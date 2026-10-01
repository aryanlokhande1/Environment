"""Build the versioned May-arrival and Gold event-name correction bundle.

Only immutable March-April Gold Events and the V1 PL cohort/donor artifacts are
used.  The V1 directory is never modified.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


START = pd.Timestamp("2026-03-01")
MAY = pd.Timestamp("2026-05-01")
END = pd.Timestamp("2026-06-01")
SEED = 20260502


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def stable_rng(*parts: object) -> np.random.Generator:
    raw = sha256("|".join(map(str, parts)).encode()).digest()
    return np.random.default_rng(int.from_bytes(raw[:8], "little"))


def stable_id(prefix: str, *parts: object) -> str:
    return prefix + sha256("|".join(map(str, parts)).encode()).hexdigest()[:28]


def largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    raw = weights / weights.sum() * total
    result = np.floor(raw).astype(int)
    remainder = total - int(result.sum())
    if remainder:
        result[np.argsort(raw - result)[-remainder:]] += 1
    return result


def scan_history(paths: list[Path], lifecycle: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    owners = lifecycle.set_index("application_id")
    application_ids = set(owners.index)
    observed: list[pd.DataFrame] = []
    creations: list[pd.DataFrame] = []
    source_order = 0
    columns = ["application_id", "context_id", "event_datetime", "event_name",
               "journey_stage", "journey_substage", "source_type"]
    for file_order, path in enumerate(paths):
        for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=262_144):
            frame = batch.to_pandas()
            frame = frame.loc[frame.application_id.notna()]
            frame["application_id"] = frame.application_id.astype(str)
            frame = frame.loc[frame.application_id.isin(application_ids)].copy()
            if frame.empty:
                continue
            frame["_file_order"] = file_order
            frame["_source_order"] = np.arange(source_order, source_order + len(frame))
            source_order += len(frame)
            observed.append(frame)
            creations.append(frame.loc[frame.journey_substage.eq("Application Created")].copy())
    events = pd.concat(observed, ignore_index=True)
    candidates = pd.concat(creations, ignore_index=True).merge(
        lifecycle, on="application_id", how="inner", validate="many_to_one",
        suffixes=("", "_owner"),
    )
    candidates["event_datetime"] = pd.to_datetime(candidates.event_datetime)
    candidates["application_created_at"] = pd.to_datetime(candidates.application_created_at)
    exact = candidates.loc[
        candidates.event_datetime.eq(candidates.application_created_at)
        & candidates.context_id.astype(str).eq(candidates.context_id_owner.astype(str))
    ].sort_values(["application_id", "_file_order", "_source_order"], kind="stable")
    exact = exact.drop_duplicates("application_id", keep="first")
    return events, exact


def build_event_name_model(events: pd.DataFrame, output: Path, source_hashes: dict[str, str]) -> dict:
    frame = events[["journey_stage", "journey_substage", "source_type", "event_name"]].copy()
    missing = int(frame.event_name.isna().sum())
    if missing:
        raise RuntimeError(f"PL event-name fitting rows unexpectedly contain {missing} NULL values")
    frame = frame.fillna("").astype(str)
    levels = (
        ("STAGE_SUBSTAGE_SOURCE", ["journey_stage", "journey_substage", "source_type"]),
        ("STAGE_SUBSTAGE", ["journey_stage", "journey_substage"]),
        ("SUBSTAGE_SOURCE", ["journey_substage", "source_type"]),
        ("SUBSTAGE", ["journey_substage"]),
    )
    rows: list[pd.DataFrame] = []
    for level, keys in levels:
        counts = frame.groupby([*keys, "event_name"], dropna=False).size().rename("support_count").reset_index()
        counts["probability"] = counts.support_count / counts.groupby(keys).support_count.transform("sum")
        counts["model_level"] = level
        for name in ("journey_stage", "journey_substage", "source_type"):
            if name not in counts:
                counts[name] = ""
        rows.append(counts[["model_level", "journey_stage", "journey_substage",
                            "source_type", "event_name", "support_count", "probability"]])
    model = pd.concat(rows, ignore_index=True).sort_values(
        ["model_level", "journey_stage", "journey_substage", "source_type", "event_name"],
        kind="stable",
    )
    model_path = output / "event_name_mapping.csv"
    model.to_csv(model_path, index=False)
    report = {
        "model_version": "PL_GOLD_EVENT_NAME_CONDITIONAL_V1",
        "source_period": "2026-03-01 through 2026-04-30; no May data",
        "population": "Personal Loan application-owned events",
        "fitting_rows": len(frame),
        "missing_event_name_rows": missing,
        "mapping_rows": len(model),
        "backoff_hierarchy": [level for level, _ in levels],
        "source_files_sha256": source_hashes,
        "model_sha256": digest(model_path),
    }
    manifest_path = output / "event_name_mapping_manifest.json"
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def profile(daily: pd.Series) -> dict:
    daily = daily.sort_index().astype(int)
    total = int(daily.sum())
    rows = pd.DataFrame({"date": daily.index, "count": daily.values})
    rows["weekday"] = rows.date.dt.day_name()
    rows["week_of_month"] = np.minimum((rows.date.dt.day - 1) // 7 + 1, 4)
    rows["normalized_share"] = rows["count"] / total
    weekday = rows.groupby("weekday").agg(count=("count", "sum"), days=("date", "size"))
    weekday["mean_per_day"] = weekday["count"] / weekday["days"]
    weekday["normalized_share"] = weekday["count"] / total
    week = rows.groupby("week_of_month")["count"].sum()
    weekend = rows.loc[rows.date.dt.dayofweek.ge(5), "count"].sum()
    return {
        "total": total,
        "daily": [{**row, "date": row["date"].date().isoformat()}
                  for row in rows.to_dict("records")],
        "weekday": {key: {k: float(v) for k, v in value.items()}
                    for key, value in weekday.to_dict("index").items()},
        "week_of_month": {str(k): int(v) for k, v in week.items()},
        "weekend_share": float(weekend / total),
        "weekday_to_weekend_ratio": float((total - weekend) / weekend),
    }


def build_arrivals(source: Path, exact: pd.DataFrame, output: Path) -> dict:
    schedule = pd.read_parquet(source / "may_application_arrivals.parquet")
    total = len(schedule)
    history_days = pd.date_range(START, MAY - pd.Timedelta(days=1), freq="D")
    exact_daily = exact.groupby(exact.event_datetime.dt.normalize()).size().reindex(history_days, fill_value=0)
    may_days = pd.date_range(MAY, END - pd.Timedelta(days=1), freq="D")

    # Source magnitude is censored/nonstationary, so month phase is not a valid
    # forecast key. Preserve the total V1 donor population, estimate weekday
    # mass over both months, and bootstrap positive same-weekday daily residuals.
    weekday_means = exact_daily.groupby(exact_daily.index.dayofweek).mean()
    weekday_weights = np.array([
        weekday_means[weekday] * sum(day.dayofweek == weekday for day in may_days)
        for weekday in range(7)
    ])
    weekday_totals = largest_remainder(weekday_weights, total)
    target_counts: dict[pd.Timestamp, int] = {}
    for weekday in range(7):
        targets = may_days[may_days.dayofweek == weekday]
        donors = exact_daily[(exact_daily.index.dayofweek == weekday) & exact_daily.gt(0)].to_numpy(float)
        rng = stable_rng(SEED, "may-arrival-v2-residual", weekday)
        residuals = rng.choice(donors, size=len(targets), replace=True)
        allocations = largest_remainder(residuals, int(weekday_totals[weekday]))
        target_counts.update(zip(targets, map(int, allocations), strict=True))

    rng = stable_rng(SEED, "may-arrival-v2-record-assignment")
    schedule = schedule.iloc[rng.permutation(len(schedule))].reset_index(drop=True)
    assigned: list[pd.DataFrame] = []
    cursor = 0
    for target in may_days:
        count = target_counts[target]
        block = schedule.iloc[cursor:cursor + count].copy()
        cursor += count
        old_time = pd.to_datetime(block.activation_datetime)
        block["application_created_at"] = target + (old_time - old_time.dt.normalize())
        block["activation_datetime"] = block.application_created_at
        block["application_id"] = [
            stable_id("MAYAPP2_", SEED, donor, target.date(), index)
            for index, donor in enumerate(block.donor_application_id.astype(str))
        ]
        block["population_origin"] = "EMPIRICAL_MAY_ARRIVAL_V2"
        assigned.append(block)
    if cursor != total:
        raise RuntimeError("V2 arrival allocation did not preserve the donor population")
    candidate = pd.concat(assigned, ignore_index=True)
    candidate.sort_values(["donor_context_id", "application_created_at", "application_id"],
                          kind="stable", inplace=True)
    contexts: dict[str, tuple[pd.Timestamp, int]] = {}
    mapped = []
    for row in candidate.itertuples(index=False):
        previous = contexts.get(str(row.donor_context_id))
        group = 0 if previous is None else previous[1] + int(
            row.application_created_at < previous[0] + pd.Timedelta(days=30))
        contexts[str(row.donor_context_id)] = (pd.Timestamp(row.application_created_at), group)
        mapped.append(stable_id("MAYCTX2_", SEED, row.donor_context_id, group))
    candidate["snapshot_context_id"] = mapped
    candidate.sort_values(["activation_datetime", "snapshot_context_id", "application_id"],
                          kind="stable", inplace=True)
    if not candidate.application_id.is_unique:
        raise RuntimeError("V2 May application ID collision")
    schedule_path = output / "may_application_arrivals.parquet"
    candidate.to_parquet(schedule_path, index=False)

    baseline_daily = pd.read_parquet(source / "may_application_arrivals.parquet", columns=["activation_datetime"])
    baseline_daily = baseline_daily.groupby(
        pd.to_datetime(baseline_daily.activation_datetime).dt.normalize()).size().reindex(may_days, fill_value=0)
    candidate_daily = candidate.groupby(
        pd.to_datetime(candidate.activation_datetime).dt.normalize()).size().reindex(may_days, fill_value=0)
    historical_profile = profile(exact_daily)
    baseline_profile = profile(baseline_daily)
    candidate_profile = profile(candidate_daily)
    criteria = {
        "arrival_days": {"required": 31, "actual": int(candidate_daily.gt(0).sum())},
        "share_by_may_11_max": {"required": 0.60,
                                "actual": float(candidate_daily.iloc[:11].sum() / total)},
        "share_by_may_15_max": {"required": 0.75,
                                "actual": float(candidate_daily.iloc[:15].sum() / total)},
        "share_may_24_31_min": {"required": 0.12,
                                "actual": float(candidate_daily.iloc[23:].sum() / total)},
        "weekend_share_abs_delta_max": {
            "required": 0.03,
            "actual": abs(candidate_profile["weekend_share"] - historical_profile["weekend_share"]),
        },
    }
    passed = (
        criteria["arrival_days"]["actual"] == 31
        and criteria["share_by_may_11_max"]["actual"] <= 0.60
        and criteria["share_by_may_15_max"]["actual"] <= 0.75
        and criteria["share_may_24_31_min"]["actual"] >= 0.12
        and criteria["weekend_share_abs_delta_max"]["actual"] <= 0.03
    )
    if not passed:
        raise RuntimeError(f"V2 arrival acceptance criteria failed: {criteria}")
    report = {
        "model_version": "PL_WEEKDAY_RESIDUAL_BOOTSTRAP_V2",
        "source_period": "2026-03-01 through 2026-04-30; no May data",
        "seed": SEED,
        "method": ("preserve the exact V1 donor population; remove unsupported month-phase mapping; "
                   "allocate weekday mass from March-April weekday means and bootstrap positive "
                   "same-weekday empirical daily residuals; preserve donor time-of-day"),
        "root_cause": ("V1 copied the source's severe within-month coverage decay and April 22-30 "
                       "zero-creation boundary into May through exact month-phase donor matching."),
        "simulated_may_arrivals": total,
        "simulated_may_contexts": int(candidate.snapshot_context_id.nunique()),
        "historical_exact_creation_donors": len(exact),
        "historical_profile": historical_profile,
        "baseline_profile": baseline_profile,
        "candidate_profile": candidate_profile,
        "acceptance_criteria_defined_before_full_run": criteria,
        "acceptance_status": "PASS",
        "schedule_sha256": digest(schedule_path),
    }
    manifest_path = output / "may_arrival_manifest.json"
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def artifact_record(logical: str, filename: str, digest_value: str, version: str,
                    support_count: int, support_unit: str, keys: list[str], backoff: list[str]) -> dict:
    return {
        "logical_name": logical, "filename": filename, "sha256": digest_value,
        "model_artifact_version": version,
        "source_period": "2026-03-01 through 2026-04-30; immutable historical only",
        "population": "Personal Loan", "support_count": support_count,
        "support_unit": support_unit, "conditioning_keys": keys,
        "backoff_hierarchy": backoff, "authoritative_runtime_input": True,
    }


def build(source: Path, output: Path, history: list[Path]) -> dict:
    if output.exists():
        raise FileExistsError(f"versioned output bundle already exists: {output}")
    lifecycle = pd.read_parquet(
        source / "application_lifecycle_state.parquet",
        columns=["application_id", "context_id", "application_created_at"],
    ).rename(columns={"context_id": "context_id_owner"})
    lifecycle["application_id"] = lifecycle.application_id.astype(str)
    source_hashes = {path.name: digest(path) for path in history}
    events, exact = scan_history(history, lifecycle)
    shutil.copytree(source, output)
    event_report = build_event_name_model(events, output, source_hashes)
    arrival_report = build_arrivals(source, exact, output)

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["bundle_version"] = "gold_events_v2"
    manifest["fitting_policy"] = "immutable March-April historical only; simulated and synthetic rows forbidden"
    records = [row for row in manifest["artifacts"]
               if row["logical_name"] not in {"may_application_arrivals", "may_arrival_provenance",
                                               "event_name_mapping", "event_name_mapping_manifest"}]
    records.append(artifact_record(
        "may_application_arrivals", "may_application_arrivals.parquet",
        arrival_report["schedule_sha256"], arrival_report["model_version"],
        arrival_report["simulated_may_arrivals"], "historical PL creation donor",
        ["weekday", "positive empirical daily residual"], ["WEEKDAY", "GLOBAL"],
    ))
    arrival_manifest = output / "may_arrival_manifest.json"
    records.append(artifact_record(
        "may_arrival_provenance", arrival_manifest.name, digest(arrival_manifest),
        arrival_report["model_version"], arrival_report["historical_exact_creation_donors"],
        "arrival model manifest", [], [],
    ))
    records.append(artifact_record(
        "event_name_mapping", "event_name_mapping.csv", event_report["model_sha256"],
        event_report["model_version"], event_report["fitting_rows"],
        "historical PL application-owned Gold event",
        ["journey_stage", "journey_substage", "source_type"], event_report["backoff_hierarchy"],
    ))
    name_manifest = output / "event_name_mapping_manifest.json"
    records.append(artifact_record(
        "event_name_mapping_manifest", name_manifest.name, digest(name_manifest),
        event_report["model_version"], event_report["fitting_rows"],
        "model manifest", [], [],
    ))
    manifest["artifacts"] = sorted(records, key=lambda row: row["logical_name"])
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    hashes = {row["filename"]: row["sha256"] for row in manifest["artifacts"]}
    hashes["manifest.json"] = digest(manifest_path)
    (output / "hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True), encoding="utf-8")
    return {"bundle": str(output), "bundle_version": "gold_events_v2",
            "arrivals": arrival_report["simulated_may_arrivals"],
            "arrival_acceptance": arrival_report["acceptance_criteria_defined_before_full_run"],
            "event_name_mapping_rows": event_report["mapping_rows"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("artifacts/gold_events_v1"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gold_events_v2"))
    parser.add_argument("--history", type=Path, action="append", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, args.history), indent=2))


if __name__ == "__main__":
    main()
