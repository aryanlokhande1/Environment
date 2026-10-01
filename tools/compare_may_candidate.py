"""One-pass March/April, baseline May, and candidate May comparison report."""
from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def load_may(run: Path) -> pd.DataFrame:
    paths = sorted((run / "may").glob("*/gold_events.parquet"))
    columns = ["application_id", "context_id", "event_datetime", "event_name",
               "journey_stage", "journey_substage", "source_type"]
    return pd.concat([pd.read_parquet(path, columns=columns) for path in paths], ignore_index=True)


def load_history(paths: list[Path], application_ids: set[str]) -> tuple[pd.DataFrame, pd.Series]:
    owned = []
    daily = Counter()
    columns = ["application_id", "context_id", "event_datetime", "event_name",
               "journey_stage", "journey_substage", "source_type"]
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=262_144):
            frame = batch.to_pandas()
            dates = pd.to_datetime(frame.event_datetime).dt.normalize()
            daily.update(dates.value_counts().to_dict())
            frame = frame.loc[frame.application_id.notna()].copy()
            frame["application_id"] = frame.application_id.astype(str)
            owned.append(frame.loc[frame.application_id.isin(application_ids)])
    index = pd.date_range("2026-03-01", "2026-04-30", freq="D")
    return pd.concat(owned, ignore_index=True), pd.Series(daily).reindex(index, fill_value=0).astype(int)


def tv(left: pd.Series, right: pd.Series) -> float:
    keys = left.index.union(right.index)
    return float(0.5 * (left.reindex(keys, fill_value=0) - right.reindex(keys, fill_value=0)).abs().sum())


def distribution(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].fillna("<NULL>").value_counts(normalize=True)


def transition_distribution(frame: pd.DataFrame) -> pd.Series:
    ordered = frame.sort_values(["application_id", "event_datetime"], kind="stable").copy()
    ordered["prior"] = ordered.groupby("application_id").journey_substage.shift()
    pairs = ordered.loc[ordered.prior.notna(), ["prior", "journey_substage"]].fillna("<NULL>")
    return pairs.value_counts(normalize=True)


def profile(frame: pd.DataFrame, arrivals: set[str] | None = None) -> dict:
    frame = frame.copy()
    frame["event_datetime"] = pd.to_datetime(frame.event_datetime)
    frame["application_id"] = frame.application_id.astype(str)
    frame.sort_values(["application_id", "event_datetime"], kind="stable", inplace=True)
    created = frame.loc[frame.journey_substage.eq("Application Created")]
    aip = frame.loc[frame.journey_substage.eq("AIP Approved")]
    ptp = frame.loc[frame.journey_substage.eq("Push to Partner")]
    daily = frame.groupby(frame.event_datetime.dt.normalize()).size()
    lengths = frame.groupby("application_id").size()
    gaps = frame.groupby("application_id").event_datetime.diff().dt.total_seconds().dropna()
    first_created = created.groupby("application_id").event_datetime.min()
    next_rows = frame.merge(first_created.rename("created_at"), left_on="application_id", right_index=True)
    next_rows = next_rows.loc[next_rows.event_datetime.gt(next_rows.created_at)]
    next_delay = (next_rows.groupby("application_id").event_datetime.min()
                  - first_created).dt.total_seconds().dropna()
    spans = frame.groupby("application_id").event_datetime.agg(["min", "max"])
    result = {
        "events": len(frame), "event_active_contexts": int(frame.context_id.nunique()),
        "events_per_event_active_context": float(len(frame) / frame.context_id.nunique()),
        "created_rows": len(created), "created_unique_applications": int(created.application_id.nunique()),
        "aip_rows": len(aip), "aip_unique_applications": int(aip.application_id.nunique()),
        "aip_repeated_rows": int(len(aip) - aip.application_id.nunique()),
        "ptp_rows": len(ptp), "ptp_unique_applications": int(ptp.application_id.nunique()),
        "missing_event_name": int(frame.event_name.isna().sum() + frame.event_name.eq("").sum()),
        "mean_sequence_length": float(lengths.mean()),
        "median_sequence_length": float(lengths.median()),
        "median_inter_event_seconds": float(gaps.median()),
        "p90_inter_event_seconds": float(gaps.quantile(.9)),
        "median_creation_to_next_seconds": float(next_delay.median()),
        "p90_creation_to_next_seconds": float(next_delay.quantile(.9)),
        "multi_day_application_share": float((spans["max"].dt.normalize() > spans["min"].dt.normalize()).mean()),
        "daily_event_share_by_11": float(daily.loc[daily.index.day <= 11].sum() / len(frame)),
        "daily_event_share_by_15": float(daily.loc[daily.index.day <= 15].sum() / len(frame)),
        "daily_event_share_24_31": float(daily.loc[daily.index.day >= 24].sum() / len(frame)),
        "daily_event_counts": {str(key.date()): int(value) for key, value in daily.items()},
    }
    if arrivals is not None:
        is_arrival = frame.application_id.isin(arrivals)
        result.update({
            "carried_forward_event_applications": int(frame.loc[~is_arrival, "application_id"].nunique()),
            "carried_forward_aip_applications": int(aip.loc[~aip.application_id.isin(arrivals), "application_id"].nunique()),
            "carried_forward_aip_rows": int((~aip.application_id.isin(arrivals)).sum()),
            "arrival_aip_applications": int(aip.loc[aip.application_id.isin(arrivals), "application_id"].nunique()),
            "arrival_ptp_applications": int(ptp.loc[ptp.application_id.isin(arrivals), "application_id"].nunique()),
            "carried_forward_ptp_applications": int(ptp.loc[~ptp.application_id.isin(arrivals), "application_id"].nunique()),
        })
    return result


def arrival_profile(path: Path) -> dict:
    frame = pd.read_parquet(path, columns=["application_id", "activation_datetime"])
    dates = pd.to_datetime(frame.activation_datetime)
    daily = frame.groupby(dates.dt.normalize()).size()
    return {
        "arrivals": len(frame), "active_days": len(daily),
        "share_by_11": float(daily.loc[daily.index.day <= 11].sum() / len(frame)),
        "share_by_15": float(daily.loc[daily.index.day <= 15].sum() / len(frame)),
        "share_24_31": float(daily.loc[daily.index.day >= 24].sum() / len(frame)),
        "weekend_share": float(frame.loc[dates.dt.dayofweek.ge(5)].shape[0] / len(frame)),
        "daily_counts": {str(key.date()): int(value) for key, value in daily.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, action="append", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lifecycle = pd.read_parquet("artifacts/gold_events_v1/application_lifecycle_state.parquet",
                                columns=["application_id"])
    history, history_daily = load_history(args.history, set(lifecycle.application_id.astype(str)))
    baseline = load_may(args.baseline)
    candidate = load_may(args.candidate)
    baseline_arrival_path = Path("artifacts/gold_events_v1/may_application_arrivals.parquet")
    candidate_arrival_path = Path("artifacts/gold_events_v2/may_application_arrivals.parquet")
    baseline_ids = set(pd.read_parquet(baseline_arrival_path, columns=["application_id"]).application_id.astype(str))
    candidate_ids = set(pd.read_parquet(candidate_arrival_path, columns=["application_id"]).application_id.astype(str))
    march = history.loc[pd.to_datetime(history.event_datetime).dt.month.eq(3)]
    april = history.loc[pd.to_datetime(history.event_datetime).dt.month.eq(4)]
    historical = profile(history)
    baseline_profile = profile(baseline, baseline_ids)
    candidate_profile = profile(candidate, candidate_ids)
    transition_tv = tv(transition_distribution(candidate), transition_distribution(history))
    substage_tv = tv(distribution(candidate, "journey_substage"), distribution(history, "journey_substage"))
    criteria = {
        "event_name_missing_equals_zero": candidate_profile["missing_event_name"] == 0,
        "created_unique_equals_arrival_population": (
            candidate_profile["created_unique_applications"] == len(candidate_ids)),
        "event_share_by_11_max_0_70": candidate_profile["daily_event_share_by_11"] <= .70,
        "event_share_24_31_min_0_08": candidate_profile["daily_event_share_24_31"] >= .08,
        "transition_tv_max_0_35": transition_tv <= .35,
        "substage_tv_max_0_35": substage_tv <= .35,
        "inter_event_median_ratio_0_25_to_4": .25 <= (
            candidate_profile["median_inter_event_seconds"] / historical["median_inter_event_seconds"]) <= 4,
    }
    report = {
        "acceptance_status": "PASS" if all(criteria.values()) else "FAIL",
        "acceptance_criteria": criteria,
        "transition_tv_candidate_vs_historical": transition_tv,
        "substage_tv_candidate_vs_historical": substage_tv,
        "historical_application_owned": historical,
        "march_historical_application_owned": profile(march),
        "april_historical_application_owned": profile(april),
        "baseline_may": baseline_profile,
        "candidate_may": candidate_profile,
        "baseline_arrivals": arrival_profile(baseline_arrival_path),
        "candidate_arrivals": arrival_profile(candidate_arrival_path),
        "historical_daily_all_gold": {str(key.date()): int(value) for key, value in history_daily.items()},
        "june_leakage": int((pd.to_datetime(candidate.event_datetime) >= pd.Timestamp("2026-06-01")).sum()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["acceptance_status"], "criteria": criteria,
        "transition_tv": transition_tv, "substage_tv": substage_tv,
        "baseline": baseline_profile, "candidate": candidate_profile,
    }, indent=2))


if __name__ == "__main__":
    main()
