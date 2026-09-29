"""Transparent piecewise empirical hazards for continuation and terminal PTP."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HazardResult:
    when: pd.Timestamp | None
    evidence: dict[str, Any]


class _PiecewiseHazard:
    def __init__(self, rows: pd.DataFrame, donors: pd.DataFrame, manifest: dict[str, Any]) -> None:
        self.rows = rows.copy()
        self.donors = donors.copy()
        self.manifest = dict(manifest)

    @staticmethod
    def _partial_probability(hazard: float, fraction: float) -> float:
        if hazard <= 0 or fraction <= 0:
            return 0.0
        if hazard >= 1:
            return 1.0
        return float(1.0 - (1.0 - hazard) ** fraction)

    def _sample(self, rows: pd.DataFrame, donors: pd.DataFrame, *, origin: pd.Timestamp,
                now: pd.Timestamp, horizon: pd.Timestamp, deadline: pd.Timestamp,
                rng: np.random.Generator, model_level: str, support_key: dict[str, str]) -> HazardResult:
        origin, now, horizon, deadline = map(pd.Timestamp, (origin, now, horizon, deadline))
        start = max(0.0, (now - origin).total_seconds())
        stop = min((horizon - origin).total_seconds(), (deadline - origin).total_seconds())
        evaluated: list[dict[str, Any]] = []
        if stop <= start:
            return HazardResult(None, {
                "model_level": model_level, "support_key": support_key,
                "reason": "NO_AT_RISK_TIME_IN_WINDOW", "candidate_hazards": evaluated,
            })
        rows = rows.sort_values("elapsed_lower_seconds", kind="stable")
        for row in rows.itertuples(index=False):
            lower, upper = float(row.elapsed_lower_seconds), float(row.elapsed_upper_seconds)
            overlap_start, overlap_stop = max(start, lower), min(stop, upper)
            if overlap_stop <= overlap_start or int(row.risk_set) <= 0:
                continue
            eligible = donors.loc[
                donors.duration_seconds.ge(overlap_start)
                & donors.duration_seconds.lt(overlap_stop)
            ]
            fraction = (overlap_stop - overlap_start) / (upper - lower)
            probability = (self._partial_probability(float(row.hazard), fraction)
                           if not eligible.empty else 0.0)
            draw = float(rng.random())
            item = {
                "elapsed_lower_seconds": int(lower), "elapsed_upper_seconds": int(upper),
                "risk_set": int(row.risk_set), "events": int(getattr(row, "ptp_events",
                                                                       getattr(row, "continuation_events", 0))),
                "full_interval_hazard": float(row.hazard),
                "window_probability": probability, "rng_draw": draw,
            }
            evaluated.append(item)
            if draw >= probability:
                continue
            if eligible.empty:
                # An interval with positive empirical events should always have
                # a corresponding donor.  Failing is safer than inventing time.
                raise RuntimeError("positive empirical hazard has no in-window timing donor")
            donor_index = int(rng.integers(0, len(eligible)))
            delay = float(eligible.iloc[donor_index].duration_seconds)
            when = origin + pd.Timedelta(seconds=delay)
            item["timing_donor_index"] = donor_index
            return HazardResult(when, {
                "model_level": model_level, "support_key": support_key,
                "support_count": int(row.risk_set), "selected_interval": item,
                "candidate_hazards": evaluated, "sampled_delay_seconds": delay,
                "reason": "EMPIRICAL_HAZARD_FIRED",
            })
        return HazardResult(None, {
            "model_level": model_level, "support_key": support_key,
            "support_count": int(rows.risk_set.max()) if not rows.empty else 0,
            "candidate_hazards": evaluated,
            "reason": "EMPIRICAL_INACTIVITY_THIS_WINDOW",
        })


class PtpHazardModel(_PiecewiseHazard):
    """Conditional residual PTP risk after the latest eligible AIP."""

    def __init__(self, artifact_dir: Path) -> None:
        directory = Path(artifact_dir)
        manifest = json.loads((directory / "ptp_hazard_manifest.json").read_text(encoding="utf-8"))
        hazard_path, donor_path = directory / "ptp_hazard.csv", directory / "ptp_hazard_event_delays.parquet"
        if manifest.get("model_version") != "PL_APPLICATION_PTP_PIECEWISE_HAZARD_V2":
            raise ValueError("unsupported PTP hazard version")
        if (sha256(hazard_path.read_bytes()).hexdigest() != manifest.get("hazard_sha256")
                or sha256(donor_path.read_bytes()).hexdigest() != manifest.get("event_delay_sha256")):
            raise ValueError("PTP hazard artifact hash mismatch")
        super().__init__(pd.read_csv(hazard_path).fillna(""), pd.read_parquet(donor_path), manifest)

    @staticmethod
    def age_bucket(state: dict[str, Any], aip: pd.Timestamp) -> str:
        age = (pd.Timestamp(aip) - pd.Timestamp(state["creation_datetime"])).total_seconds()
        return "AGE_LT_1D" if age < 86_400 else "AGE_1D_7D" if age < 7 * 86_400 else "AGE_7D_30D"

    def sample_window(self, state: dict[str, Any], now: pd.Timestamp, horizon: pd.Timestamp,
                      rng: np.random.Generator) -> HazardResult:
        aip = pd.Timestamp(state["latest_aip_datetime"])
        bucket = self.age_bucket(state, aip)
        rows = self.rows.loc[
            self.rows.model_level.eq("APPLICATION_AGE")
            & self.rows.application_age_bucket.eq(bucket)
        ]
        level = "APPLICATION_AGE"
        donors = self.donors.loc[self.donors.application_age_bucket.eq(bucket)]
        if rows.empty:
            rows = self.rows.loc[self.rows.model_level.eq("GLOBAL")]
            donors = self.donors
            level = "GLOBAL"
        return self._sample(rows, donors, origin=aip, now=now, horizon=horizon,
                            deadline=pd.Timestamp(state["deadline"]), rng=rng,
                            model_level=level, support_key={"application_age_bucket": bucket})


class StageContinuationModel(_PiecewiseHazard):
    """Conditional risk of another visible non-PTP journey event."""

    def __init__(self, artifact_dir: Path) -> None:
        directory = Path(artifact_dir)
        manifest = json.loads((directory / "stage_continuation_manifest.json").read_text(encoding="utf-8"))
        hazard_path = directory / "stage_continuation_hazard.csv"
        donor_path = directory / "stage_continuation_delays.parquet"
        if manifest.get("model_version") != "PL_STAGE_CONTINUATION_PIECEWISE_HAZARD_V2":
            raise ValueError("unsupported continuation hazard version")
        if (sha256(hazard_path.read_bytes()).hexdigest() != manifest.get("hazard_sha256")
                or sha256(donor_path.read_bytes()).hexdigest() != manifest.get("delay_sha256")):
            raise ValueError("continuation artifact hash mismatch")
        super().__init__(pd.read_csv(hazard_path).fillna(""), pd.read_parquet(donor_path), manifest)

    def sample_window(self, state: dict[str, Any], now: pd.Timestamp, horizon: pd.Timestamp,
                      rng: np.random.Generator) -> HazardResult:
        substage, stage = str(state["journey_substage"]), str(state["journey_stage"])
        rows = self.rows.loc[self.rows.model_level.eq("STATE")
                             & self.rows.journey_substage.eq(substage)]
        donors = self.donors.loc[self.donors.journey_substage.eq(substage)]
        level, key = "STATE", {"journey_substage": substage}
        if rows.empty:
            rows = self.rows.loc[self.rows.model_level.eq("STAGE")
                                 & self.rows.journey_stage.eq(stage)]
            donors = self.donors.loc[self.donors.journey_stage.eq(stage)]
            level, key = "STAGE", {"journey_stage": stage}
        if rows.empty:
            rows = self.rows.loc[self.rows.model_level.eq("GLOBAL")]
            donors = self.donors
            level, key = "GLOBAL", {}
        return self._sample(rows, donors, origin=pd.Timestamp(state["last_journey_datetime"]),
                            now=now, horizon=horizon, deadline=pd.Timestamp(state["deadline"]),
                            rng=rng, model_level=level, support_key=key)
