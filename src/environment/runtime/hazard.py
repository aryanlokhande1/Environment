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

    def _sample_corrected(self, rows: pd.DataFrame, donors: pd.DataFrame, *,
                          origin: pd.Timestamp, now: pd.Timestamp, horizon: pd.Timestamp,
                          deadline: pd.Timestamp, rng: np.random.Generator,
                          model_level: str, support_key: dict[str, str]) -> HazardResult:
        """Invert the conditional empirical survival distribution once.

        h is events/risk_set conditional on survival to a bin's left edge.
        Each in-bin timing donor receives h/n conditional mass. Condition on
        survival to now, including partial bins, and retain no-event mass.
        The caller schedules through deadline once per episode, not per poll.
        """
        origin, now, horizon, deadline = map(pd.Timestamp, (origin, now, horizon, deadline))
        start = max(0., (now - origin).total_seconds())
        stop = min((horizon - origin).total_seconds(), (deadline - origin).total_seconds())
        survival = 1.0
        survived_to_start = 1.0
        times, masses = [], []
        durations = donors.duration_seconds.to_numpy(dtype=float, copy=False)
        for row in rows.sort_values("elapsed_lower_seconds", kind="stable").itertuples(index=False):
            lower, upper, h = float(row.elapsed_lower_seconds), float(row.elapsed_upper_seconds), float(row.hazard)
            if not 0 <= h <= 1 or upper <= lower:
                raise ValueError("invalid empirical hazard bin")
            pool = np.sort(durations[(durations >= lower) & (durations < upper)])
            if h > 0 and not len(pool):
                raise ValueError("positive empirical hazard has no timing donors")
            if len(pool):
                mass = survival * h / len(pool)
                survived_to_start -= mass * int((pool < start).sum())
                selected = pool[(pool >= start) & (pool < stop)]
                times.extend(selected.tolist())
                masses.extend([mass] * len(selected))
            survival *= 1 - h
        draw = float(rng.random())
        evidence = {"model_level": model_level, "support_key": support_key,
                    "support_count": int(rows.risk_set.max()) if not rows.empty else 0,
                    "sampling_semantics": "CONDITIONAL_EMPIRICAL_SURVIVAL",
                    "survival_at_start": max(0., survived_to_start), "rng_draw": draw}
        if survived_to_start > 1e-12 and masses:
            probabilities = np.asarray(masses) / survived_to_start
            index = int(np.searchsorted(np.cumsum(probabilities), draw, side="right"))
            if index < len(times):
                evidence.update(reason="EMPIRICAL_HAZARD_FIRED", sampled_delay_seconds=times[index])
                return HazardResult(origin + pd.Timedelta(seconds=times[index]), evidence)
        evidence["reason"] = "EMPIRICAL_INACTIVITY_THIS_EPISODE"
        return HazardResult(None, evidence)

    def _sample(self, rows: pd.DataFrame, donors: pd.DataFrame, *, origin: pd.Timestamp,
                now: pd.Timestamp, horizon: pd.Timestamp, deadline: pd.Timestamp,
                rng: np.random.Generator, model_level: str, support_key: dict[str, str]) -> HazardResult:
        if getattr(self, "corrected", False):
            return self._sample_corrected(rows, donors, origin=origin, now=now, horizon=horizon,
                                          deadline=deadline, rng=rng, model_level=model_level,
                                          support_key=support_key)
        origin, now, horizon, deadline = map(pd.Timestamp, (origin, now, horizon, deadline))
        start = max(0.0, (now - origin).total_seconds())
        stop = min((horizon - origin).total_seconds(), (deadline - origin).total_seconds())
        evaluated: list[dict[str, Any]] = []
        if stop <= start:
            return HazardResult(None, {
                "model_level": model_level, "support_key": support_key,
                "reason": "NO_AT_RISK_TIME_IN_WINDOW", "candidate_hazards": evaluated,
            })
        if not rows.elapsed_lower_seconds.is_monotonic_increasing:
            rows = rows.sort_values("elapsed_lower_seconds", kind="stable")
        donor_durations = donors.duration_seconds.to_numpy(dtype=float, copy=False)
        for row in rows.itertuples(index=False):
            lower, upper = float(row.elapsed_lower_seconds), float(row.elapsed_upper_seconds)
            overlap_start, overlap_stop = max(start, lower), min(stop, upper)
            if overlap_stop <= overlap_start or int(row.risk_set) <= 0:
                continue
            eligible_indices = np.flatnonzero(
                (donor_durations >= overlap_start) & (donor_durations < overlap_stop))
            fraction = (overlap_stop - overlap_start) / (upper - lower)
            probability = (self._partial_probability(float(row.hazard), fraction)
                           if len(eligible_indices) else 0.0)
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
            if not len(eligible_indices):
                # An interval with positive empirical events should always have
                # a corresponding donor.  Failing is safer than inventing time.
                raise RuntimeError("positive empirical hazard has no in-window timing donor")
            donor_index = int(rng.integers(0, len(eligible_indices)))
            delay = float(donor_durations[eligible_indices[donor_index]])
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
        self._age_rows = {str(key): group for key, group in
                          self.rows.loc[self.rows.model_level.eq("APPLICATION_AGE")]
                          .groupby("application_age_bucket", observed=True, sort=False)}
        self._age_donors = {str(key): group for key, group in
                            self.donors.groupby("application_age_bucket", observed=True, sort=False)}
        self._global_rows = self.rows.loc[self.rows.model_level.eq("GLOBAL")]

    @staticmethod
    def age_bucket(state: dict[str, Any], aip: pd.Timestamp) -> str:
        age = (pd.Timestamp(aip) - pd.Timestamp(state["creation_datetime"])).total_seconds()
        return "AGE_LT_1D" if age < 86_400 else "AGE_1D_7D" if age < 7 * 86_400 else "AGE_7D_30D"

    def sample_window(self, state: dict[str, Any], now: pd.Timestamp, horizon: pd.Timestamp,
                      rng: np.random.Generator) -> HazardResult:
        aip = pd.Timestamp(state["latest_aip_datetime"])
        bucket = self.age_bucket(state, aip)
        rows = self._age_rows.get(bucket, self.rows.iloc[0:0])
        level = "APPLICATION_AGE"
        donors = self._age_donors.get(bucket, self.donors.iloc[0:0])
        if rows.empty:
            rows = self._global_rows
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
        state_rows = self.rows.loc[self.rows.model_level.eq("STATE")]
        stage_rows = self.rows.loc[self.rows.model_level.eq("STAGE")]
        self._state_rows = {str(key): group for key, group in
                            state_rows.groupby("journey_substage", observed=True, sort=False)}
        self._stage_rows = {str(key): group for key, group in
                            stage_rows.groupby("journey_stage", observed=True, sort=False)}
        self._state_donors = {str(key): group for key, group in
                              self.donors.groupby("journey_substage", observed=True, sort=False)}
        self._stage_donors = {str(key): group for key, group in
                              self.donors.groupby("journey_stage", observed=True, sort=False)}
        self._global_rows = self.rows.loc[self.rows.model_level.eq("GLOBAL")]

    def sample_window(self, state: dict[str, Any], now: pd.Timestamp, horizon: pd.Timestamp,
                      rng: np.random.Generator) -> HazardResult:
        substage, stage = str(state["journey_substage"]), str(state["journey_stage"])
        rows = self._state_rows.get(substage, self.rows.iloc[0:0])
        donors = self._state_donors.get(substage, self.donors.iloc[0:0])
        level, key = "STATE", {"journey_substage": substage}
        if rows.empty:
            rows = self._stage_rows.get(stage, self.rows.iloc[0:0])
            donors = self._stage_donors.get(stage, self.donors.iloc[0:0])
            level, key = "STAGE", {"journey_stage": stage}
        if rows.empty:
            rows = self._global_rows
            donors = self.donors
            level, key = "GLOBAL", {}
        return self._sample(rows, donors, origin=pd.Timestamp(state["last_journey_datetime"]),
                            now=now, horizon=horizon, deadline=pd.Timestamp(state["deadline"]),
                            rng=rng, model_level=level, support_key=key)
