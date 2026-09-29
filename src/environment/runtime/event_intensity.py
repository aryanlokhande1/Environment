"""Occurrence and empirical waiting-time sampler for PL journey events."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


class EmpiricalEventIntensity:
    def __init__(self, artifact_dir: Path):
        artifact_dir = Path(artifact_dir)
        model_path = artifact_dir / "empirical_event_intensity.csv"
        gaps_path = artifact_dir / "empirical_inter_event_gaps.parquet"
        manifest = json.loads((artifact_dir / "empirical_event_intensity_manifest.json").read_text(encoding="utf-8"))
        if "no May or simulated rows" not in manifest["source_period"]:
            raise ValueError("event intensity may fit only immutable March-April history")
        if sha256(model_path.read_bytes()).hexdigest() != manifest["model_sha256"]:
            raise ValueError("event-intensity model hash mismatch")
        if sha256(gaps_path.read_bytes()).hexdigest() != manifest["gap_sha256"]:
            raise ValueError("event-intensity gap pool hash mismatch")
        model = pd.read_csv(model_path).fillna({"journey_substage": ""})
        gaps = pd.read_parquet(gaps_path)
        self.manifest = manifest
        self.probabilities = {(str(row.model_level), str(row.journey_substage), int(row.count_bucket)):
                              float(row.p_no_later_observed_event)
                              for row in model.itertuples(index=False)}
        self.gaps_by_state_count = {
            (str(state), int(count)): group.gap_microseconds.to_numpy(dtype=np.int64)
            for (state, count), group in gaps.groupby(["journey_substage", "count_bucket"], observed=True)
        }
        self.gaps_by_state = {
            str(state): group.gap_microseconds.to_numpy(dtype=np.int64)
            for state, group in gaps.groupby("journey_substage", observed=True)
        }
        self.global_gaps = gaps.gap_microseconds.to_numpy(dtype=np.int64)
        if not len(self.global_gaps):
            raise ValueError("event-intensity gap pool is empty")

    def sample_wait(self, substage: str, events_today_before: int,
                    rng: np.random.Generator) -> tuple[int | None, str]:
        """Return a microsecond wait, or None for observed-telemetry silence."""
        bucket = min(int(events_today_before), 8)
        for level, state, count in (("STATE_COUNT", str(substage), bucket),
                                    ("STATE", str(substage), -1), ("GLOBAL", "", -1)):
            probability = self.probabilities.get((level, state, count))
            if probability is None:
                continue
            if float(rng.random()) < probability:
                return None, level
            pool = (self.gaps_by_state_count.get((state, bucket)) if level == "STATE_COUNT" else
                    self.gaps_by_state.get(state) if level == "STATE" else self.global_gaps)
            if pool is None or len(pool) < 50:
                pool = self.gaps_by_state.get(str(substage), self.global_gaps)
            return int(pool[int(rng.integers(0, len(pool)))]), level
        raise RuntimeError("empirical intensity has no supported fallback")
