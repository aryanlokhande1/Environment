"""Joint observational campaign engagement and journey-response motifs."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CampaignMotifResult:
    engagement_events: tuple[tuple[str, int], ...]
    next_journey_substage: str | None
    journey_delay_seconds: int | None
    release_delay_seconds: int
    evidence: dict[str, Any]


class CampaignMotifModel:
    LEVELS = (
        ("STATE_CHANNEL_TIME_PRIOR", ("current_journey_substage", "channel", "time_bucket", "prior_send_bucket")),
        ("STATE_CHANNEL_TIME", ("current_journey_substage", "channel", "time_bucket")),
        ("STATE_CHANNEL", ("current_journey_substage", "channel")),
        ("STAGE_CHANNEL", ("current_journey_stage", "channel")),
        ("CHANNEL_TIME_PRIOR", ("channel", "time_bucket", "prior_send_bucket")),
        ("CHANNEL", ("channel",)),
        ("GLOBAL", ()),
    )

    def __init__(self, artifact_dir: Path) -> None:
        directory = Path(artifact_dir)
        manifest = json.loads((directory / "campaign_response_motif_manifest.json").read_text(encoding="utf-8"))
        model_path = directory / "campaign_response_motif_model.csv"
        donor_path = directory / "campaign_response_motif_donors.parquet"
        if manifest.get("model_version") not in {
                "PL_JOINT_CAMPAIGN_RESPONSE_MOTIF_V1", "PL_JOINT_CAMPAIGN_RESPONSE_MOTIF_V2"}:
            raise ValueError("unsupported campaign motif model")
        if (sha256(model_path.read_bytes()).hexdigest() != manifest.get("model_sha256")
                or sha256(donor_path.read_bytes()).hexdigest() != manifest.get("donor_sha256")):
            raise ValueError("campaign motif artifact hash mismatch")
        self.manifest = manifest
        self.rows = pd.read_csv(model_path).fillna("")
        self.donors = pd.read_parquet(donor_path).fillna({
            "current_journey_stage": "", "current_journey_substage": "",
            "channel": "", "time_bucket": "",
        })

    @staticmethod
    def _normalize_bucket(value: str) -> str:
        return str(value).upper()

    def _lookup(self, state: dict[str, Any], channel: str, time_bucket: str
                ) -> tuple[pd.DataFrame, pd.DataFrame, str, dict[str, str]]:
        values = {
            "current_journey_substage": str(state["journey_substage"]),
            "current_journey_stage": str(state["journey_stage"]),
            "channel": str(channel).upper(), "time_bucket": self._normalize_bucket(time_bucket),
            "prior_send_bucket": ("0" if int(state.get("campaign_sends_seen", 0)) == 0 else
                                  "1" if int(state.get("campaign_sends_seen", 0)) == 1 else "2_PLUS"),
        }
        for level, keys in self.LEVELS:
            rows = self.rows.loc[self.rows.model_level.eq(level)]
            donors = self.donors
            for key in keys:
                rows = rows.loc[rows[key].astype(str).eq(values[key])]
                donors = donors.loc[donors[key].astype(str).eq(values[key])]
            if not rows.empty:
                return rows, donors, level, {key: values[key] for key in keys}
        raise RuntimeError("campaign motif model has no global fallback")

    def sample(self, state: dict[str, Any], channel: str, time_bucket: str,
               rng: np.random.Generator, forbidden_journey: set[str] | None = None
               ) -> CampaignMotifResult:
        rows, donors, level, keys = self._lookup(state, channel, time_bucket)
        probabilities = rows.probability.astype(float).to_numpy(copy=True)
        probabilities /= probabilities.sum()
        draw = float(rng.random())
        index = int(np.searchsorted(np.cumsum(probabilities), draw, side="right"))
        index = min(index, len(rows) - 1)
        selected = rows.iloc[index]
        engagement = str(selected.engagement_sequence)
        journey = str(selected.next_journey_substage)
        matched = donors.loc[
            donors.engagement_sequence.astype(str).eq(engagement)
            & donors.next_journey_substage.astype(str).eq(journey)
        ]
        if matched.empty:
            raise RuntimeError("campaign motif has no timing donor")
        donor_index = int(rng.integers(0, len(matched)))
        donor = matched.iloc[donor_index]
        events: list[tuple[str, int]] = []
        if "campaign_open_read" in engagement and pd.notna(donor.open_delay_seconds):
            events.append(("campaign_open_read", max(1, int(round(float(donor.open_delay_seconds))))))
        if "campaign_clicked" in engagement and pd.notna(donor.click_delay_seconds):
            events.append(("campaign_clicked", max(1, int(round(float(donor.click_delay_seconds))))))
        events.sort(key=lambda item: item[1])
        blocked = journey in (forbidden_journey or set())
        next_journey = None if journey == "NO_JOURNEY_RESPONSE" or blocked else journey
        journey_delay = (None if next_journey is None or pd.isna(donor.journey_delay_seconds)
                         else max(1, int(round(float(donor.journey_delay_seconds)))))
        release = max(1, int(round(float(donor.observed_horizon_seconds))))
        options = [{
            "engagement_sequence": str(row.engagement_sequence),
            "next_journey_substage": str(row.next_journey_substage),
            "probability": float(probability), "count": int(row["count"]),
        } for (_, row), probability in zip(rows.iterrows(), probabilities)]
        evidence = {
            "model_version": self.manifest["model_version"],
            "model_hash": self.manifest["model_sha256"],
            "model_level": level, "backoff_level": level,
            "conditioning": keys, "support_count": int(selected.denominator),
            "candidate_distribution": options, "rng_draw": draw,
            "selected_engagement_sequence": engagement,
            "selected_journey_response": journey,
            "journey_response_blocked_by_guard": blocked,
            "timing_donor_index": donor_index,
            "causal_interpretation": "NONE_OBSERVATIONAL_ASSOCIATION_ONLY",
        }
        return CampaignMotifResult(tuple(events), next_journey, journey_delay, release, evidence)
