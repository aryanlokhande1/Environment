"""Conditioned empirical SEND/NO_ACTION policy at daily opportunities."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def age_bucket(days: float) -> str:
    if days < 1:
        return "LT_1D"
    if days < 3:
        return "1D_3D"
    if days < 7:
        return "3D_7D"
    if days < 14:
        return "7D_14D"
    if days < 21:
        return "14D_21D"
    return "21D_30D"


def prior_send_bucket(count: int) -> str:
    return "0" if count == 0 else "1" if count == 1 else "2_3" if count <= 3 else "4_PLUS"


def recent_send_gap_bucket(seconds: float | None) -> str:
    if seconds is None:
        return "NO_PRIOR_SEND"
    days = seconds / 86_400
    return "LT_1D" if days < 1 else "1D_3D" if days < 3 else "3D_7D" if days < 7 else "7D_PLUS"


@dataclass(frozen=True)
class CampaignDecision:
    send: bool
    evidence: dict[str, Any]


class CampaignDecisionPolicy:
    KEYS = {
        "STATE_AGE_PRIOR_GAP": ("current_state", "age_bucket", "prior_send_bucket", "recent_send_gap_bucket"),
        "STATE_AGE_GAP": ("current_state", "age_bucket", "recent_send_gap_bucket"),
        "STATE_PRIOR_GAP": ("current_state", "prior_send_bucket", "recent_send_gap_bucket"),
        "STATE_GAP": ("current_state", "recent_send_gap_bucket"),
        "AGE_PRIOR_GAP": ("age_bucket", "prior_send_bucket", "recent_send_gap_bucket"),
        "AGE_GAP": ("age_bucket", "recent_send_gap_bucket"),
        "ORIGIN_GAP": ("population_origin", "recent_send_gap_bucket"),
        "GAP": ("recent_send_gap_bucket",),
        "STATE_AGE_PRIOR": ("current_state", "age_bucket", "prior_send_bucket"),
        "STATE_AGE": ("current_state", "age_bucket"),
        "STATE_PRIOR": ("current_state", "prior_send_bucket"),
        "STATE": ("current_state",),
        "AGE_PRIOR": ("age_bucket", "prior_send_bucket"),
        "AGE": ("age_bucket",),
        "ORIGIN": ("population_origin",),
        "GLOBAL": (),
    }

    def __init__(self, artifact_dir: Path) -> None:
        directory = Path(artifact_dir)
        manifest_path = directory / "campaign_decision_policy_manifest.json"
        model_path = directory / "campaign_decision_policy.csv"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("model_version") not in {
                "PL_ACTIVE_DAY_SEND_NO_ACTION_POLICY_V1", "PL_ACTIVE_DAY_SEND_NO_ACTION_POLICY_V2"}:
            raise ValueError("unsupported campaign decision policy")
        if sha256(model_path.read_bytes()).hexdigest() != self.manifest["model_sha256"]:
            raise ValueError("campaign decision policy hash mismatch")
        model = pd.read_csv(model_path, keep_default_na=False)
        self.hierarchy = list(self.manifest["conditioning_hierarchy"])
        self.rows: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
        for row in model.to_dict("records"):
            level = str(row["model_level"])
            keys = self.KEYS[level]
            self.rows[(level, tuple(str(row[key]) for key in keys))] = row
        if ("GLOBAL", ()) not in self.rows:
            raise ValueError("campaign policy lacks global fallback")

    def lookup(self, state: Mapping[str, Any], now: pd.Timestamp) -> dict[str, Any]:
        creation = pd.Timestamp(state["creation_datetime"])
        last_send = state.get("last_campaign_send_datetime")
        gap_seconds = (None if not last_send else
                       max(0.0, (pd.Timestamp(now) - pd.Timestamp(last_send)).total_seconds()))
        values = {
            "current_state": str(state["journey_substage"]),
            "age_bucket": age_bucket(max(0.0, (pd.Timestamp(now) - creation).total_seconds() / 86_400)),
            "prior_send_bucket": prior_send_bucket(int(state.get("campaign_sends_seen", 0))),
            "recent_send_gap_bucket": recent_send_gap_bucket(gap_seconds),
            "population_origin": ("CARRIED_FORWARD" if str(state.get("population_origin")) == "MAY1_ACTIVE_COHORT"
                                  else "NEW_ARRIVAL"),
        }
        for level in self.hierarchy:
            keys = self.KEYS[level]
            row = self.rows.get((level, tuple(values[key] for key in keys)))
            if row is not None:
                return {**row, "lookup_values": values, "backoff_level": level}
        raise RuntimeError("global campaign decision fallback is missing")

    def sample(self, state: Mapping[str, Any], now: pd.Timestamp,
               rng: np.random.Generator) -> CampaignDecision:
        row = self.lookup(state, now)
        draw = float(rng.random())
        probability = float(row["send_probability"])
        return CampaignDecision(draw < probability, {
            "model": "campaign_decision_policy",
            "model_version": self.manifest["model_version"],
            "model_hash": self.manifest["model_sha256"],
            "opportunity": "ACTIVE_APPLICATION_DAY",
            "conditioning_level": row["model_level"],
            "conditioning_keys": str(row["conditioning_keys"]).split("+") if row["conditioning_keys"] != "NONE" else [],
            "conditioning_values": row["lookup_values"],
            "support_count": int(row["support_count"]),
            "support_unit": row["support_unit"],
            "send_days": int(row["send_days"]),
            "candidate_distribution": {"SEND": probability, "NO_ACTION": 1.0 - probability},
            "rng_draw": draw, "selected_outcome": "SEND" if draw < probability else "NO_ACTION",
            "backoff_level": row["backoff_level"],
        })
