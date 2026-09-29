"""Minimal frozen-runtime primitives used by the daily closed-loop environment."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pandas as pd
from .progression import MAPPING_SHA256

CONTRACT_VERSION = "2.0"
PRODUCT_SCOPE = "PERSONAL_LOAN"
THEME_PLACEHOLDER = "THEME_PLACEHOLDER"
CHANNELS = ("SMS", "WA", "GRCS")
TIME_BUCKETS = ("MORNING", "AFTERNOON", "EVENING", "NIGHT")
TRANSACTION_EVENT = "TRANSACTION_EVENT"
EXPIRY_REASON = "APPLICATION_EXPIRED_30_DAYS"
SUCCESS_REASON = "PUSH_TO_PARTNER_SUCCESS"
EVENT_PRIORITY = {TRANSACTION_EVENT: 0, "TERMINAL_EVENT": 1, "CAMPAIGN_SEND_EVENT": 2,
                  "CAMPAIGN_RESPONSE_EVENT": 3, "JOURNEY_EVENT": 4,
                  "ACTION_CONDITIONED_JOURNEY_EVENT": 4, "CAMPAIGN_MOTIF_RELEASE_EVENT": 5}

def stable_seed(*parts: object) -> int:
    return int.from_bytes(sha256("|".join(map(str, parts)).encode("utf-8")).digest()[:8], "big")

@dataclass(frozen=True)
class CampaignAction:
    channel: str
    theme: str = THEME_PLACEHOLDER
    time_bucket: str = "MORNING"

    def __post_init__(self) -> None:
        channel, theme, bucket = self.channel.strip().upper(), self.theme.strip().upper(), self.time_bucket.strip().upper()
        if channel not in CHANNELS: raise ValueError(f"unsupported canonical channel: {channel}")
        if theme != THEME_PLACEHOLDER: raise ValueError("only the approved placeholder theme is available")
        if bucket not in TIME_BUCKETS: raise ValueError(f"unsupported time bucket: {bucket}")
        object.__setattr__(self, "channel", channel)
        object.__setattr__(self, "theme", theme)
        object.__setattr__(self, "time_bucket", bucket)

    @property
    def action_id(self) -> str:
        return f"{self.channel}|{self.theme}|{self.time_bucket}"

@dataclass(frozen=True)
class PolicyChoice:
    action: CampaignAction | None
    decision_id: str | None = None
    policy_source: str = "BASELINE_DETERMINISTIC"

def can_approve_aip(state: Mapping[str, Any]) -> bool:
    return bool(state["form_filled_seen"] and state["professional_details_seen"])

def can_push_to_partner(state: Mapping[str, Any]) -> bool:
    return bool(can_approve_aip(state) and state["aip_approved_seen"] and not state["done"])

def _window(day: pd.Timestamp, bucket: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = day.normalize()
    ranges = {"MORNING": (6, 12), "AFTERNOON": (12, 17), "EVENING": (17, 21), "NIGHT": (21, 30)}
    low, high = ranges[bucket]
    return start + pd.Timedelta(hours=low), start + pd.Timedelta(hours=high)

def resolve_bucket_time(decision_time: pd.Timestamp | str, bucket: str,
                        deadline: pd.Timestamp | str, rng: np.random.Generator) -> pd.Timestamp | None:
    now, expiry, bucket = pd.Timestamp(decision_time), pd.Timestamp(deadline), bucket.strip().upper()
    if bucket not in TIME_BUCKETS: raise ValueError(f"unsupported time bucket: {bucket}")
    if now >= expiry: return None
    for day in (now.normalize() - pd.Timedelta(days=1), now.normalize(), now.normalize() + pd.Timedelta(days=1)):
        begin, end = _window(day, bucket)
        low, high = max(now, begin), min(expiry, end)
        lower_us, upper_us = (low.value + 999) // 1000, (high.value + 999) // 1000
        if lower_us < upper_us:
            return pd.Timestamp(int(rng.integers(lower_us, upper_us)), unit="us")
    return None

class PersonalLoanSimulator:
    """Loads only the model components consumed by the daily environment."""
    def __init__(self, artifact_dir: Path, *, seed: int, start: pd.Timestamp | str,
                 end: pd.Timestamp | str, baseline_action: CampaignAction | None = None,
                 max_events_per_context: int = 256):
        self.artifact_dir, self.seed = Path(artifact_dir), int(seed)
        self.start, self.end = pd.Timestamp(start), pd.Timestamp(end)
        if self.end <= self.start: raise ValueError("simulation end must be after start")
        self.baseline_action = baseline_action
        from .empirical_policy import EmpiricalBaselinePolicy
        self.empirical_policy = None if baseline_action is not None else EmpiricalBaselinePolicy(self.artifact_dir, seed=self.seed)
        self.max_events_per_context = max_events_per_context
        self.model_manifest = json.loads((self.artifact_dir / "model_manifest.json").read_text(encoding="utf-8"))
        manifest = self.model_manifest
        if manifest.get("product_scope") != PRODUCT_SCOPE or manifest.get("contract_version") != CONTRACT_VERSION:
            raise ValueError("a validated Personal Loan contract 2.0 artifact is required")
        if manifest.get("ptp_progression_mapping_sha256") != MAPPING_SHA256 or manifest.get("historical_full_ptp_retained") is not True:
            raise ValueError("PL model must retain full historical PTP evidence")
        if self.start != pd.Timestamp(manifest.get("snapshot_as_of")):
            raise ValueError("runtime start must match the frozen snapshot as-of time")
        transitions = pd.read_csv(self.artifact_dir / "natural_transition_model.csv")
        if manifest.get("ptp_transaction_excluded_from_natural") is not True or transitions.response.eq("Push to Partner").any():
            raise ValueError("PTP must be excluded from natural transitions")
        self.natural = {str(state): group.to_dict("records") for state, group in transitions.groupby("state", sort=False)}
        from .natural_transition import RegimeNaturalTransitionModel
        from .hazard import PtpHazardModel
        from .campaign_motif import CampaignMotifModel
        self.natural_regime = RegimeNaturalTransitionModel(self.artifact_dir)
        self.ptp_hazard, self.ptp_model = PtpHazardModel(self.artifact_dir), None
        stages = pd.read_csv(self.artifact_dir / "lifecycle_business_mapping.csv")
        self.stage_map = dict(zip(stages.journey_substage.astype(str), stages.journey_stage.astype(str)))
        self.campaign_motif = CampaignMotifModel(self.artifact_dir)
        self.action_model, self.campaign_response = None, {}

    def cohort(self, limit: int | None = None) -> list[dict[str, Any]]:
        snapshot = pd.read_parquet(self.artifact_dir / "starting_snapshot.parquet")
        lifecycle = pd.read_parquet(self.artifact_dir / "application_lifecycle_state.parquet")
        snapshot = snapshot.merge(lifecycle, on="application_id", how="left", validate="one_to_one")
        seen = pd.read_parquet(self.artifact_dir / "application_observed_substages.parquet")
        snapshot = snapshot.merge(seen, on="application_id", how="left", validate="one_to_one")
        if not snapshot.product_id.eq(PRODUCT_SCOPE).all(): raise ValueError("non-PL product in snapshot")
        if not snapshot.snapshot_context_id.is_unique: raise ValueError("multiple applications per starting context")
        ranked = sorted(snapshot.to_dict("records"), key=lambda row: sha256(f"{self.seed}|{row['snapshot_context_id']}".encode()).hexdigest())
        return ranked if limit is None else ranked[:limit]
