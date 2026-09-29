"""Validated application-lifecycle regime lookup for natural transitions."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

PD = frozenset({"Professional Details Submission", "Professional details"})
DOWNSTREAM = frozenset({
    *PD, "AIP Approved", "listing", "Product Selection", "Address Submission",
    "Bank Approval Pending", "Bank Approved", "Push to Partner",
})
FOCUS = frozenset({"Application OTP Verification", "Application Resume"})


@dataclass(frozen=True)
class NaturalLookup:
    options: list[dict[str, Any]]
    regime: str
    model_level: str
    support_count: int
    backoff_level: str


class RegimeNaturalTransitionModel:
    def __init__(self, artifact_dir: Path) -> None:
        directory = Path(artifact_dir)
        base_path = directory / "natural_transition_model.csv"
        manifest_path = directory / "natural_transition_regime_manifest.json"
        regime_path = directory / "natural_transition_regime_model.csv"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("model_version") != "PL_APPLICATION_NATURAL_STATE_REGIME_V1":
            raise ValueError("unsupported natural regime artifact")
        if sha256(base_path.read_bytes()).hexdigest() != self.manifest["base_natural_sha256"]:
            raise ValueError("base natural transition artifact hash mismatch")
        if sha256(regime_path.read_bytes()).hexdigest() != self.manifest["model_sha256"]:
            raise ValueError("natural regime transition artifact hash mismatch")
        base = pd.read_csv(base_path)
        regime = pd.read_csv(regime_path)
        if base.response.eq("Push to Partner").any() or regime.response.eq("Push to Partner").any():
            raise ValueError("PTP leaked into natural transition lookup")
        self.base = {str(key): group.to_dict("records")
                     for key, group in base.groupby("state", observed=True, sort=False)}
        self.regime = {(str(state), str(kind)): group.to_dict("records")
                       for (state, kind), group in regime.groupby(["state", "regime"], observed=True, sort=False)}

    @staticmethod
    def classify(state: Mapping[str, Any]) -> str:
        current = str(state.get("journey_substage", ""))
        if current not in FOCUS:
            return "UNCONDITIONED"
        seen = set(map(str, state.get("observed_journey_substages", ())))
        return "REVISIT" if seen & DOWNSTREAM else "FIRST_PASS"

    def lookup(self, state: Mapping[str, Any]) -> NaturalLookup | None:
        current = str(state.get("journey_substage", ""))
        regime = self.classify(state)
        rows = self.regime.get((current, regime))
        level = "STATE_REGIME"
        backoff = "NONE"
        if rows is None and current in FOCUS:
            rows = self.regime.get(("__FOCUS__", regime))
            level, backoff = "FOCUS_REGIME", "POOLED_FOCUS_REGIME"
        if rows is None:
            rows = self.base.get(current)
            level, backoff = "STATE", "UNCONDITIONED_STATE"
        if not rows:
            return None
        support = int(rows[0].get("outgoing_transition_count", sum(int(row["support_count"]) for row in rows)))
        return NaturalLookup([dict(row) for row in rows], regime, level, support, backoff)
