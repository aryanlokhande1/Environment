"""Deterministic observational PL baseline action propensity, not RL."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .base import CampaignAction, stable_seed


class EmpiricalBaselinePolicy:
    def __init__(self, artifact_dir: Path, *, seed: int):
        artifact_dir = Path(artifact_dir)
        path = artifact_dir / "empirical_action_propensity.csv"
        manifest = json.loads((artifact_dir / "empirical_action_propensity_manifest.json").read_text(encoding="utf-8"))
        if sha256(path.read_bytes()).hexdigest() != manifest["model_sha256"]:
            raise ValueError("empirical action propensity hash mismatch")
        if manifest["accepted_conditioning_levels"]["GLOBAL"] != 1:
            raise ValueError("global PL action propensity is not supported")
        self.model_sha256 = manifest["model_sha256"]
        self.seed = int(seed)
        model = pd.read_csv(path, keep_default_na=False)
        self.states = {str(key): group.to_dict("records") for key, group in
                       model.loc[model.model_level.eq("STATE")].groupby("current_journey_substage")}
        self.stages = {str(key): group.to_dict("records") for key, group in
                       model.loc[model.model_level.eq("STAGE")].groupby("current_journey_stage")}
        self.global_rows = model.loc[model.model_level.eq("GLOBAL")].to_dict("records")

    def sample(self, state: Mapping[str, Any], context_id: str, application_id: str,
               decision_sequence: int) -> tuple[CampaignAction, str]:
        substage = str(state["journey_substage"])
        stage = str(state["journey_stage"])
        if substage in self.states:
            rows, level = self.states[substage], "STATE"
        elif stage in self.stages:
            rows, level = self.stages[stage], "STAGE"
        else:
            rows, level = self.global_rows, "GLOBAL"
        probabilities = np.asarray([float(row["probability"]) for row in rows], dtype=float)
        probabilities /= probabilities.sum()
        rng = np.random.default_rng(stable_seed(self.seed, "empirical-action", context_id,
                                                 application_id, decision_sequence, substage, stage))
        selected = rows[int(rng.choice(len(rows), p=probabilities))]
        return CampaignAction(str(selected["channel"]), time_bucket=str(selected["time_bucket"])), level
