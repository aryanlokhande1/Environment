"""Public single-application world-model API."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd

from environment.contracts.action import EnvironmentAction
from environment.contracts.result import EnvironmentResult
from environment.models.artifact_loader import ArtifactLoader
from environment.runtime.base import CampaignAction, PolicyChoice
from environment.runtime.daily_closed_loop import DailyClosedLoopEnvironment
from environment.runtime.gold_events_adapter import to_gold_row
from .state import EnvironmentState
from .scheduler import get_decision_opportunities

class Environment:
    """Frozen empirical environment; it never fits or mutates model artifacts."""
    def __init__(self, artifact_dir: str | Path = "artifacts/gold_events_v1", *,
                 seed: int = 20260502, run_id: str = "environment-local"):
        self.loader = ArtifactLoader(artifact_dir)
        self.artifact_hashes = self.loader.validate()
        self.seed, self.run_id = int(seed), str(run_id)
        self._runtime = DailyClosedLoopEnvironment(Path(artifact_dir), seed=self.seed, run_id=self.run_id)

    @staticmethod
    def get_decision_opportunities(state: EnvironmentState, start_time: str | pd.Timestamp,
                                   end_time: str | pd.Timestamp) -> list[pd.Timestamp]:
        return get_decision_opportunities(state, start_time, end_time)

    def step(self, state: EnvironmentState, action: EnvironmentAction,
             decision_time: str | pd.Timestamp) -> EnvironmentResult:
        if state.terminal:
            raise ValueError("terminal application cannot be stepped")
        when = pd.Timestamp(decision_time)
        if when.tzinfo is not None:
            when = when.tz_localize(None)
        if not pd.Timestamp("2026-05-01") <= when < pd.Timestamp("2026-06-01"):
            raise ValueError("frozen gold_events_v1 runtime supports decision dates in May 2026")
        campaign = None if not action.campaign_sent else CampaignAction(
            action.channel_id or "", action.theme or "THEME_PLACEHOLDER", action.time_bucket or "")
        def policy(_: dict[str, Any], __: pd.Timestamp) -> PolicyChoice:
            return PolicyChoice(campaign, policy_source="EXTERNAL_RL_ACTION")
        first_step = not bool(state.payload.get("_environment_started", False))
        if first_step:
            state.payload["activation_datetime"] = when.isoformat()
        events, decisions, _ = self._runtime.process_day(
            state.payload, state.pending_events, when.normalize(), newly_activated=first_step, policy=policy)
        state.payload["_environment_started"] = True
        gold = [row for event in events if (row := to_gold_row(event)) is not None]
        return EnvironmentResult(
            events=gold, state=state, terminal=state.terminal,
            pending_events=list(state.pending_events),
            explanations=list(self._runtime.audit_records), decisions=decisions,
        )
