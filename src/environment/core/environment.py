"""Public single-application world-model API."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import pandas as pd

from environment.contracts.action import EnvironmentAction
from environment.contracts.result import EnvironmentResult
from environment.models.artifact_loader import ArtifactLoader
from environment.runtime.base import CampaignAction, PolicyChoice
from environment.runtime.daily_closed_loop import DailyClosedLoopEnvironment
from environment.runtime.gold_events_adapter import EventNameMapper, to_gold_row
from environment.runtime.semantics import CORRECTED, LEGACY, FINAL, validate_runtime_version
from .state import EnvironmentState
from .scheduler import get_decision_opportunities

class Environment:
    """Frozen empirical environment; it never fits or mutates model artifacts."""
    def __init__(self, artifact_dir: str | Path = "artifacts/gold_events_v2", *,
                 seed: int = 20260502, run_id: str = "environment-local",
                 runtime_version: str = CORRECTED, stochastic_namespace: str | None = None,
                 simulation_start: str | None = None, simulation_end: str | None = None):
        self.runtime_version = validate_runtime_version(runtime_version)
        self.stochastic_namespace = (str(stochastic_namespace) if stochastic_namespace is not None else "corrected-v2-reference-20260502") if self.runtime_version == FINAL else None
        if self.runtime_version == FINAL and not self.stochastic_namespace:
            raise ValueError("stochastic_namespace must be non-empty")
        if self.runtime_version != FINAL and stochastic_namespace is not None:
            raise ValueError("stochastic_namespace requires corrected-v2")
        self.simulation_start = pd.Timestamp(simulation_start) if simulation_start is not None else None
        self.simulation_end = pd.Timestamp(simulation_end) if simulation_end is not None else None
        for bound in (self.simulation_start, self.simulation_end):
            if bound is not None and (pd.isna(bound) or bound.tzinfo is not None):
                raise ValueError("simulation bounds must be valid timezone-naive timestamps")
        if self.simulation_start is not None and self.simulation_end is not None and self.simulation_end <= self.simulation_start:
            raise ValueError("simulation_end must follow simulation_start")
        self.loader = ArtifactLoader(artifact_dir)
        self.artifact_hashes = self.loader.validate()
        if self.runtime_version == FINAL and self.loader.manifest.get("bundle_version") != "gold_events_v3":
            raise ValueError("corrected-v2 requires gold_events_v3 empirical contract")
        if self.runtime_version != FINAL and self.loader.manifest.get("bundle_version") == "gold_events_v3":
            raise ValueError("gold_events_v3 requires corrected-v2 semantics")
        self.seed, self.run_id = int(seed), str(run_id)
        self.event_name_mapper = (
            EventNameMapper(self.loader.path("event_name_mapping"), seed=self.seed)
            if "event_name_mapping" in self.loader.records else None
        )
        self._runtime = DailyClosedLoopEnvironment(Path(artifact_dir), seed=self.seed, run_id=self.run_id,
                                                     runtime_version=self.runtime_version, stochastic_namespace=self.stochastic_namespace)

    @staticmethod
    def get_decision_opportunities(state: EnvironmentState, start_time: str | pd.Timestamp,
                                   end_time: str | pd.Timestamp) -> list[pd.Timestamp]:
        return get_decision_opportunities(state, start_time, end_time)

    def step(self, state: EnvironmentState, action: EnvironmentAction | Mapping[str, Any],
             decision_time: str | pd.Timestamp) -> EnvironmentResult:
        when = pd.Timestamp(decision_time)
        if self.runtime_version == FINAL:
            raise ValueError("corrected-v2 uses advance or ExternalAgentSession.decide with an explicit next decision time")
        previous = state.payload.get("_last_decision_time")
        if previous is not None and when.normalize() <= pd.Timestamp(previous).normalize():
            raise ValueError("at most one decision is supported per active application-day")
        return self._advance(state, action, when, when.normalize() + pd.Timedelta(days=1),
                             exact_send_time=False)

    def advance(self, state: EnvironmentState, action: EnvironmentAction | Mapping[str, Any],
                start_time: str | pd.Timestamp, end_time: str | pd.Timestamp, *,
                end_inclusive: bool = True) -> EnvironmentResult:
        """Apply an external action at ``start_time`` and advance to ``end_time``.

        ``NO_ACTION`` is the WAIT action. Organic and already-pending events may
        occur inside the interval, but the interval is never shortened to ask
        for another decision. A SEND is realized at the externally selected
        start timestamp and its empirically sampled effects may remain pending.
        """
        return self._advance(state, action, pd.Timestamp(start_time), pd.Timestamp(end_time),
                             exact_send_time=True, end_inclusive=end_inclusive)

    def _advance(self, state: EnvironmentState, action: EnvironmentAction | Mapping[str, Any],
                 when: pd.Timestamp, horizon: pd.Timestamp, *, exact_send_time: bool,
                 end_inclusive: bool = True) -> EnvironmentResult:
        action = EnvironmentAction.parse(action)
        saved_version = state.payload.get("_runtime_version",
            LEGACY if state.payload.get("_environment_started") else None)
        if saved_version is not None and saved_version != self.runtime_version:
            raise ValueError("state runtime_version does not match Environment")
        if state.terminal:
            raise ValueError("terminal application cannot be stepped")
        if when.tzinfo is not None or horizon.tzinfo is not None:
            raise ValueError("simulation timestamps must be timezone-naive")
        if pd.isna(when) or pd.isna(horizon) or not when < horizon:
            raise ValueError("simulation interval must be increasing and valid")
        if self.runtime_version == FINAL:
            if self.simulation_start is not None and when < self.simulation_start:
                raise ValueError("start_time is before configured simulation horizon")
            if self.simulation_end is not None and horizon > self.simulation_end:
                raise ValueError("end_time exceeds configured simulation horizon")
            saved_namespace = state.payload.get("_stochastic_namespace")
            if saved_namespace is not None and saved_namespace != self.stochastic_namespace:
                raise ValueError("state stochastic_namespace mismatch")
        elif not pd.Timestamp("2026-05-01") <= when < horizon <= pd.Timestamp("2026-06-01"):
            raise ValueError("simulation interval must be increasing and contained in May 2026")
        deadline = state.payload.get("deadline")
        if deadline is not None and when >= pd.Timestamp(deadline):
            raise ValueError("decision_time is at or after application expiry")
        current = state.payload.get("_simulation_time")
        if current is not None and when != pd.Timestamp(current):
            raise ValueError("start_time must equal the previously committed simulation time")
        state.payload["_runtime_version"] = self.runtime_version
        campaign = None if not action.campaign_sent else CampaignAction(
            action.channel_id or "", action.theme or "THEME_PLACEHOLDER", action.time_bucket or "")
        def policy(_: dict[str, Any], __: pd.Timestamp) -> PolicyChoice:
            return PolicyChoice(campaign, policy_source="EXTERNAL_RL_ACTION")
        first_step = not bool(state.payload.get("_environment_started", False))
        if first_step:
            state.payload["activation_datetime"] = when.isoformat()
        if self.runtime_version == FINAL:
            events, decisions, explanations = [], [], []
            cursor = when
            first_segment = True
            while cursor < horizon and not state.terminal:
                boundary = min(horizon, cursor.normalize() + pd.Timedelta(days=1))
                segment_events, segment_decisions, _ = self._runtime.process_day(
                    state.payload, state.pending_events, cursor.normalize(),
                    newly_activated=first_step and first_segment, policy=policy,
                    decision_time=cursor, horizon=boundary, action_at_decision_time=True,
                    decision_enabled=first_segment,
                    include_horizon_events=(boundary < horizon or end_inclusive))
                events.extend(segment_events); decisions.extend(segment_decisions)
                explanations.extend(self._runtime.audit_records)
                cursor = boundary
                first_segment = False
            self._runtime.audit_records = explanations
        else:
            events, decisions, _ = self._runtime.process_day(
                state.payload, state.pending_events, when.normalize(), newly_activated=first_step,
                policy=policy, decision_time=when, horizon=horizon,
                action_at_decision_time=exact_send_time)
        state.payload["_environment_started"] = True
        state.payload["_last_decision_time"] = when.isoformat()
        state.payload["_simulation_time"] = horizon.isoformat()
        gold = [row for event in events
                if (row := to_gold_row(event, self.event_name_mapper)) is not None]
        return EnvironmentResult(
            events=gold, state=state, terminal=state.terminal,
            pending_events=list(state.pending_events),
            explanations=list(self._runtime.audit_records), decisions=decisions,
        )
