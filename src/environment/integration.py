"""Minimal deterministic local Transformer/RL/Environment closed loop."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from environment.contracts.action import EnvironmentAction
from environment.contracts.integration import RLPolicy, Transformer
from environment.contracts.result import EnvironmentResult
from environment.core.environment import Environment
from environment.core.state import EnvironmentState, reconstruct_state_from_gold
from environment.persistence.checkpoint import CheckpointStore
from environment.persistence.local_io import LocalIO


@dataclass(frozen=True)
class ClosedLoopStep:
    result: EnvironmentResult
    action: EnvironmentAction
    next_opportunities: tuple[pd.Timestamp, ...]
    cumulative_history_path: Path
    checkpoint_path: Path


class LocalClosedLoop:
    """Coordinate one locally persisted decision without implementing ML.

    The caller supplies the Transformer and RL policy. Both the Parquet file
    replacement and JSON checkpoint replacement are atomic local operations;
    live database/S3 transactions remain adapter responsibilities.
    """

    def __init__(self, environment: Environment, *, input_history: str | Path,
                 cumulative_history: str | Path, checkpoint: str | Path):
        self.environment = environment
        self.input_history = Path(input_history)
        self.cumulative_history = Path(cumulative_history)
        self.checkpoint = Path(checkpoint)
        self.io = LocalIO()
        self.checkpoints = CheckpointStore()

    def _history(self) -> pd.DataFrame:
        if self.cumulative_history.exists():
            return self.io.load_history(self.cumulative_history)
        history = self.io.load_history(self.input_history)
        self.io.write_history(history, self.cumulative_history)
        return history

    def _state(self, history: pd.DataFrame, application_id: str,
               decision_time: pd.Timestamp) -> EnvironmentState:
        if not self.checkpoint.exists():
            return reconstruct_state_from_gold(history, application_id, decision_time)
        stored = self.checkpoints.load(self.checkpoint)
        if stored.get("run_id") != self.environment.run_id:
            raise ValueError("checkpoint run_id does not match Environment run_id")
        if str(stored.get("application_id")) != str(application_id):
            raise ValueError("checkpoint application_id does not match requested application")
        return EnvironmentState(dict(stored["state"]), list(stored.get("pending_events", [])))

    def run_step(self, application_id: str, decision_time: str | pd.Timestamp,
                 transformer: Transformer, policy: RLPolicy) -> ClosedLoopStep:
        when = pd.Timestamp(decision_time)
        if when.tzinfo is not None:
            raise ValueError("decision_time must be timezone-naive")
        history = self._history()
        state = self._state(history, application_id, when)
        visible = history.loc[pd.to_datetime(history.event_datetime).le(when)].copy()
        embedding = np.asarray(transformer.encode(visible))
        if embedding.shape != (192,):
            raise ValueError(f"Transformer embedding must have shape (192,), got {embedding.shape}")
        action = EnvironmentAction.parse(policy.choose_action(embedding))
        mask = EnvironmentAction.mask(
            terminal=state.terminal,
            before_expiry=when < pd.Timestamp(state.payload["deadline"]),
        )
        if action.campaign_sent and not mask["SEND"]:
            raise ValueError("SEND is masked for this application state")
        result = self.environment.step(state, action, when)
        if result.events:
            self.io.append_history(result.events, self.cumulative_history)
        self.checkpoints.save({
            "run_id": self.environment.run_id,
            "application_id": str(application_id),
            "decision_time": when.isoformat(),
            "state": result.state.payload,
            "pending_events": result.pending_events,
            "artifact_hashes": self.environment.artifact_hashes,
        }, self.checkpoint)
        next_start = when.normalize() + pd.Timedelta(days=1)
        next_end = min(pd.Timestamp("2026-06-01"), pd.Timestamp(result.state.payload["deadline"]))
        opportunities = tuple(self.environment.get_decision_opportunities(
            result.state, next_start, next_end)) if next_start < next_end else ()
        return ClosedLoopStep(result, action, opportunities,
                              self.cumulative_history, self.checkpoint)
