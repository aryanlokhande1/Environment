"""Minimal deterministic local Transformer/RL/Environment closed loop."""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
from uuid import uuid4
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from environment.contracts.action import EnvironmentAction
from environment.contracts.integration import RLPolicy, Transformer
from environment.contracts.result import EnvironmentResult
from environment.core.environment import Environment
from environment.core.state import EnvironmentState, reconstruct_state_from_gold
from environment.persistence.checkpoint import CheckpointStore, file_hash, atomic_bytes, ensure_directory
from environment.runtime.semantics import LEGACY, CORRECTED, FINAL
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
    corrected mode uses immutable snapshots and a final checkpoint commit marker;
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
        if self.environment.runtime_version != LEGACY and self.checkpoint.exists():
            stored = self.checkpoints.load(self.checkpoint)
            self._validate_identity(stored)
            self.checkpoints.recover_history(self.checkpoint, self.cumulative_history)
        if self.cumulative_history.exists():
            history = self.io.load_history(self.cumulative_history)
            if self.environment.runtime_version != LEGACY and not self.checkpoint.exists():
                if not history.equals(self.io.load_history(self.input_history)):
                    raise ValueError("cumulative history has no committed checkpoint")
            return history
        history = self.io.load_history(self.input_history)
        self.io.write_history(history, self.cumulative_history)
        return history

    def _validate_identity(self, stored: dict[str, Any]) -> None:
        expected = {"run_id": self.environment.run_id,
                    "runtime_version": self.environment.runtime_version,
                    "seed": self.environment.seed,
                    "artifact_hashes": self.environment.artifact_hashes}
        if self.environment.runtime_version == FINAL:
            expected["stochastic_namespace"] = self.environment.stochastic_namespace
            expected["simulation_start"] = None if self.environment.simulation_start is None else self.environment.simulation_start.isoformat()
            expected["simulation_end"] = None if self.environment.simulation_end is None else self.environment.simulation_end.isoformat()
        for key, value in expected.items():
            actual = stored.get(key, LEGACY if key == "runtime_version" else None)
            if actual != value:
                # Old checkpoints are only accepted for explicit legacy replay.
                if self.environment.runtime_version == LEGACY and key == "seed" and key not in stored:
                    continue
                raise ValueError(f"checkpoint configuration mismatch: {key}")
        if "input_history_sha256" in stored and file_hash(self.input_history) != stored["input_history_sha256"]:
            raise ValueError("input history hash mismatch")

    def _state(self, history: pd.DataFrame, application_id: str,
               decision_time: pd.Timestamp) -> EnvironmentState:
        if not self.checkpoint.exists():
            return reconstruct_state_from_gold(history, application_id, decision_time)
        stored = self.checkpoints.load(self.checkpoint)
        self._validate_identity(stored)
        if stored.get("run_id") != self.environment.run_id:
            raise ValueError("checkpoint run_id does not match Environment run_id")
        if str(stored.get("application_id")) != str(application_id):
            raise ValueError("checkpoint application_id does not match requested application")
        return EnvironmentState(dict(stored["state"]), list(stored.get("pending_events", [])))

    def run_step(self, application_id: str, decision_time: str | pd.Timestamp,
                 transformer: Transformer, policy: RLPolicy) -> ClosedLoopStep:
        if self.environment.runtime_version == FINAL:
            raise ValueError("corrected-v2 uses ExternalAgentSession.decide(action, next_decision_time)")
        # Serialize cooperating writers for the single-application local loop.
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with self.checkpoint.with_suffix(self.checkpoint.suffix + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._run_step(application_id, decision_time, transformer, policy)

    def _run_step(self, application_id: str, decision_time: str | pd.Timestamp,
                  transformer: Transformer, policy: RLPolicy) -> ClosedLoopStep:
        when = pd.Timestamp(decision_time)
        if when.tzinfo is not None:
            raise ValueError("decision_time must be timezone-naive")
        history = self._history()
        state = self._state(history, application_id, when)
        if self.environment.runtime_version != LEGACY and self.checkpoint.exists():
            stored = self.checkpoints.load(self.checkpoint)
            if stored["decision_time"] == when.isoformat():
                # This boundary already committed. Do not call either model or
                # append its Gold again on a lost-acknowledgement retry.
                action = EnvironmentAction.parse(stored["action"])
                result = EnvironmentResult(events=[], state=state, terminal=state.terminal,
                    pending_events=list(state.pending_events), explanations=[], decisions=[])
                return self._step_result(result, action, when)
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
        stored = {
            "run_id": self.environment.run_id, "runtime_version": self.environment.runtime_version,
            "seed": self.environment.seed, "input_history_sha256": file_hash(self.input_history),
            "action": {"campaign_sent": action.campaign_sent, "channel_id": action.channel_id,
                       "theme": action.theme, "time_bucket": action.time_bucket},
            "application_id": str(application_id),
            "decision_time": when.isoformat(),
            "state": result.state.payload,
            "pending_events": result.pending_events,
            "artifact_hashes": self.environment.artifact_hashes,
        }
        if self.environment.runtime_version != LEGACY:
            self._commit(stored, result.events)
        else:
            if result.events:
                self.io.append_history(result.events, self.cumulative_history)
            self.checkpoints.save(stored, self.checkpoint)
        return self._step_result(result, action, when)

    def _commit(self, stored: dict[str, Any], events: list[dict[str, Any]]) -> None:
        # Same manifest-last boundary as the day-wise runner. Staged snapshots
        # are never exposed as cumulative history before the commit marker.
        directory = self.checkpoint.parent / (self.checkpoint.name + ".commits") / uuid4().hex
        ensure_directory(directory)
        history_path, state_path = directory / "history.parquet", directory / "checkpoint.json"
        previous_hash = file_hash(self.cumulative_history)
        atomic_bytes(history_path, self.cumulative_history.read_bytes())
        if events:
            self.io.append_history(events, history_path)
        self._commit_hook("after_history")
        self.checkpoints.save(stored, state_path)
        self._commit_hook("after_checkpoint")
        marker = {"history": str(history_path.relative_to(self.checkpoint.parent)),
                  "checkpoint": str(state_path.relative_to(self.checkpoint.parent)),
                  "history_sha256": file_hash(history_path),
                  "checkpoint_sha256": file_hash(state_path),
                  "previous_history_sha256": previous_hash}
        self.checkpoints.save({**stored, "local_commit": marker}, self.checkpoint)
        self._commit_hook("after_commit")
        self.checkpoints.recover_history(self.checkpoint, self.cumulative_history)

    @staticmethod
    def _commit_hook(boundary: str) -> None:
        """Crash injection seam; no runtime behavior in production."""

    def _step_result(self, result: EnvironmentResult, action: EnvironmentAction,
                     when: pd.Timestamp) -> ClosedLoopStep:
        next_start = when.normalize() + pd.Timedelta(days=1)
        next_end = min(pd.Timestamp("2026-06-01"), pd.Timestamp(result.state.payload["deadline"]))
        opportunities = tuple(self.environment.get_decision_opportunities(
            result.state, next_start, next_end)) if next_start < next_end else ()
        return ClosedLoopStep(result, action, opportunities,
                              self.cumulative_history, self.checkpoint)
