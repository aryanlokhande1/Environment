import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from environment import Environment, EnvironmentAction
from environment.core.state import EnvironmentState, active_application_for_context
from environment.runtime.base import EVENT_PRIORITY, PolicyChoice
from environment.runtime.gold_events_adapter import GOLD_COLUMNS
from environment.runtime.natural_transition import NaturalLookup
from environment.simulation import MaySimulationRunner, _register_lifecycle


ROOT = Path(__file__).parents[1]
BUNDLE = ROOT / "artifacts" / "gold_events_v2"


def _record(context="context", application="application", creation="2026-05-01"):
    created = pd.Timestamp(creation)
    return {
        "context_id": context,
        "application_id": application,
        "creation_datetime": created.isoformat(),
        "deadline": (created + pd.Timedelta(days=30)).isoformat(),
    }


def _arrival_state(run_id="lifecycle-test"):
    environment = Environment(BUNDLE, run_id=run_id)
    raw = environment._runtime.arrivals.iloc[0].to_dict()
    when = pd.Timestamp(raw["activation_datetime"])
    return environment, when, EnvironmentState(environment._runtime.new_state(raw, activation=when))


def test_new_context_emits_exactly_one_application_created():
    environment, when, state = _arrival_state("one-created")
    result = environment.advance(
        state, EnvironmentAction.no_action(), when, when.normalize() + pd.Timedelta(days=1))
    assert sum(row["journey_substage"] == "Application Created" for row in result.events) == 1


def test_same_context_next_day_reuses_application():
    lifecycle = _record()
    assert active_application_for_context([lifecycle], "context", "2026-05-02") == "application"


def test_same_context_day_29_reuses_application():
    lifecycle = _record()
    assert active_application_for_context([lifecycle], "context", "2026-05-30") == "application"


def test_expiry_boundary_terminates_old_application():
    environment, _, state = _arrival_state("expiry-boundary")
    state.payload.update(_record(state.payload["context_id"], state.payload["application_id"]))
    state.payload["activation_datetime"] = "2026-05-01T00:00:00"
    pending = [{
        "when": "2026-05-31T00:00:00", "priority": EVENT_PRIORITY["TERMINAL_EVENT"],
        "order": 1, "kind": "TERMINAL_EVENT", "payload": {},
    }]
    events, _, _ = environment._runtime.process_day(
        state.payload, pending, pd.Timestamp("2026-05-31"),
        policy=lambda *_: PolicyChoice(None, policy_source="TEST"))
    assert state.terminal
    assert state.payload["termination_reason"] == "APPLICATION_EXPIRED_30_DAYS"
    assert events[0]["event_datetime"] == pd.Timestamp("2026-05-31")


def test_new_lifecycle_allowed_only_after_expiry():
    registry = {"context": [_record()]}
    assert active_application_for_context(registry["context"], "context", "2026-05-30") == "application"
    assert active_application_for_context(registry["context"], "context", "2026-05-31") is None
    _register_lifecycle(registry, _record("context", "application-2", "2026-05-31"))
    assert active_application_for_context(
        registry["context"], "context", "2026-05-31") == "application-2"


def test_valid_carried_application_is_reused_in_may():
    historical = _record("carried-context", "historical-app", "2026-04-15")
    registry = {"carried-context": [historical]}
    assert active_application_for_context(
        registry["carried-context"], "carried-context", "2026-05-01") == "historical-app"
    with pytest.raises(RuntimeError, match="overlapping application lifecycles"):
        _register_lifecycle(registry, _record("carried-context", "synthetic-app", "2026-05-01"))


def test_natural_application_created_proposal_is_blocked_without_renormalization(monkeypatch):
    environment, when, state = _arrival_state("blocked-created")
    runtime = environment._runtime
    lookup = NaturalLookup([{
        "response": "Application Created", "probability": 1.0,
        "support_count": 19, "model_level": "TEST",
    }], "REVISIT", "TEST", 19, "NONE")
    monkeypatch.setattr(runtime.sim.natural_regime, "lookup", lambda _: lookup)
    monkeypatch.setattr(runtime.continuation, "sample_window", lambda *_: SimpleNamespace(
        when=when + pd.Timedelta(hours=1), evidence={"model_level": "TEST"}))
    runtime.audit_records = []
    pending = []
    runtime._schedule_natural(state.payload, pending, when, when.normalize() + pd.Timedelta(days=1))
    assert pending == []
    evidence = runtime.audit_records[-1]["explanation"]
    assert evidence["reason"] == "ACTIVE_LIFECYCLE_ALREADY_CREATED"
    assert evidence["business_guards"]["blocked_mass_renormalized"] is False
    assert evidence["candidate_distribution"] == [{
        "response": "Application Created", "probability": 1.0}]


def test_ptp_is_terminal_and_clears_future_events():
    environment, when, state = _arrival_state("ptp-terminal")
    state.payload.update({
        "form_filled_seen": True, "professional_details_seen": True,
        "aip_approved_seen": True, "latest_aip_datetime": when.isoformat(),
        "ptp_generation": 1,
    })
    pending = [
        {"when": (when + pd.Timedelta(minutes=1)).isoformat(), "priority": 0, "order": 1,
         "kind": "TRANSACTION_EVENT", "payload": {
             "response": "Push to Partner", "ptp_outcome": True, "ptp_generation": 1,
             "support_count": 1, "model_level": "TEST"}},
        {"when": (when + pd.Timedelta(minutes=2)).isoformat(), "priority": 4, "order": 2,
         "kind": "JOURNEY_EVENT", "payload": {
             "response": "Application Resume", "natural_generation": 0}},
    ]
    events, _, _ = environment._runtime.process_day(
        state.payload, pending, when.normalize(), newly_activated=False,
        policy=lambda *_: PolicyChoice(None, policy_source="TEST"), decision_time=when)
    assert state.terminal and pending == []
    assert [row["journey_substage"] for row in events] == ["Push to Partner"]


def test_checkpoint_resume_preserves_active_application_identity(tmp_path):
    march, april = tmp_path / "march.parquet", tmp_path / "april.parquet"
    def historical(when):
        row = {column: "fixture" for column in GOLD_COLUMNS}
        row.update(event_datetime=pd.Timestamp(when), visitnum=1.0, day_of_week_num=1)
        return pd.DataFrame([row])
    historical("2026-03-01").to_parquet(march)
    historical("2026-04-01").to_parquet(april)
    runner = MaySimulationRunner(
        run_id="identity-resume", historical_sources=[str(march), str(april)],
        artifact_dir=BUNDLE, output_root=tmp_path / "runs", initial_limit=1, arrival_limit=1)
    runner.initialize()
    runner.run_day("2026-05-01")
    with gzip.open(runner.run_dir / "checkpoints" / "2026-05-01.json.gz", "rt") as stream:
        checkpoint = json.load(stream)
    initial = next(record for rows in checkpoint["lifecycle_registry"].values() for record in rows
                   if not record["application_id"].startswith("MAYAPP"))
    records = checkpoint["lifecycle_registry"][initial["context_id"]]
    assert active_application_for_context(records, initial["context_id"], "2026-05-01") == initial["application_id"]
    assert MaySimulationRunner.open("identity-resume", output_root=tmp_path / "runs").run_day(
        "2026-05-01")["status"] == "IDEMPOTENT_RETRY"


def test_active_lookup_has_no_cross_context_application_leakage():
    records = [_record("context-a", "app-a"), _record("context-b", "app-b")]
    assert active_application_for_context(records, "context-a", "2026-05-02") == "app-a"
    assert active_application_for_context(records, "context-b", "2026-05-02") == "app-b"
    assert active_application_for_context(records, "context-c", "2026-05-02") is None
