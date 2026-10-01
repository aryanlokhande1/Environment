from hashlib import sha256
import json
import gzip
from pathlib import Path

import pandas as pd

from environment import Environment, EnvironmentAction, EnvironmentState
from environment.runtime.gold_events_adapter import GOLD_COLUMNS
from environment.simulation import MaySimulationRunner, ScriptedSend

ROOT = Path(__file__).parents[1]
BUNDLE = ROOT / "artifacts" / "gold_events_v1"


def _historical_row(when, application_id):
    row = {column: "historical" for column in GOLD_COLUMNS}
    row.update({
        "application_id": application_id, "context_id": f"ctx-{application_id}",
        "event_datetime": pd.Timestamp(when), "event_name": None,
        "journey_stage": "Application", "journey_substage": "Application Created",
        "product_id": "PERSONAL_LOAN", "visitnum": 1.0,
        "day_of_week_num": pd.Timestamp(when).weekday() + 1,
    })
    return row


def _historical_files(tmp_path):
    march = tmp_path / "fixture_march.parquet"
    april = tmp_path / "fixture_april.parquet"
    pd.DataFrame([_historical_row("2026-03-01", "history-march")]).to_parquet(march, index=False)
    pd.DataFrame([_historical_row("2026-04-01", "history-april")]).to_parquet(april, index=False)
    return march, april


def test_may_daily_append_resume_and_combined_export(tmp_path):
    march, april = _historical_files(tmp_path)
    probe = Environment(BUNDLE, run_id="probe-may")
    application_id = probe._runtime.initial_population()[0]["application_id"]
    source_hashes = {path: sha256(path.read_bytes()).hexdigest() for path in (march, april)}
    runner = MaySimulationRunner(
        run_id="fixture-run", historical_sources=[str(march), str(april)],
        artifact_dir=BUNDLE, output_root=tmp_path / "runs",
        initial_limit=1, arrival_limit=1,
        scripted_sends=[ScriptedSend(application_id, "2026-05-01T00:00:00", "SMS", "MORNING")],
    )
    manifest = runner.initialize()
    assert manifest["historical_rows"] == 2
    may1 = runner.run_day("2026-05-01")
    assert may1["gold_rows"] >= 1
    assert may1["campaign_sends"] == 1
    with gzip.open(runner.run_dir / may1["checkpoint"], "rt", encoding="utf-8") as stream:
        may1_checkpoint = json.load(stream)
    may1_rows = pd.read_parquet(runner.run_dir / may1["gold_partition"])
    latest_key = may1_rows.loc[
        may1_rows.application_id.astype(str).eq(str(application_id)), "cumulative_row_key"].iloc[-1]
    assert may1_checkpoint["states"][str(application_id)]["last_visible_row_key"] == latest_key
    may2 = runner.run_day("2026-05-02")
    assert may2["status"] == "COMMITTED"
    assert runner.run_day("2026-05-02")["status"] == "IDEMPOTENT_RETRY"
    assert all(sha256(path.read_bytes()).hexdigest() == digest
               for path, digest in source_hashes.items())

    exported = runner.export_combined()
    report = runner.validate_run(require_complete=False)
    assert exported["rows"] == 2 + may1["gold_rows"] + may2["gold_rows"]
    assert report["status"] == "PASS"
    combined = pd.read_parquet(runner.combined_path)
    assert list(combined.columns) == list(GOLD_COLUMNS)
    assert len(combined) == exported["rows"]
    assert json.loads(runner.summary_path.read_text())["completed_day"] == "2026-05-02"


def test_wait_interval_and_pending_effect_across_day_boundary():
    env = Environment(BUNDLE, run_id="wait-contract")
    raw = env._runtime.initial_population()[0]
    payload = env._runtime.new_state(raw, activation=pd.Timestamp("2026-05-01"))
    payload.update({
        "journey_stage": "UNSUPPORTED_TEST_STAGE",
        "journey_substage": "UNSUPPORTED_TEST_STATE",
        "observed_journey_substages": ["Application Created"],
        "form_filled_seen": False, "professional_details_seen": False,
        "aip_approved_seen": False, "latest_aip_datetime": None,
        "last_journey_datetime": "2026-05-01T00:00:00",
    })
    state = EnvironmentState(payload)
    waited = env.advance(state, EnvironmentAction.no_action(),
                         "2026-05-01T00:00:00", "2026-05-01T03:00:00")
    assert waited.events == []
    assert state.payload["_simulation_time"] == "2026-05-01T03:00:00"

    state.payload["_simulation_time"] = "2026-05-02T00:00:00"
    state.payload["_last_decision_time"] = "2026-05-01T00:00:00"
    state.pending_events.append({
        "when": "2026-05-02T01:00:00", "priority": 3, "order": 99,
        "kind": "CAMPAIGN_RESPONSE_EVENT",
        "payload": {
            "decision_id": "prior", "action_id": "SMS|THEME_PLACEHOLDER|NIGHT",
            "channel": "SMS", "theme": "THEME_PLACEHOLDER", "time_bucket": "NIGHT",
            "decision_time": "2026-05-01T23:00:00",
            "resolved_send_time": "2026-05-01T23:00:00",
            "policy_source": "SCRIPTED_TEST", "response": "campaign_open_read",
        },
    })
    resumed = env.advance(state, EnvironmentAction.no_action(),
                          "2026-05-02T00:00:00", "2026-05-02T02:00:00")
    assert [row["journey_substage"] for row in resumed.events] == ["campaign_open_read"]
    assert resumed.events[0]["event_datetime"] == pd.Timestamp("2026-05-02T01:00:00")
