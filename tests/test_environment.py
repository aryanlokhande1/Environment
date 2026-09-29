from pathlib import Path
import pandas as pd
from environment import Environment, EnvironmentAction, EnvironmentState

BUNDLE = Path(__file__).parents[1] / "artifacts" / "gold_events_v1"

def test_no_action_step_is_deterministic_and_audited():
    first = Environment(BUNDLE, run_id="test-a")
    raw = first._runtime.initial_population()[0]
    payload = first._runtime.new_state(raw, activation=pd.Timestamp("2026-05-01T00:00:00"))
    result = first.step(EnvironmentState(payload), EnvironmentAction.no_action(), "2026-05-01T00:00:00")
    second = Environment(BUNDLE, run_id="test-a")
    replay_payload = second._runtime.new_state(raw, activation=pd.Timestamp("2026-05-01T00:00:00"))
    replay = second.step(EnvironmentState(replay_payload), EnvironmentAction.no_action(), "2026-05-01T00:00:00")
    assert result.decisions[0]["status"] == "NO_ACTION"
    assert result.explanations
    assert all("is_simulated" in row and row["is_simulated"] for row in result.events)
    assert result.events == replay.events
    assert result.pending_events == replay.pending_events

def test_campaign_action_contract_and_opportunity_api():
    action = EnvironmentAction.campaign("sms", "afternoon")
    assert action.channel_id == "SMS"
    state = EnvironmentState({"done": False})
    points = Environment.get_decision_opportunities(state, "2026-05-02", "2026-05-03")
    assert len(points) == 1
