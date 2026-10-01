from pathlib import Path

import numpy as np
import pandas as pd

from environment import Environment, EnvironmentAction, EnvironmentState
from environment.runtime.base import CampaignAction, PolicyChoice
from environment.runtime.hazard import PtpHazardModel
from environment.runtime.natural_transition import RegimeNaturalTransitionModel

BUNDLE = Path(__file__).parents[1] / "artifacts" / "gold_events_v1"


def _quiet_state(runtime):
    raw = runtime.initial_population()[0]
    state = runtime.new_state(raw, activation=pd.Timestamp("2026-05-01"))
    state.update({
        "journey_stage": "UNSUPPORTED_TEST_STAGE",
        "journey_substage": "UNSUPPORTED_TEST_STATE",
        "observed_journey_substages": ["Application Created"],
        "form_filled_seen": False,
        "professional_details_seen": False,
        "aip_approved_seen": False,
        "latest_aip_datetime": None,
        "last_journey_datetime": "2026-05-01T00:00:00",
    })
    return state


def test_send_is_realized_and_gold_is_separate_from_audit():
    env = Environment(BUNDLE, run_id="send-realized")
    state = EnvironmentState(_quiet_state(env._runtime))
    result = env.step(state, EnvironmentAction.campaign("SMS", "MORNING"), "2026-05-01")
    assert result.decisions[0]["status"] == "REALIZED"
    sends = [row for row in result.events if row["journey_substage"] == "campaign_sent"]
    assert len(sends) == 1
    assert sends[0]["channel"] == "SMS"
    assert sends[0]["is_simulated"] is True
    assert all("explanation" not in row for row in result.events)
    assert any(row["category"] == "CAMPAIGN_RESPONSE_MOTIF" for row in result.explanations)


def test_competing_send_censors_prior_pending_response():
    env = Environment(BUNDLE, run_id="censor")
    runtime = env._runtime
    state = _quiet_state(runtime)
    state["pending_sequence"] = 1
    pending = [{
        "when": "2026-05-01T18:00:00", "priority": 3, "order": 1,
        "kind": "CAMPAIGN_RESPONSE_EVENT",
        "payload": {"decision_id": "old", "response": "campaign_open_read"},
    }]

    def policy(_, __):
        return PolicyChoice(CampaignAction("SMS", time_bucket="MORNING"),
                            policy_source="TEST_POLICY")

    events, _, _ = runtime.process_day(state, pending, pd.Timestamp("2026-05-01"), policy=policy)
    assert not any(row.get("decision_id") == "old" for row in events)
    censor = [row for row in runtime.audit_records
              if row["category"] == "CAMPAIGN_RESPONSE_CENSORING"]
    assert censor and censor[0]["explanation"]["reason"] == "COMPETING_SEND_BOUNDARY"


def test_terminal_event_suppresses_a_scheduled_send():
    env = Environment(BUNDLE, run_id="terminal-suppression")
    runtime = env._runtime
    state = _quiet_state(runtime)
    state.update({
        "form_filled_seen": True, "professional_details_seen": True,
        "aip_approved_seen": True, "latest_aip_datetime": "2026-05-01T00:00:00",
        "ptp_generation": 1,
    })
    state["pending_sequence"] = 1
    pending = [{
        "when": "2026-05-01T01:00:00", "priority": 0, "order": 1,
        "kind": "TRANSACTION_EVENT", "payload": {
            "response": "Push to Partner", "ptp_outcome": True,
            "ptp_generation": 1, "support_count": 1, "model_level": "TEST",
        },
    }]

    def policy(_, __):
        return PolicyChoice(CampaignAction("SMS", time_bucket="MORNING"),
                            policy_source="TEST_POLICY")

    events, decisions, _ = runtime.process_day(
        state, pending, pd.Timestamp("2026-05-01"), policy=policy)
    assert state["done"] and state["termination_reason"] == "PUSH_TO_PARTNER_SUCCESS"
    assert decisions[0]["status"] == "SUPPRESSED_BY_TERMINAL"
    assert decisions[0]["suppression_reason"] == "PUSH_TO_PARTNER_SUCCESS"
    assert not any(row["event_type"] == "CAMPAIGN_SEND_EVENT" for row in events)


def test_first_pass_revisit_and_ptp_hazard_contracts():
    model = RegimeNaturalTransitionModel(BUNDLE)
    base = {"journey_substage": "Application Resume", "observed_journey_substages": []}
    assert model.classify(base) == "FIRST_PASS"
    revisit = {**base, "observed_journey_substages": ["Professional Details Submission"]}
    assert model.classify(revisit) == "REVISIT"

    hazard = PtpHazardModel(BUNDLE)
    state = {
        "creation_datetime": "2026-05-01T00:00:00",
        "latest_aip_datetime": "2026-05-01T01:00:00",
        "deadline": "2026-05-31T00:00:00",
    }
    outcomes = [hazard.sample_window(
        state, pd.Timestamp("2026-05-01T01:00:00"), pd.Timestamp("2026-05-02"),
        np.random.default_rng(seed)) for seed in range(50)]
    fired = [result for result in outcomes if result.when is not None]
    assert fired
    assert all(pd.Timestamp("2026-05-01T01:00:00") <= result.when < pd.Timestamp("2026-05-02")
               for result in fired)
