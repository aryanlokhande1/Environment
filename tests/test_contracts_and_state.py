import pandas as pd
import pytest

from environment import EnvironmentAction, EnvironmentState, reconstruct_state_from_gold
from environment.core.scheduler import get_decision_opportunities


def _row(application_id, context_id, when, substage="Application Created"):
    return {
        "application_id": application_id,
        "context_id": context_id,
        "event_datetime": pd.Timestamp(when),
        "journey_stage": "Application",
        "journey_substage": substage,
    }


def test_action_parsing_and_masking_are_strict():
    action = EnvironmentAction.parse({
        "campaign_sent": True, "channel_id": "sms", "time_bucket": "morning",
    })
    assert action.channel_id == "SMS"
    assert EnvironmentAction.mask(terminal=False, before_expiry=True) == {
        "NO_ACTION": True, "SEND": True,
    }
    assert not EnvironmentAction.mask(terminal=True, before_expiry=True)["SEND"]
    with pytest.raises(ValueError, match="unknown action fields"):
        EnvironmentAction.parse({"campaign_sent": False, "campain": True})
    with pytest.raises(ValueError, match="boolean"):
        EnvironmentAction.parse({"campaign_sent": 1})


def test_daily_fallback_lists_each_active_day_and_respects_deadline():
    state = EnvironmentState({
        "done": False,
        "activation_datetime": "2026-05-01T09:30:00",
        "deadline": "2026-05-03T12:00:00",
    })
    assert get_decision_opportunities(state, "2026-05-01", "2026-05-05") == [
        pd.Timestamp("2026-05-01T09:30:00"),
        pd.Timestamp("2026-05-02T00:00:00"),
        pd.Timestamp("2026-05-03T00:00:00"),
    ]
    state.payload["done"] = True
    assert get_decision_opportunities(state, "2026-05-01", "2026-05-05") == []


def test_reconstruction_has_no_future_or_cross_application_leakage():
    history = pd.DataFrame([
        _row("a1", "customer", "2026-05-01", "Application Created"),
        _row("a1", "customer", "2026-05-02", "Professional Details Submission"),
        _row("a1", "customer", "2026-05-04", "AIP Approved"),
        _row("a2", "customer", "2026-05-01", "Application Created"),
        _row("a2", "customer", "2026-05-02", "AIP Approved"),
    ])
    state = reconstruct_state_from_gold(history, "a1", "2026-05-03")
    assert state.payload["professional_details_seen"]
    assert not state.payload["aip_approved_seen"]
    assert state.payload["application_id"] == "a1"


def test_reconstruction_rejects_post_terminal_rows():
    history = pd.DataFrame([
        _row("a", "c", "2026-05-01", "Application Created"),
        _row("a", "c", "2026-05-02", "AIP Approved"),
        _row("a", "c", "2026-05-03", "Push to Partner"),
        _row("a", "c", "2026-05-04", "Application Resume"),
    ])
    with pytest.raises(ValueError, match="after terminal PTP"):
        reconstruct_state_from_gold(history, "a", "2026-05-05")
