from pathlib import Path

import pandas as pd

from environment import Environment, EnvironmentAction
from environment.core.state import EnvironmentState
from environment.models.artifact_loader import ArtifactLoader
from environment.runtime.gold_events_adapter import EventNameMapper, to_gold_row


ROOT = Path(__file__).parents[1]
BUNDLE = ROOT / "artifacts" / "gold_events_v2"


def test_v2_arrivals_cover_may_without_changing_population():
    ArtifactLoader(BUNDLE).validate()
    arrivals = pd.read_parquet(BUNDLE / "may_application_arrivals.parquet")
    daily = arrivals.groupby(pd.to_datetime(arrivals.activation_datetime).dt.normalize()).size()

    assert len(arrivals) == 48_373
    assert len(daily) == 31
    assert (daily > 0).all()
    assert daily.iloc[:11].sum() / len(arrivals) <= 0.60
    assert daily.iloc[23:].sum() / len(arrivals) >= 0.12


def test_v2_event_name_mapping_is_deterministic_and_nonempty():
    mapper = EventNameMapper(BUNDLE / "event_name_mapping.csv", seed=20260502)
    event = {
        "run_id": "test", "event_key": "event-1", "event_type": "JOURNEY_EVENT",
        "application_id": "app", "context_id": "context",
        "event_datetime": pd.Timestamp("2026-05-02T10:00:00"),
        "journey_stage": "Onboarding", "journey_substage": "Application Created",
        "source_type": "journey", "time_bucket": None, "channel": None,
        "trigger_type": None,
    }
    first = to_gold_row(event, mapper)
    second = to_gold_row(event, mapper)

    assert first is not None
    assert first["event_name"]
    assert first["event_name"] == second["event_name"]


def test_v2_arrival_emits_one_gold_creation_row():
    environment = Environment(BUNDLE, run_id="v2-arrival-test")
    raw = environment._runtime.arrivals.iloc[0].to_dict()
    when = pd.Timestamp(raw["activation_datetime"])
    state = EnvironmentState(environment._runtime.new_state(raw, activation=when))

    result = environment.advance(
        state, EnvironmentAction.no_action(), when, when.normalize() + pd.Timedelta(days=1))
    created = [row for row in result.events
               if row["journey_substage"] == "Application Created"]

    assert len(created) == 1
    assert created[0]["event_name"]
