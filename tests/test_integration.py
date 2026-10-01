from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from environment import Environment, EnvironmentAction, LocalClosedLoop

BUNDLE = Path(__file__).parents[1] / "artifacts" / "gold_events_v1"


class FakeTransformer:
    def encode(self, cumulative_gold_history):
        assert cumulative_gold_history.event_datetime.max() <= pd.Timestamp("2026-05-01T09:00:00")
        return np.zeros(192, dtype=np.float32)


class FakePolicy:
    def choose_action(self, embedding):
        assert embedding.shape == (192,)
        return {"campaign_sent": False}


def _starting_history():
    return pd.DataFrame([{
        "application_id": "app", "context_id": "ctx",
        "event_datetime": pd.Timestamp("2026-05-01T08:00:00"),
        "journey_stage": "Application", "journey_substage": "Application Created",
    }])


def test_minimal_local_closed_loop_and_checkpoint_restart(tmp_path):
    source = tmp_path / "input.parquet"
    output = tmp_path / "run-a" / "history.parquet"
    checkpoint = tmp_path / "run-a" / "checkpoint.json"
    _starting_history().to_parquet(source, index=False)
    first = LocalClosedLoop(Environment(BUNDLE, run_id="run-a"), input_history=source,
                            cumulative_history=output, checkpoint=checkpoint)
    step = first.run_step("app", "2026-05-01T09:00:00", FakeTransformer(), FakePolicy())
    assert output.exists() and checkpoint.exists()
    assert step.action == EnvironmentAction.no_action()
    stored = first.checkpoints.load(checkpoint)
    assert stored["run_id"] == "run-a"
    assert stored["state"]["_last_decision_time"] == "2026-05-01T09:00:00"

    class NextTransformer:
        def encode(self, history):
            return np.zeros(192)

    resumed = LocalClosedLoop(Environment(BUNDLE, run_id="run-a"), input_history=source,
                              cumulative_history=output, checkpoint=checkpoint)
    resumed.run_step("app", "2026-05-02T00:00:00", NextTransformer(), FakePolicy())
    assert resumed.checkpoints.load(checkpoint)["state"]["decision_sequence"] == 2


def test_closed_loop_rejects_wrong_embedding_shape(tmp_path):
    source = tmp_path / "input.parquet"
    _starting_history().to_parquet(source, index=False)

    class BadTransformer:
        def encode(self, history):
            return np.zeros(191)

    loop = LocalClosedLoop(Environment(BUNDLE, run_id="bad"), input_history=source,
                           cumulative_history=tmp_path / "out.parquet",
                           checkpoint=tmp_path / "checkpoint.json")
    with pytest.raises(ValueError, match=r"shape \(192,\)"):
        loop.run_step("app", "2026-05-01T09:00:00", BadTransformer(), FakePolicy())
