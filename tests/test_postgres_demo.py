from datetime import datetime
from pathlib import Path

import pandas as pd

from environment.persistence.postgres_demo import (
    DEMO_COPY_COLUMNS,
    _iter_copy_rows,
    validate_combined_parquet,
)
from environment.runtime.gold_events_adapter import GOLD_COLUMNS


def _combined_fixture(path: Path) -> None:
    rows = []
    for context_id, event_time in (
        ("historical-customer", datetime(2026, 4, 30, 23, 59, 59)),
        ("simulated-customer", datetime(2026, 5, 1, 0, 0, 0)),
    ):
        row = {column: None for column in GOLD_COLUMNS}
        row.update({
            "application_id": f"app-{context_id}",
            "context_id": context_id,
            "event_datetime": event_time,
            "event_name": "event",
            "journey_stage": "stage",
            "journey_substage": "substage",
            "visitnum": 1.0,
            "day_of_week_num": 1,
        })
        rows.append(row)
    pd.DataFrame(rows).reindex(columns=list(GOLD_COLUMNS)).to_parquet(path, index=False)


def test_demo_input_and_origin_mapping(tmp_path: Path):
    path = tmp_path / "combined.parquet"
    _combined_fixture(path)

    metadata = validate_combined_parquet(path)
    rows = list(_iter_copy_rows(path, datetime(2026, 5, 1)))
    origin = DEMO_COPY_COLUMNS.index("data_origin")
    simulated = DEMO_COPY_COLUMNS.index("is_simulated")

    assert metadata["row_count"] == 2
    assert len(metadata["sha256"]) == 64
    assert rows[0][origin] == "HISTORICAL_APRIL"
    assert rows[0][simulated] is False
    assert rows[1][origin] == "SIMULATED_MAY"
    assert rows[1][simulated] is True


def test_environment_demo_sql_defines_requested_surface():
    sql = (Path(__file__).parents[1] / "sql" / "environment_demo.sql").read_text(
        encoding="utf-8"
    ).lower()

    assert "create schema if not exists environment_demo" in sql
    assert "event_row_id bigint generated always as identity primary key" in sql
    assert "context_id text not null" in sql
    for view in (
        "v_monthly_summary",
        "v_may_daily_summary",
        "v_simulation_overview",
        "v_may_journey_summary",
        "v_context_journey",
    ):
        assert f"create or replace view environment_demo.{view}" in sql
