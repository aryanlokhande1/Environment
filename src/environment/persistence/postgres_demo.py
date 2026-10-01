"""PostgreSQL loader for the cumulative Gold Events demonstration dataset.

The simulator remains independent of PostgreSQL.  This module is an explicit
delivery adapter which copies an already validated combined Parquet into the
``environment_demo`` analytics schema.
"""
from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import os
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from environment.runtime.gold_events_adapter import GOLD_COLUMNS


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCHEMA_PATH = ROOT / "sql" / "environment_demo.sql"
DEFAULT_COMBINED_PATH = (
    ROOT / "data" / "output" / "runs" / "may-no-action-20260502"
    / "combined_gold_events.parquet"
)
DEMO_COPY_COLUMNS = (
    "context_id", "application_id", "event_datetime", "event_name",
    "journey_stage", "journey_substage", "data_origin", "is_simulated",
    "site_subsection", "channel_id", "platform_type", "product_id",
    "partner_id", "trigger_type", "time_bucket", "visitnum", "utm_source",
    "utm_medium", "utm_campaign", "source_type", "campaign_name",
    "session_id", "channel", "utm_channel", "day_of_week_num", "initiated_by",
)


def _psycopg():
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "PostgreSQL support is not installed; run "
            "'python -m pip install -e .[postgres]'"
        ) from exc
    return psycopg


def connect():
    """Connect using DATABASE_URL or standard libpq PG* environment variables."""
    psycopg = _psycopg()
    try:
        return psycopg.connect(os.environ.get("DATABASE_URL", ""))
    except psycopg.OperationalError as exc:
        raise RuntimeError(
            "PostgreSQL connection failed; configure DATABASE_URL or the "
            "PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD variables: "
            f"{exc}"
        ) from exc


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_combined_parquet(path: str | Path) -> dict[str, Any]:
    """Validate the exact canonical Gold export before touching PostgreSQL."""
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"combined Gold Parquet not found: {source}")
    parquet = pq.ParquetFile(source)
    actual = tuple(parquet.schema_arrow.names)
    if actual != GOLD_COLUMNS:
        raise ValueError(
            "combined Parquet must contain the canonical Gold columns in order; "
            f"expected {list(GOLD_COLUMNS)}, got {list(actual)}"
        )
    event_type = parquet.schema_arrow.field("event_datetime").type
    if not pa.types.is_timestamp(event_type) or event_type.tz is not None:
        raise ValueError("event_datetime must be a timezone-naive Arrow timestamp")
    return {
        "path": str(source),
        "row_count": parquet.metadata.num_rows,
        "row_group_count": parquet.metadata.num_row_groups,
        "sha256": _file_sha256(source),
    }


def _iter_copy_rows(
    path: str | Path,
    simulation_start: datetime,
    *,
    batch_size: int = 16_384,
) -> Iterator[tuple[Any, ...]]:
    if simulation_start.tzinfo is not None:
        raise ValueError("simulation_start must be timezone-naive")
    parquet = pq.ParquetFile(Path(path))
    for batch in parquet.iter_batches(columns=list(GOLD_COLUMNS), batch_size=batch_size):
        for source in batch.to_pylist():
            event_time = source["event_datetime"]
            if event_time is None:
                raise ValueError("event_datetime cannot be NULL")
            if source["context_id"] is None:
                raise ValueError("context_id cannot be NULL")
            simulated = event_time >= simulation_start
            if simulated:
                data_origin = "SIMULATED_MAY"
            elif event_time >= datetime(2026, 4, 1):
                data_origin = "HISTORICAL_APRIL"
            else:
                data_origin = "HISTORICAL_MARCH"
            values = {
                **source,
                "data_origin": data_origin,
                "is_simulated": simulated,
            }
            yield tuple(values[column] for column in DEMO_COPY_COLUMNS)


def initialize_demo_schema(schema_path: str | Path = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    path = Path(schema_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"database schema SQL not found: {path}")
    with connect() as connection:
        connection.execute(path.read_text(encoding="utf-8"))
    return {"status": "PASS", "schema": "environment_demo", "sql": str(path)}


def load_combined_parquet(
    path: str | Path = DEFAULT_COMBINED_PATH,
    *,
    simulation_start: datetime = datetime(2026, 5, 1),
    replace: bool = False,
) -> dict[str, Any]:
    """Atomically load one combined export.

    Existing rows are protected by default.  ``replace=True`` truncates only
    ``environment_demo.combined_gold_events`` inside the same transaction as
    the copy, so a failed copy rolls the prior contents back.
    """
    source = Path(path).resolve()
    metadata = validate_combined_parquet(source)
    columns_sql = ", ".join(DEMO_COPY_COLUMNS)
    copy_sql = (
        "COPY environment_demo.combined_gold_events "
        f"({columns_sql}) FROM STDIN"
    )
    with connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "LOCK TABLE environment_demo.combined_gold_events IN ACCESS EXCLUSIVE MODE"
            )
            cursor.execute("SELECT count(*) FROM environment_demo.combined_gold_events")
            existing = int(cursor.fetchone()[0])
            if existing and not replace:
                raise RuntimeError(
                    "environment_demo.combined_gold_events is not empty "
                    f"({existing} rows); pass --replace to reload it atomically"
                )
            if replace:
                cursor.execute(
                    "TRUNCATE environment_demo.combined_gold_events RESTART IDENTITY"
                )
            loaded = 0
            with cursor.copy(copy_sql) as copy:
                for row in _iter_copy_rows(source, simulation_start):
                    copy.write_row(row)
                    loaded += 1
            if loaded != metadata["row_count"]:
                raise RuntimeError(
                    f"copy row count mismatch: expected {metadata['row_count']}, wrote {loaded}"
                )
            cursor.execute("ANALYZE environment_demo.combined_gold_events")
    return {
        "status": "PASS",
        "schema": "environment_demo",
        "table": "combined_gold_events",
        "simulation_start": simulation_start.isoformat(),
        **metadata,
    }


def validate_demo_database(
    *,
    expected_total: int | None = None,
    expected_historical: int | None = None,
    expected_simulated: int | None = None,
) -> dict[str, Any]:
    """Validate table invariants, requested views, and optional exact counts."""
    with connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT combined_event_count, historical_event_count,
                          simulated_event_count, combined_context_count,
                          simulated_context_count, combined_application_count,
                          simulated_application_count, first_event_datetime,
                          last_event_datetime, first_simulated_event_datetime,
                          last_simulated_event_datetime, simulated_active_day_count
                   FROM environment_demo.v_simulation_overview"""
            )
            row = cursor.fetchone()
            columns = [description.name for description in cursor.description]
            overview = dict(zip(columns, row))
            cursor.execute(
                """SELECT count(*) FROM environment_demo.combined_gold_events
                   WHERE context_id IS NULL OR event_datetime IS NULL
                      OR is_simulated <> (data_origin = 'SIMULATED_MAY')"""
            )
            invalid_rows = int(cursor.fetchone()[0])
            cursor.execute(
                """SELECT viewname FROM pg_catalog.pg_views
                   WHERE schemaname = 'environment_demo'
                     AND viewname = ANY(%s)
                   ORDER BY viewname""",
                ([
                    "v_monthly_summary", "v_may_daily_summary",
                    "v_simulation_overview", "v_may_journey_summary",
                    "v_context_journey",
                ],),
            )
            views = [value[0] for value in cursor.fetchall()]

    expected_views = {
        "v_monthly_summary", "v_may_daily_summary", "v_simulation_overview",
        "v_may_journey_summary", "v_context_journey",
    }
    violations: list[str] = []
    if invalid_rows:
        violations.append(f"invalid_rows={invalid_rows}")
    missing_views = sorted(expected_views - set(views))
    if missing_views:
        violations.append(f"missing_views={missing_views}")
    for label, expected, actual_key in (
        ("total", expected_total, "combined_event_count"),
        ("historical", expected_historical, "historical_event_count"),
        ("simulated", expected_simulated, "simulated_event_count"),
    ):
        if expected is not None and int(overview[actual_key]) != expected:
            violations.append(
                f"{label}_rows expected {expected}, got {overview[actual_key]}"
            )
    return {
        "status": "PASS" if not violations else "FAIL",
        "schema": "environment_demo",
        "overview": overview,
        "views": views,
        "violations": violations,
    }
