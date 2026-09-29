"""Optional PostgreSQL transactional state store."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any

def connect():
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("install the 'postgres' extra to use PostgreSQL") from exc
    return psycopg.connect(host=os.getenv("PGHOST"), port=os.getenv("PGPORT", "5432"),
                           dbname=os.getenv("PGDATABASE"), user=os.getenv("PGUSER"),
                           password=os.getenv("PGPASSWORD"))

def initialize_schema(schema_path: str | Path = "sql/environment_schema.sql") -> None:
    sql = Path(schema_path).read_text(encoding="utf-8")
    with connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql)

class PostgresStateStore:
    def commit_checkpoint(self, run_id: str, application_id: str, checkpoint_time: Any,
                          state_json: str, pending_json: str) -> None:
        statement = """INSERT INTO environment_checkpoints
            (run_id, application_id, checkpoint_time, state_json, pending_json)
            VALUES (%s,%s,%s,%s::jsonb,%s::jsonb)
            ON CONFLICT (run_id, application_id) DO UPDATE SET
            checkpoint_time=EXCLUDED.checkpoint_time, state_json=EXCLUDED.state_json,
            pending_json=EXCLUDED.pending_json"""
        with connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(statement, (run_id, application_id, checkpoint_time, state_json, pending_json))
