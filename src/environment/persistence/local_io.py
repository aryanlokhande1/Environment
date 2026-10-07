"""Local filesystem history and output adapter."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Mapping, Any
import pandas as pd

from environment.runtime.gold_events_adapter import GOLD_COLUMNS
from .checkpoint import durable_replace, ensure_directory

REQUIRED_HISTORY_COLUMNS = frozenset({
    "application_id", "context_id", "event_datetime", "journey_stage", "journey_substage",
})

class LocalIO:
    def load_history(self, uri: str | Path) -> pd.DataFrame:
        rows = pd.read_parquet(Path(uri))
        self.validate_history(rows)
        return rows

    @staticmethod
    def validate_history(rows: pd.DataFrame) -> None:
        missing = REQUIRED_HISTORY_COLUMNS - set(rows.columns)
        if missing:
            raise ValueError(f"Gold history missing columns: {sorted(missing)}")
        timestamps = pd.to_datetime(rows["event_datetime"], errors="coerce")
        if timestamps.isna().any():
            raise ValueError("Gold history contains invalid event_datetime values")
        if getattr(timestamps.dt, "tz", None) is not None:
            raise ValueError("Gold history timestamps must be timezone-naive")
        owners = (rows.dropna(subset=["application_id", "context_id"])
                  .astype({"application_id": str, "context_id": str})
                  .groupby("application_id", observed=True)["context_id"].nunique())
        if owners.gt(1).any():
            raise ValueError("an application_id cannot belong to multiple contexts")
        if "cumulative_row_key" in rows and rows.cumulative_row_key.dropna().duplicated().any():
            raise ValueError("duplicate cumulative_row_key in Gold history")

    def write_history(self, rows: pd.DataFrame, uri: str | Path) -> None:
        self.validate_history(rows)
        path = Path(uri)
        ensure_directory(path.parent)
        temporary = path.with_suffix(path.suffix + ".tmp")
        rows.to_parquet(temporary, index=False)
        durable_replace(temporary, path)

    def append_history(self, events: Iterable[Mapping[str, Any]], uri: str | Path) -> pd.DataFrame:
        """Atomically append unique, chronological Gold rows to a local history."""
        path = Path(uri)
        existing = self.load_history(path)
        incoming = pd.DataFrame(list(events))
        if incoming.empty:
            return existing
        missing_gold = set(GOLD_COLUMNS) - set(incoming.columns)
        if missing_gold:
            raise ValueError(f"emitted rows do not satisfy Gold schema: {sorted(missing_gold)}")
        self.validate_history(incoming)
        if "cumulative_row_key" not in incoming or incoming.cumulative_row_key.isna().any():
            raise ValueError("simulated Gold rows require cumulative_row_key")
        if incoming.cumulative_row_key.duplicated().any():
            raise ValueError("duplicate emitted cumulative_row_key")
        if "cumulative_row_key" in existing:
            overlap = set(existing.cumulative_row_key.dropna()) & set(incoming.cumulative_row_key)
            if overlap:
                raise ValueError("attempted to append an existing cumulative_row_key")
        existing_times = pd.to_datetime(existing.event_datetime)
        incoming_times = pd.to_datetime(incoming.event_datetime)
        for application_id, group in incoming.assign(_time=incoming_times).groupby("application_id"):
            prior = existing_times.loc[existing.application_id.astype(str).eq(str(application_id))]
            if not prior.empty and group._time.min() < prior.max():
                raise ValueError(f"out-of-order append for application_id {application_id}")
        columns = list(dict.fromkeys([*existing.columns, *incoming.columns]))
        combined = pd.concat([existing.reindex(columns=columns), incoming.reindex(columns=columns)],
                             ignore_index=True)
        combined = combined.sort_values(["event_datetime", "application_id"], kind="stable").reset_index(drop=True)
        self.write_history(combined, path)
        return combined

    load_starting_gold_history = load_history
    write_run_output = write_history
    write_final_cumulative_output = write_history
