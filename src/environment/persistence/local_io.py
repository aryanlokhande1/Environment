"""Local filesystem history and output adapter."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

class LocalIO:
    def load_history(self, uri: str | Path) -> pd.DataFrame:
        return pd.read_parquet(Path(uri))

    def write_history(self, rows: pd.DataFrame, uri: str | Path) -> None:
        path = Path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        rows.to_parquet(temporary, index=False)
        temporary.replace(path)

    load_starting_gold_history = load_history
    write_run_output = write_history
    write_final_cumulative_output = write_history
