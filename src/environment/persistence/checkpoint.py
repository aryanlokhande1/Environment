"""Atomic JSON checkpoint/state storage."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

class CheckpointStore:
    def save(self, value: dict[str, Any], uri: str | Path) -> None:
        path = Path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, sort_keys=True, default=str), encoding="utf-8")
        temporary.replace(path)

    def load(self, uri: str | Path) -> dict[str, Any]:
        return json.loads(Path(uri).read_text(encoding="utf-8"))

    write_checkpoint = save
