"""Strict manifest-only artifact bundle loader."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class ArtifactRecord:
    logical_name: str
    filename: str
    sha256: str
    metadata: dict[str, Any]

class ArtifactLoader:
    def __init__(self, bundle: str | Path):
        self.bundle = Path(bundle)
        path = self.bundle / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"artifact manifest not found: {path}")
        self.manifest = json.loads(path.read_text(encoding="utf-8"))
        if self.manifest.get("bundle_version") != "gold_events_v1":
            raise ValueError("unsupported artifact bundle version")
        self.records = {row["logical_name"]: ArtifactRecord(
            row["logical_name"], row["filename"], row["sha256"], row
        ) for row in self.manifest["artifacts"]}
        if len(self.records) != len(self.manifest["artifacts"]):
            raise ValueError("duplicate logical artifact names")

    def validate(self) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for record in self.records.values():
            path = self.bundle / record.filename
            if not path.is_file():
                raise FileNotFoundError(f"declared artifact missing: {record.filename}")
            digest = sha256(path.read_bytes()).hexdigest()
            if digest != record.sha256:
                raise ValueError(f"artifact hash mismatch: {record.filename}")
            if not record.metadata.get("authoritative_runtime_input"):
                raise ValueError(f"non-authoritative file declared: {record.filename}")
            hashes[record.logical_name] = digest
        return hashes

    def path(self, logical_name: str) -> Path:
        record = self.records[logical_name]
        path = self.bundle / record.filename
        if sha256(path.read_bytes()).hexdigest() != record.sha256:
            raise ValueError(f"artifact hash mismatch: {record.filename}")
        return path
