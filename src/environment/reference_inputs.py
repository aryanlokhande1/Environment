"""Validation for non-runtime historical reference inputs."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def validate_reference_inputs(spec_path: str | Path = "config/reference_inputs.json") -> dict[str, Any]:
    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    root = spec_path.parents[1] / spec["base_directory"]
    results: list[dict[str, Any]] = []
    manifest_record = next(row for row in spec["files"] if row["name"] == "reference_manifest")
    manifest_path = root / manifest_record["filename"]
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing reference input: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing_manifest = set(manifest_record["required_json_fields"]) - set(manifest)
    if missing_manifest:
        raise ValueError(f"reference manifest missing fields: {sorted(missing_manifest)}")
    inventory = {row["filename"]: row for row in manifest["files"]}
    for record in spec["files"]:
        path = root / record["filename"]
        if not path.is_file():
            raise FileNotFoundError(f"missing reference input: {path}")
        if record["name"] == "reference_manifest":
            continue
        declared = inventory.get(record["filename"])
        if declared is None or "sha256" not in declared or "row_count" not in declared:
            raise ValueError(f"reference manifest lacks hash/row_count for {record['filename']}")
        digest = sha256(path.read_bytes()).hexdigest()
        if digest != declared["sha256"]:
            raise ValueError(f"reference hash mismatch: {record['filename']}")
        parquet = pq.ParquetFile(path)
        missing = set(record["required_columns"]) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{record['filename']} missing columns: {sorted(missing)}")
        if parquet.metadata.num_rows != int(declared["row_count"]):
            raise ValueError(f"reference row count mismatch: {record['filename']}")
        results.append({"filename": record["filename"], "sha256": digest,
                        "row_count": parquet.metadata.num_rows})
    return {"status": "ok", "schema_version": spec["schema_version"], "files": results}
