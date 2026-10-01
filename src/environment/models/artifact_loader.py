"""Strict manifest-only artifact bundle loader."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import csv
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


REQUIRED_ARTIFACT_COLUMNS = {
    "application_lifecycle_state": {"application_id", "context_id", "application_created_at"},
    "application_observed_substages": {"application_id", "observed_journey_substages"},
    "campaign_decision_policy": {"model_level", "support_count", "send_probability"},
    "campaign_response_motif_donors": {"engagement_sequence", "next_journey_substage",
                                        "observed_horizon_seconds"},
    "campaign_response_motif": {"model_level", "engagement_sequence", "next_journey_substage",
                                 "denominator", "probability"},
    "empirical_action_propensity": {"model_level", "channel", "time_bucket", "probability"},
    "empirical_event_intensity": {"model_level", "journey_substage", "count_bucket",
                                   "p_no_later_observed_event"},
    "empirical_inter_event_gaps": {"journey_substage", "count_bucket", "gap_microseconds"},
    "event_name_mapping": {"model_level", "journey_stage", "journey_substage",
                           "source_type", "event_name", "support_count", "probability"},
    "lifecycle_business_mapping": {"journey_substage", "journey_stage"},
    "may_application_arrivals": {"application_id", "activation_datetime", "snapshot_context_id",
                                  "snapshot_stage", "snapshot_substage"},
    "natural_transition": {"state", "response", "support_count", "probability"},
    "natural_transition_regime": {"state", "regime", "response", "support_count", "probability"},
    "ptp_hazard": {"model_level", "elapsed_lower_seconds", "elapsed_upper_seconds",
                   "risk_set", "hazard"},
    "ptp_hazard_event_delays": {"application_age_bucket", "duration_seconds"},
    "stage_continuation_delays": {"journey_stage", "journey_substage", "duration_seconds"},
    "stage_continuation_hazard": {"model_level", "elapsed_lower_seconds",
                                  "elapsed_upper_seconds", "risk_set", "hazard"},
    "starting_snapshot": {"application_id", "snapshot_context_id", "snapshot_event_datetime",
                          "snapshot_stage", "snapshot_substage", "product_id"},
}

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
        if self.manifest.get("bundle_version") not in {"gold_events_v1", "gold_events_v2"}:
            raise ValueError("unsupported artifact bundle version")
        if self.manifest.get("contract_version") != "2.0":
            raise ValueError("unsupported artifact contract version")
        if self.manifest.get("population") != "Personal Loan":
            raise ValueError("artifact bundle population must be Personal Loan")
        artifacts = self.manifest.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError("artifact manifest must contain a non-empty artifacts list")
        for row in artifacts:
            required = {"logical_name", "filename", "sha256", "model_artifact_version",
                        "source_period", "population", "support_unit", "conditioning_keys",
                        "backoff_hierarchy", "authoritative_runtime_input"}
            missing = required - set(row)
            if missing:
                raise ValueError(f"artifact record missing fields: {sorted(missing)}")
            filename = str(row["filename"])
            relative = Path(filename)
            if relative.is_absolute() or ".." in relative.parts or relative.name != filename:
                raise ValueError(f"unsafe artifact filename: {filename!r}")
            if row["population"] != "Personal Loan":
                raise ValueError(f"non-PL artifact declared: {filename}")
        self.records = {row["logical_name"]: ArtifactRecord(
            row["logical_name"], row["filename"], row["sha256"], row
        ) for row in artifacts}
        if len(self.records) != len(artifacts):
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
            self._validate_schema(record, path)
            hashes[record.logical_name] = digest
        return hashes

    @staticmethod
    def _validate_schema(record: ArtifactRecord, path: Path) -> None:
        if path.suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"JSON artifact must contain an object: {record.filename}")
            return
        required = REQUIRED_ARTIFACT_COLUMNS.get(record.logical_name)
        if required is None:
            return
        if path.suffix == ".parquet":
            columns = set(pq.ParquetFile(path).schema_arrow.names)
        elif path.suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as stream:
                columns = set(next(csv.reader(stream), []))
        else:
            raise ValueError(f"unsupported artifact file type: {record.filename}")
        missing = required - columns
        if missing:
            raise ValueError(f"artifact schema mismatch for {record.filename}: {sorted(missing)}")

    def path(self, logical_name: str) -> Path:
        record = self.records[logical_name]
        path = self.bundle / record.filename
        if sha256(path.read_bytes()).hexdigest() != record.sha256:
            raise ValueError(f"artifact hash mismatch: {record.filename}")
        return path
