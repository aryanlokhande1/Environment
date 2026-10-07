"""Frozen byte contracts: no runtime newline normalization or provenance rehashing."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import pytest

from environment.models import ArtifactLoader
from environment.runtime.progression import MAPPING_SHA256

ROOT = Path(__file__).parents[1]
# References are semantic mappings, never inferred by matching digest values.
LOCAL_REFERENCES = {'campaign_decision_policy_manifest.json': {'model_sha256': 'campaign_decision_policy.csv'},
 'campaign_response_motif_manifest.json': {'model_sha256': 'campaign_response_motif_model.csv',
                                           'donor_sha256': 'campaign_response_motif_donors.parquet'},
 'empirical_action_propensity_manifest.json': {'model_sha256': 'empirical_action_propensity.csv'},
 'empirical_event_intensity_manifest.json': {'model_sha256': 'empirical_event_intensity.csv',
                                             'gap_sha256': 'empirical_inter_event_gaps.parquet'},
 'event_name_mapping_manifest.json': {'model_sha256': 'event_name_mapping.csv'},
 'may_arrival_manifest.json': {'schedule_sha256': 'may_application_arrivals.parquet'},
 'model_manifest.json': {'natural_model_sha256': 'natural_transition_model.csv',
                         'starting_snapshot_sha256': 'starting_snapshot.parquet'},
 'natural_transition_regime_manifest.json': {'base_natural_sha256': 'natural_transition_model.csv',
                                             'model_sha256': 'natural_transition_regime_model.csv',
                                             'observed_substages_sha256': 'application_observed_substages.parquet'},
 'ptp_hazard_manifest.json': {'hazard_sha256': 'ptp_hazard.csv',
                              'event_delay_sha256': 'ptp_hazard_event_delays.parquet'},
 'stage_continuation_manifest.json': {'hazard_sha256': 'stage_continuation_hazard.csv',
                                      'delay_sha256': 'stage_continuation_delays.parquet'}}

# Historical fitting inputs and reports are provenance, not bundled runtime files.
# Their original digests must survive serialization migrations unchanged.
EVIDENCE_FIELDS = {
    "campaign_decision_policy_manifest.json": {
        "phase1_audit_sha256", "phase1_evidence_sha256", "source_campaign_sha256",
        "source_cohort_sha256",
    },
    "campaign_response_motif_manifest.json": {
        "source_campaign_sha256", "source_cohort_sha256", "source_files_sha256", "summary_sha256",
    },
    "empirical_action_propensity_manifest.json": {"source_campaign_sha256", "source_cohort_sha256"},
    "empirical_event_intensity_manifest.json": {
        "daily_sha256", "source_cohort_sha256", "source_file_sha256",
    },
    "event_name_mapping_manifest.json": {"source_files_sha256"},
    "may_arrival_manifest.json": {
        "donor_days_sha256", "historical_daily_sha256", "reference_sha256", "source_cohort_sha256",
    },
    "model_manifest.json": {
        "cohort_sha256", "old_mixed_model_sha256", "ptp_evidence_audit_sha256",
        "ptp_progression_mapping_sha256",
    },
    "natural_transition_regime_manifest.json": {
        "source_cohort_sha256", "source_files_sha256", "stability_sha256",
    },
    "ptp_hazard_manifest.json": {"fit_summary_sha256", "source_cohort_sha256", "source_files_sha256"},
    "stage_continuation_manifest.json": {"report_sha256", "source_cohort_sha256", "source_files_sha256"},
}


@pytest.mark.parametrize("version", ["gold_events_v1", "gold_events_v2"])
def test_frozen_artifact_byte_contract(version):
    bundle = ROOT / "artifacts" / version
    ArtifactLoader(bundle).validate()  # Strict raw hashes and schemas, including binaries.
    manifest = json.loads((bundle / "manifest.json").read_bytes())
    inventory = {row["filename"] for row in manifest["artifacts"]}
    assert {p.name for p in bundle.iterdir() if p.is_file()} == inventory | {"manifest.json", "hashes.json"}
    for path in bundle.iterdir():
        if path.suffix in {".csv", ".json"}:
            data = path.read_bytes()
            data.decode("utf-8", errors="strict")
            assert b"\r" not in data, path  # LF only; binaries are never normalized.
        if path.suffix != ".json" or path.name in {"manifest.json", "hashes.json"}:
            continue
        value = json.loads(path.read_bytes())
        local = LOCAL_REFERENCES[path.name]
        evidence = EVIDENCE_FIELDS[path.name]
        if path.name == "may_arrival_manifest.json" and version == "gold_events_v2":
            evidence = set()  # V2 arrivals embed the historical profile instead.
        assert {key for key in value if "sha256" in key} == local.keys() | evidence
        for key, filename in local.items():
            assert filename in inventory
            assert value[key] == sha256((bundle / filename).read_bytes()).hexdigest(), (path, key)
        if path.name == "model_manifest.json":
            # This is a canonical rule-object digest, not a file-byte digest.
            assert value["ptp_progression_mapping_sha256"] == MAPPING_SHA256
    hashes = json.loads((bundle / "hashes.json").read_bytes())
    assert set(hashes) == inventory | {"manifest.json"}
    for filename, expected in hashes.items():
        assert sha256((bundle / filename).read_bytes()).hexdigest() == expected, filename


@pytest.mark.parametrize("autocrlf", ["true", "false", "input"])
def test_git_checkout_attributes_are_explicit(autocrlf, tmp_path):
    # Exercise a fresh Git checkout with conflicting global-style newline settings.
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitattributes").write_bytes((ROOT / ".gitattributes").read_bytes())
    samples = {}
    for suffix in ("py", "md", "json", "yaml", "yml", "csv", "txt"):
        samples[f"sample.{suffix}"] = b"first\nsecond\n"
    for suffix in ("parquet", "pkl", "pickle", "npy", "npz"):
        samples[f"sample.{suffix}"] = b"\x00first\r\nsecond\n"
    for filename, data in samples.items():
        (tmp_path / filename).write_bytes(data)
    git = ["git", "-C", str(tmp_path), "-c", f"core.autocrlf={autocrlf}"]
    subprocess.run(git + ["add", "."], check=True, capture_output=True)
    for filename in samples:
        (tmp_path / filename).unlink()
    subprocess.run(git + ["checkout-index", "--all"], check=True, capture_output=True)
    for filename, data in samples.items():
        assert (tmp_path / filename).read_bytes() == data, (autocrlf, filename)
