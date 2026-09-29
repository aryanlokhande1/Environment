"""Build the frozen bundle inventory without fitting or changing model data."""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts" / "gold_events_v1"

GROUPS = {
    "natural": ("journey state occurrence", ["journey_substage", "FIRST_PASS/REVISIT"], ["STATE_REGIME", "STATE", "GLOBAL"]),
    "continuation": ("state occurrence at risk", ["journey_substage", "elapsed interval"], ["STATE", "STAGE", "GLOBAL"]),
    "ptp": ("eligible AIP application at risk", ["elapsed since AIP", "application age", "no PTP yet"], ["AIP_AGE", "GLOBAL_ELAPSED"]),
    "campaign_decision": ("active application-day", ["journey_substage", "stage", "age", "recent sends"], ["STATE", "STAGE", "GLOBAL"]),
    "campaign_motif": ("uniquely attributed campaign send", ["pre-send state", "channel", "time_bucket", "campaign history"], ["STATE_CHANNEL_BUCKET", "STATE_CHANNEL", "STAGE_CHANNEL", "GLOBAL_CHANNEL", "GLOBAL"]),
    "action": ("observed campaign action", ["journey_substage", "journey_stage"], ["STATE", "STAGE", "GLOBAL"]),
    "intensity": ("active application-day", ["journey_substage", "events already today"], ["STATE_COUNT", "STATE", "GLOBAL"]),
    "arrival": ("historical observed application creation", ["weekday", "month phase"], ["MATCHED_DONOR_DAY"]),
    "cohort": ("application lifecycle", ["application_id"], ["EXACT_APPLICATION"]),
    "mapping": ("observed journey row", ["journey_substage"], ["MOST_SUPPORTED_STAGE"]),
}

def category(name: str) -> str:
    if name.startswith("natural_") or name == "model_manifest.json": return "natural"
    if name.startswith("stage_continuation"): return "continuation"
    if name.startswith("ptp_hazard"): return "ptp"
    if name.startswith("campaign_decision"): return "campaign_decision"
    if name.startswith("campaign_response_motif"): return "campaign_motif"
    if name.startswith("empirical_action"): return "action"
    if name.startswith("empirical_event") or name.startswith("empirical_inter"): return "intensity"
    if name.startswith("may_"): return "arrival"
    if name.startswith("lifecycle_business"): return "mapping"
    return "cohort"

def metadata_for(name: str) -> dict:
    key = category(name)
    support_unit, conditioning, backoff = GROUPS[key]
    manifests = {
        "natural": "model_manifest.json", "continuation": "stage_continuation_manifest.json",
        "ptp": "ptp_hazard_manifest.json", "campaign_decision": "campaign_decision_policy_manifest.json",
        "campaign_motif": "campaign_response_motif_manifest.json", "action": "empirical_action_propensity_manifest.json",
        "intensity": "empirical_event_intensity_manifest.json", "arrival": "may_arrival_manifest.json",
        "cohort": "model_manifest.json", "mapping": "model_manifest.json",
    }
    source = {}
    manifest_name = manifests.get(key)
    if manifest_name and (BUNDLE / manifest_name).exists():
        source = json.loads((BUNDLE / manifest_name).read_text(encoding="utf-8"))
    support = next((source.get(field) for field in (
        "support_count", "active_application_days", "eligible_aip_applications",
        "mature_application_active_days", "mature_applications", "total_uniquely_owned_sends",
        "eligible_application_days", "send_count", "transition_rows", "source_application_rows",
        "observed_creation_applications", "starting_applications") if source.get(field) is not None), None)
    if support is None and isinstance(source.get("fit_counts"), dict):
        support = source["fit_counts"].get("retained_motifs")
    return {
        "model_artifact_version": source.get("model_version", source.get("contract_version", "V3_FROZEN")),
        "source_period": source.get("source_period", "2026-03-01 through 2026-04-30; immutable historical only"),
        "population": "Personal Loan", "support_count": support, "support_unit": support_unit,
        "conditioning_keys": conditioning, "backoff_hierarchy": backoff,
        "authoritative_runtime_input": True,
    }

def main() -> None:
    files = sorted(path for path in BUNDLE.iterdir() if path.is_file() and path.name not in {"manifest.json", "hashes.json"})
    artifacts = []
    for path in files:
        logical = path.stem.replace("_manifest", "_provenance").replace("_model", "")
        artifacts.append({"logical_name": logical, "filename": path.name,
                          "sha256": sha256(path.read_bytes()).hexdigest(), **metadata_for(path.name)})
    manifest = {
        "bundle_version": "gold_events_v1", "contract_version": "2.0",
        "population": "Personal Loan", "source_period": "2026-03-01 through 2026-04-30",
        "fitting_policy": "immutable historical only; simulated and synthetic rows forbidden",
        "artifact_loading": "manifest-only", "artifacts": artifacts,
    }
    manifest_path = BUNDLE / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    hashes = {row["filename"]: row["sha256"] for row in artifacts}
    hashes["manifest.json"] = sha256(manifest_path.read_bytes()).hexdigest()
    (BUNDLE / "hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
