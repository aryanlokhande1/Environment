from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
BUNDLE = ROOT / "artifacts" / "gold_events_v1"

def test_manifest_hashes_and_forbidden_runtime_dependencies():
    manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    for row in manifest["artifacts"]:
        assert sha256((BUNDLE / row["filename"]).read_bytes()).hexdigest() == row["sha256"]
    forbidden = ("goldfish.", "src.goldfish", "C:\\Aryan\\", "stage_mapping.csv",
                 "environment_model_v2.csv", "from .ptp_outcome", "ptp_outcome_manifest.json")
    for base in (ROOT / "src", ROOT / "config"):
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".json"}:
                text = path.read_text(encoding="utf-8")
                assert not any(token in text for token in forbidden), (path, forbidden)

def test_fixture_contract_covers_required_scenarios():
    fixtures = json.loads((ROOT / "tests" / "fixtures" / "scenarios.json").read_text(encoding="utf-8"))
    required = {"no_action_organic_continuation", "temporary_inactivity", "campaign_send",
                "campaign_open_read", "campaign_click", "campaign_response", "application_created",
                "otp", "resume_first_pass", "resume_revisit", "professional_details", "aip",
                "product_bank_path", "ptp", "expiry", "carried_forward_application",
                "revisit_loop", "sparse_backoff"}
    assert {row["name"] for row in fixtures} == required
    for row in fixtures:
        assert {"history_state", "action", "decision_time", "seed", "expected"} <= row.keys()
        assert {"artifact", "backoff", "result_category", "terminal", "explanation_fields"} <= row["expected"].keys()
