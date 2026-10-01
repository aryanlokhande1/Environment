from pathlib import Path
import shutil

import pandas as pd
import pytest
from environment.persistence import CheckpointStore, LocalIO, S3IO
from environment.models import ArtifactLoader
from environment.runtime.gold_events_adapter import GOLD_COLUMNS

BUNDLE = Path(__file__).parents[1] / "artifacts" / "gold_events_v1"

def test_local_io_and_atomic_checkpoint(tmp_path):
    frame = pd.DataFrame({
        "application_id": ["a"], "context_id": ["c"],
        "event_datetime": [pd.Timestamp("2026-05-01")],
        "journey_stage": ["Application"], "journey_substage": ["Application Created"],
    })
    target = tmp_path / "history.parquet"
    LocalIO().write_history(frame, target)
    assert LocalIO().load_history(target).equals(frame)
    checkpoint = tmp_path / "checkpoint.json"
    CheckpointStore().save({"state": {"application_id": "a"}, "pending": []}, checkpoint)
    assert CheckpointStore().load(checkpoint)["state"]["application_id"] == "a"


def test_atomic_gold_append_rejects_duplicates_and_out_of_order(tmp_path):
    target = tmp_path / "history.parquet"
    initial = pd.DataFrame({
        "application_id": ["a"], "context_id": ["c"],
        "event_datetime": [pd.Timestamp("2026-05-01")],
        "journey_stage": ["Application"], "journey_substage": ["Application Created"],
    })
    io = LocalIO()
    io.write_history(initial, target)
    row = {column: None for column in GOLD_COLUMNS}
    row.update({
        "application_id": "a", "context_id": "c",
        "event_datetime": pd.Timestamp("2026-05-02"),
        "journey_stage": "Application", "journey_substage": "Application Resume",
        "cumulative_row_key": "S|one", "is_simulated": True,
    })
    combined = io.append_history([row], target)
    assert len(combined) == 2
    with pytest.raises(ValueError, match="existing cumulative_row_key"):
        io.append_history([row], target)
    older = {**row, "cumulative_row_key": "S|two",
             "event_datetime": pd.Timestamp("2026-04-30")}
    with pytest.raises(ValueError, match="out-of-order"):
        io.append_history([older], target)


def test_s3_bundle_download_is_manifest_led_and_hash_verified(tmp_path):
    class FakeS3:
        def download_file(self, bucket, key, destination):
            assert bucket == "bucket"
            source = BUNDLE / key.rsplit("/", 1)[-1]
            shutil.copyfile(source, destination)

    target = S3IO(FakeS3()).download_artifact_bundle(
        "s3://bucket/frozen/manifest.json", tmp_path / "bundle")
    assert len(ArtifactLoader(target).validate()) == 26
