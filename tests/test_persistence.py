import pandas as pd
from environment.persistence import CheckpointStore, LocalIO

def test_local_io_and_atomic_checkpoint(tmp_path):
    frame = pd.DataFrame({"application_id": ["a"], "value": [1]})
    target = tmp_path / "history.parquet"
    LocalIO().write_history(frame, target)
    assert LocalIO().load_history(target).equals(frame)
    checkpoint = tmp_path / "checkpoint.json"
    CheckpointStore().save({"state": {"application_id": "a"}, "pending": []}, checkpoint)
    assert CheckpointStore().load(checkpoint)["state"]["application_id"] == "a"
