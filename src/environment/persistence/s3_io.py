"""Optional S3 object adapter; credentials come from the standard AWS chain."""
from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlparse
import pandas as pd

class S3IO:
    def __init__(self, client=None):
        if client is None:
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError("install the 's3' extra to use S3IO") from exc
            client = boto3.client("s3")
        self.client = client

    @staticmethod
    def _parts(uri: str) -> tuple[str, str]:
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
            raise ValueError("expected s3://bucket/key URI")
        return parsed.netloc, parsed.path.lstrip("/")

    def load_starting_gold_history(self, uri: str) -> pd.DataFrame:
        bucket, key = self._parts(uri)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "history.parquet"
            self.client.download_file(bucket, key, str(path))
            return pd.read_parquet(path)

    def write_run_output(self, rows: pd.DataFrame, uri: str) -> None:
        bucket, key = self._parts(uri)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "output.parquet"
            rows.to_parquet(path, index=False)
            self.client.upload_file(str(path), bucket, key)

    write_final_cumulative_output = write_run_output

    def write_checkpoint(self, payload: bytes, uri: str) -> None:
        bucket, key = self._parts(uri)
        self.client.put_object(Bucket=bucket, Key=key, Body=payload)

    def download_artifact_bundle(self, manifest_uri: str, target: str | Path) -> Path:
        """Download a caller-enumerated manifest; artifact keys remain manifest-driven."""
        bucket, key = self._parts(manifest_uri)
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        self.client.download_file(bucket, key, str(target / "manifest.json"))
        return target
