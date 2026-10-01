"""Optional S3 object adapter; credentials come from the standard AWS chain."""
from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlparse
import json
import pandas as pd

from environment.models.artifact_loader import ArtifactLoader

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

    def load_checkpoint(self, uri: str) -> bytes:
        bucket, key = self._parts(uri)
        response = self.client.get_object(Bucket=bucket, Key=key)
        return response["Body"].read()

    def download_artifact_bundle(self, manifest_uri: str, target: str | Path) -> Path:
        """Download and hash-validate every file declared by an S3 manifest."""
        bucket, key = self._parts(manifest_uri)
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        manifest_path = target / "manifest.json"
        self.client.download_file(bucket, key, str(manifest_path))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        prefix = key.rsplit("/", 1)[0] if "/" in key else ""
        for row in manifest.get("artifacts", []):
            filename = str(row.get("filename", ""))
            relative = Path(filename)
            if not filename or relative.is_absolute() or ".." in relative.parts or relative.name != filename:
                raise ValueError(f"unsafe artifact filename in manifest: {filename!r}")
            artifact_key = f"{prefix}/{filename}" if prefix else filename
            self.client.download_file(bucket, artifact_key, str(target / filename))
        ArtifactLoader(target).validate()
        return target
