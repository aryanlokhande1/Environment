"""Durable atomic files and manifest-last local simulation commits."""
from __future__ import annotations
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from hashlib import sha256


def file_hash(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ensure_directory(path: Path) -> None:
    missing = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        descriptor = os.open(directory.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def durable_replace(temporary: Path, path: Path) -> None:
    """Flush file content, rename atomically, then flush the directory entry."""
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: Path, data: bytes) -> None:
    ensure_directory(path.parent)
    descriptor, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        durable_replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class CheckpointStore:
    def save(self, value: dict[str, Any], uri: str | Path) -> None:
        atomic_bytes(Path(uri), json.dumps(value, sort_keys=True, default=str).encode('utf-8'))

    def load(self, uri: str | Path) -> dict[str, Any]:
        return json.loads(Path(uri).read_text(encoding='utf-8'))

    def recover_history(self, checkpoint: Path, history: Path) -> dict[str, Any]:
        """The checkpoint is the final commit marker, never a staged state file.

        Immutable history/state snapshots are authoritative. A public history
        projection may be the immediately preceding committed version following
        a crash, but arbitrary corruption is rejected rather than overwritten.
        """
        stored = self.load(checkpoint)
        commit = stored.get('local_commit')
        if commit is None:
            raise ValueError('checkpoint has no committed history manifest')
        history_snapshot = checkpoint.parent / commit['history']
        state_snapshot = checkpoint.parent / commit['checkpoint']
        if (file_hash(history_snapshot) != commit['history_sha256']
                or file_hash(state_snapshot) != commit['checkpoint_sha256']):
            raise ValueError('committed history/checkpoint hash mismatch')
        payload = {key: value for key, value in stored.items() if key != 'local_commit'}
        if self.load(state_snapshot) != payload:
            raise ValueError('checkpoint state disagrees with commit snapshot')
        if history.exists():
            digest = file_hash(history)
            if digest not in {commit['history_sha256'], commit['previous_history_sha256']}:
                raise ValueError('cumulative history disagrees with checkpoint')
        elif commit['previous_history_sha256'] is not None:
            raise ValueError('committed cumulative history is missing')
        if not history.exists() or file_hash(history) != commit['history_sha256']:
            atomic_bytes(history, history_snapshot.read_bytes())
        return stored

    write_checkpoint = save
