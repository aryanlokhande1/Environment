"""Run-scoped, restartable local May simulation over immutable Gold history."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import gzip
import json
import logging
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from environment.contracts.action import EnvironmentAction
from environment.runtime.semantics import CORRECTED, LEGACY, FINAL, validate_runtime_version
from environment.core.environment import Environment
from environment.core.state import EnvironmentState, active_application_for_context
from environment.runtime.gold_events_adapter import GOLD_COLUMNS
from environment.persistence.checkpoint import atomic_bytes, durable_replace, ensure_directory

START = pd.Timestamp("2026-05-01")
END = pd.Timestamp("2026-06-01")
CHECKPOINT_VERSION = "MAY_LOCAL_RUN_V1"
SIMULATION_COLUMNS = (
    "cumulative_row_key", "run_id", "is_simulated", "simulation_event_key",
    "source_file", "source_row", *GOLD_COLUMNS,
)
GOLD_ARROW_SCHEMA = pa.schema([
    pa.field(name, pa.timestamp("ns") if name == "event_datetime" else
             pa.float64() if name == "visitnum" else
             pa.int64() if name == "day_of_week_num" else pa.string())
    for name in GOLD_COLUMNS
])
LOGGER = logging.getLogger(__name__)


def _lifecycle_record(state: EnvironmentState | dict[str, Any]) -> dict[str, str]:
    payload = state.payload if isinstance(state, EnvironmentState) else state
    return {
        "context_id": str(payload["context_id"]),
        "application_id": str(payload["application_id"]),
        "creation_datetime": pd.Timestamp(payload["creation_datetime"]).isoformat(),
        "deadline": pd.Timestamp(payload["deadline"]).isoformat(),
    }


def _register_lifecycle(registry: dict[str, list[dict[str, str]]],
                        state: EnvironmentState | dict[str, Any]) -> None:
    record = _lifecycle_record(state)
    creation, deadline = pd.Timestamp(record["creation_datetime"]), pd.Timestamp(record["deadline"])
    bucket = registry.setdefault(record["context_id"], [])
    for existing in bucket:
        if existing["application_id"] == record["application_id"]:
            raise RuntimeError(f"application_id lifecycle already registered: {record['application_id']}")
        if (creation < pd.Timestamp(existing["deadline"])
                and pd.Timestamp(existing["creation_datetime"]) < deadline):
            raise RuntimeError(f"overlapping application lifecycles for context: {record['context_id']}")
    bucket.append(record)


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      default=_json_default).encode("utf-8")


def _atomic_json(path: Path, value: Any) -> None:
    ensure_directory(path.parent)
    atomic_bytes(path, _json_bytes(value) + b"\n")


def _atomic_gzip_json(path: Path, value: Any) -> str:
    ensure_directory(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
            compressed.write(_json_bytes(value))
    digest = sha256(temporary.read_bytes()).hexdigest()
    durable_replace(temporary, path)
    return digest


def _read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_parquet(path: Path, rows: list[dict[str, Any]], columns: Iterable[str]) -> str:
    ensure_directory(path.parent)
    frame = pd.DataFrame(rows).reindex(columns=list(columns))
    if "event_datetime" in frame:
        frame["event_datetime"] = pd.to_datetime(frame["event_datetime"])
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    digest = _sha256(temporary)
    durable_replace(temporary, path)
    return digest


def _atomic_jsonl_gzip(path: Path, rows: list[dict[str, Any]]) -> str:
    ensure_directory(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
            for row in rows:
                compressed.write(_json_bytes(row) + b"\n")
    digest = _sha256(temporary)
    durable_replace(temporary, path)
    return digest


@dataclass(frozen=True)
class ScriptedSend:
    application_id: str
    when: str
    channel: str
    time_bucket: str

    def __post_init__(self) -> None:
        when = pd.Timestamp(self.when)
        if when.tzinfo is not None or not START <= when < END:
            raise ValueError("scripted SEND time must be timezone-naive and within May 2026")
        EnvironmentAction.campaign(self.channel, self.time_bucket)

    def timestamp(self) -> pd.Timestamp:
        return pd.Timestamp(self.when)


@dataclass(frozen=True)
class HistoricalFile:
    source_uri: str
    local_path: str
    sha256: str
    row_count: int
    size: int
    mtime_ns: int


class MaySimulationRunner:
    """Deterministic scripted driver using the production Environment API."""

    def __init__(self, *, run_id: str, historical_sources: Iterable[str] = (),
                 artifact_dir: str | Path = "artifacts/gold_events_v2",
                 output_root: str | Path = "data/output/runs", seed: int = 20260502,
                 initial_limit: int | None = None, arrival_limit: int | None = None,
                 scripted_sends: Iterable[ScriptedSend] = (), send_example: bool = False,
                 runtime_version: str = CORRECTED, stochastic_namespace: str | None = None):
        if not run_id or any(character in run_id for character in "\\/:*?\"<>|"):
            raise ValueError("run_id must be a non-empty filesystem-safe name")
        self.runtime_version = validate_runtime_version(runtime_version)
        self.stochastic_namespace = (str(stochastic_namespace) if stochastic_namespace is not None else "corrected-v2-reference-20260502") if self.runtime_version == FINAL else None
        self.run_id = run_id
        self.run_dir = Path(output_root) / run_id
        self.artifact_dir = Path(artifact_dir)
        self.seed = int(seed)
        self.initial_limit = initial_limit
        self.arrival_limit = arrival_limit
        self.requested_sources = tuple(map(str, historical_sources))
        self.scripted_sends = tuple(scripted_sends)
        self.send_example = bool(send_example)
        self.environment = Environment(self.artifact_dir, seed=self.seed, run_id=self.run_id,
                                       runtime_version=self.runtime_version, stochastic_namespace=self.stochastic_namespace)
        if initial_limit is not None or arrival_limit is not None:
            self.environment._runtime = self.environment._runtime.__class__(
                self.artifact_dir, seed=self.seed, run_id=self.run_id,
                initial_limit=initial_limit, arrival_limit=arrival_limit,
                runtime_version=self.runtime_version, stochastic_namespace=self.stochastic_namespace)

    @classmethod
    def open(cls, run_id: str, *, output_root: str | Path = "data/output/runs") -> "MaySimulationRunner":
        manifest_path = Path(output_root) / run_id / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"run manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return cls(
            run_id=run_id,
            historical_sources=[row["source_uri"] for row in manifest["historical_files"]],
            artifact_dir=manifest["artifact_dir"], output_root=output_root,
            runtime_version=manifest.get("runtime_version", LEGACY),
            stochastic_namespace=manifest.get("stochastic_namespace"),
            seed=int(manifest["seed"]), initial_limit=manifest.get("initial_limit"),
            arrival_limit=manifest.get("arrival_limit"),
            scripted_sends=[ScriptedSend(**row) for row in manifest.get("scripted_sends", [])],
            send_example=bool(manifest.get("send_example", False)),
        )

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "run_manifest.json"

    @property
    def summary_path(self) -> Path:
        return self.run_dir / "simulation_summary.json"

    @property
    def combined_path(self) -> Path:
        return self.run_dir / "combined_gold_events.parquet"

    def _expand_sources(self) -> list[str]:
        sources = list(self.requested_sources)
        if not sources and self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            sources = [row["source_uri"] for row in manifest["historical_files"]]
        expanded: list[str] = []
        for source in sources:
            if source.startswith("s3://"):
                expanded.append(source)
                continue
            path = Path(source)
            if path.is_dir():
                expanded.extend(str(item) for item in sorted(path.glob("*.parquet")))
            else:
                expanded.append(str(path))
        if not expanded:
            raise FileNotFoundError("no historical Parquet inputs configured")
        return expanded

    def _resolve_sources(self) -> list[Path]:
        resolved: list[Path] = []
        cache = self.run_dir / "source_cache"
        for source in self._expand_sources():
            if not source.startswith("s3://"):
                path = Path(source).resolve()
                if not path.is_file():
                    raise FileNotFoundError(f"historical input not found: {path}")
                resolved.append(path)
                continue
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError("install the 's3' extra to load S3 history") from exc
            parsed = urlparse(source)
            if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
                raise ValueError(f"invalid S3 URI: {source}")
            cache.mkdir(parents=True, exist_ok=True)
            name = sha256(source.encode()).hexdigest()[:12] + "-" + Path(parsed.path).name
            target = cache / name
            if not target.exists():
                boto3.client("s3").download_file(parsed.netloc, parsed.path.lstrip("/"), str(target))
            resolved.append(target)
        return resolved

    def validate_historical_inputs(self) -> list[HistoricalFile]:
        sources = self._expand_sources()
        paths = self._resolve_sources()
        expected_hashes = self.environment._runtime.sim.campaign_motif.manifest.get(
            "source_files_sha256", {})
        records: list[HistoricalFile] = []
        total_rows = 0
        minimum: pd.Timestamp | None = None
        maximum: pd.Timestamp | None = None
        for source, path in zip(sources, paths):
            parquet = pq.ParquetFile(path)
            columns = set(parquet.schema_arrow.names)
            missing = set(GOLD_COLUMNS) - columns
            if missing:
                raise ValueError(f"historical input {path} missing Gold columns: {sorted(missing)}")
            digest = _sha256(path)
            expected = expected_hashes.get(Path(source).name)
            if expected is not None and digest != expected:
                raise ValueError(f"historical source hash mismatch: {path}")
            for batch in parquet.iter_batches(columns=["event_datetime"], batch_size=262_144):
                bounds = pc.min_max(batch.column(0)).as_py()
                low, high = pd.Timestamp(bounds["min"]), pd.Timestamp(bounds["max"])
                minimum = low if minimum is None else min(minimum, low)
                maximum = high if maximum is None else max(maximum, high)
            stat = path.stat()
            total_rows += parquet.metadata.num_rows
            records.append(HistoricalFile(source, str(path), digest, parquet.metadata.num_rows,
                                          stat.st_size, stat.st_mtime_ns))
        if minimum is None or maximum is None:
            raise ValueError("historical input is empty")
        if minimum < pd.Timestamp("2026-03-01") or maximum >= START:
            raise ValueError(f"historical input must contain only March-April 2026 rows; got {minimum} to {maximum}")
        LOGGER.info("validated historical Gold", extra={"rows": total_rows,
                                                         "minimum": str(minimum), "maximum": str(maximum)})
        return records

    def initialize(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self._assert_manifest_compatible(manifest)
            return manifest
        historical = self.validate_historical_inputs()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        states: dict[str, dict[str, Any]] = {}
        pending: dict[str, list[dict[str, Any]]] = {}
        for raw in self.environment._runtime.initial_population():
            state = self.environment._runtime.new_state(raw, activation=START)
            states[state["application_id"]] = state
            pending[state["application_id"]] = []
        lifecycle_registry: dict[str, list[dict[str, str]]] = {}
        if self.runtime_version == FINAL:
            # Prior historical lifecycles are registry constraints, not active
            # simulator states. Preserve their original customer identity.
            historical_life = pd.read_parquet(self.artifact_dir / "application_lifecycle_state.parquet").dropna(subset=["context_id", "application_created_at"])
            # Register the proven prior lifecycles selected by the identity
            # builder, rather than treating every historical registry row as a
            # live world constraint. Immutable source anomalies remain audited.
            prior_ids = set(self.environment._runtime.arrivals.prior_application_id.dropna())
            historical_life = historical_life.loc[historical_life.application_id.isin(prior_ids)]
            for row in historical_life.itertuples(index=False):
                _register_lifecycle(lifecycle_registry, dict(context_id=str(row.context_id), application_id=str(row.application_id), creation_datetime=row.application_created_at, deadline=row.application_created_at+pd.Timedelta(days=30)))
        for state in states.values():
            records = lifecycle_registry.get(state["context_id"], [])
            if not any(r["application_id"] == state["application_id"] for r in records):
                _register_lifecycle(lifecycle_registry, state)
        manifest = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "run_id": self.run_id, "seed": self.seed, "runtime_version": self.runtime_version,
            "artifact_dir": str(self.artifact_dir.resolve()),
            "artifact_hashes": self.environment.artifact_hashes,
            "historical_files": [asdict(row) for row in historical],
            "historical_rows": sum(row.row_count for row in historical),
            "initial_carried_applications": len(states),
            "simulation_start": START.isoformat(), "simulation_end": END.isoformat(),
            "driver": "SCRIPTED_EXTERNAL_ACTIONS",
            "default_action": "WAIT_NO_ACTION",
            "initial_limit": self.initial_limit, "arrival_limit": self.arrival_limit,
            "scripted_sends": [asdict(row) for row in self.scripted_sends],
            "send_example": self.send_example,
        }
        if self.runtime_version == FINAL:
            manifest["stochastic_namespace"] = self.stochastic_namespace
        _atomic_json(self.manifest_path, manifest)
        checkpoint = {
            "checkpoint_version": CHECKPOINT_VERSION, "run_id": self.run_id,
            "runtime_version": self.runtime_version,
            "completed_day": "2026-04-30", "states": states, "pending": pending,
            "lifecycle_registry": lifecycle_registry,
            "terminal_counts": {}, "daily": [], "may_gold_rows": 0,
            "arrivals_activated": 0, "arrivals_reused": 0, "scripted_outcomes": [],
        }
        if self.runtime_version == FINAL:
            checkpoint["stochastic_namespace"] = self.stochastic_namespace
        self._commit_initial_checkpoint(checkpoint)
        self._write_summary(checkpoint, status="INITIALIZED")
        return manifest

    def _assert_manifest_compatible(self, manifest: dict[str, Any]) -> None:
        expected = {
            "run_id": self.run_id, "seed": self.seed, "runtime_version": self.runtime_version,
            "artifact_hashes": self.environment.artifact_hashes,
            "initial_limit": self.initial_limit, "arrival_limit": self.arrival_limit,
            "scripted_sends": [asdict(row) for row in self.scripted_sends],
            "send_example": self.send_example,
        }
        if self.runtime_version == FINAL:
            expected["stochastic_namespace"] = self.stochastic_namespace
        for key, value in expected.items():
            if manifest.get(key, LEGACY if key == "runtime_version" else None) != value:
                raise ValueError(f"run configuration changed across restart: {key}")

    def _commit_initial_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        path = self.run_dir / "checkpoints" / "2026-04-30.json.gz"
        digest = _atomic_gzip_json(path, checkpoint)
        _atomic_json(self.run_dir / "checkpoints" / "2026-04-30.commit.json", {
            "completed_day": "2026-04-30", "checkpoint": str(path.relative_to(self.run_dir)),
            "checkpoint_sha256": digest,
        })

    def _latest_checkpoint(self) -> tuple[dict[str, Any], dict[str, Any]]:
        commits = sorted((self.run_dir / "checkpoints").glob("*.commit.json"))
        if not commits:
            raise RuntimeError("run has not been initialized")
        commit = json.loads(commits[-1].read_text(encoding="utf-8"))
        path = self.run_dir / commit["checkpoint"]
        if _sha256(path) != commit["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {path}")
        checkpoint = _read_gzip_json(path)
        if checkpoint.get("runtime_version", LEGACY) != self.runtime_version:
            raise ValueError("checkpoint runtime_version mismatch")
        if self.runtime_version == FINAL and checkpoint.get("stochastic_namespace") != self.stochastic_namespace:
            raise ValueError("checkpoint stochastic_namespace mismatch")
        return checkpoint, commit

    def _rules_for(self, application_id: str, start: pd.Timestamp,
                   end: pd.Timestamp) -> list[ScriptedSend]:
        return sorted((rule for rule in self.scripted_sends
                       if rule.application_id == application_id
                       and start <= rule.timestamp() < end), key=lambda row: row.timestamp())

    def _advance_application(self, state: EnvironmentState, start: pd.Timestamp,
                             end: pd.Timestamp, rules: list[ScriptedSend]
                             ) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
        events: list[dict] = []
        decisions: list[dict] = []
        explanations: list[dict] = []
        outcomes: list[dict] = []
        cursor = start
        for index, rule in enumerate(rules):
            when = rule.timestamp()
            if cursor < when and not state.terminal:
                result = self.environment.advance(state, EnvironmentAction.no_action(), cursor, when, **({"end_inclusive": True} if self.runtime_version == FINAL else {}))
                events.extend(result.events); decisions.extend(result.decisions)
                explanations.extend(result.explanations)
            cursor = when
            if state.terminal:
                outcomes.append({**asdict(rule), "status": "SKIPPED_TERMINAL"})
                continue
            segment_end = rules[index + 1].timestamp() if index + 1 < len(rules) else end
            result = self.environment.advance(
                state, EnvironmentAction.campaign(rule.channel, rule.time_bucket), cursor, segment_end,
                **({"end_inclusive": segment_end < end} if self.runtime_version == FINAL else {}))
            events.extend(result.events); decisions.extend(result.decisions)
            explanations.extend(result.explanations)
            outcomes.append({**asdict(rule), "status": result.decisions[-1]["status"]})
            cursor = segment_end
        if cursor < end and not state.terminal:
            result = self.environment.advance(state, EnvironmentAction.no_action(), cursor, end,
                **({"end_inclusive": False} if self.runtime_version == FINAL else {}))
            events.extend(result.events); decisions.extend(result.decisions)
            explanations.extend(result.explanations)
        return events, decisions, explanations, outcomes

    def run_day(self, day: str | pd.Timestamp) -> dict[str, Any]:
        day = pd.Timestamp(day).normalize()
        if not START <= day < END:
            raise ValueError("run-day accepts May 1-31, 2026 only")
        self.initialize()
        commit_path = self.run_dir / "checkpoints" / f"{day.date()}.commit.json"
        if commit_path.exists():
            return {**json.loads(commit_path.read_text(encoding="utf-8")),
                    "status": "IDEMPOTENT_RETRY"}
        checkpoint, prior_commit = self._latest_checkpoint()
        expected = pd.Timestamp(checkpoint["completed_day"]) + pd.Timedelta(days=1)
        if day != expected:
            raise RuntimeError(f"cannot skip days: next required day is {expected.date()}")
        following = day + pd.Timedelta(days=1)
        states = {app: EnvironmentState(payload, list(checkpoint["pending"].get(app, [])))
                  for app, payload in checkpoint["states"].items()}
        lifecycle_registry = checkpoint.get("lifecycle_registry")
        if lifecycle_registry is None:
            lifecycle_registry = {}
            for state in states.values():
                _register_lifecycle(lifecycle_registry, state)
        registered_applications = {
            record["application_id"] for records in lifecycle_registry.values() for record in records
        }
        activated: set[str] = set()
        reused_arrivals = 0
        for raw in self.environment._runtime.arrivals_for_day(day):
            activation = pd.Timestamp(raw["activation_datetime"])
            context = str(raw["snapshot_context_id"])
            active_app = active_application_for_context(
                lifecycle_registry.get(context, ()), context, activation)
            if active_app is not None:
                reused_arrivals += 1
                continue
            state_payload = self.environment._runtime.new_state(
                raw, activation=activation)
            app = state_payload["application_id"]
            if app in registered_applications:
                raise RuntimeError(f"arrival reused application_id: {app}")
            _register_lifecycle(lifecycle_registry, state_payload)
            registered_applications.add(app)
            states[app] = EnvironmentState(state_payload, [])
            activated.add(app)
        example_rule: ScriptedSend | None = None
        if self.send_example and day == pd.Timestamp("2026-05-03"):
            candidates = sorted(app for app, state in states.items()
                                if not state.terminal
                                and pd.Timestamp(state.payload["activation_datetime"]) <= day + pd.Timedelta(hours=9)
                                and pd.Timestamp(state.payload["deadline"]) > day + pd.Timedelta(hours=9))
            if candidates:
                example_rule = ScriptedSend(candidates[0], "2026-05-03T09:00:00", "WA", "MORNING")
        gold: list[dict] = []
        decisions: list[dict] = []
        explanations: list[dict] = []
        scripted_outcomes = list(checkpoint.get("scripted_outcomes", []))
        terminal_counts = Counter(checkpoint.get("terminal_counts", {}))
        for app in sorted(states, key=lambda value: (states[value].payload["context_id"], value)):
            state = states[app]
            start = (pd.Timestamp(state.payload["activation_datetime"])
                     if app in activated else day)
            rules = self._rules_for(app, start, following)
            if example_rule is not None and example_rule.application_id == app:
                rules.append(example_rule)
            app_gold, app_decisions, app_explanations, outcomes = self._advance_application(
                state, start, following, sorted(rules, key=lambda row: row.timestamp()))
            gold.extend(app_gold); decisions.extend(app_decisions)
            explanations.extend(app_explanations); scripted_outcomes.extend(outcomes)
            if app_gold:
                state.payload["last_visible_row_key"] = app_gold[-1]["cumulative_row_key"]
            if state.terminal:
                terminal_counts[state.payload["termination_reason"]] += 1
        surviving = {app: state for app, state in states.items() if not state.terminal}
        event_order = {row["cumulative_row_key"]: i for i, row in enumerate(gold)} if self.runtime_version == FINAL else {}
        gold.sort(key=lambda row: (pd.Timestamp(row["event_datetime"]), str(row["application_id"]),
                                   event_order[row["cumulative_row_key"]] if self.runtime_version == FINAL else str(row["cumulative_row_key"])))
        event_path = self.run_dir / "may" / str(day.date()) / "gold_events.parquet"
        event_hash = _atomic_parquet(event_path, gold, SIMULATION_COLUMNS)
        decision_columns = sorted({key for row in decisions for key in row})
        decision_path = self.run_dir / "audit" / str(day.date()) / "decisions.parquet"
        decision_hash = _atomic_parquet(decision_path, decisions, decision_columns)
        explanation_path = self.run_dir / "audit" / str(day.date()) / "explanations.jsonl.gz"
        explanation_hash = _atomic_jsonl_gzip(explanation_path, explanations)
        status_counts = Counter(str(row["status"]) for row in decisions)
        daily = {
            "day": str(day.date()), "applications_processed": len(states),
            "active_applications": len(surviving), "arrivals": len(activated),
            "gold_rows": len(gold), "decisions": len(decisions),
            "decision_statuses": dict(status_counts),
            "ptp": sum(row["journey_substage"] == "Push to Partner" for row in gold),
            "campaign_sends": sum(row["journey_substage"] == "campaign_sent" for row in gold),
        }
        next_checkpoint = {
            "checkpoint_version": CHECKPOINT_VERSION, "run_id": self.run_id,
            "runtime_version": self.runtime_version,
            "completed_day": str(day.date()),
            "states": {app: state.payload for app, state in surviving.items()},
            "pending": {app: state.pending_events for app, state in surviving.items()},
            "lifecycle_registry": lifecycle_registry,
            "terminal_counts": dict(terminal_counts),
            "daily": [*checkpoint.get("daily", []), daily],
            "may_gold_rows": int(checkpoint.get("may_gold_rows", 0)) + len(gold),
            "arrivals_activated": int(checkpoint.get("arrivals_activated", 0)) + len(activated),
            "arrivals_reused": int(checkpoint.get("arrivals_reused", 0)) + reused_arrivals,
            "scripted_outcomes": scripted_outcomes,
        }
        if self.runtime_version == FINAL:
            next_checkpoint["stochastic_namespace"] = self.stochastic_namespace
        checkpoint_path = self.run_dir / "checkpoints" / f"{day.date()}.json.gz"
        checkpoint_hash = _atomic_gzip_json(checkpoint_path, next_checkpoint)
        commit = {
            **daily, "status": "COMMITTED",
            "previous_checkpoint_sha256": prior_commit["checkpoint_sha256"],
            "checkpoint": str(checkpoint_path.relative_to(self.run_dir)),
            "checkpoint_sha256": checkpoint_hash,
            "gold_partition": str(event_path.relative_to(self.run_dir)),
            "gold_partition_sha256": event_hash,
            "decision_partition_sha256": decision_hash,
            "explanation_partition_sha256": explanation_hash,
        }
        _atomic_json(commit_path, commit)
        self._write_summary(next_checkpoint, status="COMPLETED" if following == END else "RUNNING")
        return commit

    def run_through(self, last_day: str | pd.Timestamp = "2026-05-31") -> list[dict[str, Any]]:
        last = pd.Timestamp(last_day).normalize()
        if not START <= last < END:
            raise ValueError("last_day must be in May 2026")
        self.initialize()
        return [self.run_day(day) for day in pd.date_range(START, last, freq="D")]

    def _write_summary(self, checkpoint: dict[str, Any], *, status: str) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        summary = {
            "run_id": self.run_id, "status": status, "seed": self.seed,
            "runtime_version": self.runtime_version,
            "completed_day": checkpoint["completed_day"],
            "historical_rows": manifest["historical_rows"],
            "may_simulated_rows": checkpoint["may_gold_rows"],
            "combined_rows": manifest["historical_rows"] + checkpoint["may_gold_rows"],
            "initial_carried_applications": manifest["initial_carried_applications"],
            "arrivals_activated": checkpoint["arrivals_activated"],
            "arrivals_reused": checkpoint.get("arrivals_reused", 0),
            "active_applications": len(checkpoint["states"]),
            "terminal_distribution": checkpoint["terminal_counts"],
            "daily": checkpoint["daily"],
            "scripted_outcomes": checkpoint.get("scripted_outcomes", []),
            "combined_parquet": str(self.combined_path),
        }
        if self.runtime_version == FINAL:
            summary["stochastic_namespace"] = self.stochastic_namespace
        _atomic_json(self.summary_path, summary)

    def export_combined(self) -> dict[str, Any]:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        paths = [Path(row["local_path"]) for row in manifest["historical_files"]]
        may_paths = sorted((self.run_dir / "may").glob("*/gold_events.parquet"))
        if not paths:
            raise RuntimeError("run has no historical inputs")
        schema = GOLD_ARROW_SCHEMA
        ensure_directory(self.combined_path.parent)
        temporary = self.combined_path.with_suffix(".parquet.tmp")
        rows = 0
        with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
            for path in paths:
                source = pq.ParquetFile(path)
                for batch in source.iter_batches(columns=list(GOLD_COLUMNS), batch_size=262_144):
                    table = pa.Table.from_batches([batch]).cast(schema, safe=False)
                    writer.write_table(table)
                    rows += table.num_rows
            for path in may_paths:
                table = pq.read_table(path, columns=list(GOLD_COLUMNS)).cast(schema, safe=False)
                writer.write_table(table)
                rows += table.num_rows
        durable_replace(temporary, self.combined_path)
        expected = manifest["historical_rows"] + sum(
            pq.ParquetFile(path).metadata.num_rows for path in may_paths)
        if rows != expected:
            raise RuntimeError(f"combined export row mismatch: {rows} != {expected}")
        return {"path": str(self.combined_path), "rows": rows, "sha256": _sha256(self.combined_path)}

    def validate_run(self, *, require_complete: bool = True) -> dict[str, Any]:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        bundle_manifest = json.loads(
            (Path(manifest["artifact_dir"]) / "manifest.json").read_text(encoding="utf-8"))
        require_event_name = bundle_manifest.get("bundle_version") == "gold_events_v2"
        commits = sorted((self.run_dir / "checkpoints").glob("2026-05-*.commit.json"))
        if require_complete and len(commits) != 31:
            raise ValueError(f"completed May run requires 31 commits; found {len(commits)}")
        keys: set[str] = set()
        owner: dict[str, str] = {}
        terminal: dict[str, pd.Timestamp] = {}
        last_event: dict[str, pd.Timestamp] = {}
        creation_counts: Counter[str] = Counter()
        simulated_applications: set[str] = set()
        ptp_applications: set[str] = set()
        may_rows = 0
        missing_event_name = 0
        june_leakage = 0
        violations: list[str] = []
        for event_path in sorted((self.run_dir / "may").glob("*/gold_events.parquet")):
            frame = pd.read_parquet(event_path)
            missing_names = int((frame.event_name.isna() | frame.event_name.eq("")).sum())
            missing_event_name += missing_names
            if require_event_name and missing_names:
                violations.append(f"missing simulated event_name: {event_path}")
            day = pd.Timestamp(event_path.parent.name)
            times = pd.to_datetime(frame.event_datetime)
            if not frame.empty and (times.lt(day).any() or times.ge(day + pd.Timedelta(days=1)).any()):
                violations.append(f"event outside partition day: {event_path}")
            june_leakage += int(times.ge(END).sum())
            for row in frame.itertuples(index=False):
                key = str(row.cumulative_row_key)
                if key in keys:
                    violations.append(f"duplicate simulated row key: {key}")
                keys.add(key)
                app, context, when = str(row.application_id), str(row.context_id), pd.Timestamp(row.event_datetime)
                if app in owner and owner[app] != context:
                    violations.append(f"application owner changed: {app}")
                owner[app] = context
                simulated_applications.add(app)
                if app in last_event and when < last_event[app]:
                    violations.append(f"out-of-order simulated event: {app}")
                last_event[app] = when
                if app in terminal and when > terminal[app]:
                    violations.append(f"post-terminal event: {app}")
                if str(row.journey_substage) == "Push to Partner":
                    terminal[app] = when
                    ptp_applications.add(app)
                if str(row.journey_substage) == "Application Created":
                    creation_counts[app] += 1
            may_rows += len(frame)
        latest, _ = self._latest_checkpoint()
        arrivals = self.environment._runtime.arrivals
        if not require_complete:
            completed_through = pd.Timestamp(latest["completed_day"]) + pd.Timedelta(days=1)
            arrivals = arrivals.loc[pd.to_datetime(arrivals.activation_datetime).lt(completed_through)]
        arrival_applications = set(arrivals.application_id.astype(str))
        duplicate_creation_rows = sum(max(0, count - 1) for count in creation_counts.values())
        missing_arrival_creations = arrival_applications - creation_counts.keys()
        nonarrival_creations = creation_counts.keys() - arrival_applications
        if duplicate_creation_rows:
            violations.append(f"duplicate simulated Application Created rows: {duplicate_creation_rows}")
        if missing_arrival_creations:
            violations.append(f"May arrivals without Application Created: {len(missing_arrival_creations)}")
        if nonarrival_creations:
            violations.append(f"Application Created emitted for carried applications: {len(nonarrival_creations)}")
        lifecycle_registry = latest.get("lifecycle_registry", {})
        overlapping_contexts = 0
        for records in lifecycle_registry.values():
            ordered = sorted(records, key=lambda record: pd.Timestamp(record["creation_datetime"]))
            if any(pd.Timestamp(current["creation_datetime"]) < pd.Timestamp(previous["deadline"])
                   for previous, current in zip(ordered, ordered[1:])):
                overlapping_contexts += 1
        if overlapping_contexts:
            violations.append(f"contexts with overlapping 30-day application lifecycles: {overlapping_contexts}")
        if june_leakage:
            violations.append(f"simulated rows on or after June 1: {june_leakage}")
        historical_unchanged = True
        for record in manifest["historical_files"]:
            stat = Path(record["local_path"]).stat()
            historical_unchanged &= stat.st_size == record["size"] and stat.st_mtime_ns == record["mtime_ns"]
        combined_rows = None
        combined_schema_ok = None
        if self.combined_path.exists():
            combined = pq.ParquetFile(self.combined_path)
            combined_rows = combined.metadata.num_rows
            combined_schema_ok = combined.schema_arrow.names == list(GOLD_COLUMNS)
            if combined_rows != manifest["historical_rows"] + may_rows:
                violations.append("combined Parquet row count mismatch")
            if not combined_schema_ok:
                violations.append("combined Parquet does not have canonical Gold schema")
        if not historical_unchanged:
            violations.append("historical source size or modification time changed")
        report = {
            "run_id": self.run_id, "runtime_version": self.runtime_version,
            "status": "PASS" if not violations and (not require_complete or len(commits) == 31) else "FAIL",
            "committed_days": len(commits), "historical_rows": manifest["historical_rows"],
            "may_rows": may_rows, "combined_rows": combined_rows,
            "combined_schema_ok": combined_schema_ok,
            "historical_sources_unchanged": historical_unchanged,
            "unique_simulated_row_keys": len(keys), "violations": violations,
            "intended_may_arrivals": len(arrival_applications),
            "unique_may_arrival_application_ids": len(arrival_applications),
            "application_created_rows": sum(creation_counts.values()),
            "duplicate_application_created_rows": duplicate_creation_rows,
            "contexts_multiple_applications_inside_30_days": overlapping_contexts,
            "carried_forward_applications_reused": manifest["initial_carried_applications"],
            "carried_forward_applications_with_may_events": len(simulated_applications - arrival_applications),
            "unique_ptp_applications": len(ptp_applications),
            "missing_event_name": missing_event_name,
            "june_leakage": june_leakage,
            "terminal_distribution": latest.get("terminal_counts", {}),
            "pending_after_boundary": sum(len(rows) for rows in latest.get("pending", {}).values()),
        }
        _atomic_json(self.run_dir / "validation_report.json", report)
        if report["status"] != "PASS":
            raise ValueError(f"run validation failed: {violations}")
        return report

    def inspect(self, *, sample_applications: int = 3) -> dict[str, Any]:
        summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
        may_paths = sorted((self.run_dir / "may").glob("*/gold_events.parquet"))
        frames = [pd.read_parquet(path, columns=["application_id", "context_id", "event_datetime",
                                                  "journey_stage", "journey_substage", "channel"])
                  for path in may_paths]
        may = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        samples: dict[str, list[dict[str, Any]]] = {}
        if not may.empty:
            ranked = sorted(may.application_id.dropna().astype(str).unique(),
                            key=lambda value: sha256(f"{self.seed}|preview|{value}".encode()).hexdigest())
            for app in ranked[:sample_applications]:
                journey = may.loc[may.application_id.astype(str).eq(app)].sort_values("event_datetime")
                samples[app] = journey.assign(
                    event_datetime=lambda frame: frame.event_datetime.astype(str)).to_dict("records")
        combined_contexts = combined_applications = None
        if self.combined_path.exists():
            identities = pq.read_table(self.combined_path, columns=["context_id", "application_id"])
            combined_contexts = int(pc.count_distinct(identities["context_id"]).as_py())
            combined_applications = int(pc.count_distinct(identities["application_id"]).as_py())
        return {
            "historical_event_count": summary["historical_rows"],
            "may_simulated_event_count": len(may),
            "total_combined_rows": summary["historical_rows"] + len(may),
            "daily_may_event_counts": ({str(key.date()): int(value) for key, value in
                may.groupby(pd.to_datetime(may.event_datetime).dt.normalize()).size().items()} if not may.empty else {}),
            "may_distinct_context_ids": int(may.context_id.nunique()) if not may.empty else 0,
            "may_distinct_application_ids": int(may.application_id.nunique()) if not may.empty else 0,
            "combined_distinct_context_ids": combined_contexts,
            "combined_distinct_application_ids": combined_applications,
            "sample_chronological_journeys": samples,
            "final_lifecycle_terminal_distribution": summary["terminal_distribution"],
            "active_applications_at_boundary": summary["active_applications"],
            "combined_parquet": str(self.combined_path),
        }
