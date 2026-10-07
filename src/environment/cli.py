from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import yaml

from .models.artifact_loader import ArtifactLoader
from .persistence.postgres_demo import (
    DEFAULT_COMBINED_PATH,
    initialize_demo_schema,
    load_combined_parquet,
    validate_combined_parquet,
    validate_demo_database,
)
from .reference_inputs import validate_reference_inputs
from .simulation import MaySimulationRunner, ScriptedSend
from .runtime.semantics import CORRECTED, RUNTIME_VERSIONS


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}


def _parse_scripted_send(value: str) -> ScriptedSend:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "scripted SEND must be APPLICATION_ID,ISO_TIMESTAMP,CHANNEL,TIME_BUCKET")
    try:
        return ScriptedSend(parts[0], parts[1], parts[2], parts[3])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _add_common_run_arguments(parser: argparse.ArgumentParser, *, historical: bool) -> None:
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", default="config/environment.yaml")
    parser.add_argument("--output-root")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--runtime-version", choices=RUNTIME_VERSIONS)
    if historical:
        parser.add_argument("--historical", action="append", default=[],
                            help="Parquet file, directory, or S3 URI; repeat for multiple inputs")
        parser.add_argument("--initial-limit", type=int)
        parser.add_argument("--arrival-limit", type=int)
        parser.add_argument("--scripted-send", action="append", type=_parse_scripted_send, default=[])
        parser.add_argument("--send-example", action="store_true")


def _new_runner(args: argparse.Namespace) -> MaySimulationRunner:
    config = _load_config(args.config)
    environment = config.get("environment", {})
    simulation = config.get("simulation", {})
    sources = args.historical or simulation.get("historical_inputs", [])
    return MaySimulationRunner(
        run_id=args.run_id, historical_sources=sources,
        artifact_dir=args.artifact_dir or environment.get("artifact_uri", "artifacts/gold_events_v2"),
        output_root=args.output_root or simulation.get("output_root", "data/output/runs"),
        seed=args.seed if args.seed is not None else int(environment.get("seed", 20260502)),
        runtime_version=args.runtime_version or environment.get("runtime_version", CORRECTED),
        initial_limit=args.initial_limit, arrival_limit=args.arrival_limit,
        scripted_sends=args.scripted_send, send_example=args.send_example,
    )


def _existing_runner(args: argparse.Namespace) -> MaySimulationRunner:
    config = _load_config(args.config)
    output_root = args.output_root or config.get("simulation", {}).get("output_root", "data/output/runs")
    runner = MaySimulationRunner.open(args.run_id, output_root=output_root)
    if args.runtime_version is not None and args.runtime_version != runner.runtime_version:
        raise ValueError("requested runtime_version differs from saved run")
    return runner


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(prog="environment")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-bundle")
    validate.add_argument("--artifact-dir", default="artifacts/gold_events_v2")
    references = commands.add_parser("validate-reference-inputs")
    references.add_argument("--spec", default="config/reference_inputs.json")
    validate_input = commands.add_parser("validate-input")
    _add_common_run_arguments(validate_input, historical=True)
    initialize = commands.add_parser("initialize-may")
    _add_common_run_arguments(initialize, historical=True)
    run_day = commands.add_parser("run-day")
    _add_common_run_arguments(run_day, historical=False)
    run_day.add_argument("--day", required=True)
    run_may = commands.add_parser("run-may")
    _add_common_run_arguments(run_may, historical=True)
    run_may.add_argument("--through", default="2026-05-31")
    run_may.add_argument("--no-export", action="store_true")
    resume = commands.add_parser("resume-may")
    _add_common_run_arguments(resume, historical=False)
    resume.add_argument("--through", default="2026-05-31")
    resume.add_argument("--no-export", action="store_true")
    inspect = commands.add_parser("inspect-run")
    _add_common_run_arguments(inspect, historical=False)
    inspect.add_argument("--sample-applications", type=int, default=3)
    export = commands.add_parser("export-combined")
    _add_common_run_arguments(export, historical=False)
    validate_run = commands.add_parser("validate-run")
    _add_common_run_arguments(validate_run, historical=False)
    validate_run.add_argument("--allow-partial", action="store_true")
    demo_input = commands.add_parser(
        "postgres-demo-validate-input",
        help="validate the combined Parquet intended for environment_demo",
    )
    demo_input.add_argument("--input", default=str(DEFAULT_COMBINED_PATH))
    demo_init = commands.add_parser(
        "postgres-demo-init",
        help="create environment_demo table, indexes, and analytics views",
    )
    demo_init.add_argument("--schema-sql", default="sql/environment_demo.sql")
    demo_load = commands.add_parser(
        "postgres-demo-load",
        help="atomically copy combined Gold Events Parquet into PostgreSQL",
    )
    demo_load.add_argument("--input", default=str(DEFAULT_COMBINED_PATH))
    demo_load.add_argument("--simulation-start", type=datetime.fromisoformat,
                           default=datetime(2026, 5, 1))
    demo_load.add_argument("--replace", action="store_true")
    demo_validate = commands.add_parser(
        "postgres-demo-validate",
        help="validate environment_demo counts, invariants, and views",
    )
    demo_validate.add_argument("--expected-total", type=int)
    demo_validate.add_argument("--expected-historical", type=int)
    demo_validate.add_argument("--expected-simulated", type=int)

    args = parser.parse_args()
    try:
        if args.command == "validate-bundle":
            _print({"status": "ok", "artifacts": len(ArtifactLoader(args.artifact_dir).validate())})
        elif args.command == "validate-reference-inputs":
            _print(validate_reference_inputs(args.spec))
        elif args.command == "validate-input":
            runner = _new_runner(args)
            files = runner.validate_historical_inputs()
            _print({"status": "ok", "files": [asdict(row) for row in files],
                    "rows": sum(row.row_count for row in files)})
        elif args.command == "initialize-may":
            _print(_new_runner(args).initialize())
        elif args.command == "run-day":
            _print(_existing_runner(args).run_day(args.day))
        elif args.command == "run-may":
            runner = _new_runner(args)
            commits = runner.run_through(args.through)
            result: dict[str, Any] = {
                "days": commits, "summary": json.loads(runner.summary_path.read_text(encoding="utf-8"))}
            if not args.no_export and args.through == "2026-05-31":
                result["combined"] = runner.export_combined()
                result["validation"] = runner.validate_run()
            _print(result)
        elif args.command == "resume-may":
            runner = _existing_runner(args)
            commits = runner.run_through(args.through)
            result = {"days": commits,
                      "summary": json.loads(runner.summary_path.read_text(encoding="utf-8"))}
            if not args.no_export and args.through == "2026-05-31":
                result["combined"] = runner.export_combined()
                result["validation"] = runner.validate_run()
            _print(result)
        elif args.command == "inspect-run":
            _print(_existing_runner(args).inspect(sample_applications=args.sample_applications))
        elif args.command == "export-combined":
            _print(_existing_runner(args).export_combined())
        elif args.command == "validate-run":
            _print(_existing_runner(args).validate_run(require_complete=not args.allow_partial))
        elif args.command == "postgres-demo-validate-input":
            _print({"status": "PASS", **validate_combined_parquet(args.input)})
        elif args.command == "postgres-demo-init":
            _print(initialize_demo_schema(args.schema_sql))
        elif args.command == "postgres-demo-load":
            _print(load_combined_parquet(
                args.input,
                simulation_start=args.simulation_start,
                replace=args.replace,
            ))
        elif args.command == "postgres-demo-validate":
            result = validate_demo_database(
                expected_total=args.expected_total,
                expected_historical=args.expected_historical,
                expected_simulated=args.expected_simulated,
            )
            _print(result)
            if result["status"] != "PASS":
                parser.exit(1)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"{args.command} failed: {exc}\n")


if __name__ == "__main__":
    main()
