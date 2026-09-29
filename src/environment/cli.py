from __future__ import annotations
import argparse
import json
from .models.artifact_loader import ArtifactLoader

def main() -> None:
    parser = argparse.ArgumentParser(prog="environment")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-bundle")
    validate.add_argument("--artifact-dir", default="artifacts/gold_events_v1")
    args = parser.parse_args()
    if args.command == "validate-bundle":
        hashes = ArtifactLoader(args.artifact_dir).validate()
        print(json.dumps({"status": "ok", "artifacts": len(hashes)}, indent=2))

if __name__ == "__main__":
    main()
