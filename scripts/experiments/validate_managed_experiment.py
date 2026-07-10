#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv
from pathlib import Path

REQUIRED = ["experiment_id", "phase", "config_path", "command", "result_dir", "metrics_path"]

def load_registry(exp_id: str) -> dict:
    path = Path("experiments/registry.csv")
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("experiment_id") == exp_id:
                return row
    return {}

def main() -> None:
    ap = argparse.ArgumentParser(description="Validate a managed experiment before running it.")
    ap.add_argument("--experiment_id", required=True)
    ap.add_argument("--require_git_policy", action="store_true")
    args = ap.parse_args()

    row = load_registry(args.experiment_id)
    missing = []

    if not row:
        missing.append("registry entry in experiments/registry.csv")
    else:
        for key in REQUIRED:
            if not row.get(key):
                missing.append(key)

        config_path = row.get("config_path", "")
        if config_path and not Path(config_path).exists():
            missing.append(f"config file not found: {config_path}")

        result_dir = row.get("result_dir", "")
        if not result_dir:
            missing.append("result_dir")

        metrics_path = row.get("metrics_path", "")
        if not metrics_path:
            missing.append("metrics_path")

        if args.require_git_policy:
            if not row.get("git_branch"):
                missing.append("git_branch / create_branch decision")
            # git_commit may be empty before run, so this script does not require a commit hash.

    if missing:
        print("Managed experiment is NOT ready.")
        print("Missing or unclear fields:")
        for item in missing:
            print(f"- {item}")
        print()
        print("Ask the user to clarify these fields before running the experiment.")
        raise SystemExit(1)

    print("Managed experiment is ready to run.")
    print(f"experiment_id: {args.experiment_id}")
    print(f"command: {row.get('command')}")
    print(f"metrics_path: {row.get('metrics_path')}")

if __name__ == "__main__":
    main()
