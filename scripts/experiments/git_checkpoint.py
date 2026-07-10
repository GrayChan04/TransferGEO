#!/usr/bin/env python3
from __future__ import annotations

import argparse, subprocess
from pathlib import Path

def run(cmd, check=True):
    print("+", " ".join(cmd))
    return subprocess.run(cmd, text=True, check=check, capture_output=False)

def main() -> None:
    ap = argparse.ArgumentParser(description="Create a git checkpoint for an experiment.")
    ap.add_argument("--experiment_id", required=True)
    ap.add_argument("--message", default="")
    ap.add_argument("--branch", default="")
    ap.add_argument("--paths", nargs="*", default=[
        "configs/experiments",
        "experiments",
        "results/processed",
        "private/06_结果解读.md",
        "private/10_下一步行动.md",
    ])
    ap.add_argument("--create_branch", action="store_true")
    args = ap.parse_args()

    if args.create_branch:
        branch = args.branch or f"exp/{args.experiment_id}"
        subprocess.run(["git", "checkout", "-B", branch], check=False)

    existing = [p for p in args.paths if Path(p).exists()]
    if existing:
        run(["git", "add", *existing], check=False)
    msg = args.message or f"exp: {args.experiment_id}"
    run(["git", "commit", "-m", msg], check=False)
    subprocess.run(["git", "rev-parse", "HEAD"], check=False)

if __name__ == "__main__":
    main()
