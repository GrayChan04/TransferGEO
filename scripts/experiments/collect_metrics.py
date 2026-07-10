#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path

def main() -> None:
    ap = argparse.ArgumentParser(description="Collect metrics from a metrics JSON file.")
    ap.add_argument("--metrics_path", required=True)
    ap.add_argument("--allow_missing", action="store_true")
    args = ap.parse_args()

    p = Path(args.metrics_path)
    if not p.exists():
        if args.allow_missing:
            print("{}")
            return
        raise SystemExit(f"Metrics file not found: {p}")

    data = json.loads(p.read_text(encoding="utf-8"))
    print(json.dumps(data, ensure_ascii=False, sort_keys=True))

if __name__ == "__main__":
    main()
