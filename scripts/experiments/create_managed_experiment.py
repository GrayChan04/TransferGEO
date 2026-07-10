#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json
from pathlib import Path
from datetime import datetime, timezone

REGISTRY_FIELDS = ["experiment_id","phase","config_path","command","result_dir","metrics_path","status","git_branch","git_commit","notes_cn"]

def append_or_update_registry(row: dict, path: Path = Path("experiments/registry.csv")) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    updated = False
    for r in rows:
        if r.get("experiment_id") == row["experiment_id"]:
            r.update(row)
            updated = True
    if not updated:
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        w.writeheader()
        w.writerows(rows)

def dump_simple_yaml(data: dict, indent: int = 0) -> str:
    lines = []
    sp = " " * indent
    for k, v in data.items():
        if isinstance(v, dict):
            lines.append(f"{sp}{k}:")
            lines.append(dump_simple_yaml(v, indent + 2))
        else:
            if isinstance(v, str):
                if any(c in v for c in [":", "#", "{", "}", "[", "]"]) or v.strip() != v:
                    v = json.dumps(v, ensure_ascii=False)
            lines.append(f"{sp}{k}: {v}")
    return "\n".join(lines)

def main() -> None:
    ap = argparse.ArgumentParser(description="Create a managed experiment config and registry entry.")
    ap.add_argument("--experiment_id", required=True)
    ap.add_argument("--phase", default="pilot")
    ap.add_argument("--claim", default="")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--method", default="")
    ap.add_argument("--seed", default="")
    ap.add_argument("--params_json", default="{}", help='JSON object, e.g. {"top_k":5,"model":"gpt-4.1-mini"}')
    ap.add_argument("--output_dir", default="")
    ap.add_argument("--metrics_path", default="")
    ap.add_argument("--command", required=True, help="Command to run this experiment.")
    ap.add_argument("--notes_cn", default="")
    ap.add_argument("--git_branch", default="")
    args = ap.parse_args()

    exp_id = args.experiment_id
    config_path = Path("configs/experiments") / f"{exp_id}.yaml"
    output_dir = args.output_dir or f"results/processed/{exp_id}"
    metrics_path = args.metrics_path or f"{output_dir}/metrics.json"
    params = json.loads(args.params_json or "{}")

    config = {
        "experiment_id": exp_id,
        "phase": args.phase,
        "claim": args.claim,
        "dataset": args.dataset,
        "method": args.method,
        "seed": args.seed,
        "params": params,
        "output_dir": output_dir,
        "metrics_path": metrics_path,
        "command": args.command,
        "notes_cn": args.notes_cn,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(dump_simple_yaml(config) + "\n", encoding="utf-8")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    row = {
        "experiment_id": exp_id,
        "phase": args.phase,
        "config_path": str(config_path),
        "command": args.command,
        "result_dir": output_dir,
        "metrics_path": metrics_path,
        "status": "planned",
        "git_branch": args.git_branch or f"exp/{exp_id}",
        "git_commit": "",
        "notes_cn": args.notes_cn,
    }
    append_or_update_registry(row)
    print(f"Created managed experiment: {exp_id}")
    print(f"Config: {config_path}")
    print(f"Result dir: {output_dir}")
    print(f"Metrics path: {metrics_path}")

if __name__ == "__main__":
    main()
