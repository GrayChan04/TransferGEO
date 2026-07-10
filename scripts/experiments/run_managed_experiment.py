#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, subprocess, shlex, time
from datetime import datetime, timezone
from pathlib import Path

REGISTRY = Path("experiments/registry.csv")

def now() -> str:
    return datetime.now(timezone.utc).isoformat()

def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"

def load_registry(exp_id: str) -> dict:
    if not REGISTRY.exists():
        return {}
    with REGISTRY.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("experiment_id") == exp_id:
                return row
    return {}

def update_registry(exp_id: str, updates: dict) -> None:
    rows = []
    fields = ["experiment_id","phase","config_path","command","result_dir","metrics_path","status","git_branch","git_commit","notes_cn"]
    if REGISTRY.exists():
        with REGISTRY.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
            fields = list(rows[0].keys()) if rows else fields
    found = False
    for row in rows:
        if row.get("experiment_id") == exp_id:
            row.update(updates)
            found = True
    if not found:
        row = {"experiment_id": exp_id}
        row.update(updates)
        rows.append(row)
    for k in updates:
        if k not in fields:
            fields.append(k)
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def read_metrics(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return "{}"
    try:
        return json.dumps(json.loads(p.read_text(encoding="utf-8")), ensure_ascii=False, sort_keys=True)
    except Exception:
        return json.dumps({"raw_metrics_text": p.read_text(encoding="utf-8", errors="ignore")[:4000]}, ensure_ascii=False)

def append_experiment_log(row: dict) -> None:
    path = Path("experiments/experiment_log.csv")
    fields = ["experiment_id","start_time","end_time","duration","phase","claim","config","command","git_commit","dataset","method","seed","status","metrics_json","cost_json","result_path","notes_cn","next_action_cn"]
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})

def write_run_card(exp_id: str, command: str, status: str, metrics_json: str, cost_json: str, stdout_path: str, stderr_path: str, next_action: str) -> None:
    p = Path("experiments/run_cards") / f"{exp_id}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# 实验运行卡：{exp_id}

## 状态

{status}

## 命令

```bash
{command}
```

## Metrics

```json
{metrics_json}
```

## LLM Cost / Token Metadata

```json
{cost_json}
```

## Stdout

{stdout_path}

## Stderr

{stderr_path}

## 下一步

{next_action}
"""
    p.write_text(content, encoding="utf-8")

def main() -> None:
    ap = argparse.ArgumentParser(description="Run a managed experiment and record logs/results.")
    ap.add_argument("--experiment_id", required=True)
    ap.add_argument("--command", default="")
    ap.add_argument("--config", default="")
    ap.add_argument("--phase", default="")
    ap.add_argument("--claim", default="")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--method", default="")
    ap.add_argument("--seed", default="")
    ap.add_argument("--result_dir", default="")
    ap.add_argument("--metrics_path", default="")
    ap.add_argument("--notes_cn", default="")
    ap.add_argument("--next_action_cn", default="")
    ap.add_argument("--cost_json", default="{}", help="JSON string for LLM model/token/cost metadata.")
    ap.add_argument("--shell", action="store_true", help="Run command via shell. Default uses shlex splitting.")
    ap.add_argument("--create_branch", action="store_true")
    ap.add_argument("--git_commit", action="store_true")
    ap.add_argument("--force", action="store_true", help="Allow running when result_dir is non-empty.")
    args = ap.parse_args()

    exp_id = args.experiment_id
    reg = load_registry(exp_id)

    command = args.command or reg.get("command", "")
    if not command:
        raise SystemExit("No command provided. Pass --command or create registry entry first.")

    config = args.config or reg.get("config_path", "")
    result_dir = args.result_dir or reg.get("result_dir", f"results/processed/{exp_id}")
    metrics_path = args.metrics_path or reg.get("metrics_path", f"{result_dir}/metrics.json")
    phase = args.phase or reg.get("phase", "")
    result_path_obj = Path(result_dir)
    if result_path_obj.exists() and any(result_path_obj.iterdir()) and not args.force:
        raise SystemExit(
            f"Result directory is not empty: {result_dir}\n"
            "Refusing to overwrite existing experiment outputs. "
            "Use --force only if the user explicitly allows overwrite."
        )
    result_path_obj.mkdir(parents=True, exist_ok=True)
    Path("experiments/stdout").mkdir(parents=True, exist_ok=True)
    Path("experiments/stderr").mkdir(parents=True, exist_ok=True)

    if args.create_branch:
        branch = reg.get("git_branch") or f"exp/{exp_id}"
        subprocess.run(["git", "checkout", "-B", branch], check=False)

    start = now()
    t0 = time.time()
    stdout_path = f"experiments/stdout/{exp_id}.out"
    stderr_path = f"experiments/stderr/{exp_id}.err"

    with open(stdout_path, "w", encoding="utf-8") as out, open(stderr_path, "w", encoding="utf-8") as err:
        if args.shell:
            proc = subprocess.run(command, shell=True, stdout=out, stderr=err, text=True)
        else:
            proc = subprocess.run(shlex.split(command), stdout=out, stderr=err, text=True)

    end = now()
    duration = f"{time.time() - t0:.2f}s"
    status = "completed" if proc.returncode == 0 else "failed"
    metrics_json = read_metrics(metrics_path)
    commit = git_commit()

    append_experiment_log({
        "experiment_id": exp_id,
        "start_time": start,
        "end_time": end,
        "duration": duration,
        "phase": phase,
        "claim": args.claim,
        "config": config,
        "command": command,
        "git_commit": commit,
        "dataset": args.dataset,
        "method": args.method,
        "seed": args.seed,
        "status": status,
        "metrics_json": metrics_json,
        "cost_json": args.cost_json,
        "result_path": result_dir,
        "notes_cn": args.notes_cn,
        "next_action_cn": args.next_action_cn,
    })

    update_registry(exp_id, {
        "status": status,
        "config_path": config,
        "command": command,
        "result_dir": result_dir,
        "metrics_path": metrics_path,
        "git_commit": commit,
    })

    write_run_card(exp_id, command, status, metrics_json, args.cost_json, stdout_path, stderr_path, args.next_action_cn)

    with Path("experiments/commands.md").open("a", encoding="utf-8") as f:
        f.write(f"\n## {exp_id}\n\n```bash\n{command}\n```\n")

    if status == "failed":
        with Path("experiments/failures.md").open("a", encoding="utf-8") as f:
            f.write(f"\n| {end} | {exp_id} | returncode={proc.returncode} | 查看 {stderr_path} | TBD | {args.next_action_cn} |\n")

    if args.git_commit:
        subprocess.run(["git", "add", "configs/experiments", "experiments", result_dir], check=False)
        subprocess.run(["git", "commit", "-m", f"exp: {exp_id}"], check=False)

    print(f"Experiment {exp_id}: {status}")
    print(f"Metrics: {metrics_json}")
    print(f"Stdout: {stdout_path}")
    print(f"Stderr: {stderr_path}")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

if __name__ == "__main__":
    main()
