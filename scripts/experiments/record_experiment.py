#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, subprocess
from datetime import datetime, timezone
from pathlib import Path

FIELDS = ["experiment_id","start_time","end_time","duration","phase","claim","config","command","git_commit","dataset","method","seed","status","metrics_json","cost_json","result_path","notes_cn","next_action_cn"]

def git_commit():
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
    except Exception:
        return "unknown"

def main():
    ap=argparse.ArgumentParser()
    for f in FIELDS:
        if f != "git_commit":
            ap.add_argument(f"--{f}", default="")
    ap.add_argument("--log", default="experiments/experiment_log.csv")
    args=ap.parse_args()
    now=datetime.now(timezone.utc).isoformat()
    row={f:getattr(args,f,"") for f in FIELDS if f!="git_commit"}
    row["git_commit"]=git_commit()
    row["start_time"]=row["start_time"] or now
    row["end_time"]=row["end_time"] or now
    path=Path(args.log); path.parent.mkdir(parents=True, exist_ok=True)
    exists=path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=FIELDS)
        if not exists: w.writeheader()
        w.writerow({k:row.get(k,"") for k in FIELDS})
    exp=row["experiment_id"] or "unknown"
    Path("experiments/run_cards").mkdir(parents=True, exist_ok=True)
    Path(f"experiments/run_cards/{exp}.md").write_text(f"# 实验运行卡：{exp}\n\n## 命令\n\n```bash\n{row['command']}\n```\n\n## 结果\n\n{row['metrics_json']}\n\n## 中文解释\n\n{row['notes_cn']}\n\n## 下一步\n\n{row['next_action_cn']}\n", encoding="utf-8")
    with Path("experiments/commands.md").open("a", encoding="utf-8") as f:
        f.write(f"\n## {exp}\n\n```bash\n{row['command']}\n```\n")
    if row["status"].lower() in {"failed","error","fail"}:
        with Path("experiments/failures.md").open("a", encoding="utf-8") as f:
            f.write(f"\n| {now} | {exp} | {row['notes_cn']} | TBD | TBD | {row['next_action_cn']} |\n")
    with Path("private/10_下一步行动.md").open("a", encoding="utf-8") as f:
        f.write(f"\n- [{now}] 实验 {exp} 下一步：{row['next_action_cn']}\n")
    print(f"已记录实验：{exp}")

if __name__=="__main__":
    main()
