#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", "--decision", dest="event", required=True, help="要记录的事件")
    ap.add_argument("--result", default="", help="这个事件导致的结果")
    ap.add_argument("--next_action", default="", help="下一步行动")
    ap.add_argument("--details", default="", help="可选补充说明；默认不写长记录")
    args = ap.parse_args()

    now = datetime.now(timezone.utc).isoformat()
    Path("experiments/decision_log.md").open("a", encoding="utf-8").write(
        f"| {now} | {args.event} | {args.result} | {args.next_action} |\n"
    )

    if args.details:
        Path("private/dialogues").mkdir(parents=True, exist_ok=True)
        Path("private/dialogues", now[:10] + "_event.md").write_text(
            f"# 事件记录\n\n## 时间\n{now}\n\n## 事件\n{args.event}\n\n## 结果\n{args.result}\n\n## 下一步\n{args.next_action}\n\n## 补充说明\n{args.details}\n",
            encoding="utf-8"
        )
    print("已记录事件。")

if __name__ == "__main__":
    main()
