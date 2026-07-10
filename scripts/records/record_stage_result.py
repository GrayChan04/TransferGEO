#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True, help="实验组或阶段事件，例如 pilot 完成")
    ap.add_argument("--result", required=True, help="阶段性结论")
    ap.add_argument("--next_action", default="", help="下一步行动")
    args = ap.parse_args()

    now = datetime.now(timezone.utc).isoformat()
    Path("private/06_结果解读.md").open("a", encoding="utf-8").write(
        f"| {now} | {args.event} | {args.result} | {args.next_action} |\n"
    )
    print("已记录阶段性结果解读。")

if __name__ == "__main__":
    main()
