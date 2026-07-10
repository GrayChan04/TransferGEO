#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True, help="例如 paper draft v1 / submission draft / rebuttal revision / camera-ready")
    ap.add_argument("--result", required=True, help="该版本完成了什么")
    ap.add_argument("--next_action", default="", help="下一步")
    args = ap.parse_args()

    now = datetime.now(timezone.utc).isoformat()
    Path("private/12_重大写作版本记录.md").open("a", encoding="utf-8").write(
        f"| {now} | {args.event} | {args.result} | {args.next_action} |\n"
    )
    print("已记录重大写作版本。")

if __name__ == "__main__":
    main()
