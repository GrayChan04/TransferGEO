#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

STATE = Path("private/state.yaml")

def read_stage() -> tuple[str, str]:
    if not STATE.exists():
        return "0", "模板/环境初始化"
    text = STATE.read_text(encoding="utf-8")
    stage_id = "0"
    stage_name = "模板/环境初始化"
    for line in text.splitlines():
        if line.startswith("stage_id:"):
            stage_id = line.split(":",1)[1].strip()
        if line.startswith("stage_name:"):
            stage_name = line.split(":",1)[1].strip()
    return stage_id, stage_name

def main() -> None:
    sid, name = read_stage()
    print(f"当前阶段：{sid}. {name}")

if __name__ == "__main__":
    main()
