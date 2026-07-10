#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import csv

def read_state():
    state = {"stage_id": "0", "stage_name": "模板/环境初始化", "current_goal": "初始化论文项目", "next_stage_name": "选题与研究问题定义"}
    p = Path("private/state.yaml")
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                state[k.strip()] = v.strip().strip('"')
    return state

def file_status(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return "缺少"
    if p.suffix not in {".md", ".csv", ".yaml", ".yml", ".tex", ".bib"}:
        return "存在"
    text = p.read_text(encoding="utf-8", errors="ignore")
    if "TBD" in text or len(text.strip()) < 40:
        return "待填写"
    return "已有内容"

def recent_experiments(limit=3):
    p = Path("experiments/experiment_log.csv")
    if not p.exists():
        return []
    try:
        rows = list(csv.DictReader(p.open(encoding="utf-8")))
        return rows[-limit:]
    except Exception:
        return []

def main():
    state = read_state()
    stage_id = state.get("stage_id", "0")
    stage_name = state.get("stage_name", "模板/环境初始化")
    goal = state.get("current_goal", "")
    next_stage = state.get("next_stage_name", "")

    print("=" * 72)
    print("AI Paper Factory：项目导航")
    print("=" * 72)
    print(f"当前阶段：{stage_id}. {stage_name}")
    print(f"当前目标：{goal}")
    print(f"下一阶段：{next_stage}")
    print()

    print("你不用记 Skill。现在最常用的是：")
    print("- paper-orchestrator-cn：接管导航，告诉你下一步")
    print("- academic-research-suite：论文阅读、综述、写作、review")
    print("- grill-me：质询 idea / 方法 / 实验设计")
    print("- experiment-manager-cn：实验设计与记录")
    print("- session-recorder-cn：记录重要决策")
<<<<<<< HEAD
    print("- Supervisor-Skills：科研副导师式 idea/投稿前审查（增强项）")
    print("- Superpowers：复杂代码实现、重构、TDD（增强插件）")
=======
>>>>>>> 1bd3e85eba289b200cbc1799c28eb5dd4f06b03f
    print()

    key_files = [
        "private/01_研究问题.md",
        "private/03_方法卡.md",
        "private/04_实验计划.md",
        "private/10_下一步行动.md",
        "experiments/experiment_log.csv",
        "paper/claim_tracker.csv",
        "paper/references.bib",
        "experiments/registry.csv",
        "venues/_TEMPLATE/raw_official_requirements.md",
    ]
    print("关键文件状态：")
    for f in key_files:
        print(f"- {f}: {file_status(f)}")
    print()

    recent = recent_experiments()
    if recent:
        print("最近实验：")
        for r in recent:
            print(f"- {r.get('experiment_id','')} | {r.get('status','')} | next: {r.get('next_action_cn','')}")
        print()

    print("建议你复制给 Codex 的一句话：")
    print()
    print("Use $paper-orchestrator-cn to 接管导航。请告诉我当前阶段、该用哪个 Skill、要读写哪些文件、要运行哪些命令。")
    print()

    print("常用命令：")
    print("- python tools/project_brief.py")
    print("- python tools/template_doctor.py")
    print("- python tools/check_project_stage.py")
    print("- python scripts/anonymize/check_anonymity.py")
    print("- bash scripts/slides/html_to_editable_pptx.sh slides/html/deck.html slides/pptx/deck.pptx .slide")
    print("=" * 72)

if __name__ == "__main__":
    main()
