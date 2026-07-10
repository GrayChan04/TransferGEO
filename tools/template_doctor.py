#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

errors = []
warnings = []

def exists(path: str):
    if not Path(path).exists():
        errors.append(f"缺少：{path}")

for p in [
    "AGENTS.md", "README.md", "skills.manifest.json", "private/state.yaml",
    "experiments/experiment_log.csv", "paper/claim_tracker.csv",
    "configs/experiments/TEMPLATE.yaml",
    "scripts/experiments/run_managed_experiment.py",
    "scripts/experiments/validate_managed_experiment.py",
    "docs/CLARIFICATION_POLICY.md",
    "docs/EXPERIMENT_SAFETY.md",
    "docs/AUTO_REMINDER.md",
    "docs/FINAL_V1_SCOPE.md",
<<<<<<< HEAD
    "docs/EXTERNAL_SKILLS.md",
=======
>>>>>>> 1bd3e85eba289b200cbc1799c28eb5dd4f06b03f
    "scripts/experiments/create_managed_experiment.py",
    "experiments/registry.csv",
    "paper/references.bib",
    "private/12_重大写作版本记录.md",
    "scripts/slides/html_to_editable_pptx.sh",
    "tools/install_external_skills.sh",
    "tools/project_brief.py",
    "docs/COMMAND_CENTER.md"
]:
    exists(p)

try:
    json.load(open("skills.manifest.json", encoding="utf-8"))
except Exception as e:
    errors.append(f"skills.manifest.json 无法解析：{e}")

for skill_dir in Path(".agents/skills").glob("*"):
    if skill_dir.is_dir() and not (skill_dir / "SKILL.md").exists():
        errors.append(f"Skill 缺少 SKILL.md：{skill_dir}")


for deleted in ["private/09_GitHub管理记录.md", "private/11_引用与证据清单.md", "private/12_写作修改记录.md"]:
    if Path(deleted).exists():
        errors.append(f"应删除但仍存在：{deleted}")


for deleted in ["paper/refs.bib", "references/bib/official", "references/bib/arxiv", "scripts/references/merge_bibtex_to_paper.py"]:
    if Path(deleted).exists():
        errors.append(f"应删除但仍存在：{deleted}")


elog = Path("experiments/experiment_log.csv")
if elog.exists():
    header = elog.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    if "cost_json" not in header:
        errors.append("experiments/experiment_log.csv 缺少 cost_json 字段")

gitignore = Path(".gitignore").read_text(encoding="utf-8") if Path(".gitignore").exists() else ""
if "references/pdfs/*.pdf" not in gitignore:
    warnings.append("建议默认忽略 references/pdfs/*.pdf")

bad_versions = ["v" + x for x in ["3.1", "3.2", "3.3", "3.4"]]
for p in Path(".").rglob("*"):
    if p.is_file() and p.as_posix() != "tools/template_doctor.py" and p.suffix in {".md",".txt",".py",".sh",".ps1",".json",".yaml",".yml"}:
        if any(part in {".git","node_modules"} for part in p.parts):
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for v in bad_versions:
            if v in text:
                warnings.append(f"发现历史版本字符串 {v}: {p}")

<<<<<<< HEAD

try:
    manifest = json.load(open("skills.manifest.json", encoding="utf-8"))
    names = {s.get("name") for s in manifest.get("external_skills", [])}
    for expected_skill in ["supervisor-skills", "superpowers"]:
        if expected_skill not in names:
            errors.append(f"skills.manifest.json 缺少第三方增强项：{expected_skill}")
except Exception as e:
    errors.append(f"无法检查 skills.manifest.json 增强项：{e}")

=======
>>>>>>> 1bd3e85eba289b200cbc1799c28eb5dd4f06b03f
print("Template doctor")
print("Errors:", len(errors))
for e in errors:
    print("  ERROR:", e)
print("Warnings:", len(warnings))
for w in warnings[:20]:
    print("  WARN:", w)

raise SystemExit(1 if errors else 0)
