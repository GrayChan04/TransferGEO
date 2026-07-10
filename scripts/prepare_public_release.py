#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import shutil

EXCLUDE = ["private","experiments",".agents","references/pdfs","data/raw","data/private","results/raw","results/cache"]

def excluded(rel):
    s=rel.as_posix()
    return any(s==e or s.startswith(e+"/") for e in EXCLUDE)

dst=Path("public_release/package")
if dst.exists(): shutil.rmtree(dst)
dst.mkdir(parents=True)
for p in Path(".").rglob("*"):
    r=p.relative_to(".")
    if r.as_posix().startswith("public_release/package") or excluded(r): continue
    t=dst/r
    if p.is_dir(): t.mkdir(parents=True, exist_ok=True)
    elif p.is_file(): t.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p,t)
print("公开发布包已生成，请运行 scripts/anonymize/check_anonymity.py 并人工检查。")
