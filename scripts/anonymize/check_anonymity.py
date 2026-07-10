#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re

patterns = [
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("github_url", re.compile(r"github\.com/[A-Za-z0-9_.-]+")),
    ("absolute_path", re.compile(r"(/Users/|/home/|C:\\Users\\)")),
    ("acknowledgement", re.compile(r"acknowledg(e)?ment", re.I)),
]
targets = ["paper", "README.md", "public_release"]
findings=[]
for t in targets:
    p=Path(t)
    paths=[p] if p.is_file() else list(p.rglob("*")) if p.exists() else []
    for f in paths:
        if f.is_file() and f.suffix.lower() in {".tex",".md",".bib",".txt"}:
            text=f.read_text(encoding="utf-8", errors="ignore")
            for name,pat in patterns:
                for m in pat.finditer(text):
                    findings.append((name,str(f),m.group(0)[:120]))
if not findings:
    print("未发现明显匿名风险。")
else:
    print("发现可能匿名风险：")
    for x in findings:
        print(" -", x)
    raise SystemExit(1)
