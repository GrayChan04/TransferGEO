#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv
from pathlib import Path

FIELDS="paper_key,title,authors,year,venue,official_bib_path,arxiv_bib_path,pdf_path,note_path,tags,used_in_paper,claim_supported".split(",")

def main():
    ap=argparse.ArgumentParser()
    for f in FIELDS: ap.add_argument(f"--{f}", default="")
    ap.add_argument("--metadata", default="references/metadata.csv")
    a=ap.parse_args()
    p=Path(a.metadata); exists=p.exists(); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=FIELDS)
        if not exists: w.writeheader()
        w.writerow({k:getattr(a,k,"") for k in FIELDS})
    print("已添加文献：", a.paper_key)
if __name__=="__main__": main()
