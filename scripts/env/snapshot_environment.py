#!/usr/bin/env python3
from __future__ import annotations

import json, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

def cmd(c):
    try: return subprocess.check_output(c, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e: return f"unavailable: {e}"

data={
  "time_utc": datetime.now(timezone.utc).isoformat(),
  "python": sys.version,
  "platform": platform.platform(),
  "git_commit": cmd(["git","rev-parse","HEAD"]),
  "pip_freeze": cmd([sys.executable,"-m","pip","freeze"]),
  "nvidia_smi": cmd(["nvidia-smi"])
}
out=Path("experiments/env_snapshots")/f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(out)
