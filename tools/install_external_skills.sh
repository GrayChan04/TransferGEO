#!/usr/bin/env bash
set -euo pipefail

python - <<'PY2'
import json, os, subprocess, sys
from pathlib import Path

manifest = json.load(open('skills.manifest.json', encoding='utf-8'))
installer = Path(os.environ.get('CODEX_HOME', str(Path.home() / '.codex'))) / 'skills/.system/skill-installer/scripts/install-skill-from-github.py'
dest_root = Path(os.environ.get('CODEX_HOME', str(Path.home() / '.codex'))) / 'skills'

print('== Installing / reminding external skills ==')
for s in manifest.get('external_skills', []):
    name = s.get('name', '')
    method = s.get('install_method', 'skill_installer_git')
    enabled = s.get('enabled_by_default', True)
    if not enabled:
        print(f'[skip] {name}: enabled_by_default=false')
        continue
    if method == 'skill_installer_git':
        if not installer.exists():
            print(f'[manual] {name}: Codex skill installer not found: {installer}')
            print(f"         Repo: {s.get('repo')}  Path: {s.get('path')}  Ref: {s.get('ref','main')}")
            continue
        if (dest_root / name).exists():
            print(f'[skip] {name}: already installed')
            continue
        print(f"[install] {name} from {s.get('repo')}:{s.get('path')}")
        subprocess.run([sys.executable, str(installer), '--repo', s['repo'], '--ref', s.get('ref','main'), '--path', s.get('path','.'), '--name', name, '--method', 'auto'], check=False)
    elif method == 'codex_prompt':
        print(f'[prompt] {name}: install inside Codex with:')
        print(f"         {s.get('install_prompt')}")
    elif method == 'codex_plugin_marketplace':
        print(f'[plugin] {name}: install as Codex plugin:')
        print(f"         CLI: {s.get('codex_cli_install', '/plugins')}")
        print(f"         App: {s.get('codex_app_install', 'Plugins sidebar')}")
    else:
        print(f'[unknown] {name}: install_method={method}. Check skills.manifest.json')
print()        
print('After installation, restart Codex and run /skills or /plugins to verify availability.')
PY2
