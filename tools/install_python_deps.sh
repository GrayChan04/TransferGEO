#!/usr/bin/env bash
set -euo pipefail
GROUP="${1:-base}"
REQ="requirements/${GROUP}.txt"
[ -f "$REQ" ] || { echo "缺少 $REQ"; exit 1; }
python -m pip install --upgrade pip
if grep -vE '^\s*(#|$)' "$REQ" >/dev/null; then
  python -m pip install -r "$REQ"
else
  echo "$REQ 没有实际依赖，跳过。"
fi
