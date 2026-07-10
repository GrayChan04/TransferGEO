#!/usr/bin/env bash
set -euo pipefail
cd tools/slides
npm install
npx playwright install chromium
echo "Slides 工具安装完成。"
