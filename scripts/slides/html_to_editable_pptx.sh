#!/usr/bin/env bash
set -euo pipefail
INPUT="${1:-slides/html/deck.html}"
OUTPUT="${2:-slides/pptx/deck.pptx}"
SELECTOR="${3:-.slide}"
node tools/slides/html_to_editable_pptx.mjs --input "$INPUT" --output "$OUTPUT" --selector "$SELECTOR"
