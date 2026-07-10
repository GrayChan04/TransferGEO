#!/usr/bin/env bash
set -euo pipefail

echo "初始化论文项目..."
mkdir -p private/dialogues private/handoffs private/literature private/reviews
mkdir -p experiments/run_cards experiments/env_snapshots
mkdir -p references/pdfs references/bib/official references/bib/arxiv references/notes
mkdir -p venues/_TEMPLATE/paper_template
mkdir -p slides/source slides/html slides/pptx slides/pdf slides/screenshots
mkdir -p data/raw data/processed results/raw results/processed

if [ "${SKIP_SKILLS:-0}" != "1" ]; then
  bash tools/install_external_skills.sh || true
fi

python tools/template_doctor.py || true
python tools/check_project_stage.py || true
python tools/project_brief.py || true

cat <<'EOF'

下一步在 Codex 中输入：

Use $paper-orchestrator-cn to 初始化这篇论文项目。请先提醒我当前阶段，然后根据我的目标建议应该调用哪些 Skill。

EOF
