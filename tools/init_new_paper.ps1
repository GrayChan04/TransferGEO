Write-Host "初始化论文项目..."
$dirs = @(
"private/dialogues","private/handoffs","private/literature","private/reviews",
"experiments/run_cards","experiments/env_snapshots",
"references/pdfs","references/bib/official","references/bib/arxiv","references/notes",
"venues/_TEMPLATE/paper_template",
"slides/source","slides/html","slides/pptx","slides/pdf","slides/screenshots",
"data/raw","data/processed","results/raw","results/processed"
)
foreach ($d in $dirs) { New-Item -ItemType Directory -Force -Path $d | Out-Null }

if ($env:SKIP_SKILLS -ne "1") {
  powershell -ExecutionPolicy Bypass -File tools/install_external_skills.ps1
}

python tools/template_doctor.py
python tools/check_project_stage.py

Write-Host "下一步在 Codex 中输入："
Write-Host "Use `$paper-orchestrator-cn to 初始化这篇论文项目。请先提醒我当前阶段，然后根据我的目标建议应该调用哪些 Skill。"
