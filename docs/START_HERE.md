# 你不用记命令

最重要的一句：

```text
Use $paper-orchestrator-cn to 接管导航。请告诉我当前阶段、该用哪个 Skill、要读写哪些文件、要运行哪些命令。
```

最重要的命令：

```bash
python tools/project_brief.py
```

---

# START HERE

## 你第一次使用只需要做这些

```bash
bash tools/init_new_paper.sh
python tools/template_doctor.py
```

然后启动 Codex：

```text
Use $paper-orchestrator-cn to 初始化这篇论文项目。请先提醒我当前阶段，并根据我的目标建议应该调用哪些 Skill。
```

## 常用命令

检查阶段：

```bash
python tools/check_project_stage.py
```

安装外部 Skill：

```bash
bash tools/install_external_skills.sh
```

记录实验：

```bash
python scripts/experiments/record_experiment.py --experiment_id pilot_001 --status planned --notes_cn "计划 pilot 实验"
```

记录重要决策：

```bash
python scripts/records/record_decision.py --decision "确定目标会议为 TBD" --reason "与研究方向匹配"
```

HTML 生成可编辑 PPTX：

```bash
bash scripts/slides/html_to_editable_pptx.sh slides/html/deck.html slides/pptx/deck.pptx .slide
```

投稿前匿名检查：

```bash
python scripts/anonymize/check_anonymity.py
```
