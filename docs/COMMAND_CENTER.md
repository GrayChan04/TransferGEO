# Command Center：你不用记命令

你只需要记住一句 Codex 指令：

```text
Use $paper-orchestrator-cn to 接管导航。请告诉我当前阶段、该用哪个 Skill、要读写哪些文件、要运行哪些命令。
```

你也只需要记住一个命令：

```bash
python tools/project_brief.py
```

## 常见情况

### 我不知道现在该做什么

```text
Use $paper-orchestrator-cn to 接管导航。请根据当前项目状态告诉我下一步。
```

### 我要读论文

```text
Use $paper-orchestrator-cn to 帮我启动文献阅读流程，并告诉我该调用哪个 Skill、记录到哪些文件。
```

它应该推荐：

- `$academic-research-suite`
- `$reference-manager-cn`

### 我要设计方法

```text
Use $paper-orchestrator-cn to 帮我进入方法设计阶段，并告诉我应该如何用 grill-me 质询 idea。
```

它应该推荐：

- `$grill-me`
- `$session-recorder-cn`

### 我要跑实验

```text
Use $paper-orchestrator-cn to 帮我设计并记录下一个实验。
```

它应该推荐：

- `$experiment-manager-cn`
- `scripts/experiments/record_experiment.py`

### 我要做 PPT

```text
Use $paper-orchestrator-cn to 帮我生成组会 slides，并提醒我如何导出可编辑 PPTX。
```

它应该推荐：

- `$frontend-slides`
- `$visual-slides-cn`

### 我要投稿

```text
Use $paper-orchestrator-cn to 帮我进入投稿准备阶段，并检查会议要求、匿名风险和 claim evidence。
```

它应该推荐：

- `$venue-manager-cn`
- `scripts/anonymize/check_anonymity.py`
- `paper/claim_tracker.csv`

## 轻量记录命令

记录重要事件：

```bash
python scripts/records/record_decision.py --event "确定 benchmark" --result "选择 TBD" --next_action "准备 pilot"
```

记录一组实验的阶段性结论：

```bash
python scripts/records/record_stage_result.py --event "pilot 完成" --result "pipeline 跑通，但 metric 需要校准" --next_action "补 baseline"
```

记录重大写作版本：

```bash
python scripts/records/record_writing_version.py --event "paper draft v1" --result "完成 Method 和 Experiment 初稿" --next_action "补 Related Work"
```

## Managed Experiment 命令

一般情况下你不需要自己记命令，只要对 Codex 说：

```text
Use $experiment-manager-cn to 跑一个 managed experiment：<自然语言参数和结果路径>。结束后记录 experiment_log，并提交 git commit。
```

底层会使用：

```bash
python scripts/experiments/create_managed_experiment.py ...
python scripts/experiments/run_managed_experiment.py --experiment_id <experiment_id> --git_commit
```

## 指令不明确时

如果你只说：

```text
帮我跑个实验
```

Codex 应该追问，而不是直接跑。你可以这样补充：

```text
experiment_id 是 pilot_001；
dataset 是 data/processed/toy_geo_eval.jsonl；
method 是 baseline_rag_v0；
seed 是 42；
输出到 results/processed/pilot_001；
metrics 在 results/processed/pilot_001/metrics.json；
命令是 python src/experiments/run_eval.py --config configs/experiments/pilot_001.yaml；
需要创建分支并提交 git commit。
```

## 覆盖保护和 LLM cost

如果你要重复运行同一个实验，Codex 应该先问你是否允许覆盖旧结果。

允许覆盖时才使用：

```bash
--force
```

如果是 LLM 实验，可以让 Codex 记录 cost/token：

```text
请把 model、prompt_tokens、completion_tokens、estimated_cost_usd 记录到 cost_json。
```

## 最重要的自动提醒

Codex 应该在你不知道下一步、忘记命令、或者进入新论文阶段时主动提醒：

```bash
python tools/project_brief.py
```

以及：

```text
Use $paper-orchestrator-cn to 接管导航。请告诉我当前阶段、该用哪个 Skill、要读写哪些文件、要运行哪些命令。
```

## 第三方增强项

### Supervisor-Skills

```text
Use Supervisor-Skills to evaluate my research idea based on private/01_研究问题.md and private/03_方法卡.md. Focus on novelty, feasibility, risks, and CCF A potential.
```

### Superpowers

```text
Use Superpowers to implement src/experiments/run_eval.py based on configs/experiments/TEMPLATE.yaml. Keep it compatible with $experiment-manager-cn and scripts/experiments/run_managed_experiment.py.
```

## 第三方增强项

### Supervisor-Skills

```text
Use Supervisor-Skills to evaluate my research idea based on private/01_研究问题.md and private/03_方法卡.md. Focus on novelty, feasibility, risks, and CCF A potential.
```

### Superpowers

```text
Use Superpowers to implement src/experiments/run_eval.py based on configs/experiments/TEMPLATE.yaml. Keep it compatible with $experiment-manager-cn and scripts/experiments/run_managed_experiment.py.
```
