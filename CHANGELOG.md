# Changelog

## v1

- BibTeX 流程简化：统一人工复制到 `paper/references.bib`。
- `paper/main.tex` 改为读取 `references.bib`。
- 新增 `experiments/registry.csv`。
- 新增 `configs/experiments/TEMPLATE.yaml`。
- 新增 `scripts/experiments/create_managed_experiment.py`。
- 新增 `scripts/experiments/run_managed_experiment.py`。
- 新增 `scripts/experiments/collect_metrics.py`。
- 新增 `scripts/experiments/git_checkpoint.py`。
- 更新 `experiment-manager-cn`，支持自然语言实验请求落地、运行、记录和版本管理。

## v1

- 新增 `docs/CLARIFICATION_POLICY.md`。
- 新增 `scripts/experiments/validate_managed_experiment.py`。
- 更新 `AGENTS.md`：高影响任务缺少关键参数时必须追问。
- 更新 `$paper-orchestrator-cn`：项目导航时先做“缺失信息检查”。
- 更新 `$experiment-manager-cn`：自动跑实验前必须检查 experiment_id / command / dataset / method / seed / output_dir / metrics_path / git 策略等关键字段。

## v1

- 给 managed experiment 增加 result_dir 覆盖保护：默认不覆盖非空结果目录，除非显式 `--force`。
- 给实验日志增加 `cost_json` 字段，用于记录 LLM model / token / cost 信息。
- 保持不添加 dataset fingerprint，避免模板变重。
- 再次检查并删除不必要的 BibTeX 自动合并逻辑和冗余引用记录。

## v1

- 将两个最重要入口写入 Codex 自动提醒：
  - `python tools/project_brief.py`
  - `Use $paper-orchestrator-cn to 接管导航。请告诉我当前阶段、该用哪个 Skill、要读写哪些文件、要运行哪些命令。`
- 新增 `docs/AUTO_REMINDER.md`。
- 更新 `AGENTS.md`、`paper-orchestrator-cn`、`README.md` 和 `docs/COMMAND_CENTER.md`。
