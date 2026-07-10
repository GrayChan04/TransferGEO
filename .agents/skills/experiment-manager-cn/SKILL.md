---
name: experiment-manager-cn
description: 中文实验设计落地、自动执行和记录。支持自然语言实验请求 → config → registry → run → metrics → experiment_log → git checkpoint。
---

# 中文 Managed Experiment Manager

## 定位

高层实验设计可以交给 `$academic-research-suite`，但项目内实验执行、记录和版本管理由本 Skill 接管。

用户只需要用自然语言说明实验参数和结果位置，你负责生成配置、运行命令、记录结果和做 git checkpoint。

## 推荐分工

- `$academic-research-suite`：设计实验应该做什么，例如 baselines、metrics、ablation、robustness、reproducibility。
- `$experiment-manager-cn`：把实验落地并执行，例如生成 config、写 registry、运行命令、记录结果、提交 git。

## 必须执行的步骤

1. 读取 `private/state.yaml`、`private/04_实验计划.md`、`experiments/registry.csv`。
2. 根据用户自然语言请求生成或更新 `configs/experiments/<experiment_id>.yaml`。
3. 写入或更新 `experiments/registry.csv`。
4. 如用户要求，创建实验分支 `exp/<experiment_id>`。
5. 运行环境快照：

```bash
python scripts/env/snapshot_environment.py
```

6. 执行实验：

```bash
python scripts/experiments/run_managed_experiment.py --experiment_id <experiment_id>
```

必要时传入 `--command`、`--metrics_path`、`--result_dir`、`--git_commit`。

7. 读取 `metrics_path`。
8. 自动写入 `experiments/experiment_log.csv`。
9. 如果失败，写入 `experiments/failures.md`。
10. 如果一组实验完成，再更新 `private/06_结果解读.md`。
11. 如用户要求，执行 git add / commit。
12. 输出下一步建议。

## 创建 managed experiment

```bash
python scripts/experiments/create_managed_experiment.py \
  --experiment_id pilot_001 \
  --phase pilot \
  --claim "验证 pipeline 是否跑通" \
  --dataset data/processed/toy_geo_eval.jsonl \
  --method baseline_rag_v0 \
  --seed 42 \
  --params_json '{"top_k":5,"model":"gpt-4.1-mini"}' \
  --output_dir results/processed/pilot_001 \
  --metrics_path results/processed/pilot_001/metrics.json \
  --command "python src/experiments/run_eval.py --config configs/experiments/pilot_001.yaml" \
  --notes_cn "pilot 实验"
```

## 执行 managed experiment

```bash
python scripts/experiments/run_managed_experiment.py \
  --experiment_id pilot_001 \
  --claim "验证 pipeline 是否跑通" \
  --dataset data/processed/toy_geo_eval.jsonl \
  --method baseline_rag_v0 \
  --seed 42 \
  --notes_cn "pipeline pilot" \
  --next_action_cn "检查 metrics 并补 baseline" \
  --create_branch \
  --git_commit
```

## 失败处理

如果实验失败，必须：

- 查看 `experiments/stderr/<experiment_id>.err`
- 写入 `experiments/failures.md`
- 给出下一步 debug 建议
- 不要更新 `private/06_结果解读.md`，除非这是一个实验组的阶段性结论

## 阶段性结果

单次实验只进 `experiments/experiment_log.csv`。

只有完成一组实验，例如 pilot/main/ablation/robustness/rebuttal，才更新：

```text
private/06_结果解读.md
```

## 自动跑实验前的澄清门

在生成 config 或运行实验之前，必须检查以下字段：

```text
experiment_id
phase
dataset
method
seed
params
output_dir
metrics_path
command
是否 create_branch
是否 git_commit
失败时策略
```

如果缺少 `command`、`dataset`、`output_dir` 或 `metrics_path`，不得猜测，必须追问。

如果缺少 `experiment_id`，可以建议一个 ID，例如 `pilot_001`，但必须让用户确认。

如果用户没有说明 git 策略，必须询问：

```text
是否需要创建实验分支 exp/<experiment_id>？
实验完成后是否需要 git commit？
```

推荐先运行：

```bash
python scripts/experiments/validate_managed_experiment.py --experiment_id <experiment_id>
```

如果 validation 报缺失字段，先向用户追问，不要运行实验。

## 覆盖保护

在运行 managed experiment 前，如果 `result_dir` 已存在且非空，默认必须停止并追问用户是否允许覆盖。

只有用户明确允许覆盖时，才可以加：

```bash
--force
```

不要替用户默认覆盖结果。

## LLM cost/token 记录

如果实验涉及 LLM API、LLM judge、生成式搜索调用或其他 token/cost 开销，需要尽量记录：

```text
cost_json
```

示例：

```json
{"model":"gpt-4.1-mini","prompt_tokens":120000,"completion_tokens":30000,"estimated_cost_usd":1.25}
```

传给 runner：

```bash
--cost_json '{"model":"gpt-4.1-mini","prompt_tokens":120000,"completion_tokens":30000,"estimated_cost_usd":1.25}'
```
