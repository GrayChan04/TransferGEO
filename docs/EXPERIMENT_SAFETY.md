# Experiment Safety

## 覆盖保护

Managed experiment 默认不会覆盖非空结果目录。

如果 `result_dir` 已存在且里面有文件，`run_managed_experiment.py` 会停止，并提示用户确认。

只有用户明确允许覆盖时，才可以使用：

```bash
--force
```

不要替用户默认加 `--force`。

## LLM cost/token 记录

LLM 相关实验可以用 `cost_json` 记录模型、token 和费用估算：

```json
{
  "model": "gpt-4.1-mini",
  "prompt_tokens": 120000,
  "completion_tokens": 30000,
  "estimated_cost_usd": 1.25
}
```

使用：

```bash
python scripts/experiments/run_managed_experiment.py \
  --experiment_id pilot_001 \
  --cost_json '{"model":"gpt-4.1-mini","prompt_tokens":120000,"completion_tokens":30000,"estimated_cost_usd":1.25}'
```

## 不记录 dataset fingerprint

当前模板不默认记录 dataset fingerprint。原因是保持轻量；如果某个项目确实需要数据版本追踪，可以在该论文项目内单独添加。
