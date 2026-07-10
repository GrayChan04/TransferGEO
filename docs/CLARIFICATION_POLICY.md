# Clarification Policy：指令不明确时必须追问

这个模板的目标不是让 Codex 盲目执行，而是让 Codex 成为可靠的科研协作者。

## 核心原则

```text
低风险可以假设，高风险必须追问。
```

## 高风险任务

以下任务缺少关键信息时，Codex 必须继续追问：

| 任务 | 为什么不能猜 |
|---|---|
| 自动跑实验 | 可能浪费算力、覆盖结果、记录错误日志 |
| 修改 config | 可能导致结果不可复现 |
| 改 dataset / baseline / metric | 会影响论文结论 |
| git commit / branch | 会影响版本历史 |
| public release | 可能泄露 private / PDF / reviewer 信息 |
| 投稿要求 | deadline、页数、匿名要求不能凭记忆 |
| 新增依赖 | 会污染环境或破坏复现 |

## 自动实验前必须确认

`$experiment-manager-cn` 在跑 managed experiment 前，必须确认：

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
是否创建 git branch
是否 git commit
失败时策略
```

如果用户只说“帮我跑个实验”，Codex 应该追问，而不是直接猜。

## 推荐追问格式

```text
我需要先确认几个关键信息，避免误跑实验或记录错误结果：

1. experiment_id 用什么？例如 pilot_001。
2. 运行命令是什么？例如 python src/experiments/run_eval.py --config ...
3. metrics.json 会输出到哪里？
4. 是否需要创建分支 exp/<experiment_id>？
5. 实验结束后是否 git commit？

你可以直接按编号回答。
```

## 允许假设的低风险情况

以下情况可以说明假设后继续：

- 生成空模板
- 整理 Markdown
- 创建目录
- 总结已有实验日志
- 给出命令示例
- 根据已有文件做非破坏性分析
