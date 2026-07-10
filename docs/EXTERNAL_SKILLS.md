# External Skills and Plugins

本模板把第三方能力分成两类：

1. **核心第三方 Skill**：日常科研工作经常用，初始化时会尝试安装或提醒安装。
2. **增强第三方 Skill / 插件**：特定场景调用，不接管模板核心流程。

所有第三方来源都记录在：

```text
skills.manifest.json
skills.lock.md
```

安装/提醒入口：

```bash
bash tools/install_external_skills.sh
```

Windows：

```powershell
powershell -ExecutionPolicy Bypass -File tools/install_external_skills.ps1
```

## Supervisor-Skills

定位：科研副导师增强包。

适合场景：

- idea 评估
- Introduction / story line 梳理
- 技术类 / benchmark 类论文结构审查
- 投稿前 pre-submission review
- 图表设计建议

使用边界：

- 不接管项目导航；项目导航仍由 `$paper-orchestrator-cn` 负责。
- 不接管自动实验执行；实验执行仍由 `$experiment-manager-cn` 负责。
- 不替代 `paper/claim_tracker.csv`、`experiments/experiment_log.csv` 等本地记录。

推荐调用：

```text
Use Supervisor-Skills to evaluate my research idea based on private/01_研究问题.md and private/03_方法卡.md. Focus on novelty, feasibility, risks, and CCF A potential.
```

投稿前：

```text
Use Supervisor-Skills to perform a pre-submission review based on paper/main.tex, paper/claim_tracker.csv, and experiments/experiment_log.csv.
```

## Superpowers

定位：工程实现增强插件。

适合场景：

- 实现实验代码
- 重构实验 pipeline
- 写测试
- 使用 git worktree 做复杂开发
- 把实现任务拆解成计划并执行

使用边界：

- 不接管论文总控；总控仍由 `$paper-orchestrator-cn` 负责。
- 不接管实验日志；实验运行与记录仍由 `$experiment-manager-cn` 负责。
- 不默认修改 managed experiment 接口。

推荐调用：

```text
Use Superpowers to implement src/experiments/run_eval.py based on configs/experiments/TEMPLATE.yaml. Keep it compatible with $experiment-manager-cn and scripts/experiments/run_managed_experiment.py.
```

重构时：

```text
Use Superpowers to plan and execute a refactor of the experiment pipeline. Do not change configs/experiments/*.yaml, experiments/registry.csv, or scripts/experiments/run_managed_experiment.py without confirmation.
```

## 推荐分工

| 任务 | 推荐能力 |
|---|---|
| 不知道下一步 | `$paper-orchestrator-cn` |
| 文献、实验设计、论文写作 | `$academic-research-suite` |
| idea / 方法 / 实验质询 | `$grill-me` |
| 科研导师式评估、投稿前自查 | Supervisor-Skills |
| 自动跑实验、记录结果、git checkpoint | `$experiment-manager-cn` |
| 复杂代码实现、重构、TDD | Superpowers |
| 记录重要决策 | `$session-recorder-cn` |
| 可编辑 PPT | `$frontend-slides` + `$visual-slides-cn` |
