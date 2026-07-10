# AI Paper Factory Template v1

这是一个轻量、可复制、面向 AI 科研论文产出的 GitHub Template Repository。

目标：以后每一篇新论文都从这个模板创建新仓库，然后根据具体 idea、benchmark、实验环境和目标会议扩展。

## 设计原则

1. 模板只保留“论文产出基础设施”，不预装具体 AI 实验库。
2. 通用学术能力交给第三方 Skill：
   - academic-research-suite
   - handoff
   - grill-me
   - excalidraw
   - frontend-slides
3. 本地 Skill 只负责中文记录、阶段管理、实验记录、会议要求、参考文献、依赖和 slides 产物。
4. 除论文正文和中稿后公开材料外，所有内部讨论和科研记录默认中文。
5. 默认生成可编辑 PPTX，而不是整页截图式 PPTX。

## 第一次使用

```bash
bash tools/init_new_paper.sh
```

然后启动 Codex：

```text
Use $paper-orchestrator-cn to 初始化这篇论文项目。请先提醒我当前论文阶段，然后根据我的目标建议应该调用哪些 Skill。
```

## 目录

```text
.agents/skills/     本地中文项目 Skill
private/            中文内部科研记录，不公开
experiments/        实验日志、运行卡、失败记录
references/         PDF、BibTeX、阅读笔记、文献索引
venues/             会议官网要求和模板
paper/              英文 LaTeX 论文
slides/             HTML 和可编辑 PPTX
assets/             图表和公开素材
tools/              初始化、自检、依赖、Skill 安装工具
scripts/            实验、文献、匿名检查、slides 转换脚本
requirements/       按需安装的 Python 依赖分组
public_release/     中稿后公开发布包
```

## 你不用记命令

每次不知道下一步时，只要运行：

```bash
python tools/project_brief.py
```

或者在 Codex 中说：

```text
Use $paper-orchestrator-cn to 接管导航。请告诉我当前阶段、该用哪个 Skill、要读写哪些文件、要运行哪些命令。
```

Codex 应该主动提醒当前阶段、推荐 Skill、列出关键文件和可复制命令。

## 轻量记录原则

这个版本采用事件级记录，不记录普通过程。

默认只记录：

```text
时间 | 事件 | 结果 | 下一步
```

普通单次实验只进入：

```text
experiments/experiment_log.csv
```

只有 pilot / main results / ablation 等一组实验完成后，才写：

```text
private/06_结果解读.md
```

论文写作只记录重大版本：

```text
private/12_重大写作版本记录.md
```

Claim 和证据统一使用：

```text
paper/claim_tracker.csv
```

## BibTeX 简化规则

你只需要把所有文章的 BibTeX 复制粘贴到：

```text
paper/references.bib
```

模板不再自动抓取、拆分或合并 BibTeX。

## Managed Experiment：自然语言自动跑实验

理想使用方式：

```text
Use $experiment-manager-cn to 跑一个 pilot 实验：
- experiment_id: pilot_001
- top_k: 5
- model: gpt-4.1-mini
- dataset: data/processed/toy_geo_eval.jsonl
- seed: 42
- 输出到 results/processed/pilot_001/
- metrics 在 results/processed/pilot_001/metrics.json
- 结束后自动记录 experiment_log，并提交 git commit。
```

Codex 应该自动生成 config、写 registry、运行实验、记录结果、失败归档、必要时 git commit。

## 指令不明确时，Codex 必须追问

这个模板不鼓励 Codex 盲目猜测。

尤其是自动跑实验时，如果缺少这些信息，Codex 必须先追问：

```text
experiment_id
dataset
method
seed
params
output_dir
metrics_path
command
是否创建 git branch
是否 git commit
```

你可以只用自然语言说目标，但如果信息不够，Codex 应该继续问你，而不是乱跑。

## 自动导航提醒

如果你忘了命令、文件或 Skill，Codex 应该主动提醒你这两个入口：

```bash
python tools/project_brief.py
```

```text
Use $paper-orchestrator-cn to 接管导航。请告诉我当前阶段、该用哪个 Skill、要读写哪些文件、要运行哪些命令。
```

这个规则已经写入 `AGENTS.md`、`paper-orchestrator-cn` 和 `docs/AUTO_REMINDER.md`。
<<<<<<< HEAD

## 第三方增强项：Supervisor-Skills 和 Superpowers

这两个已经纳入 `skills.manifest.json` 和 `skills.lock.md`，并由：

```bash
bash tools/install_external_skills.sh
```

统一提醒安装。

- **Supervisor-Skills**：科研副导师增强，适合 idea 评估、论文结构、投稿前审查、图表建议。
- **Superpowers**：工程实现增强，适合复杂代码实现、重构、TDD、git worktree、计划执行。

核心流程不变：

```text
项目导航：$paper-orchestrator-cn
自动实验：$experiment-manager-cn
实验设计：$academic-research-suite
导师式审查：Supervisor-Skills
复杂工程实现：Superpowers
```

## 第三方增强项：Supervisor-Skills 和 Superpowers

这两个已纳入 `skills.manifest.json` 和 `skills.lock.md`，并由以下命令统一提醒安装：

```bash
bash tools/install_external_skills.sh
```

- **Supervisor-Skills**：科研副导师增强，适合 idea 评估、论文结构、投稿前审查、图表建议。
- **Superpowers**：工程实现增强，适合复杂代码实现、重构、TDD、git worktree、计划执行。

核心流程不变：

```text
项目导航：$paper-orchestrator-cn
自动实验：$experiment-manager-cn
实验设计：$academic-research-suite
导师式审查：Supervisor-Skills
复杂工程实现：Superpowers
```
=======
>>>>>>> 1bd3e85eba289b200cbc1799c28eb5dd4f06b03f
