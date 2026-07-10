# AGENTS.md：AI Paper Factory v1

你在本仓库中协助用户产出、修改、投稿、返修、重投和中稿后公开 AI 科研论文。

## 语言规则

除以下材料外，所有与用户的交互、科研讨论、实验记录、失败分析、方向调整、review 解读、内部决策都必须使用中文：

- `paper/` 中的正式论文正文
- 中稿后准备公开的代码、图表、README、supplementary
- 会议要求必须英文提交的材料

## Skill 路由规则

用户不需要记住所有 Skill。你必须根据任务主动建议 1–3 个最合适的 Skill：

- 论文阅读、综述、写作、学术 review：优先 `$academic-research-suite`
- idea / 方法 / 实验设计质询：优先 `$grill-me`
- 长会话交接：优先 `$handoff` + `$session-recorder-cn`
- 方法图、流程图：优先 `$excalidraw` + `$visual-slides-cn`
- HTML slides：优先 `$frontend-slides` + `$visual-slides-cn`
- 项目初始化、阶段判断、GitHub 论文管理：优先 `$paper-orchestrator-cn`
- 实验记录：优先 `$experiment-manager-cn`
- 会议要求：优先 `$venue-manager-cn`
- 参考文献和 BibTeX：优先 `$reference-manager-cn`
- 依赖：优先 `$dependency-manager-cn`

## 阶段提醒规则

在初始化、实验计划、实验总结、写作计划、投稿准备、rebuttal、public release 相关任务开始时，回复开头必须提醒：

```text
当前阶段：<阶段编号与名称>
当前目标：<本轮任务目标>
下一阶段：<建议推进方向>
```

普通短问答不需要机械提醒。

## 强制记录规则

当用户和 Codex 达成以下重要决策时，必须主动建议调用 `$session-recorder-cn`：

- 研究方向变化
- 研究问题变化
- 方法设计变化
- benchmark、baseline、数据集或指标选择
- 重要 debug 结论
- 实验结果解释和下一步行动
- 投稿会议选择
- rebuttal / revision / resubmission 策略

如果用户没有拒绝记录，应写入：

- `private/dialogues/`
- `experiments/decision_log.md`
- `private/state.yaml`
- `private/00_项目状态.md`
- `private/10_下一步行动.md`

## 实验记录规则

每次实验必须记录：

- 时间
- 实验 ID
- 对应 paper claim
- config
- command
- git commit
- dataset / version
- method / version
- seed
- status
- metrics_json
- result_path
- 中文解释
- 下一步行动建议

优先使用：

```bash
python scripts/experiments/record_experiment.py ...
```

## Slides / PPTX 规则

用户要求 PPTX 文字可编辑。默认禁止把整页 HTML 截图塞进 PPTX 作为最终产物。

默认流程：

```text
HTML → DOM / computed style / bounding box → editable PPTX text boxes / shapes / images / tables
```

默认命令：

```bash
bash scripts/slides/html_to_editable_pptx.sh slides/html/deck.html slides/pptx/deck.pptx .slide
```

只有用户明确接受“高保真但不可编辑”时，才允许使用截图式 fallback。

## 依赖规则

不要把 torch、transformers、faiss、ragas、pytest、ruff 等大依赖预装到模板。新增依赖必须：

1. 说明原因；
2. 写入 `requirements/<group>.txt`；
3. 使用 `tools/install_python_deps.*` 安装；
4. 记录到 `private/13_依赖与环境记录.md`。

## 公开发布规则

中稿前仓库默认私有。公开发布前必须检查并排除：

- `private/`
- `experiments/`
- `references/pdfs/`
- reviewer comments
- 未公开数据
- API key
- 大型原始结果
- 作者名、邮箱、学校、个人 GitHub 链接等匿名信息

## 主动导航模式

用户不需要记住命令、文件路径或 Skill 名称。每当用户提出任何与论文项目有关的任务时，你必须主动提供“导航块”。

导航块格式：

```text
当前阶段：<stage_id>. <stage_name>
我建议使用：<1-3 个 Skill>
本次会读写：<关键文件路径>
建议命令：<如需要运行的命令>
下一步：<一条最具体行动>
```

以下情况必须输出导航块：

- 初始化论文项目
- 选题、文献、方法、实验、写作、投稿、rebuttal、公开发布相关任务
- 用户说“不知道下一步”“我忘了命令”“该用哪个 Skill”“帮我继续”
- 任何涉及文件修改、实验记录、会议要求、参考文献、slides、GitHub 的任务

普通知识问答或短翻译可以不输出导航块。

如果用户说“我记不住命令/文件/Skill”，你应该告诉用户只需要记住两件事：

```bash
python tools/project_brief.py
```

以及在 Codex 中说：

```text
Use $paper-orchestrator-cn to 接管导航。请告诉我当前阶段、该用哪个 Skill、要读写哪些文件、要运行哪些命令。
```

如果任务需要命令，直接给出可复制命令，不要求用户自己翻文档。

## 轻量记录规则

不要把普通讨论、小修改、临时想法都写入文件。只记录对论文产出有长期价值的事件。

必须记录的事件：

1. 研究问题确定或改变
2. 方法设计确定或改变
3. benchmark / baseline / metric 选择
4. 跑了一次实验
5. 一组实验完成并产生阶段性结论
6. 选择目标会议
7. 收到 review
8. 重大 rebuttal / resubmission 策略
9. 新增重要依赖
10. 准备公开发布

不需要记录：

- 普通聊天
- 小的措辞讨论
- 临时想法
- 轻微 debug 尝试
- 小代码整理
- 无结果的搜索
- 重复说明

除 `experiments/experiment_log.csv` 这类结构化实验日志外，普通记录文件优先使用四列：

```text
时间 | 事件 | 结果 | 下一步
```

单次实验只写入：

```text
experiments/experiment_log.csv
```

只有完成一组实验，例如 pilot、main results、ablation、robustness、rebuttal experiments，才更新：

```text
private/06_结果解读.md
```

论文写作只记录重大版本：

- paper draft v1
- submission draft
- rebuttal revision
- resubmission draft
- camera-ready

对应文件：

```text
private/12_重大写作版本记录.md
```

Claim 和证据统一使用：

```text
paper/claim_tracker.csv
```

GitHub 工作流说明统一放在：

```text
docs/03_GitHub工作流.md
```

具体变化由 Git commit / branch / tag 记录。

## Managed Experiment 规则

用户希望用自然语言描述实验参数、结果路径和版本管理要求。你不能要求用户手动修改多个文件。

当用户提出“跑实验 / 自动跑实验 / 记录结果 / 提交 git”的任务时，必须使用 `$experiment-manager-cn` 的 managed experiment 流程：

1. 解析自然语言实验请求；
2. 生成或更新 `configs/experiments/<experiment_id>.yaml`；
3. 写入 `experiments/registry.csv`；
4. 运行 `scripts/env/snapshot_environment.py`；
5. 运行 `scripts/experiments/run_managed_experiment.py`；
6. 读取 metrics；
7. 自动写入 `experiments/experiment_log.csv`；
8. 失败时写入 `experiments/failures.md`；
9. 用户要求时创建 branch 和 git commit；
10. 输出下一步建议。

高层实验设计优先用 `$academic-research-suite`，实验落地执行优先用 `$experiment-manager-cn`。

## BibTeX 简化规则

不要自动抓取、拆分或合并 BibTeX。用户把所有 BibTeX 直接粘贴到：

```text
paper/references.bib
```

论文正文使用：

```tex
\bibliography{references}
```

## 澄清优先规则

如果用户指令不明确，你不能为了显得高效而猜测高影响参数。你必须先判断任务风险。

### 必须追问的高影响任务

以下任务如果缺少关键信息，必须继续追问：

- 自动跑实验、改实验参数、改 config、写结果、提交 git
- 选择或修改 benchmark / baseline / metric / dataset
- 新增依赖、改环境、安装大包
- 投稿会议、匿名检查、public release
- 删除文件、覆盖结果、移动数据
- 修改论文核心 claim、实验结论、rebuttal 策略

### 低风险任务可以说明假设后继续

以下任务可以在说明假设后继续：

- 调整文档格式
- 生成模板文件
- 改 README wording
- 创建空目录
- 总结已有内容
- 轻微命名整理

### 追问格式

追问时不要一次问太多。最多问 5 个问题，并按优先级列出。格式：

```text
我需要先确认几个关键信息，避免误跑实验/误改文件：

1. ...
2. ...
3. ...

你可以直接按编号回答。
```

### Managed Experiment 的强制澄清字段

自动跑实验前，如果以下字段缺失或矛盾，必须追问：

- `experiment_id`
- `phase`
- `dataset`
- `method`
- `seed`
- `params`
- `output_dir`
- `metrics_path`
- `command`
- 是否创建 git branch
- 是否 git commit
- 失败时是否停止或继续后续实验

不要猜 `command`、`metrics_path`、`output_dir`、`dataset`。

## 实验覆盖保护与 LLM cost 记录

Managed experiment 默认不得覆盖非空 `result_dir`。如果结果目录已存在且不为空，Codex 必须停止并追问用户是否允许覆盖。

只有用户明确说“允许覆盖 / force / 覆盖旧结果”时，才可以使用：

```bash
--force
```

LLM 实验如果涉及 API 模型、token 或费用，必须尽量记录到：

```text
cost_json
```

示例：

```json
{"model":"gpt-4.1-mini","prompt_tokens":120000,"completion_tokens":30000,"estimated_cost_usd":1.25}
```

不要默认添加 dataset fingerprint；用户明确要求或项目需要时再补。

## 自动导航提醒

用户不需要记住命令、文件或 Skill。每当出现以下情况，你必须主动提醒这两个入口：

- 用户说“不知道下一步”“忘了命令”“忘了文件”“该用哪个 Skill”
- 用户开始一个新的论文阶段
- 用户准备跑实验、写论文、整理文献、做 slides、投稿、rebuttal 或 public release
- 用户的任务描述不够明确，需要先导航或追问
- 长任务结束后需要给出下一步

必须提醒：

```bash
python tools/project_brief.py
```

以及：

```text
Use $paper-orchestrator-cn to 接管导航。请告诉我当前阶段、该用哪个 Skill、要读写哪些文件、要运行哪些命令。
```

推荐输出格式：

```text
不知道下一步时，可以先运行：
python tools/project_brief.py

或者直接对 Codex 说：
Use $paper-orchestrator-cn to 接管导航。请告诉我当前阶段、该用哪个 Skill、要读写哪些文件、要运行哪些命令。
```

注意：不要让用户自己翻 README 找命令。你应该直接给出可复制的下一步。

## Supervisor-Skills / Superpowers 路由规则

### Supervisor-Skills

当用户需要以下任务时，可以建议 Supervisor-Skills：

- 科研副导师式 idea 评估
- 论文 story line / Introduction 结构审查
- 技术类或 benchmark 类论文结构审查
- 投稿前 pre-submission review
- 科研图表设计建议

边界：

- 不接管项目导航；导航仍由 `$paper-orchestrator-cn` 负责。
- 不接管 managed experiment；实验执行仍由 `$experiment-manager-cn` 负责。
- 它的结论如果会影响研究方向、论文结构或投稿策略，需要用 `$session-recorder-cn` 记录为轻量事件。

### Superpowers

当用户进入复杂工程实现阶段时，可以建议 Superpowers：

- 实现实验代码
- 重构 experiment pipeline
- 写测试
- 复杂代码计划与执行
- git worktree / TDD / code review 流程

边界：

- 不接管论文项目总控。
- 不替代 `$experiment-manager-cn` 的实验执行与日志系统。
- 不允许未经确认修改 managed experiment 接口：
  - `configs/experiments/*.yaml`
  - `experiments/registry.csv`
  - `scripts/experiments/run_managed_experiment.py`

默认优先级：

- “该做什么科研工作” → `$paper-orchestrator-cn`
- “该怎么设计实验” → `$academic-research-suite`
- “把实验跑起来并记录” → `$experiment-manager-cn`
- “实现或重构代码” → Superpowers
- “导师式评价或投稿前挑刺” → Supervisor-Skills

## 新增第三方增强 Skill / 插件路由

### Supervisor-Skills

当用户需要以下任务时，可以建议 Supervisor-Skills：

- 科研副导师式 idea 评估
- 论文 story line / Introduction 结构审查
- 技术类或 benchmark 类论文结构审查
- 投稿前 pre-submission review
- 科研图表设计建议

边界：

- 不接管项目导航；导航仍由 `$paper-orchestrator-cn` 负责。
- 不接管 managed experiment；实验执行仍由 `$experiment-manager-cn` 负责。
- 它的结论如果会影响研究方向、论文结构或投稿策略，需要用 `$session-recorder-cn` 记录为轻量事件。

### Superpowers

当用户进入复杂工程实现阶段时，可以建议 Superpowers：

- 实现实验代码
- 重构 experiment pipeline
- 写测试
- 复杂代码计划与执行
- git worktree / TDD / code review 流程

边界：

- 不接管论文项目总控。
- 不替代 `$experiment-manager-cn` 的实验执行与日志系统。
- 不允许未经确认修改 managed experiment 接口：
  - `configs/experiments/*.yaml`
  - `experiments/registry.csv`
  - `scripts/experiments/run_managed_experiment.py`

### 默认优先级

如果任务是“该做什么科研工作”，先用 `$paper-orchestrator-cn`。
如果任务是“该怎么设计实验”，优先 `$academic-research-suite`。
如果任务是“把实验跑起来并记录”，优先 `$experiment-manager-cn`。
如果任务是“实现或重构代码”，可以建议 Superpowers。
如果任务是“导师式评价或投稿前挑刺”，可以建议 Supervisor-Skills。
