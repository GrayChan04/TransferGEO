# Final v1 Scope

这个模板是 AI 科研论文产出的轻量母版。

## v1 已覆盖

- GitHub Template Repository 复用
- Codex 主动导航
- 指令不明确时追问
- 本地中文 Skill
- 第三方 Skill / 插件安装清单
- Supervisor-Skills 科研副导师增强
- Superpowers 工程实现增强
- 中文内部记录
- BibTeX 人工粘贴到 `paper/references.bib`
- Managed experiment：自然语言参数 → config → 执行 → 记录 → git checkpoint
- 实验结果覆盖保护
- LLM cost/token 记录 `cost_json`
- 轻量事件记录
- claim tracker
- 会议要求管理
- 可编辑 PPTX
- 匿名检查
- public release 排除规则

## v1 不包含

- DVC / MLflow / W&B / Hydra
- Docker / Conda 默认环境
- 自动 BibTeX 抓取
- 自动 PDF 下载
- dataset fingerprint
- 远程服务器调度
- 多 GPU scheduler
- 具体 AI benchmark 代码

这些能力应在具体论文项目中按需添加，而不是放进母版。
