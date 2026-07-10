# 前 30 分钟使用指南

1. 初始化：

```bash
bash tools/init_new_paper.sh
```

2. 自检：

```bash
python tools/template_doctor.py
```

3. 启动 Codex：

```text
Use $paper-orchestrator-cn to 初始化这篇论文项目。请提醒我当前阶段，并建议应该调用哪些 Skill。
```

4. 先填这 3 个文件：

```text
private/01_研究问题.md
private/03_方法卡.md
private/04_实验计划.md
```

5. 用 `$academic-research-suite` 做初步文献 scoping。

6. 用 `$grill-me` 质询研究问题和实验设计。

7. 用 `$session-recorder-cn` 保存第一轮重要决策。
