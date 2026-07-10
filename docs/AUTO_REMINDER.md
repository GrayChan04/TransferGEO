# Auto Reminder：Codex 自动导航提醒

这个文件定义模板中最重要的两个入口，Codex 应该主动提醒用户。

## 命令行入口

```bash
python tools/project_brief.py
```

用途：

- 显示当前阶段
- 显示当前目标
- 显示下一阶段
- 显示常用 Skill
- 显示关键文件状态
- 给出下一步 Codex 指令

## Codex 入口

```text
Use $paper-orchestrator-cn to 接管导航。请告诉我当前阶段、该用哪个 Skill、要读写哪些文件、要运行哪些命令。
```

用途：

- 让 Codex 接管导航
- 主动推荐 Skill
- 主动列出关键文件
- 主动给出命令
- 主动判断是否需要追问

## 什么时候提醒

Codex 在这些情况下必须主动提醒：

- 用户说不知道下一步
- 用户忘了命令、文件或 Skill
- 用户准备跑实验
- 用户准备写论文
- 用户准备整理文献
- 用户准备投稿或 rebuttal
- 用户准备 public release
- 用户的指令不明确
- 一个长任务结束后需要建议下一步
<<<<<<< HEAD

## Supervisor-Skills / Superpowers 自动提醒

当用户问“这个 idea 值不值得做”“投稿前帮我挑刺”“导师会怎么评价”时，Codex 应提醒可以使用 Supervisor-Skills。

当用户问“帮我实现实验代码”“重构 pipeline”“写测试”“复杂代码开发”时，Codex 应提醒可以使用 Superpowers。

但任何时候都要强调：

- 项目导航仍由 `$paper-orchestrator-cn` 负责；
- 自动实验执行和记录仍由 `$experiment-manager-cn` 负责。

## Supervisor-Skills / Superpowers 自动提醒

当用户问“这个 idea 值不值得做”“投稿前帮我挑刺”“导师会怎么评价”时，Codex 应提醒可以使用 Supervisor-Skills。

当用户问“帮我实现实验代码”“重构 pipeline”“写测试”“复杂代码开发”时，Codex 应提醒可以使用 Superpowers。

但任何时候都要强调：

- 项目导航仍由 `$paper-orchestrator-cn` 负责；
- 自动实验执行和记录仍由 `$experiment-manager-cn` 负责。
=======
>>>>>>> 1bd3e85eba289b200cbc1799c28eb5dd4f06b03f
