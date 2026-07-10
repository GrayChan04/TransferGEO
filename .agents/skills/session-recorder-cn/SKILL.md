---
name: session-recorder-cn
description: 中文轻量事件记录。只记录重要科研决策和阶段性结论，不记录普通聊天；默认使用时间、事件、结果、下一步。
---

# 中文轻量事件记录

## 只记录重要事件

需要记录：

- 研究问题确定或改变
- 方法设计确定或改变
- benchmark / baseline / metric 选择
- 一组实验完成后的阶段性结论
- 投稿会议选择
- 收到 review 后的回应策略
- 重大 rebuttal / revision / resubmission 决策
- 准备公开发布

不需要记录：

- 普通聊天
- 小措辞修改
- 临时想法
- 轻微 debug 尝试
- 无结果的搜索
- 重复说明

## 默认记录格式

```text
时间 | 事件 | 结果 | 下一步
```

## 默认写入

- 重要决策：`experiments/decision_log.md`
- 阶段性结果：`private/06_结果解读.md`
- 重大写作版本：`private/12_重大写作版本记录.md`
- 下一步任务：`private/10_下一步行动.md`

不要默认生成长篇会话记录。只有用户要求“保存本次讨论 / handoff / 详细记录”时，才写入 `private/dialogues/` 或 `private/handoffs/`。
