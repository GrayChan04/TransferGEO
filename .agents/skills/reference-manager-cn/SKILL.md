---
name: reference-manager-cn
description: 中文参考文献管理。BibTeX 由用户人工复制到 paper/references.bib；本 Skill 只管理阅读笔记、metadata 和 claim-citation 对齐。
---

# 参考文献管理

## BibTeX 规则

本模板不自动抓取或合并 BibTeX。

用户把每篇论文的 BibTeX 直接复制到：

```text
paper/references.bib
```

Codex 的职责是：

1. 检查 `paper/references.bib` 中是否有对应 key；
2. 提醒用户缺失 BibTeX；
3. 帮用户整理 `references/metadata.csv`；
4. 帮用户建立中文阅读笔记；
5. 把重要 citation 对齐到 `paper/claim_tracker.csv`。

## 阅读笔记

每篇论文建议总结：背景、问题、方法、数据集、指标、结论、局限性、与本项目关系。
