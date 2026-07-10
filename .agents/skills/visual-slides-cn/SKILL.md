---
name: visual-slides-cn
description: 中文图表和 slides 管理。登记方法图、实验图、HTML slides、可编辑 PPTX 和汇报材料。
---

# 图表与 Slides 管理

默认生成可编辑 PPTX：

```bash
bash scripts/slides/html_to_editable_pptx.sh slides/html/deck.html slides/pptx/deck.pptx .slide
```

不要默认使用整页截图式 PPTX。只有用户明确接受“不可编辑但高保真”时才用截图 fallback。

每个产物要记录 source、output、claim、是否可公开。
