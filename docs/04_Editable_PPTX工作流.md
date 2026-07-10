# Editable PPTX 工作流

默认流程：

```text
HTML → DOM 解析 → PptxGenJS text boxes / shapes / tables / images → editable PPTX
```

## 推荐 HTML

```html
<section class="slide">
  <h1>Method Overview</h1>
  <p>Our method contains three modules.</p>
  <ul>
    <li>Retriever</li>
    <li>Optimizer</li>
    <li>Evaluator</li>
  </ul>
</section>
```

## 生成命令

```bash
bash scripts/slides/html_to_editable_pptx.sh slides/html/deck.html slides/pptx/deck.pptx .slide
```

## 现实限制

可编辑 PPTX 优先保证文字可编辑和基本布局。复杂 CSS、动画、渐变、多层 SVG 可能需要手工精修。
