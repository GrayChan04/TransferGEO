# 参考文献

v1 Managed 采用最简单、最可控的 BibTeX 流程：

1. 你从论文官网、ACM、ACL Anthology、arXiv、Google Scholar 等页面复制 BibTeX。
2. 直接粘贴到：

```text
paper/references.bib
```

3. 论文中引用对应 BibTeX key。

不再维护 `references/bib/official/` 和 `references/bib/arxiv/` 双路径，也不自动合并 BibTeX。

`references/metadata.csv` 只用于记录论文阅读和项目关系，不负责生成 BibTeX。
