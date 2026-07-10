---
name: dependency-manager-cn
description: 中文依赖管理。根据当前 idea、benchmark、实验环境决定是否新增 Python/Node 依赖。
---

# 依赖管理

不要预装大依赖。新增依赖流程：

1. 说明原因
2. 写入 `requirements/<group>.txt`
3. 运行 `tools/install_python_deps.sh <group>`
4. 记录到 `private/13_依赖与环境记录.md`
