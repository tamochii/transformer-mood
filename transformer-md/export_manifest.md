# Transformers NotebookLM Export

这个目录是为 `NotebookLM` 整理的一份 `transformers` 文档导出包，目标是：

- 尽量保留项目里高价值、可直接阅读的 Markdown 文档
- 控制总量，不超过 `300` 个 `.md` 文件
- 优先保留中文官方文档，减少英文全量文档带来的噪音和体积

## 导出内容

- `repo_root/`
  - 根目录核心说明文档
  - 包含 `README.md`、`CONTRIBUTING.md`、`CODE_OF_CONDUCT.md`、`SECURITY.md`、`MIGRATION_GUIDE_V5.md`、`awesome-transformers.md`
- `docs_zh/`
  - `docs/source/zh/` 下的全部中文官方文档
- `examples/`
  - `examples/` 下的 Markdown 说明文档和任务示例 README
- `notebooks/`
  - `notebooks/README.md`

## 为什么没有导出英文全量文档

`docs/source/en/` 当前有数百篇 Markdown 文档，直接全量导出会明显超过你要求的规模上限，也会把大量模型页、参考页和细分主题页一起带进去。

这份导出优先保证：

- 文档量可控
- 主题覆盖尽量完整
- 更适合直接喂给 `NotebookLM`
- 所有导出后的 Markdown 文件名全局唯一，避免多个 `README.md` 重名

如果你后面需要，我还可以继续给你再做一份：

- 英文核心文档精简版
- 英文任务文档版
- 英文 API / 模型文档专题版

## 目录规模

当前这份导出预计包含：

- 中文官方文档：`71`
- 根目录核心文档：`6`
- 示例文档：`23`
- notebooks 文档：`1`
- 本说明文件：`1`

合计：`102` 个 Markdown 文件。
