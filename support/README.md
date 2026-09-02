# 辅助材料（非日常使用入口）

主体代码及日常命令见根目录 [README](../README.md)。本目录不增加新模型。

- `docs/`：原根目录的实现说明、数据构造说明及详细运行说明。
- `experiments/`：正式计算实验、XGBoost实验、论文结果总结。
- `maintenance/`：模型清单、审计、验收、源码快照及已执行的一次性维护脚本。

所有模块命令仍从项目根目录运行：实验命令前缀现在是
`python -m support.experiments.`，维护命令前缀是 `python -m support.maintenance.`。
例如 `python -m support.maintenance.build_model_manifest`。
这些是需要时使用的工具，不是尚待完成的工作；不要为了阅读项目重复训练或清理。

本次只调整目录与引用，不修改模型算法、权重、数据或实验指标。
`data/`、`artifacts/`、`results/`、`logs/`、`outputs/` 保持原位置。
旧实验清单、日志和归档内部的路径是当时运行记录，未批量篡改；需要复现历史
版本时使用对应源码快照。已归档实验不应使用当前代码覆盖重跑。

旧路径对照：根目录的详细Markdown说明 → `support/docs/`；
`experiments/` → `support/experiments/`；
原pipeline中的清单/审计/验收工具及实验目录中的快照/清理工具 → `support/maintenance/`。
