# 详细代码目录与运行约定

日常使用先看根目录 `README.md`。本文件是补充说明，以下路径均相对项目根目录。

项目代码按职责拆分为以下 Python 包：

```text
common/    三层共享的评测工具
layer1/    目标 O4 结构门控：构造、训练、推理与测试
layer2/    可追溯的成功文献先例检索
layer3/    α/β 立体选择性：预处理、图构造、训练与测试
baseline/  传统机器学习基线与结果汇总
support/experiments/ 正式计算实验、结果汇总与实验协议（非日常入口）
support/maintenance/ 模型清单、审计、验收与历史维护工具
support/docs/ 详细实现说明
models/    第一层和第三层复用的神经网络组件
pipeline/  三层统一推理、条件筛选、条件库和状态定义
artifacts/ 冻结 checkpoint、指标、manifest 与验收报告
logs/      正式训练运行日志（按实验版本集中保存，不放在项目根目录）
results/   正式任务、基线、预测导出、消融和误差分析结果
environment.server.yml  服务器冻结推理、审计与测试的精确依赖版本
```

所有 Python 入口都应从项目根目录通过模块方式运行，例如：

```bash
python -m layer1.train_hard_feasibility --help
python -m layer1.predict_hard_feasibility --help
python -m layer2.retrieve_literature_evidence --help
python -m layer3.train_gine --help
python -m baseline.run_baseline_ml
python -m support.experiments.run_formal_computational_v1 --help
python -m support.experiments.finalize_formal_computational_v1 --help
python -m support.maintenance.build_model_manifest --help
python -m support.maintenance.audit_frozen_models --help
python -m pipeline.predict_three_layer --help
```

第三层的正式计算实验协议、固定矩阵、运行方式和输出布局见
`support/experiments/FORMAL_COMPUTATIONAL_V1.md`，已完成结果与结论见
`support/experiments/FORMAL_COMPUTATIONAL_V1_WORK_SUMMARY.md`。正式运行的中间文件不得散落到项目根目录；
模型检查点和逐轮历史放在 `artifacts/checkpoints/formal_computational_v1/`，单次指标放在
`artifacts/metrics/formal_computational_v1/`，最终汇总放在
`results/formal_computational_v1/`。

冻结版统一推理使用 `artifacts/model_manifest_v2.json`。正式执行顺序固定为：

1. 第一层三个种子的可行性分数取均值，以 `0.5` 判断是否通过；
2. 未通过的候选停止，不运行第二、第三层；
3. 第二层给出身份计数、原始 Tanimoto 和近邻来源，不定义适用域且不作硬门控；
4. 第三层使用每个种子各自的验证集阈值投票，`0=Alpha`、`1=Beta`。

第一层的固定名称是“目标 O4 结构门控”。它只判断目标 O4 是否满足必要结构
条件，不判断实验一定成功。

第三层导出的数值字段统一命名为 `Beta_Score`。文献证据和第三层分数都不能解释为
真实反应成功概率。文献先例类型、`Final_Status` 和种子分歧分别记录，不能混用。

示例：

```bash
python -m pipeline.predict_three_layer \
  --input candidates.csv \
  --output predictions.csv \
  --manifest artifacts/model_manifest_v2.json \
  --neighbors-output nearest_success_reactions.csv
```

输入至少需要 `Donor_Canonical_SMILES` 和 `Acceptor_Canonical_SMILES`。供体类型可由
结构唯一推断时省略；若受体含多个可识别糖环，必须提供 `Target_O4_Index`。溶剂、
催化剂/活化剂、温度和时间缺失时会使用各自的缺失标记，不会伪造数值。

`Final_Status` 只描述流水线终态：`invalid_input`、`structurally_infeasible`、
`predicted` 或 `internal_error`。供受体历史由 `Pair_History` 单独表示；当前候选条件的
证据由 `Condition_Transfer_Evidence` 和各项身份计数、Tanimoto 字段表示；
种子分歧记录在两个 `Seed_Disagreement` 字段及 `Warnings` 中。第一层拒绝后，
第二、第三层字段保持空值或 `not_evaluated`，并写入明确的跳过原因。

条件推荐层不把种子多数票解释为成功概率：目标构型获得 3/3 支持记为“优先”，
2/3 支持记为“探索”，最多 1/3 支持则“不推荐”。同一投票等级内按当前候选条件的
同底物、单一底物相同和类似底物证据排序；同供体与同受体等级相同，
再参考启用手性的非补偿联合结构相似度。
`Pair_History` 仅展示，不参与候选条件排序。
`Task_Mode=exploratory_rescue` 的第三类任务
始终标记为“探索性救援”，不能解释为反应成功预测。

要求“更换原条件”的正式任务会用两种证据排除原文模板：规范化后的溶剂、催化剂、
温度和时间完全相同，或 `Source_Reaction_ID` 出现在模板支持反应列表中。后者用于
处理混合溶剂分隔符差异和原表条件字段不完整的情况。

测试命令：

```bash
python -m unittest discover -s layer1/tests -t . -v
python -m unittest discover -s layer2/tests -t . -v
python -m unittest discover -s layer3/tests -t . -v
```

不要直接进入子目录运行脚本，否则项目根目录可能不在 Python 的模块搜索路径中。
