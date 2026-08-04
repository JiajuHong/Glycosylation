# 代码目录与运行约定

项目代码按职责拆分为以下 Python 包：

```text
common/    三层共享的评测工具
layer1/    目标 O4 硬结构可行性：构造、训练、推理与测试
layer2/    文献支持度与适用域：U 池、评测与评分
layer3/    α/β 立体选择性：预处理、图构造、训练与测试
baseline/  传统机器学习基线与结果汇总
models/    第一层和第三层复用的神经网络组件
pipeline/  冻结模型清单、六 checkpoint 审计与三层统一推理
artifacts/ 冻结 checkpoint、指标、manifest 与验收报告
environment.server.yml  服务器冻结推理、审计与测试的精确依赖版本
```

所有 Python 入口都应从项目根目录通过模块方式运行，例如：

```bash
python -m layer1.train_hard_feasibility --help
python -m layer1.predict_hard_feasibility --help
python -m layer2.score_soft_compatibility --help
python -m layer3.train_gine --help
python -m baseline.run_baseline_ml
python -m pipeline.build_model_manifest --help
python -m pipeline.audit_frozen_models --help
python -m pipeline.predict_three_layer --help
```

冻结版统一推理使用 `artifacts/model_manifest_v2.json`。正式执行顺序固定为：

1. 第一层三个种子的可行性分数取均值，以 `0.5` 判断是否通过；
2. 未通过的候选停止，不运行第二、第三层；
3. 第二层给出文献支持度、适用域和近邻证据，不作硬门控；
4. 第三层使用每个种子各自的验证集阈值投票，`0=Alpha`、`1=Beta`。

第一层的固定名称是“硬结构可行性模型”。它只判断目标 O4 是否满足必要结构
条件，不判断实验一定成功。

第三层导出的数值字段统一命名为 `Beta_Score`，第二层与第三层分数都不能解释为
真实反应成功概率。`Final_Status`、`Soft_Domain_Status` 和种子分歧分别记录，不能混用。

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
`predicted` 或 `internal_error`。适用域内外由 `Soft_Domain_Status` 单独表示；
种子分歧记录在两个 `Seed_Disagreement` 字段及 `Warnings` 中。第一层拒绝后，
第二、第三层字段保持空值或 `not_evaluated`，并写入明确的跳过原因。

条件推荐层不把种子多数票解释为高置信度：目标构型获得 3/3 支持且位于适用域内
记为“优先”，3/3 支持但处于适用域边界记为“谨慎”，2/3 分歧或适用域外只记为
“探索”，最多 1/3 支持则“不推荐”。`Task_Mode=exploratory_rescue` 的第三类任务
始终标记为“探索性救援”，不能解释为反应成功预测。

测试命令：

```bash
python -m unittest discover -s layer1/tests -t . -v
python -m unittest discover -s layer3/tests -t . -v
```

不要直接进入子目录运行脚本，否则项目根目录可能不在 Python 的模块搜索路径中。
