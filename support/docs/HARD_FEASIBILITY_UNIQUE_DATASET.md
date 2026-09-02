# 目标 O4 硬结构可行性：唯一结构组合数据集

## 分组定义

结构组由以下两个字段唯一确定：

```text
Donor_Canonical_SMILES + Original_Acceptor_Canonical_SMILES
```

反应条件不参与分组，也不进入第一层结构模型。1561 条正反应归并为 1316 个唯一结构组。

## 数据层级

- `hard_feasibility_full_pairs_3122.csv`：1561 个原始反应各保留一条正样本和一条程序生成负样本，用于追溯。
- `hard_feasibility_structural_groups_1316.csv`：每个结构组一行，保留所有 Parent ID、Reaction ID 和重复次数。
- `hard_feasibility_unique_pairs_2632.csv`：1316 个结构组各一正一负，包含完整配对与审计元数据。
- `hard_feasibility_model_ready_2632.csv`：最小化模型输入表，移除条件、原始受体副本、Variant 和构造验证等直接泄漏字段。

## Model-ready 字段

```text
Sample_ID
Pair_ID
Structural_Group_ID
hard_feasibility
Donor_Canonical_SMILES
Acceptor_Canonical_SMILES
Donor_Type
Target_C4_Index
Target_O4_Index
```

`Sample_ID`、`Pair_ID` 和 `Structural_Group_ID` 仅用于追溯、配对和数据划分，不作为数值或类别特征输入模型。

## 重建

```bash
python -m layer1.build_unique_hard_feasibility_dataset
```

脚本只要发现生成失败、合作方已知结构不匹配、重复模型输入、跨标签结构冲突、组内索引不一致、正负配对缺失或禁止字段进入 model-ready 表，就会以非零状态退出。
