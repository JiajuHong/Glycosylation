# 第三阶段：程序复现目标 O4 乙酰化规则

## 目标

从原始正反应受体及其已标注的 `Acceptor_O4_Index` / `Acceptor_C4_Index` 出发，确定性生成目标 O4 封闭受体：

```text
原始受体 O4-H  →  O4-C(=O)CH3
```

程序不读取产物、产率或 α/β 标签，也不以合作方封闭结构作为生成输入。合作方结构只用于最后一步的独立一致性比较。

## 运行

```bash
python -m layer1.regenerate_o4_negatives
```

默认输入：

- `data/processed/negative_cleaned_861.csv`
- `data/processed/glyco_model_local.csv`

默认输出：

- `data/processed/o4_acetylated_regenerated_861.csv`
- `results/o4_acetylation_validation_861.csv`
- `results/o4_acetylation_validation_summary.csv`

只要任意样本未通过全部检查，脚本便以非零状态退出。

## 八项逐条验证

1. 原始受体 SMILES 可由 RDKit 解析。
2. C4/O4 索引存在、元素正确，且 C4—O4 成键。
3. 修改前 O4 为具有氢的游离羟基氧。
4. 修改后新增键严格为 `O4-C(=O)-C`，O4 不再带氢。
5. 修改后分子通过 RDKit 完整价态与芳香性清理，并可由生成的 canonical SMILES 重新解析。
6. 删除新增乙酰基后，异构 canonical SMILES 与原始受体完全一致；原有手性原子标签和数量不变。
7. 原有全部原子属性和键属性不变，且全分子严格只增加 3 个原子与 3 条键。
8. 程序生成结构与合作方封闭结构的异构 canonical SMILES 完全一致。

## 当前结果

861 条样本的八项检查均通过，合作方结构一致率为 100%。
