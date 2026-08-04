# 目标 O4 硬结构可行性：统一位点标注

## 原则

正样本使用已验证的游离 4-OH 糖环角色。负样本不搜索游离 OH，而是通过确定性 O4 乙酰化过程中建立的原子映射，把正样本的糖环原子身份传递到封闭 canonical 结构。

统一角色：

```text
C1, O5, C2, C3, C4, C5, O4, O4_EXTERNAL
```

糖环拓扑必须满足：

```text
C1-O5-C5-C4-C3-C2-C1
C4-O4
```

## O4_EXTERNAL 约定

- 正样本：O4 外部只有隐式氢，角色列表为空，显式索引使用 `-1`。
- 负样本：O4 外部原子为乙酰基羰基碳，角色列表包含其 canonical 原子索引。

## 输出

- `data/processed/hard_feasibility_site_annotated_2632.csv`：带状态、来源和完整审计字段的标注表。
- `data/processed/hard_feasibility_site_model_ready_2632.csv`：移除 Variant、O4 状态文本、标注状态和条件字段后的模型表。
- `results/hard_feasibility_site_annotation_validation_1316.csv`：每个正负对 11 项检查。
- `results/hard_feasibility_site_annotation_qa.csv`：全数据汇总检查。

## 重建

```bash
python -m layer1.annotate_hard_feasibility_sites
```

任一结构组发生角色缺失、拓扑错误、索引越界、正样本 O4 非游离、负样本 O4 未乙酰化、原子映射不完整、立体化学变化或重复模型输入时，脚本以非零状态退出。
