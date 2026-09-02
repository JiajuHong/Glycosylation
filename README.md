# 糖基化立体选择性项目

给定供体、受体和反应条件，预测固定受体4-O位点糖基化的α/β立体倾向，
并用结构检查和文献先例辅助筛选条件。不是反应成功率或产率预测。

## 只看主体代码，从这里开始

| 你想了解什么 | 先看这个文件 |
|---|---|
| 整个项目怎么运行 | [pipeline/predict_three_layer.py](pipeline/predict_three_layer.py) |
| 怎样筛选候选条件 | [pipeline/screen_conditions.py](pipeline/screen_conditions.py) |
| 主模型由哪些部分组成 | [models/glyco_gine_models.py](models/glyco_gine_models.py) |
| 怎样处理手性 | [models/chiral_gine_encoder.py](models/chiral_gine_encoder.py)、[models/chiral_gine_conv.py](models/chiral_gine_conv.py) |
| 供体和受体局部怎样交互 | [models/local_modules.py](models/local_modules.py) |
| 溶剂、催化剂、温度、时间怎样编码 | [models/condition_encoder.py](models/condition_encoder.py) |
| 主模型怎样训练、读入数据 | [layer3/train_gine.py](layer3/train_gine.py)、[layer3/glyco_dataset.py](layer3/glyco_dataset.py) |
| 第一层怎样检查O4结构 | [layer1/predict_hard_feasibility.py](layer1/predict_hard_feasibility.py) |
| 第二层怎样检索文献 | [layer2/retrieve_literature_evidence.py](layer2/retrieve_literature_evidence.py) |

## 主体目录

```text
pipeline/   使用入口：整条流程、条件筛选、条件库和状态定义
models/     神经网络组件：手性编码、局部交互、条件编码、分类器
layer3/     核心α/β模型：预处理、构图、训练、预测导出
layer1/     O4结构门控：数据准备、训练和预测
layer2/     当前条件的文献先例检索
common/     共享评价函数
baseline/   传统对照：LR、随机森林、XGBoost（均输入条件）
```

各目录下的 `tests/` 是自动测试，不是额外的模型。数据准备脚本可以在需要
重建训练数据时再看，不必按文件顺序阅读。

## 当前正式方案

1. 第一层判断目标O4的必要结构前提；不输入条件，不通过就停止。
2. 第二层提供当前候选条件的文献证据与来源，不作硬门控或成功率预测。
3. 第三层使用 **Chiral-GINE + 全局/局部结构 + 双向交互L3 + 条件编码 + MLP**，
   预测Alpha/Beta；三个种子分别按验证集阈值投票。

L1/L2/L3是第三层交互输出的三种消融配置，不是项目的三层。
正式模型是L3；XGBoost分类头仅做过实验，没有替换正式MLP。
条件在分类前与结构表示拼接；不直接调制局部交叉注意力。

## 日常只需要这两个命令

在项目根目录、已安装依赖的Python环境中运行。
服务器目录为 `/home/jjhong/gly`，已有环境Python为
`/home/jjhong/.conda/envs/one/bin/python`；依赖见 `environment.server.yml`。

预测你已经确定的底物和条件：

```bash
python -m pipeline.predict_three_layer \
  --input candidates.csv --output predictions.csv \
  --manifest artifacts/model_manifest_v2.json
```

查看条件筛选入口的输入要求和选项：

```bash
python -m pipeline.screen_conditions --help
```

输入供受体SMILES和已知的溶剂、催化剂/活化剂、温度、时间；若受体含多个
可识别糖环，需要指定 `Target_O4_Index`。缺失条件用缺失标记，不伪造数值。
输出的Beta分数、文献证据和推荐等级都不等于实验成功概率。

## 其他目录现在可以不看，但不要删除

- `data/`、`artifacts/`：运行必需的数据、词表、权重和模型清单。
- `results/`、`logs/`、`outputs/`：已有结果、日志和生成文件。
- `support/`：实验对照、审计维护工具和详细历史说明，已与主体入口分开。

需要写论文时再看 [计算实验总结](support/experiments/FORMAL_COMPUTATIONAL_V1_WORK_SUMMARY.md)。
需要复现或维护时看 [辅助材料导航](support/README.md)。
