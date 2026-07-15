# SMI-TED 分子级特征提取

本项目使用 IBM 官方 SMI-TED 289M 的冻结分子表示作为全局二维化学语义特征。模型不参与下游反向传播，也不替代 Chiral-GINE 对手性与图拓扑的建模。

## 固定的数据处理约定

- Chiral-GINE 继续读取原始 `Donor_Canonical_SMILES` 和 `Acceptor_Canonical_SMILES`，保留立体化学。
- SMI-TED 单独使用 RDKit 生成的 canonical、non-isomeric SMILES，即 `isomericSmiles=False`。
- 供体和受体合并去重，每个唯一的 non-isomeric 分子只编码一次。
- 使用 IBM 官方 `load_smi_ted()` 加载 `smi-ted-Light_40.pt`，使用官方 `model.encode()` 得到 768 维瓶颈表示。
- 不执行 PCA、L2 归一化、提前降维或供体/受体拼接。
- 开发阶段不运行五折交叉验证；本步骤只做一次冻结特征提取。

## 服务器执行环境

所有模型计算在 `server1` 完成：

```text
项目目录：/home/jjhong/gly
Conda 环境：one
Python：/home/jjhong/.conda/envs/one/bin/python
```

IBM 官方源码固定到 `materials` 仓库提交：

```text
dc01519f434c0c10227a1d1955cd8a025b136151
```

执行命令：

```bash
cd /home/jjhong/gly
/home/jjhong/.conda/envs/one/bin/python scripts/extract_smi_ted_features.py \
  --csv data/processed/glyco_model_local.csv \
  --output-dir data/processed/smi_ted_features \
  --smi-ted-code-dir third_party/materials/models/smi_ted \
  --model-dir third_party/smi_ted_model \
  --batch-size 8 \
  --source-revision dc01519f434c0c10227a1d1955cd8a025b136151
```

服务器没有可用外网出口，所以官方源码和权重先下载到本机，再复制到服务器；本机不加载或运行 SMI-TED。提取脚本仍调用 IBM 官方 loader 和 encoder，只把 loader 的 Hugging Face 文件解析器指向已经下载的本地官方文件。

## 自动验证

脚本在写出最终结果前检查：

1. 两个输入字段均存在且所有 SMILES 可由 RDKit 解析。
2. SMI-TED 实际输入不超过官方 202-token 上限，禁止静默截断。
3. 每批输出形状为 `[batch, 768]`，且不存在 NaN 或 Inf。
4. 同一小批分子重复编码的最大绝对误差不超过 `1e-6`。
5. 逐反应供体、受体数组与唯一分子数组及索引映射完全一致。
6. 记录删除立体信息后发生合并的立体异构体组，供审计使用；这些差异仍由 Chiral-GINE 分支处理。

## 输出文件

输出目录为 `data/processed/smi_ted_features/`：

```text
smi_ted_unique_embeddings.npy   唯一分子的原始 768 维表示
donor_smi_ted_768.npy           与 1561 条反应逐行对齐的供体表示
acceptor_smi_ted_768.npy        与 1561 条反应逐行对齐的受体表示
donor_smi_ted_indices.npy       供体到唯一分子表的索引
acceptor_smi_ted_indices.npy    受体到唯一分子表的索引
smi_ted_molecule_index.csv      唯一分子、token 长度及出现次数
reaction_smi_ted_mapping.csv    反应行到供体/受体特征的完整映射
stereo_collision_audit.csv      non-isomeric 规范化导致的立体异构体合并审计
metadata.json                   输入、模型、环境、校验结果与文件 SHA-256
```

`donor_smi_ted_768.npy` 和 `acceptor_smi_ted_768.npy` 的第 `i` 行严格对应 `glyco_model_local.csv` 的第 `i` 行。后续模型中应使用两个不共享参数的投影层，再分别与供体和受体图表示融合。
