# 🛠️ 维度不匹配错误解决方案

## ❌ 错误信息

```
RuntimeError: Expected size for first two dimensions of batch2 tensor to be: [1, 132] but got: [1, 66].
```

## 🔍 问题原因

这个错误发生在训练开始时，您看到：
```
Embedding already exists! Loading train
Embedding already exists! Loading test
```

**根本原因**：之前缓存的嵌入文件是用**不同的配置**生成的（例如不同数量的基因通路或原型），与当前模型配置不匹配。

## ✅ 解决方案（三选一）

### 方法 1: 使用清理脚本（最简单）

```bash
cd scripts
chmod +x clean_embeddings.sh
./clean_embeddings.sh splits/TCGA_BLCA_survival_k=0

# 然后重新训练
cd survival
./train_blca_causal_separation.sh 0 0
```

### 方法 2: 手动删除缓存

```bash
# 删除旧的嵌入缓存
rm -rf splits/TCGA_BLCA_survival_k=0/embeddings/*.pkl

# 重新训练（会自动重新生成嵌入）
cd scripts/survival
./train_blca_causal_separation.sh 0 0
```

### 方法 3: 使用 --overwrite 参数（已包含在脚本中）

训练脚本已经添加了 `--overwrite` 参数，会自动覆盖旧的嵌入：

```bash
cd scripts/survival
./train_blca_causal_separation.sh 0 0
```

## 📊 验证修复

成功后，您应该看到：

```
Init Causal Separation Model...
Constructing unsupervised slide embedding...

Aggregating train set features...
100%|████████████████████| 289/289 [XX:XX<00:00, X.XXit/s]

Aggregating val set features...
100%|████████████████████| 72/72 [XX:XX<00:00, X.XXit/s]

Aggregating test set features...
100%|████████████████████| 73/73 [XX:XX<00:00, X.XXit/s]

Init optimizer...
No EarlyStopping...

########## TRAIN Epoch: 0 ##########
Batch 10/289 | Loss: X.XXXX | Causal: X.XXXX | Confound: X.XXXX | Full: X.XXXX
```

**关键点**：
- ✅ 不再显示 "Embedding already exists! Loading"
- ✅ 显示 "Aggregating X set features..."
- ✅ 显示进度条
- ✅ 训练正常开始

## 🔧 为什么会发生这个问题？

嵌入文件缓存机制是为了加速训练，避免每次都重新计算。但是当您更改了以下任何配置时，旧的缓存就会不匹配：

- ❌ 改变了基因通路数量（`omics_modality`）
- ❌ 改变了原型数量（`n_proto`）
- ❌ 改变了特征维度（`in_dim`）
- ❌ 改变了模型类型（`model_histo_type`）
- ❌ 改变了输出类型（`out_type`）

## 💡 预防措施

### 1. 使用 --overwrite 参数

训练脚本现在默认包含此参数，会自动覆盖旧缓存：

```bash
python -m training.main_survival \
    --overwrite \
    [其他参数...]
```

### 2. 为不同配置使用不同的split目录

```bash
# 配置1
SPLIT_DIR="splits/TCGA_BLCA_survival_config1_k=0"

# 配置2
SPLIT_DIR="splits/TCGA_BLCA_survival_config2_k=0"
```

### 3. 定期清理旧缓存

```bash
# 清理所有旧的嵌入
find splits -name "embeddings" -type d -exec rm -rf {}/*.pkl \;
```

## 📝 相关文件位置

嵌入缓存文件通常在：
```
splits/TCGA_BLCA_survival_k=0/embeddings/
├── feats_h5_PANTHER_embeddings_proto_16_allcat_em_1_eps_0.1_tau_1.0.pkl
└── [其他配置的pkl文件]
```

文件名编码了生成时的配置：
- `feats_h5`: 数据源
- `PANTHER`: 模型类型
- `proto_16`: 16个原型
- `allcat`: 输出类型
- `em_1`: EM迭代次数
- `eps_0.1`, `tau_1.0`: 其他参数

## 🎯 快速诊断

如果您不确定是否有缓存问题：

```bash
# 检查是否存在嵌入缓存
ls -lh splits/TCGA_BLCA_survival_k=0/embeddings/

# 如果存在文件，且您改过配置，建议删除
rm splits/TCGA_BLCA_survival_k=0/embeddings/*.pkl
```

## ❓ 常见问题

**Q: 删除缓存会丢失数据吗？**
A: 不会。嵌入只是中间结果，会从原始特征文件重新生成。

**Q: 重新生成嵌入需要多久？**
A: 取决于数据量，通常几分钟到十几分钟。

**Q: 可以保留缓存吗？**
A: 可以，但只有在配置完全相同时才安全。建议删除以避免问题。

**Q: --overwrite 每次都重新生成吗？**
A: 是的，这样可以确保配置一致。

## 📚 相关文档

- 完整配置指南: `BLCA_SETUP_GUIDE.md`
- 快速开始: `START_HERE.md`
- 故障排除: 本文档

---

**更新日期**: 2025-10-21
**适用版本**: v1.2+
**状态**: ✅ 已测试
