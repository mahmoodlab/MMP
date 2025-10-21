# TCGA BLCA 因果分离模型配置指南

## ✅ 已修复的问题

1. ✓ **配置文件路径错误**: `model_histo_config` 默认值已从 `ABMIL_default` 改为 `PANTHER_default`
2. ✓ **out_type 参数错误**: 默认值已从 `param_cat` 改为 `allcat`
3. ✓ **特征维度自动检测**: 添加了维度检测工具和明确的配置说明

## 📁 您的数据配置

根据您提供的信息：

```bash
病理切片特征路径: /data/TCGA/BLCA/features/pt_files
```

## 🚀 快速开始

### 步骤 0: 检测特征维度（重要！）

```bash
# 首先检测您的特征维度
python check_feature_dim.py /data/TCGA/BLCA/features/pt_files

# 输出示例:
# ✅ 所有特征文件的维度一致: 768
# 📋 您需要在训练时设置: --in_dim 768
```

如果您的特征是 **768维**（常见于ViT模型），训练脚本已经自动配置好了。

### 方法 1: 使用专用脚本（推荐）

```bash
cd scripts/survival
chmod +x train_blca_causal_separation.sh

# 如果需要，编辑脚本修改 IN_DIM=768 为您的实际维度

./train_blca_causal_separation.sh 0 0
```

### 方法 2: 直接Python命令

```bash
cd src
python -m training.main_survival \
    --split_dir splits/TCGA_BLCA_survival_k=0 \
    --split_names 0 \
    --data_source /data/TCGA/BLCA/features/pt_files \
    --in_dim 768 \
    --model_histo_type PANTHER \
    --model_histo_config PANTHER_default \
    --model_mm_type causal_separation \
    --n_proto 16 \
    --out_type allcat \
    --em_iter 1 \
    --tau 1.0 \
    --selection_method gumbel \
    --lambda_causal_consistency 1.0 \
    --lambda_confound_random 0.5 \
    --lambda_sparsity 0.01 \
    --lambda_balance 0.1 \
    --max_epochs 20 \
    --lr 2e-4 \
    --results_dir results/BLCA_causal_separation
```

**⚠️ 重要**: 将 `--in_dim 768` 改为您实际的特征维度！

## ⚙️ 必需的配置步骤

### 1. 准备数据分割文件

确保您有正确的split文件：

```bash
splits/TCGA_BLCA_survival_k=0/
├── train_0.csv
├── val_0.csv
└── test_0.csv
```

每个CSV文件应包含列：
- `slide_id`: 切片ID
- `os_survival_days`: 总生存时间（天）
- `os_censorship`: 删失标志 (0=事件, 1=删失)

### 2. 准备基因组数据（可选）

如果使用多模态（组织学+基因组）：

```bash
data_csvs/TCGA_BLCA/
├── rnaseq/
│   └── tcga_blca_rna_clean.csv     # RNA-seq数据
└── signatures/
    └── hallmark_gene_sets.csv      # 基因集签名
```

如果只用组织学数据，设置：
```bash
--omics_modality none
```

### 3. 准备原型（如果需要）

如果使用预先计算的原型：

```bash
# 先运行原型聚类
cd src
python -m training.main_prototype \
    --mode faiss \
    --data_source /data/TCGA/BLCA/features/pt_files \
    --split_dir splits/TCGA_BLCA_survival_k=0 \
    --n_proto 16 \
    --in_dim 1024 \
    --n_proto_patches 100000
```

## 🔧 关键参数说明

### 必须正确设置的参数

| 参数 | 正确值 | 说明 |
|------|--------|------|
| `--in_dim` | `768` | ⚠️ **最重要！** 必须匹配您的特征维度 |
| `--model_histo_config` | `PANTHER_default` | ⚠️ 必须匹配模型类型 |
| `--out_type` | `allcat` | ⚠️ PANTHER只支持: allcat, weight_param_cat, weight_avg_all |
| `--data_source` | `/data/TCGA/BLCA/features/pt_files` | 您的特征路径 |

### PANTHER支持的out_type选项

- ✅ `allcat`: 拼接所有参数 (推荐，用于因果分离)
- ✅ `weight_param_cat`: 加权参数拼接
- ✅ `weight_avg_all`: 加权平均（均值+方差）
- ✅ `weight_avg_mean`: 只加权平均均值
- ❌ `param_cat`: 不支持！会报错

## 🐛 故障排除

### 错误 1: 特征维度不匹配（最常见！）

```bash
RuntimeError: Expected size for first two dimensions of batch2 tensor to be: [1, 1024] but got: [1, 768]
```

**原因**: 您的特征是768维，但模型期望1024维（或其他维度）

**解决方案**:
```bash
# 1. 首先检测您的特征维度
python check_feature_dim.py /data/TCGA/BLCA/features/pt_files

# 2. 在训练命令中明确指定维度
--in_dim 768  # 使用您检测到的实际维度

# 3. 如果您有预先计算的原型，需要重新生成
cd src
python -m training.main_prototype \
    --mode faiss \
    --data_source /data/TCGA/BLCA/features/pt_files \
    --split_dir splits/TCGA_BLCA_survival_k=0 \
    --n_proto 16 \
    --in_dim 768 \
    --n_proto_patches 100000
```

### 错误 2: Config path doesn't exist

```bash
AssertionError: Config path ./configs/ABMIL_default/config.json doesn't exist!
```

**解决方案**:
```bash
# 明确指定配置
--model_histo_config PANTHER_default
```

### 错误 2: Out mode not implemented

```bash
NotImplementedError: Out mode param_cat not implemented
```

**解决方案**:
```bash
# 使用正确的out_type
--out_type allcat
```

### 错误 3: 找不到数据文件

```bash
FileNotFoundError: [Errno 2] No such file or directory
```

**解决方案**:
1. 检查路径是否正确（使用绝对路径）
2. 检查文件是否为 `.pt` 格式
3. 检查split CSV中的slide_id是否与特征文件名匹配

### 错误 4: omics数据不匹配

```bash
KeyError: 'slide_id not found'
```

**解决方案**:
```bash
# 如果没有RNA数据，暂时只用组织学
--omics_modality none
--model_mm_type causal_separation  # 仍然可以用因果分离
```

## 📊 预期输出

训练成功开始时应该看到：

```
Init Causal Separation Model...
Constructing unsupervised slide embedding...

Aggregating train set features...
Aggregating val set features...
Aggregating test set features...

Init optimizer ...
Setup EarlyStopping...

########## TRAIN Epoch: 0 ##########
Batch 10/XX | Loss: X.XXXX | Causal: X.XXXX | Confound: X.XXXX | Full: X.XXXX
```

## 🔍 验证设置

运行前快速验证：

```bash
# 1. 检查数据路径
ls /data/TCGA/BLCA/features/pt_files/*.pt | head -5

# 2. 检查split文件
cat splits/TCGA_BLCA_survival_k=0/train_0.csv | head -5

# 3. 检查配置文件存在
ls configs/PANTHER_default/config.json

# 4. 测试导入（在Python中）
python -c "from mil_models.model_causal_separation import CausalSeparationModel; print('OK')"
```

## 📝 完整配置模板

创建一个配置文件 `config_blca.sh`:

```bash
#!/bin/bash

# === 数据路径 ===
export DATA_ROOT="/data/TCGA/BLCA/features/pt_files"
export SPLIT_DIR="splits/TCGA_BLCA_survival_k=0"

# === 模型配置 ===
export MODEL_HISTO_TYPE="PANTHER"
export MODEL_HISTO_CONFIG="PANTHER_default"
export MODEL_MM_TYPE="causal_separation"
export OUT_TYPE="allcat"

# === 原型配置 ===
export N_PROTO=16
export EM_ITER=1
export TAU=1.0

# === 因果分离配置 ===
export SELECTION_METHOD="gumbel"
export LAMBDA_CAUSAL=1.0
export LAMBDA_CONFOUND=0.5
export LAMBDA_SPARSITY=0.01
export LAMBDA_BALANCE=0.1

# === 训练配置 ===
export MAX_EPOCHS=20
export LR=2e-4
export BATCH_SIZE=1
```

然后：
```bash
source config_blca.sh
cd src
python -m training.main_survival \
    --data_source "$DATA_ROOT" \
    --split_dir "$SPLIT_DIR" \
    --model_histo_type "$MODEL_HISTO_TYPE" \
    --model_histo_config "$MODEL_HISTO_CONFIG" \
    --model_mm_type "$MODEL_MM_TYPE" \
    --out_type "$OUT_TYPE" \
    --n_proto "$N_PROTO" \
    --em_iter "$EM_ITER" \
    --tau "$TAU" \
    --selection_method "$SELECTION_METHOD" \
    --lambda_causal_consistency "$LAMBDA_CAUSAL" \
    --lambda_confound_random "$LAMBDA_CONFOUND" \
    --lambda_sparsity "$LAMBDA_SPARSITY" \
    --lambda_balance "$LAMBDA_BALANCE" \
    --max_epochs "$MAX_EPOCHS" \
    --lr "$LR" \
    --batch_size "$BATCH_SIZE" \
    --results_dir results/BLCA_test
```

## 💡 优化建议

### 对于BLCA数据集

1. **样本量**: TCGA-BLCA约有400个样本
   - 建议使用5折交叉验证
   - 每折约80个样本用于测试

2. **训练参数调整**:
   ```bash
   --max_epochs 30          # BLCA可能需要更多epoch
   --es_patience 15         # 增加patience
   --lr 1e-4               # 可能需要更小的学习率
   ```

3. **因果分离参数**:
   ```bash
   --selection_method topk           # 对小数据集可能更稳定
   --top_k_ratio 0.6                # 选择60%的原型
   --lambda_sparsity 0.02           # 增加稀疏性
   ```

## 📞 获取帮助

如果遇到问题：

1. 查看详细文档: `CAUSAL_SEPARATION_README.md`
2. 查看快速开始: `QUICK_START.md`
3. 检查实现细节: `IMPLEMENTATION_SUMMARY.md`

---

**更新日期**: 2025-10-21
**适用版本**: v1.1+
**状态**: ✅ 已测试修复
