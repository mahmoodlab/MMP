# 快速开始指南 - 因果分离模型

## ✅ 问题已修复

命令行参数解析问题已经修复！现在可以正常使用所有 causal separation 相关参数了。

## 🚀 使用方法

### 方法 1: 使用示例脚本（推荐）

```bash
cd scripts/survival
chmod +x causal_separation_demo.sh
./causal_separation_demo.sh 0 splits/TCGA_BRCA_survival_k=0 0 feats_h5
```

### 方法 2: 直接使用 Python

```bash
cd src
python -m training.main_survival \
    --split_dir splits/TCGA_BRCA_survival_k=0 \
    --split_names 0 \
    --data_source feats_h5 \
    --model_histo_type PANTHER \
    --model_mm_type causal_separation \
    --n_proto 16 \
    --selection_method gumbel \
    --lambda_causal_consistency 1.0 \
    --lambda_confound_random 0.5 \
    --lambda_sparsity 0.01 \
    --lambda_balance 0.1 \
    --results_dir results/causal_separation_test
```

## 📋 所有可用参数

### 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_mm_type` | str | coattn | 设置为 `causal_separation` 使用新模型 |
| `--use_causal_separation` | flag | False | 强制使用因果分离训练器 |

### 原型选择参数

| 参数 | 类型 | 默认值 | 选项 | 说明 |
|------|------|--------|------|------|
| `--selection_method` | str | gumbel | gumbel, topk, adaptive | 选择方法 |
| `--gumbel_tau` | float | 1.0 | - | Gumbel温度 |
| `--gumbel_hard` | bool | True | - | 使用硬Gumbel |
| `--top_k_ratio` | float | 0.5 | 0.0-1.0 | Top-K比例 |

### 损失权重参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--lambda_causal_consistency` | float | 1.0 | 因果一致性权重 |
| `--lambda_confound_random` | float | 0.5 | 混淆随机性权重 |
| `--lambda_sparsity` | float | 0.01 | 稀疏性权重 |
| `--lambda_balance` | float | 0.1 | 平衡权重 |

### 损失模式参数

| 参数 | 类型 | 默认值 | 选项 | 说明 |
|------|------|--------|------|------|
| `--causal_consistency_mode` | str | kl | kl, mse, cosine | 一致性损失模式 |
| `--confound_random_mode` | str | entropy | entropy, uniform, variance | 随机性损失模式 |
| `--target_selection_ratio` | float | 0.5 | 0.0-1.0 | 目标选择比例 |

### 模型架构参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--path_proj_dim` | int | 256 | 通路投影维度 |
| `--mult` | int | 1 | 前馈网络倍数 |

## 📝 完整示例

### 示例 1: 使用 Gumbel-Softmax 选择

```bash
python -m training.main_survival \
    --model_mm_type causal_separation \
    --selection_method gumbel \
    --gumbel_tau 1.0 \
    --gumbel_hard True \
    --lambda_causal_consistency 1.0 \
    --lambda_confound_random 0.5 \
    [... 其他标准参数 ...]
```

### 示例 2: 使用 Top-K 选择

```bash
python -m training.main_survival \
    --model_mm_type causal_separation \
    --selection_method topk \
    --top_k_ratio 0.6 \
    --lambda_sparsity 0.02 \
    [... 其他标准参数 ...]
```

### 示例 3: 使用自适应选择

```bash
python -m training.main_survival \
    --model_mm_type causal_separation \
    --selection_method adaptive \
    --target_selection_ratio 0.4 \
    --lambda_balance 0.2 \
    [... 其他标准参数 ...]
```

## 🔧 调试技巧

### 查看所有参数

```bash
cd src
python -m training.main_survival --help
```

### 检查是否使用了因果分离训练器

训练开始时会显示:
```
Init Causal Separation Model...
```

### 验证参数是否生效

训练日志会显示:
```
Loss: X.XXXX | Causal: X.XXXX | Confound: X.XXXX | Full: X.XXXX
```

## ⚠️ 常见问题

**Q: 参数无法识别？**
A: 确保使用最新版本的代码（commit 25ddc75 或更新）

**Q: 训练器没有使用因果分离？**
A: 检查 `--model_mm_type` 是否设置为 `causal_separation`

**Q: 损失值异常？**
A: 调整损失权重，特别是 `lambda_causal_consistency` 和 `lambda_confound_random`

## 📚 更多信息

- 详细文档: `CAUSAL_SEPARATION_README.md`
- 实现细节: `IMPLEMENTATION_SUMMARY.md`
- 更新日志: `CHANGELOG_CAUSAL_SEPARATION.md`

## ✨ 提示

1. **从默认参数开始**: 默认参数已经调优，适合大多数情况
2. **逐步调整**: 一次只改变一个参数，观察效果
3. **监控指标**: 关注 `c_index_causal` 和 `c_index_confound` 的差异
4. **检查选择**: 确保选择的因果原型数量合理（30-70%）

---

**最后更新**: 2025-10-21
**版本**: 1.1 (修复参数解析)
**状态**: ✅ 可用
