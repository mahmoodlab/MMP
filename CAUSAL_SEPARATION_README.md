# Causal-Confounding Prototype Separation for MMP

## 概述

本项目实现了基于**因果原型解耦**的多模态生存预测方案。该方案通过学习区分因果原型和混淆原型，提高模型的泛化能力和可解释性。

### 核心思想

现有的MMP模型使用GMM进行原型聚合时只执行一轮EM算法，缺乏端到端的学习。本方案通过引入**原型选择层**和**三分支预测架构**，实现可微分的因果-混淆原型分离。

## 架构设计

### 整体流程

```
输入层:
├─ 组织学: 16个原型表征 {z_h,1, ..., z_h,16}
└─ 基因组: 50个通路表征 {z_g,1, ..., z_g,50}

原型选择层 (新增):
├─ 组织学选择网络 G_h: 学习选择权重 w_h ∈ [0,1]^16
│   └─ 输出: α_h,i = σ(MLP(z_h,i)) 表示原型i是因果的概率
├─ 基因组选择网络 G_g: 学习选择权重 w_g ∈ [0,1]^50
│   └─ 输出: α_g,j = σ(MLP(z_g,j)) 表示通路j是因果的概率
└─ Gumbel-Softmax技巧实现可微分的离散选择

原型分离层:
├─ 因果原型: z^c_h, z^c_g (通过软掩码加权)
└─ 混淆原型: z^s_h, z^s_g (通过软掩码加权)

三分支预测网络:
├─ 因果分支 F_causal:
│   ├─ 输入: z^c_h 和 z^c_g (只用因果原型)
│   ├─ Transformer融合
│   └─ 输出: 风险预测 ŷ_causal
│
├─ 混淆分支 F_confound:
│   ├─ 输入: z^s_h 和 z^s_g (只用混淆原型)
│   ├─ Transformer融合
│   └─ 输出: 风险预测 ŷ_confound
│
└─ 全部分支 F_full:
    ├─ 输入: 所有原型
    ├─ Transformer融合 (类似原始MMP)
    └─ 输出: 风险预测 ŷ_full
```

### 损失函数

模型使用多个损失函数的组合：

1. **生存损失** (Cox/NLL/Ranking): 三个分支各自的生存预测损失
2. **因果一致性损失**: 确保因果分支的预测接近全部分支
   - `L_consistency = KL(P_full || P_causal)` 或 `MSE(logits_full, logits_causal)`
3. **混淆随机性损失**: 鼓励混淆分支的预测接近随机
   - `L_random = -Entropy(P_confound)` (最大化熵)
4. **稀疏性损失**: 鼓励选择较少的因果原型
   - `L_sparsity = ||α||_1`
5. **平衡损失**: 防止选择所有或没有原型的退化解
   - `L_balance = (count(α > 0.5) - target)^2`

总损失:
```
L_total = L_survival_full +
          λ_1 * L_consistency +
          λ_2 * L_random +
          λ_3 * L_sparsity +
          λ_4 * L_balance
```

## 文件结构

### 新增文件

```
MMP/
├── src/
│   ├── mil_models/
│   │   ├── prototype_selection.py          # 原型选择网络
│   │   ├── model_causal_separation.py      # 三分支因果分离模型
│   │   ├── causal_losses.py                # 因果分离损失函数
│   │   └── model_factory.py                # 已修改: 添加causal_separation支持
│   └── training/
│       └── trainer_causal_separation.py    # 因果分离模型训练器
├── scripts/
│   └── survival/
│       └── causal_separation_demo.sh       # 示例训练脚本
├── test_causal_separation.py               # 端到端测试脚本
└── CAUSAL_SEPARATION_README.md             # 本文档
```

### 核心模块说明

#### 1. `prototype_selection.py`

实现三种原型选择策略:

- **GumbelSoftmaxSelector**: 使用Gumbel-Softmax进行可微分离散选择
  - 训练时: 软选择 (提供梯度)
  - 推理时: 硬选择 (argmax)
  - 支持straight-through estimator

- **TopKSelector**: 基于学习的重要性分数选择top-K个原型
  - 更简单直观
  - 可控的选择数量

- **AdaptiveSelector**: 学习自适应阈值
  - 每个样本可以有不同的选择阈值
  - 更灵活

#### 2. `model_causal_separation.py`

三分支架构的主模型:

- **SharedTransformerBranch**: 可复用的Transformer分支
- **CausalSeparationModel**: 完整的三分支模型
  - 支持所有原型选择方法
  - 兼容PANTHER/OT/H2T等嵌入模型
  - 输出三个分支的预测和选择信息

#### 3. `causal_losses.py`

损失函数集合:

- **CausalConsistencyLoss**: 因果一致性 (KL/MSE/Cosine)
- **ConfoundingRandomnessLoss**: 混淆随机性 (Entropy/Uniform/Variance)
- **SelectionSparsityLoss**: L1稀疏性
- **SelectionBalanceLoss**: 平衡约束
- **CausalSeparationLoss**: 组合所有损失
- **compute_causal_separation_metrics**: 监控指标计算

#### 4. `trainer_causal_separation.py`

专门的训练循环:

- 支持三分支模型的前向传播
- 计算组合损失
- 记录所有分支的C-index
- 跟踪原型选择统计

## 使用方法

### 1. 环境准备

确保已安装所有依赖:

```bash
conda env create -f env.yaml
conda activate mmp
```

### 2. 准备数据和原型

首先运行原型聚类 (与原始MMP相同):

```bash
cd src
python -m training.main_prototype \
    --mode faiss \
    --data_source /path/to/feats_h5 \
    --split_dir splits/TCGA_BRCA_survival_k=0 \
    --n_proto 16 \
    --in_dim 1024 \
    --n_proto_patches 100000
```

### 3. 训练因果分离模型

使用提供的脚本:

```bash
cd scripts/survival
chmod +x causal_separation_demo.sh
./causal_separation_demo.sh 0 splits/TCGA_BRCA_survival_k=0 0 feats_h5
```

或者直接使用Python:

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

### 4. 测试所有模块

运行端到端测试:

```bash
python test_causal_separation.py
```

这将测试:
- ✓ 所有模块导入
- ✓ 原型选择网络
- ✓ 三分支模型
- ✓ 损失函数
- ✓ 模型工厂集成
- ✓ 所有选择方法

## 配置参数

### 模型参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `model_mm_type` | 模型类型 | `causal_separation` |
| `selection_method` | 选择方法 | `gumbel` (选项: gumbel/topk/adaptive) |
| `gumbel_tau` | Gumbel温度 | `1.0` |
| `gumbel_hard` | 硬Gumbel | `True` |
| `top_k_ratio` | Top-K比例 | `0.5` |

### 损失权重

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `lambda_causal_consistency` | 因果一致性权重 | `1.0` |
| `lambda_confound_random` | 混淆随机性权重 | `0.5` |
| `lambda_sparsity` | 稀疏性权重 | `0.01` |
| `lambda_balance` | 平衡权重 | `0.1` |

### 损失模式

| 参数 | 说明 | 选项 |
|------|------|------|
| `causal_consistency_mode` | 一致性损失模式 | kl/mse/cosine |
| `confound_random_mode` | 随机性损失模式 | entropy/uniform/variance |
| `target_selection_ratio` | 目标选择比例 | `0.5` |

## 实验建议

### 超参数调优

1. **选择方法**:
   - `gumbel`: 最灵活，适合端到端学习
   - `topk`: 最简单，可控性强
   - `adaptive`: 最智能，需要更多数据

2. **损失权重**:
   - 从默认值开始: `[1.0, 0.5, 0.01, 0.1]`
   - 如果因果分支性能差: 增加 `lambda_causal_consistency`
   - 如果混淆分支仍有预测能力: 增加 `lambda_confound_random`
   - 如果选择太多原型: 增加 `lambda_sparsity`
   - 如果退化解 (全选/全不选): 增加 `lambda_balance`

3. **温度参数**:
   - 高温 (>1.0): 更软的选择，训练初期
   - 低温 (<1.0): 更硬的选择，微调阶段
   - 可以使用退火策略: tau从1.0降到0.5

### 评估指标

关注以下指标:

1. **性能指标**:
   - `c_index_full`: 全部分支的C-index (主要指标)
   - `c_index_causal`: 因果分支的C-index (应接近full)
   - `c_index_confound`: 混淆分支的C-index (应接近随机0.5)

2. **选择指标**:
   - `histo_causal_count`: 选择的组织学因果原型数量
   - `gene_causal_count`: 选择的基因因果原型数量
   - `histo_causal_ratio`: 组织学因果原型比例

3. **分离指标**:
   - `causal_full_similarity`: 因果与全部分支的相似度 (应该高)
   - `confound_entropy`: 混淆分支的预测熵 (应该高，接近随机)

### 预期结果

成功的因果分离应该满足:

1. ✓ `c_index_causal` ≈ `c_index_full` (±0.02)
2. ✓ `c_index_confound` ≈ 0.5 (随机水平)
3. ✓ 选择的因果原型数量合理 (30-70%的原型)
4. ✓ `confound_entropy` > `causal_entropy`

## 理论优势

### 相比原始MMP的改进

1. **端到端学习**: 原型选择可微分，与预测任务联合优化
2. **因果推断**: 显式分离因果特征和混淆特征
3. **可解释性**: 可以识别哪些原型对预测重要
4. **泛化能力**: 通过去除混淆因素提升泛化

### 潜在应用

1. **特征发现**: 识别对生存预测重要的组织学/基因模式
2. **领域适应**: 因果特征可能在不同数据集间更稳定
3. **对抗鲁棒性**: 减少对混淆因素的依赖
4. **生物学洞察**: 发现真正的预后标志物

## 未来扩展

### 可能的改进方向

1. **动态原型数量**: 让模型自动决定使用多少因果原型
2. **层次化选择**: 在patch级别和原型级别都进行选择
3. **对比学习**: 使用对比损失增强因果-混淆分离
4. **多任务学习**: 同时优化生存预测和其他任务
5. **不确定性估计**: 量化原型选择的置信度
6. **因果图学习**: 学习原型之间的因果关系

### 高级损失函数

可以添加更多约束:

- **独立性约束**: 确保因果和混淆原型独立
- **不变性约束**: 因果原型在不同域间保持不变
- **对抗训练**: 使用判别器区分因果/混淆
- **信息瓶颈**: 最小化互信息

## 故障排除

### 常见问题

**Q: 训练不稳定，损失波动大？**
A: 尝试:
- 降低学习率
- 增加Gumbel温度
- 减小损失权重
- 使用梯度裁剪

**Q: 所有原型都被选为因果/混淆？**
A: 增加`lambda_balance`权重，或调整`target_selection_ratio`

**Q: 因果分支性能远低于全部分支？**
A:
- 增加`lambda_causal_consistency`
- 减小`lambda_sparsity` (允许选择更多原型)
- 检查是否有足够的训练数据

**Q: 混淆分支C-index仍然很高？**
A:
- 增加`lambda_confound_random`
- 尝试不同的`confound_random_mode`
- 可能数据中混淆因素较少

## 引用

如果使用本代码，请引用:

```bibtex
@misc{causal_separation_mmp,
  title={Causal-Confounding Prototype Separation for Multimodal Survival Prediction},
  author={Your Name},
  year={2025},
  howpublished={https://github.com/yourusername/MMP}
}
```

## 许可

本项目遵循与MMP相同的许可协议。

## 联系方式

如有问题或建议，请提issue或联系作者。

---

**最后更新**: 2025-10-21
**版本**: 1.0
**作者**: Claude (AI Assistant)
