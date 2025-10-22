# 原型聚类改进实施指南
# Prototype Clustering Improvement Implementation Guide

## 📋 目录

1. [问题诊断](#问题诊断)
2. [快速开始](#快速开始)
3. [详细实施步骤](#详细实施步骤)
4. [性能评估](#性能评估)
5. [故障排查](#故障排查)
6. [高级选项](#高级选项)

---

## 🔍 问题诊断

### 当前实现的问题

根据对代码库的分析 (`/home/user/MMP/src/mil_models/PANTHER/networks.py`)，当前实现存在以下问题:

#### 1. **原型固定不可训练** ❌

**位置**: `src/mil_models/PANTHER/networks.py:62-66`

```python
self.m = nn.Parameter(torch.from_numpy(weights),
                     requires_grad=not fix_proto)  # fix_proto=True (默认)
```

**影响**:
- 原型在 k-means 初始化后完全冻结
- 无法通过反向传播学习更好的原型表示
- 原型分布与下游任务目标(生存预测)缺乏梯度连接

#### 2. **EM 迭代次数不足** ⚠️

**位置**: `src/mil_models/model_configs.py:88`

```python
em_iter: int = 3  # 默认只有 3 次迭代
```

**影响**:
- patch 到原型的分配不充分
- 可能陷入局部最优

#### 3. **非端到端学习** ❌

**流程**:
```
K-means 初始化 → 固定原型 → 每次 forward 运行 3 轮 EM → 原型不更新
```

**影响**:
- 原型表示与任务目标脱节
- 无法学习任务相关的特征模式

---

## 🚀 快速开始

### 方案一: 最小修改 (推荐首先尝试)

**只需修改配置文件，即可启用端到端训练！**

#### 步骤 1: 修改配置

编辑你的训练配置文件 (例如 `configs/TCGA-LUAD.yaml`):

```yaml
# 原来的配置
model:
  fix_proto: true   # 改为 false
  em_iter: 3        # 改为 5 或更高

# 改为
model:
  fix_proto: false  # ✅ 启用原型学习
  em_iter: 5        # ✅ 增加 EM 迭代次数
```

#### 步骤 2: 重新训练

```bash
# 使用修改后的配置训练
python src/training/main_prototype.py \
    --config configs/TCGA-LUAD.yaml \
    --fix_proto False \
    --em_iter 5
```

#### 步骤 3: 监控训练

观察以下指标:
- **训练稳定性**: 损失是否平稳下降
- **原型多样性**: 使用 t-SNE 可视化原型是否分散
- **C-index**: 验证集性能是否提升

**预期提升**: C-index +2-5%

---

### 方案二: 使用改进版网络 (推荐)

我们提供了改进版的 `ImprovedDirNIWNet`，包含额外功能:
- ✅ 原型多样性正则化 (防止原型崩溃)
- ✅ Gumbel-Softmax 重参数化 (增强可微分性)
- ✅ 可选的完整协方差矩阵

#### 步骤 1: 集成改进版网络

修改 `src/mil_models/PANTHER/layers.py`:

```python
# 原来的导入
from .networks import DirNIWNet

# 改为
from .networks_improved import ImprovedDirNIWNet as DirNIWNet
```

#### 步骤 2: 更新配置

```yaml
model:
  fix_proto: false
  em_iter: 5

  # 新增配置
  diversity_reg: 0.01      # 多样性正则化权重
  use_gumbel: false        # 是否使用 Gumbel-Softmax (可选)
  gumbel_temp: 1.0         # Gumbel 温度 (如果 use_gumbel=true)
  cov_type: "diag"         # 协方差类型: diag, spherical, full
```

#### 步骤 3: 修改训练循环

在训练脚本中添加多样性损失 (例如 `src/training/main_prototype.py`):

```python
# 在训练循环中
def train_epoch(model, train_loader, optimizer, criterion):
    for batch in train_loader:
        # ... 前向传播 ...

        # 主任务损失
        survival_loss = criterion(pred, target)

        # 多样性损失
        diversity_loss = model.panther.compute_diversity_loss()

        # 总损失
        total_loss = survival_loss + 0.01 * diversity_loss

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

#### 步骤 4: 训练

```bash
python src/training/main_prototype.py \
    --config configs/improved_panther.yaml
```

**预期提升**: C-index +3-7%

---

## 📝 详细实施步骤

### 阶段 1: 端到端基础改进 (1 周)

#### 1.1 修改模型配置

**文件**: `src/mil_models/model_configs.py`

```python
@dataclass
class PANTHERConfig(PretrainedConfig):
    # 原来
    fix_proto: bool = True   # 改为 False
    em_iter: int = 3         # 改为 5

    # 新增
    diversity_reg: float = 0.01  # 多样性正则化权重
```

#### 1.2 添加多样性正则化

**文件**: `src/mil_models/PANTHER/networks.py`

在 `DirNIWNet` 类中添加:

```python
class DirNIWNet(nn.Module):
    # ... 现有代码 ...

    def compute_diversity_loss(self):
        """防止原型崩溃到相似表示"""
        # 归一化原型
        m_norm = F.normalize(self.m, dim=-1)  # (p, d)

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(m_norm, m_norm.t())  # (p, p)

        # 只计算非对角元素
        mask = ~torch.eye(self.p, dtype=bool, device=self.m.device)
        off_diagonal_sim = similarity_matrix[mask]

        # 多样性损失 = 最小化相似度
        return off_diagonal_sim.abs().mean()
```

#### 1.3 修改训练循环

**文件**: `src/training/main_prototype.py`

```python
# 在训练函数中
for epoch in range(num_epochs):
    for batch in train_loader:
        # 前向传播
        logits, hazards = model(features)

        # 主任务损失
        survival_loss = criterion(logits, labels)

        # 多样性损失
        diversity_loss = model.panther.compute_diversity_loss()

        # 总损失 (添加权重)
        total_loss = survival_loss + config.diversity_reg * diversity_loss

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 记录日志
        if batch_idx % 10 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}:')
            print(f'  Survival Loss: {survival_loss.item():.4f}')
            print(f'  Diversity Loss: {diversity_loss.item():.4f}')
```

#### 1.4 使用原型专用优化器 (可选但推荐)

```python
# 分离原型参数和其他参数
proto_params = [model.panther.m, model.panther.V_]
proto_param_ids = [id(p) for p in proto_params]

other_params = [
    p for p in model.parameters()
    if id(p) not in proto_param_ids and p.requires_grad
]

# 原型专用优化器 (更小的学习率)
proto_optimizer = optim.AdamW(proto_params, lr=5e-5, weight_decay=1e-5)

# 主优化器
main_optimizer = optim.AdamW(other_params, lr=1e-4, weight_decay=1e-5)

# 训练时同时更新
for batch in train_loader:
    # ... 前向传播 ...

    main_optimizer.zero_grad()
    proto_optimizer.zero_grad()

    total_loss.backward()

    main_optimizer.step()
    proto_optimizer.step()
```

---

### 阶段 2: 因果增强 (2-3 周)

#### 2.1 实现对抗性去混淆

**创建新文件**: `src/mil_models/PANTHER/causal_prototype.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalPrototypeNet(nn.Module):
    """带因果推理的原型网络"""

    def __init__(self, base_proto_net, confounder_dim=128):
        super().__init__()

        self.base_net = base_proto_net
        self.p = base_proto_net.p
        self.d = base_proto_net.d

        # 混淆因子编码器 (从 WSI 级别特征提取)
        self.confounder_encoder = nn.Sequential(
            nn.Linear(self.d, confounder_dim),
            nn.ReLU(),
            nn.Linear(confounder_dim, confounder_dim)
        )

        # 混淆因子判别器 (对抗训练)
        self.confounder_discriminator = nn.Sequential(
            nn.Linear(self.p, 64),
            nn.ReLU(),
            nn.Linear(64, confounder_dim)
        )

    def forward(self, patches, wsi_features=None):
        """
        Args:
            patches: (B, N, d) - patch 特征
            wsi_features: (B, d) - WSI 级别特征 (用于提取混淆因子)

        Returns:
            pi, mu, Sigma, qq, confounders, pred_confounders
        """
        B, N, d = patches.shape

        # 如果没有提供 WSI 特征，使用 patches 的平均值
        if wsi_features is None:
            wsi_features = patches.mean(dim=1)  # (B, d)

        # 编码混淆因子
        confounders = self.confounder_encoder(wsi_features)  # (B, confounder_dim)

        # 计算原型分配
        pi, mu, Sigma, qq = self.base_net.map_em(patches)

        # 聚合为 WSI 级别的原型表示
        proto_repr = qq.mean(1)  # (B, p)

        # 预测混淆因子 (对抗训练目标)
        pred_confounders = self.confounder_discriminator(proto_repr)

        return pi, mu, Sigma, qq, confounders, pred_confounders

    def compute_adversarial_loss(self, confounders, pred_confounders):
        """
        对抗损失: 原型网络试图使原型表示与混淆因子无关

        对于原型网络: 最大化预测误差 (负号)
        对于判别器: 最小化预测误差
        """
        return F.mse_loss(pred_confounders, confounders)
```

#### 2.2 修改训练循环支持因果学习

```python
# 创建因果原型网络
causal_model = CausalPrototypeNet(model.panther)

# 分离优化器
proto_optimizer = optim.AdamW(model.panther.parameters(), lr=5e-5)
discriminator_optimizer = optim.AdamW(
    causal_model.confounder_discriminator.parameters(),
    lr=1e-4
)

# 训练循环
for epoch in range(num_epochs):
    for batch in train_loader:
        patches = batch['features']  # (B, N, d)
        wsi_features = batch.get('wsi_features', None)

        # 前向传播
        pi, mu, Sigma, qq, confounders, pred_confounders = \
            causal_model(patches, wsi_features)

        # === 步骤 1: 训练混淆因子判别器 ===
        discriminator_optimizer.zero_grad()
        disc_loss = causal_model.compute_adversarial_loss(
            confounders.detach(),
            pred_confounders
        )
        disc_loss.backward()
        discriminator_optimizer.step()

        # === 步骤 2: 训练原型网络 (对抗) ===
        proto_optimizer.zero_grad()

        # 主任务损失
        logits = survival_head(qq)
        survival_loss = criterion(logits, labels)

        # 对抗损失 (负号: 最大化判别器误差)
        adv_loss = -causal_model.compute_adversarial_loss(
            confounders,
            pred_confounders
        )

        # 总损失
        total_loss = survival_loss + 0.1 * adv_loss

        total_loss.backward()
        proto_optimizer.step()
```

---

### 阶段 3: 知识增强 (1-2 个月)

#### 3.1 集成生物医学文本编码器

**安装依赖**:

```bash
pip install transformers
```

**实现**:

```python
from transformers import AutoModel, AutoTokenizer

class MultimodalPrototypeNet(nn.Module):
    """多模态原型网络 (视觉 + 文本)"""

    def __init__(
        self,
        visual_proto_net,
        text_encoder='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        prototype_descriptions=None
    ):
        super().__init__()

        self.visual_proto_net = visual_proto_net
        self.p = visual_proto_net.p
        self.d = visual_proto_net.d

        # 文本编码器
        self.text_encoder = AutoModel.from_pretrained(text_encoder)
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder)

        # 投影层 (文本 → 视觉空间)
        self.text_projection = nn.Linear(768, self.d)

        # 原型文本描述
        if prototype_descriptions is None:
            # 默认病理学描述
            self.prototype_descriptions = [
                f"histological pattern {i+1}" for i in range(self.p)
            ]
        else:
            self.prototype_descriptions = prototype_descriptions

        # 冻结文本编码器 (可选)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def encode_text_prototypes(self):
        """编码文本描述为向量"""
        text_embeddings = []

        for desc in self.prototype_descriptions:
            # Tokenize
            tokens = self.tokenizer(
                desc,
                return_tensors='pt',
                padding=True,
                truncation=True
            ).to(next(self.parameters()).device)

            # 编码
            with torch.no_grad():
                text_feat = self.text_encoder(**tokens).last_hidden_state[:, 0, :]  # [CLS]

            # 投影
            text_feat = self.text_projection(text_feat)
            text_embeddings.append(text_feat)

        return torch.cat(text_embeddings, dim=0)  # (p, d)

    def compute_alignment_loss(self):
        """计算视觉原型与文本描述的对齐损失"""
        # 文本原型嵌入
        text_prototypes = self.encode_text_prototypes()  # (p, d)

        # 视觉原型嵌入
        visual_prototypes = self.visual_proto_net.m  # (p, d)

        # 余弦相似度对齐
        visual_norm = F.normalize(visual_prototypes, dim=-1)
        text_norm = F.normalize(text_prototypes, dim=-1)

        # 对齐损失 (最大化对应原型的相似度)
        alignment_loss = -torch.sum(visual_norm * text_norm) / self.p

        return alignment_loss
```

#### 3.2 使用 LLM 生成原型描述

**创建脚本**: `scripts/generate_prototype_descriptions.py`

```python
import openai
import json

def generate_prototype_descriptions(
    n_proto=16,
    cancer_type="lung adenocarcinoma",
    api_key=None
):
    """使用 GPT-4 生成病理学原型描述"""

    client = openai.OpenAI(api_key=api_key)

    prompt = f"""
    As a pathologist, describe {n_proto} distinct histological patterns
    commonly observed in {cancer_type} whole slide images.

    For each pattern, provide:
    1. A concise name (3-5 words)
    2. Key visual features
    3. Clinical significance

    Format the output as a JSON list with keys: "name", "features", "significance"
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    descriptions = json.loads(response.choices[0].message.content)

    return descriptions

# 使用示例
if __name__ == '__main__':
    descriptions = generate_prototype_descriptions(
        n_proto=16,
        cancer_type="lung adenocarcinoma",
        api_key="your-api-key"
    )

    # 保存
    with open('configs/prototype_descriptions.json', 'w') as f:
        json.dump(descriptions, f, indent=2)

    print("生成的原型描述:")
    for i, desc in enumerate(descriptions):
        print(f"{i+1}. {desc['name']}")
        print(f"   特征: {desc['features']}")
        print(f"   意义: {desc['significance']}\n")
```

---

## 📊 性能评估

### 评估指标

#### 1. 主任务性能
- **C-index (Concordance Index)**: 生存预测准确率
- **Survival Loss**: 生存分析损失

#### 2. 原型质量
- **原型多样性**: 原型间的平均余弦距离
- **原型使用率**: 每个原型被分配的 patch 数量分布
- **原型稳定性**: 训练过程中原型位置的变化

#### 3. 可解释性
- **原型-病理模式对应**: 人工评估原型是否对应已知病理模式
- **原型-文本对齐度**: (如果使用多模态方法) 视觉-文本相似度

### 可视化工具

**创建脚本**: `scripts/visualize_prototypes.py`

```python
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_prototype_evolution(checkpoint_paths, save_path='proto_evolution.png'):
    """可视化原型在训练过程中的演化"""

    fig, axes = plt.subplots(1, len(checkpoint_paths), figsize=(5*len(checkpoint_paths), 5))

    for i, ckpt_path in enumerate(checkpoint_paths):
        # 加载检查点
        checkpoint = torch.load(ckpt_path)
        prototypes = checkpoint['model_state_dict']['panther.m'].numpy()  # (p, d)

        # t-SNE 降维
        tsne = TSNE(n_components=2, random_state=42)
        proto_2d = tsne.fit_transform(prototypes)

        # 绘制
        ax = axes[i] if len(checkpoint_paths) > 1 else axes
        ax.scatter(proto_2d[:, 0], proto_2d[:, 1], s=100, alpha=0.6)

        # 标注原型编号
        for j, (x, y) in enumerate(proto_2d):
            ax.annotate(f'P{j}', (x, y), fontsize=8)

        ax.set_title(f'Epoch {checkpoint["epoch"]}')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f'保存可视化到 {save_path}')


def analyze_prototype_usage(model, data_loader, save_path='proto_usage.png'):
    """分析原型使用情况"""

    model.eval()
    proto_counts = torch.zeros(model.panther.p)

    with torch.no_grad():
        for batch in data_loader:
            features = batch['features'].cuda()

            # 计算分配
            pi, mu, Sigma, qq = model.panther.map_em(features)

            # 统计每个原型的使用次数 (软分配)
            proto_counts += qq.sum(dim=(0, 1)).cpu()

    # 归一化
    proto_usage = proto_counts / proto_counts.sum()

    # 绘制
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(proto_usage)), proto_usage.numpy())
    plt.xlabel('原型编号')
    plt.ylabel('使用频率')
    plt.title('原型使用情况分布')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f'保存原型使用情况到 {save_path}')

    # 打印统计
    print(f"\n原型使用统计:")
    print(f"  平均使用率: {proto_usage.mean().item():.4f}")
    print(f"  标准差: {proto_usage.std().item():.4f}")
    print(f"  最大使用率: {proto_usage.max().item():.4f} (原型 {proto_usage.argmax().item()})")
    print(f"  最小使用率: {proto_usage.min().item():.4f} (原型 {proto_usage.argmin().item()})")


# 使用示例
if __name__ == '__main__':
    # 可视化原型演化
    checkpoint_paths = [
        'checkpoints/epoch_10.pth',
        'checkpoints/epoch_20.pth',
        'checkpoints/epoch_30.pth',
        'checkpoints/epoch_40.pth',
    ]
    visualize_prototype_evolution(checkpoint_paths)

    # 分析原型使用
    model = torch.load('checkpoints/best.pth')['model']
    # data_loader = ...
    # analyze_prototype_usage(model, data_loader)
```

---

## 🐛 故障排查

### 问题 1: 原型崩溃 (所有原型收敛到相似表示)

**症状**:
- 原型间余弦相似度 > 0.9
- 所有原型的 t-SNE 可视化聚集在一起

**解决方案**:
1. **增加多样性正则化权重**:
   ```yaml
   diversity_reg: 0.05  # 从 0.01 增加到 0.05
   ```

2. **使用更小的原型学习率**:
   ```yaml
   proto_lr: 1e-5  # 从 5e-5 降低到 1e-5
   ```

3. **添加原型初始化约束**:
   ```python
   # 定期重置不活跃的原型
   def reset_inactive_prototypes(model, usage_threshold=0.01):
       """重置使用率低的原型"""
       proto_usage = compute_prototype_usage(model, data_loader)

       for i, usage in enumerate(proto_usage):
           if usage < usage_threshold:
               # 从数据中重新采样
               model.panther.m[i] = sample_from_data()
               print(f'重置原型 {i} (使用率: {usage:.4f})')
   ```

---

### 问题 2: 训练不稳定 (损失震荡)

**症状**:
- 损失曲线剧烈震荡
- 梯度范数异常大

**解决方案**:
1. **添加梯度裁剪**:
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

2. **使用更保守的学习率调度**:
   ```python
   scheduler = optim.lr_scheduler.ReduceLROnPlateau(
       optimizer,
       mode='max',
       factor=0.5,
       patience=5,
       verbose=True
   )
   ```

3. **增加 EM 正则化强度**:
   ```yaml
   tau: 0.01  # 从 0.001 增加到 0.01
   ```

---

### 问题 3: 性能提升不明显

**症状**:
- 端到端训练后 C-index 提升 < 1%

**诊断步骤**:
1. **检查原型是否真的在学习**:
   ```python
   # 打印原型梯度
   print(f'原型梯度范数: {model.panther.m.grad.norm().item()}')

   # 比较训练前后的原型
   proto_before = model.panther.m.clone()
   # ... 训练 ...
   proto_after = model.panther.m
   proto_change = (proto_after - proto_before).norm()
   print(f'原型变化量: {proto_change.item()}')
   ```

2. **检查原型使用是否均衡**:
   ```python
   analyze_prototype_usage(model, val_loader)
   ```

3. **尝试不同的超参数**:
   ```yaml
   # 实验配置 A: 激进学习
   proto_lr: 1e-4
   diversity_reg: 0.02
   em_iter: 7

   # 实验配置 B: 保守学习
   proto_lr: 1e-5
   diversity_reg: 0.001
   em_iter: 3
   ```

---

## 🎯 高级选项

### 选项 1: 层次化原型 (Hierarchical Prototypes)

为不同层次的特征学习不同原型:

```python
class HierarchicalPrototypeNet(nn.Module):
    def __init__(self, d, p_global=16, p_local=4):
        super().__init__()

        # 全局原型 (整个 WSI)
        self.global_prototypes = ImprovedDirNIWNet(p_global, d)

        # 局部原型 (每个全局原型内部)
        self.local_prototypes = nn.ModuleList([
            ImprovedDirNIWNet(p_local, d)
            for _ in range(p_global)
        ])

    def forward(self, patches):
        # 全局分配
        _, _, _, qq_global = self.global_prototypes.map_em(patches)

        # 局部分配
        qq_local_list = []
        for i, local_proto_net in enumerate(self.local_prototypes):
            # 选择分配到全局原型 i 的 patches
            mask = qq_global[:, :, i] > 0.1  # 阈值过滤
            local_patches = patches[mask]

            if local_patches.size(0) > 0:
                _, _, _, qq_local = local_proto_net.map_em(local_patches.unsqueeze(0))
                qq_local_list.append(qq_local)

        return qq_global, qq_local_list
```

---

### 选项 2: 动态原型数量

根据数据自动调整原型数量:

```python
class AdaptivePrototypeNet(nn.Module):
    def __init__(self, d, p_init=16, p_max=32):
        super().__init__()

        self.d = d
        self.p = p_init
        self.p_max = p_max

        # 初始原型
        self.prototypes = nn.Parameter(torch.randn(p_init, d))

    def split_prototype(self, proto_idx):
        """分裂一个原型为两个"""
        if self.p >= self.p_max:
            return

        # 获取原型
        proto = self.prototypes[proto_idx]

        # 添加小扰动创建两个新原型
        new_proto1 = proto + 0.1 * torch.randn_like(proto)
        new_proto2 = proto - 0.1 * torch.randn_like(proto)

        # 替换原原型并添加新原型
        new_prototypes = torch.cat([
            self.prototypes[:proto_idx],
            new_proto1.unsqueeze(0),
            new_proto2.unsqueeze(0),
            self.prototypes[proto_idx+1:]
        ], dim=0)

        self.prototypes = nn.Parameter(new_prototypes)
        self.p += 1

    def merge_prototypes(self, proto_idx1, proto_idx2):
        """合并两个原型"""
        if self.p <= 2:
            return

        # 平均两个原型
        merged_proto = (
            self.prototypes[proto_idx1] +
            self.prototypes[proto_idx2]
        ) / 2

        # 删除两个原型，添加合并后的原型
        keep_mask = torch.ones(self.p, dtype=bool)
        keep_mask[proto_idx1] = False
        keep_mask[proto_idx2] = False

        new_prototypes = torch.cat([
            self.prototypes[keep_mask],
            merged_proto.unsqueeze(0)
        ], dim=0)

        self.prototypes = nn.Parameter(new_prototypes)
        self.p -= 1
```

---

## 📚 参考文献

### 端到端原型学习
1. Snell, J., et al. (2017). "Prototypical Networks for Few-shot Learning". NeurIPS.
2. Li, W., et al. (2018). "This Looks Like That: Deep Learning for Interpretable Image Recognition". NeurIPS.

### 因果表示学习
3. Arjovsky, M., et al. (2019). "Invariant Risk Minimization". arXiv.
4. Schölkopf, B., et al. (2021). "Toward Causal Representation Learning". Proceedings of the IEEE.

### 多模态病理学
5. Huang, Z., et al. (2023). "PLIP: Pathology Language-Image Pre-training". Nature Medicine.
6. Lu, M., et al. (2024). "Visual-Language Foundation Model for Pathology". arXiv.

### EM 算法和可微分优化
7. Kim, M., et al. (2022). "Differentiable Expectation-Maximization for Set Representation Learning". ICLR.

---

## 🎓 总结

### 推荐实施路线图

**第 1 周**: 基础改进
- ✅ 设置 `fix_proto=False`
- ✅ 增加 `em_iter=5`
- ✅ 添加多样性正则化
- ✅ 实现可视化工具

**第 2-3 周**: 性能优化
- ✅ 调优超参数 (学习率、正则化权重)
- ✅ 实验不同优化器配置
- ✅ 分析原型使用情况

**第 4-6 周**: 高级方法 (可选)
- ✅ 实现因果推理模块
- ✅ 集成文本编码器
- ✅ 尝试层次化原型

### 预期性能提升

| 方法 | C-Index 提升 | 实现难度 | 时间成本 |
|------|------------|---------|---------|
| 端到端基础 (fix_proto=False) | +2-5% | ⭐ 低 | 1 周 |
| + 多样性正则化 | +1-3% | ⭐⭐ 中 | 1 周 |
| + 因果推理 | +2-4% | ⭐⭐⭐ 高 | 2-3 周 |
| + 多模态对齐 | +3-6% | ⭐⭐⭐⭐ 很高 | 1-2 月 |

**总体预期**: C-index +5-15%

---

## 📞 支持

如有问题,请查看:
- 详细方案文档: `docs/prototype_improvement_proposals.md`
- 代码实现: `src/mil_models/PANTHER/networks_improved.py`
- 训练脚本: `src/training/train_improved_prototype.py`

祝实验顺利! 🚀
