# 原型聚类改进方案（Prototype Clustering Improvement Proposals）

## 📋 当前问题总结

### 1. 现状分析

当前实现基于 PANTHER (ICLR 2022)，主要流程：
- K-means 初始化原型 → 固定原型参数 → 每次 forward 运行 3 轮 EM → 原型不更新
- **关键问题**：原型与下游任务目标（生存预测）缺乏梯度连接

### 2. 性能瓶颈

| 问题 | 影响 | 严重程度 |
|------|------|----------|
| 原型固定不可训练 (`fix_proto=True`) | 无法学习任务相关的原型表示 | 🔴 高 |
| EM 迭代次数少 (3次) | patch-prototype 分配不充分 | 🟡 中 |
| 非端到端学习 | 原型与任务目标脱节 | 🔴 高 |
| 对角协方差矩阵假设 | 无法捕捉特征间相关性 | 🟡 中 |

---

## 🎯 改进方案

## 方案一：端到端可微分原型学习 (End-to-End Differentiable Prototypes)

### 核心思想
让原型参数通过反向传播学习，同时增强 EM 算法的可微分性。

### 实现策略

#### 1.1 启用原型梯度更新
```python
# 修改 /src/mil_models/model_configs.py
@dataclass
class PANTHERConfig(PretrainedConfig):
    fix_proto: bool = False  # 改为 False，允许原型学习
    em_iter: int = 5         # 增加 EM 迭代次数
    proto_lr: float = 1e-4   # 原型专用学习率（可选）
```

#### 1.2 可学习的协方差矩阵
```python
class DirNIWNet(nn.Module):
    def __init__(self, p, d, eps=0.1, cov_type='diag', ...):
        self.cov_type = cov_type

        if cov_type == 'diag':
            # 当前实现：对角协方差
            self.V_ = nn.Parameter(torch.randn(p, d), requires_grad=True)

        elif cov_type == 'full':
            # 完整协方差矩阵 (需要保证正定性)
            # 使用 Cholesky 分解参数化: Sigma = LL^T
            self.L_diag = nn.Parameter(torch.ones(p, d))      # 对角元素
            self.L_lower = nn.Parameter(torch.zeros(p, d*(d-1)//2))  # 下三角

        elif cov_type == 'spherical':
            # 球形协方差 (单一方差参数)
            self.V_ = nn.Parameter(torch.randn(p, 1), requires_grad=True)

    def get_covariance(self):
        if self.cov_type == 'full':
            # 构造正定协方差矩阵
            L = self.construct_lower_triangular()
            Sigma = torch.bmm(L, L.transpose(1, 2))
            return Sigma
        else:
            return self.eps * F.softplus(self.V_)
```

#### 1.3 Gumbel-Softmax 软分配增强
为了增强 EM 的可微分性，使用 Gumbel-Softmax 重参数化：

```python
def gumbel_softmax_em(self, data, temperature=1.0, hard=False):
    """使用 Gumbel-Softmax 的可微分 EM"""
    B, N, d = data.shape

    # E-step: 计算后验概率
    _, qq, _ = mog_eval((pi, mu, Sigma), data)

    # 添加 Gumbel 噪声使采样可微分
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(qq) + 1e-10) + 1e-10)
    qq_gumbel = F.softmax((qq + gumbel_noise) / temperature, dim=-1)

    if hard:
        # Straight-through estimator
        qq_hard = F.one_hot(qq_gumbel.argmax(-1), num_classes=self.p).float()
        qq = (qq_hard - qq_gumbel).detach() + qq_gumbel  # 前向硬分配，反向软梯度

    # M-step: 使用软分配更新参数
    pi, mu, Sigma = self.map_m_step(data, weight=qq, tau=self.tau, prior=prior)

    return pi, mu, Sigma, qq
```

### 优点
- ✅ 原型可以学习任务相关特征
- ✅ 实现简单，只需修改配置和少量代码
- ✅ 保持 PANTHER 框架完整性

### 缺点
- ⚠️ 训练可能不稳定（原型可能崩溃到相似表示）
- ⚠️ 需要正则化防止过拟合

---

## 方案二：基于因果推理的原型分配 (Causal Prototype Assignment)

### 核心思想
使用因果推断框架建模 patch → prototype → outcome 的因果关系，学习具有因果解释性的原型。

### 理论基础

#### 2.1 结构因果模型 (SCM)
```
Confounders (C): WSI级别特征（组织类型、染色质量等）
    ↓
Patch Features (X) → Prototype Assignment (Z) → Survival Outcome (Y)
```

#### 2.2 反事实推理目标
学习满足以下性质的原型：
- **独立性**: P(Z|do(X)) - 消除混淆因子影响
- **可解释性**: 每个原型对应特定的生物学模式
- **因果效应**: 原型对生存结果有显著因果作用

### 实现策略

#### 2.1 对抗性去混淆 (Adversarial Deconfounding)
```python
class CausalPrototypeNet(nn.Module):
    def __init__(self, d, p, confounder_dim=128):
        super().__init__()

        # 混淆因子编码器 (从 WSI 级别特征提取)
        self.confounder_encoder = nn.Sequential(
            nn.Linear(d, confounder_dim),
            nn.ReLU(),
            nn.Linear(confounder_dim, confounder_dim)
        )

        # 原型网络（去混淆）
        self.prototype_net = DirNIWNet(p, d)

        # 混淆因子判别器 (对抗训练)
        self.confounder_discriminator = nn.Sequential(
            nn.Linear(p, 64),
            nn.ReLU(),
            nn.Linear(64, confounder_dim)
        )

    def forward(self, patches, wsi_features):
        B, N, d = patches.shape

        # 编码混淆因子
        confounders = self.confounder_encoder(wsi_features)  # (B, confounder_dim)

        # 计算原型分配
        pi, mu, Sigma = self.prototype_net.mode()
        _, qq, _ = mog_eval((pi, mu, Sigma), patches)
        proto_assignment = qq.exp()  # (B, N, p)

        # 聚合为 WSI 级别的原型表示
        proto_repr = proto_assignment.mean(1)  # (B, p)

        # 对抗训练：原型表示应该独立于混淆因子
        pred_confounders = self.confounder_discriminator(proto_repr)

        return proto_assignment, confounders, pred_confounders

    def compute_loss(self, proto_assignment, confounders, pred_confounders,
                     survival_pred, survival_label, lambda_adv=0.1):
        # 主任务损失 (生存预测)
        survival_loss = self.survival_criterion(survival_pred, survival_label)

        # 对抗损失 (最大化混淆因子预测误差)
        # 原型网络试图使原型表示与混淆因子无关
        adv_loss = -F.mse_loss(pred_confounders, confounders.detach())

        total_loss = survival_loss + lambda_adv * adv_loss

        return total_loss
```

训练流程：
```python
# 判别器训练
confounder_loss = F.mse_loss(pred_confounders, confounders)
confounder_loss.backward()
discriminator_optimizer.step()

# 原型网络训练（对抗）
total_loss = prototype_net.compute_loss(...)
total_loss.backward()
prototype_optimizer.step()
```

#### 2.2 因果介入增强 (Causal Intervention)
使用 do-calculus 实现原型分配的因果效应估计：

```python
def causal_intervention_loss(self, patches, prototypes, outcome):
    """
    估计原型对结果的平均因果效应 (ACE):
    ACE(z) = E[Y|do(Z=z)] - E[Y|do(Z=z')]
    """
    B, N, d = patches.shape
    p = prototypes.shape[1]

    # 计算每个原型的因果效应
    causal_effects = []

    for k in range(p):
        # 介入：强制分配到原型 k
        assignment_k = torch.zeros(B, N, p).to(patches.device)
        assignment_k[:, :, k] = 1.0

        # 计算介入后的预测
        proto_repr_k = self.aggregate_prototypes(patches, assignment_k, prototypes)
        outcome_k = self.survival_predictor(proto_repr_k)

        causal_effects.append(outcome_k)

    causal_effects = torch.stack(causal_effects, dim=1)  # (B, p)

    # 最大化原型间的因果效应差异（增强可解释性）
    diversity_loss = -torch.std(causal_effects, dim=1).mean()

    return diversity_loss
```

#### 2.3 反事实正则化
```python
def counterfactual_regularization(self, patches, prototypes):
    """
    反事实正则化：如果 patch 被分配到不同原型，结果应该不同
    """
    B, N, d = patches.shape

    # 实际分配
    factual_assignment = self.compute_assignment(patches, prototypes)
    factual_repr = self.aggregate_prototypes(patches, factual_assignment, prototypes)
    factual_outcome = self.survival_predictor(factual_repr)

    # 反事实分配（随机扰动）
    counterfactual_assignment = factual_assignment + 0.1 * torch.randn_like(factual_assignment)
    counterfactual_assignment = F.softmax(counterfactual_assignment, dim=-1)

    counterfactual_repr = self.aggregate_prototypes(patches, counterfactual_assignment, prototypes)
    counterfactual_outcome = self.survival_predictor(counterfactual_repr)

    # 分配变化 vs 结果变化的一致性
    assignment_change = F.kl_div(
        factual_assignment.log(),
        counterfactual_assignment,
        reduction='batchmean'
    )
    outcome_change = F.mse_loss(factual_outcome, counterfactual_outcome)

    # 如果分配变化大，结果也应该变化大
    consistency_loss = F.mse_loss(
        assignment_change * 10,  # 缩放到相似量级
        outcome_change
    )

    return consistency_loss
```

### 优点
- ✅ 因果可解释性强
- ✅ 原型具有明确的生物学意义
- ✅ 减少虚假相关性

### 缺点
- ⚠️ 实现复杂度高
- ⚠️ 需要仔细设计混淆因子和因果图
- ⚠️ 计算开销较大

---

## 方案三：基于预训练大模型的原型学习 (LLM-Enhanced Prototypes)

### 核心思想
利用生物医学预训练大模型（如 BioGPT, PubMedBERT, PathCLIP）的先验知识指导原型学习。

### 实现策略

#### 3.1 多模态原型对齐 (Multimodal Prototype Alignment)

使用 CLIP-style 对比学习将视觉原型与文本描述对齐：

```python
class MultimodalPrototypeNet(nn.Module):
    def __init__(self, d, p, text_encoder='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'):
        super().__init__()

        # 视觉编码器 (现有的 patch encoder)
        self.visual_encoder = ...  # 已有的特征提取器

        # 文本编码器 (生物医学 BERT)
        self.text_encoder = AutoModel.from_pretrained(text_encoder)
        self.text_projection = nn.Linear(768, d)  # 投影到视觉空间

        # 可学习的原型
        self.visual_prototypes = nn.Parameter(torch.randn(p, d), requires_grad=True)

        # 原型文本描述 (可以是病理学术语)
        self.prototype_descriptions = [
            "tumor infiltrating lymphocytes in stroma",
            "necrotic tumor tissue with hemorrhage",
            "dense tumor cell clusters with high mitotic activity",
            "normal epithelial tissue with glandular structure",
            # ... 更多描述
        ]

    def forward(self, patches, return_similarity=True):
        B, N, d = patches.shape

        # 编码文本描述
        text_embeddings = []
        for desc in self.prototype_descriptions:
            tokens = self.tokenizer(desc, return_tensors='pt', padding=True).to(patches.device)
            text_feat = self.text_encoder(**tokens).last_hidden_state[:, 0, :]  # [CLS]
            text_feat = self.text_projection(text_feat)
            text_embeddings.append(text_feat)

        text_embeddings = torch.cat(text_embeddings, dim=0)  # (p, d)

        # 对齐损失 (视觉原型 ↔ 文本描述)
        alignment_loss = F.mse_loss(self.visual_prototypes, text_embeddings.detach())

        # 计算 patch 到原型的相似度
        patches_norm = F.normalize(patches, dim=-1)
        prototypes_norm = F.normalize(self.visual_prototypes, dim=-1)

        # 余弦相似度
        similarity = torch.matmul(patches_norm, prototypes_norm.t())  # (B, N, p)
        assignment = F.softmax(similarity / 0.07, dim=-1)  # 温度缩放

        if return_similarity:
            return assignment, alignment_loss, similarity

        return assignment, alignment_loss
```

#### 3.2 基于 LLM 的原型初始化

使用生物医学知识库引导原型初始化：

```python
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMGuidedPrototypeInitializer:
    def __init__(self, n_proto, feature_dim, use_api=False):
        self.n_proto = n_proto
        self.feature_dim = feature_dim

        if use_api:
            # 使用 OpenAI API (GPT-4)
            self.client = openai.OpenAI()
        else:
            # 使用开源生物医学 LLM
            self.model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT")
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT")

    def generate_prototype_descriptions(self, cancer_type="lung cancer"):
        """使用 LLM 生成病理学原型描述"""

        prompt = f"""
        As a pathologist, describe {self.n_proto} distinct histological patterns
        commonly observed in {cancer_type} whole slide images.

        For each pattern, provide:
        1. Brief name (3-5 words)
        2. Key visual features
        3. Clinical significance

        Format as JSON list.
        """

        if hasattr(self, 'client'):
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            descriptions = response.choices[0].message.content
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_length=500)
            descriptions = self.tokenizer.decode(outputs[0])

        return descriptions

    def embed_descriptions(self, descriptions, text_encoder):
        """将文本描述编码为原型初始化向量"""
        embeddings = []

        for desc in descriptions:
            tokens = text_encoder.tokenizer(desc, return_tensors='pt', padding=True)
            embedding = text_encoder(**tokens).last_hidden_state.mean(1)  # 平均池化
            embeddings.append(embedding)

        prototype_init = torch.cat(embeddings, dim=0)  # (n_proto, text_dim)

        # 投影到视觉特征空间
        projection = nn.Linear(prototype_init.shape[1], self.feature_dim)
        visual_prototypes = projection(prototype_init)

        return visual_prototypes
```

使用示例：
```python
# 初始化
initializer = LLMGuidedPrototypeInitializer(n_proto=16, feature_dim=1024)

# 生成病理学描述
descriptions = initializer.generate_prototype_descriptions(cancer_type="lung adenocarcinoma")
# 输出示例:
# [
#   "Tumor infiltrating lymphocytes (TILs) in stromal regions",
#   "Solid tumor nests with central necrosis",
#   "Lepidic growth pattern with preserved alveolar structures",
#   ...
# ]

# 编码为视觉原型
text_encoder = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
prototype_init = initializer.embed_descriptions(descriptions, text_encoder)

# 使用初始化原型
model = MultimodalPrototypeNet(d=1024, p=16)
model.visual_prototypes.data = prototype_init
```

#### 3.3 知识蒸馏增强原型学习

使用预训练基础模型作为教师模型：

```python
class KnowledgeDistillationPrototype(nn.Module):
    def __init__(self, student_proto_net, teacher_model, temperature=2.0):
        super().__init__()
        self.student = student_proto_net
        self.teacher = teacher_model.eval()  # 冻结教师模型
        self.temperature = temperature

    def forward(self, patches):
        # 学生模型预测
        student_assignment, _, _ = self.student(patches)
        student_repr = self.student.aggregate(patches, student_assignment)
        student_logits = self.student.survival_head(student_repr)

        # 教师模型预测
        with torch.no_grad():
            teacher_features = self.teacher.extract_features(patches)
            teacher_logits = self.teacher.survival_head(teacher_features)

        # 知识蒸馏损失
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        return student_logits, kd_loss
```

#### 3.4 基于 RAG 的动态原型更新

使用检索增强生成 (RAG) 动态调整原型：

```python
class RAGPrototypeUpdater:
    def __init__(self, vector_db_path, llm_model):
        # 向量数据库 (存储病理知识)
        self.vector_db = FAISS.load_local(vector_db_path)
        self.llm = llm_model

    def update_prototypes(self, current_prototypes, validation_performance):
        """根据验证集性能动态调整原型"""

        # 1. 识别表现差的原型
        proto_performance = self.analyze_prototype_usage(validation_performance)
        weak_prototypes = [i for i, perf in enumerate(proto_performance) if perf < 0.5]

        # 2. 从知识库检索相关病理模式
        for proto_idx in weak_prototypes:
            query = f"Alternative histological pattern for prototype {proto_idx}"
            relevant_docs = self.vector_db.similarity_search(query, k=3)

            # 3. 使用 LLM 生成改进建议
            context = "\n".join([doc.page_content for doc in relevant_docs])
            prompt = f"""
            Context: {context}

            Current prototype {proto_idx} has low predictive performance.
            Suggest an improved histological pattern to replace it.
            """

            suggestion = self.llm.generate(prompt)

            # 4. 更新原型
            new_embedding = self.embed_suggestion(suggestion)
            current_prototypes[proto_idx] = new_embedding

        return current_prototypes
```

### 优点
- ✅ 利用海量生物医学知识
- ✅ 原型具有语义可解释性
- ✅ 可以生成人类可读的原型描述

### 缺点
- ⚠️ 依赖外部大模型（可能需要 API 费用）
- ⚠️ 文本-视觉对齐需要大量数据
- ⚠️ 实时推理速度较慢

---

## 方案四：混合方案（推荐）

### 核心策略
结合方案一、二、三的优点，分阶段实现：

#### 阶段 1: 端到端基础改进 (短期 - 1周)
- ✅ 设置 `fix_proto=False`
- ✅ 增加 `em_iter=5`
- ✅ 添加原型多样性正则化

```python
def diversity_loss(prototypes):
    """防止原型崩溃到相似表示"""
    proto_norm = F.normalize(prototypes, dim=-1)
    similarity_matrix = torch.matmul(proto_norm, proto_norm.t())

    # 最小化非对角元素（最大化原型间差异）
    mask = ~torch.eye(prototypes.shape[0], dtype=bool).to(prototypes.device)
    diversity = similarity_matrix[mask].abs().mean()

    return diversity
```

#### 阶段 2: 因果增强 (中期 - 2-3周)
- ✅ 实现对抗性去混淆
- ✅ 添加因果效应正则化

#### 阶段 3: 知识增强 (长期 - 1-2月)
- ✅ 集成文本编码器
- ✅ 使用 LLM 生成原型描述
- ✅ 实现知识蒸馏

---

## 📊 预期性能提升

| 方案 | C-Index 提升估计 | 实现复杂度 | 计算开销 | 可解释性 |
|------|----------------|-----------|---------|---------|
| 方案一 (端到端) | +2-5% | ⭐ 低 | +10% | ⭐⭐ 中 |
| 方案二 (因果) | +3-7% | ⭐⭐⭐ 高 | +30% | ⭐⭐⭐ 高 |
| 方案三 (LLM) | +5-10% | ⭐⭐⭐⭐ 很高 | +50% | ⭐⭐⭐⭐ 很高 |
| 方案四 (混合) | +7-12% | ⭐⭐⭐ 高 | +40% | ⭐⭐⭐⭐ 很高 |

注：提升估计基于相关论文和经验，实际效果取决于数据集特性。

---

## 🔧 实现优先级建议

### 立即实施 (高优先级)
1. 启用 `fix_proto=False` 并训练观察效果
2. 增加 EM 迭代次数到 5-7
3. 添加原型多样性正则化

### 中期实施 (中优先级)
4. 实现 Gumbel-Softmax 重参数化
5. 添加对抗性去混淆模块
6. 实现因果效应分析工具

### 长期研究 (探索性)
7. 集成生物医学文本编码器
8. 尝试 RAG 动态原型更新
9. 实现多模态原型对齐

---

## 📚 相关文献

1. **端到端原型学习**:
   - "Prototypical Networks for Few-shot Learning" (Snell et al., NeurIPS 2017)
   - "Deep Embedded Clustering" (Xie et al., ICML 2016)

2. **因果表示学习**:
   - "Invariant Risk Minimization" (Arjovsky et al., 2019)
   - "Causal Representation Learning" (Schölkopf et al., 2021)

3. **多模态病理学**:
   - "PLIP: Pathology Language-Image Pre-training" (Huang et al., Nature Medicine 2023)
   - "PathCLIP: Vision-Language Model for Pathology" (Lu et al., 2024)

4. **知识蒸馏**:
   - "Distilling the Knowledge in a Neural Network" (Hinton et al., 2015)

---

## 🚀 快速开始

想要立即实施改进？运行以下命令启动端到端训练：

```bash
python src/training/main_prototype.py \
    --fix_proto False \
    --em_iter 5 \
    --proto_lr 1e-4 \
    --add_diversity_loss True \
    --diversity_weight 0.01
```

---

**作者**: Claude Code
**日期**: 2025-10-22
**版本**: 1.0
