"""
改进版 PANTHER 网络 - 端到端可训练原型
Improved PANTHER Networks with End-to-End Trainable Prototypes

主要改进:
1. 可学习的原型参数 (fix_proto=False)
2. 原型多样性正则化 (防止原型崩溃)
3. Gumbel-Softmax 重参数化 (增强可微分性)
4. 可选的完整协方差矩阵

使用示例:
    model = ImprovedDirNIWNet(
        p=16,
        d=1024,
        fix_proto=False,  # 关键：启用原型学习
        diversity_reg=0.01,
        use_gumbel=True
    )
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mog_eval(mog, data):
    """
    评估高斯混合模型的对数似然

    Args:
        mog: tuple of (pi, mu, Sigma)
            pi: (B, p) - 混合系数
            mu: (B, p, d) - 均值
            Sigma: (B, p, d) - 对角协方差
        data: (B, N, d) - 输入数据

    Returns:
        jll: (B, N, p) - 联合对数似然
        cll: (B, N, p) - 条件对数似然 (后验概率的对数)
        mll: (B, N) - 边际对数似然
    """
    B, N, d = data.shape
    pi, mu, Sigma = mog

    # 联合对数似然: log p(x, z) = log p(x|z) + log p(z)
    # log p(x|z) = log N(x; mu_z, Sigma_z)
    jll = -0.5 * (
        d * np.log(2*np.pi) +
        Sigma.log().sum(-1).unsqueeze(1) +  # log det(Sigma)
        torch.bmm(data**2, 1./Sigma.permute(0,2,1)) +  # x^T Sigma^{-1} x
        ((mu**2) / Sigma).sum(-1).unsqueeze(1) +  # mu^T Sigma^{-1} mu
        -2. * torch.bmm(data, (mu/Sigma).permute(0,2,1))  # -2 x^T Sigma^{-1} mu
    ) + pi.log().unsqueeze(1)  # + log p(z)

    # 边际对数似然: log p(x) = log sum_z p(x, z)
    mll = jll.logsumexp(-1)

    # 条件对数似然: log p(z|x) = log p(x, z) - log p(x)
    cll = jll - mll.unsqueeze(-1)

    return jll, cll, mll


class ImprovedDirNIWNet(nn.Module):
    """
    改进的 Dirichlet-Normal-Inverse-Wishart 网络

    改进点:
    1. 原型参数可学习 (requires_grad=True)
    2. 多样性正则化 (防止原型崩溃)
    3. Gumbel-Softmax 重参数化
    4. 可选的完整协方差矩阵
    """

    def __init__(
        self,
        p,
        d,
        eps=0.1,
        load_proto=True,
        proto_path=None,
        fix_proto=False,  # 默认改为可训练
        diversity_reg=0.01,  # 多样性正则化权重
        use_gumbel=False,  # 是否使用 Gumbel-Softmax
        gumbel_temp=1.0,  # Gumbel 温度
        cov_type='diag',  # 协方差类型: 'diag', 'spherical', 'full'
    ):
        super(ImprovedDirNIWNet, self).__init__()

        self.p = p  # 原型数量
        self.d = d  # 特征维度
        self.eps = eps
        self.load_proto = load_proto
        self.proto_path = proto_path
        self.fix_proto = fix_proto
        self.diversity_reg = diversity_reg
        self.use_gumbel = use_gumbel
        self.gumbel_temp = gumbel_temp
        self.cov_type = cov_type

        # 初始化原型均值
        if self.load_proto and proto_path is not None:
            print(f'Loading prototypes from {proto_path}')
            weights = np.load(proto_path)
            print(f'Prototype shape: {weights.shape}')
            assert weights.shape[1] == p and weights.shape[2] == d, \
                f"Expected shape (1, {p}, {d}), got {weights.shape}"

            # 原型均值 (可学习)
            self.m = nn.Parameter(
                torch.from_numpy(weights).squeeze(0).float(),
                requires_grad=not fix_proto  # 关键修改
            )
        else:
            print('Initializing prototypes randomly')
            self.m = nn.Parameter(
                0.1 * torch.randn(p, d),
                requires_grad=not fix_proto
            )

        # 初始化协方差参数
        if cov_type == 'diag':
            # 对角协方差矩阵
            self.V_ = nn.Parameter(
                torch.randn(p, d),
                requires_grad=not fix_proto
            )
        elif cov_type == 'spherical':
            # 球形协方差 (单一参数)
            self.V_ = nn.Parameter(
                torch.randn(p, 1),
                requires_grad=not fix_proto
            )
        elif cov_type == 'full':
            # 完整协方差矩阵 (使用 Cholesky 分解参数化)
            # Sigma = LL^T, 保证正定性
            self.L_diag = nn.Parameter(
                torch.ones(p, d),  # 对角元素 (必须为正)
                requires_grad=not fix_proto
            )
            self.L_lower = nn.Parameter(
                torch.zeros(p, d * (d - 1) // 2),  # 下三角元素
                requires_grad=not fix_proto
            )
        else:
            raise ValueError(f"Unknown covariance type: {cov_type}")

        if not fix_proto:
            print(f"原型参数可训练 (requires_grad=True)")
        else:
            print(f"原型参数固定 (requires_grad=False)")

    def get_covariance(self):
        """获取协方差矩阵"""
        if self.cov_type == 'diag':
            # 对角协方差
            return self.eps * F.softplus(self.V_)

        elif self.cov_type == 'spherical':
            # 球形协方差
            V_scalar = self.eps * F.softplus(self.V_)
            return V_scalar.expand(self.p, self.d)

        elif self.cov_type == 'full':
            # 完整协方差矩阵
            # 构造下三角矩阵 L
            L = torch.zeros(self.p, self.d, self.d).to(self.L_diag.device)

            # 填充对角元素 (使用 softplus 保证为正)
            diag_indices = torch.arange(self.d)
            L[:, diag_indices, diag_indices] = F.softplus(self.L_diag)

            # 填充下三角元素
            tril_indices = torch.tril_indices(self.d, self.d, offset=-1)
            L[:, tril_indices[0], tril_indices[1]] = self.L_lower

            # Sigma = LL^T
            Sigma = torch.bmm(L, L.transpose(1, 2))  # (p, d, d)

            # 返回对角元素 (用于当前的 GMM 评估函数)
            return torch.diagonal(Sigma, dim1=1, dim2=2)  # (p, d)

    def forward(self):
        """返回原型的先验分布"""
        V = self.get_covariance()
        return self.m, V

    def mode(self, prior=None):
        """
        返回先验分布的众数

        Returns:
            pi: (p,) - 混合系数
            mu: (p, d) - 均值
            Sigma: (p, d) - 协方差
        """
        if prior is None:
            m, V = self.forward()
        else:
            m, V = prior

        # 均匀混合系数
        pi = torch.ones(self.p).to(m.device) / self.p
        mu = m
        Sigma = V

        return pi, mu, Sigma

    def compute_diversity_loss(self):
        """
        计算原型多样性损失 (防止原型崩溃到相似表示)

        Returns:
            diversity_loss: scalar - 原型相似度的平均值 (越小越好)
        """
        # 归一化原型
        m_norm = F.normalize(self.m, dim=-1)  # (p, d)

        # 计算原型间的余弦相似度矩阵
        similarity_matrix = torch.matmul(m_norm, m_norm.t())  # (p, p)

        # 只计算非对角元素 (不同原型之间的相似度)
        mask = ~torch.eye(self.p, dtype=bool, device=self.m.device)
        off_diagonal_sim = similarity_matrix[mask]

        # 多样性损失 = 平均相似度
        # 我们希望最小化这个值，即最大化原型间的差异
        diversity_loss = off_diagonal_sim.abs().mean()

        return diversity_loss

    def map_m_step(self, data, weight, tau=1.0, prior=None):
        """
        MAP (最大后验) M-step 估计

        Args:
            data: (B, N, d) - 输入数据
            weight: (B, N, p) - 后验权重 (E-step 的输出)
            tau: float - 正则化强度 (先验权重)
            prior: tuple - (m, V) 先验参数

        Returns:
            pi: (B, p) - 更新后的混合系数
            mu: (B, p, d) - 更新后的均值
            Sigma: (B, p, d) - 更新后的协方差
        """
        B, N, d = data.shape

        if prior is None:
            m, V = self.forward()
        else:
            m, V = prior

        # 每个原型的总权重
        wsum = weight.sum(1)  # (B, p)

        # 正则化 (加入先验强度)
        wsum_reg = wsum + tau

        # 加权数据和
        wxsum = torch.bmm(weight.permute(0, 2, 1), data)  # (B, p, d)

        # 加权平方数据和
        wxxsum = torch.bmm(weight.permute(0, 2, 1), data**2)  # (B, p, d)

        # 更新混合系数
        pi = wsum_reg / wsum_reg.sum(1, keepdim=True)  # (B, p)

        # 更新均值 (带先验正则化)
        mu = (wxsum + m.unsqueeze(0) * tau) / wsum_reg.unsqueeze(-1)  # (B, p, d)

        # 更新协方差 (带先验正则化)
        Sigma = (wxxsum + (V + m**2).unsqueeze(0) * tau) / wsum_reg.unsqueeze(-1) - mu**2  # (B, p, d)

        # 确保协方差为正
        Sigma = Sigma.clamp(min=1e-6)

        return pi, mu, Sigma

    def gumbel_softmax_em(self, data, mask=None, num_iters=3, tau=1.0,
                         hard=False, prior=None):
        """
        使用 Gumbel-Softmax 重参数化的 EM 算法

        Args:
            data: (B, N, d) - 输入数据
            mask: (B, N) - 数据掩码
            num_iters: int - EM 迭代次数
            tau: float - M-step 正则化强度
            hard: bool - 是否使用硬分配 (Straight-through estimator)
            prior: tuple - (m, V) 先验参数

        Returns:
            pi: (B, p) - 混合系数
            mu: (B, p, d) - 均值
            Sigma: (B, p, d) - 协方差
            qq: (B, N, p) - 后验概率 (软分配)
        """
        B, N, d = data.shape

        if mask is None:
            mask = torch.ones(B, N).to(data.device)

        # 从先验初始化
        pi, mu, Sigma = self.mode(prior)
        pi = pi.unsqueeze(0).repeat(B, 1)        # (B, p)
        mu = mu.unsqueeze(0).repeat(B, 1, 1)     # (B, p, d)
        Sigma = Sigma.unsqueeze(0).repeat(B, 1, 1)  # (B, p, d)

        # EM 迭代
        for emiter in range(num_iters):
            # E-step: 计算后验概率
            _, cll, _ = mog_eval((pi, mu, Sigma), data)  # cll: (B, N, p)

            # Gumbel-Softmax 重参数化
            # 添加 Gumbel 噪声使采样可微分
            gumbel_noise = -torch.log(
                -torch.log(torch.rand_like(cll) + 1e-10) + 1e-10
            )

            # 温度缩放的 softmax
            qq_soft = F.softmax((cll + gumbel_noise) / self.gumbel_temp, dim=-1)

            # 应用掩码
            qq_soft = qq_soft * mask.unsqueeze(-1)

            if hard:
                # Straight-through estimator: 前向使用硬分配，反向使用软梯度
                qq_hard = F.one_hot(
                    qq_soft.argmax(-1),
                    num_classes=self.p
                ).float()
                qq = (qq_hard - qq_soft).detach() + qq_soft
            else:
                qq = qq_soft

            # M-step: 更新参数
            pi, mu, Sigma = self.map_m_step(data, weight=qq, tau=tau, prior=prior)

        return pi, mu, Sigma, qq

    def map_em(self, data, mask=None, num_iters=3, tau=1.0, prior=None):
        """
        标准 MAP-EM 算法

        Args:
            data: (B, N, d) - 输入数据
            mask: (B, N) - 数据掩码
            num_iters: int - EM 迭代次数
            tau: float - 正则化强度
            prior: tuple - (m, V) 先验参数

        Returns:
            pi: (B, p) - 混合系数
            mu: (B, p, d) - 均值
            Sigma: (B, p, d) - 协方差
            qq: (B, N, p) - 后验概率
        """
        B, N, d = data.shape

        if mask is None:
            mask = torch.ones(B, N).to(data.device)

        # 如果使用 Gumbel-Softmax，调用对应函数
        if self.use_gumbel:
            return self.gumbel_softmax_em(
                data, mask, num_iters, tau, hard=False, prior=prior
            )

        # 从先验初始化
        pi, mu, Sigma = self.mode(prior)
        pi = pi.unsqueeze(0).repeat(B, 1)        # (B, p)
        mu = mu.unsqueeze(0).repeat(B, 1, 1)     # (B, p, d)
        Sigma = Sigma.unsqueeze(0).repeat(B, 1, 1)  # (B, p, d)

        # EM 迭代
        for emiter in range(num_iters):
            # E-step: 计算后验概率
            _, cll, _ = mog_eval((pi, mu, Sigma), data)
            qq = cll.exp() * mask.unsqueeze(-1)  # (B, N, p)

            # M-step: 更新参数
            pi, mu, Sigma = self.map_m_step(data, weight=qq, tau=tau, prior=prior)

        return pi, mu, Sigma, qq


# 兼容性：保持原始类名
DirNIWNet = ImprovedDirNIWNet


if __name__ == '__main__':
    """测试改进的原型网络"""

    # 设置参数
    batch_size = 4
    n_patches = 1000
    feature_dim = 1024
    n_prototypes = 16

    # 创建随机数据
    data = torch.randn(batch_size, n_patches, feature_dim)
    mask = torch.ones(batch_size, n_patches)

    print("=" * 80)
    print("测试改进的原型网络")
    print("=" * 80)

    # 测试 1: 固定原型 (原始行为)
    print("\n[测试 1] 固定原型 (fix_proto=True)")
    model_fixed = ImprovedDirNIWNet(
        p=n_prototypes,
        d=feature_dim,
        fix_proto=True,
        load_proto=False
    )

    print(f"原型参数 requires_grad: {model_fixed.m.requires_grad}")
    pi, mu, Sigma, qq = model_fixed.map_em(data, mask, num_iters=3, tau=0.001)
    print(f"输出形状 - pi: {pi.shape}, mu: {mu.shape}, Sigma: {Sigma.shape}, qq: {qq.shape}")

    # 测试 2: 可训练原型
    print("\n[测试 2] 可训练原型 (fix_proto=False)")
    model_trainable = ImprovedDirNIWNet(
        p=n_prototypes,
        d=feature_dim,
        fix_proto=False,
        load_proto=False,
        diversity_reg=0.01
    )

    print(f"原型参数 requires_grad: {model_trainable.m.requires_grad}")
    pi, mu, Sigma, qq = model_trainable.map_em(data, mask, num_iters=5, tau=0.001)

    # 计算多样性损失
    diversity_loss = model_trainable.compute_diversity_loss()
    print(f"原型多样性损失: {diversity_loss.item():.4f}")

    # 测试反向传播
    loss = qq.sum() + 0.01 * diversity_loss
    loss.backward()
    print(f"原型梯度范数: {model_trainable.m.grad.norm().item():.4f}")

    # 测试 3: Gumbel-Softmax
    print("\n[测试 3] Gumbel-Softmax 重参数化 (use_gumbel=True)")
    model_gumbel = ImprovedDirNIWNet(
        p=n_prototypes,
        d=feature_dim,
        fix_proto=False,
        load_proto=False,
        use_gumbel=True,
        gumbel_temp=0.5
    )

    pi_g, mu_g, Sigma_g, qq_g = model_gumbel.map_em(data, mask, num_iters=5, tau=0.001)
    print(f"Gumbel 分配 (前5个patch): \n{qq_g[0, :5, :]}")

    # 测试 4: 不同协方差类型
    print("\n[测试 4] 不同协方差类型")

    for cov_type in ['diag', 'spherical']:
        model_cov = ImprovedDirNIWNet(
            p=n_prototypes,
            d=feature_dim,
            fix_proto=False,
            load_proto=False,
            cov_type=cov_type
        )

        V = model_cov.get_covariance()
        print(f"协方差类型: {cov_type}, 形状: {V.shape}")

    print("\n" + "=" * 80)
    print("所有测试完成!")
    print("=" * 80)
