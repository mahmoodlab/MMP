"""
Prototype Selection Network with Causal-Confounding Separation

This module implements a learnable prototype selection mechanism that separates
prototypes into causal and confounding groups using Gumbel-Softmax trick for
differentiable discrete selection.

Author: Claude
Date: 2025-10-21
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GumbelSoftmax(nn.Module):
    """
    Gumbel-Softmax implementation for differentiable discrete sampling.

    During training, uses Gumbel-Softmax to provide gradients.
    During inference, uses hard selection (argmax).
    """

    def __init__(self, tau=1.0, hard=False):
        """
        Args:
            tau (float): Temperature parameter. Lower values make the distribution sharper.
            hard (bool): If True, uses straight-through estimator (hard sampling with soft gradients).
        """
        super(GumbelSoftmax, self).__init__()
        self.tau = tau
        self.hard = hard

    def forward(self, logits):
        """
        Args:
            logits (torch.Tensor): Unnormalized log probabilities of shape (..., n_classes)

        Returns:
            torch.Tensor: Sampled probabilities of the same shape as logits
        """
        if self.training:
            # Sample from Gumbel distribution
            gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
            gumbels = (logits + gumbels) / self.tau
            y_soft = F.softmax(gumbels, dim=-1)

            if self.hard:
                # Straight through estimator
                index = y_soft.max(dim=-1, keepdim=True)[1]
                y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
                # This line allows gradients to flow through y_soft
                ret = y_hard - y_soft.detach() + y_soft
            else:
                ret = y_soft
        else:
            # During inference, use hard selection
            index = logits.max(dim=-1, keepdim=True)[1]
            ret = torch.zeros_like(logits).scatter_(-1, index, 1.0)

        return ret


class PrototypeSelectionNetwork(nn.Module):
    """
    Prototype Selection Network that learns to select causal vs confounding prototypes.

    Architecture:
        Input: prototype representations (B, n_proto, proto_dim)
        ├─ MLP: projects each prototype to selection logits
        ├─ Gumbel-Softmax: differentiable binary selection (causal vs confounding)
        └─ Output: selection weights ∈ [0,1]^n_proto
    """

    def __init__(self,
                 proto_dim,
                 hidden_dim=256,
                 n_classes=2,  # 2 for binary: causal vs confounding
                 tau=1.0,
                 hard=False,
                 use_gumbel=True):
        """
        Args:
            proto_dim (int): Dimension of each prototype
            hidden_dim (int): Hidden dimension for MLP
            n_classes (int): Number of selection classes (2 for causal/confounding)
            tau (float): Gumbel-Softmax temperature
            hard (bool): Use hard Gumbel-Softmax
            use_gumbel (bool): If False, use standard sigmoid instead
        """
        super(PrototypeSelectionNetwork, self).__init__()

        self.proto_dim = proto_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.use_gumbel = use_gumbel

        # MLP for learning selection logits
        self.selection_mlp = nn.Sequential(
            nn.Linear(proto_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim // 2, n_classes)
        )

        if use_gumbel:
            self.gumbel = GumbelSoftmax(tau=tau, hard=hard)

    def forward(self, proto_repr, return_logits=False):
        """
        Args:
            proto_repr (torch.Tensor): Prototype representations of shape (B, n_proto, proto_dim)
            return_logits (bool): If True, also return the raw logits

        Returns:
            dict: {
                'selection_probs': (B, n_proto, n_classes) - Selection probabilities
                'causal_mask': (B, n_proto) - Binary mask for causal prototypes
                'confound_mask': (B, n_proto) - Binary mask for confounding prototypes
                'logits': (B, n_proto, n_classes) - Raw logits (if return_logits=True)
            }
        """
        B, n_proto, proto_dim = proto_repr.shape

        # Compute selection logits for each prototype
        # Shape: (B, n_proto, n_classes)
        logits = self.selection_mlp(proto_repr)

        if self.use_gumbel:
            # Apply Gumbel-Softmax for differentiable selection
            selection_probs = self.gumbel(logits)
        else:
            # Use standard softmax
            selection_probs = F.softmax(logits, dim=-1)

        # Extract causal and confounding masks
        # Assuming class 0 = confounding, class 1 = causal
        confound_mask = selection_probs[:, :, 0]  # (B, n_proto)
        causal_mask = selection_probs[:, :, 1]    # (B, n_proto)

        result = {
            'selection_probs': selection_probs,
            'causal_mask': causal_mask,
            'confound_mask': confound_mask,
        }

        if return_logits:
            result['logits'] = logits

        return result


class TopKPrototypeSelector(nn.Module):
    """
    Alternative to Gumbel-Softmax: Select top-K prototypes based on learned scores.

    This provides a simpler alternative that directly selects the top-K most important
    prototypes as causal, and the rest as confounding.
    """

    def __init__(self, proto_dim, hidden_dim=256, top_k_ratio=0.5):
        """
        Args:
            proto_dim (int): Dimension of each prototype
            hidden_dim (int): Hidden dimension for MLP
            top_k_ratio (float): Ratio of prototypes to select as causal (0.0 to 1.0)
        """
        super(TopKPrototypeSelector, self).__init__()

        self.proto_dim = proto_dim
        self.hidden_dim = hidden_dim
        self.top_k_ratio = top_k_ratio

        # MLP for learning importance scores
        self.score_mlp = nn.Sequential(
            nn.Linear(proto_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim // 2, 1)  # Single importance score
        )

    def forward(self, proto_repr, return_scores=False):
        """
        Args:
            proto_repr (torch.Tensor): Prototype representations of shape (B, n_proto, proto_dim)
            return_scores (bool): If True, also return the raw importance scores

        Returns:
            dict: {
                'causal_mask': (B, n_proto) - Binary mask for causal prototypes
                'confound_mask': (B, n_proto) - Binary mask for confounding prototypes
                'scores': (B, n_proto) - Importance scores (if return_scores=True)
            }
        """
        B, n_proto, proto_dim = proto_repr.shape

        # Compute importance scores for each prototype
        # Shape: (B, n_proto, 1) -> (B, n_proto)
        scores = self.score_mlp(proto_repr).squeeze(-1)

        # Calculate top-K
        k = max(1, int(n_proto * self.top_k_ratio))

        # Get top-K indices
        _, top_k_indices = torch.topk(scores, k, dim=-1)

        # Create binary masks
        causal_mask = torch.zeros_like(scores)
        causal_mask.scatter_(1, top_k_indices, 1.0)
        confound_mask = 1.0 - causal_mask

        # Apply sigmoid to scores for soft weighting (optional)
        soft_weights = torch.sigmoid(scores)

        result = {
            'causal_mask': causal_mask,
            'confound_mask': confound_mask,
            'soft_weights': soft_weights,
        }

        if return_scores:
            result['scores'] = scores

        return result


class AdaptivePrototypeSelector(nn.Module):
    """
    Adaptive prototype selector that learns a threshold for causal/confounding separation.

    Instead of using a fixed threshold or top-K, this learns an adaptive threshold
    that can vary per sample.
    """

    def __init__(self, proto_dim, hidden_dim=256, init_threshold=0.5):
        """
        Args:
            proto_dim (int): Dimension of each prototype
            hidden_dim (int): Hidden dimension for MLP
            init_threshold (float): Initial threshold value
        """
        super(AdaptivePrototypeSelector, self).__init__()

        self.proto_dim = proto_dim
        self.hidden_dim = hidden_dim

        # MLP for computing prototype importance scores
        self.score_mlp = nn.Sequential(
            nn.Linear(proto_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim // 2, 1)
        )

        # MLP for computing adaptive threshold (per sample)
        self.threshold_mlp = nn.Sequential(
            nn.Linear(proto_dim * 2, hidden_dim),  # Use aggregated statistics
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Threshold in [0, 1]
        )

        # Initialize threshold
        self.register_buffer('init_threshold', torch.tensor(init_threshold))

    def forward(self, proto_repr, return_scores=False):
        """
        Args:
            proto_repr (torch.Tensor): Prototype representations of shape (B, n_proto, proto_dim)
            return_scores (bool): If True, also return the raw scores and thresholds

        Returns:
            dict: {
                'causal_mask': (B, n_proto) - Soft mask for causal prototypes
                'confound_mask': (B, n_proto) - Soft mask for confounding prototypes
                'scores': (B, n_proto) - Importance scores (if return_scores=True)
                'threshold': (B, 1) - Adaptive threshold (if return_scores=True)
            }
        """
        B, n_proto, proto_dim = proto_repr.shape

        # Compute importance scores for each prototype
        scores = self.score_mlp(proto_repr).squeeze(-1)  # (B, n_proto)
        scores_sigmoid = torch.sigmoid(scores)

        # Compute adaptive threshold using aggregated statistics
        proto_mean = proto_repr.mean(dim=1)  # (B, proto_dim)
        proto_std = proto_repr.std(dim=1)    # (B, proto_dim)
        proto_stats = torch.cat([proto_mean, proto_std], dim=-1)  # (B, 2*proto_dim)

        threshold = self.threshold_mlp(proto_stats)  # (B, 1)

        # Soft masking using sigmoid
        # causal_mask is high when score > threshold
        causal_mask = torch.sigmoid((scores_sigmoid - threshold) * 10)  # Temperature=10 for sharpness
        confound_mask = 1.0 - causal_mask

        result = {
            'causal_mask': causal_mask,
            'confound_mask': confound_mask,
        }

        if return_scores:
            result['scores'] = scores
            result['threshold'] = threshold

        return result


if __name__ == '__main__':
    """
    Test the prototype selection networks
    """
    # Test configuration
    batch_size = 4
    n_proto_histo = 16
    n_proto_gene = 50
    proto_dim = 512

    print("=" * 80)
    print("Testing Prototype Selection Networks")
    print("=" * 80)

    # Create dummy data
    histo_proto = torch.randn(batch_size, n_proto_histo, proto_dim)
    gene_proto = torch.randn(batch_size, n_proto_gene, proto_dim)

    # Test 1: Gumbel-Softmax selector
    print("\n1. Testing Gumbel-Softmax Selector:")
    print("-" * 80)
    selector_gumbel = PrototypeSelectionNetwork(
        proto_dim=proto_dim,
        hidden_dim=256,
        tau=1.0,
        hard=True,
        use_gumbel=True
    )

    result = selector_gumbel(histo_proto, return_logits=True)
    print(f"Selection probs shape: {result['selection_probs'].shape}")
    print(f"Causal mask shape: {result['causal_mask'].shape}")
    print(f"Example causal mask: {result['causal_mask'][0]}")
    print(f"Number of causal prototypes (sample 0): {result['causal_mask'][0].sum().item():.2f}")

    # Test 2: Top-K selector
    print("\n2. Testing Top-K Selector:")
    print("-" * 80)
    selector_topk = TopKPrototypeSelector(
        proto_dim=proto_dim,
        hidden_dim=256,
        top_k_ratio=0.5
    )

    result = selector_topk(histo_proto, return_scores=True)
    print(f"Causal mask shape: {result['causal_mask'].shape}")
    print(f"Example causal mask: {result['causal_mask'][0]}")
    print(f"Number of causal prototypes (sample 0): {result['causal_mask'][0].sum().item():.0f}")
    print(f"Scores range: [{result['scores'].min():.3f}, {result['scores'].max():.3f}]")

    # Test 3: Adaptive selector
    print("\n3. Testing Adaptive Selector:")
    print("-" * 80)
    selector_adaptive = AdaptivePrototypeSelector(
        proto_dim=proto_dim,
        hidden_dim=256,
        init_threshold=0.5
    )

    result = selector_adaptive(histo_proto, return_scores=True)
    print(f"Causal mask shape: {result['causal_mask'].shape}")
    print(f"Example causal mask (soft): {result['causal_mask'][0]}")
    print(f"Effective number of causal prototypes (sample 0): {result['causal_mask'][0].sum().item():.2f}")
    print(f"Threshold (sample 0): {result['threshold'][0].item():.3f}")

    # Test backward pass
    print("\n4. Testing Gradient Flow:")
    print("-" * 80)
    selector = PrototypeSelectionNetwork(proto_dim=proto_dim, use_gumbel=True, hard=True)
    selector.train()

    histo_proto_grad = histo_proto.clone().requires_grad_(True)
    result = selector(histo_proto_grad)

    # Dummy loss
    loss = result['causal_mask'].sum()
    loss.backward()

    print(f"Gradient computed: {histo_proto_grad.grad is not None}")
    print(f"Gradient norm: {histo_proto_grad.grad.norm().item():.6f}")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
