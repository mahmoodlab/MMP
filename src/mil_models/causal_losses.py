"""
Loss Functions for Causal-Confounding Prototype Separation

This module implements various loss functions designed to enforce the separation
of causal and confounding prototypes:

1. Causal Consistency Loss: Ensures the causal branch prediction is close to full branch
2. Confounding Randomness Loss: Penalizes confounding branch from having predictive power
3. Selection Sparsity Loss: Encourages sparse selection of causal prototypes
4. Selection Balance Loss: Prevents trivial solutions (selecting all or none)

Author: Claude
Date: 2025-10-21
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CausalConsistencyLoss(nn.Module):
    """
    Ensures the causal branch produces predictions similar to the full branch.

    This is based on the assumption that causal features should be sufficient
    to achieve performance similar to using all features.

    Loss = KL(P_full || P_causal) or MSE(logits_full, logits_causal)
    """

    def __init__(self, mode='kl', reduction='mean'):
        """
        Args:
            mode: 'kl' for KL divergence, 'mse' for mean squared error, 'cosine' for cosine similarity
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.mode = mode
        self.reduction = reduction

    def forward(self, causal_logits, full_logits):
        """
        Args:
            causal_logits: (B, num_classes) - Predictions from causal branch
            full_logits: (B, num_classes) - Predictions from full branch

        Returns:
            loss: Scalar or (B,) depending on reduction
        """
        if self.mode == 'kl':
            # KL divergence between distributions
            causal_probs = F.softmax(causal_logits, dim=-1)
            full_probs = F.softmax(full_logits, dim=-1)

            # KL(full || causal) - we want causal to match full
            loss = F.kl_div(
                torch.log(causal_probs + 1e-8),
                full_probs,
                reduction='none'
            ).sum(dim=-1)

            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()

        elif self.mode == 'mse':
            # Mean squared error on logits
            loss = F.mse_loss(causal_logits, full_logits, reduction=self.reduction)

        elif self.mode == 'cosine':
            # Cosine similarity loss
            # We want high similarity, so loss = 1 - cosine_similarity
            cosine_sim = F.cosine_similarity(causal_logits, full_logits, dim=-1)
            loss = 1 - cosine_sim

            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return loss


class ConfoundingRandomnessLoss(nn.Module):
    """
    Encourages the confounding branch to have random/uninformative predictions.

    This can be achieved by:
    1. Maximizing entropy of predictions (uniform distribution)
    2. Minimizing prediction variance
    3. Pushing predictions towards a uniform/random baseline
    """

    def __init__(self, mode='entropy', num_classes=4, reduction='mean'):
        """
        Args:
            mode: 'entropy' to maximize entropy, 'uniform' to match uniform distribution,
                  'variance' to minimize variance
            num_classes: Number of output classes
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.mode = mode
        self.num_classes = num_classes
        self.reduction = reduction

        # Create uniform target distribution
        self.register_buffer('uniform_dist', torch.ones(num_classes) / num_classes)

    def forward(self, confound_logits):
        """
        Args:
            confound_logits: (B, num_classes) - Predictions from confounding branch

        Returns:
            loss: Scalar or (B,) depending on reduction
        """
        if self.mode == 'entropy':
            # Maximize entropy = minimize negative entropy
            confound_probs = F.softmax(confound_logits, dim=-1)
            entropy = -(confound_probs * torch.log(confound_probs + 1e-8)).sum(dim=-1)

            # We want to maximize entropy, so minimize negative entropy
            loss = -entropy

            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()

        elif self.mode == 'uniform':
            # Push predictions towards uniform distribution
            confound_probs = F.softmax(confound_logits, dim=-1)

            # KL divergence to uniform distribution
            uniform = self.uniform_dist.unsqueeze(0).expand_as(confound_probs)
            loss = F.kl_div(
                torch.log(confound_probs + 1e-8),
                uniform,
                reduction='none'
            ).sum(dim=-1)

            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()

        elif self.mode == 'variance':
            # Minimize variance across samples
            # This encourages all samples to have similar predictions
            confound_probs = F.softmax(confound_logits, dim=-1)
            mean_pred = confound_probs.mean(dim=0)
            variance = ((confound_probs - mean_pred) ** 2).sum(dim=-1)

            if self.reduction == 'mean':
                loss = variance.mean()
            elif self.reduction == 'sum':
                loss = variance.sum()
            else:
                loss = variance

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return loss


class SelectionSparsityLoss(nn.Module):
    """
    Encourages sparse selection of causal prototypes.

    This prevents the model from selecting too many prototypes as "causal".
    Uses L1 regularization on the selection masks.
    """

    def __init__(self, reduction='mean'):
        """
        Args:
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, causal_mask):
        """
        Args:
            causal_mask: (B, n_proto) - Binary or soft selection mask

        Returns:
            loss: Scalar or (B,) depending on reduction
        """
        # L1 norm encourages sparsity
        loss = causal_mask.sum(dim=-1)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class SelectionBalanceLoss(nn.Module):
    """
    Prevents trivial solutions where all or no prototypes are selected.

    This encourages the model to select a reasonable number of causal prototypes
    by penalizing deviation from a target selection ratio.
    """

    def __init__(self, target_ratio=0.5, reduction='mean'):
        """
        Args:
            target_ratio: Desired ratio of causal prototypes (0.0 to 1.0)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.target_ratio = target_ratio
        self.reduction = reduction

    def forward(self, causal_mask):
        """
        Args:
            causal_mask: (B, n_proto) - Binary or soft selection mask

        Returns:
            loss: Scalar or (B,) depending on reduction
        """
        n_proto = causal_mask.shape[1]
        target_count = n_proto * self.target_ratio

        # Compute actual count
        actual_count = causal_mask.sum(dim=-1)

        # MSE loss between actual and target
        loss = (actual_count - target_count) ** 2

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class CausalSeparationLoss(nn.Module):
    """
    Combined loss function for the causal separation model.

    This aggregates all the component losses with configurable weights.
    """

    def __init__(self,
                 survival_loss_fn='cox',
                 lambda_causal_consistency=1.0,
                 lambda_confound_random=0.5,
                 lambda_sparsity=0.01,
                 lambda_balance=0.1,
                 causal_consistency_mode='kl',
                 confound_random_mode='entropy',
                 target_selection_ratio=0.5,
                 num_classes=4):
        """
        Args:
            survival_loss_fn: Base survival loss ('cox', 'nll', 'ranking')
            lambda_causal_consistency: Weight for causal consistency loss
            lambda_confound_random: Weight for confounding randomness loss
            lambda_sparsity: Weight for selection sparsity loss
            lambda_balance: Weight for selection balance loss
            causal_consistency_mode: Mode for causal consistency loss
            confound_random_mode: Mode for confounding randomness loss
            target_selection_ratio: Target ratio for selection balance
            num_classes: Number of output classes
        """
        super().__init__()

        self.survival_loss_fn = survival_loss_fn
        self.lambda_causal_consistency = lambda_causal_consistency
        self.lambda_confound_random = lambda_confound_random
        self.lambda_sparsity = lambda_sparsity
        self.lambda_balance = lambda_balance

        # Component losses
        self.causal_consistency_loss = CausalConsistencyLoss(mode=causal_consistency_mode)
        self.confound_random_loss = ConfoundingRandomnessLoss(
            mode=confound_random_mode,
            num_classes=num_classes
        )
        self.sparsity_loss = SelectionSparsityLoss()
        self.balance_loss = SelectionBalanceLoss(target_ratio=target_selection_ratio)

    def forward(self, model_outputs):
        """
        Args:
            model_outputs: Dictionary with keys:
                - 'causal_logits': (B, num_classes)
                - 'confound_logits': (B, num_classes)
                - 'full_logits': (B, num_classes)
                - 'causal_loss': Survival loss for causal branch
                - 'confound_loss': Survival loss for confounding branch
                - 'full_loss': Survival loss for full branch
                - 'histo_causal_mask': (B, n_histo_proto)
                - 'gene_causal_mask': (B, n_gene_proto)

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        # Base survival losses
        survival_loss_causal = model_outputs.get('causal_loss', 0.0)
        survival_loss_confound = model_outputs.get('confound_loss', 0.0)
        survival_loss_full = model_outputs.get('full_loss', 0.0)

        # Causal consistency: causal branch should match full branch
        causal_consistency = self.causal_consistency_loss(
            model_outputs['causal_logits'],
            model_outputs['full_logits']
        )

        # Confounding randomness: confounding branch should be uninformative
        confound_random = self.confound_random_loss(
            model_outputs['confound_logits']
        )

        # Selection sparsity: encourage selecting fewer causal prototypes
        sparsity_histo = self.sparsity_loss(model_outputs['histo_causal_mask'])
        sparsity_gene = self.sparsity_loss(model_outputs['gene_causal_mask'])
        sparsity = (sparsity_histo + sparsity_gene) / 2

        # Selection balance: prevent trivial solutions
        balance_histo = self.balance_loss(model_outputs['histo_causal_mask'])
        balance_gene = self.balance_loss(model_outputs['gene_causal_mask'])
        balance = (balance_histo + balance_gene) / 2

        # Combine all losses
        total_loss = (
            survival_loss_full +  # Main survival loss from full branch
            self.lambda_causal_consistency * causal_consistency +
            self.lambda_confound_random * confound_random +
            self.lambda_sparsity * sparsity +
            self.lambda_balance * balance
        )

        # Prepare loss dictionary for logging
        loss_dict = {
            'total_loss': total_loss.item() if torch.is_tensor(total_loss) else total_loss,
            'survival_loss_causal': survival_loss_causal.item() if torch.is_tensor(survival_loss_causal) else survival_loss_causal,
            'survival_loss_confound': survival_loss_confound.item() if torch.is_tensor(survival_loss_confound) else survival_loss_confound,
            'survival_loss_full': survival_loss_full.item() if torch.is_tensor(survival_loss_full) else survival_loss_full,
            'causal_consistency': causal_consistency.item() if torch.is_tensor(causal_consistency) else causal_consistency,
            'confound_random': confound_random.item() if torch.is_tensor(confound_random) else confound_random,
            'sparsity': sparsity.item() if torch.is_tensor(sparsity) else sparsity,
            'balance': balance.item() if torch.is_tensor(balance) else balance,
        }

        return total_loss, loss_dict


def compute_causal_separation_metrics(model_outputs):
    """
    Compute additional metrics for monitoring causal separation.

    Args:
        model_outputs: Dictionary from model forward pass

    Returns:
        metrics_dict: Dictionary with monitoring metrics
    """
    metrics = {}

    # Selection statistics
    histo_causal_mask = model_outputs['histo_causal_mask']
    gene_causal_mask = model_outputs['gene_causal_mask']

    metrics['histo_causal_count'] = histo_causal_mask.sum(dim=1).mean().item()
    metrics['gene_causal_count'] = gene_causal_mask.sum(dim=1).mean().item()

    metrics['histo_causal_ratio'] = histo_causal_mask.mean().item()
    metrics['gene_causal_ratio'] = gene_causal_mask.mean().item()

    # Prediction similarity between branches
    causal_logits = model_outputs['causal_logits']
    confound_logits = model_outputs['confound_logits']
    full_logits = model_outputs['full_logits']

    # Cosine similarity
    causal_full_sim = F.cosine_similarity(causal_logits, full_logits, dim=-1).mean().item()
    confound_full_sim = F.cosine_similarity(confound_logits, full_logits, dim=-1).mean().item()
    causal_confound_sim = F.cosine_similarity(causal_logits, confound_logits, dim=-1).mean().item()

    metrics['causal_full_similarity'] = causal_full_sim
    metrics['confound_full_similarity'] = confound_full_sim
    metrics['causal_confound_similarity'] = causal_confound_sim

    # Prediction entropy (higher = more uncertain/random)
    causal_probs = F.softmax(causal_logits, dim=-1)
    confound_probs = F.softmax(confound_logits, dim=-1)
    full_probs = F.softmax(full_logits, dim=-1)

    causal_entropy = -(causal_probs * torch.log(causal_probs + 1e-8)).sum(dim=-1).mean().item()
    confound_entropy = -(confound_probs * torch.log(confound_probs + 1e-8)).sum(dim=-1).mean().item()
    full_entropy = -(full_probs * torch.log(full_probs + 1e-8)).sum(dim=-1).mean().item()

    metrics['causal_entropy'] = causal_entropy
    metrics['confound_entropy'] = confound_entropy
    metrics['full_entropy'] = full_entropy

    return metrics


if __name__ == '__main__':
    """
    Test the loss functions
    """
    print("=" * 80)
    print("Testing Causal Separation Loss Functions")
    print("=" * 80)

    batch_size = 8
    num_classes = 4
    n_histo_proto = 16
    n_gene_proto = 50

    # Create dummy model outputs
    model_outputs = {
        'causal_logits': torch.randn(batch_size, num_classes),
        'confound_logits': torch.randn(batch_size, num_classes),
        'full_logits': torch.randn(batch_size, num_classes),
        'causal_loss': torch.tensor(1.5),
        'confound_loss': torch.tensor(2.0),
        'full_loss': torch.tensor(1.2),
        'histo_causal_mask': torch.rand(batch_size, n_histo_proto),
        'gene_causal_mask': torch.rand(batch_size, n_gene_proto),
    }

    # Test individual loss components
    print("\n1. Testing Causal Consistency Loss:")
    print("-" * 80)
    loss_fn = CausalConsistencyLoss(mode='kl')
    loss = loss_fn(model_outputs['causal_logits'], model_outputs['full_logits'])
    print(f"KL loss: {loss.item():.4f}")

    loss_fn = CausalConsistencyLoss(mode='mse')
    loss = loss_fn(model_outputs['causal_logits'], model_outputs['full_logits'])
    print(f"MSE loss: {loss.item():.4f}")

    # Test confounding randomness loss
    print("\n2. Testing Confounding Randomness Loss:")
    print("-" * 80)
    loss_fn = ConfoundingRandomnessLoss(mode='entropy', num_classes=num_classes)
    loss = loss_fn(model_outputs['confound_logits'])
    print(f"Entropy loss: {loss.item():.4f}")

    loss_fn = ConfoundingRandomnessLoss(mode='uniform', num_classes=num_classes)
    loss = loss_fn(model_outputs['confound_logits'])
    print(f"Uniform loss: {loss.item():.4f}")

    # Test sparsity loss
    print("\n3. Testing Selection Sparsity Loss:")
    print("-" * 80)
    loss_fn = SelectionSparsityLoss()
    loss = loss_fn(model_outputs['histo_causal_mask'])
    print(f"Sparsity loss: {loss.item():.4f}")

    # Test balance loss
    print("\n4. Testing Selection Balance Loss:")
    print("-" * 80)
    loss_fn = SelectionBalanceLoss(target_ratio=0.5)
    loss = loss_fn(model_outputs['histo_causal_mask'])
    print(f"Balance loss: {loss.item():.4f}")

    # Test combined loss
    print("\n5. Testing Combined Causal Separation Loss:")
    print("-" * 80)
    loss_fn = CausalSeparationLoss(
        survival_loss_fn='nll',
        lambda_causal_consistency=1.0,
        lambda_confound_random=0.5,
        lambda_sparsity=0.01,
        lambda_balance=0.1,
        num_classes=num_classes
    )

    total_loss, loss_dict = loss_fn(model_outputs)
    print(f"Total loss: {total_loss:.4f}")
    print("\nLoss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")

    # Test metrics
    print("\n6. Testing Monitoring Metrics:")
    print("-" * 80)
    metrics = compute_causal_separation_metrics(model_outputs)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Test backward pass
    print("\n7. Testing Gradient Flow:")
    print("-" * 80)
    total_loss.backward()
    print("Backward pass successful!")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
