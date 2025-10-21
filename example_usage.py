#!/usr/bin/env python
"""
Quick Start Example for Causal-Confounding Separation Model

This is a minimal working example showing how to use the causal separation model.

Author: Claude
Date: 2025-10-21
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from mil_models.model_causal_separation import CausalSeparationModel
from mil_models.causal_losses import CausalSeparationLoss, compute_causal_separation_metrics

# ============================================================================
# Example: Creating and using the causal separation model
# ============================================================================

print("=" * 80)
print("Causal-Confounding Separation Model - Quick Start Example")
print("=" * 80)

# Configuration
batch_size = 4
n_histo_proto = 16      # Number of histology prototypes
n_gene_pathways = 50    # Number of gene pathways
proto_dim = 256         # Prototype dimension

# For PANTHER model: each prototype has [prob, mean, cov]
# Total dimension per prototype: 1 + proto_dim + proto_dim
histo_in_dim = 1 + proto_dim + proto_dim

print(f"\nConfiguration:")
print(f"  Batch size: {batch_size}")
print(f"  Histology prototypes: {n_histo_proto}")
print(f"  Gene pathways: {n_gene_pathways}")
print(f"  Prototype dimension: {proto_dim}")

# ============================================================================
# Step 1: Create dummy data (replace with your real data)
# ============================================================================

print(f"\nStep 1: Preparing data...")

# Histology prototype features
# Shape: (batch_size, n_histo_proto, histo_in_dim)
x_path = torch.randn(batch_size, n_histo_proto, histo_in_dim)

# Gene pathway features (list of tensors, one per pathway)
# Each pathway has some number of genes
omic_sizes = [281] * n_gene_pathways  # e.g., 50 pathways, each with 281 genes
x_omics = [torch.randn(batch_size, size) for size in omic_sizes]

# Survival labels
label = torch.randint(0, 4, (batch_size,))           # Discretized survival time
censorship = torch.randint(0, 2, (batch_size,))      # 0 = event, 1 = censored

print(f"  ✓ Data prepared")
print(f"    - Histology features: {x_path.shape}")
print(f"    - Gene features: {len(x_omics)} pathways")
print(f"    - Labels: {label.shape}")

# ============================================================================
# Step 2: Create the model
# ============================================================================

print(f"\nStep 2: Creating model...")

model = CausalSeparationModel(
    omic_sizes=omic_sizes,
    histo_in_dim=histo_in_dim,
    num_classes=4,                      # 4 bins for NLL loss
    path_proj_dim=proto_dim,
    num_coattn_layers=1,
    histo_model='PANTHER',
    histo_agg='mean',
    numOfproto=n_histo_proto,
    selection_method='gumbel',          # Options: 'gumbel', 'topk', 'adaptive'
    tau=1.0,                            # Gumbel temperature
    hard=True,                          # Use hard Gumbel
    top_k_ratio=0.5                     # For topk method
)

n_params = sum(p.numel() for p in model.parameters())
print(f"  ✓ Model created")
print(f"    - Total parameters: {n_params:,}")

# ============================================================================
# Step 3: Forward pass
# ============================================================================

print(f"\nStep 3: Forward pass...")

model.train()
results, logs = model(
    x_path, x_omics,
    label=label,
    censorship=censorship,
    loss_fn='nll',
    return_selection=True
)

print(f"  ✓ Forward pass completed")
print(f"\n  Outputs:")
print(f"    - Causal branch logits: {results['causal_logits'].shape}")
print(f"    - Confound branch logits: {results['confound_logits'].shape}")
print(f"    - Full branch logits: {results['full_logits'].shape}")
print(f"\n  Selection info:")
print(f"    - Histo causal mask: {results['histo_causal_mask'].shape}")
print(f"    - Gene causal mask: {results['gene_causal_mask'].shape}")
print(f"    - Avg causal histo prototypes: {results['histo_causal_mask'].sum(1).mean():.2f}/{n_histo_proto}")
print(f"    - Avg causal gene prototypes: {results['gene_causal_mask'].sum(1).mean():.2f}/{n_gene_pathways}")

# ============================================================================
# Step 4: Compute loss
# ============================================================================

print(f"\nStep 4: Computing loss...")

causal_loss_fn = CausalSeparationLoss(
    survival_loss_fn='nll',
    lambda_causal_consistency=1.0,
    lambda_confound_random=0.5,
    lambda_sparsity=0.01,
    lambda_balance=0.1,
    num_classes=4
)

total_loss, loss_dict = causal_loss_fn(results)

print(f"  ✓ Loss computed")
print(f"\n  Total loss: {total_loss:.4f}")
print(f"\n  Loss components:")
for key, value in loss_dict.items():
    if key != 'total_loss':
        print(f"    - {key}: {value:.4f}")

# ============================================================================
# Step 5: Compute monitoring metrics
# ============================================================================

print(f"\nStep 5: Computing monitoring metrics...")

metrics = compute_causal_separation_metrics(results)

print(f"  ✓ Metrics computed")
print(f"\n  Key metrics:")
print(f"    - Histo causal ratio: {metrics['histo_causal_ratio']:.2%}")
print(f"    - Gene causal ratio: {metrics['gene_causal_ratio']:.2%}")
print(f"    - Causal-Full similarity: {metrics['causal_full_similarity']:.4f}")
print(f"    - Confound-Full similarity: {metrics['confound_full_similarity']:.4f}")
print(f"    - Causal entropy: {metrics['causal_entropy']:.4f}")
print(f"    - Confound entropy: {metrics['confound_entropy']:.4f}")

# ============================================================================
# Step 6: Backward pass
# ============================================================================

print(f"\nStep 6: Backward pass...")

total_loss.backward()

print(f"  ✓ Gradients computed")

# Check a few gradients
sample_params = list(model.parameters())[:3]
grad_norms = [p.grad.norm().item() if p.grad is not None else 0.0 for p in sample_params]
print(f"  ✓ Sample gradient norms: {[f'{g:.6f}' for g in grad_norms]}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("Example completed successfully!")
print("=" * 80)

print("\nNext steps:")
print("  1. Replace dummy data with your real histology and gene data")
print("  2. Adjust hyperparameters (loss weights, selection method, etc.)")
print("  3. Train using the full training pipeline (see scripts/survival/causal_separation_demo.sh)")
print("  4. Evaluate on validation/test sets")
print("  5. Analyze which prototypes are selected as causal")

print("\nFor more details, see CAUSAL_SEPARATION_README.md")
print("=" * 80)
