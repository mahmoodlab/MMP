#!/usr/bin/env python
"""
End-to-End Test for Causal-Confounding Separation Model

This script tests all components of the causal separation architecture:
1. Prototype selection networks
2. Three-branch prediction network
3. Loss functions
4. Model integration with training pipeline

Usage:
    python test_causal_separation.py

Author: Claude
Date: 2025-10-21
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
import numpy as np

print("=" * 80)
print("Testing Causal-Confounding Separation Model")
print("=" * 80)

# ============================================================================
# Test 1: Import all modules
# ============================================================================
print("\n[Test 1] Importing modules...")
print("-" * 80)

try:
    from mil_models.prototype_selection import (
        PrototypeSelectionNetwork,
        TopKPrototypeSelector,
        AdaptivePrototypeSelector,
        GumbelSoftmax
    )
    print("✓ Prototype selection modules imported successfully")
except Exception as e:
    print(f"✗ Failed to import prototype selection modules: {e}")
    sys.exit(1)

try:
    from mil_models.model_causal_separation import (
        CausalSeparationModel,
        SharedTransformerBranch
    )
    print("✓ Causal separation model imported successfully")
except Exception as e:
    print(f"✗ Failed to import causal separation model: {e}")
    sys.exit(1)

try:
    from mil_models.causal_losses import (
        CausalConsistencyLoss,
        ConfoundingRandomnessLoss,
        SelectionSparsityLoss,
        SelectionBalanceLoss,
        CausalSeparationLoss,
        compute_causal_separation_metrics
    )
    print("✓ Causal loss functions imported successfully")
except Exception as e:
    print(f"✗ Failed to import causal loss functions: {e}")
    sys.exit(1)

try:
    from mil_models.model_factory import create_multimodal_survival_model
    print("✓ Model factory imported successfully")
except Exception as e:
    print(f"✗ Failed to import model factory: {e}")
    sys.exit(1)

# ============================================================================
# Test 2: Test Prototype Selection Networks
# ============================================================================
print("\n[Test 2] Testing Prototype Selection Networks...")
print("-" * 80)

batch_size = 4
n_proto_histo = 16
n_proto_gene = 50
proto_dim = 256

# Create dummy prototype representations
histo_proto = torch.randn(batch_size, n_proto_histo, proto_dim)
gene_proto = torch.randn(batch_size, n_proto_gene, proto_dim)

# Test Gumbel-Softmax selector
print("\nTesting Gumbel-Softmax selector...")
try:
    selector = PrototypeSelectionNetwork(
        proto_dim=proto_dim,
        tau=1.0,
        hard=True,
        use_gumbel=True
    )
    result = selector(histo_proto, return_logits=True)

    assert 'causal_mask' in result
    assert 'confound_mask' in result
    assert result['causal_mask'].shape == (batch_size, n_proto_histo)

    print(f"  ✓ Output shapes correct")
    print(f"  ✓ Avg causal prototypes: {result['causal_mask'].sum(1).mean():.2f}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test Top-K selector
print("\nTesting Top-K selector...")
try:
    selector = TopKPrototypeSelector(
        proto_dim=proto_dim,
        top_k_ratio=0.5
    )
    result = selector(histo_proto, return_scores=True)

    assert 'causal_mask' in result
    assert result['causal_mask'].sum(1).mean() == n_proto_histo * 0.5

    print(f"  ✓ Top-K selection working correctly")
    print(f"  ✓ Causal prototypes: {result['causal_mask'].sum(1).mean():.0f}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test gradient flow
print("\nTesting gradient flow through selector...")
try:
    selector = PrototypeSelectionNetwork(proto_dim=proto_dim, use_gumbel=True, hard=True)
    selector.train()

    histo_grad = histo_proto.clone().requires_grad_(True)
    result = selector(histo_grad)
    loss = result['causal_mask'].sum()
    loss.backward()

    assert histo_grad.grad is not None
    print(f"  ✓ Gradients computed successfully")
    print(f"  ✓ Gradient norm: {histo_grad.grad.norm():.6f}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# ============================================================================
# Test 3: Test Three-Branch Model
# ============================================================================
print("\n[Test 3] Testing Three-Branch Causal Separation Model...")
print("-" * 80)

# Create dummy data
# For PANTHER: each prototype has [prob, mean, cov]
# Total per prototype: 1 + proto_dim + proto_dim
histo_in_dim = 1 + proto_dim + proto_dim
x_path = torch.randn(batch_size, n_proto_histo, histo_in_dim)

# Gene data
omic_sizes = [281] * n_proto_gene
x_omics = [torch.randn(batch_size, size) for size in omic_sizes]

# Labels
label = torch.randint(0, 4, (batch_size,))
censorship = torch.randint(0, 2, (batch_size,))

print("\nCreating model...")
try:
    model = CausalSeparationModel(
        omic_sizes=omic_sizes,
        histo_in_dim=histo_in_dim,
        num_classes=4,
        path_proj_dim=proto_dim,
        num_coattn_layers=1,
        histo_model='PANTHER',
        numOfproto=n_proto_histo,
        selection_method='gumbel',
        tau=1.0,
        hard=True
    )
    print(f"  ✓ Model created successfully")
    print(f"  ✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"  ✗ Failed to create model: {e}")
    sys.exit(1)

print("\nTesting forward pass...")
try:
    model.train()
    results, logs = model(
        x_path, x_omics,
        label=label,
        censorship=censorship,
        loss_fn='nll',
        return_selection=True
    )

    # Check outputs
    assert 'causal_logits' in results
    assert 'confound_logits' in results
    assert 'full_logits' in results
    assert 'histo_causal_mask' in results
    assert 'gene_causal_mask' in results

    print(f"  ✓ Forward pass successful")
    print(f"  ✓ Causal logits shape: {results['causal_logits'].shape}")
    print(f"  ✓ Confound logits shape: {results['confound_logits'].shape}")
    print(f"  ✓ Full logits shape: {results['full_logits'].shape}")
    print(f"  ✓ Avg causal histo prototypes: {results['histo_causal_mask'].sum(1).mean():.2f}")
    print(f"  ✓ Avg causal gene prototypes: {results['gene_causal_mask'].sum(1).mean():.2f}")
except Exception as e:
    print(f"  ✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTesting backward pass...")
try:
    total_loss = results['causal_loss'] + results['confound_loss'] + results['full_loss']
    total_loss.backward()
    print(f"  ✓ Backward pass successful")
except Exception as e:
    print(f"  ✗ Backward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Test 4: Test Loss Functions
# ============================================================================
print("\n[Test 4] Testing Loss Functions...")
print("-" * 80)

# Create fresh outputs for loss testing
model.zero_grad()
with torch.no_grad():
    results, logs = model(
        x_path, x_omics,
        label=label,
        censorship=censorship,
        loss_fn='nll',
        return_selection=True
    )

print("\nTesting individual loss components...")
try:
    # Causal consistency loss
    cc_loss = CausalConsistencyLoss(mode='kl')
    loss = cc_loss(results['causal_logits'], results['full_logits'])
    print(f"  ✓ Causal consistency loss (KL): {loss.item():.4f}")

    # Confounding randomness loss
    cr_loss = ConfoundingRandomnessLoss(mode='entropy', num_classes=4)
    loss = cr_loss(results['confound_logits'])
    print(f"  ✓ Confounding randomness loss: {loss.item():.4f}")

    # Sparsity loss
    sp_loss = SelectionSparsityLoss()
    loss = sp_loss(results['histo_causal_mask'])
    print(f"  ✓ Selection sparsity loss: {loss.item():.4f}")

    # Balance loss
    bl_loss = SelectionBalanceLoss(target_ratio=0.5)
    loss = bl_loss(results['histo_causal_mask'])
    print(f"  ✓ Selection balance loss: {loss.item():.4f}")

except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTesting combined loss...")
try:
    causal_loss_fn = CausalSeparationLoss(
        survival_loss_fn='nll',
        lambda_causal_consistency=1.0,
        lambda_confound_random=0.5,
        lambda_sparsity=0.01,
        lambda_balance=0.1,
        num_classes=4
    )

    total_loss, loss_dict = causal_loss_fn(results)

    print(f"  ✓ Total loss: {total_loss:.4f}")
    print(f"  ✓ Loss components:")
    for key, value in loss_dict.items():
        print(f"      - {key}: {value:.4f}")

except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTesting monitoring metrics...")
try:
    metrics = compute_causal_separation_metrics(results)
    print(f"  ✓ Computed {len(metrics)} monitoring metrics")
    print(f"  ✓ Key metrics:")
    for key in ['histo_causal_ratio', 'gene_causal_ratio',
                'causal_full_similarity', 'causal_entropy', 'confound_entropy']:
        if key in metrics:
            print(f"      - {key}: {metrics[key]:.4f}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Test 5: Test Model Factory Integration
# ============================================================================
print("\n[Test 5] Testing Model Factory Integration...")
print("-" * 80)

try:
    # Create a mock args object
    class Args:
        model_mm_type = 'causal_separation'
        model_histo_type = 'PANTHER'
        feat_dim = histo_in_dim
        path_proj_dim = 256
        num_coattn_layers = 1
        histo_agg = 'mean'
        append_embed = 'none'
        n_proto = n_proto_histo
        selection_method = 'gumbel'
        gumbel_tau = 1.0
        gumbel_hard = True
        top_k_ratio = 0.5
        loss_fn = 'nll'
        n_label_bins = 4

    args = Args()

    model = create_multimodal_survival_model(args, omic_sizes=omic_sizes)

    print(f"  ✓ Model created through factory successfully")
    print(f"  ✓ Model type: {type(model).__name__}")

    # Test forward pass
    results, logs = model(
        x_path, x_omics,
        label=label,
        censorship=censorship,
        loss_fn='nll',
        return_selection=True
    )

    print(f"  ✓ Forward pass through factory model successful")

except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Test 6: Test Different Selection Methods
# ============================================================================
print("\n[Test 6] Testing Different Selection Methods...")
print("-" * 80)

for method in ['gumbel', 'topk', 'adaptive']:
    print(f"\nTesting {method} selection...")
    try:
        model = CausalSeparationModel(
            omic_sizes=omic_sizes,
            histo_in_dim=histo_in_dim,
            num_classes=4,
            path_proj_dim=proto_dim,
            num_coattn_layers=1,
            histo_model='PANTHER',
            numOfproto=n_proto_histo,
            selection_method=method,
            tau=1.0,
            hard=True,
            top_k_ratio=0.5
        )

        with torch.no_grad():
            results, logs = model(
                x_path, x_omics,
                label=label,
                censorship=censorship,
                loss_fn='nll',
                return_selection=True
            )

        avg_causal = results['histo_causal_mask'].sum(1).mean().item()
        print(f"  ✓ {method}: avg causal prototypes = {avg_causal:.2f}")

    except Exception as e:
        print(f"  ✗ {method} failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "=" * 80)
print("ALL TESTS PASSED! ✓")
print("=" * 80)
print("\nSummary:")
print("  ✓ All modules imported successfully")
print("  ✓ Prototype selection networks working")
print("  ✓ Three-branch model architecture working")
print("  ✓ Loss functions working")
print("  ✓ Model factory integration working")
print("  ✓ All selection methods working")
print("\nThe causal-confounding separation model is ready to use!")
print("=" * 80)
