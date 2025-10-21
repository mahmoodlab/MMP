# Overfitting and Selection Collapse: Diagnosis and Fixes

## Problem Summary

Your training revealed three critical issues:

### 1. Severe Overfitting
- **Train C-Index**: 0.9446 (excellent)
- **Test C-Index**: 0.5723 (barely better than random 0.5)
- **Gap**: 0.37 (extremely high, should be < 0.10)

### 2. Histology Prototype Collapse
- **Expected**: ~5-8 histology prototypes selected (out of 16)
- **Actual**: 0.00 histology prototypes selected
- **Impact**: Model completely ignores histology features

### 3. Excessive Gene Selection
- **Expected**: ~15-20 gene prototypes (out of 50)
- **Actual**: 23.76 gene prototypes selected (47.5%)
- **Impact**: Not sparse enough, defeats purpose of selection

### 4. Confound Branch Not Random
- **Expected**: C-Index ≈ 0.5 (random)
- **Actual Train**: 0.6809
- **Actual Test**: 0.5009
- **Impact**: Confound branch is learning some patterns

## Root Causes

### Original Hyperparameters (Too Weak)
```bash
LAMBDA_SPARSITY=0.01        # Way too small!
LAMBDA_BALANCE=0.1          # Too weak
LAMBDA_CONFOUND_RANDOM=0.5  # Insufficient
TARGET_SELECTION_RATIO=0.5  # Too high (expects 50% selection)
WEIGHT_DECAY=1e-5           # Too small for preventing overfitting
LR=2e-4                     # Slightly too high
GUMBEL_TAU=1.0              # Too soft (not encouraging discrete selection)
```

### Why Histology Collapsed
1. **Weak Balance Penalty**: Lambda=0.1 wasn't strong enough to prevent collapse
2. **Gumbel-Softmax Instability**: Soft selection can get stuck in local minima
3. **Feature Quality**: Histology features may genuinely be less predictive than genomics
4. **Gradient Flow**: Selection network for histology might not be getting strong gradients

### Why Overfitting Occurred
1. **Low Weight Decay**: 1e-5 is too small for this model size
2. **Weak Regularization**: Sparsity and balance losses were too weak
3. **Small Dataset**: TCGA BLCA is relatively small (~400 samples)
4. **High Capacity**: Model has many parameters relative to data size

## Solutions

### Solution 1: Improved Gumbel-Softmax Configuration (Updated Script)

**File**: `scripts/survival/train_blca_causal_separation.sh`

```bash
# Updated hyperparameters
GUMBEL_TAU=0.5                  # ↓ Sharper selection (was 1.0)
LAMBDA_SPARSITY=2.0             # ↑↑ 200x stronger (was 0.01)
LAMBDA_BALANCE=0.5              # ↑ 5x stronger (was 0.1)
LAMBDA_CONFOUND_RANDOM=1.0      # ↑ 2x stronger (was 0.5)
TARGET_SELECTION_RATIO=0.3      # ↓ Expect 30% selection (was 50%)
WEIGHT_DECAY=1e-4               # ↑ 10x stronger (was 1e-5)
LR=1e-4                         # ↓ More stable (was 2e-4)
MAX_EPOCHS=30                   # ↑ More epochs with regularization
```

**Expected Results**:
- Histology: ~5 prototypes selected (30% of 16)
- Genomics: ~15 prototypes selected (30% of 50)
- Test C-Index: Should improve to 0.60-0.65
- Overfitting gap: Should reduce to < 0.15

### Solution 2: TopK Selection Method (New Script)

**File**: `scripts/survival/train_blca_topk.sh`

TopK selector is more stable than Gumbel-Softmax because:
- **Deterministic**: Always selects exactly top K prototypes
- **No Temperature Annealing**: Doesn't require tuning tau
- **Stronger Gradients**: Clear ranking signal

```bash
SELECTION_METHOD="topk"
TOP_K_RATIO=0.3                 # Select top 30%
LAMBDA_BALANCE=1.0              # Strong balance to prevent modality collapse
```

**When to Use**:
- If Gumbel-Softmax continues to show histology collapse
- If you want more interpretable selection (exactly K prototypes)
- If training is unstable

### Solution 3: Analysis Tool

**File**: `scripts/analyze_selection.py`

Monitor selection patterns during training:

```bash
python scripts/analyze_selection.py --results_dir results/BLCA_causal_separation_gumbel_k0
```

**Output**:
- Selection statistics per modality
- Overfitting diagnosis
- Actionable recommendations

## Recommended Training Strategy

### Step 1: Try Updated Gumbel Configuration

```bash
cd scripts/survival
./train_blca_causal_separation.sh 0 0
```

**Monitor**:
- Avg Histo Causal Prototypes should be > 2 (not 0!)
- Test C-Index should improve each epoch (not stay flat)
- Train-Test gap should be < 0.15

### Step 2: If Histology Still Collapses, Try TopK

```bash
./train_blca_topk.sh 0 0
```

TopK enforces hard selection and may prevent collapse.

### Step 3: Analyze Results

```bash
python ../analyze_selection.py --results_dir results/BLCA_causal_separation_gumbel_k0
```

## Expected Healthy Results

```
Validation Results:
  C-Index (Causal):   0.62-0.66  (↑ from 0.57)
  C-Index (Confound): 0.48-0.52  (↓ to random)
  C-Index (Full):     0.63-0.67  (↑ from 0.57)
  Avg Histo Causal:   4-6        (↑ from 0!)
  Avg Gene Causal:    12-18      (↓ from 24)
```

**Key Improvements**:
1. **No Histology Collapse**: 4-6 prototypes selected
2. **Sparser Gene Selection**: 12-18 instead of 24
3. **Reduced Overfitting**: Train-Test gap < 0.15
4. **Random Confound**: C-Index ≈ 0.5

## Advanced Tuning (If Needed)

### If Test Performance Still Poor

1. **Add Dropout** (requires code modification):
   - Add dropout to selection networks
   - Add dropout to fusion layers

2. **Data Augmentation**:
   - Feature noise injection
   - Sample reweighting

3. **Different Loss Function**:
   - Try `LOSS_FN="nll"` instead of `"cox"`
   - NLL may be more robust for small datasets

### If Histology Keeps Collapsing

1. **Separate Sparsity Penalties** (requires code modification):
   ```python
   # In causal_losses.py
   sparsity = lambda_histo * sparsity_histo + lambda_gene * sparsity_gene
   ```
   - Use weaker sparsity for histology (e.g., 0.5)
   - Use stronger sparsity for genomics (e.g., 3.0)

2. **Minimum Selection Constraint**:
   - Modify SelectionBalanceLoss to enforce minimum prototypes
   - E.g., penalize if histo < 3 prototypes

3. **Check Feature Quality**:
   ```python
   # Test if histology features are predictive at all
   python -m training.main_survival \
       --model_mm_type simple_fusion \  # Baseline without selection
       ...
   ```

## Debugging Checklist

- [ ] Histology prototypes selected > 0
- [ ] Gene prototypes selected < 30
- [ ] Test C-Index improves across epochs
- [ ] Confound C-Index ≈ 0.5 (random)
- [ ] Train-Test gap < 0.15
- [ ] Selection ratio close to target (30%)

## Files Modified

1. ✅ `scripts/survival/train_blca_causal_separation.sh` - Updated hyperparameters
2. ✅ `scripts/survival/train_blca_topk.sh` - New TopK configuration
3. ✅ `scripts/analyze_selection.py` - Selection analysis tool
4. ✅ `OVERFITTING_AND_SELECTION_FIXES.md` - This document

## Next Steps

1. Run training with updated configuration
2. Monitor selection statistics during training
3. Run analysis script on results
4. If issues persist, try TopK method
5. Report back with new results

Good luck! The updated hyperparameters should significantly improve your results.
