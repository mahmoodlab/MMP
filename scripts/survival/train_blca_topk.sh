#!/bin/bash

# Causal Separation Model Training Script for TCGA BLCA (TopK Selector)
#
# This version uses TopK selection which is more stable than Gumbel-Softmax
# and may prevent the histology selection collapse issue.
#
# Usage: ./train_blca_topk.sh [gpu_id] [split_name]

# ============================================================================
# Configuration
# ============================================================================

# GPU and basic settings
GPU_ID=${1:-0}
SPLIT_NAME=${2:-"0"}

export CUDA_VISIBLE_DEVICES=$GPU_ID

# Data paths - MODIFY THESE TO MATCH YOUR SETUP
DATA_ROOT="/data/TCGA/BLCA/features/pt_files"
SPLIT_DIR="splits/TCGA_BLCA_survival_k=0"
OMIC_SOURCE="data_csvs/TCGA_BLCA/rnaseq/tcga_blca_rna_clean.csv"
OMIC_NAMES="data_csvs/TCGA_BLCA/signatures/hallmark_gene_sets.csv"

# Feature dimension
IN_DIM=768

# Navigate to src directory
cd "$(dirname "$0")/../../src" || exit

# ============================================================================
# Model Configuration
# ============================================================================

MODEL_HISTO_TYPE="PANTHER"
MODEL_HISTO_CONFIG="PANTHER_default"
MODEL_MM_TYPE="causal_separation"

# Prototype configuration
N_PROTO=16
OUT_TYPE="allcat"
EM_ITER=1
TAU=1.0
OT_EPS=0.1

# ============================================================================
# Training Configuration
# ============================================================================

MAX_EPOCHS=30
LR=1e-4
WEIGHT_DECAY=1e-4               # Higher weight decay to prevent overfitting
BATCH_SIZE=1
ACCUM_STEPS=32
PRINT_EVERY=100

# Loss configuration
LOSS_FN="cox"

# ============================================================================
# Causal Separation Parameters - TOPK METHOD
# ============================================================================

# Use TopK selection (deterministic, more stable)
SELECTION_METHOD="topk"
GUMBEL_TAU=0.5
GUMBEL_HARD="true"
TOP_K_RATIO=0.3                 # Select top 30% based on learned scores

# Loss weights - STRONG REGULARIZATION
LAMBDA_CAUSAL_CONSISTENCY=1.0
LAMBDA_CONFOUND_RANDOM=1.0      # Push confound toward random
LAMBDA_SPARSITY=1.0             # Moderate sparsity (TopK already enforces sparsity)
LAMBDA_BALANCE=1.0              # Strong balance constraint to prevent histo collapse

# Loss modes
CAUSAL_CONSISTENCY_MODE="kl"
CONFOUND_RANDOM_MODE="entropy"
TARGET_SELECTION_RATIO=0.3      # Expect 30% selection

# ============================================================================
# Other Parameters
# ============================================================================

# Early stopping
EARLY_STOPPING="true"
ES_PATIENCE=10
ES_MIN_EPOCHS=5

# Omics configuration
OMICS_MODALITY="pathway"

# Multimodal
NUM_COATTN_LAYERS=1
HISTO_AGG="mean"
APPEND_EMBED="none"

# Results directory
RESULTS_DIR="results/BLCA_causal_topk_k${SPLIT_NAME}"

# ============================================================================
# Print Configuration
# ============================================================================

echo "========================================="
echo "TCGA BLCA Causal Separation Training (TopK)"
echo "========================================="
echo "GPU ID: $GPU_ID"
echo "Split Name: $SPLIT_NAME"
echo "Selection Method: $SELECTION_METHOD (top ${TOP_K_RATIO})"
echo "Results Dir: $RESULTS_DIR"
echo "========================================="
echo ""

# ============================================================================
# Run Training
# ============================================================================

python -m training.main_survival \
    --split_dir "$SPLIT_DIR" \
    --split_names "$SPLIT_NAME" \
    --data_source "$DATA_ROOT" \
    --omic_source "$OMIC_SOURCE" \
    --omic_names "$OMIC_NAMES" \
    --omics_modality "$OMICS_MODALITY" \
    --model_histo_type "$MODEL_HISTO_TYPE" \
    --model_histo_config "$MODEL_HISTO_CONFIG" \
    --model_mm_type "$MODEL_MM_TYPE" \
    --in_dim "$IN_DIM" \
    --n_proto "$N_PROTO" \
    --out_type "$OUT_TYPE" \
    --em_iter "$EM_ITER" \
    --tau "$TAU" \
    --ot_eps "$OT_EPS" \
    --loss_fn "$LOSS_FN" \
    --max_epochs "$MAX_EPOCHS" \
    --lr "$LR" \
    --wd "$WEIGHT_DECAY" \
    --batch_size "$BATCH_SIZE" \
    --accum_steps "$ACCUM_STEPS" \
    --print_every "$PRINT_EVERY" \
    --num_coattn_layers "$NUM_COATTN_LAYERS" \
    --histo_agg "$HISTO_AGG" \
    --append_embed "$APPEND_EMBED" \
    --selection_method "$SELECTION_METHOD" \
    --gumbel_tau "$GUMBEL_TAU" \
    --top_k_ratio "$TOP_K_RATIO" \
    --lambda_causal_consistency "$LAMBDA_CAUSAL_CONSISTENCY" \
    --lambda_confound_random "$LAMBDA_CONFOUND_RANDOM" \
    --lambda_sparsity "$LAMBDA_SPARSITY" \
    --lambda_balance "$LAMBDA_BALANCE" \
    --causal_consistency_mode "$CAUSAL_CONSISTENCY_MODE" \
    --confound_random_mode "$CONFOUND_RANDOM_MODE" \
    --target_selection_ratio "$TARGET_SELECTION_RATIO" \
    --early_stopping "$EARLY_STOPPING" \
    --es_patience "$ES_PATIENCE" \
    --es_min_epochs "$ES_MIN_EPOCHS" \
    --overwrite \
    --results_dir "$RESULTS_DIR"

echo ""
echo "========================================="
echo "Training completed!"
echo "Results saved to: $RESULTS_DIR"
echo "========================================="
