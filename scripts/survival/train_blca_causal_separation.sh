#!/bin/bash

# Causal Separation Model Training Script for TCGA BLCA
#
# This script is configured for your TCGA BLCA dataset
# Features path: /data/TCGA/BLCA/features/pt_files
#
# Usage: ./train_blca_causal_separation.sh [gpu_id] [split_name]
#
# Example: ./train_blca_causal_separation.sh 0 0

# ============================================================================
# Configuration
# ============================================================================

# GPU and basic settings
GPU_ID=${1:-0}
SPLIT_NAME=${2:-"0"}

export CUDA_VISIBLE_DEVICES=$GPU_ID

# Data paths - MODIFY THESE TO MATCH YOUR SETUP
DATA_ROOT="/data/TCGA/BLCA/features/pt_files"
SPLIT_DIR="splits/TCGA_BLCA_survival_k=0"  # Update this to your actual split directory
OMIC_SOURCE="data_csvs/TCGA_BLCA/rnaseq/tcga_blca_rna_clean.csv"  # Update if you have BLCA RNA data
OMIC_NAMES="data_csvs/TCGA_BLCA/signatures/hallmark_gene_sets.csv"  # Or your gene signature file

# IMPORTANT: Set this to match your feature dimension
# Common values: 768 (ViT-based), 1024 (ResNet-based), 2048 (ResNet50)
IN_DIM=768  # Your features are 768-dimensional

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
OUT_TYPE="allcat"  # IMPORTANT: Must be 'allcat' for PANTHER, not 'param_cat'
EM_ITER=1
TAU=1.0
OT_EPS=0.1

# ============================================================================
# Training Configuration
# ============================================================================

MAX_EPOCHS=20
LR=2e-4
WEIGHT_DECAY=1e-5
BATCH_SIZE=1
ACCUM_STEPS=32
PRINT_EVERY=10

# Loss configuration
LOSS_FN="cox"  # Options: 'cox', 'nll', 'rank'

# ============================================================================
# Causal Separation Parameters
# ============================================================================

# Selection method: 'gumbel', 'topk', or 'adaptive'
SELECTION_METHOD="gumbel"
GUMBEL_TAU=1.0
GUMBEL_HARD=True
TOP_K_RATIO=0.5

# Loss weights
LAMBDA_CAUSAL_CONSISTENCY=1.0
LAMBDA_CONFOUND_RANDOM=0.5
LAMBDA_SPARSITY=0.01
LAMBDA_BALANCE=0.1

# Loss modes
CAUSAL_CONSISTENCY_MODE="kl"      # Options: 'kl', 'mse', 'cosine'
CONFOUND_RANDOM_MODE="entropy"    # Options: 'entropy', 'uniform', 'variance'
TARGET_SELECTION_RATIO=0.5

# ============================================================================
# Other Parameters
# ============================================================================

# Early stopping
EARLY_STOPPING=True
ES_PATIENCE=10
ES_MIN_EPOCHS=5

# Omics configuration
OMICS_MODALITY="pathway"  # Options: 'pathway', 'functional', 'none'

# Multimodal
NUM_COATTN_LAYERS=1
HISTO_AGG="mean"
APPEND_EMBED="none"

# Results directory
RESULTS_DIR="results/BLCA_causal_separation_${SELECTION_METHOD}_k${SPLIT_NAME}"

# ============================================================================
# Print Configuration
# ============================================================================

echo "========================================="
echo "TCGA BLCA Causal Separation Training"
echo "========================================="
echo "GPU ID: $GPU_ID"
echo "Split Name: $SPLIT_NAME"
echo "Data Root: $DATA_ROOT"
echo "Split Dir: $SPLIT_DIR"
echo "Model: $MODEL_MM_TYPE with $MODEL_HISTO_TYPE"
echo "Selection Method: $SELECTION_METHOD"
echo "Out Type: $OUT_TYPE"
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
