#!/bin/bash

# Causal-Confounding Separation Model Demo Script
#
# This script demonstrates how to train the causal separation model
# for survival prediction with multimodal data.
#
# Usage: ./causal_separation_demo.sh [gpu_id] [split_dir] [split_name] [data_root]
#
# Author: Claude
# Date: 2025-10-21

# Parse arguments
GPU_ID=${1:-0}
SPLIT_DIR=${2:-"splits/TCGA_BRCA_survival_k=0"}
SPLIT_NAME=${3:-"0"}
DATA_ROOT=${4:-"feats_h5"}

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Navigate to src directory
cd ../../src || exit

# Model configuration
MODEL_HISTO_TYPE="PANTHER"
MODEL_HISTO_CONFIG="PANTHER_default"
MODEL_MM_TYPE="causal_separation"

# Prototype configuration
N_PROTO=16
OUT_TYPE="allcat"
EM_ITER=1
TAU=1.0
OT_EPS=0.1

# Training configuration
MAX_EPOCHS=20
LR=2e-4
WEIGHT_DECAY=1e-5
BATCH_SIZE=1
ACCUM_STEPS=32
PRINT_EVERY=10

# Loss configuration
LOSS_FN="cox"  # Options: 'cox', 'nll', 'rank'

# Causal separation specific parameters
SELECTION_METHOD="gumbel"  # Options: 'gumbel', 'topk', 'adaptive'
GUMBEL_TAU=1.0
GUMBEL_HARD="True"
TOP_K_RATIO=0.5

# Loss weights
LAMBDA_CAUSAL_CONSISTENCY=1.0
LAMBDA_CONFOUND_RANDOM=0.5
LAMBDA_SPARSITY=0.01
LAMBDA_BALANCE=0.1

# Loss modes
CAUSAL_CONSISTENCY_MODE="kl"  # Options: 'kl', 'mse', 'cosine'
CONFOUND_RANDOM_MODE="entropy"  # Options: 'entropy', 'uniform', 'variance'
TARGET_SELECTION_RATIO=0.5

# Early stopping
EARLY_STOPPING="True"
ES_PATIENCE=10
ES_MIN_EPOCHS=5

# Omics configuration
OMICS_MODALITY="pathway"  # Options: 'pathway', 'functional', 'none'
OMIC_SOURCE="data_csvs/TCGA_BRCA/rnaseq/tcga_brca_rna_clean.csv"
OMIC_NAMES="data_csvs/TCGA_BRCA/signatures/hallmark_gene_sets.csv"

# Other hyperparameters
NUM_COATTN_LAYERS=1
HISTO_AGG="mean"
APPEND_EMBED="none"

# Results directory
RESULTS_DIR="results/causal_separation_${MODEL_HISTO_TYPE}_${SELECTION_METHOD}_k${SPLIT_NAME}"

echo "========================================="
echo "Causal-Confounding Separation Training"
echo "========================================="
echo "GPU ID: $GPU_ID"
echo "Split Dir: $SPLIT_DIR"
echo "Split Name: $SPLIT_NAME"
echo "Data Root: $DATA_ROOT"
echo "Model: $MODEL_MM_TYPE with $MODEL_HISTO_TYPE"
echo "Selection Method: $SELECTION_METHOD"
echo "Results Dir: $RESULTS_DIR"
echo "========================================="
echo ""

# Run training
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
    --n_proto "$N_PROTO" \
    --out_type "$OUT_TYPE" \
    --em_iter "$EM_ITER" \
    --tau "$TAU" \
    --ot_eps "$OT_EPS" \
    --loss_fn "$LOSS_FN" \
    --max_epochs "$MAX_EPOCHS" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --batch_size "$BATCH_SIZE" \
    --accum_steps "$ACCUM_STEPS" \
    --print_every "$PRINT_EVERY" \
    --num_coattn_layers "$NUM_COATTN_LAYERS" \
    --histo_agg "$HISTO_AGG" \
    --append_embed "$APPEND_EMBED" \
    --selection_method "$SELECTION_METHOD" \
    --gumbel_tau "$GUMBEL_TAU" \
    --gumbel_hard "$GUMBEL_HARD" \
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
    --results_dir "$RESULTS_DIR" \
    --use_causal_separation  # Flag to use the new trainer

echo ""
echo "========================================="
echo "Training completed!"
echo "Results saved to: $RESULTS_DIR"
echo "========================================="
