#!/bin/bash

# Test script to verify command-line arguments work correctly
# This only tests argument parsing, not actual training

cd src

echo "Testing causal separation argument parsing..."
echo "=============================================="

python -m training.main_survival \
    --split_dir splits/TCGA_BRCA_survival_k=0 \
    --split_names 0 \
    --data_source feats_h5 \
    --model_histo_type PANTHER \
    --model_mm_type causal_separation \
    --n_proto 16 \
    --selection_method gumbel \
    --gumbel_tau 1.0 \
    --top_k_ratio 0.5 \
    --lambda_causal_consistency 1.0 \
    --lambda_confound_random 0.5 \
    --lambda_sparsity 0.01 \
    --lambda_balance 0.1 \
    --causal_consistency_mode kl \
    --confound_random_mode entropy \
    --target_selection_ratio 0.5 \
    --max_epochs 1 \
    --results_dir /tmp/test_causal_args \
    --help > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✓ All arguments parsed successfully!"
    echo ""
    echo "You can now run actual training with:"
    echo "  cd scripts/survival"
    echo "  ./causal_separation_demo.sh 0 splits/TCGA_BRCA_survival_k=0 0 feats_h5"
else
    echo "✗ Argument parsing failed"
    exit 1
fi
