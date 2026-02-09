#!/bin/bash
# Multi-seed MBPO training for Walker2d-Medium-Expert-v2 (Sparse 72.5%)
# Usage: bash bash_scr/mult_seed/mbpo_walker2d_medium_expert_sparse3.sh

echo "============================================"
echo "Multi-Seed MBPO Training: Walker2d-Medium-Expert-Sparse3"
echo "============================================"
echo ""

# Array of random seeds to test
seeds=(42 123 456)

# Shared results file for all seeds
RESULTS_FILE="results/walker2d-medium-expert-v2_sparse_78/kde/mbpo_multiseed_results.csv"

# Loop through each seed
for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo ">>> Training MBPO with seed = $seed"
    echo "=========================================="

    python mopo.py \
        --config configs/kde/gormpo_walker2d_medium_expert_sparse_3.yaml \
        --data_path /public/d4rl/walker2d_filtered_dataset.pkl \
        --algo-name mbpo \
        --reward-penalty-coef 0.0 \
        --seed $seed \
        --epoch 1000 \
        --devid 5 \
        --results_output $RESULTS_FILE
    echo "✓ MBPO training complete for seed $seed"
    echo ""
done

echo "============================================"
echo "All MBPO multi-seed experiments completed!"
echo "============================================"
echo ""

# Compute normalized scores across all seeds
echo "Computing normalized scores..."
python helpers/normalizer.py $RESULTS_FILE walker2d-medium-expert-v2
