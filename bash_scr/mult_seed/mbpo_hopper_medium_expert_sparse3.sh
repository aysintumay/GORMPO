#!/bin/bash
# Multi-seed MBPO training for Hopper-Medium-Expert-v2 (Sparse 72.5%)
# Usage: bash bash_scr/mult_seed/mbpo_hopper_medium_expert_sparse3.sh

echo "============================================"
echo "Multi-Seed MBPO Training: Hopper-Medium-Expert-Sparse3"
echo "============================================"
echo ""

# Array of random seeds to test
seeds=(2 3 4)

# Shared results file for all seeds
RESULTS_FILE="results/hopper-medium-expert-v2_sparse_73/kde/mbpo_multiseed_results.csv"

# Loop through each seed
for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo ">>> Training MBPO with seed = $seed"
    echo "=========================================="

    python mopo.py \
        --config configs/kde/gormpo_hopper_medium_expert_sparse_3.yaml \
        --algo-name mbpo \
        --reward-penalty-coef 0.0 \
        --rollout-length 5 \
        --seed $seed \
        --epoch 3000 \
        --devid 6 \
        --results_output $RESULTS_FILE
            # --dynamics-model-dir 'true' \

    echo "✓ MBPO training complete for seed $seed"
    echo ""
done

echo "============================================"
echo "All MBPO multi-seed experiments completed!"
echo "============================================"
echo ""

# Compute normalized scores across all seeds
echo "Computing normalized scores..."
python helpers/normalizer.py $RESULTS_FILE hopper-medium-expert-v2
