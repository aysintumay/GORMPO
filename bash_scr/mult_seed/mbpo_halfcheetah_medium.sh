#!/bin/bash
# Multi-seed MBPO training for HalfCheetah-Medium-v2
# Usage: bash bash_scr/mult_Seed/mbpo_halfcheetah_medium.sh

set -e  # Exit on error

echo "============================================"
echo "Multi-Seed MBPO Training: HalfCheetah-Medium"
echo "============================================"
echo ""

# Array of random seeds to test
seeds=(42 123 456)

# Shared results file for all seeds
RESULTS_FILE="results/halfcheetah-medium-v2/mbpo/mbpo_multiseed_results.csv"

# Loop through each seed
for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo ">>> Training MBPO with seed = $seed"
    echo "=========================================="

    python mopo.py \
        --task halfcheetah-medium-v2 \
        --algo-name mbpo \
        --dynamics-model-dir 'true' \
        --config configs/kde/mbpo_halfcheetah.yaml \
        --reward-penalty-coef 0.0 \
        --seed $seed \
        --epoch 1000 \
        --rollout-length 5 \
        --devid 2 \
        --results_output $RESULTS_FILE

    echo "MBPO training complete for seed $seed"
    echo ""
done

echo "============================================"
echo "All MBPO multi-seed experiments completed!"
echo "============================================"
echo ""

# Compute normalized scores across all seeds
echo "Computing normalized scores..."
python helpers/normalizer.py $RESULTS_FILE halfcheetah-medium-v2