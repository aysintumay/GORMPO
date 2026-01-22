#!/bin/bash
# Multi-seed MBPO training for Hopper-Medium-v2
# Usage: bash bash_scr/mult_Seed/mbpo_hopper_medium.sh

set -e  # Exit on error

echo "============================================"
echo "Multi-Seed MBPO Training: Hopper-Medium"
echo "============================================"
echo ""

# Array of random seeds to test
seeds=(42 123 456)

# Shared results file for all seeds
RESULTS_FILE="results/hopper-medium-v2/mbpo/mbpo_multiseed_results.csv"

# Loop through each seed
for seed in "${seeds[@]}"; do

    python mopo.py \
        --task hopper-medium-v2 \
        --algo-name mbpo \
        --config configs/kde/mbpo_hopper.yaml \
        --dynamics-model-dir 'true' \
        --reward-penalty-coef 0.0 \
        --seed $seed \
        --epoch 1000 \
        --rollout-length 5 \
        --devid 1 \
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
python helpers/normalizer.py $RESULTS_FILE hopper-medium-v2
