#!/bin/bash
# Multi-seed MBPO training for Hopper-Medium-Replay-v2
# Usage: bash bash_scr/mult_Seed/mbpo_hopper_medium_replay.sh

set -e  # Exit on error

echo "============================================"
echo "Multi-Seed MBPO Training: Hopper-Medium-Replay"
echo "============================================"
echo ""

# Array of random seeds to test
seeds=(42 123 456)

# Shared results file for all seeds
RESULTS_FILE="results/hopper-medium-replay-v2/mbpo/mbpo_multiseed_results.csv"

# Loop through each seed
for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo ">>> Training MBPO with seed = $seed"
    echo "=========================================="

    python mopo.py \
        --task hopper-medium-replay-v2 \
        --algo-name mbpo \
        --config configs/kde/mbpo_hopper_medium_replay.yaml \
        --dynamics-model-dir 'true' \
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
python helpers/normalizer.py $RESULTS_FILE hopper-medium-replay-v2
