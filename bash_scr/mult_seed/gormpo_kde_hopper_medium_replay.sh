#!/bin/bash
# Multi-seed GORMPO-KDE training for Hopper-Medium-Replay-v2
# Usage: bash bash_scr/mult_seed/gormpo_kde_hopper_medium_replay.sh

source venv/bin/activate

set -e  # Exit on error

echo "============================================"
echo "Multi-Seed GORMPO-KDE Training: Hopper-Medium-Replay"
echo "============================================"
echo ""

# Array of random seeds to test
seeds=(42 123 456)

# Shared results file for all seeds
RESULTS_FILE="results/hopper-medium-replay/kde/gormpo_kde_multiseed_results.csv"

# Loop through each seed
for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo ">>> Training with seed = $seed"
    echo "=========================================="

    # Step 1: Train KDE density estimator for this seed
    echo "Step 1/2: Training KDE density estimator (seed $seed)..."
    python kde_module/kde.py \
        --config configs/kde/hopper_medium_replay.yaml \
        --seed $seed \
        --save_path /public/gormpo/models/hopper_medium_replay/kde_$seed \
        --devid 7
    echo "KDE training complete for seed $seed"
    echo ""

    # Step 2: Train GORMPO policy using the trained KDE model
    echo "Step 2/2: Training GORMPO-KDE policy (seed $seed)..."
    python mopo.py \
        --config configs/kde/mbpo_hopper_medium_replay.yaml \
        --seed $seed \
        --classifier_model_name /public/gormpo/models/hopper_medium_replay/kde_$seed \
        --epoch 1000 \
        --devid 7 \
        --results_output $RESULTS_FILE
    echo "GORMPO-KDE training complete for seed $seed"
    echo ""
done

echo "============================================"
echo "All GORMPO-KDE multi-seed experiments completed!"
echo "============================================"
echo ""

# Compute normalized scores across all seeds
echo "Computing normalized scores..."
python helpers/normalizer.py $RESULTS_FILE hopper-medium-replay-v2
