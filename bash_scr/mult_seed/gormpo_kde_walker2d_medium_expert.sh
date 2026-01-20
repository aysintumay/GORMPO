#!/bin/bash
# Multi-seed GORMPO-KDE training for Walker2d-Medium-Expert-v2
# Usage: bash bash_scr/mult_seed/gormpo_kde_walker2d_medium_expert.sh

source venv/bin/activate

set -e  # Exit on error

echo "============================================"
echo "Multi-Seed GORMPO-KDE Training: Walker2d-Medium-Expert"
echo "============================================"
echo ""

# Array of random seeds to test
seeds=(42 123 456)

# Shared results file for all seeds
RESULTS_FILE="results/walker2d-medium-expert/kde/gormpo_kde_multiseed_results.csv"

# Loop through each seed
for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo ">>> Training with seed = $seed"
    echo "=========================================="

    # Step 1: Train KDE density estimator for this seed
    echo "Step 1/2: Training KDE density estimator (seed $seed)..."
    python kde_module/kde.py \
        --config configs/kde/walker2d_medium_expert.yaml \
        --seed $seed \
        --save_path /public/gormpo/models/walker2d_medium_expert/kde_$seed \
        --devid 1
    echo "KDE training complete for seed $seed"
    echo ""

    # Step 2: Train GORMPO policy using the trained KDE model
    echo "Step 2/2: Training GORMPO-KDE policy (seed $seed)..."
    python mopo.py \
        --config configs/kde/mbpo_walker2d_medium_expert.yaml \
        --seed $seed \
        --classifier_model_name /public/gormpo/models/walker2d_medium_expert/kde_$seed \
        --epoch 1000 \
        --devid 1 \
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
python helpers/normalizer.py $RESULTS_FILE walker2d-medium-expert-v2
