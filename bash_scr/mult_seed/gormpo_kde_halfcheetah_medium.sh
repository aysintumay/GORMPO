#!/bin/bash
# Multi-seed GORMPO-KDE training for HalfCheetah-Medium-v2
# Usage: bash bash_scr/mult_seed/gormpo_kde_halfcheetah_medium.sh

source venv/bin/activate

set -e  # Exit on error

echo "============================================"
echo "Multi-Seed GORMPO-KDE Training: HalfCheetah-Medium"
echo "============================================"
echo ""

# Array of random seeds to test
seeds=(42 123 456)

# Shared results file for all seeds
RESULTS_FILE="results/halfcheetah-medium/kde/gormpo_kde_multiseed_results.csv"

# Loop through each seed
for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo ">>> Training with seed = $seed"
    echo "=========================================="

    # Step 1: Train KDE density estimator for this seed
    echo "Step 1/2: Training KDE density estimator (seed $seed)..."
    python kde_module/kde.py \
        --config configs/kde/halfcheetah_medium.yaml \
        --seed $seed \
        --save_path /public/gormpo/models/halfcheetah_medium/kde_$seed \
        --devid 1
    echo "KDE training complete for seed $seed"
    echo ""

    # Step 2: Train GORMPO policy using the trained KDE model
    echo "Step 2/2: Training GORMPO-KDE policy (seed $seed)..."
    python mopo.py \
        --config configs/kde/mbpo_halfcheetah.yaml \
        --seed $seed \
        --classifier_model_name /public/gormpo/models/halfcheetah_medium/kde_$seed \
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
python helpers/normalizer.py $RESULTS_FILE halfcheetah-medium-v2
