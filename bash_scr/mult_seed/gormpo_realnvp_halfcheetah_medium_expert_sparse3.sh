#!/bin/bash
# Multi-seed GORMPO-RealNVP training for HalfCheetah-Medium-Expert-v2 (Sparse 72.5%)
# Usage: bash bash_scr/mult_seed/gormpo_realnvp_halfcheetah_medium_expert_sparse3.sh

echo "============================================"
echo "Multi-Seed GORMPO-RealNVP Training: HalfCheetah-Medium-Expert-Sparse3"
echo "============================================"
echo ""

# Array of random seeds to test
seeds=(42 123 456)

# Shared results file for all seeds
RESULTS_FILE="results/halfcheetah-medium-expert-v2_sparse_72.5/realnvp/gormpo_realnvp_multiseed_results.csv"

# Loop through each seed
for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo ">>> Training with seed = $seed"
    echo "=========================================="

    # Step 1: Train RealNVP density estimator for this seed
    echo "Step 1/2: Training RealNVP density estimator (seed $seed)..."
    python realnvp_module/realnvp.py \
        --config configs/realnvp/halfcheetah_medium_expert_sparse_3.yaml \
        --seed $seed \
        --model_save_path /public/gormpo/models/halfcheetah_medium_expert_sparse_3/realnvp_$seed \
        --device cuda:0
    echo "✓ RealNVP training complete for seed $seed"
    echo ""

    # Step 2: Train GORMPO policy using the trained RealNVP model
    echo "Step 2/2: Training GORMPO-RealNVP policy (seed $seed)..."
    python mopo.py \
        --config configs/realnvp/gormpo_halfcheetah_medium_expert_sparse_3.yaml \
        --seed $seed \
        --classifier_model_name /public/gormpo/models/halfcheetah_medium_expert_sparse_3/realnvp_$seed \
        --epoch 1000 \
        --devid 0 \
        --results_output $RESULTS_FILE
    echo "✓ GORMPO-RealNVP training complete for seed $seed"
    echo ""
done

echo "============================================"
echo "All GORMPO-RealNVP multi-seed experiments completed!"
echo "============================================"
echo ""

# Compute normalized scores across all seeds
echo "Computing normalized scores..."
python helpers/normalizer.py $RESULTS_FILE halfcheetah-medium-expert-v2
