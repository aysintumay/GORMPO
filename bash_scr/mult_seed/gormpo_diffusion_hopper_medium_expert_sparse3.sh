#!/bin/bash
# Multi-seed GORMPO-Diffusion training for Hopper-Medium-Expert-v2 (Sparse 78%)
# Usage: bash bash_scr/mult_seed/gormpo_diffusion_hopper_medium_expert_sparse3.sh

echo "============================================"
echo "Multi-Seed GORMPO-Diffusion Training: Hopper-Medium-Expert-Sparse3"
echo "============================================"
echo ""

# Array of random seeds to test
seeds=(42 123 456)

# Shared results file for all seeds
RESULTS_FILE="results/hopper-medium-expert-v2_sparse_78/diffusion/gormpo_diffusion_multiseed_results.csv"

# Loop through each seed
for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo ">>> Training with seed = $seed"
    echo "=========================================="

    # Train GORMPO policy using the pre-trained Diffusion model
    echo "Training GORMPO-Diffusion policy (seed $seed)..."
    python mopo.py \
        --config configs/diffusion/gormpo_hopper_medium_expert_sparse_3.yaml \
        --seed $seed \
        --epoch 1000 \
        --devid 0 \
        --results_output $RESULTS_FILE
    echo "✓ GORMPO-Diffusion training complete for seed $seed"
    echo ""
done

echo "============================================"
echo "All GORMPO-Diffusion multi-seed experiments completed!"
echo "============================================"
echo ""