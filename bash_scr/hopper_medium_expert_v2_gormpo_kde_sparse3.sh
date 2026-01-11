#!/bin/bash

set -e  # Exit on error
echo "============================================"
echo "Training GORMPO with KDE on Hopper-Medium-Expert-v2"
echo "Testing multiple reward penalty coefficients"
echo "============================================"
echo ""

# Array of penalty coefficients to test
penalty_coeffs=( 0.5 0.8)

# Loop through each penalty coefficient
for coef in "${penalty_coeffs[@]}"; do
    echo "=========================================="
    echo ">>> Training with reward-penalty-coef = $coef"
    echo "=========================================="

    python mopo.py \
        --config configs/kde/gormpo_hopper_medium_expert_sparse_3.yaml \
        --reward-penalty-coef $coef \
        --epoch 1000 \
        --devid 3

    echo "✓ Training complete for penalty coefficient $coef"
    echo ""
done

echo "============================================"
echo "All GORMPO-VAE experiments completed!"
echo "============================================"
