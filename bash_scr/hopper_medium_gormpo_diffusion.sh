#!/bin/bash

set -e  # Exit on error
echo "============================================"
echo "Training GORMPO with DDPM on Hopper-Medium"
echo "Testing multiple reward penalty coefficients"
echo "============================================"
echo ""

# Array of penalty coefficients to test
penalty_coeffs=( 0.8 0.5 0.3  0.1 0.05)

# Loop through each penalty coefficient
for coef in "${penalty_coeffs[@]}"; do
    echo "=========================================="
    echo ">>> Training with reward-penalty-coef = $coef"
    echo "=========================================="

    python mopo.py \
        --config configs/diffusion/hopper_normal.yaml \
        --reward-penalty-coef $coef \
        --epoch 500 \
        --devid 4

    echo "✓ Training complete for penalty coefficient $coef"
    echo ""
done

echo "============================================"
echo "All GORMPO-DDPM experiments completed!"
echo "============================================"
