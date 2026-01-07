#!/bin/bash

set -e  # Exit on error
echo "============================================"
echo "Training GORMPO with RealNVP on Hopper-Medium"
echo "Testing multiple reward penalty coefficients"
echo "============================================"
echo ""

# Array of penalty coefficients to test
penalty_coeffs=(0.05 0.1 0.3 0.5 0.8)

# Loop through each penalty coefficient
for coef in "${penalty_coeffs[@]}"; do
    echo "=========================================="
    echo ">>> Training with reward-penalty-coef = $coef"
    echo "=========================================="

    python mopo.py \
        --config configs/realnvp/hopper_linear_normal.yaml \
        --reward-penalty-coef $coef \
        --epoch 500 \
        --devid 7

    echo "✓ Training complete for penalty coefficient $coef"
    echo ""
done

echo "============================================"
echo "All GORMPO-RealNVP experiments completed!"
echo "============================================"
