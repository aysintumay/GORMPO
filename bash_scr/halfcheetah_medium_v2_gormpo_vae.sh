#!/bin/bash

set -e  # Exit on error
echo "============================================"
echo "Training GORMPO with VAE on HalfCheetah-Medium-v2"
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
        --config configs/vae/mbpo_halfcheetah.yaml \
        --reward-penalty-coef $coef \
        --epoch 500 \
        --devid 7

    echo "✓ Training complete for penalty coefficient $coef"
    echo ""
done

echo "============================================"
echo "All GORMPO-VAE experiments completed!"
echo "============================================"
