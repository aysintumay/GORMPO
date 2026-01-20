#!/bin/bash

set -e  # Exit on error
echo "============================================"
echo "Training GORMPO with NeuralODE on Walker2d-Medium-Expert-v2 (Sparse 73%)"
echo "Testing multiple reward penalty coefficients"
echo "============================================"
echo ""

# Array of penalty coefficients to test
penalty_coeffs=(0.05 0.5)

# Loop through each penalty coefficient
for coef in "${penalty_coeffs[@]}"; do
    echo "=========================================="
    echo ">>> Training with reward-penalty-coef = $coef"
    echo "=========================================="

    python mopo.py \
        --config configs/neuralODE/gormpo_walker2d_medium_expert_sparse_3.yaml \
        --reward-penalty-coef $coef \
        --epoch 1000 \
        --devid 3 \
        --dynamics-model-dir dynamics

    echo "✓ Training complete for penalty coefficient $coef"
    echo ""
done

echo "============================================"
echo "All GORMPO-NeuralODE experiments completed!"
echo "============================================"
