#!/bin/bash

set -e  # Exit on error
echo "============================================"
echo "Training GORMPO with NeuralODE on Walker2d-Medium-Expert-v2 (Sparse 73%)"
echo "Testing multiple reward penalty coefficients"
echo "============================================"
echo ""
RESULTS_FILE="results/hyperparameter/walker2d-medium-expert-sparse/gormpo_neuralODE_multilambda_results.csv"

# Array of penalty coefficients to test
penalty_coeffs=(0.1 0.3 0.5 0.7)
seeds=(42 123)
for coef in "${penalty_coeffs[@]}"; do
    # Loop through each seed
    for seed in "${seeds[@]}"; do
        echo "=========================================="
        echo ">>> Training with reward-penalty-coef = $coef"
        echo "=========================================="

        python mopo.py \
            --config configs/neuralODE/gormpo_walker2d_medium_expert_sparse_3.yaml \
            --seed $seed \
            --reward-penalty-coef $coef \
            --epoch 2500 \
            --devid 6 \
            --results_output $RESULTS_FILE \
            --dynamics-model-dir 'true'

        echo "✓ Training complete for penalty coefficient $coef"
        echo ""
    done
done

echo "============================================"
echo "All GORMPO-NeuralODE experiments completed!"
echo "============================================"
# Compute normalized scores across all seeds
echo "Computing normalized scores..."
python helpers/normalizer.py $RESULTS_FILE walker2d-medium-expert-v2
