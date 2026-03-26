#!/bin/bash

set -e  # Exit on error
echo "============================================"
echo "Training GORMPO with Diffusion on Hopper-Medium-Expert-v2 (Sparse 78%)"
echo "Testing multiple reward penalty coefficients"
echo "============================================"
echo ""
RESULTS_FILE="results/hyperparameter/hopper-medium-expert-sparse/gormpo_diffusion_multilambda_results.csv"

# Array of penalty coefficients to test
penalty_coeffs=( 0.5 0.7)
seeds=(42 123)
for coef in "${penalty_coeffs[@]}"; do
    # Loop through each seed
    for seed in "${seeds[@]}"; do
        echo "=========================================="
        echo ">>> Training with reward-penalty-coef = $coef"
        echo "=========================================="

        python mopo.py \
            --config configs/diffusion/gormpo_hopper_medium_expert_sparse_3.yaml \
            --seed $seed \
            --classifier_model_name /public/gormpo/models/hopper_medium_expert_sparse_3/diffusion_$seed/checkpoint.pt \
            --reward-penalty-coef $coef \
            --epoch 2500 \
            --devid 4 \
            --results_output $RESULTS_FILE \
            --dynamics-model-dir 'true'

        echo "✓ Training complete for penalty coefficient $coef"
        echo ""
    done
done

echo "============================================"
echo "All GORMPO-Diffusion experiments completed!"
echo "============================================"
# Compute normalized scores across all seeds
echo "Computing normalized scores..."
python helpers/normalizer.py $RESULTS_FILE hopper-medium-expert-v2
