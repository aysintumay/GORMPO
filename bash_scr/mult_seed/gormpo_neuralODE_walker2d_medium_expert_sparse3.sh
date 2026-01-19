#!/bin/bash
# Multi-seed GORMPO-NeuralODE training for Walker2d-Medium-Expert-v2 (Sparse 73%)
# Usage: bash bash_scr/mult_seed/gormpo_neuralODE_walker2d_medium_expert_sparse3.sh

source venv/bin/activate

set -e  # Exit on error

echo "============================================"
echo "Multi-Seed GORMPO-NeuralODE Training: Walker2d-Medium-Expert-Sparse3"
echo "============================================"
echo ""

# Array of random seeds to test
seeds=(42 123 456)

# Shared results file for all seeds
RESULTS_FILE="results/walker2d-medium-expert-v2_sparse_73/neuralODE/gormpo_neuralODE_multiseed_results.csv"

# Loop through each seed
for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo ">>> Training with seed = $seed"
    echo "=========================================="

    # Step 1: Train Neural ODE density model for this seed
    echo "Step 1/2: Training Neural ODE density model (seed $seed)..."
    python neuralODE/neural_ode_density.py \
        --config configs/neuralODE/walker2d_medium_expert_sparse_3_train.yaml \
        --seed $seed \
        --epochs 10 \
        --out /public/gormpo/models/walker2d_medium_expert_sparse_3/neuralODE_$seed
    echo "✓ Neural ODE training complete for seed $seed"
    echo ""

    # Step 2: Train GORMPO policy using the trained Neural ODE model
    echo "Step 2/2: Training GORMPO-NeuralODE policy (seed $seed)..."
    python mopo.py \
        --config configs/neuralODE/gormpo_walker2d_medium_expert_sparse_3.yaml \
        --seed $seed \
        --classifier_model_name /public/gormpo/models/walker2d_medium_expert_sparse_3/neuralODE_$seed \
        --epoch 1000 \
        --devid 0 \
        --results_output $RESULTS_FILE
    echo "✓ GORMPO-NeuralODE training complete for seed $seed"
    echo ""
done

echo "============================================"
echo "All GORMPO-NeuralODE multi-seed experiments completed!"
echo "============================================"
echo ""

# Compute normalized scores across all seeds
echo "Computing normalized scores..."
python helpers/normalizer.py $RESULTS_FILE walker2d-medium-expert-v2
