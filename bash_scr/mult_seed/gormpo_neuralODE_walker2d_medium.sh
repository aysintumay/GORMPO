#!/bin/bash
# Multi-seed GORMPO-NeuralODE training for Walker2d-Medium-v2
# Usage: bash bash_scr/mult_seed/gormpo_neuralODE_walker2d_medium.sh

source venv/bin/activate

set -e  # Exit on error

echo "============================================"
echo "Multi-Seed GORMPO-NeuralODE Training: Walker2d-Medium"
echo "============================================"
echo ""

# Array of random seeds to test
seeds=(42 123 456)

# Shared results file for all seeds
RESULTS_FILE="results/walker2d-medium/neuralODE/gormpo_neuralODE_multiseed_results.csv"

# Loop through each seed
for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo ">>> Training with seed = $seed"
    echo "=========================================="

    # Step 1: Train Neural ODE density model for this seed
    # echo "Step 1/2: Training Neural ODE density model (seed $seed)..."
    # python neuralODE/neural_ode_density.py \
    #     --config configs/neuralODE/walker2d_train.yaml \
    #     --seed $seed \
    #     --epochs 10 \
    #     --out /public/gormpo/models/walker2d_medium/neuralODE_$seed
    # echo "Neural ODE training complete for seed $seed"
    # echo ""

    # Step 2: Train GORMPO policy using the trained Neural ODE model
    echo "Step 2/2: Training GORMPO-NeuralODE policy (seed $seed)..."
    python mopo.py \
        --config configs/neuralODE/walker2d_normal.yaml \
        --seed $seed \
        --epoch 1000 \
        --devid 4 \
        --results_output $RESULTS_FILE
    echo "GORMPO-NeuralODE training complete for seed $seed"
    echo ""
done

echo "============================================"
echo "All GORMPO-NeuralODE multi-seed experiments completed!"
echo "============================================"
echo ""

# Compute normalized scores across all seeds
echo "Computing normalized scores..."
python helpers/normalizer.py $RESULTS_FILE walker2d-medium-v2
