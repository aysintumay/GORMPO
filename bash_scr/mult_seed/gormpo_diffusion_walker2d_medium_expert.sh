#!/bin/bash
# Multi-seed GORMPO-Diffusion training for Walker2d-Medium-Expert-v2
# Usage: bash bash_scr/mult_seed/gormpo_diffusion_walker2d_medium_expert.sh

source venv/bin/activate

set -e  # Exit on error

echo "============================================"
echo "Multi-Seed GORMPO-Diffusion Training: Walker2d-Medium-Expert"
echo "============================================"
echo ""

# Array of random seeds to test
seeds=(42 123 456)

# Shared results file for all seeds
RESULTS_FILE="results/walker2d-medium-expert/diffusion/gormpo_diffusion_multiseed_results.csv"

# Loop through each seed
for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo ">>> Training with seed = $seed"
    echo "=========================================="

    # Step 1: Train Diffusion model for this seed
    echo "Step 1/2: Training Diffusion model (seed $seed)..."
    python diffusion/ddim_training_unconditional.py \
        --config diffusion/configs/unconditional_training/walker2d_mlp_expert.yaml \
        --seed $seed \
        --epochs 100 \
        --out /public/gormpo/models/walker2d_medium_expert/diffusion_$seed
    echo "Diffusion model training complete for seed $seed"
    echo ""

    # Step 2: Train GORMPO policy using the trained Diffusion model
    echo "Step 2/2: Training GORMPO-Diffusion policy (seed $seed)..."
    python mopo.py \
        --config configs/diffusion/walker2d_medium_expert.yaml \
        --seed $seed \
        --classifier_model_name /public/gormpo/models/walker2d_medium_expert/diffusion_$seed/checkpoint.pt \
        --epoch 1000 \
        --devid 6 \
        --results_output $RESULTS_FILE
    echo "GORMPO-Diffusion training complete for seed $seed"
    echo ""
done

echo "============================================"
echo "All GORMPO-Diffusion multi-seed experiments completed!"
echo "============================================"
echo ""

# Compute normalized scores across all seeds
echo "Computing normalized scores..."
python helpers/normalizer.py $RESULTS_FILE walker2d-medium-expert-v2
