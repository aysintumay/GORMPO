#!/bin/bash
# Multi-seed GORMPO-Diffusion training for Hopper-Medium-Expert-v2 (Sparse 78%)
# Usage: bash bash_scr/mult_seed/gormpo_diffusion_hopper_medium_expert_sparse3.sh

source venv/bin/activate

set -e  # Exit on error

echo "============================================"
echo "Multi-Seed GORMPO-Diffusion Training: Hopper-Medium-Expert-Sparse3"
echo "============================================"
echo ""

# Array of random seeds to test
seeds=(42 123 456)

# Shared results file for all seeds
RESULTS_FILE="results/hopper-medium-expert-v2_sparse_78/diffusion/gormpo_diffusion_multiseed_results.csv"

# Loop through each seed
for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo ">>> Training with seed = $seed"
    echo "=========================================="

    # Step 1: Train Diffusion model for this seed
    # echo "Step 1/2: Training Diffusion model (seed $seed)..."
    # python diffusion/ddim_training_unconditional.py \
    #     --config diffusion/configs/unconditional_training/hopper_mlp_expert_sparse_78.yaml \
    #     --seed $seed \
    #     --epochs 100 \
    #     --device cuda:1 \
    #     --out /public/gormpo/models/hopper_medium_expert_sparse_3/diffusion_$seed
    # echo "✓ Diffusion model training complete for seed $seed"
    # echo ""

    # Step 2: Train GORMPO policy using the trained Diffusion model
    echo "Step 2/2: Training GORMPO-Diffusion policy (seed $seed)..."
    python mopo.py \
        --config configs/diffusion/gormpo_hopper_medium_expert_sparse_3.yaml \
        --seed $seed \
        --dynamics-model-dir 'true' \
        --rollout-length 5 \
        --classifier_model_name /public/gormpo/models/hopper_medium_expert_sparse_3/diffusion_$seed/checkpoint.pt \
        --epoch 1000 \
        --devid 1 \
        --results_output $RESULTS_FILE
    echo "✓ GORMPO-Diffusion training complete for seed $seed"
    echo ""
done

echo "============================================"
echo "All GORMPO-Diffusion multi-seed experiments completed!"
echo "============================================"
echo ""

# Compute normalized scores across all seeds
echo "Computing normalized scores..."
python helpers/normalizer.py $RESULTS_FILE hopper-medium-expert-v2