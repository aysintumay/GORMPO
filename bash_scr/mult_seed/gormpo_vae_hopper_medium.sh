#!/bin/bash
# Multi-seed GORMPO-VAE training for Hopper-Medium-v2
# Usage: bash bash_scr/mult_seed/gormpo_vae_hopper_medium.sh

source venv/bin/activate

set -e  # Exit on error

echo "============================================"echo "Multi-Seed GORMPO-VAE Training: Hopper-Medium"
echo "============================================"
echo ""

# Array of random seeds to test
seeds=(1 2 3)

# Shared results file for all seeds
RESULTS_FILE="results/hopper-medium/vae/gormpo_vae_multiseed_results.csv"

# Loop through each seed
for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo ">>> Training with seed = $seed"
    echo "=========================================="

    # Step 1: Train VAE density estimator for this seed
    # echo "Step 1/2: Training VAE density estimator (seed $seed)..."
    # python vae_module/vae.py \
    #     --config configs/vae/hopper.yaml \
    #     --seed $seed \
    #     --model_save_path /public/gormpo/models/hopper_medium/vae_$seed \
    #     --device cuda:2
    # echo "VAE training complete for seed $seed"
    # echo ""

    # Step 2: Train GORMPO policy using the trained VAE model
    echo "Step 2/2: Training GORMPO-VAE policy (seed $seed)..."
    python mopo.py \
        --config configs/vae/mbpo_hopper.yaml \
        --seed $seed \
        --classifier_model_name /public/gormpo/models/hopper_medium/vae \
        --epoch 1000 \
        --devid 2 \
        --results_output $RESULTS_FILE
    echo "GORMPO-VAE training complete for seed $seed"
    echo ""
done

echo "============================================"
echo "All GORMPO-VAE multi-seed experiments completed!"
echo "============================================"
echo ""

# Compute normalized scores across all seeds
echo "Computing normalized scores..."
python helpers/normalizer.py $RESULTS_FILE hopper-medium-v2
