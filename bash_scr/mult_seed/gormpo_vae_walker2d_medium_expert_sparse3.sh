#!/bin/bash
# Multi-seed GORMPO-VAE training for Walker2d-Medium-Expert-v2 (Sparse 72.5%)
# Usage: bash bash_scr/mult_seed/gormpo_vae_walker2d_medium_expert_sparse3.sh

echo "============================================"
echo "Multi-Seed GORMPO-VAE Training: Walker2d-Medium-Expert-Sparse3"
echo "============================================"
echo ""

# Array of random seeds to test
seeds=(42 123 456)

# Shared results file for all seeds
RESULTS_FILE="results/walker2d-medium-expert-v2_sparse_78/vae/gormpo_vae_multiseed_results.csv"

# Loop through each seed
for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo ">>> Training with seed = $seed"
    echo "=========================================="

    # Step 1: Train VAE density estimator for this seed
    echo "Step 1/2: Training VAE density estimator (seed $seed)..."
    python vae_module/vae.py \
        --config configs/vae/walker2d_medium_expert_sparse_3.yaml \
        --seed $seed \
        --model_save_path /public/gormpo/models/walker2d_medium_expert_sparse_3/vae_$seed \
        --device cuda:4
    echo "✓ VAE training complete for seed $seed"
    echo ""

    # Step 2: Train GORMPO policy using the trained VAE model
    echo "Step 2/2: Training GORMPO-VAE policy (seed $seed)..."
    python mopo.py \
        --config configs/vae/gormpo_walker2d_medium_expert_sparse_3.yaml \
        --seed $seed \
        --classifier_model_name /public/gormpo/models/walker2d_medium_expert_sparse_3/vae_$seed \
        --epoch 1000 \
        --devid 4 \
        --results_output $RESULTS_FILE
    echo "✓ GORMPO-VAE training complete for seed $seed"
    echo ""
done

echo "============================================"
echo "All GORMPO-VAE multi-seed experiments completed!"
echo "============================================"
echo ""

# Compute normalized scores across all seeds
echo "Computing normalized scores..."
python helpers/normalizer.py $RESULTS_FILE walker2d-medium-expert-v2
