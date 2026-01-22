#!/bin/bash
# Multi-seed GORMPO-RealNVP training for Hopper-Medium-Replay-v2
# Usage: bash bash_scr/mult_seed/gormpo_realnvp_hopper_medium_replay.sh

source venv/bin/activate

set -e  # Exit on error

echo "============================================"
echo "Multi-Seed GORMPO-RealNVP Training: Hopper-Medium-Replay"
echo "============================================"
echo ""

# Array of random seeds to test
seeds=(42 123 456)

# Shared results file for all seeds
RESULTS_FILE="results/hopper-medium-replay/realnvp/gormpo_realnvp_multiseed_results.csv"

# Loop through each seed
for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo ">>> Training with seed = $seed"
    echo "=========================================="

    # Step 1: Train RealNVP density estimator for this seed
    echo "Step 1/2: Training RealNVP density estimator (seed $seed)..."
    python realnvp_module/realnvp.py \
        --config configs/realnvp/hopper_medium_replay.yaml \
        --seed $seed \
        --model_save_path /public/gormpo/models/hopper_medium_replay/realnvp_$seed \
        --device cuda:7
    echo "RealNVP training complete for seed $seed"
    echo ""

    # Step 2: Train GORMPO policy using the trained RealNVP model
    echo "Step 2/2: Training GORMPO-RealNVP policy (seed $seed)..."
    python mopo.py \
        --config configs/realnvp/mbpo_hopper_medium_replay.yaml \
        --seed $seed \
        --classifier_model_name /public/gormpo/models/hopper_medium_replay/realnvp_$seed \
        --epoch 1000 \
        --devid 7 \
        --results_output $RESULTS_FILE
    echo "GORMPO-RealNVP training complete for seed $seed"
    echo ""
done

echo "============================================"
echo "All GORMPO-RealNVP multi-seed experiments completed!"
echo "============================================"
echo ""

# Compute normalized scores across all seeds
echo "Computing normalized scores..."
python helpers/normalizer.py $RESULTS_FILE hopper-medium-replay-v2
