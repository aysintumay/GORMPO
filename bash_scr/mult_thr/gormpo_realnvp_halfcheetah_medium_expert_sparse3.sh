#!/bin/bash
# Threshold sensitivity sweep: GORMPO-RealNVP, HalfCheetah-Medium-Expert-v2 (Sparse 72.5%)
# Usage: bash bash_scr/mult_thr/gormpo_realnvp_halfcheetah_medium_expert_sparse3.sh

source venv/bin/activate

set -e

echo "============================================"
echo "Threshold Sensitivity GORMPO-RealNVP: HalfCheetah-Medium-Expert-Sparse3"
echo "============================================"
echo ""

seeds=(42 123 456)
thresholds=(1 5 10 15 20)

RESULTS_FILE="results/thr_experiments/realnvp/halfcheetah_medium_expert_sparse3_individual.csv"
AGG_FILE="results/thr_experiments/realnvp/halfcheetah_medium_expert_sparse3_aggregated.csv"
mkdir -p "$(dirname $RESULTS_FILE)"
echo "Results will be saved to: $RESULTS_FILE"
echo ""

for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo ">>> Seed = $seed: computing threshold candidates"
    echo "=========================================="

    # Step 1: compute & save threshold_candidates into the RealNVP model metadata
    python realnvp_module/realnvp.py \
        --config configs/realnvp/halfcheetah_medium_expert_sparse_3.yaml \
        --seed $seed \
        --compute_thresholds_only \
        --model_save_path /public/gormpo/models/halfcheetah_medium_expert_sparse_3/realnvp_$seed 

    echo "✓ Threshold candidates saved for seed=$seed"
    echo ""

    for thr in "${thresholds[@]}"; do
        echo "=========================================="
        echo ">>> Training with seed=$seed, threshold_percentile=$thr"
        echo "=========================================="

        # Step 2: train GORMPO-RealNVP policy using the pre-computed threshold
        python mopo.py \
            --config configs/realnvp/gormpo_halfcheetah_medium_expert_sparse_3.yaml \
            --seed $seed \
            --classifier_model_name /public/gormpo/models/halfcheetah_medium_expert_sparse_3/realnvp_$seed \
            --threshold_percentile $thr \
            --epoch 1000 \
            --devid 0 \
            --dynamics-model-dir 'true' \
            --results_output $RESULTS_FILE

        echo "✓ Done seed=$seed thr=$thr"
        echo ""
    done
done

echo "============================================"
echo "All GORMPO-RealNVP threshold sensitivity experiments completed!"
echo "Results saved to: $RESULTS_FILE"
echo "============================================"
echo ""

echo "Computing normalized scores..."
python helpers/normalizer.py $RESULTS_FILE halfcheetah-medium-expert-v2

echo "Aggregating results across seeds..."
python helpers/aggregate_thr_results.py $RESULTS_FILE $AGG_FILE
