#!/bin/bash

set -e  # Exit on error
echo "============================================"
echo "Threshold Sensitivity GORMPO-Diffusion: HalfCheetah-Medium-Expert-Sparse3"
echo "============================================"
echo ""

# Array of random seeds to test
seeds=(42)

# Array of threshold percentiles to sweep
thresholds=(1 5 10 15 20)

# Define shared results file path
timestamp=$(date +"%m%d_%H%M%S")
results_dir="results/thr_experiments/diffusion"
results_file="${results_dir}/halfcheetah_medium_expert_sparse3_individual.csv"
agg_file="${results_dir}/halfcheetah_medium_expert_sparse3_aggregated.csv"

mkdir -p "$results_dir"
echo "Results will be saved to: $results_file"
echo ""

# Loop through each seed and threshold
for seed in "${seeds[@]}"; do
    for thr in "${thresholds[@]}"; do
        echo "=========================================="
        echo ">>> Training with seed = $seed, threshold_percentile = $thr"
        echo "=========================================="

        python mopo.py \
            --config configs/diffusion/gormpo_halfcheetah_medium_expert_sparse_3.yaml \
            --seed $seed \
            --epoch 1 \
            --devid 0 \
            --penalty_type tanh \
            --threshold_percentile $thr \
            --results_output $results_file

        echo "✓ Done seed=$seed thr=$thr"
        echo ""
    done
done

echo "============================================"
echo "All GORMPO-Diffusion threshold sensitivity experiments completed!"
echo "Results saved to: $results_file"
echo "============================================"

echo "Aggregating results across seeds..."
python helpers/aggregate_thr_results.py $results_file $agg_file
