#!/bin/bash
set -e
# COMBO on Walker2d-Medium-Expert (sparse D4RL), 3 seeds.
# Hypers (paper / run_combo comments): rollout-length=1, cql-weight=5.0, real-ratio=0.8
# Uses the currently active conda environment (activate LEQ2 / etc. before running).
#
# Identical to COMBO_WALKER2D.sh except it trains on the sparse dataset via
# run_combo_in_sparse.py --dataset-path (instead of the env's default D4RL data).
#
# Usage (from GORMPO root):
#   conda activate LEQ2
#   CUDA_VISIBLE_DEVICES=2 bash bash_scr/mult_seed/COMBO_WALKER2D_SPARSE.sh

TASK="${TASK:-walker2d-medium-expert-v2}"
DATASET_PATH="${DATASET_PATH:-/public/d4rl/sparse_datasets/walker2d_medium_expert_sparse_73.pkl}"
ROLLOUT_LENGTH="${ROLLOUT_LENGTH:-1}"
CQL_WEIGHT="${CQL_WEIGHT:-5.0}"
REAL_RATIO="${REAL_RATIO:-0.8}"
EPOCH="${EPOCH:-1000}"

DEVID="${DEVID:-0}"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$DEVID"
fi

if [ -z "${SEEDS+x}" ] || [ -z "$SEEDS" ]; then
    SEEDS=(42 123 456)
else
    # shellcheck disable=SC2206
    SEEDS=($SEEDS)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GORMPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OFFLINERLKIT_DIR="${OFFLINERLKIT_DIR:-$GORMPO_ROOT/../OfflineRL-Kit2}"

if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "ERROR: no active conda environment (CONDA_PREFIX unset)."
    echo "Activate an env first, e.g.: conda activate LEQ2"
    exit 1
fi

if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: sparse dataset not found: $DATASET_PATH"
    exit 1
fi

echo "============================================"
echo "COMBO: $TASK (sparse D4RL medium-expert)"
echo "  Dataset:       $DATASET_PATH"
echo "  OfflineRL-Kit: $OFFLINERLKIT_DIR"
echo "  Conda env:     ${CONDA_DEFAULT_ENV:-$CONDA_PREFIX}"
echo "  Python:        $(command -v python)"
echo "  Seeds:         ${SEEDS[*]}"
echo "  rollout-length=$ROLLOUT_LENGTH  cql-weight=$CQL_WEIGHT  real-ratio=$REAL_RATIO"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "============================================"
echo ""

for seed in "${SEEDS[@]}"; do
    echo "=========================================="
    echo ">>> seed = $seed"
    echo "=========================================="

    # Resume: *_sparse log dirs are this script's runs (tagged by run_combo_in_sparse.py).
    # Glob sorts lexically, so the last match wins = newest run.
    policy_done=""; dyn_dir=""
    for md in "$OFFLINERLKIT_DIR/log/$TASK/combo/seed_${seed}&timestamp_"*_sparse/model; do
        [ -d "$md" ] || continue
        if [ -f "$md/policy.pth" ]; then policy_done="$md"
        elif [ -f "$md/dynamics.pth" ]; then dyn_dir="$md"; fi
    done

    if [ -n "$policy_done" ]; then
        echo "  ✓ policy already trained ($policy_done) — skipping seed."
        echo ""
        continue
    fi

    DYN_ARGS=()
    if [ -n "$dyn_dir" ]; then
        echo "  ↻ dynamics already trained ($dyn_dir) — skipping dynamics, policy only."
        DYN_ARGS=(--load-dynamics-path "$dyn_dir")
    fi

    (
        cd "$OFFLINERLKIT_DIR"
        python run_example/run_combo_in_sparse.py \
            --task "$TASK" \
            --seed "$seed" \
            --dataset-path "$DATASET_PATH" \
            "${DYN_ARGS[@]}" \
            --rollout-length "$ROLLOUT_LENGTH" \
            --cql-weight "$CQL_WEIGHT" \
            --real-ratio "$REAL_RATIO" \
            --epoch "$EPOCH"
    )
    echo "  ✓ COMBO complete for seed $seed"
    echo ""
done

echo "Done. Logs under OfflineRL-Kit2/log/${TASK}/combo/"
