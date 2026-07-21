#!/bin/bash
set -e
# COMBO on MCS (Abiomed), 3 seeds.
# Trains dynamics + COMBO policy via OfflineRL-Kit2/run_example/run_combo.py
# Uses the currently active conda environment (activate LEQ2 / etc. before running).
#
# Usage (from GORMPO root):
#   conda activate LEQ2
#   CUDA_VISIBLE_DEVICES=2 bash bash_scr/mult_seed/COMBO_MCS.sh

TASK="${TASK:-abiomed-v0}"
DATASET_PATH="${DATASET_PATH:-}"
ROLLOUT_LENGTH="${ROLLOUT_LENGTH:-5}"
CQL_WEIGHT="${CQL_WEIGHT:-1.0}"
REAL_RATIO="${REAL_RATIO:-0.5}"
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
GORMPO_ABIOMED_DIR="${GORMPO_ABIOMED_DIR:-$GORMPO_ROOT/../GORMPO_abiomed}"

if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "ERROR: no active conda environment (CONDA_PREFIX unset)."
    echo "Activate an env first, e.g.: conda activate LEQ2"
    exit 1
fi

if [ -z "$DATASET_PATH" ]; then
    DATASET_PATH="$GORMPO_ABIOMED_DIR/synthetic_data/SAC_5000eps_stochastic.npz"
fi
if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: MCS dataset not found: $DATASET_PATH"
    exit 1
fi

echo "============================================"
echo "COMBO: $TASK (MCS)"
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

    # Resume: skip finished work. *_sparse dirs are sparse-dataset runs, not ours.
    policy_done=""; dyn_dir=""
    for md in "$OFFLINERLKIT_DIR/log/$TASK/combo/seed_${seed}&timestamp_"*/model; do
        [ -d "$md" ] || continue
        case "$md" in *_sparse/model) continue;; esac
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
        python run_example/run_combo.py \
            --task "$TASK" \
            --seed "$seed" \
            "${DYN_ARGS[@]}" \
            --dataset-path "$DATASET_PATH" \
            --rollout-length "$ROLLOUT_LENGTH" \
            --cql-weight "$CQL_WEIGHT" \
            --real-ratio "$REAL_RATIO" \
            --epoch "$EPOCH"
    )
    echo "  ✓ COMBO complete for seed $seed"
    echo ""
done

echo "Done. Logs under OfflineRL-Kit2/log/${TASK}/combo/"
