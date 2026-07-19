#!/bin/bash
set -e
# COMBO on HalfCheetah-Medium-Expert (full D4RL), 3 seeds.
# Hypers (paper / run_combo comments): rollout-length=5, cql-weight=5.0
# Uses the currently active conda environment (activate LEQ2 / etc. before running).
#
# Usage (from GORMPO root):
#   conda activate LEQ2
#   CUDA_VISIBLE_DEVICES=2 bash bash_scr/mult_seed/COMBO_HALFCHEETAH.sh

TASK="${TASK:-halfcheetah-medium-expert-v2}"
ROLLOUT_LENGTH="${ROLLOUT_LENGTH:-5}"
CQL_WEIGHT="${CQL_WEIGHT:-5.0}"
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

if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "ERROR: no active conda environment (CONDA_PREFIX unset)."
    echo "Activate an env first, e.g.: conda activate LEQ2"
    exit 1
fi

echo "============================================"
echo "COMBO: $TASK (full D4RL medium-expert)"
echo "  Dataset:       D4RL default for $TASK"
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
    (
        cd "$OFFLINERLKIT_DIR"
        python run_example/run_combo.py \
            --task "$TASK" \
            --seed "$seed" \
            --rollout-length "$ROLLOUT_LENGTH" \
            --cql-weight "$CQL_WEIGHT" \
            --real-ratio "$REAL_RATIO" \
            --epoch "$EPOCH"
    )
    echo "  ✓ COMBO complete for seed $seed"
    echo ""
done

echo "Done. Logs under OfflineRL-Kit2/log/${TASK}/combo/"
