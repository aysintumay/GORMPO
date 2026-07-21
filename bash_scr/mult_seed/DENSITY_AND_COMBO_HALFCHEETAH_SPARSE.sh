#!/bin/bash
set -e
# COMBO + KDE density guardian on HalfCheetah-Medium-Expert (sparse 72.5%).
#
# Mirrors DENSITY_AND_LEQ_HALFCHEETAH.sh, but Step 3 optimises the policy with
# COMBO (OfflineRL-Kit2/run_example/run_combo.py) instead of LEQ, applying the
# same density-guardian OOD penalty to model-rollout rewards.
#
# Pipeline (per seed):
#   1. Train KDE guardian on sparse offline data   (GORMPO kde_module)
#   2. Train dynamics ensemble                     (OfflineRL-Kit2)
#   3. Train COMBO with guardian OOD penalty       (OfflineRL-Kit2)
#
# ------------------------------------------------------------------------------
# ⚠ PENDING PYTHON SUPPORT
#   run_combo.py does NOT yet accept the guardian flags this script passes, so it
#   will fail on unrecognized arguments until OfflineRL-Kit2 is extended to add:
#       --guardian-model-name   <path>   # load the trained KDE guardian
#       --guardian-penalty-coef <float>  # OOD penalty coefficient λ
#   and to penalise model-rollout rewards for OOD (next_obs, action) pairs inside
#   COMBOPolicy, mirroring LEQ2/algos/leq/learner.py:413-419:
#       ood     = guardian["model"].score_samples(concat[next_obs, action]) < thr
#       rewards = rewards - guardian_penalty_coef * ood
#   (--load-dynamics-path, --task, --seed, --cql-weight, --rollout-length,
#    --real-ratio, --epoch are already supported by run_combo.py.)
# ------------------------------------------------------------------------------
#
# Usage (from GORMPO root):
#   bash bash_scr/mult_seed/DENSITY_AND_COMBO_HALFCHEETAH_SPARSE.sh
#
# Env overrides:
#   SEEDS, GUARDIAN_PENALTY_COEF, ROLLOUT_LENGTH, CQL_WEIGHT, REAL_RATIO, EPOCH,
#   DEVID_KDE, DEVID_DYN, LEQ2_ENV, OFFLINERLKIT_DIR

TASK="${TASK:-halfcheetah-medium-expert-v2}"
KDE_CONFIG="${KDE_CONFIG:-configs/kde/halfcheetah_medium_expert_sparse_3.yaml}"
GUARDIAN_ROOT="${GUARDIAN_ROOT:-/public/gormpo/models/halfcheetah_medium_expert_sparse_3}"
GUARDIAN_PENALTY_COEF="${GUARDIAN_PENALTY_COEF:-0.5}"

# COMBO hypers (halfcheetah-medium-expert-v2: rollout-length=5, cql-weight=5.0)
ROLLOUT_LENGTH="${ROLLOUT_LENGTH:-5}"
CQL_WEIGHT="${CQL_WEIGHT:-5.0}"
REAL_RATIO="${REAL_RATIO:-0.5}"
EPOCH="${EPOCH:-1000}"

DEVID_KDE="${DEVID_KDE:-0}"
DEVID_DYN="${DEVID_DYN:-0}"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$DEVID_DYN"
fi

# Default seeds from the LEQ HalfCheetah script; override with SEEDS="42 123 456"
if [ -z "${SEEDS+x}" ] || [ -z "$SEEDS" ]; then
    SEEDS=(67 68 69)
else
    # shellcheck disable=SC2206
    SEEDS=($SEEDS)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GORMPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OFFLINERLKIT_DIR="${OFFLINERLKIT_DIR:-$GORMPO_ROOT/../OfflineRL-Kit2}"
LEQ2_ENV="${LEQ2_ENV:-LEQ2}"

echo "============================================"
echo "COMBO + KDE guardian: $TASK (sparse HalfCheetah)"
echo "  GORMPO:        $GORMPO_ROOT"
echo "  OfflineRL-Kit: $OFFLINERLKIT_DIR"
echo "  Seeds:         ${SEEDS[*]}"
echo "  Guardian λ:    $GUARDIAN_PENALTY_COEF"
echo "  rollout-length=$ROLLOUT_LENGTH  cql-weight=$CQL_WEIGHT  real-ratio=$REAL_RATIO"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "============================================"
echo ""

for seed in "${SEEDS[@]}"; do
    GUARDIAN_PATH="${GUARDIAN_ROOT}/kde_${seed}"

    echo "=========================================="
    echo ">>> seed = $seed"
    echo "=========================================="

    # Resume: skip the whole seed if its policy is already trained.
    policy_done=""
    for md in "$OFFLINERLKIT_DIR/log/$TASK/combo/seed_${seed}&timestamp_"*/model; do
        [ -d "$md" ] || continue
        case "$md" in *_sparse/model) continue;; esac
        [ -f "$md/policy.pth" ] && policy_done="$md"
    done
    if [ -n "$policy_done" ]; then
        echo "  ✓ policy already trained ($policy_done) — skipping seed."
        echo ""
        continue
    fi

    # --- Step 1: KDE density guardian ---
    echo "Step 1/3: KDE guardian → $GUARDIAN_PATH"
    if [ -f "${GUARDIAN_PATH}_metadata.pkl" ] && [ -f "${GUARDIAN_PATH}.faiss" ]; then
        echo "  Guardian already exists, skipping."
    else
        (cd "$GORMPO_ROOT" && python kde_module/kde.py \
            --config "$KDE_CONFIG" \
            --seed "$seed" \
            --save_path "$GUARDIAN_PATH" \
            --devid "$DEVID_KDE")
        echo "  ✓ KDE training complete"
    fi
    echo ""

    # --- Step 2: Dynamics ensemble ---
    DYN_DIR="$OFFLINERLKIT_DIR/models/dynamics-ensemble/${seed}/${TASK}"
    echo "Step 2/3: Dynamics ensemble → $DYN_DIR"
    if [ -d "$DYN_DIR" ] && [ -n "$(ls -A "$DYN_DIR" 2>/dev/null)" ]; then
        echo "  Dynamics already exist, skipping."
    else
        conda run --no-capture-output -n "$LEQ2_ENV" bash -c \
            "cd '$OFFLINERLKIT_DIR' && \
                CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' \
                python run_example/run_dynamics.py \
                    --task '$TASK' --seed '$seed'"
        echo "  ✓ Dynamics training complete"
    fi
    echo ""

    # --- Step 3: COMBO + guardian ---
    echo "Step 3/3: COMBO + guardian (λ=$GUARDIAN_PENALTY_COEF)"
    conda run --no-capture-output -n "$LEQ2_ENV" bash -c \
        "cd '$OFFLINERLKIT_DIR' && \
            CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' \
            python run_example/run_combo.py \
                --task '$TASK' \
                --seed '$seed' \
                --rollout-length '$ROLLOUT_LENGTH' \
                --cql-weight '$CQL_WEIGHT' \
                --real-ratio '$REAL_RATIO' \
                --epoch '$EPOCH' \
                --load-dynamics-path '$DYN_DIR' \
                --guardian-model-name '$GUARDIAN_PATH' \
                --guardian-penalty-coef '$GUARDIAN_PENALTY_COEF'"
    echo "  ✓ COMBO training complete for seed $seed"
    echo ""
done

echo "============================================"
echo "Done. Logs under OfflineRL-Kit2/log/${TASK}/combo/"
echo "Guardians in ${GUARDIAN_ROOT}/kde_<seed>.*"
echo "============================================"
