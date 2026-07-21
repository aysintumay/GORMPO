#!/bin/bash
set -e
# COMBO + KDE density guardian on MCS (Abiomed).
#
# Mirrors DENSITY_AND_LEQ_MCS.sh, but Step 3 optimises the policy with COMBO
# (OfflineRL-Kit2/run_example/run_combo.py) instead of LEQ, applying the same
# density-guardian OOD penalty to model-rollout rewards.
#
# Pipeline (per seed):
#   1. Train KDE guardian                          (GORMPO_abiomed mbpo_kde)
#   2. Train dynamics ensemble on MCS .npz         (OfflineRL-Kit2)
#   3. Train COMBO with guardian OOD penalty       (OfflineRL-Kit2)
#
# ------------------------------------------------------------------------------
# ⚠ PENDING PYTHON SUPPORT
#   run_combo.py does NOT yet accept the flags this script passes, so it will fail
#   on unrecognized arguments until OfflineRL-Kit2 is extended to add:
#       --guardian-model-name   <path>   # load the trained KDE guardian
#       --guardian-penalty-coef <float>  # OOD penalty coefficient λ
#       --dataset-path          <npz>    # offline MCS/Abiomed dataset
#   plus abiomed-v0 env + .npz dataset loading (run_combo.py currently assumes a
#   D4RL gym task), and it must penalise model-rollout rewards for OOD
#   (next_obs, action) pairs inside COMBOPolicy, mirroring
#   LEQ2/algos/leq/learner.py:413-419:
#       ood     = guardian["model"].score_samples(concat[next_obs, action]) < thr
#       rewards = rewards - guardian_penalty_coef * ood
#   (--load-dynamics-path, --task, --seed, --cql-weight, --rollout-length,
#    --real-ratio, --epoch are already supported by run_combo.py.)
# ------------------------------------------------------------------------------
#
# Usage (from GORMPO root):
#   bash bash_scr/mult_seed/DENSITY_AND_COMBO_MCS.sh
#   CUDA_VISIBLE_DEVICES=2 bash bash_scr/mult_seed/DENSITY_AND_COMBO_MCS.sh
#
# Env overrides:
#   DATASET_PATH, SEEDS, GUARDIAN_PENALTY_COEF, ROLLOUT_LENGTH, CQL_WEIGHT,
#   REAL_RATIO, EPOCH, DEVID_KDE, DEVID_DYN, LEQ2_ENV, OFFLINERLKIT_DIR,
#   GORMPO_ABIOMED_DIR, GUARDIAN_BASE

TASK="${TASK:-abiomed-v0}"
KDE_CONFIG="${KDE_CONFIG:-config/kde/real.yaml}"
GUARDIAN_BASE="${GUARDIAN_BASE:-/public/gormpo/models/abiomed}"
GUARDIAN_PENALTY_COEF="${GUARDIAN_PENALTY_COEF:-0.5}"

# COMBO hypers (match COMBO_MCS.sh)
ROLLOUT_LENGTH="${ROLLOUT_LENGTH:-5}"
CQL_WEIGHT="${CQL_WEIGHT:-1.0}"
REAL_RATIO="${REAL_RATIO:-0.5}"
EPOCH="${EPOCH:-1000}"

DEVID_KDE="${DEVID_KDE:-0}"
DEVID_DYN="${DEVID_DYN:-0}"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$DEVID_DYN"
fi

if [ -z "${SEEDS+x}" ] || [ -z "$SEEDS" ]; then
    SEEDS=(42 123 456)
else
    # shellcheck disable=SC2206
    SEEDS=($SEEDS)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GORMPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GORMPO_ABIOMED_DIR="${GORMPO_ABIOMED_DIR:-$GORMPO_ROOT/../GORMPO_abiomed}"
GORMPO_CORMPO="${GORMPO_ABIOMED_DIR}/cormpo"
OFFLINERLKIT_DIR="${OFFLINERLKIT_DIR:-$GORMPO_ROOT/../OfflineRL-Kit2}"
LEQ2_ENV="${LEQ2_ENV:-LEQ2}"

DATASET_PATH="${DATASET_PATH:-$GORMPO_ABIOMED_DIR/synthetic_data/SAC_5000eps_stochastic.npz}"
if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: MCS dataset not found: $DATASET_PATH"
    echo "Set DATASET_PATH to an abiomed offline .npz"
    exit 1
fi

echo "============================================"
echo "COMBO + KDE guardian: $TASK (MCS / Abiomed)"
echo "  GORMPO:         $GORMPO_ROOT"
echo "  GORMPO_abiomed: $GORMPO_ABIOMED_DIR"
echo "  OfflineRL-Kit:  $OFFLINERLKIT_DIR"
echo "  Dataset:        $DATASET_PATH"
echo "  Seeds:          ${SEEDS[*]}"
echo "  Guardian λ:     $GUARDIAN_PENALTY_COEF"
echo "  rollout-length=$ROLLOUT_LENGTH  cql-weight=$CQL_WEIGHT  real-ratio=$REAL_RATIO"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "============================================"
echo ""

for seed in "${SEEDS[@]}"; do
    # Matches existing layout: .../trained_kde_<seed>/trained_kde_1.{faiss,_metadata.pkl}
    GUARDIAN_DIR="${GUARDIAN_BASE}/trained_kde_${seed}"
    GUARDIAN_PATH="${GUARDIAN_DIR}/trained_kde_1"

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
        mkdir -p "$GUARDIAN_DIR"
        (cd "$GORMPO_CORMPO" && python mbpo_kde/kde.py \
            --config "$KDE_CONFIG" \
            --seed "$seed" \
            --save_path "$GUARDIAN_DIR" \
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
                    --task '$TASK' --seed '$seed' \
                    --dataset-path '$DATASET_PATH'"
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
                --dataset-path '$DATASET_PATH' \
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
echo "Guardians in ${GUARDIAN_BASE}/trained_kde_<seed>/trained_kde_1.*"
echo "============================================"
