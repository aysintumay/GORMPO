#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/venv/bin/python}"

TASK_HALF="${TASK_HALF:-halfcheetah-medium-expert-v2}"
DATA_HALF="${DATA_HALF:-/public/d4rl/sparse_datasets/halfcheetah_medium_expert_sparse_72.5.pkl}"
POL_HALF="${POL_HALF:-${ROOT_DIR}/notebooks/policy_paths_halfcheetah.json}"

TASK_HOP="${TASK_HOP:-hopper-medium-expert-v2}"
DATA_HOP="${DATA_HOP:-/public/d4rl/sparse_datasets/hopper_medium_expert_sparse_78.pkl}"
POL_HOP="${POL_HOP:-${ROOT_DIR}/notebooks/policy_paths_hopper.json}"

TASK_WALK="${TASK_WALK:-walker2d-medium-expert-v2}"
DATA_WALK="${DATA_WALK:-/public/d4rl/sparse_datasets/walker2d_medium_expert_sparse_73.pkl}"
POL_WALK="${POL_WALK:-${ROOT_DIR}/notebooks/policy_paths_walker2d.json}"

PANEL_COLS="${PANEL_COLS:-6}"
N_ROLLOUT_EPISODES="${N_ROLLOUT_EPISODES:-100}"
MAX_OFFLINE_SAMPLES="${MAX_OFFLINE_SAMPLES:-10000}"
MAX_ROLLOUT_SAMPLES="${MAX_ROLLOUT_SAMPLES:-5000000}"

SUPPORT_SOURCE="${SUPPORT_SOURCE:-rollout}" # rollout is what you want for policy support overlap

SAVE_PDF_FLAG="--save-pdf"
if [[ "${SAVE_PDF:-1}" -eq 0 ]]; then
  SAVE_PDF_FLAG=""
fi

DETERMINISTIC_FLAG="--deterministic"
if [[ "${DETERMINISTIC:-1}" -eq 0 ]]; then
  DETERMINISTIC_FLAG=""
fi

OUT_DIR="${OUT_DIR:-results/tsne_policy_vs_dataset_sparse}"

DEVICE="cpu"

echo "Running t-SNE overlap plots (policy support vs dataset support)"
echo "Python: ${PYTHON_BIN}"
echo "Output: ${OUT_DIR}"
echo "Device: ${DEVICE}"

run_one () {
  local task="$1"
  local dataset_path="$2"
  local policy_json="$3"
  echo "---- $task ----"
  "${PYTHON_BIN}" "${ROOT_DIR}/notebooks/tsne_policy_vs_dataset.py" \
    --task "${task}" \
    --dataset-path "${dataset_path}" \
    --policy-json "${policy_json}" \
    --support-source "${SUPPORT_SOURCE}" \
    "${DETERMINISTIC_FLAG}" \
    --device "${DEVICE}" \
    --n-rollout-episodes "${N_ROLLOUT_EPISODES}" \
    --max-offline-samples "${MAX_OFFLINE_SAMPLES}" \
    --max-rollout-samples "${MAX_ROLLOUT_SAMPLES}" \
    --panel-cols "${PANEL_COLS}" \
    --output-dir "${OUT_DIR}" \
    ${SAVE_PDF_FLAG}
}

run_one "${TASK_HALF}"  "${DATA_HALF}"  "${POL_HALF}"
run_one "${TASK_HOP}"   "${DATA_HOP}"   "${POL_HOP}"
run_one "${TASK_WALK}"  "${DATA_WALK}"  "${POL_WALK}"

echo "All runs completed."

