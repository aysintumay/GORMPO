#!/bin/bash
# Train KDE models on medium-replay-v2 datasets
# Usage: bash bash_scr/train_kde_medium_replay.sh

set -e  # Exit on error

echo "============================================"
echo "Training MBPO Models on Medium-Replay-v2"
echo "============================================"
echo ""


echo ">>> Training GORMPO on HalfCheetah-Medium-Replay-v2..."
python mopo.py --config configs/kde/mbpo_halfcheetah_medium_replay.yaml --reward-penalty-coef 0.0 --algo-name mbpo --devid 4
echo "✓ HalfCheetah MBPO training complete"
echo ""
echo ">>> Training GORMPO on Hopper-Medium-Replay-v2..."
python mopo.py --config configs/kde/mbpo_hopper_medium_replay.yaml --reward-penalty-coef 0.0 --algo-name mbpo --devid 4
echo "✓ Hopper MBPO training complete"
echo ""
echo ">>> Training GORMPO on Walker2d-Medium-Replay-v2..."
python mopo.py --config configs/kde/mbpo_walker2d_medium_replay.yaml --reward-penalty-coef 0.0 --algo-name mbpo --devid 4
echo "✓ Walker2d MBPO training complete"
echo ""
