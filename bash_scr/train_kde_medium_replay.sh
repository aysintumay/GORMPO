#!/bin/bash
# Train KDE models on medium-replay-v2 datasets
# Usage: bash bash_scr/train_kde_medium_replay.sh

set -e  # Exit on error

echo "============================================"
echo "Training KDE Models on Medium-Replay-v2"
echo "============================================"
echo ""

# Train HalfCheetah KDE
echo ">>> Training KDE on HalfCheetah-Medium-Replay-v2..."
python kde_module/kde.py --config configs/kde/halfcheetah_medium_replay.yaml --devid 7
echo "✓ HalfCheetah KDE training complete"
echo ""

# Train Hopper KDE
echo ">>> Training KDE on Hopper-Medium-Replay-v2..."
python kde_module/kde.py --config configs/kde/hopper_medium_replay.yaml --devid 7
echo "✓ Hopper KDE training complete"
echo ""

# Train Walker2d KDE
echo ">>> Training KDE on Walker2d-Medium-Replay-v2..."
python kde_module/kde.py --config configs/kde/walker2d_medium_replay.yaml --devid 7
echo "✓ Walker2d KDE training complete"
echo ""

echo "============================================"
echo "All KDE models trained successfully!"
echo "Models saved to: /public/gormpo/models/{env}_medium_replay/kde"
echo "============================================"
