#!/bin/bash
# Train RealNVP models on medium-replay-v2 datasets
# Usage: bash train_realnvp_medium_replay.sh

set -e  # Exit on error

echo "============================================"
echo "Training RealNVP Models on Medium-Replay-v2"
echo "============================================"
echo ""

# Train HalfCheetah RealNVP
echo ">>> Training RealNVP on HalfCheetah-Medium-Replay-v2..."
python realnvp_module/realnvp.py --config configs/realnvp/halfcheetah_medium_replay.yaml --device cuda:4
echo "✓ HalfCheetah RealNVP training complete"
echo ""

# Train Hopper RealNVP
echo ">>> Training RealNVP on Hopper-Medium-Replay-v2..."
python realnvp_module/realnvp.py --config configs/realnvp/hopper_medium_replay.yaml --device cuda:4
echo "✓ Hopper RealNVP training complete"
echo ""

# Train Walker2d RealNVP
echo ">>> Training RealNVP on Walker2d-Medium-Replay-v2..."
python realnvp_module/realnvp.py --config configs/realnvp/walker2d_medium_replay.yaml --device cuda:4
echo "✓ Walker2d RealNVP training complete"
echo ""

echo "============================================"
echo "All RealNVP models trained successfully!"
echo "Models saved to: /public/gormpo/models/{env}_medium_replay/realnvp"
echo "============================================"
