#!/bin/bash
# Train VAE models on medium-replay-v2 datasets
# Usage: bash train_vae_medium_replay.sh

set -e  # Exit on error

echo "============================================"
echo "Training VAE Models on Medium-Replay-v2"
echo "============================================"
echo ""

# Train HalfCheetah VAE
echo ">>> Training VAE on HalfCheetah-Medium-Replay-v2..."
python vae_module/vae.py --config configs/vae/halfcheetah_medium_replay.yaml --device cuda:4
echo "✓ HalfCheetah VAE training complete"
echo ""

# Train Hopper VAE
echo ">>> Training VAE on Hopper-Medium-Replay-v2..."
python vae_module/vae.py --config configs/vae/hopper_medium_replay.yaml --device cuda:4
echo "✓ Hopper VAE training complete"
echo ""

# Train Walker2d VAE
echo ">>> Training VAE on Walker2d-Medium-Replay-v2..."
python vae_module/vae.py --config configs/vae/walker2d_medium_replay.yaml --device cuda:4
echo "✓ Walker2d VAE training complete"
echo ""

echo "============================================"
echo "All VAE models trained successfully!"
echo "Models saved to: /public/gormpo/models/{env}_medium_replay/vae"
echo "============================================"
