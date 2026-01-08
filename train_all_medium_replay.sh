#!/bin/bash
# Train all density models (KDE, VAE, RealNVP) on all medium-replay-v2 datasets
# Usage: bash train_all_medium_replay.sh [model_type]
#   model_type: kde, vae, realnvp, or all (default: all)

MODEL_TYPE=${1:-all}

echo "============================================"
echo "Training Density Models on Medium-Replay-v2"
echo "Model Type: $MODEL_TYPE"
echo "============================================"
echo ""

case $MODEL_TYPE in
    kde)
        echo "Training KDE models only..."
        bash train_kde_medium_replay.sh
        ;;
    vae)
        echo "Training VAE models only..."
        bash train_vae_medium_replay.sh
        ;;
    realnvp)
        echo "Training RealNVP models only..."
        bash train_realnvp_medium_replay.sh
        ;;
    all)
        echo "Training all density models (KDE, VAE, RealNVP)..."
        echo ""

        echo "=== Step 1/3: Training KDE Models ==="
        bash train_kde_medium_replay.sh
        echo ""

        echo "=== Step 2/3: Training VAE Models ==="
        bash train_vae_medium_replay.sh
        echo ""

        echo "=== Step 3/3: Training RealNVP Models ==="
        bash train_realnvp_medium_replay.sh
        echo ""

        echo "============================================"
        echo "All density models trained successfully!"
        echo "============================================"
        ;;
    *)
        echo "Error: Invalid model type '$MODEL_TYPE'"
        echo "Usage: bash train_all_medium_replay.sh [kde|vae|realnvp|all]"
        exit 1
        ;;
esac
