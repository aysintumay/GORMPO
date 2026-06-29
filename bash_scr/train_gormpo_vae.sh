
set -e  # Exit on error
echo "============================================"
echo "Training GORMPO Models with VAE models"
echo "============================================"
echo ""


echo ">>> Training GORMPO on HalfCheetah-Medium-Replay-v2..."
python mopo.py --config configs/vae/gormpo_halfcheetah_medium_replay.yaml --devid 3
echo "✓ HalfCheetah GORMPO-VAE training complete"
echo ""
echo ">>> Training GORMPO on Hopper-Medium-Replay-v2..."
python mopo.py --config configs/vae/gormpo_hopper_medium_replay.yaml --devid 3
echo "✓ Hopper GORMPO-vae training complete"
echo ""
echo ">>> Training GORMPO on Walker2d-Medium-Replay-v2..."
python mopo.py --config configs/vae/gormpo_walker2d_medium_replay.yaml --devid 3
echo "✓ Walker2d GORMPO-VAE training complete"
echo ""