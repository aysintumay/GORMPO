
set -e  # Exit on error
echo "============================================"
echo "Training GORMPO Models with ReaLNVP models"
echo "============================================"
echo ""


echo ">>> Training GORMPO on HalfCheetah-Medium-Replay-v2..."
python mopo.py --config configs/realnvp/mbpo_halfcheetah_medium_replay.yaml --devid 3
echo "✓ HalfCheetah GORMPO-RealNVP training complete"
echo ""
echo ">>> Training GORMPO on Hopper-Medium-Replay-v2..."
python mopo.py --config configs/realnvp/mbpo_hopper_medium_replay.yaml --devid 3
echo "✓ Hopper GORMPO-RealNVP training complete"
echo ""
echo ">>> Training GORMPO on Walker2d-Medium-Replay-v2..."
python mopo.py --config configs/realnvp/mbpo_walker2d_medium_replay.yaml --devid 3
echo "✓ Walker2d GORMPO-RealNVP training complete"
echo ""