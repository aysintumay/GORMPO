
set -e  # Exit on error
echo "============================================"
echo "Training GORMPO Models with KDE models"
echo "============================================"
echo ""


echo ">>> Training GORMPO on HalfCheetah-Medium-Replay-v2..."
python mopo.py --config configs/kde/mbpo_halfcheetah_medium_replay.yaml --devid 3
echo "✓ HalfCheetah GORMPO-KDE training complete"
echo ""
echo ">>> Training GORMPO on Hopper-Medium-Replay-v2..."
python mopo.py --config configs/kde/mbpo_hopper_medium_replay.yaml --devid 3
echo "✓ Hopper GORMPO-KDE training complete"
echo ""
echo ">>> Training GORMPO on Walker2d-Medium-Replay-v2..."
python mopo.py --config configs/kde/mbpo_walker2d_medium_replay.yaml --devid 3
echo "✓ Walker2d GORMPO-KDE training complete"
echo ""