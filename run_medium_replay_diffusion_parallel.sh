#!/bin/bash

# Script to run medium_replay diffusion GORMPO experiments in parallel
# Creates a logs directory and runs each environment configuration as a background process

# Create logs directory if it doesn't exist
mkdir -p logs/parallel_runs

# Get timestamp for unique log naming
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "Starting parallel GORMPO training runs at ${TIMESTAMP}"
echo "Logs will be saved to logs/parallel_runs/"
echo "----------------------------------------"

# Run HalfCheetah medium-replay
echo "Starting HalfCheetah medium-replay..."
python mopo.py --config configs/diffusion/halfcheetah_medium_replay.yaml \
    > logs/parallel_runs/halfcheetah_medium_replay_${TIMESTAMP}.log 2>&1 &
PID_HALFCHEETAH=$!
echo "  - HalfCheetah started (PID: ${PID_HALFCHEETAH})"

# Run Hopper medium-replay
echo "Starting Hopper medium-replay..."
python mopo.py --config configs/diffusion/hopper_medium_replay.yaml \
    > logs/parallel_runs/hopper_medium_replay_${TIMESTAMP}.log 2>&1 &
PID_HOPPER=$!
echo "  - Hopper started (PID: ${PID_HOPPER})"

# Run Walker2d medium-replay
echo "Starting Walker2d medium-replay..."
python mopo.py --config configs/diffusion/walker2d_medium_replay.yaml \
    > logs/parallel_runs/walker2d_medium_replay_${TIMESTAMP}.log 2>&1 &
PID_WALKER2D=$!
echo "  - Walker2d started (PID: ${PID_WALKER2D})"

echo "----------------------------------------"
echo "All experiments started. Monitoring progress..."
echo "You can monitor logs in real-time with:"
echo "  tail -f logs/parallel_runs/halfcheetah_medium_replay_${TIMESTAMP}.log"
echo "  tail -f logs/parallel_runs/hopper_medium_replay_${TIMESTAMP}.log"
echo "  tail -f logs/parallel_runs/walker2d_medium_replay_${TIMESTAMP}.log"
echo ""
echo "Waiting for all experiments to complete..."

# Wait for all background processes
wait $PID_HALFCHEETAH
HALFCHEETAH_EXIT=$?
echo "HalfCheetah completed (exit code: ${HALFCHEETAH_EXIT})"

wait $PID_HOPPER
HOPPER_EXIT=$?
echo "Hopper completed (exit code: ${HOPPER_EXIT})"

wait $PID_WALKER2D
WALKER2D_EXIT=$?
echo "Walker2d completed (exit code: ${WALKER2D_EXIT})"

echo "----------------------------------------"
echo "All experiments completed!"
echo "Exit codes: HalfCheetah=${HALFCHEETAH_EXIT}, Hopper=${HOPPER_EXIT}, Walker2d=${WALKER2D_EXIT}"

# Check if any experiments failed
if [ $HALFCHEETAH_EXIT -ne 0 ] || [ $HOPPER_EXIT -ne 0 ] || [ $WALKER2D_EXIT -ne 0 ]; then
    echo "WARNING: Some experiments failed. Check the log files for details."
    exit 1
else
    echo "SUCCESS: All experiments completed successfully!"
    exit 0
fi
