#!/bin/bash
#
# Endurance Study for Adaptive Power Management
#
# This script runs a long-duration (1 hour) experiment to evaluate the
# "break-even" efficiency of the adaptive power management system under
# realistic, sparse RF surveillance workloads.
#
# It uses:
#   - Extended Duration: 3600s (1 hour)
#   - High Sparsity: 10x longer idle times (sparsity-factor=10.0)
#   - Long Hysteresis: 60s (to prevent thrashing)
#
# Usage:
#   ./run_endurance_study.sh [MODEL] [ENABLE_GPU_SCALING]
#
# Example:
#   ./run_endurance_study.sh lstm_ae true
#

set -e  # Exit on error

# Configuration
MODEL=${1:-lstm_ae}
ENABLE_GPU_SCALING=${2:-false}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="endurance_study_${MODEL}_${TIMESTAMP}"
DURATION=3600  # 1 hour
HYSTERESIS=60.0
SPARSITY=10.0

# Create output directory
mkdir -p "$OUTPUT_BASE"

echo "======================================================="
echo "ENDURANCE STUDY - ADAPTIVE POWER MANAGEMENT"
echo "======================================================="
echo "Model: $MODEL"
echo "Timestamp: $TIMESTAMP"
echo "Duration: ${DURATION}s (1 hour)"
echo "Hysteresis: ${HYSTERESIS}s"
echo "Sparsity Factor: ${SPARSITY}x"
echo "GPU Frequency Scaling: $ENABLE_GPU_SCALING"
echo ""
echo "This study simulates a realistic 'Guard Duty' cycle:"
echo "  - Long periods of idle/low activity"
echo "  - Occasional bursts of signal activity"
echo "  - System should downshift during long idles"
echo "  - System should NOT thrash during short pauses"
echo ""
echo "Expected runtime: ~1 hour per workload"
echo "======================================================="
echo ""

# Set model paths
MODEL_PATH="src/output/weights/${MODEL}_best.pth"
ENGINE_PATH="src/output/engines/${MODEL}.engine"

# Build TensorRT flag
if [ -f "$ENGINE_PATH" ]; then
    TRT_FLAG="--use-tensorrt --engine-path $ENGINE_PATH"
else
    TRT_FLAG=""
fi

# Build frequency scaling flag
if [ "$ENABLE_GPU_SCALING" = "true" ]; then
    FREQ_FLAG="--enable-frequency-scaling"
else
    FREQ_FLAG=""
fi

# Run experiment for Bursty and Periodic workloads (most relevant for endurance)
WORKLOADS=("bursty" "periodic")

for workload in "${WORKLOADS[@]}"; do
    echo ""
    echo "========================================================="
    echo "TESTING WORKLOAD: $workload"
    echo "========================================================="
    echo ""

    # Strategy: Run a quick calibration first to get thresholds.
    # Since we are not on the device, this is simulation mode anyway.
    echo "Step 1: Calibrating thresholds (quick run)..."
    CALIB_OUTPUT="$OUTPUT_BASE/calibration_${workload}.json"
    
    # We use a short duration for calibration
    python src/adaptive_benchmark.py \
        --model "$MODEL" \
        --model-path "$MODEL_PATH" \
        $TRT_FLAG \
        --workload "$workload" \
        --duration 30 \
        --auto-calibrate \
        --target-sla 10.0 \
        --output-dir "$OUTPUT_BASE/calibration" \
        > "$OUTPUT_BASE/calibration_log_${workload}.txt" 2>&1 || true

    echo "Step 2: Running Endurance Test..."
    
    python src/adaptive_benchmark.py \
        --model "$MODEL" \
        --model-path "$MODEL_PATH" \
        $TRT_FLAG \
        --workload "$workload" \
        --duration "$DURATION" \
        --use-model-defaults \
        --hysteresis-time "$HYSTERESIS" \
        --sparsity-factor "$SPARSITY" \
        $FREQ_FLAG \
        --run-baselines \
        --batch-size 1 \
        --num-channels 1 \
        --output-dir "$OUTPUT_BASE/results"

    echo ""
    echo "✅ Completed: $workload"
    
    # Cooldown between workloads
    echo "⏳ Long Thermal Cooldown (2 minutes)..."
    sleep 120
done

echo ""
echo "========================================================="
echo "STUDY COMPLETE!"
echo "========================================================="
echo "Results saved to: $OUTPUT_BASE/"
