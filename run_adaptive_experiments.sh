#!/bin/bash
#
# Adaptive Power Management Experiment Launcher
# Runs comprehensive experiments for the paper extension
#
# Usage:
#   ./run_adaptive_experiments.sh [MODEL] [USE_TENSORRT] [USE_MODEL_DEFAULTS]
#
# Optional environment variables:
#   BATCH_SIZE=N              - Batch size for batched inference (default: 1)
#   NUM_CHANNELS=N            - Number of concurrent channels (default: 1)
#   DURATION=X                - Experiment duration in seconds (default: 60)
#   WORKLOAD=pattern          - Run single workload only: bursty, continuous, variable, periodic (default: all)
#   AUTO_CALIBRATE=true       - Automatically calibrate thresholds (default: false)
#   TARGET_SLA=X              - Target SLA in ms for auto-calibration (default: 10.0)
#   AUTO_ADJUST_SLA=true      - Auto-adjust SLA if unreachable (default: false)
#   ENABLE_FREQUENCY_SCALING=true - Enable fine-grained GPU frequency scaling (default: false)
#   SKIP_SWITCHING=true       - Skip switching characterization phase (default: false)
#   THERMAL_COOLDOWN=X        - Thermal cooldown seconds between workloads (default: 60)
#   SWITCHING_TRIALS=X        - Number of switching trials (default: 20)
#   SWITCHING_STABILIZATION=X - Stabilization time in seconds (default: 2.0)
#
# Examples:
#   # Default: single-channel, single-sample
#   ./run_adaptive_experiments.sh lstm_ae false true
#
#   # Batched inference (8 samples per batch)
#   BATCH_SIZE=8 ./run_adaptive_experiments.sh lstm_ae false true
#
#   # Multi-channel (10 concurrent channels)
#   NUM_CHANNELS=10 ./run_adaptive_experiments.sh lstm_ae false true
#
#   # Hybrid: 10 channels, each processing batches of 8
#   BATCH_SIZE=8 NUM_CHANNELS=10 ./run_adaptive_experiments.sh lstm_ae false true
#
#   # Auto-calibrate thresholds for optimal performance
#   AUTO_CALIBRATE=true ./run_adaptive_experiments.sh lstm_ae false false
#
#   # Auto-calibrate with custom SLA target
#   AUTO_CALIBRATE=true TARGET_SLA=15.0 ./run_adaptive_experiments.sh lstm_ae
#
#   # Auto-adjust SLA if target is unreachable (discovers minimum achievable)
#   AUTO_CALIBRATE=true AUTO_ADJUST_SLA=true ./run_adaptive_experiments.sh lstm_ae
#
#   # Multi-channel with auto-adjustment (realistic workload scaling)
#   AUTO_CALIBRATE=true AUTO_ADJUST_SLA=true NUM_CHANNELS=10 ./run_adaptive_experiments.sh lstm_ae
#
#   # Skip switching characterization for faster experiments
#   SKIP_SWITCHING=true ./run_adaptive_experiments.sh lstm_ae false true
#
#   # Quick experiment: skip switching, short cooldown
#   SKIP_SWITCHING=true THERMAL_COOLDOWN=10 ./run_adaptive_experiments.sh lstm_ae
#
#   # Custom duration (120 seconds)
#   DURATION=120 ./run_adaptive_experiments.sh lstm_ae false true
#
#   # Single workload only (bursty)
#   WORKLOAD=bursty ./run_adaptive_experiments.sh lstm_ae false true
#
#   # Quick single workload test: bursty, 30s duration, skip switching
#   SKIP_SWITCHING=true DURATION=30 WORKLOAD=bursty ./run_adaptive_experiments.sh lstm_ae
#
#   # Multi-channel scaling test: 10 channels, bursty workload, 120s
#   NUM_CHANNELS=10 WORKLOAD=bursty DURATION=120 ./run_adaptive_experiments.sh lstm_ae
#

set -e  # Exit on error

# Configuration
MODEL=${1:-ae}
USE_TENSORRT=${2:-false}
USE_MODEL_DEFAULTS=${3:-true}  # Enable model-specific thresholds by default
BATCH_SIZE=${BATCH_SIZE:-1}    # Default: single-sample
NUM_CHANNELS=${NUM_CHANNELS:-1}  # Default: single-channel
DURATION=${DURATION:-60}  # Default: 60 seconds experiment duration
WORKLOAD=${WORKLOAD:-all}  # Default: run all workloads (or specify: bursty, continuous, variable, periodic)
AUTO_CALIBRATE=${AUTO_CALIBRATE:-false}  # Default: use fixed/model-defaults
TARGET_SLA=${TARGET_SLA:-10.0}  # Default: 10ms SLA
AUTO_ADJUST_SLA=${AUTO_ADJUST_SLA:-false}  # Default: don't auto-adjust
ENABLE_FREQUENCY_SCALING=${ENABLE_FREQUENCY_SCALING:-false} # Default: disable frequency scaling
SKIP_SWITCHING=${SKIP_SWITCHING:-false}  # Default: run switching characterization
THERMAL_COOLDOWN=${THERMAL_COOLDOWN:-60}  # Default: 60 seconds between workloads
SWITCHING_TRIALS=${SWITCHING_TRIALS:-20}  # Default: 20 trials for switching overhead
SWITCHING_STABILIZATION=${SWITCHING_STABILIZATION:-2.0}  # Default: 2.0s stabilization
OUTPUT_BASE="adaptive_experiments_$(date +%Y%m%d_%H%M%S)"

echo "========================================"
echo "ADAPTIVE POWER MANAGEMENT EXPERIMENTS"
echo "========================================"
echo "Model: $MODEL"
echo "Use TensorRT: $USE_TENSORRT"
echo "Use Model Defaults: $USE_MODEL_DEFAULTS"
echo "Auto Calibrate: $AUTO_CALIBRATE"
if [ "$AUTO_CALIBRATE" = "true" ]; then
    echo "Target SLA: ${TARGET_SLA}ms"
    echo "Auto Adjust SLA: $AUTO_ADJUST_SLA"
fi
echo "Frequency Scaling: $ENABLE_FREQUENCY_SCALING"
echo "Batch Size: $BATCH_SIZE"
echo "Num Channels: $NUM_CHANNELS"
echo "Duration: ${DURATION}s"
echo "Workload: $WORKLOAD"
echo "Skip Switching: $SKIP_SWITCHING"
if [ "$SKIP_SWITCHING" = "false" ]; then
    echo "Switching Trials: $SWITCHING_TRIALS"
    echo "Switching Stabilization: ${SWITCHING_STABILIZATION}s"
fi
echo "Thermal Cooldown: ${THERMAL_COOLDOWN}s"
echo "Output directory: $OUTPUT_BASE"
echo ""

# Create output directories
mkdir -p "$OUTPUT_BASE"
mkdir -p "$OUTPUT_BASE/switching_overhead"
mkdir -p "$OUTPUT_BASE/results"
mkdir -p "$OUTPUT_BASE/figures"

# Set model paths
MODEL_PATH="src/output/weights/${MODEL}_best.pth"
ENGINE_PATH="src/output/engines/${MODEL}.engine"

# Check if model files exist
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: Model weights not found at $MODEL_PATH"
    echo "   Please train the model first using: cd src && python train.py --model $MODEL"
    exit 1
fi

if [ "$USE_TENSORRT" = "true" ] && [ ! -f "$ENGINE_PATH" ]; then
    echo "⚠️  Warning: TensorRT engine not found at $ENGINE_PATH"
    echo "   Will use PyTorch model instead"
    USE_TENSORRT="false"
fi

# Phase 1: Mode Switching Characterization (conditional)
if [ "$SKIP_SWITCHING" = "false" ]; then
    echo ""
    echo "========================================="
    echo "PHASE 1: Mode Switching Characterization"
    echo "========================================="
    echo ""

    python src/characterize_switching_overhead.py \
        --trials "$SWITCHING_TRIALS" \
        --stabilization-time "$SWITCHING_STABILIZATION" \
        --performance-test \
        --samples 100 \
        --output-dir "$OUTPUT_BASE/switching_overhead"

    echo ""
    echo "✅ Phase 1 complete: Switching overhead characterized"
    echo "   Results: $OUTPUT_BASE/switching_overhead/"
    echo ""
else
    echo ""
    echo "========================================="
    echo "PHASE 1: Mode Switching (SKIPPED)"
    echo "========================================="
    echo "⏭️  Skipping switching characterization (SKIP_SWITCHING=true)"
    echo ""
fi

# Phase 2: Adaptive Benchmarking
echo ""
echo "==============================="
echo "PHASE 2: Adaptive Benchmarking"
echo "==============================="
echo ""

# Define workload patterns
if [ "$WORKLOAD" = "all" ]; then
    WORKLOADS=("bursty" "continuous" "variable" "periodic")
else
    # Single workload mode
    WORKLOADS=("$WORKLOAD")
    echo "🎯 Single workload mode: $WORKLOAD"
    echo ""
fi

# Build TensorRT flag
if [ "$USE_TENSORRT" = "true" ]; then
    TRT_FLAG="--use-tensorrt --engine-path $ENGINE_PATH"
else
    TRT_FLAG=""
fi

# Build auto-calibrate flag
if [ "$AUTO_CALIBRATE" = "true" ]; then
    CALIBRATE_FLAG="--auto-calibrate --target-sla $TARGET_SLA"
    if [ "$AUTO_ADJUST_SLA" = "true" ]; then
        CALIBRATE_FLAG="$CALIBRATE_FLAG --auto-adjust-sla"
    fi
else
    CALIBRATE_FLAG=""
fi

# Build frequency scaling flag
if [ "$ENABLE_FREQUENCY_SCALING" = "true" ]; then
    FREQ_FLAG="--enable-frequency-scaling"
else
    FREQ_FLAG=""
fi

# Run benchmark for each workload
for workload in "${WORKLOADS[@]}"; do
    echo ""
    echo "Testing workload: $workload"
    echo "-----------------------------------"

    # Build command based on whether we're using model defaults or auto-calibration
    if [ "$AUTO_CALIBRATE" = "true" ]; then
        # Use auto-calibrated thresholds (overrides model defaults)
        python src/adaptive_benchmark.py \
            --model "$MODEL" \
            --model-path "$MODEL_PATH" \
            $TRT_FLAG \
            --workload "$workload" \
            --duration "$DURATION" \
            $CALIBRATE_FLAG \
            $FREQ_FLAG \
            --run-baselines \
            --max-samples 200 \
            --batch-size "$BATCH_SIZE" \
            --num-channels "$NUM_CHANNELS" \
            --output-dir "$OUTPUT_BASE/results"
    elif [ "$USE_MODEL_DEFAULTS" = "true" ]; then
        # Use model-specific thresholds and hysteresis
        python src/adaptive_benchmark.py \
            --model "$MODEL" \
            --model-path "$MODEL_PATH" \
            $TRT_FLAG \
            --workload "$workload" \
            --duration "$DURATION" \
            --use-model-defaults \
            $FREQ_FLAG \
            --run-baselines \
            --max-samples 200 \
            --batch-size "$BATCH_SIZE" \
            --num-channels "$NUM_CHANNELS" \
            --output-dir "$OUTPUT_BASE/results"
    else
        # Use explicit threshold and hysteresis values
        python src/adaptive_benchmark.py \
            --model "$MODEL" \
            --model-path "$MODEL_PATH" \
            $TRT_FLAG \
            --workload "$workload" \
            --duration "$DURATION" \
            --latency-threshold 20.0 \
            --hysteresis-time 3.0 \
            $FREQ_FLAG \
            --run-baselines \
            --max-samples 200 \
            --batch-size "$BATCH_SIZE" \
            --num-channels "$NUM_CHANNELS" \
            --output-dir "$OUTPUT_BASE/results"
    fi

    echo ""
    echo "✅ Workload $workload complete"

    # Thermal cooldown between workloads
    if [ "$workload" != "periodic" ]; then
        echo "⏳ Thermal cooldown: ${THERMAL_COOLDOWN} seconds..."
        sleep "$THERMAL_COOLDOWN"
    fi
done

echo ""
echo "✅ Phase 2 complete: All workloads benchmarked"
echo "   Results: $OUTPUT_BASE/results/"
echo ""

# Phase 3: Visualization
echo ""
echo "========================"
echo "PHASE 3: Visualization"
echo "========================"
echo ""

python src/visualize_adaptive_results.py \
    --results-dir "$OUTPUT_BASE/results" \
    --model "$MODEL" \
    --workloads "${WORKLOADS[@]}" \
    --output-dir "$OUTPUT_BASE/figures"

echo ""
echo "✅ Phase 3 complete: Visualizations generated"
echo "   Figures: $OUTPUT_BASE/figures/"
echo ""

# Create summary report
echo ""
echo "Creating summary report..."

SUMMARY_FILE="$OUTPUT_BASE/EXPERIMENT_SUMMARY.md"

cat > "$SUMMARY_FILE" << EOF
# Adaptive Power Management Experiment Summary

**Date**: $(date)
**Model**: $MODEL
**TensorRT**: $USE_TENSORRT
**Model-Specific Defaults**: $USE_MODEL_DEFAULTS

## Experiment Configuration

- **Power Management**: Three-tier adaptive (15W/25W/MAXN) $(if [ "$ENABLE_FREQUENCY_SCALING" = "true" ]; then echo "+ Frequency Scaling"; fi)
- **Threshold Mode**: $(if [ "$AUTO_CALIBRATE" = "true" ]; then echo "Auto-calibrated (SLA: ${TARGET_SLA}ms)"; elif [ "$USE_MODEL_DEFAULTS" = "true" ]; then echo "Model-specific (auto-configured)"; else echo "Manual (20.0ms threshold, 3.0s hysteresis)"; fi)
- **Batch Size**: $BATCH_SIZE $(if [ "$BATCH_SIZE" -gt 1 ]; then echo "(batched inference)"; else echo "(single-sample)"; fi)
- **Channels**: $NUM_CHANNELS $(if [ "$NUM_CHANNELS" -gt 1 ]; then echo "(multi-channel concurrent)"; else echo "(single-channel)"; fi)
- **Duration per Workload**: 60 seconds
- **Workload Patterns**: ${WORKLOADS[*]}
- **Test Samples**: 200

## Directory Structure

\`\`\`
$OUTPUT_BASE/
├── switching_overhead/    # Mode switching characterization results
├── results/               # Raw benchmark results (JSON)
├── figures/              # Visualizations (PNG + summary tables)
└── EXPERIMENT_SUMMARY.md  # This file
\`\`\`

## Results

### Mode Switching Overhead

See: \`switching_overhead/characterization_results.json\`

### Workload Benchmarks

EOF

# Add links to results for each workload
for workload in "${WORKLOADS[@]}"; do
    echo "- **$workload**: \`results/${MODEL}_adaptive_${workload}_results.json\`" >> "$SUMMARY_FILE"
done

cat >> "$SUMMARY_FILE" << EOF

### Visualizations

- Energy-Latency Trade-off: \`figures/${MODEL}_*_energy_latency.png\`
- Latency Timeline: \`figures/${MODEL}_*_timeline.png\`
- Efficiency Comparison: \`figures/${MODEL}_efficiency_comparison.png\`
- Energy Comparison: \`figures/${MODEL}_energy_comparison.png\`
- Latency Comparison: \`figures/${MODEL}_latency_comparison.png\`
- Detailed Summary: \`figures/${MODEL}_summary.md\`

## Next Steps

1. Review the figures in \`figures/\` directory
2. Analyze detailed results in \`results/\` directory
3. Compare with static power mode baselines
4. Identify optimal parameter settings for your workload

## Notes

EOF

# Add switching overhead summary if available
if [ -f "$OUTPUT_BASE/switching_overhead/characterization_results.json" ]; then
    cat >> "$SUMMARY_FILE" << EOF

### Switching Overhead Summary

\`\`\`json
$(cat "$OUTPUT_BASE/switching_overhead/characterization_results.json" | head -30)
...
\`\`\`

EOF
fi

echo "✅ Summary report created: $SUMMARY_FILE"

# Final summary
echo ""
echo "========================================"
echo "🎉 ALL EXPERIMENTS COMPLETE!"
echo "========================================"
echo ""
echo "Results directory: $OUTPUT_BASE/"
echo ""
echo "Key files:"
echo "  - Summary report: $SUMMARY_FILE"
echo "  - Detailed results: $OUTPUT_BASE/results/"
echo "  - Figures: $OUTPUT_BASE/figures/"
echo ""
echo "Next steps:"
echo "  1. Review figures: ls -lh $OUTPUT_BASE/figures/"
echo "  2. Check summary table: cat $OUTPUT_BASE/figures/${MODEL}_summary.md"
echo "  3. Analyze energy savings and latency impact"
echo ""
echo "For detailed documentation, see: ADAPTIVE_POWER_MANAGEMENT.md"
echo ""
