#!/bin/bash
#
# Adaptive Power Management Experiment Launcher
# Runs comprehensive experiments for the paper extension
#

set -e  # Exit on error

# Configuration
MODEL=${1:-ae}
USE_TENSORRT=${2:-false}
OUTPUT_BASE="adaptive_experiments_$(date +%Y%m%d_%H%M%S)"

echo "========================================"
echo "ADAPTIVE POWER MANAGEMENT EXPERIMENTS"
echo "========================================"
echo "Model: $MODEL"
echo "Use TensorRT: $USE_TENSORRT"
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

# Phase 1: Mode Switching Characterization
echo ""
echo "========================================="
echo "PHASE 1: Mode Switching Characterization"
echo "========================================="
echo ""

python src/characterize_switching_overhead.py \
    --trials 20 \
    --stabilization-time 2.0 \
    --performance-test \
    --samples 100 \
    --output-dir "$OUTPUT_BASE/switching_overhead"

echo ""
echo "✅ Phase 1 complete: Switching overhead characterized"
echo "   Results: $OUTPUT_BASE/switching_overhead/"
echo ""

# Phase 2: Adaptive Benchmarking
echo ""
echo "==============================="
echo "PHASE 2: Adaptive Benchmarking"
echo "==============================="
echo ""

# Define workload patterns
WORKLOADS=("bursty" "continuous" "variable" "periodic")

# Build TensorRT flag
if [ "$USE_TENSORRT" = "true" ]; then
    TRT_FLAG="--use-tensorrt --engine-path $ENGINE_PATH"
else
    TRT_FLAG=""
fi

# Run benchmark for each workload
for workload in "${WORKLOADS[@]}"; do
    echo ""
    echo "Testing workload: $workload"
    echo "-----------------------------------"

    python src/adaptive_benchmark.py \
        --model "$MODEL" \
        --model-path "$MODEL_PATH" \
        $TRT_FLAG \
        --workload "$workload" \
        --duration 60 \
        --latency-threshold 10.0 \
        --hysteresis-time 5.0 \
        --run-baselines \
        --max-samples 200 \
        --output-dir "$OUTPUT_BASE/results"

    echo ""
    echo "✅ Workload $workload complete"

    # Thermal cooldown between workloads
    if [ "$workload" != "periodic" ]; then
        echo "⏳ Thermal cooldown: 60 seconds..."
        sleep 60
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

## Experiment Configuration

- **Latency Threshold**: 10.0 ms
- **Hysteresis Time**: 5.0 seconds
- **Duration per Workload**: 60 seconds
- **Workload Patterns**: ${WORKLOADS[*]}
- **Test Samples**: 1000

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
