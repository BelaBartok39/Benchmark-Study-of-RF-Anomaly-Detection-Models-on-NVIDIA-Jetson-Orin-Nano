#!/bin/bash
#
# Workload Scaling Study for Adaptive Power Management
#
# This script systematically tests different channel configurations
# to discover hardware-specific achievable SLAs and evaluate adaptive
# power management performance across different deployment scenarios.
#
# The auto-calibration system will discover the minimum achievable SLA
# for each workload configuration and document the results.
#
# Usage:
#   ./run_workload_scaling_study.sh [MODEL]
#
# Example:
#   ./run_workload_scaling_study.sh lstm_ae
#

set -e  # Exit on error

# Configuration
NUM_CHANNELS=${NUM_CHANNELS:-1}
MODEL=${1:-lstm_ae}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="workload_scaling_study_${MODEL}_${TIMESTAMP}"
RESULTS_FILE="$OUTPUT_BASE/scaling_summary.txt"

# Test configurations: different channel counts
CHANNEL_CONFIGS=(1 2 3 5 10)

# Create output directory
mkdir -p "$OUTPUT_BASE"

echo "======================================================="
echo "WORKLOAD SCALING STUDY - ADAPTIVE POWER MANAGEMENT"
echo "======================================================="
echo "Model: $MODEL"
echo "Timestamp: $TIMESTAMP"
echo "Channel configurations: ${CHANNEL_CONFIGS[@]}"
echo ""
echo "This study will:"
echo "  1. Profile hardware at each workload level"
echo "  2. Discover minimum achievable SLA for each configuration"
echo "  3. Run adaptive experiments with discovered SLAs"
echo "  4. Generate comparison report"
echo ""
echo "Expected runtime: ~15-20 minutes per configuration"
echo "======================================================="
echo ""

# Initialize results file
cat > "$RESULTS_FILE" << EOF
========================================================
WORKLOAD SCALING STUDY RESULTS
========================================================
Model: $MODEL
Date: $(date)
Jetson Model: NVIDIA Orin Nano

This study evaluates adaptive power management across different
workload intensities (number of concurrent RF channels).

========================================================
CONFIGURATION SUMMARY
========================================================

EOF

# Run experiments for each channel configuration
for num_channels in "${CHANNEL_CONFIGS[@]}"; do
    echo ""
    echo "========================================================="
    echo "TESTING: $num_channels Channel(s)"
    echo "========================================================="
    echo ""

    # Create subdirectory for this configuration
    CONFIG_DIR="$OUTPUT_BASE/channels_${num_channels}"
    mkdir -p "$CONFIG_DIR"

    # Step 1: Discovery phase - Find achievable SLA
    echo "Step 1/2: Discovering achievable SLA for $num_channels channel(s)..."
    echo "         (This will profile hardware and auto-adjust if needed)"
    echo ""

    # Run with auto-calibration and auto-adjustment
    # Redirect output to capture suggested SLA
    AUTO_CALIBRATE=true \
    AUTO_ADJUST_SLA=true \
    NUM_CHANNELS=$num_channels \
    TARGET_SLA=10.0 \
    ./run_adaptive_experiments.sh "$MODEL" false false \
        2>&1 | tee "$CONFIG_DIR/discovery_output.log"

    # Extract the adjusted SLA from the output (if it was adjusted)
    # Look for "Adjusted to: XXXms" in the calibration output
    DISCOVERED_SLA=$(grep -oP "Adjusted to:\s+\K[0-9.]+(?=ms)" "$CONFIG_DIR/discovery_output.log" | head -1)

    if [ -z "$DISCOVERED_SLA" ]; then
        # No adjustment needed, 10ms was achievable
        DISCOVERED_SLA="10.0"
        SLA_STATUS="✓ Achievable (no adjustment needed)"
    else
        SLA_STATUS="⚠ Auto-adjusted from 10.0ms"
    fi

    echo ""
    echo "========================================================="
    echo "RESULTS: $num_channels Channel(s)"
    echo "========================================================="
    echo "Discovered SLA: ${DISCOVERED_SLA}ms"
    echo "Status: $SLA_STATUS"
    echo ""

    # Step 2: Move results to organized directory
    LATEST_RESULTS=$(ls -td adaptive_experiments_* | head -1)
    if [ -d "$LATEST_RESULTS" ]; then
        mv "$LATEST_RESULTS" "$CONFIG_DIR/experiment_results"
        echo "Results saved to: $CONFIG_DIR/experiment_results"
    fi

    # Append to summary file
    cat >> "$RESULTS_FILE" << EOF
Configuration: $num_channels Channel(s)
  Target SLA:      10.0ms
  Achieved SLA:    ${DISCOVERED_SLA}ms
  Status:          $SLA_STATUS
  Results Dir:     $CONFIG_DIR/experiment_results/

EOF

    echo ""
    echo "✓ Completed: $num_channels channel(s) - SLA=${DISCOVERED_SLA}ms"
    echo ""

    # Brief pause between configurations
    sleep 2
done

# Generate final summary
echo ""
echo "========================================================="
echo "STUDY COMPLETE!"
echo "========================================================="
echo ""
echo "Results summary:"
cat "$RESULTS_FILE"

# Add analysis section
cat >> "$RESULTS_FILE" << EOF

========================================================
ANALYSIS RECOMMENDATIONS
========================================================

Review the discovered SLAs for each configuration:

1. Deployment Guidance:
   - Single channel (1-2): Lowest power modes, best efficiency
   - Medium load (3-5): Balanced power/performance
   - Heavy load (10+): Requires higher power modes, focus on meeting SLA

2. Paper Integration:
   - Table: Channel count vs Achievable SLA vs Energy savings
   - Graph: Workload scaling (x=channels, y=energy/inference)
   - Discussion: Trade-offs between workload and efficiency

3. Novel Contributions:
   - Hardware-aware SLA discovery (not manual tuning)
   - Workload-specific power management strategies
   - Practical deployment guidance for Jetson Orin Nano

4. Key Findings to Highlight:
   - Compare energy savings across configurations
   - Show how adaptive benefit changes with workload
   - Demonstrate real-world applicability

For detailed results, see individual directories:
$(ls -d $OUTPUT_BASE/channels_*)

========================================================
EOF

echo ""
echo "📊 Full results saved to: $OUTPUT_BASE/"
echo "📄 Summary report: $RESULTS_FILE"
echo ""
echo "Next steps:"
echo "  1. Review the scaling summary file"
echo "  2. Compare energy efficiency across configurations"
echo "  3. Use results for paper Table/Figure"
echo "  4. Document deployment recommendations"
echo ""
echo "✅ Workload scaling study complete!"
