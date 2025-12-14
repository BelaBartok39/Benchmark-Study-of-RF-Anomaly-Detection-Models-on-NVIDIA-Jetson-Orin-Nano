#!/bin/bash
#
# APM Paper Study - Automated Endurance Benchmark
#
# This script runs a comprehensive endurance study comparing Static MAXN
# vs Adaptive Power Management for publication in the APM paper.
#
# For each model (ae, aae, cnn_ae, resnet_ae, lstm_ae):
#   - Runs 60-minute tests for MAXN static mode
#   - Runs 60-minute tests for Adaptive mode
#   - Tests all 4 workloads (bursty, periodic, continuous, variable)
#   - Generates visualizations automatically after each model
#
# Total estimated runtime: ~8 hours
#
# Usage:
#   ./run_apm_paper_study.sh
#

set -e  # Exit on error

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
STUDY_BASE="apm_paper_study_${TIMESTAMP}"
DURATION=1800  # 30 minutes
HYSTERESIS=30.0 # Set to 60 for one-hour durations
SPARSITY=5.0 # Set to 10.0 for one-hour durations
NUM_CHANNELS=1
WORKLOADS=("bursty" "periodic" "continuous" "variable")
MODELS=("resnet_ae")

# Create base study directory
mkdir -p "$STUDY_BASE"
# ... (rest of script until python command) ...
        python src/adaptive_benchmark.py \
            --model "$model" \
            --model-path "$MODEL_PATH" \
            $TRT_FLAG \
            --workload "$workload" \
            --duration "$DURATION" \
            --auto-calibrate \
            --target-sla 20.0 \
            --hysteresis-time "$HYSTERESIS" \
            --sparsity-factor "$SPARSITY" \
            --run-baselines \
            --batch-size 1 \
            --num-channels "$NUM_CHANNELS" \
            --output-dir "$model_output/results" \
            2>&1 | tee -a "$model_output/${workload}_run_log.txt"
# ... (rest of script until visualization) ...
    python src/visualize_adaptive_results.py \
        --results-dir "$model_output/results" \
        --model "$model" \
        --workloads "${WORKLOADS[@]}" \
        --output-dir "$model_output/figures" \
        2>&1 | tee -a "$model_output/visualization_log.txt"

    echo "✅ Visualizations complete for $model"  | tee -a "$LOG_FILE"
    echo "📊 Figures saved to: $model_output/figures/"  | tee -a "$LOG_FILE"

    # Push results to GitHub
    echo "--------------------------------------------------------------------" | tee -a "$LOG_FILE"
    echo "Pushing results to GitHub..." | tee -a "$LOG_FILE"
    git add -A
    git commit -m "Benchmark results for $model (SLA 20ms)"
    git push
    echo "✅ Pushed to GitHub" | tee -a "$LOG_FILE"

    echo ""  | tee -a "$LOG_FILE"
    echo "========================================================================"  | tee -a "$LOG_FILE"
    echo "✅ MODEL COMPLETE: $model"  | tee -a "$LOG_FILE"
    echo "End time: $(date)"  | tee -a "$LOG_FILE"
    echo "========================================================================"  | tee -a "$LOG_FILE"
    echo ""  | tee -a "$LOG_FILE"
}

# Main execution loop
echo "Starting benchmark at $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

for model in "${MODELS[@]}"; do
    run_model_benchmark "$model"

    # Extra cooldown between models
    if [ "$model" != "${MODELS[-1]}" ]; then
        echo "⏳ Extended cooldown between models (5 minutes)..."  | tee -a "$LOG_FILE"
        sleep 300
    fi
done

# Final summary
echo ""  | tee -a "$LOG_FILE"
echo "========================================================================"  | tee -a "$LOG_FILE"
echo "🎉 STUDY COMPLETE!"  | tee -a "$LOG_FILE"
echo "========================================================================"  | tee -a "$LOG_FILE"
echo "Completion time: $(date)"  | tee -a "$LOG_FILE"
echo ""  | tee -a "$LOG_FILE"
echo "Results organized by model in: $STUDY_BASE/"  | tee -a "$LOG_FILE"
echo ""  | tee -a "$LOG_FILE"
echo "Directory structure:"  | tee -a "$LOG_FILE"
for model in "${MODELS[@]}"; do
    echo "  $STUDY_BASE/${model}_results/"  | tee -a "$LOG_FILE"
    echo "    ├── results/           (JSON result files + calibration data)"  | tee -a "$LOG_FILE"
    echo "    └── figures/           (Visualization plots)"  | tee -a "$LOG_FILE"
done
echo ""  | tee -a "$LOG_FILE"
echo "📊 All visualizations have been generated automatically."  | tee -a "$LOG_FILE"
echo "========================================================================"  | tee -a "$LOG_FILE"
