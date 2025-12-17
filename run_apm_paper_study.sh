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
MODELS=("ae" "aae" "cnn_ae" "resnet_ae" "lstm_ae")
# MODELS=("resnet_ae") # Running only ResNet for the 20ms SLA experiment

# Create base study directory
mkdir -p "$STUDY_BASE"

echo "========================================================================"
echo "APM PAPER STUDY - AUTOMATED ENDURANCE BENCHMARK"
echo "========================================================================"
echo "Study ID: $STUDY_BASE"
echo "Duration per run: ${DURATION}s (30 minutes)"
echo "Hysteresis: ${HYSTERESIS}s"
echo "Sparsity Factor: ${SPARSITY}x"
echo "Num Channels: $NUM_CHANNELS"
echo ""
echo "Models to test: ${MODELS[@]}"
echo "Workloads: ${WORKLOADS[@]}"
echo ""
echo "Comparison: Static MAXN vs Adaptive Power Management"
echo ""
echo "Expected total runtime: ~2 hours (Single Model)"
echo "  - 5 models × 4 workloads × 2 modes × 30 min = 4 runs × 30 min = 1200 min"
echo "  - Plus cooldown periods and visualization generation"
echo "========================================================================"
echo ""

# Log file
LOG_FILE="$STUDY_BASE/study_log.txt"
echo "Study started at $(date)" | tee -a "$LOG_FILE"

# Function to run single model benchmark
run_model_benchmark() {
    local model=$1
    local model_output="$STUDY_BASE/${model}_results"

    echo "" | tee -a "$LOG_FILE"
    echo "========================================================================"  | tee -a "$LOG_FILE"
    echo "BENCHMARKING MODEL: $model"  | tee -a "$LOG_FILE"
    echo "========================================================================"  | tee -a "$LOG_FILE"
    echo "Start time: $(date)"  | tee -a "$LOG_FILE"
    echo ""  | tee -a "$LOG_FILE"

    # Set model paths
    MODEL_PATH="src/output/weights/${model}_best.pth"
    ENGINE_PATH="src/output/engines/${model}.engine"

    # Build TensorRT flag
    if [ -f "$ENGINE_PATH" ]; then
        TRT_FLAG="--use-tensorrt --engine-path $ENGINE_PATH"
        echo "Using TensorRT engine: $ENGINE_PATH"  | tee -a "$LOG_FILE"
    else
        TRT_FLAG=""
        echo "Using PyTorch model: $MODEL_PATH"  | tee -a "$LOG_FILE"
    fi

    # Create model output directory
    mkdir -p "$model_output/results"

    # Run for each workload
    for workload in "${WORKLOADS[@]}"; do
        echo ""  | tee -a "$LOG_FILE"
        echo "--------------------------------------------------------------------"  | tee -a "$LOG_FILE"
        echo "Model: $model | Workload: $workload"  | tee -a "$LOG_FILE"
        echo "--------------------------------------------------------------------"  | tee -a "$LOG_FILE"
        echo "Time: $(date)"  | tee -a "$LOG_FILE"
        echo ""  | tee -a "$LOG_FILE"

        # Run MAXN baseline + Adaptive with auto-calibration
        # Auto-calibration will:
        #   1. Profile hardware (30s)
        #   2. Determine optimal thresholds
        #   3. Use those thresholds for the endurance tests
        echo "Step 1/2: Running endurance tests with auto-calibration (MAXN + Adaptive, ~2 hours)..."  | tee -a "$LOG_FILE"
        echo "  - Hardware profiling: ~30s"  | tee -a "$LOG_FILE"
        echo "  - MAXN baseline: 30 min"  | tee -a "$LOG_FILE"
        echo "  - Adaptive test: 30 min"  | tee -a "$LOG_FILE"
        python src/adaptive_benchmark.py \
            --model "$model" \
            --model-path "$MODEL_PATH" \
            $TRT_FLAG \
            --workload "$workload" \
            --duration "$DURATION" \
            --auto-calibrate \
            --target-sla 10.0 \
            --hysteresis-time "$HYSTERESIS" \
            --sparsity-factor "$SPARSITY" \
            --run-baselines \
            --batch-size 1 \
            --num-channels "$NUM_CHANNELS" \
            --output-dir "$model_output/results" \
            2>&1 | tee -a "$model_output/${workload}_run_log.txt"

        echo "✅ Completed: $model - $workload"  | tee -a "$LOG_FILE"
        echo "Time: $(date)"  | tee -a "$LOG_FILE"

        # Step 2: Thermal cooldown between workloads
        if [ "$workload" != "${WORKLOADS[-1]}" ]; then
            echo "⏳ Thermal cooldown (2 minutes)..."  | tee -a "$LOG_FILE"
            sleep 120
        fi
    done

    # Generate visualizations for this model
    echo ""  | tee -a "$LOG_FILE"
    echo "--------------------------------------------------------------------"  | tee -a "$LOG_FILE"
    echo "Generating visualizations for $model..."  | tee -a "$LOG_FILE"
    echo "--------------------------------------------------------------------"  | tee -a "$LOG_FILE"

    mkdir -p "$model_output/figures"

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
    git commit -m "Benchmark results for $model (SLA 10ms)"
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
