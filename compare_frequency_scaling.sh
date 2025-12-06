#!/bin/bash
#
# Quick comparison script for GPU Frequency Scaling
# Runs a short bursty workload with and without frequency scaling to measure energy savings.
#
# Usage: ./compare_frequency_scaling.sh [MODEL]
#

set -e

MODEL=${1:-lstm_ae}
OUTPUT_DIR="frequency_comparison_$(date +%Y%m%d_%H%M%S)"

echo "=================================================="
echo "GPU FREQUENCY SCALING COMPARISON: $MODEL"
echo "=================================================="
echo "Output Directory: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# Common settings for a quick but representative test
DURATION=30
WORKLOAD="bursty"
BATCH_SIZE=1

# 1. Run WITHOUT Frequency Scaling (Baseline)
echo "--------------------------------------------------"
echo "1. Running BASELINE (No Frequency Scaling)..."
echo "--------------------------------------------------"

ENABLE_FREQUENCY_SCALING=false \
DURATION=$DURATION \
WORKLOAD=$WORKLOAD \
BATCH_SIZE=$BATCH_SIZE \
./run_adaptive_experiments.sh "$MODEL" false true > "$OUTPUT_DIR/baseline.log" 2>&1

# Move results to subfolder
mkdir -p "$OUTPUT_DIR/baseline"
mv adaptive_experiments_*/* "$OUTPUT_DIR/baseline/"
rmdir adaptive_experiments_*

echo "✓ Baseline run complete."

# Cooldown
echo "⏳ Cooling down for 15s..."
sleep 15

# 2. Run WITH Frequency Scaling
echo "--------------------------------------------------"
echo "2. Running WITH FREQUENCY SCALING..."
echo "--------------------------------------------------"

ENABLE_FREQUENCY_SCALING=true \
DURATION=$DURATION \
WORKLOAD=$WORKLOAD \
BATCH_SIZE=$BATCH_SIZE \
./run_adaptive_experiments.sh "$MODEL" false true > "$OUTPUT_DIR/scaling.log" 2>&1

# Move results to subfolder
mkdir -p "$OUTPUT_DIR/scaling"
mv adaptive_experiments_*/* "$OUTPUT_DIR/scaling/"
rmdir adaptive_experiments_*

echo "✓ Frequency scaling run complete."

# 3. Analyze and Visualize
echo "--------------------------------------------------"
echo "3. Analyzing Results..."
echo "--------------------------------------------------"

# Create a simple python script to compare and plot
cat > "$OUTPUT_DIR/analyze_comparison.py" << 'EOF'
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

try:
    # Find the result files
    base_dir = "baseline/results"
    scale_dir = "scaling/results"
    
    # Locate the adaptive results JSON (filename varies by timestamp/model)
    base_file = [f for f in os.listdir(base_dir) if "adaptive" in f and "bursty" in f][0]
    scale_file = [f for f in os.listdir(scale_dir) if "adaptive" in f and "bursty" in f][0]
    
    base_data = load_json(os.path.join(base_dir, base_file))
    scale_data = load_json(os.path.join(scale_dir, scale_file))
    
    # Extract metrics
    metrics = {
        'Total Energy (J)': ('total_energy_j', 1.0),
        'Avg Power (W)': ('avg_power_w', 1.0),
        'Energy/Inf (mJ)': ('energy_per_inference_j', 1000.0),
        'P95 Latency (ms)': ('p95_latency_ms', 1.0)
    }
    
    # Print Comparison Table
    print(f"\n{'Metric':<20} | {'Baseline':<12} | {'With Scaling':<12} | {'Change':<10}")
    print("-" * 60)
    
    results = {}
    for name, (key, factor) in metrics.items():
        b_val = base_data.get(key, 0) * factor
        s_val = scale_data.get(key, 0) * factor
        
        # Calculate percent change
        if b_val != 0:
            pct_change = ((s_val - b_val) / b_val) * 100
            change_str = f"{pct_change:+.1f}%"
        else:
            change_str = "N/A"
            
        print(f"{name:<20} | {b_val:<12.2f} | {s_val:<12.2f} | {change_str:<10}")
        results[name] = (b_val, s_val, change_str)

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Energy Plot
    labels = ['Baseline', 'With Scaling']
    energy_vals = [results['Total Energy (J)'][0], results['Total Energy (J)'][1]]
    colors = ['#3498db', '#2ecc71']
    
    bars = ax1.bar(labels, energy_vals, color=colors, alpha=0.8, edgecolor='black', width=0.5)
    ax1.set_ylabel('Total Energy (Joules)', fontweight='bold')
    ax1.set_title(f'Energy Consumption Comparison\n({results["Total Energy (J)"][2]})', fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f} J',
                ha='center', va='bottom', fontweight='bold')

    # Efficiency Plot
    eff_vals = [results['Energy/Inf (mJ)'][0], results['Energy/Inf (mJ)'][1]]
    bars2 = ax2.bar(labels, eff_vals, color=colors, alpha=0.8, edgecolor='black', width=0.5)
    ax2.set_ylabel('Energy per Inference (mJ)', fontweight='bold')
    ax2.set_title(f'Efficiency Comparison\n({results["Energy/Inf (mJ)"][2]})', fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f} mJ',
                ha='center', va='bottom', fontweight='bold')

    plt.suptitle('Impact of GPU Frequency Scaling', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('comparison_plot.png', dpi=300)
    print(f"\n📊 Plot saved to {os.getcwd()}/comparison_plot.png")

except Exception as e:
    print(f"\n❌ Error analyzing results: {e}")
    sys.exit(1)
EOF

# Run analysis
cd "$OUTPUT_DIR"
python3 analyze_comparison.py
cd ..

echo ""
echo "=================================================="
echo "COMPARISON COMPLETE"
echo "=================================================="
echo "Results saved in: $OUTPUT_DIR"
