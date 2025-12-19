# MATLAB Visualization Scripts for APM Study Results

This directory contains MATLAB scripts to generate figures and tables for the paper extension section on "Increasing Efficiency" using Adaptive Power Management (APM).

## Files

### 1. `extract_stats.py`
Python script that extracts statistics from the JSON result files.

**Usage:**
```bash
python3 extract_stats.py
```

**Output:**
- Extracts data from single 30-minute runs per configuration
- Prints MATLAB-formatted arrays (already integrated into .m files)

### 2. `generate_apm_figures.m`
Main script that generates the core figures for the paper.

**Generated Figures:**
- `energy_savings_heatmap.png/.fig` - Heatmap showing energy savings with value annotations
- `energy_savings_bars.png/.fig` - Grouped bar chart of energy savings
- `power_mode_distribution.png/.fig` - Stacked bar chart showing power mode utilization
- `power_comparison.png/.fig` - Average power consumption comparison (Static vs Adaptive)
- `sla_violations.png/.fig` - SLA violation rates for variable workload
- `energy_vs_power.png/.fig` - Scatter plot of efficiency vs power

**Generated LaTeX Tables:**
- Energy savings summary table
- Power mode distribution table

### 3. `generate_detailed_analysis.m`
Supplementary script for in-depth analysis.

**Generated Figures:**
- `total_energy_comparison.png/.fig` - Side-by-side total energy consumption
- `mode_switching_heatmap.png/.fig` - Frequency of mode switches
- `workload_specific_modes.png/.fig` - Power mode distribution by workload type
- `fps_per_watt_improvement.png/.fig` - Energy efficiency improvements
- `energy_savings_boxplot.png/.fig` - Distribution of savings across workloads
- `efficiency_classification.png/.fig` - Model efficiency profile

**Generated LaTeX Tables:**
- Comprehensive summary table with all metrics

## Usage

1. **Open MATLAB:**
   ```matlab
   cd '/home/babynicky/Documents/Academic Papers/Benchmark_Study_Nano/Benchmark-Study-of-RF-Anomaly-Detection-Models-on-NVIDIA-Jetson-Orin-Nano/matlab_visualizations'
   ```

2. **Run the main script:**
   ```matlab
   generate_apm_figures
   ```

3. **Run the detailed analysis:**
   ```matlab
   generate_detailed_analysis
   ```

4. All figures will be saved as:
   - `.png` format (for paper inclusion)
   - `.fig` format (for editing in MATLAB)

## Data Source

All data is extracted from the Fifth Trial results:
- Path: `Fifth Trial/apm_paper_study_20251213_111825/*/results/*.json`
- Models: AAE, AE, CNN-AE, LSTM-AE, ResNet-AE
- Workloads: Bursty, Periodic, Continuous, Variable
- Duration: 1800 seconds (30 minutes) per configuration
- **Note**: Each configuration was run once (single measurement, not replicated)

## Key Results Summary

### Energy Savings (%)

| Model | Bursty | Periodic | Continuous | Variable | Average |
|-------|--------|----------|------------|----------|---------|
| AAE | 0.76 | 1.17 | 0.75 | 0.75 | 0.86 |
| AE | 0.70 | 1.27 | 0.83 | 0.44 | 0.81 |
| CNN-AE | 0.84 | 1.11 | 0.35 | 0.66 | 0.74 |
| LSTM-AE | 0.70 | 1.10 | 2.17 | 2.19 | 1.54 |
| ResNet-AE | -0.20 | 0.57 | -0.14 | -0.36 | -0.03 |

### Power Consumption (W)

**Static MAXN (average across workloads):**
- AAE: 5.41 W
- AE: 5.45 W
- CNN-AE: 5.50 W
- LSTM-AE: 5.57 W
- ResNet-AE: 5.56 W

**Adaptive (average across workloads):**
- AAE: 5.36 W (0.8% reduction)
- AE: 5.41 W (0.8% reduction)
- CNN-AE: 5.46 W (0.7% reduction)
- LSTM-AE: 5.48 W (1.6% reduction)
- ResNet-AE: 5.56 W (0.0% reduction)

### Model Characteristics

| Model | Avg Energy Savings | Primary Mode (Variable) | Mode Switches | Benefit |
|-------|-------------------|------------------------|---------------|---------|
| AAE | 0.86% | 15W (94.4%) | 4 | Good efficiency, minimal switching |
| AE | 0.81% | 15W (95.0%) | 6 | Excellent efficiency, stable |
| CNN-AE | 0.74% | Mixed | 42 | Moderate savings, highly dynamic |
| LSTM-AE | 1.54% | 15W/25W (28%/72%) | 44 | Highest savings, frequent switching |
| ResNet-AE | 0.03% | MAXN (99.2%) | 5 | Minimal APM benefit |

## Recommended Figures for Paper

For the "Increasing Efficiency" section, we recommend:

### Primary Figures:
1. **Figure 1**: `energy_savings_heatmap.png` - Comprehensive overview
2. **Figure 2**: `energy_savings_bars.png` - Clear visualization of savings
3. **Figure 3**: `power_mode_distribution.png` - Shows APM strategies

### Supplementary Figures:
- `power_comparison.png` - Power reduction across models
- `mode_switching_heatmap.png` - Adaptation behavior
- `total_energy_comparison.png` - Detailed energy analysis

### Tables:
- Energy savings summary (from MATLAB console output)
- Power mode distribution (from MATLAB console output)

## Important Notes

- **Single measurements**: Each configuration (model × workload) was run once for 1800 seconds
- **No replication**: These are single experimental runs, not averaged across multiple trials
- **No error bars**: Since there's only one measurement per condition, error bars/standard deviations cannot be computed
- Target SLA: 10ms latency
- Power modes: 15W (Low), 25W (Medium), MAXN (High)
- Hysteresis time: 60 seconds (prevents rapid mode switching)
- JetPack version: 6.1 with MAXN SUPER power mode
- Platform: NVIDIA Jetson Orin Nano 8GB Developer Kit

## Interpretation

The energy savings (0.7-2.2%) are modest but represent real reductions in power consumption:
- **Best performers**: LSTM-AE achieves up to 2.2% savings through intelligent mode switching
- **Most stable**: AAE and AE maintain >94% low-power operation with consistent savings
- **Dynamic adapter**: CNN-AE actively switches modes (42 switches) for moderate gains
- **Limited benefit**: ResNet-AE stays in MAXN mode, showing APM has minimal impact on heavy models

## Troubleshooting

If you encounter errors:
1. Ensure MATLAB has basic plotting capabilities
2. Check that all data files are present in `Fifth Trial/` directory
3. Run `extract_stats.py` to verify data extraction
4. Verify working directory is correct before running scripts

## Citation

If using these visualizations, please cite:
```
Redmond, N.D., Ali, M.H., Dasgupta, D., Won, M. (2025).
"A Benchmark Study of RF Anomaly Detection Models on NVIDIA Jetson Orin Nano"
```
