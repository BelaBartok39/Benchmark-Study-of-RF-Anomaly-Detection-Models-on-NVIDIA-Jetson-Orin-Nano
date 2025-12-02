# Adaptive Power Management Experiment Summary

**Date**: Tue Dec  2 02:37:10 PM CST 2025
**Model**: lstm_ae
**TensorRT**: false
**Model-Specific Defaults**: true

## Experiment Configuration

- **Power Management**: Three-tier adaptive (15W/25W/MAXN)
- **Threshold Mode**: Model-specific (auto-configured)
- **Duration per Workload**: 60 seconds
- **Workload Patterns**: bursty continuous variable periodic
- **Test Samples**: 200

## Directory Structure

```
adaptive_experiments_20251202_140552/
├── switching_overhead/    # Mode switching characterization results
├── results/               # Raw benchmark results (JSON)
├── figures/              # Visualizations (PNG + summary tables)
└── EXPERIMENT_SUMMARY.md  # This file
```

## Results

### Mode Switching Overhead

See: `switching_overhead/characterization_results.json`

### Workload Benchmarks

- **bursty**: `results/lstm_ae_adaptive_bursty_results.json`
- **continuous**: `results/lstm_ae_adaptive_continuous_results.json`
- **variable**: `results/lstm_ae_adaptive_variable_results.json`
- **periodic**: `results/lstm_ae_adaptive_periodic_results.json`

### Visualizations

- Energy-Latency Trade-off: `figures/lstm_ae_*_energy_latency.png`
- Latency Timeline: `figures/lstm_ae_*_timeline.png`
- Efficiency Comparison: `figures/lstm_ae_efficiency_comparison.png`
- Energy Comparison: `figures/lstm_ae_energy_comparison.png`
- Latency Comparison: `figures/lstm_ae_latency_comparison.png`
- Detailed Summary: `figures/lstm_ae_summary.md`

## Next Steps

1. Review the figures in `figures/` directory
2. Analyze detailed results in `results/` directory
3. Compare with static power mode baselines
4. Identify optimal parameter settings for your workload

## Notes


### Switching Overhead Summary

```json
{
  "switching_overhead": {
    "num_trials": 20,
    "stabilization_time_s": 2.0,
    "low_to_medium_avg_ms": 22.283685207366943,
    "low_to_medium_median_ms": 22.11916446685791,
    "low_to_medium_std_ms": 1.3794229566425682,
    "low_to_medium_min_ms": 20.528078079223633,
    "low_to_medium_max_ms": 24.927139282226562,
    "low_to_medium_p95_ms": 24.581503868103027,
    "low_to_medium_times_ms": [
      22.985219955444336,
      20.638465881347656,
      23.60081672668457,
      24.563312530517578,
      23.613452911376953,
      23.90575408935547,
      20.89071273803711,
      21.115541458129883,
      21.308183670043945,
      23.248672485351562,
      21.19302749633789,
      22.88198471069336,
      24.927139282226562,
      20.807743072509766,
      23.194074630737305,
      20.528078079223633,
      22.58133888244629,
      21.65699005126953,
      21.120309829711914,
...
```

