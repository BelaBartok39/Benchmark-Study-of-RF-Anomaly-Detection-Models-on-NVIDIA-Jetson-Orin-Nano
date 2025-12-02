# Adaptive Power Management Experiment Summary

**Date**: Tue Dec  2 11:08:48 AM CST 2025
**Model**: cnn_ae
**TensorRT**: false

## Experiment Configuration

- **Latency Threshold**: 10.0 ms
- **Hysteresis Time**: 5.0 seconds
- **Duration per Workload**: 60 seconds
- **Workload Patterns**: bursty continuous variable periodic
- **Test Samples**: 1000

## Directory Structure

```
adaptive_experiments_20251202_104743/
├── switching_overhead/    # Mode switching characterization results
├── results/               # Raw benchmark results (JSON)
├── figures/              # Visualizations (PNG + summary tables)
└── EXPERIMENT_SUMMARY.md  # This file
```

## Results

### Mode Switching Overhead

See: `switching_overhead/characterization_results.json`

### Workload Benchmarks

- **bursty**: `results/cnn_ae_adaptive_bursty_results.json`
- **continuous**: `results/cnn_ae_adaptive_continuous_results.json`
- **variable**: `results/cnn_ae_adaptive_variable_results.json`
- **periodic**: `results/cnn_ae_adaptive_periodic_results.json`

### Visualizations

- Energy-Latency Trade-off: `figures/cnn_ae_*_energy_latency.png`
- Latency Timeline: `figures/cnn_ae_*_timeline.png`
- Efficiency Comparison: `figures/cnn_ae_efficiency_comparison.png`
- Energy Comparison: `figures/cnn_ae_energy_comparison.png`
- Latency Comparison: `figures/cnn_ae_latency_comparison.png`
- Detailed Summary: `figures/cnn_ae_summary.md`

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
    "low_to_high_avg_ms": 22.864198684692383,
    "low_to_high_median_ms": 21.45063877105713,
    "low_to_high_std_ms": 3.701320825210511,
    "low_to_high_min_ms": 20.16305923461914,
    "low_to_high_max_ms": 36.63039207458496,
    "low_to_high_p95_ms": 28.689384460449226,
    "high_to_low_avg_ms": 23.642408847808838,
    "high_to_low_median_ms": 23.178458213806152,
    "high_to_low_std_ms": 1.538765458987304,
    "high_to_low_min_ms": 21.801233291625977,
    "high_to_low_max_ms": 27.045011520385742,
    "high_to_low_p95_ms": 27.031195163726807,
    "avg_switch_time_ms": 23.253303766250607,
    "low_to_high_times_ms": [
      20.9197998046875,
      21.056652069091797,
      21.285533905029297,
      20.641565322875977,
      20.70140838623047,
      23.392677307128906,
      20.892620086669922,
      24.023056030273438,
      36.63039207458496,
      20.362377166748047,
      28.27143669128418,
      22.371530532836914,
...
```

