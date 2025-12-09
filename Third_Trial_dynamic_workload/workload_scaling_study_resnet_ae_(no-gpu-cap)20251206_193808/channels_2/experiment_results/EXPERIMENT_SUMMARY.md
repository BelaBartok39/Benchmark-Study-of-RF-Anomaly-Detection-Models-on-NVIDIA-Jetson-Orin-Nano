# Adaptive Power Management Experiment Summary

**Date**: Sat Dec  6 08:42:02 PM CST 2025
**Model**: resnet_ae
**TensorRT**: false
**Model-Specific Defaults**: false

## Experiment Configuration

- **Power Management**: Three-tier adaptive (15W/25W/MAXN) 
- **Threshold Mode**: Auto-calibrated (SLA: 10.0ms)
- **Batch Size**: 1 (single-sample)
- **Channels**: 2 (multi-channel concurrent)
- **Duration per Workload**: 60 seconds
- **Workload Patterns**: bursty continuous variable periodic
- **Test Samples**: 200

## Directory Structure

```
adaptive_experiments_20251206_201009/
├── switching_overhead/    # Mode switching characterization results
├── results/               # Raw benchmark results (JSON)
├── figures/              # Visualizations (PNG + summary tables)
└── EXPERIMENT_SUMMARY.md  # This file
```

## Results

### Mode Switching Overhead

See: `switching_overhead/characterization_results.json`

### Workload Benchmarks

- **bursty**: `results/resnet_ae_adaptive_bursty_results.json`
- **continuous**: `results/resnet_ae_adaptive_continuous_results.json`
- **variable**: `results/resnet_ae_adaptive_variable_results.json`
- **periodic**: `results/resnet_ae_adaptive_periodic_results.json`

### Visualizations

- Energy-Latency Trade-off: `figures/resnet_ae_*_energy_latency.png`
- Latency Timeline: `figures/resnet_ae_*_timeline.png`
- Efficiency Comparison: `figures/resnet_ae_efficiency_comparison.png`
- Energy Comparison: `figures/resnet_ae_energy_comparison.png`
- Latency Comparison: `figures/resnet_ae_latency_comparison.png`
- Detailed Summary: `figures/resnet_ae_summary.md`

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
    "low_to_medium_avg_ms": 20.5572247505188,
    "low_to_medium_median_ms": 20.328998565673828,
    "low_to_medium_std_ms": 0.5868346870200543,
    "low_to_medium_min_ms": 19.666671752929688,
    "low_to_medium_max_ms": 21.857500076293945,
    "low_to_medium_p95_ms": 21.586835384368896,
    "low_to_medium_times_ms": [
      20.85399627685547,
      20.290613174438477,
      21.857500076293945,
      21.053314208984375,
      20.233631134033203,
      21.572589874267578,
      20.27416229248047,
      19.959688186645508,
      20.61915397644043,
      20.285844802856445,
      20.75505256652832,
      19.666671752929688,
      20.175457000732422,
      21.229028701782227,
      20.7974910736084,
      21.225452423095703,
      20.36738395690918,
      19.73748207092285,
      19.90485191345215,
...
```

