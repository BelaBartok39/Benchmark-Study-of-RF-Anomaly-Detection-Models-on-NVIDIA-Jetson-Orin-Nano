# Adaptive Power Management Experiment Summary

**Date**: Tue Dec  9 02:22:02 PM CST 2025
**Model**: resnet_ae
**TensorRT**: false
**Model-Specific Defaults**: false

## Experiment Configuration

- **Power Management**: Three-tier adaptive (15W/25W/MAXN) + Frequency Scaling
- **Threshold Mode**: Auto-calibrated (SLA: 10.0ms)
- **Batch Size**: 1 (single-sample)
- **Channels**: 1 (single-channel)
- **Duration per Workload**: 60 seconds
- **Workload Patterns**: bursty continuous variable periodic
- **Test Samples**: 200

## Directory Structure

```
adaptive_experiments_20251209_135004/
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
    "low_to_medium_avg_ms": 20.871448516845703,
    "low_to_medium_median_ms": 20.753145217895508,
    "low_to_medium_std_ms": 0.6328242381098347,
    "low_to_medium_min_ms": 19.933700561523438,
    "low_to_medium_max_ms": 22.06563949584961,
    "low_to_medium_p95_ms": 22.029852867126465,
    "low_to_medium_times_ms": [
      20.241260528564453,
      20.43437957763672,
      20.988941192626953,
      22.06563949584961,
      20.138025283813477,
      19.933700561523438,
      21.390438079833984,
      20.285606384277344,
      22.027969360351562,
      20.754337310791016,
      20.82514762878418,
      20.669221878051758,
      20.530223846435547,
      21.757125854492188,
      21.86727523803711,
      20.751953125,
      21.188974380493164,
      20.188570022583008,
      20.7974910736084,
...
```

