# Adaptive Power Management Experiment Summary

**Date**: Tue Dec  9 05:24:20 PM CST 2025
**Model**: resnet_ae
**TensorRT**: false
**Model-Specific Defaults**: false

## Experiment Configuration

- **Power Management**: Three-tier adaptive (15W/25W/MAXN) + Frequency Scaling
- **Threshold Mode**: Auto-calibrated (SLA: 10.0ms)
- **Batch Size**: 1 (single-sample)
- **Channels**: 10 (multi-channel concurrent)
- **Duration per Workload**: 60 seconds
- **Workload Patterns**: bursty continuous variable periodic
- **Test Samples**: 200

## Directory Structure

```
adaptive_experiments_20251209_161218/
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
    "low_to_medium_avg_ms": 20.96090316772461,
    "low_to_medium_median_ms": 20.961403846740723,
    "low_to_medium_std_ms": 0.5726289189078521,
    "low_to_medium_min_ms": 20.035982131958008,
    "low_to_medium_max_ms": 21.91638946533203,
    "low_to_medium_p95_ms": 21.805405616760254,
    "low_to_medium_times_ms": [
      20.318984985351562,
      20.26987075805664,
      20.711898803710938,
      21.91638946533203,
      20.62845230102539,
      21.799564361572266,
      20.035982131958008,
      20.510196685791016,
      21.35324478149414,
      20.981788635253906,
      21.48127555847168,
      21.44908905029297,
      20.35379409790039,
      21.120548248291016,
      20.470380783081055,
      21.410226821899414,
      20.258188247680664,
      20.94101905822754,
      21.622419357299805,
...
```

