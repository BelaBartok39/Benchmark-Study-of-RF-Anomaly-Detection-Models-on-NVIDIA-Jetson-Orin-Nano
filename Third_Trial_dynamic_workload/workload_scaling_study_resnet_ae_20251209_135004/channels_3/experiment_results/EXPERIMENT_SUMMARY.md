# Adaptive Power Management Experiment Summary

**Date**: Tue Dec  9 03:26:53 PM CST 2025
**Model**: resnet_ae
**TensorRT**: false
**Model-Specific Defaults**: false

## Experiment Configuration

- **Power Management**: Three-tier adaptive (15W/25W/MAXN) + Frequency Scaling
- **Threshold Mode**: Auto-calibrated (SLA: 10.0ms)
- **Batch Size**: 1 (single-sample)
- **Channels**: 3 (multi-channel concurrent)
- **Duration per Workload**: 60 seconds
- **Workload Patterns**: bursty continuous variable periodic
- **Test Samples**: 200

## Directory Structure

```
adaptive_experiments_20251209_145358/
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
    "low_to_medium_avg_ms": 21.345674991607666,
    "low_to_medium_median_ms": 21.163105964660645,
    "low_to_medium_std_ms": 0.7301605861921023,
    "low_to_medium_min_ms": 20.305156707763672,
    "low_to_medium_max_ms": 23.128032684326172,
    "low_to_medium_p95_ms": 22.55431413650513,
    "low_to_medium_times_ms": [
      20.305156707763672,
      20.812273025512695,
      21.38972282409668,
      20.823955535888672,
      22.524118423461914,
      20.475149154663086,
      20.447254180908203,
      21.24476432800293,
      22.003650665283203,
      21.08144760131836,
      22.417306900024414,
      21.060466766357422,
      21.67510986328125,
      20.72453498840332,
      23.128032684326172,
      21.340131759643555,
      20.946502685546875,
      21.661043167114258,
      20.989418029785156,
...
```

