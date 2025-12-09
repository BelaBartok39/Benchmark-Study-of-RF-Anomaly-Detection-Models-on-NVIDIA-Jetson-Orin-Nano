# Adaptive Power Management Experiment Summary

**Date**: Sat Dec  6 09:59:59 PM CST 2025
**Model**: resnet_ae
**TensorRT**: false
**Model-Specific Defaults**: false

## Experiment Configuration

- **Power Management**: Three-tier adaptive (15W/25W/MAXN) 
- **Threshold Mode**: Auto-calibrated (SLA: 10.0ms)
- **Batch Size**: 1 (single-sample)
- **Channels**: 5 (multi-channel concurrent)
- **Duration per Workload**: 60 seconds
- **Workload Patterns**: bursty continuous variable periodic
- **Test Samples**: 200

## Directory Structure

```
adaptive_experiments_20251206_211431/
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
    "low_to_medium_avg_ms": 20.36222219467163,
    "low_to_medium_median_ms": 20.55203914642334,
    "low_to_medium_std_ms": 0.5584566647081961,
    "low_to_medium_min_ms": 19.475698471069336,
    "low_to_medium_max_ms": 21.554231643676758,
    "low_to_medium_p95_ms": 21.002256870269775,
    "low_to_medium_times_ms": [
      20.602703094482422,
      21.554231643676758,
      19.771099090576172,
      19.627809524536133,
      19.66071128845215,
      20.551443099975586,
      20.97320556640625,
      20.823001861572266,
      20.066261291503906,
      20.16472816467285,
      19.664287567138672,
      20.552635192871094,
      20.29561996459961,
      20.926713943481445,
      19.475698471069336,
      20.71094512939453,
      20.63131332397461,
      20.824909210205078,
      19.670963287353516,
...
```

