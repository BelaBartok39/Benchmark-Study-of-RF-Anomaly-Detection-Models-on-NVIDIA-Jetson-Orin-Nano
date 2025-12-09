# Adaptive Power Management Experiment Summary

**Date**: Sat Dec  6 08:10:07 PM CST 2025
**Model**: resnet_ae
**TensorRT**: false
**Model-Specific Defaults**: false

## Experiment Configuration

- **Power Management**: Three-tier adaptive (15W/25W/MAXN) 
- **Threshold Mode**: Auto-calibrated (SLA: 10.0ms)
- **Batch Size**: 1 (single-sample)
- **Channels**: 1 (single-channel)
- **Duration per Workload**: 60 seconds
- **Workload Patterns**: bursty continuous variable periodic
- **Test Samples**: 200

## Directory Structure

```
adaptive_experiments_20251206_193808/
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
    "low_to_medium_avg_ms": 21.07987403869629,
    "low_to_medium_median_ms": 21.21579647064209,
    "low_to_medium_std_ms": 0.5461417676747546,
    "low_to_medium_min_ms": 19.780635833740234,
    "low_to_medium_max_ms": 21.776676177978516,
    "low_to_medium_p95_ms": 21.693778038024902,
    "low_to_medium_times_ms": [
      20.701885223388672,
      19.780635833740234,
      20.48206329345703,
      21.579742431640625,
      21.689414978027344,
      21.613359451293945,
      20.647525787353516,
      21.2404727935791,
      21.37899398803711,
      21.776676177978516,
      20.318031311035156,
      21.492958068847656,
      21.00849151611328,
      20.207643508911133,
      20.855188369750977,
      21.191120147705078,
      21.459102630615234,
      21.53611183166504,
      21.45552635192871,
...
```

