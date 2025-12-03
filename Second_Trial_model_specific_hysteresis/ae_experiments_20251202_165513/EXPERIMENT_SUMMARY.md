# Adaptive Power Management Experiment Summary

**Date**: Tue Dec  2 05:26:19 PM CST 2025
**Model**: ae
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
adaptive_experiments_20251202_165513/
├── switching_overhead/    # Mode switching characterization results
├── results/               # Raw benchmark results (JSON)
├── figures/              # Visualizations (PNG + summary tables)
└── EXPERIMENT_SUMMARY.md  # This file
```

## Results

### Mode Switching Overhead

See: `switching_overhead/characterization_results.json`

### Workload Benchmarks

- **bursty**: `results/ae_adaptive_bursty_results.json`
- **continuous**: `results/ae_adaptive_continuous_results.json`
- **variable**: `results/ae_adaptive_variable_results.json`
- **periodic**: `results/ae_adaptive_periodic_results.json`

### Visualizations

- Energy-Latency Trade-off: `figures/ae_*_energy_latency.png`
- Latency Timeline: `figures/ae_*_timeline.png`
- Efficiency Comparison: `figures/ae_efficiency_comparison.png`
- Energy Comparison: `figures/ae_energy_comparison.png`
- Latency Comparison: `figures/ae_latency_comparison.png`
- Detailed Summary: `figures/ae_summary.md`

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
    "low_to_medium_avg_ms": 21.31187915802002,
    "low_to_medium_median_ms": 21.266937255859375,
    "low_to_medium_std_ms": 0.9002814999782146,
    "low_to_medium_min_ms": 18.86606216430664,
    "low_to_medium_max_ms": 23.126840591430664,
    "low_to_medium_p95_ms": 22.456860542297367,
    "low_to_medium_times_ms": [
      20.935535430908203,
      21.897315979003906,
      21.068572998046875,
      21.013736724853516,
      20.415782928466797,
      20.49398422241211,
      21.342992782592773,
      22.421598434448242,
      21.951675415039062,
      20.435094833374023,
      22.185564041137695,
      22.09162712097168,
      21.190881729125977,
      18.86606216430664,
      20.99776268005371,
      21.38686180114746,
      23.126840591430664,
      21.465778350830078,
      20.82228660583496,
...
```

