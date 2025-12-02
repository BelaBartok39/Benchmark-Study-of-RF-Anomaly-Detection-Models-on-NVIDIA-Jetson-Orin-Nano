# Adaptive Power Management Experiment Summary

**Date**: Tue Dec  2 11:34:26 AM CST 2025
**Model**: aae
**TensorRT**: false

## Experiment Configuration

- **Latency Threshold**: 10.0 ms
- **Hysteresis Time**: 5.0 seconds
- **Duration per Workload**: 60 seconds
- **Workload Patterns**: bursty continuous variable periodic
- **Test Samples**: 1000

## Directory Structure

```
adaptive_experiments_20251202_111329/
├── switching_overhead/    # Mode switching characterization results
├── results/               # Raw benchmark results (JSON)
├── figures/              # Visualizations (PNG + summary tables)
└── EXPERIMENT_SUMMARY.md  # This file
```

## Results

### Mode Switching Overhead

See: `switching_overhead/characterization_results.json`

### Workload Benchmarks

- **bursty**: `results/aae_adaptive_bursty_results.json`
- **continuous**: `results/aae_adaptive_continuous_results.json`
- **variable**: `results/aae_adaptive_variable_results.json`
- **periodic**: `results/aae_adaptive_periodic_results.json`

### Visualizations

- Energy-Latency Trade-off: `figures/aae_*_energy_latency.png`
- Latency Timeline: `figures/aae_*_timeline.png`
- Efficiency Comparison: `figures/aae_efficiency_comparison.png`
- Energy Comparison: `figures/aae_energy_comparison.png`
- Latency Comparison: `figures/aae_latency_comparison.png`
- Detailed Summary: `figures/aae_summary.md`

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
    "low_to_high_avg_ms": 21.7049241065979,
    "low_to_high_median_ms": 21.250486373901367,
    "low_to_high_std_ms": 1.052649405377324,
    "low_to_high_min_ms": 20.5075740814209,
    "low_to_high_max_ms": 23.656129837036133,
    "low_to_high_p95_ms": 23.60290288925171,
    "high_to_low_avg_ms": 23.084819316864014,
    "high_to_low_median_ms": 22.718310356140137,
    "high_to_low_std_ms": 1.3786854956627617,
    "high_to_low_min_ms": 21.361589431762695,
    "high_to_low_max_ms": 27.591466903686523,
    "high_to_low_p95_ms": 24.915850162506107,
    "avg_switch_time_ms": 22.394871711730957,
    "low_to_high_times_ms": [
      23.396968841552734,
      21.27242088317871,
      21.335840225219727,
      23.600101470947266,
      21.271944046020508,
      20.638704299926758,
      21.647930145263672,
      23.333311080932617,
      22.746801376342773,
      21.111726760864258,
      21.20828628540039,
      20.632028579711914,
...
```

