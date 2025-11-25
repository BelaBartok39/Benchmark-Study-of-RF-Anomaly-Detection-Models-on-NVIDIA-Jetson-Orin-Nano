# Adaptive Power Management Experiment Summary

**Date**: Tue Nov 25 02:59:57 PM CST 2025
**Model**: ae
**TensorRT**: false

## Experiment Configuration

- **Latency Threshold**: 10.0 ms
- **Hysteresis Time**: 5.0 seconds
- **Duration per Workload**: 60 seconds
- **Workload Patterns**: bursty continuous variable periodic
- **Test Samples**: 1000

## Directory Structure

```
adaptive_experiments_20251125_143846/
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
    "low_to_high_avg_ms": 26.662588119506836,
    "low_to_high_median_ms": 22.708892822265625,
    "low_to_high_std_ms": 8.977422054072823,
    "low_to_high_min_ms": 19.49763298034668,
    "low_to_high_max_ms": 54.552555084228516,
    "low_to_high_p95_ms": 44.46865320205689,
    "high_to_low_avg_ms": 23.874902725219727,
    "high_to_low_median_ms": 23.014187812805176,
    "high_to_low_std_ms": 2.313885576027711,
    "high_to_low_min_ms": 20.656108856201172,
    "high_to_low_max_ms": 31.692981719970703,
    "high_to_low_p95_ms": 26.56258344650269,
    "avg_switch_time_ms": 25.26874542236328,
    "low_to_high_times_ms": [
      27.913808822631836,
      21.753549575805664,
      23.58722686767578,
      21.837711334228516,
      21.784067153930664,
      43.93792152404785,
      22.533416748046875,
      23.576021194458008,
      22.718191146850586,
      24.349212646484375,
      38.28239440917969,
      21.15035057067871,
...
```

