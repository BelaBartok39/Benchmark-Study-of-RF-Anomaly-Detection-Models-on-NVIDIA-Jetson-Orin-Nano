# Adaptive Power Management Experiment Summary

**Date**: Tue Dec  2 12:21:10 PM CST 2025
**Model**: resnet_ae
**TensorRT**: false

## Experiment Configuration

- **Latency Threshold**: 10.0 ms
- **Hysteresis Time**: 5.0 seconds
- **Duration per Workload**: 60 seconds
- **Workload Patterns**: bursty continuous variable periodic
- **Test Samples**: 1000

## Directory Structure

```
adaptive_experiments_20251202_120011/
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
    "low_to_high_avg_ms": 21.94620370864868,
    "low_to_high_median_ms": 21.325230598449707,
    "low_to_high_std_ms": 2.02969006457912,
    "low_to_high_min_ms": 20.221710205078125,
    "low_to_high_max_ms": 29.19912338256836,
    "low_to_high_p95_ms": 25.07731914520264,
    "high_to_low_avg_ms": 22.822773456573486,
    "high_to_low_median_ms": 22.264719009399414,
    "high_to_low_std_ms": 1.9379039273380918,
    "high_to_low_min_ms": 20.58577537536621,
    "high_to_low_max_ms": 28.548479080200195,
    "high_to_low_p95_ms": 26.17818117141724,
    "avg_switch_time_ms": 22.384488582611084,
    "low_to_high_times_ms": [
      21.500349044799805,
      24.860382080078125,
      21.40021324157715,
      20.688533782958984,
      20.702362060546875,
      21.905899047851562,
      22.2017765045166,
      20.76268196105957,
      21.250247955322266,
      23.490190505981445,
      29.19912338256836,
      22.440433502197266,
...
```

