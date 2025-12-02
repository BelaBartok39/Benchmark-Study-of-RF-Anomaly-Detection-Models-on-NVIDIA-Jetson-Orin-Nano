# Adaptive Power Management Experiment Summary

**Date**: Tue Dec  2 10:14:23 AM CST 2025
**Model**: lstm_ae
**TensorRT**: false

## Experiment Configuration

- **Latency Threshold**: 10.0 ms
- **Hysteresis Time**: 5.0 seconds
- **Duration per Workload**: 60 seconds
- **Workload Patterns**: bursty continuous variable periodic
- **Test Samples**: 1000

## Directory Structure

```
adaptive_experiments_20251202_095317/
├── switching_overhead/    # Mode switching characterization results
├── results/               # Raw benchmark results (JSON)
├── figures/              # Visualizations (PNG + summary tables)
└── EXPERIMENT_SUMMARY.md  # This file
```

## Results

### Mode Switching Overhead

See: `switching_overhead/characterization_results.json`

### Workload Benchmarks

- **bursty**: `results/lstm_ae_adaptive_bursty_results.json`
- **continuous**: `results/lstm_ae_adaptive_continuous_results.json`
- **variable**: `results/lstm_ae_adaptive_variable_results.json`
- **periodic**: `results/lstm_ae_adaptive_periodic_results.json`

### Visualizations

- Energy-Latency Trade-off: `figures/lstm_ae_*_energy_latency.png`
- Latency Timeline: `figures/lstm_ae_*_timeline.png`
- Efficiency Comparison: `figures/lstm_ae_efficiency_comparison.png`
- Energy Comparison: `figures/lstm_ae_energy_comparison.png`
- Latency Comparison: `figures/lstm_ae_latency_comparison.png`
- Detailed Summary: `figures/lstm_ae_summary.md`

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
    "low_to_high_avg_ms": 21.127712726593018,
    "low_to_high_median_ms": 20.80678939819336,
    "low_to_high_std_ms": 0.8509780412558787,
    "low_to_high_min_ms": 20.216941833496094,
    "low_to_high_max_ms": 23.670673370361328,
    "low_to_high_p95_ms": 22.652113437652588,
    "high_to_low_avg_ms": 23.191392421722412,
    "high_to_low_median_ms": 23.013591766357422,
    "high_to_low_std_ms": 1.1563499042817291,
    "high_to_low_min_ms": 21.539688110351562,
    "high_to_low_max_ms": 25.614023208618164,
    "high_to_low_p95_ms": 25.312554836273193,
    "avg_switch_time_ms": 22.159552574157715,
    "low_to_high_times_ms": [
      20.301342010498047,
      20.769119262695312,
      22.5985050201416,
      20.216941833496094,
      20.7064151763916,
      20.489931106567383,
      21.242856979370117,
      20.364761352539062,
      20.844459533691406,
      20.473003387451172,
      21.209001541137695,
      22.32074737548828,
...
```

