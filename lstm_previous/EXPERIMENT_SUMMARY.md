# Adaptive Power Management Experiment Summary

**Date**: Tue Dec  2 12:51:19 PM CST 2025
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
adaptive_experiments_20251202_123010/
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
    "low_to_high_avg_ms": 21.82765007019043,
    "low_to_high_median_ms": 21.02351188659668,
    "low_to_high_std_ms": 2.6041713067104757,
    "low_to_high_min_ms": 20.373821258544922,
    "low_to_high_max_ms": 32.422780990600586,
    "low_to_high_p95_ms": 25.07972717285157,
    "high_to_low_avg_ms": 22.99889326095581,
    "high_to_low_median_ms": 22.783875465393066,
    "high_to_low_std_ms": 1.1219066120822863,
    "high_to_low_min_ms": 21.658658981323242,
    "high_to_low_max_ms": 26.325702667236328,
    "high_to_low_p95_ms": 25.15697479248047,
    "avg_switch_time_ms": 22.413271665573124,
    "low_to_high_times_ms": [
      21.13032341003418,
      20.802974700927734,
      20.64037322998047,
      20.735979080200195,
      21.670818328857422,
      24.69325065612793,
      21.95000648498535,
      21.160125732421875,
      20.75982093811035,
      20.936965942382812,
      20.74408531188965,
      21.95000648498535,
...
```

