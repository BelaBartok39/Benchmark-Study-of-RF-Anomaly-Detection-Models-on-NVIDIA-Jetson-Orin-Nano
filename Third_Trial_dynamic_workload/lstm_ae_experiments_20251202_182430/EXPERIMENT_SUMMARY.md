# Adaptive Power Management Experiment Summary

**Date**: Tue Dec  2 06:55:47 PM CST 2025
**Model**: lstm_ae
**TensorRT**: false
**Model-Specific Defaults**: true

## Experiment Configuration

- **Power Management**: Three-tier adaptive (15W/25W/MAXN)
- **Threshold Mode**: Model-specific (auto-configured)
- **Batch Size**: 8 (batched inference)
- **Channels**: 10 (multi-channel concurrent)
- **Duration per Workload**: 60 seconds
- **Workload Patterns**: bursty continuous variable periodic
- **Test Samples**: 200

## Directory Structure

```
adaptive_experiments_20251202_182430/
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
    "low_to_medium_avg_ms": 21.250462532043457,
    "low_to_medium_median_ms": 20.923614501953125,
    "low_to_medium_std_ms": 0.8845792964519187,
    "low_to_medium_min_ms": 19.97542381286621,
    "low_to_medium_max_ms": 23.538827896118164,
    "low_to_medium_p95_ms": 22.52253293991089,
    "low_to_medium_times_ms": [
      21.929025650024414,
      20.236492156982422,
      20.310163497924805,
      21.69632911682129,
      20.705223083496094,
      22.469043731689453,
      20.60699462890625,
      21.06642723083496,
      22.13764190673828,
      21.210193634033203,
      19.97542381286621,
      20.78080177307129,
      20.66969871520996,
      22.089719772338867,
      20.611286163330078,
      22.066593170166016,
      20.710468292236328,
      20.588159561157227,
      23.538827896118164,
...
```

