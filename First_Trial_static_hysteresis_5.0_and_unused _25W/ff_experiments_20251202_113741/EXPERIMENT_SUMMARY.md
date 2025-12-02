# Adaptive Power Management Experiment Summary

**Date**: Tue Dec  2 11:58:39 AM CST 2025
**Model**: ff
**TensorRT**: false

## Experiment Configuration

- **Latency Threshold**: 10.0 ms
- **Hysteresis Time**: 5.0 seconds
- **Duration per Workload**: 60 seconds
- **Workload Patterns**: bursty continuous variable periodic
- **Test Samples**: 1000

## Directory Structure

```
adaptive_experiments_20251202_113741/
├── switching_overhead/    # Mode switching characterization results
├── results/               # Raw benchmark results (JSON)
├── figures/              # Visualizations (PNG + summary tables)
└── EXPERIMENT_SUMMARY.md  # This file
```

## Results

### Mode Switching Overhead

See: `switching_overhead/characterization_results.json`

### Workload Benchmarks

- **bursty**: `results/ff_adaptive_bursty_results.json`
- **continuous**: `results/ff_adaptive_continuous_results.json`
- **variable**: `results/ff_adaptive_variable_results.json`
- **periodic**: `results/ff_adaptive_periodic_results.json`

### Visualizations

- Energy-Latency Trade-off: `figures/ff_*_energy_latency.png`
- Latency Timeline: `figures/ff_*_timeline.png`
- Efficiency Comparison: `figures/ff_efficiency_comparison.png`
- Energy Comparison: `figures/ff_energy_comparison.png`
- Latency Comparison: `figures/ff_latency_comparison.png`
- Detailed Summary: `figures/ff_summary.md`

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
    "low_to_high_avg_ms": 21.572434902191162,
    "low_to_high_median_ms": 20.95496654510498,
    "low_to_high_std_ms": 1.4802340518574184,
    "low_to_high_min_ms": 20.0803279876709,
    "low_to_high_max_ms": 24.852514266967773,
    "low_to_high_p95_ms": 24.400877952575684,
    "high_to_low_avg_ms": 23.363304138183594,
    "high_to_low_median_ms": 23.453116416931152,
    "high_to_low_std_ms": 1.4036918002308267,
    "high_to_low_min_ms": 19.91128921508789,
    "high_to_low_max_ms": 25.88033676147461,
    "high_to_low_p95_ms": 25.436627864837646,
    "avg_switch_time_ms": 22.46786952018738,
    "low_to_high_times_ms": [
      21.14391326904297,
      21.326303482055664,
      23.41485023498535,
      24.852514266967773,
      20.136117935180664,
      20.0803279876709,
      20.592451095581055,
      21.04926109313965,
      23.735761642456055,
      20.642757415771484,
      20.213603973388672,
      22.858381271362305,
...
```

