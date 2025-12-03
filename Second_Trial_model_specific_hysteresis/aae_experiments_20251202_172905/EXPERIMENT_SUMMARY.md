# Adaptive Power Management Experiment Summary

**Date**: Tue Dec  2 06:00:15 PM CST 2025
**Model**: aae
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
adaptive_experiments_20251202_172905/
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
    "low_to_medium_avg_ms": 22.013163566589355,
    "low_to_medium_median_ms": 21.871089935302734,
    "low_to_medium_std_ms": 1.4050431787641164,
    "low_to_medium_min_ms": 19.719839096069336,
    "low_to_medium_max_ms": 25.8328914642334,
    "low_to_medium_p95_ms": 25.017273426055908,
    "low_to_medium_times_ms": [
      23.728609085083008,
      22.47786521911621,
      20.766496658325195,
      21.004676818847656,
      22.17578887939453,
      21.129846572875977,
      21.858692169189453,
      21.0878849029541,
      22.403955459594727,
      22.411346435546875,
      21.897554397583008,
      21.045446395874023,
      21.78359031677246,
      21.998882293701172,
      20.704030990600586,
      19.719839096069336,
      25.8328914642334,
      21.378040313720703,
      24.974346160888672,
...
```

