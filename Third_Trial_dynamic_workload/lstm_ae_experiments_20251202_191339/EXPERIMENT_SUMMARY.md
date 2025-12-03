# Adaptive Power Management Experiment Summary

**Date**: Tue Dec  2 07:45:40 PM CST 2025
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
adaptive_experiments_20251202_191339/
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
    "low_to_medium_avg_ms": 21.309077739715576,
    "low_to_medium_median_ms": 21.009206771850586,
    "low_to_medium_std_ms": 0.7832561613947605,
    "low_to_medium_min_ms": 20.529985427856445,
    "low_to_medium_max_ms": 23.754358291625977,
    "low_to_medium_p95_ms": 22.330141067504886,
    "low_to_medium_times_ms": [
      21.030187606811523,
      21.701335906982422,
      20.886898040771484,
      22.211074829101562,
      20.71237564086914,
      21.329641342163086,
      20.529985427856445,
      21.533489227294922,
      20.557165145874023,
      20.877361297607422,
      21.52109146118164,
      20.531654357910156,
      21.547555923461914,
      23.754358291625977,
      20.98822593688965,
      22.25518226623535,
      20.596742630004883,
      22.045373916625977,
      20.79486846923828,
...
```

