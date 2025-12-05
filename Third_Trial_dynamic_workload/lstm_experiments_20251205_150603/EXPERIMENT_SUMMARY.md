# Adaptive Power Management Experiment Summary

**Date**: Fri Dec  5 03:19:55 PM CST 2025
**Model**: lstm_ae
**TensorRT**: false
**Model-Specific Defaults**: false

## Experiment Configuration

- **Power Management**: Three-tier adaptive (15W/25W/MAXN)
- **Threshold Mode**: Manual (10.0ms threshold, 5.0s hysteresis)
- **Batch Size**: 1 (single-sample)
- **Channels**: 1 (single-channel)
- **Duration per Workload**: 60 seconds
- **Workload Patterns**: bursty continuous variable periodic
- **Test Samples**: 200

## Directory Structure

```
adaptive_experiments_20251205_150603/
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

