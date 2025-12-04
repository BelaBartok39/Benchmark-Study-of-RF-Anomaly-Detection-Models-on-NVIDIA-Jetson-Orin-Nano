# Adaptive Power Management Experiment Summary

**Date**: Thu Dec  4 04:28:40 PM CST 2025
**Model**: cnn_ae
**TensorRT**: false
**Model-Specific Defaults**: false

## Experiment Configuration

- **Power Management**: Three-tier adaptive (15W/25W/MAXN)
- **Threshold Mode**: Manual (10.0ms threshold, 5.0s hysteresis)
- **Batch Size**: 1 (single-sample)
- **Channels**: 10 (multi-channel concurrent)
- **Duration per Workload**: 60 seconds
- **Workload Patterns**: bursty continuous variable periodic
- **Test Samples**: 200

## Directory Structure

```
adaptive_experiments_20251204_154927/
├── switching_overhead/    # Mode switching characterization results
├── results/               # Raw benchmark results (JSON)
├── figures/              # Visualizations (PNG + summary tables)
└── EXPERIMENT_SUMMARY.md  # This file
```

## Results

### Mode Switching Overhead

See: `switching_overhead/characterization_results.json`

### Workload Benchmarks

- **bursty**: `results/cnn_ae_adaptive_bursty_results.json`
- **continuous**: `results/cnn_ae_adaptive_continuous_results.json`
- **variable**: `results/cnn_ae_adaptive_variable_results.json`
- **periodic**: `results/cnn_ae_adaptive_periodic_results.json`

### Visualizations

- Energy-Latency Trade-off: `figures/cnn_ae_*_energy_latency.png`
- Latency Timeline: `figures/cnn_ae_*_timeline.png`
- Efficiency Comparison: `figures/cnn_ae_efficiency_comparison.png`
- Energy Comparison: `figures/cnn_ae_energy_comparison.png`
- Latency Comparison: `figures/cnn_ae_latency_comparison.png`
- Detailed Summary: `figures/cnn_ae_summary.md`

## Next Steps

1. Review the figures in `figures/` directory
2. Analyze detailed results in `results/` directory
3. Compare with static power mode baselines
4. Identify optimal parameter settings for your workload

## Notes

