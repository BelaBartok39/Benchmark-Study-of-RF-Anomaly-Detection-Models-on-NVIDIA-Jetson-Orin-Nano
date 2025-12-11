# Adaptive Power Management Experiment Summary

**Date**: Tue Dec  9 04:12:16 PM CST 2025
**Model**: resnet_ae
**TensorRT**: false
**Model-Specific Defaults**: false

## Experiment Configuration

- **Power Management**: Three-tier adaptive (15W/25W/MAXN) + Frequency Scaling
- **Threshold Mode**: Auto-calibrated (SLA: 10.0ms)
- **Batch Size**: 1 (single-sample)
- **Channels**: 5 (multi-channel concurrent)
- **Duration per Workload**: 60 seconds
- **Workload Patterns**: bursty continuous variable periodic
- **Test Samples**: 200

## Directory Structure

```
adaptive_experiments_20251209_152655/
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
    "low_to_medium_avg_ms": 20.88465690612793,
    "low_to_medium_median_ms": 20.74897289276123,
    "low_to_medium_std_ms": 0.9115422379627444,
    "low_to_medium_min_ms": 19.908666610717773,
    "low_to_medium_max_ms": 24.226903915405273,
    "low_to_medium_p95_ms": 21.911871433258057,
    "low_to_medium_times_ms": [
      20.439863204956055,
      21.018505096435547,
      21.147727966308594,
      21.007776260375977,
      21.09551429748535,
      20.393848419189453,
      19.908666610717773,
      21.790027618408203,
      20.446062088012695,
      20.624876022338867,
      20.952463150024414,
      21.35491371154785,
      19.969463348388672,
      20.031452178955078,
      20.873069763183594,
      21.324634552001953,
      20.2329158782959,
      20.456790924072266,
      24.226903915405273,
...
```

