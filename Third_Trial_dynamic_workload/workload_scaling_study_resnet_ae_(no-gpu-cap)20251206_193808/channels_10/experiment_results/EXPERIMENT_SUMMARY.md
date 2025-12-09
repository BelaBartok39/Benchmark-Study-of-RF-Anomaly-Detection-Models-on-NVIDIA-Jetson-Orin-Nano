# Adaptive Power Management Experiment Summary

**Date**: Sat Dec  6 11:10:08 PM CST 2025
**Model**: resnet_ae
**TensorRT**: false
**Model-Specific Defaults**: false

## Experiment Configuration

- **Power Management**: Three-tier adaptive (15W/25W/MAXN) 
- **Threshold Mode**: Auto-calibrated (SLA: 10.0ms)
- **Batch Size**: 1 (single-sample)
- **Channels**: 10 (multi-channel concurrent)
- **Duration per Workload**: 60 seconds
- **Workload Patterns**: bursty continuous variable periodic
- **Test Samples**: 200

## Directory Structure

```
adaptive_experiments_20251206_220001/
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
    "low_to_medium_avg_ms": 20.28219699859619,
    "low_to_medium_median_ms": 20.174503326416016,
    "low_to_medium_std_ms": 0.5637994047216326,
    "low_to_medium_min_ms": 19.545316696166992,
    "low_to_medium_max_ms": 21.402359008789062,
    "low_to_medium_p95_ms": 21.310627460479736,
    "low_to_medium_times_ms": [
      21.402359008789062,
      20.75052261352539,
      20.23911476135254,
      19.8974609375,
      19.93703842163086,
      19.65165138244629,
      19.655466079711914,
      19.66571807861328,
      20.305633544921875,
      21.30579948425293,
      21.104097366333008,
      20.845651626586914,
      20.561933517456055,
      19.580602645874023,
      19.545316696166992,
      20.171403884887695,
      20.177602767944336,
      20.12491226196289,
      20.699024200439453,
...
```

