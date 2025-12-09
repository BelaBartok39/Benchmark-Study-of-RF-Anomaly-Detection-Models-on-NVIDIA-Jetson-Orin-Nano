# Adaptive Power Management Experiment Summary

**Date**: Sat Dec  6 09:14:29 PM CST 2025
**Model**: resnet_ae
**TensorRT**: false
**Model-Specific Defaults**: false

## Experiment Configuration

- **Power Management**: Three-tier adaptive (15W/25W/MAXN) 
- **Threshold Mode**: Auto-calibrated (SLA: 10.0ms)
- **Batch Size**: 1 (single-sample)
- **Channels**: 3 (multi-channel concurrent)
- **Duration per Workload**: 60 seconds
- **Workload Patterns**: bursty continuous variable periodic
- **Test Samples**: 200

## Directory Structure

```
adaptive_experiments_20251206_204204/
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
    "low_to_medium_avg_ms": 20.41841745376587,
    "low_to_medium_median_ms": 20.33686637878418,
    "low_to_medium_std_ms": 0.5489415838870556,
    "low_to_medium_min_ms": 19.579648971557617,
    "low_to_medium_max_ms": 21.570682525634766,
    "low_to_medium_p95_ms": 21.393561363220215,
    "low_to_medium_times_ms": [
      20.707130432128906,
      19.716739654541016,
      19.96922492980957,
      20.397424697875977,
      21.570682525634766,
      20.212650299072266,
      20.886898040771484,
      20.795106887817383,
      20.276308059692383,
      19.762277603149414,
      20.56288719177246,
      19.97828483581543,
      21.047353744506836,
      21.384239196777344,
      20.649433135986328,
      20.871877670288086,
      19.873857498168945,
      19.984960556030273,
      19.579648971557617,
...
```

