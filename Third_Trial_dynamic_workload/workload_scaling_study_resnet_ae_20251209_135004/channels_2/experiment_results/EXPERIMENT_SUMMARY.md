# Adaptive Power Management Experiment Summary

**Date**: Tue Dec  9 02:53:56 PM CST 2025
**Model**: resnet_ae
**TensorRT**: false
**Model-Specific Defaults**: false

## Experiment Configuration

- **Power Management**: Three-tier adaptive (15W/25W/MAXN) + Frequency Scaling
- **Threshold Mode**: Auto-calibrated (SLA: 10.0ms)
- **Batch Size**: 1 (single-sample)
- **Channels**: 2 (multi-channel concurrent)
- **Duration per Workload**: 60 seconds
- **Workload Patterns**: bursty continuous variable periodic
- **Test Samples**: 200

## Directory Structure

```
adaptive_experiments_20251209_142204/
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
    "low_to_medium_avg_ms": 20.92643976211548,
    "low_to_medium_median_ms": 20.671367645263672,
    "low_to_medium_std_ms": 0.7809388103447515,
    "low_to_medium_min_ms": 20.038843154907227,
    "low_to_medium_max_ms": 22.928953170776367,
    "low_to_medium_p95_ms": 22.42748737335205,
    "low_to_medium_times_ms": [
      20.598649978637695,
      20.26820182800293,
      20.26534080505371,
      21.38352394104004,
      21.031856536865234,
      21.10910415649414,
      20.070791244506836,
      21.880388259887695,
      21.55590057373047,
      20.39337158203125,
      20.038843154907227,
      21.548032760620117,
      20.423412322998047,
      20.27273178100586,
      22.928953170776367,
      20.64347267150879,
      20.26057243347168,
      20.699262619018555,
      22.401094436645508,
...
```

