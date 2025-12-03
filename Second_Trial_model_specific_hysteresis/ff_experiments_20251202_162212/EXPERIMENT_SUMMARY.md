# Adaptive Power Management Experiment Summary

**Date**: Tue Dec  2 04:53:22 PM CST 2025
**Model**: ff
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
adaptive_experiments_20251202_162212/
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
    "low_to_medium_avg_ms": 21.782076358795166,
    "low_to_medium_median_ms": 21.825313568115234,
    "low_to_medium_std_ms": 1.1845890510628734,
    "low_to_medium_min_ms": 19.244670867919922,
    "low_to_medium_max_ms": 24.997711181640625,
    "low_to_medium_p95_ms": 23.13182353973389,
    "low_to_medium_times_ms": [
      21.904468536376953,
      21.509885787963867,
      20.884037017822266,
      19.244670867919922,
      20.516633987426758,
      20.563840866088867,
      22.86839485168457,
      22.245168685913086,
      22.209644317626953,
      24.997711181640625,
      23.033618927001953,
      22.200584411621094,
      20.561695098876953,
      21.746158599853516,
      22.245407104492188,
      21.25859260559082,
      21.225929260253906,
      21.326065063476562,
      22.94445037841797,
...
```

