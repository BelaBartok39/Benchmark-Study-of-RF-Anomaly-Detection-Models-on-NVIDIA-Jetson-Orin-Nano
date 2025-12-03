# Adaptive Power Management Experiment Summary

**Date**: Tue Dec  2 03:49:14 PM CST 2025
**Model**: cnn_ae
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
adaptive_experiments_20251202_151753/
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


### Switching Overhead Summary

```json
{
  "switching_overhead": {
    "num_trials": 20,
    "stabilization_time_s": 2.0,
    "low_to_medium_avg_ms": 22.492480278015137,
    "low_to_medium_median_ms": 22.14527130126953,
    "low_to_medium_std_ms": 2.5559781047878833,
    "low_to_medium_min_ms": 20.214080810546875,
    "low_to_medium_max_ms": 29.918432235717773,
    "low_to_medium_p95_ms": 28.599989414215088,
    "low_to_medium_times_ms": [
      22.79829978942871,
      22.179841995239258,
      20.406723022460938,
      23.609399795532227,
      20.945310592651367,
      22.80116081237793,
      20.397186279296875,
      22.207260131835938,
      22.624731063842773,
      20.214080810546875,
      29.918432235717773,
      20.240306854248047,
      23.00119400024414,
      20.569562911987305,
      20.642995834350586,
      21.10910415649414,
      20.97773551940918,
      24.56498146057129,
      28.530597686767578,
...
```

