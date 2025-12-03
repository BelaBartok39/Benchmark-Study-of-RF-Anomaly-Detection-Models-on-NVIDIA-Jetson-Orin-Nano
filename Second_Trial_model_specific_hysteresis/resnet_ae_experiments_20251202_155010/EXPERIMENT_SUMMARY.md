# Adaptive Power Management Experiment Summary

**Date**: Tue Dec  2 04:21:25 PM CST 2025
**Model**: resnet_ae
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
adaptive_experiments_20251202_155010/
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
    "low_to_medium_avg_ms": 22.1055269241333,
    "low_to_medium_median_ms": 21.35765552520752,
    "low_to_medium_std_ms": 2.3566165656398117,
    "low_to_medium_min_ms": 20.37334442138672,
    "low_to_medium_max_ms": 30.950069427490234,
    "low_to_medium_p95_ms": 25.657272338867195,
    "low_to_medium_times_ms": [
      30.950069427490234,
      25.378704071044922,
      21.2252140045166,
      20.711421966552734,
      21.01302146911621,
      20.49994468688965,
      20.402908325195312,
      21.58379554748535,
      20.37334442138672,
      21.96979522705078,
      22.005081176757812,
      21.300077438354492,
      20.664215087890625,
      21.415233612060547,
      20.91693878173828,
      20.809650421142578,
      21.646738052368164,
      23.236513137817383,
      23.153305053710938,
...
```

