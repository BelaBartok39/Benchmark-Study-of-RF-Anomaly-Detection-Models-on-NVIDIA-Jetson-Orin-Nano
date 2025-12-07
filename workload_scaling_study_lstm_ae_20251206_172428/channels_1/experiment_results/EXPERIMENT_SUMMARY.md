# Adaptive Power Management Experiment Summary

**Date**: Sat Dec  6 05:56:17 PM CST 2025
**Model**: lstm_ae
**TensorRT**: false
**Model-Specific Defaults**: false

## Experiment Configuration

- **Power Management**: Three-tier adaptive (15W/25W/MAXN) 
- **Threshold Mode**: Auto-calibrated (SLA: 10.0ms)
- **Batch Size**: 1 (single-sample)
- **Channels**: 1 (single-channel)
- **Duration per Workload**: 60 seconds
- **Workload Patterns**: bursty continuous variable periodic
- **Test Samples**: 200

## Directory Structure

```
adaptive_experiments_20251206_172428/
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
    "low_to_medium_avg_ms": 21.06635570526123,
    "low_to_medium_median_ms": 21.007418632507324,
    "low_to_medium_std_ms": 0.583461043424298,
    "low_to_medium_min_ms": 20.198822021484375,
    "low_to_medium_max_ms": 22.377729415893555,
    "low_to_medium_p95_ms": 22.114992141723633,
    "low_to_medium_times_ms": [
      20.470619201660156,
      20.494699478149414,
      22.101163864135742,
      20.49088478088379,
      21.883487701416016,
      21.21567726135254,
      20.198822021484375,
      20.6451416015625,
      21.070241928100586,
      20.60246467590332,
      22.377729415893555,
      21.090984344482422,
      21.116256713867188,
      20.650148391723633,
      21.4993953704834,
      21.500587463378906,
      20.73955535888672,
      20.62201499938965,
      21.61264419555664,
...
```

