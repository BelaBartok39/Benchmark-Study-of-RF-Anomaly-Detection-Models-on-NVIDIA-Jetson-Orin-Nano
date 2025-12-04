# Adaptive Power Management Experiment Summary

**Date**: Thu Dec  4 12:38:34 PM CST 2025
**Model**: lstm_ae
**TensorRT**: false
**Model-Specific Defaults**: true

## Experiment Configuration

- **Power Management**: Three-tier adaptive (15W/25W/MAXN)
- **Threshold Mode**: Model-specific (auto-configured)
- **Batch Size**: 1 (single-sample)
- **Channels**: 10 (multi-channel concurrent)
- **Duration per Workload**: 60 seconds
- **Workload Patterns**: bursty continuous variable periodic
- **Test Samples**: 200

## Directory Structure

```
adaptive_experiments_20251204_112345/
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
    "low_to_medium_avg_ms": 21.937549114227295,
    "low_to_medium_median_ms": 21.741628646850586,
    "low_to_medium_std_ms": 1.006570986953703,
    "low_to_medium_min_ms": 20.569801330566406,
    "low_to_medium_max_ms": 23.888111114501953,
    "low_to_medium_p95_ms": 23.472940921783447,
    "low_to_medium_times_ms": [
      21.197795867919922,
      23.051738739013672,
      21.089792251586914,
      21.716833114624023,
      21.070241928100586,
      21.76642417907715,
      22.206544876098633,
      22.794485092163086,
      22.09615707397461,
      23.45108985900879,
      20.72596549987793,
      20.92432975769043,
      20.569801330566406,
      23.888111114501953,
      21.140575408935547,
      20.967721939086914,
      23.307323455810547,
      22.57704734802246,
      21.178007125854492,
...
```

