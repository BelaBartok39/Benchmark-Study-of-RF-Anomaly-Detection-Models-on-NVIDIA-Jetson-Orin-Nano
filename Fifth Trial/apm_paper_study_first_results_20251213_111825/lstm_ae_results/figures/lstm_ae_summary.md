# Adaptive Power Management Summary

Comparison of Static MAXN (performance baseline) vs Adaptive Power Management

## Bursty Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 17.80 | 17.72 | -0.5% |
| Avg Power (W) | 5.34 | 5.30 | +0.7% |
| Total Energy (J) | 9613.47 | 9546.34 | +0.7% |
| Energy/Inference (mJ) | 291.317 | 289.283 | +0.7% |
| FPS/Watt | 3.43 | 3.46 | +0.7% |
| Violation Rate (%) | 51.41 | 0.01 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 3
- Low Power Time (15W): 0.0%
- Medium Power Time (25W): 65.3%
- High Power Time (MAXN): 34.7%
- Avg Switch Time: 21.7 ms

## Periodic Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 9.18 | 9.41 | +2.5% |
| Avg Power (W) | 5.27 | 5.21 | +1.1% |
| Total Energy (J) | 9110.09 | 9009.96 | +1.1% |
| Energy/Inference (mJ) | 759.174 | 750.830 | +1.1% |
| FPS/Watt | 1.27 | 1.28 | +1.1% |
| Violation Rate (%) | 3.40 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 48
- Low Power Time (15W): 53.1%
- Medium Power Time (25W): 46.9%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 23.0 ms

## Continuous Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 6.91 | 7.79 | +12.7% |
| Avg Power (W) | 5.72 | 5.59 | +2.2% |
| Total Energy (J) | 10295.42 | 10072.30 | +2.2% |
| Energy/Inference (mJ) | 57.197 | 55.957 | +2.2% |
| FPS/Watt | 17.48 | 17.87 | +2.2% |
| Violation Rate (%) | 0.02 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 50
- Low Power Time (15W): 53.4%
- Medium Power Time (25W): 46.6%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 23.7 ms

## Variable Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 10.83 | 10.79 | -0.4% |
| Avg Power (W) | 5.93 | 5.80 | +2.2% |
| Total Energy (J) | 10682.78 | 10448.65 | +2.2% |
| Energy/Inference (mJ) | 49.229 | 48.150 | +2.2% |
| FPS/Watt | 20.31 | 20.77 | +2.2% |
| Violation Rate (%) | 10.88 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 44
- Low Power Time (15W): 28.1%
- Medium Power Time (25W): 71.9%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 24.1 ms

