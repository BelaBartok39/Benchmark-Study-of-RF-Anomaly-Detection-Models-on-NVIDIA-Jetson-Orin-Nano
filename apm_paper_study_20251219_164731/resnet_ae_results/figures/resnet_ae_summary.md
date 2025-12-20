# Adaptive Power Management Summary

Comparison of Static MAXN (performance baseline) vs Adaptive Power Management

## Bursty Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 9.88 | 9.93 | +0.5% |
| Avg Power (W) | 5.07 | 5.17 | -2.1% |
| Total Energy (J) | 9125.71 | 9317.16 | -2.1% |
| Energy/Inference (mJ) | 276.537 | 282.338 | -2.1% |
| FPS/Watt | 3.62 | 3.54 | -2.1% |
| Violation Rate (%) | 49.97 | 47.83 | -4.3% |

**Adaptive Statistics:**
- Mode Switches: 8
- Low Power Time (15W): 0.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 100.0%
- Avg Switch Time: 21.7 ms

## Periodic Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 7.51 | 7.45 | -0.8% |
| Avg Power (W) | 5.15 | 4.99 | +3.1% |
| Total Energy (J) | 8912.60 | 8638.31 | +3.1% |
| Energy/Inference (mJ) | 742.717 | 719.859 | +3.1% |
| FPS/Watt | 1.29 | 1.34 | +3.2% |
| Violation Rate (%) | 1.67 | 1.74 | +4.5% |

**Adaptive Statistics:**
- Mode Switches: 49
- Low Power Time (15W): 53.1%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 46.9%
- Avg Switch Time: 21.8 ms

## Continuous Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 7.55 | 7.50 | -0.7% |
| Avg Power (W) | 5.46 | 5.46 | -0.0% |
| Total Energy (J) | 9825.62 | 9829.65 | -0.0% |
| Energy/Inference (mJ) | 54.587 | 54.609 | -0.0% |
| FPS/Watt | 18.32 | 18.32 | -0.0% |
| Violation Rate (%) | 0.68 | 0.65 | -4.7% |

**Adaptive Statistics:**
- Mode Switches: 3
- Low Power Time (15W): 0.1%
- Medium Power Time (25W): 0.1%
- High Power Time (MAXN): 99.8%
- Avg Switch Time: 21.1 ms

## Variable Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 9.30 | 9.21 | -1.0% |
| Avg Power (W) | 5.69 | 5.69 | -0.1% |
| Total Energy (J) | 10238.14 | 10248.08 | -0.1% |
| Energy/Inference (mJ) | 47.180 | 47.226 | -0.1% |
| FPS/Watt | 21.20 | 21.18 | -0.1% |
| Violation Rate (%) | 11.89 | 11.27 | -5.2% |

**Adaptive Statistics:**
- Mode Switches: 3
- Low Power Time (15W): 0.7%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 99.3%
- Avg Switch Time: 21.8 ms

