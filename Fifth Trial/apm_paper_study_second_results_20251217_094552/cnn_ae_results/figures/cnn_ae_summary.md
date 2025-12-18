# Adaptive Power Management Summary

Comparison of Static MAXN (performance baseline) vs Adaptive Power Management

## Bursty Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 4.81 | 4.83 | +0.4% |
| Avg Power (W) | 5.06 | 5.06 | +0.0% |
| Total Energy (J) | 9113.91 | 9112.13 | +0.0% |
| Energy/Inference (mJ) | 276.179 | 276.125 | +0.0% |
| FPS/Watt | 3.62 | 3.62 | +0.0% |
| Violation Rate (%) | 0.00 | 0.01 | N/A |

**Adaptive Statistics:**
- Mode Switches: 18
- Low Power Time (15W): 84.3%
- Medium Power Time (25W): 12.4%
- High Power Time (MAXN): 3.3%
- Avg Switch Time: 22.0 ms

## Periodic Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 4.73 | 4.69 | -0.8% |
| Avg Power (W) | 5.01 | 4.99 | +0.4% |
| Total Energy (J) | 8669.02 | 8633.60 | +0.4% |
| Energy/Inference (mJ) | 722.419 | 719.467 | +0.4% |
| FPS/Watt | 1.33 | 1.34 | +0.4% |
| Violation Rate (%) | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 3
- Low Power Time (15W): 98.2%
- Medium Power Time (25W): 1.8%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 21.6 ms

## Continuous Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 4.62 | 4.63 | +0.2% |
| Avg Power (W) | 5.41 | 5.41 | -0.0% |
| Total Energy (J) | 9740.89 | 9741.78 | -0.0% |
| Energy/Inference (mJ) | 54.116 | 54.121 | -0.0% |
| FPS/Watt | 18.48 | 18.48 | -0.0% |
| Violation Rate (%) | 0.00 | 0.01 | +42.9% |

**Adaptive Statistics:**
- Mode Switches: 38
- Low Power Time (15W): 7.5%
- Medium Power Time (25W): 54.9%
- High Power Time (MAXN): 37.6%
- Avg Switch Time: 22.4 ms

## Variable Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 4.72 | 4.65 | -1.5% |
| Avg Power (W) | 5.48 | 5.48 | -0.0% |
| Total Energy (J) | 9863.40 | 9868.43 | -0.1% |
| Energy/Inference (mJ) | 45.453 | 45.477 | -0.1% |
| FPS/Watt | 22.00 | 22.00 | -0.0% |
| Violation Rate (%) | 0.00 | 0.00 | +33.3% |

**Adaptive Statistics:**
- Mode Switches: 43
- Low Power Time (15W): 15.4%
- Medium Power Time (25W): 67.0%
- High Power Time (MAXN): 17.7%
- Avg Switch Time: 21.9 ms

