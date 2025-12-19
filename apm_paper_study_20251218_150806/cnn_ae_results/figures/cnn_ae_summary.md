# Adaptive Power Management Summary

Comparison of Static MAXN (performance baseline) vs Adaptive Power Management

## Bursty Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 4.85 | 4.74 | -2.2% |
| Avg Power (W) | 5.08 | 5.07 | +0.1% |
| Total Energy (J) | 9140.69 | 9128.63 | +0.1% |
| Energy/Inference (mJ) | 276.991 | 276.625 | +0.1% |
| FPS/Watt | 3.61 | 3.62 | +0.1% |
| Violation Rate (%) | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 12
- Low Power Time (15W): 90.2%
- Medium Power Time (25W): 9.8%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 21.5 ms

## Periodic Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 4.66 | 4.71 | +1.1% |
| Avg Power (W) | 5.02 | 5.00 | +0.3% |
| Total Energy (J) | 8684.67 | 8655.81 | +0.3% |
| Energy/Inference (mJ) | 723.723 | 721.318 | +0.3% |
| FPS/Watt | 1.33 | 1.33 | +0.3% |
| Violation Rate (%) | 0.00 | 0.01 | N/A |

**Adaptive Statistics:**
- Mode Switches: 7
- Low Power Time (15W): 94.5%
- Medium Power Time (25W): 3.6%
- High Power Time (MAXN): 1.9%
- Avg Switch Time: 22.0 ms

## Continuous Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 4.68 | 4.67 | -0.2% |
| Avg Power (W) | 5.41 | 5.41 | +0.1% |
| Total Energy (J) | 9744.92 | 9739.38 | +0.1% |
| Energy/Inference (mJ) | 54.138 | 54.108 | +0.1% |
| FPS/Watt | 18.47 | 18.49 | +0.1% |
| Violation Rate (%) | 0.01 | 0.00 | -54.5% |

**Adaptive Statistics:**
- Mode Switches: 43
- Low Power Time (15W): 12.5%
- Medium Power Time (25W): 68.4%
- High Power Time (MAXN): 19.1%
- Avg Switch Time: 22.1 ms

## Variable Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 4.64 | 4.57 | -1.5% |
| Avg Power (W) | 5.48 | 5.48 | -0.0% |
| Total Energy (J) | 9862.41 | 9868.08 | -0.1% |
| Energy/Inference (mJ) | 45.449 | 45.475 | -0.1% |
| FPS/Watt | 22.00 | 22.00 | -0.0% |
| Violation Rate (%) | 0.01 | 0.00 | -72.7% |

**Adaptive Statistics:**
- Mode Switches: 37
- Low Power Time (15W): 31.7%
- Medium Power Time (25W): 61.3%
- High Power Time (MAXN): 7.0%
- Avg Switch Time: 21.9 ms

