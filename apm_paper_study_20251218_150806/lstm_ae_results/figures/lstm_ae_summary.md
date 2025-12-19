# Adaptive Power Management Summary

Comparison of Static MAXN (performance baseline) vs Adaptive Power Management

## Bursty Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 14.34 | 14.30 | -0.3% |
| Avg Power (W) | 5.08 | 5.08 | -0.1% |
| Total Energy (J) | 9146.71 | 9154.24 | -0.1% |
| Energy/Inference (mJ) | 277.173 | 277.401 | -0.1% |
| FPS/Watt | 3.61 | 3.61 | -0.1% |
| Violation Rate (%) | 0.39 | 0.49 | +26.6% |

**Adaptive Statistics:**
- Mode Switches: 2
- Low Power Time (15W): 0.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 100.0%
- Avg Switch Time: 20.5 ms

## Periodic Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 8.85 | 8.86 | +0.1% |
| Avg Power (W) | 5.03 | 5.02 | +0.2% |
| Total Energy (J) | 8700.04 | 8678.81 | +0.2% |
| Energy/Inference (mJ) | 725.004 | 723.234 | +0.2% |
| FPS/Watt | 1.33 | 1.33 | +0.2% |
| Violation Rate (%) | 0.18 | 0.21 | +13.6% |

**Adaptive Statistics:**
- Mode Switches: 48
- Low Power Time (15W): 53.1%
- Medium Power Time (25W): 2.0%
- High Power Time (MAXN): 44.9%
- Avg Switch Time: 21.6 ms

## Continuous Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 6.88 | 6.88 | +0.1% |
| Avg Power (W) | 5.70 | 5.71 | -0.1% |
| Total Energy (J) | 10262.93 | 10276.41 | -0.1% |
| Energy/Inference (mJ) | 57.016 | 57.091 | -0.1% |
| FPS/Watt | 17.54 | 17.52 | -0.1% |
| Violation Rate (%) | 0.42 | 0.57 | +35.4% |

**Adaptive Statistics:**
- Mode Switches: 2
- Low Power Time (15W): 0.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 100.0%
- Avg Switch Time: 19.0 ms

## Variable Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 10.84 | 10.84 | +0.0% |
| Avg Power (W) | 5.99 | 6.01 | -0.3% |
| Total Energy (J) | 10792.97 | 10825.63 | -0.3% |
| Energy/Inference (mJ) | 49.737 | 49.888 | -0.3% |
| FPS/Watt | 20.11 | 20.05 | -0.3% |
| Violation Rate (%) | 11.04 | 10.91 | -1.2% |

**Adaptive Statistics:**
- Mode Switches: 2
- Low Power Time (15W): 0.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 100.0%
- Avg Switch Time: 20.2 ms

