# Adaptive Power Management Summary

Comparison of Static MAXN (performance baseline) vs Adaptive Power Management

## Bursty Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 1.77 | 1.71 | -3.6% |
| Avg Power (W) | 5.03 | 5.03 | -0.1% |
| Total Energy (J) | 9053.60 | 9058.60 | -0.1% |
| Energy/Inference (mJ) | 274.351 | 274.503 | -0.1% |
| FPS/Watt | 3.65 | 3.64 | -0.1% |
| Violation Rate (%) | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 20.3 ms

## Periodic Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 1.65 | 1.62 | -1.5% |
| Avg Power (W) | 5.00 | 4.98 | +0.3% |
| Total Energy (J) | 8646.14 | 8618.66 | +0.3% |
| Energy/Inference (mJ) | 720.512 | 718.222 | +0.3% |
| FPS/Watt | 1.33 | 1.34 | +0.3% |
| Violation Rate (%) | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 21.2 ms

## Continuous Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 1.60 | 1.61 | +0.2% |
| Avg Power (W) | 5.18 | 5.18 | -0.0% |
| Total Energy (J) | 9316.79 | 9320.77 | -0.0% |
| Energy/Inference (mJ) | 51.760 | 51.782 | -0.0% |
| FPS/Watt | 19.32 | 19.31 | -0.0% |
| Violation Rate (%) | 0.00 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 21.9 ms

## Variable Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 1.61 | 1.59 | -1.4% |
| Avg Power (W) | 5.20 | 5.20 | -0.1% |
| Total Energy (J) | 9358.49 | 9363.63 | -0.1% |
| Energy/Inference (mJ) | 43.127 | 43.150 | -0.1% |
| FPS/Watt | 23.20 | 23.18 | -0.1% |
| Violation Rate (%) | 0.00 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 20.6 ms

