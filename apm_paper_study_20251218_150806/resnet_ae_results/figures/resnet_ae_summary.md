# Adaptive Power Management Summary

Comparison of Static MAXN (performance baseline) vs Adaptive Power Management

## Bursty Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 9.64 | 9.48 | -1.7% |
| Avg Power (W) | 5.08 | 5.08 | +0.0% |
| Total Energy (J) | 9149.82 | 9147.02 | +0.0% |
| Energy/Inference (mJ) | 277.267 | 277.183 | +0.0% |
| FPS/Watt | 3.61 | 3.61 | +0.0% |
| Violation Rate (%) | 49.16 | 48.58 | -1.2% |

**Adaptive Statistics:**
- Mode Switches: 4
- Low Power Time (15W): 0.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 100.0%
- Avg Switch Time: 22.8 ms

## Periodic Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 7.62 | 7.53 | -1.1% |
| Avg Power (W) | 5.01 | 5.01 | +0.1% |
| Total Energy (J) | 8678.34 | 8664.73 | +0.2% |
| Energy/Inference (mJ) | 723.195 | 722.061 | +0.2% |
| FPS/Watt | 1.33 | 1.33 | +0.1% |
| Violation Rate (%) | 1.94 | 1.94 | +0.0% |

**Adaptive Statistics:**
- Mode Switches: 49
- Low Power Time (15W): 53.1%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 46.9%
- Avg Switch Time: 21.5 ms

## Continuous Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 7.53 | 7.47 | -0.8% |
| Avg Power (W) | 5.79 | 5.49 | +5.2% |
| Total Energy (J) | 10424.82 | 9885.20 | +5.2% |
| Energy/Inference (mJ) | 57.916 | 54.918 | +5.2% |
| FPS/Watt | 17.27 | 18.21 | +5.5% |
| Violation Rate (%) | 0.54 | 0.62 | +13.9% |

**Adaptive Statistics:**
- Mode Switches: 3
- Low Power Time (15W): 0.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 100.0%
- Avg Switch Time: 20.5 ms

## Variable Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 9.51 | 9.35 | -1.6% |
| Avg Power (W) | 5.72 | 5.72 | -0.1% |
| Total Energy (J) | 10296.39 | 10302.34 | -0.1% |
| Energy/Inference (mJ) | 47.449 | 47.476 | -0.1% |
| FPS/Watt | 21.09 | 21.07 | -0.1% |
| Violation Rate (%) | 11.82 | 11.88 | +0.5% |

**Adaptive Statistics:**
- Mode Switches: 3
- Low Power Time (15W): 0.7%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 99.2%
- Avg Switch Time: 21.4 ms

