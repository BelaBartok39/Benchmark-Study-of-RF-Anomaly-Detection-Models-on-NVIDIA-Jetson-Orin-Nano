# Adaptive Power Management Summary

Comparison of Static MAXN (performance baseline) vs Adaptive Power Management

## Bursty Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 2.47 | 2.49 | +0.8% |
| Avg Power (W) | 5.05 | 5.04 | +0.0% |
| Total Energy (J) | 9084.45 | 9081.71 | +0.0% |
| Energy/Inference (mJ) | 275.286 | 275.203 | +0.0% |
| FPS/Watt | 3.63 | 3.63 | +0.0% |
| Violation Rate (%) | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 21.7 ms

## Periodic Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 2.52 | 2.51 | -0.4% |
| Avg Power (W) | 5.01 | 4.99 | +0.4% |
| Total Energy (J) | 8665.69 | 8629.88 | +0.4% |
| Energy/Inference (mJ) | 722.141 | 719.157 | +0.4% |
| FPS/Watt | 1.33 | 1.34 | +0.4% |
| Violation Rate (%) | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 21.9 ms

## Continuous Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 2.55 | 2.53 | -0.7% |
| Avg Power (W) | 5.31 | 5.31 | -0.0% |
| Total Energy (J) | 9553.59 | 9553.07 | +0.0% |
| Energy/Inference (mJ) | 53.075 | 53.073 | +0.0% |
| FPS/Watt | 18.85 | 18.84 | -0.0% |
| Violation Rate (%) | 0.00 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 5
- Low Power Time (15W): 96.7%
- Medium Power Time (25W): 3.3%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 22.4 ms

## Variable Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 2.49 | 2.51 | +0.8% |
| Avg Power (W) | 5.35 | 5.35 | -0.0% |
| Total Energy (J) | 9638.49 | 9638.43 | +0.0% |
| Energy/Inference (mJ) | 44.417 | 44.417 | +0.0% |
| FPS/Watt | 22.52 | 22.52 | -0.0% |
| Violation Rate (%) | 0.00 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 22.9 ms

