# Adaptive Power Management Summary

Comparison of Static MAXN (performance baseline) vs Adaptive Power Management

## Bursty Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 1.73 | 1.73 | -0.3% |
| Avg Power (W) | 5.03 | 5.03 | +0.0% |
| Total Energy (J) | 9062.81 | 9059.72 | +0.0% |
| Energy/Inference (mJ) | 274.631 | 274.537 | +0.0% |
| FPS/Watt | 3.64 | 3.64 | +0.0% |
| Violation Rate (%) | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 22.7 ms

## Periodic Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 1.65 | 1.66 | +0.2% |
| Avg Power (W) | 5.02 | 4.99 | +0.6% |
| Total Energy (J) | 8686.83 | 8637.99 | +0.6% |
| Energy/Inference (mJ) | 723.903 | 719.832 | +0.6% |
| FPS/Watt | 1.33 | 1.34 | +0.6% |
| Violation Rate (%) | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 21.3 ms

## Continuous Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 1.62 | 1.62 | +0.3% |
| Avg Power (W) | 5.19 | 5.19 | -0.0% |
| Total Energy (J) | 9340.68 | 9346.34 | -0.1% |
| Energy/Inference (mJ) | 51.893 | 51.924 | -0.1% |
| FPS/Watt | 19.27 | 19.26 | -0.0% |
| Violation Rate (%) | 0.00 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 23.0 ms

## Variable Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 1.66 | 1.67 | +0.5% |
| Avg Power (W) | 5.21 | 5.22 | -0.1% |
| Total Energy (J) | 9387.21 | 9390.08 | -0.0% |
| Energy/Inference (mJ) | 43.259 | 43.272 | -0.0% |
| FPS/Watt | 23.13 | 23.11 | -0.1% |
| Violation Rate (%) | 0.00 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 23.5 ms

