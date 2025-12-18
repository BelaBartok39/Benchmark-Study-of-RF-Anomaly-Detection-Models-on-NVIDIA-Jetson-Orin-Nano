# Adaptive Power Management Summary

Comparison of Static MAXN (performance baseline) vs Adaptive Power Management

## Bursty Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 2.58 | 2.58 | -0.2% |
| Avg Power (W) | 5.04 | 5.04 | +0.0% |
| Total Energy (J) | 9078.49 | 9077.33 | +0.0% |
| Energy/Inference (mJ) | 275.106 | 275.071 | +0.0% |
| FPS/Watt | 3.64 | 3.64 | +0.0% |
| Violation Rate (%) | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 21.5 ms

## Periodic Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 2.53 | 2.52 | -0.4% |
| Avg Power (W) | 5.00 | 4.98 | +0.5% |
| Total Energy (J) | 8654.17 | 8607.73 | +0.5% |
| Energy/Inference (mJ) | 721.181 | 717.311 | +0.5% |
| FPS/Watt | 1.33 | 1.34 | +0.5% |
| Violation Rate (%) | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 21.7 ms

## Continuous Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 2.52 | 2.52 | -0.1% |
| Avg Power (W) | 5.31 | 5.31 | -0.1% |
| Total Energy (J) | 9552.18 | 9563.00 | -0.1% |
| Energy/Inference (mJ) | 53.068 | 53.128 | -0.1% |
| FPS/Watt | 18.84 | 18.82 | -0.1% |
| Violation Rate (%) | 0.00 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 22.3 ms

## Variable Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 2.53 | 2.50 | -1.3% |
| Avg Power (W) | 5.36 | 5.36 | -0.1% |
| Total Energy (J) | 9648.67 | 9655.83 | -0.1% |
| Energy/Inference (mJ) | 44.464 | 44.497 | -0.1% |
| FPS/Watt | 22.50 | 22.48 | -0.1% |
| Violation Rate (%) | 0.00 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 21.2 ms

