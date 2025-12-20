# Adaptive Power Management Summary

Comparison of Static MAXN (performance baseline) vs Adaptive Power Management

## Bursty Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 2.44 | 2.45 | +0.4% |
| Avg Power (W) | 5.07 | 5.06 | +0.1% |
| Total Energy (J) | 9124.14 | 9114.48 | +0.1% |
| Energy/Inference (mJ) | 276.489 | 276.196 | +0.1% |
| FPS/Watt | 3.62 | 3.62 | +0.1% |
| Violation Rate (%) | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 20.7 ms

## Periodic Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 2.34 | 2.33 | -0.5% |
| Avg Power (W) | 5.02 | 5.00 | +0.5% |
| Total Energy (J) | 8693.09 | 8648.01 | +0.5% |
| Energy/Inference (mJ) | 724.424 | 720.668 | +0.5% |
| FPS/Watt | 1.33 | 1.33 | +0.5% |
| Violation Rate (%) | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 23.4 ms

## Continuous Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 2.35 | 2.34 | -0.4% |
| Avg Power (W) | 5.31 | 5.31 | +0.0% |
| Total Energy (J) | 9560.26 | 9561.17 | -0.0% |
| Energy/Inference (mJ) | 53.113 | 53.118 | -0.0% |
| FPS/Watt | 18.83 | 18.83 | +0.0% |
| Violation Rate (%) | 0.00 | 0.00 | +0.0% |

**Adaptive Statistics:**
- Mode Switches: 8
- Low Power Time (15W): 93.3%
- Medium Power Time (25W): 5.0%
- High Power Time (MAXN): 1.7%
- Avg Switch Time: 22.1 ms

## Variable Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 2.35 | 2.34 | -0.3% |
| Avg Power (W) | 5.36 | 5.36 | -0.1% |
| Total Energy (J) | 9646.24 | 9653.05 | -0.1% |
| Energy/Inference (mJ) | 44.453 | 44.484 | -0.1% |
| FPS/Watt | 22.50 | 22.49 | -0.1% |
| Violation Rate (%) | 0.00 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 3
- Low Power Time (15W): 98.3%
- Medium Power Time (25W): 1.7%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 21.0 ms

