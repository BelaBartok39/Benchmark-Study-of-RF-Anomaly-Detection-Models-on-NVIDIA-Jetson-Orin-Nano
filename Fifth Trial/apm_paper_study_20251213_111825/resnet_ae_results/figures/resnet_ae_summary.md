# Adaptive Power Management Summary

Comparison of Static MAXN (performance baseline) vs Adaptive Power Management

## Bursty Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 10.03 | 10.06 | +0.2% |
| Avg Power (W) | 5.32 | 5.33 | -0.2% |
| Total Energy (J) | 9577.33 | 9596.46 | -0.2% |
| Energy/Inference (mJ) | 290.222 | 290.802 | -0.2% |
| FPS/Watt | 3.45 | 3.44 | -0.2% |
| Violation Rate (%) | 30.15 | 33.44 | +10.9% |

**Adaptive Statistics:**
- Mode Switches: 2
- Low Power Time (15W): 0.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 100.0%
- Avg Switch Time: 21.5 ms

## Periodic Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 8.57 | 8.89 | +3.7% |
| Avg Power (W) | 5.26 | 5.23 | +0.6% |
| Total Energy (J) | 9096.89 | 9044.82 | +0.6% |
| Energy/Inference (mJ) | 758.074 | 753.735 | +0.6% |
| FPS/Watt | 1.27 | 1.28 | +0.6% |
| Violation Rate (%) | 2.57 | 4.21 | +63.4% |

**Adaptive Statistics:**
- Mode Switches: 51
- Low Power Time (15W): 53.1%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 46.9%
- Avg Switch Time: 22.7 ms

## Continuous Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 8.54 | 8.51 | -0.4% |
| Avg Power (W) | 5.72 | 5.73 | -0.1% |
| Total Energy (J) | 10297.42 | 10312.29 | -0.1% |
| Energy/Inference (mJ) | 57.208 | 57.290 | -0.1% |
| FPS/Watt | 17.49 | 17.46 | -0.1% |
| Violation Rate (%) | 2.33 | 2.19 | -6.3% |

**Adaptive Statistics:**
- Mode Switches: 3
- Low Power Time (15W): 0.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 99.9%
- Avg Switch Time: 22.9 ms

## Variable Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 9.30 | 9.25 | -0.5% |
| Avg Power (W) | 5.93 | 5.95 | -0.4% |
| Total Energy (J) | 10676.89 | 10715.67 | -0.4% |
| Energy/Inference (mJ) | 49.202 | 49.381 | -0.4% |
| FPS/Watt | 20.33 | 20.25 | -0.4% |
| Violation Rate (%) | 11.70 | 10.55 | -9.9% |

**Adaptive Statistics:**
- Mode Switches: 5
- Low Power Time (15W): 0.8%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 99.2%
- Avg Switch Time: 21.9 ms

