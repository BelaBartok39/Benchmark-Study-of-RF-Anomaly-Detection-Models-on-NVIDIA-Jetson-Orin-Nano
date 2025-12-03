# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 1.93 | 1.99 | 1.97 | 2.00 | +1.7% |
| Avg Power (W) | 5.17 | 5.24 | 5.23 | 5.16 | +1.4% |
| Total Energy (J) | 310.28 | 314.08 | 313.78 | 309.35 | +1.4% |
| Energy/Inference (mJ) | 129.281 | 130.867 | 130.743 | 128.897 | +1.4% |
| FPS/Watt | 7.73 | 7.64 | 7.65 | 7.75 | +1.4% |
| Violation Rate (%) | 0.04 | 0.00 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time (15W): 100.0%
- Avg Switch Time: 0.0 ms

## Continuous Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 1.83 | 1.82 | 1.78 | 1.79 | +0.4% |
| Avg Power (W) | 5.26 | 5.26 | 5.32 | 5.26 | +1.1% |
| Total Energy (J) | 320.12 | 320.04 | 323.92 | 320.39 | +1.1% |
| Energy/Inference (mJ) | 53.353 | 53.340 | 53.987 | 53.398 | +1.1% |
| FPS/Watt | 19.02 | 19.02 | 18.80 | 19.00 | +1.1% |
| Violation Rate (%) | 0.02 | 0.00 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time (15W): 100.0%
- Avg Switch Time: 0.0 ms

## Variable Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 1.89 | 1.85 | 1.83 | 1.87 | +2.6% |
| Avg Power (W) | 5.29 | 5.28 | 5.35 | 5.36 | -0.2% |
| Total Energy (J) | 321.83 | 321.37 | 325.93 | 326.48 | -0.2% |
| Energy/Inference (mJ) | 42.910 | 42.849 | 43.457 | 43.531 | -0.2% |
| FPS/Watt | 23.65 | 23.68 | 23.35 | 23.31 | -0.2% |
| Violation Rate (%) | 0.01 | 0.00 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 2
- Low Power Time (15W): 93.4%
- Avg Switch Time: 22.4 ms

## Periodic Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 1.74 | 1.83 | 1.88 | 1.81 | -3.7% |
| Avg Power (W) | 5.19 | 5.14 | 5.16 | 5.21 | -1.1% |
| Total Energy (J) | 259.16 | 256.99 | 257.68 | 260.52 | -1.1% |
| Energy/Inference (mJ) | 143.976 | 142.771 | 143.155 | 144.733 | -1.1% |
| FPS/Watt | 5.78 | 5.83 | 5.82 | 5.75 | -1.1% |
| Violation Rate (%) | 0.06 | 0.00 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time (15W): 100.0%
- Avg Switch Time: 0.0 ms

