# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 11.94 | 12.10 | 12.01 | 12.03 | +0.1% |
| Avg Power (W) | 5.02 | 5.01 | 5.01 | 5.02 | -0.2% |
| Total Energy (J) | 305.15 | 304.77 | 304.96 | 305.54 | -0.2% |
| Energy/Inference (mJ) | 63.573 | 63.493 | 63.533 | 63.655 | -0.2% |
| FPS/Watt | 15.95 | 15.97 | 15.96 | 15.93 | -0.2% |
| Violation Rate (%) | 0.02 | 0.00 | 0.00 | 0.02 | N/A |

**Adaptive Statistics:**
- Mode Switches: 14
- Low Power Time (15W): 62.1%
- Medium Power Time (25W): 33.0%
- High Power Time (MAXN): 5.0%
- Avg Switch Time: 21.9 ms

## Continuous Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 8.52 | 8.51 | 8.51 | 8.56 | +0.5% |
| Avg Power (W) | 5.64 | 5.63 | 5.65 | 5.65 | +0.0% |
| Total Energy (J) | 343.15 | 342.26 | 343.71 | 343.72 | -0.0% |
| Energy/Inference (mJ) | 28.596 | 28.522 | 28.642 | 28.643 | -0.0% |
| FPS/Watt | 35.46 | 35.55 | 35.40 | 35.40 | +0.0% |
| Violation Rate (%) | 0.01 | 0.00 | 0.01 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 3
- Low Power Time (15W): 94.9%
- Medium Power Time (25W): 5.1%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 20.6 ms

## Variable Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 7.40 | 7.68 | 11.75 | 7.45 | -36.6% |
| Avg Power (W) | 5.95 | 5.81 | 6.16 | 5.96 | +3.3% |
| Total Energy (J) | 361.65 | 369.10 | 374.38 | 362.10 | +3.3% |
| Energy/Inference (mJ) | 24.110 | 24.606 | 24.958 | 24.140 | +3.3% |
| FPS/Watt | 42.02 | 43.03 | 40.59 | 41.97 | +3.4% |
| Violation Rate (%) | 0.00 | 0.01 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 6
- Low Power Time (15W): 87.4%
- Medium Power Time (25W): 12.6%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 21.3 ms

## Periodic Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 8.75 | 8.65 | 8.87 | 9.26 | +4.5% |
| Avg Power (W) | 4.95 | 4.94 | 4.97 | 4.92 | +1.0% |
| Total Energy (J) | 251.81 | 251.30 | 252.50 | 249.99 | +1.0% |
| Energy/Inference (mJ) | 69.948 | 69.806 | 70.139 | 69.442 | +1.0% |
| FPS/Watt | 12.11 | 12.14 | 12.08 | 12.20 | +1.0% |
| Violation Rate (%) | 0.00 | 0.03 | 0.03 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 3
- Low Power Time (15W): 93.9%
- Medium Power Time (25W): 6.1%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 21.0 ms

