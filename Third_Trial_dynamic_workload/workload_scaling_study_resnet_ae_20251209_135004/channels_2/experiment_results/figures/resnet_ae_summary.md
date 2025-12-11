# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 11.98 | 12.12 | 12.08 | 12.11 | +0.2% |
| Avg Power (W) | 5.43 | 5.42 | 5.42 | 5.47 | -0.8% |
| Total Energy (J) | 330.18 | 329.82 | 330.06 | 332.72 | -0.8% |
| Energy/Inference (mJ) | 68.788 | 68.713 | 68.763 | 69.316 | -0.8% |
| FPS/Watt | 14.74 | 14.76 | 14.75 | 14.63 | -0.8% |
| Violation Rate (%) | 0.02 | 0.02 | 0.02 | 0.02 | +0.0% |

**Adaptive Statistics:**
- Mode Switches: 16
- Low Power Time (15W): 53.3%
- Medium Power Time (25W): 38.5%
- High Power Time (MAXN): 8.2%
- Avg Switch Time: 22.0 ms

## Continuous Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 8.50 | 8.58 | 8.52 | 10.15 | +19.1% |
| Avg Power (W) | 5.98 | 5.99 | 5.99 | 6.05 | -1.0% |
| Total Energy (J) | 364.09 | 364.38 | 364.77 | 368.32 | -1.0% |
| Energy/Inference (mJ) | 30.341 | 30.365 | 30.398 | 30.693 | -1.0% |
| FPS/Watt | 33.43 | 33.40 | 33.36 | 33.04 | -1.0% |
| Violation Rate (%) | 0.01 | 0.00 | 0.00 | 0.01 | N/A |

**Adaptive Statistics:**
- Mode Switches: 7
- Low Power Time (15W): 84.7%
- Medium Power Time (25W): 10.3%
- High Power Time (MAXN): 5.0%
- Avg Switch Time: 22.0 ms

## Variable Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 8.16 | 7.68 | 11.64 | 7.37 | -36.6% |
| Avg Power (W) | 6.31 | 6.17 | 6.49 | 6.36 | +2.0% |
| Total Energy (J) | 383.41 | 386.38 | 394.55 | 386.67 | +2.0% |
| Energy/Inference (mJ) | 25.561 | 25.759 | 26.303 | 25.778 | +2.0% |
| FPS/Watt | 39.65 | 40.53 | 38.52 | 39.30 | +2.0% |
| Violation Rate (%) | 0.00 | 0.00 | 0.00 | 0.01 | N/A |

**Adaptive Statistics:**
- Mode Switches: 5
- Low Power Time (15W): 89.6%
- Medium Power Time (25W): 5.2%
- High Power Time (MAXN): 5.1%
- Avg Switch Time: 21.2 ms

## Periodic Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 8.83 | 8.99 | 8.91 | 10.76 | +20.8% |
| Avg Power (W) | 5.38 | 5.36 | 5.36 | 5.37 | -0.2% |
| Total Energy (J) | 273.53 | 272.63 | 272.62 | 273.30 | -0.3% |
| Energy/Inference (mJ) | 75.981 | 75.731 | 75.727 | 75.917 | -0.3% |
| FPS/Watt | 11.15 | 11.19 | 11.19 | 11.16 | -0.2% |
| Violation Rate (%) | 0.03 | 0.03 | 0.03 | 0.03 | +0.0% |

**Adaptive Statistics:**
- Mode Switches: 7
- Low Power Time (15W): 79.0%
- Medium Power Time (25W): 14.9%
- High Power Time (MAXN): 6.0%
- Avg Switch Time: 22.6 ms

