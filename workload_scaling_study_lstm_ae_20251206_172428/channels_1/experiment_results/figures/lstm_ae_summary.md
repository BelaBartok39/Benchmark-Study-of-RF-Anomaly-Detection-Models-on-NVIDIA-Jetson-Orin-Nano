# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 16.16 | 14.28 | 14.13 | 14.11 | -0.2% |
| Avg Power (W) | 5.35 | 5.31 | 5.25 | 5.27 | -0.4% |
| Total Energy (J) | 320.78 | 318.66 | 314.61 | 315.81 | -0.4% |
| Energy/Inference (mJ) | 133.660 | 132.773 | 131.088 | 131.588 | -0.4% |
| FPS/Watt | 7.48 | 7.53 | 7.62 | 7.59 | -0.4% |
| Violation Rate (%) | 18.67 | 16.96 | 17.17 | 17.12 | -0.2% |

**Adaptive Statistics:**
- Mode Switches: 8
- Low Power Time (15W): 0.1%
- Medium Power Time (25W): 10.5%
- High Power Time (MAXN): 89.4%
- Avg Switch Time: 22.2 ms

## Continuous Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 7.63 | 7.57 | 7.60 | 7.66 | +0.8% |
| Avg Power (W) | 5.53 | 5.66 | 5.66 | 5.57 | +1.6% |
| Total Energy (J) | 336.41 | 344.43 | 344.76 | 339.16 | +1.6% |
| Energy/Inference (mJ) | 56.068 | 57.405 | 57.459 | 56.527 | +1.6% |
| FPS/Watt | 18.09 | 17.67 | 17.65 | 17.95 | +1.7% |
| Violation Rate (%) | 0.00 | 0.03 | 0.00 | 0.02 | N/A |

**Adaptive Statistics:**
- Mode Switches: 10
- Low Power Time (15W): 71.1%
- Medium Power Time (25W): 23.6%
- High Power Time (MAXN): 5.3%
- Avg Switch Time: 23.0 ms

## Variable Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 7.63 | 10.82 | 10.84 | 10.84 | +0.0% |
| Avg Power (W) | 5.60 | 5.92 | 6.01 | 6.12 | -1.9% |
| Total Energy (J) | 351.03 | 359.94 | 365.71 | 372.86 | -2.0% |
| Energy/Inference (mJ) | 46.804 | 47.992 | 48.762 | 49.714 | -2.0% |
| FPS/Watt | 22.32 | 21.13 | 20.81 | 20.41 | -1.9% |
| Violation Rate (%) | 0.91 | 12.61 | 15.24 | 13.55 | -11.1% |

**Adaptive Statistics:**
- Mode Switches: 12
- Low Power Time (15W): 0.3%
- Medium Power Time (25W): 22.1%
- High Power Time (MAXN): 77.6%
- Avg Switch Time: 28.9 ms

## Periodic Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 8.87 | 8.94 | 9.40 | 10.82 | +15.1% |
| Avg Power (W) | 5.29 | 5.28 | 5.39 | 5.25 | +2.6% |
| Total Energy (J) | 264.37 | 263.72 | 269.14 | 262.29 | +2.5% |
| Energy/Inference (mJ) | 146.874 | 146.511 | 149.525 | 145.715 | +2.5% |
| FPS/Watt | 5.67 | 5.68 | 5.57 | 5.71 | +2.6% |
| Violation Rate (%) | 0.00 | 0.00 | 0.39 | 0.06 | -85.7% |

**Adaptive Statistics:**
- Mode Switches: 10
- Low Power Time (15W): 71.5%
- Medium Power Time (25W): 22.4%
- High Power Time (MAXN): 6.1%
- Avg Switch Time: 23.2 ms

