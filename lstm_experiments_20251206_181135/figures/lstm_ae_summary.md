# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 51.82 | 42.91 | 40.84 | 137.27 | +236.1% |
| Avg Power (W) | 5.77 | 6.55 | 6.89 | 6.06 | +12.1% |
| Total Energy (J) | 949.95 | 732.88 | 708.07 | 1145.55 | -61.8% |
| Energy/Inference (mJ) | 39.581 | 30.537 | 29.503 | 47.731 | -61.8% |
| FPS/Watt | 69.27 | 61.03 | 58.04 | 66.01 | +13.7% |
| Violation Rate (%) | 0.00 | 0.00 | 0.00 | 24.59 | N/A |

**Adaptive Statistics:**
- Mode Switches: 3
- Low Power Time (15W): 0.4%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 99.6%
- Avg Switch Time: 22.5 ms

## Continuous Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 58.72 | 45.38 | 41.71 | 135.82 | +225.7% |
| Avg Power (W) | 5.78 | 6.56 | 6.92 | 6.06 | +12.3% |
| Total Energy (J) | 2376.96 | 1842.12 | 1779.97 | 2822.95 | -58.6% |
| Energy/Inference (mJ) | 39.616 | 30.702 | 29.666 | 47.049 | -58.6% |
| FPS/Watt | 172.87 | 152.51 | 144.59 | 164.91 | +14.0% |
| Violation Rate (%) | 0.14 | 0.00 | 0.00 | 26.61 | N/A |

**Adaptive Statistics:**
- Mode Switches: 3
- Low Power Time (15W): 0.1%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 99.8%
- Avg Switch Time: 23.4 ms

## Variable Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 59.34 | 45.60 | 41.93 | 139.04 | +231.6% |
| Avg Power (W) | 5.77 | 6.55 | 6.90 | 6.08 | +11.9% |
| Total Energy (J) | 2953.95 | 2305.14 | 2216.55 | 3600.86 | -62.5% |
| Energy/Inference (mJ) | 39.386 | 30.735 | 29.554 | 48.012 | -62.5% |
| FPS/Watt | 216.59 | 190.85 | 181.05 | 205.47 | +13.5% |
| Violation Rate (%) | 0.07 | 0.00 | 0.00 | 25.67 | N/A |

**Adaptive Statistics:**
- Mode Switches: 3
- Low Power Time (15W): 0.1%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 99.9%
- Avg Switch Time: 24.5 ms

## Periodic Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 52.61 | 44.58 | 42.34 | 135.52 | +220.1% |
| Avg Power (W) | 5.76 | 6.54 | 6.88 | 6.03 | +12.4% |
| Total Energy (J) | 711.81 | 552.55 | 544.15 | 849.20 | -56.1% |
| Energy/Inference (mJ) | 39.545 | 30.697 | 30.230 | 47.178 | -56.1% |
| FPS/Watt | 52.10 | 45.90 | 43.62 | 49.78 | +14.1% |
| Violation Rate (%) | 0.00 | 0.00 | 0.00 | 25.53 | N/A |

**Adaptive Statistics:**
- Mode Switches: 3
- Low Power Time (15W): 0.5%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 99.5%
- Avg Switch Time: 24.4 ms

