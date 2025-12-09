# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 9.69 | 9.69 | 9.66 | 9.62 | -0.4% |
| Avg Power (W) | 5.21 | 5.19 | 5.20 | 5.19 | +0.1% |
| Total Energy (J) | 311.97 | 311.05 | 311.39 | 310.98 | +0.1% |
| Energy/Inference (mJ) | 129.986 | 129.604 | 129.745 | 129.575 | +0.1% |
| FPS/Watt | 7.68 | 7.71 | 7.70 | 7.71 | +0.1% |
| Violation Rate (%) | 17.33 | 17.92 | 18.71 | 18.17 | -2.9% |

**Adaptive Statistics:**
- Mode Switches: 2
- Low Power Time (15W): 0.1%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 99.9%
- Avg Switch Time: 21.3 ms

## Continuous Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 7.54 | 7.55 | 7.55 | 7.54 | -0.1% |
| Avg Power (W) | 5.10 | 5.09 | 5.09 | 5.10 | -0.1% |
| Total Energy (J) | 310.11 | 309.71 | 309.85 | 310.17 | -0.1% |
| Energy/Inference (mJ) | 51.685 | 51.619 | 51.642 | 51.696 | -0.1% |
| FPS/Watt | 19.62 | 19.64 | 19.64 | 19.62 | -0.1% |
| Violation Rate (%) | 0.33 | 0.42 | 0.43 | 0.47 | +7.7% |

**Adaptive Statistics:**
- Mode Switches: 9
- Low Power Time (15W): 0.1%
- Medium Power Time (25W): 5.6%
- High Power Time (MAXN): 94.3%
- Avg Switch Time: 21.7 ms

## Variable Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 9.33 | 9.36 | 9.37 | 9.38 | +0.1% |
| Avg Power (W) | 5.26 | 5.22 | 5.38 | 5.28 | +1.8% |
| Total Energy (J) | 319.68 | 317.56 | 327.30 | 321.53 | +1.8% |
| Energy/Inference (mJ) | 42.624 | 42.341 | 43.640 | 42.870 | +1.8% |
| FPS/Watt | 23.78 | 23.94 | 23.23 | 23.65 | +1.8% |
| Violation Rate (%) | 14.15 | 10.69 | 17.73 | 12.81 | -27.7% |

**Adaptive Statistics:**
- Mode Switches: 9
- Low Power Time (15W): 44.4%
- Medium Power Time (25W): 10.0%
- High Power Time (MAXN): 45.5%
- Avg Switch Time: 22.4 ms

## Periodic Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 7.51 | 7.53 | 7.51 | 7.76 | +3.4% |
| Avg Power (W) | 4.75 | 4.74 | 4.75 | 4.73 | +0.3% |
| Total Energy (J) | 237.22 | 236.61 | 236.94 | 236.10 | +0.4% |
| Energy/Inference (mJ) | 131.789 | 131.447 | 131.636 | 131.167 | +0.4% |
| FPS/Watt | 6.31 | 6.33 | 6.32 | 6.34 | +0.3% |
| Violation Rate (%) | 1.39 | 1.67 | 2.00 | 2.39 | +19.4% |

**Adaptive Statistics:**
- Mode Switches: 10
- Low Power Time (15W): 45.1%
- Medium Power Time (25W): 1.2%
- High Power Time (MAXN): 53.6%
- Avg Switch Time: 21.5 ms

