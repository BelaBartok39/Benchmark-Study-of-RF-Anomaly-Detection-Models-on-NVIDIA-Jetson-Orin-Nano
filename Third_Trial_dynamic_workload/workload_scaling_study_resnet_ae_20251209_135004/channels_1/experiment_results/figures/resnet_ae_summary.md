# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 9.52 | 9.62 | 9.82 | 9.83 | +0.1% |
| Avg Power (W) | 5.19 | 5.18 | 5.20 | 5.25 | -0.9% |
| Total Energy (J) | 311.05 | 310.26 | 311.76 | 314.46 | -0.9% |
| Energy/Inference (mJ) | 129.604 | 129.275 | 129.902 | 131.025 | -0.9% |
| FPS/Watt | 7.71 | 7.73 | 7.69 | 7.62 | -0.9% |
| Violation Rate (%) | 18.12 | 17.33 | 16.67 | 22.42 | +34.5% |

**Adaptive Statistics:**
- Mode Switches: 8
- Low Power Time (15W): 0.2%
- Medium Power Time (25W): 0.4%
- High Power Time (MAXN): 99.4%
- Avg Switch Time: 21.4 ms

## Continuous Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 7.62 | 8.03 | 7.53 | 9.39 | +24.7% |
| Avg Power (W) | 5.44 | 5.62 | 5.47 | 5.53 | -1.0% |
| Total Energy (J) | 330.84 | 342.16 | 333.02 | 336.26 | -1.0% |
| Energy/Inference (mJ) | 55.140 | 57.027 | 55.503 | 56.043 | -1.0% |
| FPS/Watt | 18.40 | 17.79 | 18.28 | 18.10 | -1.0% |
| Violation Rate (%) | 1.23 | 1.52 | 0.58 | 6.87 | +1077.1% |

**Adaptive Statistics:**
- Mode Switches: 2
- Low Power Time (15W): 0.1%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 99.9%
- Avg Switch Time: 20.2 ms

## Variable Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 9.26 | 9.23 | 9.23 | 9.16 | -0.7% |
| Avg Power (W) | 5.64 | 5.60 | 5.76 | 5.76 | -0.0% |
| Total Energy (J) | 343.27 | 341.03 | 350.21 | 350.27 | -0.0% |
| Energy/Inference (mJ) | 45.770 | 45.471 | 46.694 | 46.703 | -0.0% |
| FPS/Watt | 22.15 | 22.31 | 21.71 | 21.71 | -0.0% |
| Violation Rate (%) | 13.47 | 9.77 | 16.89 | 12.29 | -27.2% |

**Adaptive Statistics:**
- Mode Switches: 15
- Low Power Time (15W): 2.8%
- Medium Power Time (25W): 21.7%
- High Power Time (MAXN): 75.6%
- Avg Switch Time: 21.6 ms

## Periodic Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 7.64 | 7.62 | 7.48 | 9.30 | +24.3% |
| Avg Power (W) | 5.18 | 5.16 | 5.17 | 5.18 | -0.2% |
| Total Energy (J) | 258.83 | 257.54 | 258.35 | 258.75 | -0.2% |
| Energy/Inference (mJ) | 143.794 | 143.077 | 143.528 | 143.750 | -0.2% |
| FPS/Watt | 5.79 | 5.82 | 5.80 | 5.79 | -0.2% |
| Violation Rate (%) | 1.89 | 2.39 | 1.72 | 9.06 | +425.8% |

**Adaptive Statistics:**
- Mode Switches: 8
- Low Power Time (15W): 45.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 55.0%
- Avg Switch Time: 21.8 ms

