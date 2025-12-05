# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 28.20 | 27.71 | 27.56 | 28.44 | +3.2% |
| Avg Power (W) | 7.02 | 7.12 | 7.10 | 7.00 | +1.5% |
| Total Energy (J) | 424.34 | 429.86 | 428.85 | 429.47 | -0.1% |
| Energy/Inference (mJ) | 17.681 | 17.911 | 17.869 | 17.894 | -0.1% |
| FPS/Watt | 56.97 | 56.17 | 56.31 | 57.16 | +1.5% |
| Violation Rate (%) | 0.02 | 0.03 | 0.03 | 0.02 | -16.7% |

**Adaptive Statistics:**
- Mode Switches: 2
- Low Power Time (15W): 91.8%
- Avg Switch Time: 26.7 ms

## Continuous Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 28.02 | 27.27 | 27.85 | 29.19 | +4.8% |
| Avg Power (W) | 7.17 | 7.34 | 7.26 | 7.17 | +1.2% |
| Total Energy (J) | 1011.17 | 993.72 | 1003.47 | 1038.26 | -3.5% |
| Energy/Inference (mJ) | 16.853 | 16.562 | 16.725 | 17.304 | -3.5% |
| FPS/Watt | 139.50 | 136.32 | 137.80 | 139.44 | +1.2% |
| Violation Rate (%) | 0.02 | 0.03 | 0.02 | 0.02 | +40.0% |

**Adaptive Statistics:**
- Mode Switches: 6
- Low Power Time (15W): 79.2%
- Avg Switch Time: 25.8 ms

## Variable Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 28.03 | 27.38 | 27.62 | 28.77 | +4.2% |
| Avg Power (W) | 7.17 | 7.35 | 7.30 | 7.19 | +1.4% |
| Total Energy (J) | 1266.42 | 1245.13 | 1248.90 | 1278.23 | -2.3% |
| Energy/Inference (mJ) | 16.886 | 16.602 | 16.652 | 17.043 | -2.3% |
| FPS/Watt | 174.44 | 170.06 | 171.29 | 173.79 | +1.5% |
| Violation Rate (%) | 0.01 | 0.01 | 0.03 | 0.02 | -20.0% |

**Adaptive Statistics:**
- Mode Switches: 9
- Low Power Time (15W): 72.2%
- Avg Switch Time: 26.3 ms

## Periodic Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 27.96 | 27.40 | 27.05 | 28.29 | +4.6% |
| Avg Power (W) | 6.69 | 6.73 | 6.76 | 6.71 | +0.8% |
| Total Energy (J) | 378.99 | 380.93 | 376.70 | 379.81 | -0.8% |
| Energy/Inference (mJ) | 21.055 | 21.163 | 20.928 | 21.101 | -0.8% |
| FPS/Watt | 44.85 | 44.60 | 44.38 | 44.74 | +0.8% |
| Violation Rate (%) | 0.06 | 0.06 | 0.06 | 0.06 | +0.0% |

**Adaptive Statistics:**
- Mode Switches: 2
- Low Power Time (15W): 82.2%
- Avg Switch Time: 25.2 ms

