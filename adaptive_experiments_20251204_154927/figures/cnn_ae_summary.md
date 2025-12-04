# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 28.81 | 27.47 | 27.41 | 27.70 | +1.1% |
| Avg Power (W) | 7.20 | 7.08 | 7.08 | 7.14 | -0.9% |
| Total Energy (J) | 435.28 | 427.27 | 427.54 | 437.64 | -2.4% |
| Energy/Inference (mJ) | 18.137 | 17.803 | 17.814 | 18.235 | -2.4% |
| FPS/Watt | 55.52 | 56.49 | 56.51 | 56.03 | -0.8% |
| Violation Rate (%) | 99.98 | 100.00 | 99.99 | 100.00 | +0.0% |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 0.3%
- Avg Switch Time: 27.0 ms

## Continuous Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 28.14 | 27.37 | 27.93 | 28.32 | +1.4% |
| Avg Power (W) | 7.14 | 7.30 | 7.29 | 7.19 | +1.4% |
| Total Energy (J) | 1014.07 | 995.81 | 1007.94 | 1007.50 | +0.0% |
| Energy/Inference (mJ) | 16.901 | 16.597 | 16.799 | 16.792 | +0.0% |
| FPS/Watt | 140.00 | 137.04 | 137.13 | 139.12 | +1.5% |
| Violation Rate (%) | 99.99 | 99.98 | 100.00 | 99.99 | -0.0% |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 0.1%
- Avg Switch Time: 24.8 ms

## Variable Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 28.33 | 27.85 | 28.26 | 28.60 | +1.2% |
| Avg Power (W) | 7.15 | 7.24 | 7.28 | 7.20 | +1.1% |
| Total Energy (J) | 1269.76 | 1253.04 | 1264.79 | 1265.89 | -0.1% |
| Energy/Inference (mJ) | 16.930 | 16.707 | 16.864 | 16.878 | -0.1% |
| FPS/Watt | 174.81 | 172.69 | 171.79 | 173.66 | +1.1% |
| Violation Rate (%) | 99.98 | 99.99 | 99.98 | 99.98 | -0.0% |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 0.1%
- Avg Switch Time: 21.6 ms

## Periodic Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 27.75 | 27.00 | 27.01 | 27.36 | +1.3% |
| Avg Power (W) | 6.68 | 6.76 | 6.75 | 6.75 | -0.0% |
| Total Energy (J) | 378.48 | 376.25 | 375.91 | 376.13 | -0.1% |
| Energy/Inference (mJ) | 21.027 | 20.903 | 20.884 | 20.896 | -0.1% |
| FPS/Watt | 44.89 | 44.41 | 44.47 | 44.46 | -0.0% |
| Violation Rate (%) | 99.98 | 99.97 | 99.99 | 99.98 | -0.0% |

**Adaptive Statistics:**
- Mode Switches: 3
- Low Power Time (15W): 0.4%
- Avg Switch Time: 24.0 ms

