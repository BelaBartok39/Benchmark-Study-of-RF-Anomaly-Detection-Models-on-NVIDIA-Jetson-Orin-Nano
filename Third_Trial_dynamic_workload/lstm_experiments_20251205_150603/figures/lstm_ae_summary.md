# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 14.26 | 14.31 | 14.24 | 14.30 | +0.5% |
| Avg Power (W) | 5.23 | 5.26 | 5.26 | 5.27 | -0.1% |
| Total Energy (J) | 80.78 | 81.26 | 81.23 | 81.32 | -0.1% |
| Energy/Inference (mJ) | 134.638 | 135.438 | 135.376 | 135.537 | -0.1% |
| FPS/Watt | 7.65 | 7.60 | 7.61 | 7.60 | -0.1% |
| Violation Rate (%) | 19.67 | 20.17 | 20.33 | 20.00 | -1.6% |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 0.3%
- Avg Switch Time: 22.1 ms

## Continuous Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 8.95 | 8.08 | 8.89 | 8.19 | -7.9% |
| Avg Power (W) | 5.80 | 5.70 | 5.71 | 5.70 | +0.2% |
| Total Energy (J) | 89.62 | 88.09 | 88.14 | 87.97 | +0.2% |
| Energy/Inference (mJ) | 59.748 | 58.728 | 58.758 | 58.646 | +0.2% |
| FPS/Watt | 17.23 | 17.53 | 17.52 | 17.55 | +0.2% |
| Violation Rate (%) | 3.87 | 2.73 | 3.73 | 3.40 | -8.9% |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 0.3%
- Avg Switch Time: 18.3 ms

## Variable Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 8.89 | 8.82 | 8.83 | 8.82 | -0.1% |
| Avg Power (W) | 5.58 | 6.32 | 6.30 | 6.26 | +0.6% |
| Total Energy (J) | 111.60 | 97.55 | 97.28 | 96.71 | +0.6% |
| Energy/Inference (mJ) | 49.599 | 43.354 | 43.236 | 42.982 | +0.6% |
| FPS/Watt | 26.86 | 23.74 | 23.81 | 23.95 | +0.6% |
| Violation Rate (%) | 3.78 | 3.78 | 3.60 | 3.02 | -16.0% |

**Adaptive Statistics:**
- Mode Switches: 3
- Low Power Time (15W): 0.3%
- Avg Switch Time: 22.4 ms

## Periodic Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 8.92 | 9.66 | 10.04 | 10.22 | +1.8% |
| Avg Power (W) | 5.59 | 5.69 | 5.72 | 5.72 | +0.0% |
| Total Energy (J) | 25.41 | 25.84 | 25.96 | 25.96 | -0.0% |
| Energy/Inference (mJ) | 56.477 | 57.430 | 57.690 | 57.696 | -0.0% |
| FPS/Watt | 5.36 | 5.27 | 5.25 | 5.25 | +0.0% |
| Violation Rate (%) | 3.56 | 4.67 | 4.67 | 5.33 | +14.3% |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 1.0%
- Avg Switch Time: 19.5 ms

