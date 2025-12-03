# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 14.27 | 14.30 | 14.28 | 10.51 | -26.4% |
| Avg Power (W) | 5.40 | 5.39 | 5.33 | 6.10 | -14.3% |
| Total Energy (J) | 323.91 | 323.52 | 319.71 | 371.39 | -16.2% |
| Energy/Inference (mJ) | 134.961 | 134.800 | 133.213 | 15.474 | +88.4% |
| FPS/Watt | 7.41 | 7.42 | 7.50 | 65.62 | +774.6% |
| Violation Rate (%) | 21.92 | 19.79 | 21.08 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 2
- Low Power Time (15W): 86.3%
- Avg Switch Time: 23.8 ms

## Continuous Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 7.81 | 7.64 | 7.26 | 6.06 | -16.6% |
| Avg Power (W) | 5.22 | 5.68 | 5.81 | 8.20 | -41.1% |
| Total Energy (J) | 318.09 | 345.75 | 353.88 | 499.81 | -41.2% |
| Energy/Inference (mJ) | 53.015 | 57.625 | 58.981 | 8.330 | +85.9% |
| FPS/Watt | 19.14 | 17.61 | 17.20 | 121.96 | +608.9% |
| Violation Rate (%) | 0.42 | 0.60 | 1.12 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time (15W): 100.0%
- Avg Switch Time: 0.0 ms

## Variable Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 8.82 | 10.90 | 10.87 | 6.24 | -42.6% |
| Avg Power (W) | 5.68 | 6.16 | 6.16 | 8.55 | -38.7% |
| Total Energy (J) | 366.46 | 374.78 | 375.14 | 520.83 | -38.8% |
| Energy/Inference (mJ) | 48.861 | 49.970 | 50.019 | 6.944 | +86.1% |
| FPS/Watt | 22.01 | 20.31 | 20.29 | 146.26 | +621.0% |
| Violation Rate (%) | 2.85 | 14.29 | 14.39 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time (15W): 100.0%
- Avg Switch Time: 0.0 ms

## Periodic Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 9.71 | 9.76 | 9.46 | 6.86 | -27.4% |
| Avg Power (W) | 5.28 | 5.35 | 5.40 | 6.36 | -17.7% |
| Total Energy (J) | 263.99 | 267.21 | 269.74 | 323.61 | -20.0% |
| Energy/Inference (mJ) | 146.660 | 148.452 | 149.856 | 17.978 | +88.0% |
| FPS/Watt | 5.68 | 5.61 | 5.56 | 47.21 | +749.4% |
| Violation Rate (%) | 4.89 | 4.83 | 4.56 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time (15W): 100.0%
- Avg Switch Time: 0.0 ms

