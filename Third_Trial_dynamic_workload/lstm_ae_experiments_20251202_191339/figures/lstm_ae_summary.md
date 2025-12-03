# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 10.67 | 9.93 | 10.09 | 9.99 | -1.0% |
| Avg Power (W) | 6.02 | 6.39 | 6.45 | 6.60 | -2.4% |
| Total Energy (J) | 366.99 | 389.08 | 392.65 | 402.10 | -2.4% |
| Energy/Inference (mJ) | 15.291 | 16.212 | 16.360 | 16.754 | -2.4% |
| FPS/Watt | 66.40 | 62.64 | 62.05 | 60.62 | -2.3% |
| Violation Rate (%) | 8.00 | 4.60 | 5.43 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time (15W): 100.0%
- Avg Switch Time: 0.0 ms

## Continuous Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 9.06 | 6.07 | 6.02 | 6.07 | +0.9% |
| Avg Power (W) | 6.91 | 8.21 | 8.21 | 8.27 | -0.7% |
| Total Energy (J) | 471.16 | 500.17 | 500.47 | 503.97 | -0.7% |
| Energy/Inference (mJ) | 7.853 | 8.336 | 8.341 | 8.399 | -0.7% |
| FPS/Watt | 144.71 | 121.85 | 121.77 | 120.88 | -0.7% |
| Violation Rate (%) | 1.40 | 0.12 | 0.11 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time (15W): 100.0%
- Avg Switch Time: 0.0 ms

## Variable Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 9.07 | 6.45 | 6.35 | 6.21 | -2.3% |
| Avg Power (W) | 6.92 | 8.55 | 8.51 | 8.54 | -0.3% |
| Total Energy (J) | 584.99 | 521.21 | 518.79 | 520.36 | -0.3% |
| Energy/Inference (mJ) | 7.800 | 6.949 | 6.917 | 6.938 | -0.3% |
| FPS/Watt | 180.67 | 146.12 | 146.82 | 146.38 | -0.3% |
| Violation Rate (%) | 1.56 | 0.15 | 0.14 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time (15W): 100.0%
- Avg Switch Time: 0.0 ms

## Periodic Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 9.63 | 6.63 | 6.65 | 6.74 | +1.3% |
| Avg Power (W) | 5.96 | 6.41 | 6.34 | 6.34 | -0.0% |
| Total Energy (J) | 308.79 | 326.46 | 322.91 | 323.00 | -0.0% |
| Energy/Inference (mJ) | 17.155 | 18.136 | 17.939 | 17.944 | -0.0% |
| FPS/Watt | 50.34 | 46.79 | 47.31 | 47.29 | -0.0% |
| Violation Rate (%) | 3.60 | 1.11 | 1.33 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time (15W): 100.0%
- Avg Switch Time: 0.0 ms

