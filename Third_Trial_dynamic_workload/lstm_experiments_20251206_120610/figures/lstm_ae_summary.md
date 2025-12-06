# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 60.26 | 44.85 | 41.26 | 42.87 | +3.9% |
| Avg Power (W) | 6.24 | 7.02 | 6.89 | 6.91 | -0.4% |
| Total Energy (J) | 1084.11 | 772.34 | 713.76 | 729.36 | -2.2% |
| Energy/Inference (mJ) | 45.171 | 32.181 | 29.740 | 30.390 | -2.2% |
| FPS/Watt | 64.06 | 56.97 | 58.07 | 57.86 | -0.4% |
| Violation Rate (%) | 0.13 | 0.00 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time (15W): 100.0%
- Avg Switch Time: 0.0 ms

## Continuous Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 56.16 | 42.19 | 42.25 | 45.34 | +7.3% |
| Avg Power (W) | 5.78 | 6.90 | 6.90 | 6.83 | +1.0% |
| Total Energy (J) | 2373.24 | 1788.89 | 1782.05 | 1832.27 | -2.8% |
| Energy/Inference (mJ) | 39.554 | 29.815 | 29.701 | 30.538 | -2.8% |
| FPS/Watt | 173.13 | 144.83 | 144.92 | 146.40 | +1.0% |
| Violation Rate (%) | 0.05 | 0.00 | 0.00 | 0.04 | N/A |

**Adaptive Statistics:**
- Mode Switches: 46
- Low Power Time (15W): 61.0%
- Avg Switch Time: 24.8 ms

## Variable Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 61.15 | 41.50 | 41.47 | 42.07 | +1.4% |
| Avg Power (W) | 5.78 | 6.91 | 6.91 | 6.90 | +0.1% |
| Total Energy (J) | 2973.65 | 2218.35 | 2211.22 | 2228.85 | -0.8% |
| Energy/Inference (mJ) | 39.649 | 29.578 | 29.483 | 29.718 | -0.8% |
| FPS/Watt | 216.27 | 180.85 | 180.92 | 181.09 | +0.1% |
| Violation Rate (%) | 0.01 | 0.00 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time (15W): 100.0%
- Avg Switch Time: 0.0 ms

## Periodic Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 52.87 | 41.02 | 38.02 | 39.06 | +2.7% |
| Avg Power (W) | 5.77 | 6.89 | 6.89 | 6.90 | -0.2% |
| Total Energy (J) | 707.87 | 532.31 | 526.06 | 521.14 | +0.9% |
| Energy/Inference (mJ) | 39.326 | 29.573 | 29.225 | 28.952 | +0.9% |
| FPS/Watt | 52.00 | 43.56 | 43.55 | 43.45 | -0.2% |
| Violation Rate (%) | 0.00 | 0.00 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time (15W): 100.0%
- Avg Switch Time: 0.0 ms

