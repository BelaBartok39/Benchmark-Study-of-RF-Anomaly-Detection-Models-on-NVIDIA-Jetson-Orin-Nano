# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 14.45 | 14.48 | 14.43 | 14.49 | +0.4% |
| Avg Power (W) | 5.08 | 5.09 | 5.10 | 5.10 | +0.0% |
| Total Energy (J) | 18290.11 | 18338.60 | 18372.40 | 18368.95 | +0.0% |
| Energy/Inference (mJ) | 348.383 | 349.307 | 349.951 | 349.885 | +0.0% |
| FPS/Watt | 2.87 | 2.86 | 2.86 | 2.86 | +0.0% |
| Violation Rate (%) | 68.67 | 68.22 | 68.21 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 398
- Low Power Time (15W): 74.6%
- Medium Power Time (25W): 25.4%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 22.3 ms

## Continuous Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 8.17 | 7.54 | 7.60 | 8.87 | +16.7% |
| Avg Power (W) | 5.59 | 5.73 | 5.75 | 5.58 | +3.0% |
| Total Energy (J) | 20110.32 | 20634.30 | 20718.79 | 20106.11 | +3.0% |
| Energy/Inference (mJ) | 55.862 | 57.317 | 57.552 | 55.850 | +3.0% |
| FPS/Watt | 17.90 | 17.45 | 17.38 | 17.91 | +3.1% |
| Violation Rate (%) | 2.56 | 1.77 | 1.85 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 3
- Low Power Time (15W): 99.9%
- Medium Power Time (25W): 0.1%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 22.0 ms

## Variable Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 8.30 | 10.89 | 10.85 | 8.38 | -22.7% |
| Avg Power (W) | 5.62 | 5.92 | 6.02 | 5.61 | +6.8% |
| Total Energy (J) | 20832.35 | 21319.82 | 21689.20 | 21313.82 | +1.7% |
| Energy/Inference (mJ) | 49.366 | 50.521 | 51.396 | 50.507 | +1.7% |
| FPS/Watt | 20.87 | 19.80 | 19.46 | 20.88 | +7.3% |
| Violation Rate (%) | 2.77 | 10.24 | 11.29 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 3
- Low Power Time (15W): 99.9%
- Medium Power Time (25W): 0.1%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 22.1 ms

## Periodic Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 10.81 | 9.32 | 9.49 | 10.82 | +14.0% |
| Avg Power (W) | 5.03 | 5.04 | 5.04 | 5.00 | +0.7% |
| Total Energy (J) | 17396.60 | 17400.96 | 17397.73 | 17269.36 | +0.7% |
| Energy/Inference (mJ) | 1449.717 | 1450.080 | 1449.811 | 1439.114 | +0.7% |
| FPS/Watt | 0.66 | 0.66 | 0.66 | 0.67 | +0.7% |
| Violation Rate (%) | 6.15 | 4.23 | 4.47 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 5
- Low Power Time (15W): 99.8%
- Medium Power Time (25W): 0.2%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 22.6 ms

