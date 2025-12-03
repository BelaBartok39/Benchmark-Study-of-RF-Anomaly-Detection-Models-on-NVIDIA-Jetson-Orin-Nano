# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 9.91 | 10.15 | 10.97 | 11.13 | +1.5% |
| Avg Power (W) | 5.28 | 6.02 | 6.25 | 6.16 | +1.5% |
| Total Energy (J) | 316.56 | 361.03 | 375.37 | 369.84 | +1.5% |
| Energy/Inference (mJ) | 131.898 | 150.429 | 156.404 | 154.099 | +1.5% |
| FPS/Watt | 7.58 | 6.65 | 6.40 | 6.49 | +1.5% |
| Violation Rate (%) | 3.92 | 6.38 | 12.33 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 3
- Low Power Time (15W): 28.9%
- Avg Switch Time: 34.3 ms

## Continuous Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 8.55 | 7.53 | 7.74 | 7.91 | +2.2% |
| Avg Power (W) | 5.97 | 5.58 | 5.64 | 5.53 | +2.0% |
| Total Energy (J) | 363.30 | 339.57 | 343.52 | 336.58 | +2.0% |
| Energy/Inference (mJ) | 60.549 | 56.594 | 57.253 | 56.097 | +2.0% |
| FPS/Watt | 16.76 | 17.93 | 17.72 | 18.09 | +2.1% |
| Violation Rate (%) | 0.97 | 0.03 | 0.32 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 2
- Low Power Time (15W): 81.1%
- Avg Switch Time: 22.3 ms

## Variable Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 9.34 | 9.34 | 9.38 | 9.34 | -0.4% |
| Avg Power (W) | 5.70 | 5.83 | 5.88 | 5.78 | +1.7% |
| Total Energy (J) | 347.16 | 354.80 | 357.96 | 351.94 | +1.7% |
| Energy/Inference (mJ) | 46.288 | 47.306 | 47.728 | 46.926 | +1.7% |
| FPS/Watt | 21.92 | 21.44 | 21.25 | 21.62 | +1.7% |
| Violation Rate (%) | 0.56 | 0.71 | 0.61 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 4
- Low Power Time (15W): 70.3%
- Avg Switch Time: 22.3 ms

## Periodic Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 7.84 | 8.67 | 7.53 | 8.46 | +12.3% |
| Avg Power (W) | 5.28 | 5.20 | 4.86 | 5.38 | -10.7% |
| Total Energy (J) | 264.01 | 260.03 | 242.94 | 269.01 | -10.7% |
| Energy/Inference (mJ) | 146.671 | 144.463 | 134.964 | 149.451 | -10.7% |
| FPS/Watt | 5.68 | 5.77 | 6.17 | 5.57 | -9.7% |
| Violation Rate (%) | 0.22 | 0.78 | 0.11 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time (15W): 100.0%
- Avg Switch Time: 0.0 ms

