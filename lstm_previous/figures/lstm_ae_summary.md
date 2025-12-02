# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 15W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 14.24 | 14.44 | 14.34 | -0.6% |
| Avg Power (W) | 5.30 | 5.62 | 5.50 | +2.1% |
| Total Energy (J) | 318.13 | 336.91 | 329.74 | +2.1% |
| Energy/Inference (mJ) | 132.553 | 140.381 | 137.390 | +2.1% |
| FPS/Watt | 7.54 | 7.12 | 7.28 | +2.1% |
| Violation Rate (%) | 19.46 | 20.17 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 7
- Low Power Time: 18.3%
- Avg Switch Time: 22.7 ms

## Continuous Workload

| Metric | Static 15W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 11.13 | 9.13 | 8.19 | -10.2% |
| Avg Power (W) | 6.30 | 6.25 | 5.64 | +9.8% |
| Total Energy (J) | 383.37 | 380.73 | 343.56 | +9.8% |
| Energy/Inference (mJ) | 63.895 | 63.454 | 57.260 | +9.8% |
| FPS/Watt | 15.88 | 15.99 | 17.72 | +10.8% |
| Violation Rate (%) | 7.00 | 2.55 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 2
- Low Power Time: 86.8%
- Avg Switch Time: 23.5 ms

## Variable Workload

| Metric | Static 15W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 11.74 | 10.85 | 10.46 | -3.5% |
| Avg Power (W) | 6.46 | 6.11 | 6.36 | -4.0% |
| Total Energy (J) | 434.63 | 371.89 | 381.30 | -2.5% |
| Energy/Inference (mJ) | 57.950 | 49.586 | 50.840 | -2.5% |
| FPS/Watt | 19.34 | 20.46 | 19.67 | -3.9% |
| Violation Rate (%) | 9.13 | 15.15 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 7
- Low Power Time: 36.7%
- Avg Switch Time: 27.6 ms

## Periodic Workload

| Metric | Static 15W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 11.17 | 10.71 | 10.78 | +0.6% |
| Avg Power (W) | 6.07 | 5.45 | 5.42 | +0.4% |
| Total Energy (J) | 303.54 | 272.16 | 271.00 | +0.4% |
| Energy/Inference (mJ) | 168.632 | 151.201 | 150.557 | +0.4% |
| FPS/Watt | 4.94 | 5.51 | 5.53 | +0.4% |
| Violation Rate (%) | 8.33 | 6.17 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 5
- Low Power Time: 0.3%
- Avg Switch Time: 21.8 ms

