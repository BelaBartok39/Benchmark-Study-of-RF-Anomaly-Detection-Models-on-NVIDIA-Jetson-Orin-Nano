# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 15W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 9.84 | 10.33 | 10.21 | -1.1% |
| Avg Power (W) | 5.29 | 5.51 | 5.89 | -6.8% |
| Total Energy (J) | 317.01 | 330.74 | 352.90 | -6.7% |
| Energy/Inference (mJ) | 132.087 | 137.807 | 147.040 | -6.7% |
| FPS/Watt | 7.57 | 7.26 | 6.80 | -6.3% |
| Violation Rate (%) | 2.54 | 7.38 | 7.79 | +5.6% |

**Adaptive Statistics:**
- Mode Switches: 19
- Low Power Time: 12.7%
- Avg Switch Time: 21.7 ms

## Continuous Workload

| Metric | Static 15W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 8.13 | 8.11 | 8.22 | +1.3% |
| Avg Power (W) | 6.36 | 6.70 | 6.50 | +2.9% |
| Total Energy (J) | 387.02 | 407.49 | 395.44 | +3.0% |
| Energy/Inference (mJ) | 64.503 | 67.916 | 65.906 | +3.0% |
| FPS/Watt | 15.73 | 14.94 | 15.39 | +3.0% |
| Violation Rate (%) | 0.77 | 1.07 | 0.67 | -37.5% |

**Adaptive Statistics:**
- Mode Switches: 19
- Low Power Time: 22.5%
- Avg Switch Time: 21.7 ms

## Variable Workload

| Metric | Static 15W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 9.43 | 9.52 | 9.47 | -0.5% |
| Avg Power (W) | 5.87 | 5.96 | 6.01 | -0.8% |
| Total Energy (J) | 357.18 | 363.06 | 366.09 | -0.8% |
| Energy/Inference (mJ) | 47.623 | 48.408 | 48.811 | -0.8% |
| FPS/Watt | 21.30 | 20.96 | 20.79 | -0.8% |
| Violation Rate (%) | 0.95 | 1.67 | 1.41 | -15.2% |

**Adaptive Statistics:**
- Mode Switches: 15
- Low Power Time: 35.8%
- Avg Switch Time: 23.1 ms

## Periodic Workload

| Metric | Static 15W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 8.16 | 9.15 | 8.19 | -10.5% |
| Avg Power (W) | 5.28 | 5.36 | 5.27 | +1.6% |
| Total Energy (J) | 263.88 | 268.04 | 263.55 | +1.7% |
| Energy/Inference (mJ) | 146.603 | 148.912 | 146.417 | +1.7% |
| FPS/Watt | 5.68 | 5.60 | 5.69 | +1.7% |
| Violation Rate (%) | 0.56 | 1.33 | 1.17 | -12.5% |

**Adaptive Statistics:**
- Mode Switches: 7
- Low Power Time: 0.4%
- Avg Switch Time: 21.4 ms

