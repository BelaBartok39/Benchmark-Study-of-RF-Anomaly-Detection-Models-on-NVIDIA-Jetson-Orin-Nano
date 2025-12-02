# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 15W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 1.90 | 1.98 | 1.94 | -1.6% |
| Avg Power (W) | 5.15 | 5.15 | 5.18 | -0.5% |
| Total Energy (J) | 308.70 | 308.91 | 310.58 | -0.5% |
| Energy/Inference (mJ) | 128.623 | 128.714 | 129.406 | -0.5% |
| FPS/Watt | 7.77 | 7.76 | 7.72 | -0.5% |
| Violation Rate (%) | 0.04 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time: 100.0%
- Avg Switch Time: 0.0 ms

## Continuous Workload

| Metric | Static 15W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 1.80 | 1.79 | 1.78 | -0.5% |
| Avg Power (W) | 5.28 | 5.27 | 5.28 | -0.2% |
| Total Energy (J) | 321.31 | 321.13 | 321.68 | -0.2% |
| Energy/Inference (mJ) | 53.552 | 53.522 | 53.613 | -0.2% |
| FPS/Watt | 18.95 | 18.96 | 18.93 | -0.2% |
| Violation Rate (%) | 0.02 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time: 100.0%
- Avg Switch Time: 0.0 ms

## Variable Workload

| Metric | Static 15W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 1.71 | 2.21 | 2.22 | +0.6% |
| Avg Power (W) | 4.91 | 5.30 | 5.34 | -0.8% |
| Total Energy (J) | 298.96 | 322.77 | 325.39 | -0.8% |
| Energy/Inference (mJ) | 39.861 | 43.036 | 43.386 | -0.8% |
| FPS/Watt | 25.46 | 23.58 | 23.39 | -0.8% |
| Violation Rate (%) | 0.01 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time: 100.0%
- Avg Switch Time: 0.0 ms

## Periodic Workload

| Metric | Static 15W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 1.84 | 1.80 | 1.80 | +0.0% |
| Avg Power (W) | 5.16 | 5.15 | 5.12 | +0.7% |
| Total Energy (J) | 258.09 | 257.48 | 255.70 | +0.7% |
| Energy/Inference (mJ) | 143.386 | 143.043 | 142.057 | +0.7% |
| FPS/Watt | 5.81 | 5.82 | 5.86 | +0.7% |
| Violation Rate (%) | 0.06 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time: 100.0%
- Avg Switch Time: 0.0 ms

