# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 15W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 5.60 | 6.12 | 5.41 | -11.6% |
| Avg Power (W) | 5.42 | 5.41 | 5.34 | +1.4% |
| Total Energy (J) | 325.26 | 324.72 | 320.02 | +1.4% |
| Energy/Inference (mJ) | 135.526 | 135.298 | 133.340 | +1.4% |
| FPS/Watt | 7.38 | 7.39 | 7.50 | +1.5% |
| Violation Rate (%) | 0.08 | 0.08 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time: 100.0%
- Avg Switch Time: 0.0 ms

## Continuous Workload

| Metric | Static 15W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 4.77 | 5.07 | 5.26 | +3.7% |
| Avg Power (W) | 5.52 | 5.55 | 5.58 | -0.5% |
| Total Energy (J) | 335.80 | 337.84 | 339.78 | -0.6% |
| Energy/Inference (mJ) | 55.966 | 56.307 | 56.630 | -0.6% |
| FPS/Watt | 18.13 | 18.02 | 17.93 | -0.5% |
| Violation Rate (%) | 0.02 | 0.02 | 0.02 | +0.0% |

**Adaptive Statistics:**
- Mode Switches: 2
- Low Power Time: 91.7%
- Avg Switch Time: 24.6 ms

## Variable Workload

| Metric | Static 15W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 4.76 | 4.75 | 4.82 | +1.5% |
| Avg Power (W) | 5.57 | 5.59 | 5.60 | -0.2% |
| Total Energy (J) | 339.29 | 340.45 | 341.12 | -0.2% |
| Energy/Inference (mJ) | 45.238 | 45.393 | 45.482 | -0.2% |
| FPS/Watt | 22.43 | 22.35 | 22.31 | -0.2% |
| Violation Rate (%) | 0.01 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time: 100.0%
- Avg Switch Time: 0.0 ms

## Periodic Workload

| Metric | Static 15W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 5.16 | 5.02 | 5.23 | +4.0% |
| Avg Power (W) | 5.23 | 5.30 | 5.24 | +1.1% |
| Total Energy (J) | 261.59 | 264.72 | 261.74 | +1.1% |
| Energy/Inference (mJ) | 145.327 | 147.064 | 145.412 | +1.1% |
| FPS/Watt | 5.73 | 5.66 | 5.73 | +1.1% |
| Violation Rate (%) | 0.06 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time: 100.0%
- Avg Switch Time: 0.0 ms

