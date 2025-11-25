# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 7W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 2.69 | 2.60 | 2.65 | +1.9% |
| Avg Power (W) | 5.26 | 5.28 | 5.25 | +0.6% |
| Total Energy (J) | 315.70 | 316.52 | 314.74 | +0.6% |
| Energy/Inference (mJ) | 131.543 | 131.883 | 131.143 | +0.6% |
| FPS/Watt | 7.60 | 7.58 | 7.62 | +0.6% |
| Violation Rate (%) | 0.04 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time: 100.0%
- Avg Switch Time: 0.0 ms

## Continuous Workload

| Metric | Static 7W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 2.57 | 2.59 | 2.59 | +0.0% |
| Avg Power (W) | 5.45 | 5.46 | 5.46 | +0.0% |
| Total Energy (J) | 331.97 | 332.42 | 327.51 | +1.5% |
| Energy/Inference (mJ) | 55.329 | 55.404 | 54.585 | +1.5% |
| FPS/Watt | 18.34 | 18.32 | 18.32 | +0.0% |
| Violation Rate (%) | 0.02 | 0.03 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time: 100.0%
- Avg Switch Time: 0.0 ms

## Variable Workload

| Metric | Static 7W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 2.76 | 2.76 | 2.77 | +0.1% |
| Avg Power (W) | 5.51 | 5.53 | 5.52 | +0.2% |
| Total Energy (J) | 330.62 | 331.91 | 331.24 | +0.2% |
| Energy/Inference (mJ) | 44.083 | 44.254 | 44.165 | +0.2% |
| FPS/Watt | 22.68 | 22.60 | 22.64 | +0.2% |
| Violation Rate (%) | 0.03 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time: 100.0%
- Avg Switch Time: 0.0 ms

## Periodic Workload

| Metric | Static 7W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 2.53 | 2.72 | 3.07 | +12.6% |
| Avg Power (W) | 5.23 | 5.22 | 5.28 | -1.2% |
| Total Energy (J) | 261.26 | 260.97 | 264.07 | -1.2% |
| Energy/Inference (mJ) | 145.146 | 144.986 | 146.705 | -1.2% |
| FPS/Watt | 5.74 | 5.75 | 5.68 | -1.2% |
| Violation Rate (%) | 0.06 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time: 100.0%
- Avg Switch Time: 0.0 ms

