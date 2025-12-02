# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 15W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 2.13 | 2.17 | 2.07 | -4.6% |
| Avg Power (W) | 5.14 | 5.14 | 5.18 | -0.7% |
| Total Energy (J) | 308.15 | 308.36 | 310.60 | -0.7% |
| Energy/Inference (mJ) | 128.397 | 128.483 | 129.417 | -0.7% |
| FPS/Watt | 7.78 | 7.78 | 7.72 | -0.7% |
| Violation Rate (%) | 0.04 | 0.04 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time: 100.0%
- Avg Switch Time: 0.0 ms

## Continuous Workload

| Metric | Static 15W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 2.05 | 2.07 | 2.09 | +1.0% |
| Avg Power (W) | 5.26 | 5.26 | 5.24 | +0.4% |
| Total Energy (J) | 320.20 | 320.49 | 319.29 | +0.4% |
| Energy/Inference (mJ) | 53.367 | 53.415 | 53.216 | +0.4% |
| FPS/Watt | 19.01 | 19.00 | 19.07 | +0.4% |
| Violation Rate (%) | 0.05 | 0.00 | 0.03 | N/A |

**Adaptive Statistics:**
- Mode Switches: 4
- Low Power Time: 83.5%
- Avg Switch Time: 22.4 ms

## Variable Workload

| Metric | Static 15W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 3.43 | 3.66 | 4.84 | +32.4% |
| Avg Power (W) | 5.26 | 5.28 | 5.28 | -0.0% |
| Total Energy (J) | 320.35 | 321.16 | 321.32 | -0.0% |
| Energy/Inference (mJ) | 42.713 | 42.821 | 42.842 | -0.0% |
| FPS/Watt | 23.76 | 23.69 | 23.68 | -0.0% |
| Violation Rate (%) | 0.01 | 0.00 | 0.05 | N/A |

**Adaptive Statistics:**
- Mode Switches: 6
- Low Power Time: 75.2%
- Avg Switch Time: 22.7 ms

## Periodic Workload

| Metric | Static 15W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 2.18 | 2.96 | 2.18 | -26.3% |
| Avg Power (W) | 5.14 | 5.14 | 5.14 | +0.2% |
| Total Energy (J) | 257.01 | 257.09 | 256.69 | +0.2% |
| Energy/Inference (mJ) | 142.785 | 142.829 | 142.608 | +0.2% |
| FPS/Watt | 5.83 | 5.83 | 5.84 | +0.2% |
| Violation Rate (%) | 0.06 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time: 100.0%
- Avg Switch Time: 0.0 ms

