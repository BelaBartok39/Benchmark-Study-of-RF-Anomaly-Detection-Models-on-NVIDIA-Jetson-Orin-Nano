# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 2.09 | 2.13 | 2.23 | 2.23 | -0.1% |
| Avg Power (W) | 5.23 | 5.16 | 5.23 | 5.18 | +0.9% |
| Total Energy (J) | 313.61 | 309.12 | 313.58 | 310.67 | +0.9% |
| Energy/Inference (mJ) | 130.671 | 128.799 | 130.660 | 129.447 | +0.9% |
| FPS/Watt | 7.65 | 7.76 | 7.65 | 7.72 | +0.9% |
| Violation Rate (%) | 0.04 | 0.00 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 10
- Low Power Time (15W): 51.7%
- Avg Switch Time: 22.5 ms

## Continuous Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 2.29 | 2.09 | 2.06 | 2.14 | +3.7% |
| Avg Power (W) | 5.28 | 5.24 | 5.23 | 5.27 | -0.8% |
| Total Energy (J) | 321.47 | 319.18 | 318.60 | 321.12 | -0.8% |
| Energy/Inference (mJ) | 53.578 | 53.196 | 53.100 | 53.519 | -0.8% |
| FPS/Watt | 18.94 | 19.07 | 19.11 | 18.96 | -0.8% |
| Violation Rate (%) | 0.07 | 0.02 | 0.03 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 7
- Low Power Time (15W): 8.0%
- Avg Switch Time: 22.5 ms

## Variable Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 3.49 | 3.22 | 3.25 | 2.95 | -9.4% |
| Avg Power (W) | 5.31 | 5.32 | 5.26 | 5.26 | +0.0% |
| Total Energy (J) | 323.06 | 324.12 | 320.38 | 320.25 | +0.0% |
| Energy/Inference (mJ) | 43.074 | 43.216 | 42.718 | 42.700 | +0.0% |
| FPS/Watt | 23.56 | 23.48 | 23.75 | 23.76 | +0.0% |
| Violation Rate (%) | 0.04 | 0.03 | 0.07 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 4
- Low Power Time (15W): 16.1%
- Avg Switch Time: 22.3 ms

## Periodic Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 4.85 | 3.66 | 2.99 | 2.26 | -24.5% |
| Avg Power (W) | 5.15 | 5.20 | 5.13 | 5.18 | -1.0% |
| Total Energy (J) | 257.53 | 259.64 | 256.55 | 259.04 | -1.0% |
| Energy/Inference (mJ) | 143.072 | 144.245 | 142.529 | 143.911 | -1.0% |
| FPS/Watt | 5.82 | 5.77 | 5.84 | 5.79 | -0.9% |
| Violation Rate (%) | 0.17 | 0.00 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 7
- Low Power Time (15W): 1.5%
- Avg Switch Time: 23.1 ms

