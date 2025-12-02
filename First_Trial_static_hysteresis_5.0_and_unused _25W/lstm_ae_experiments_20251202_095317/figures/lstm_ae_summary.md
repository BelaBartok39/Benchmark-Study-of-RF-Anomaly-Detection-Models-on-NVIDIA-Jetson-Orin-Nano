# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 7W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 14.28 | 14.34 | 14.27 | -0.5% |
| Avg Power (W) | 5.34 | 5.38 | 5.37 | +0.1% |
| Total Energy (J) | 320.10 | 322.39 | 322.12 | +0.1% |
| Energy/Inference (mJ) | 133.375 | 134.331 | 134.219 | +0.1% |
| FPS/Watt | 7.50 | 7.44 | 7.45 | +0.1% |
| Violation Rate (%) | 20.62 | 21.08 | 22.58 | +7.1% |

**Adaptive Statistics:**
- Mode Switches: 9
- Low Power Time: 2.1%
- Avg Switch Time: 22.9 ms

## Continuous Workload

| Metric | Static 7W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 9.44 | 8.13 | 8.43 | +3.7% |
| Avg Power (W) | 5.62 | 5.80 | 5.87 | -1.3% |
| Total Energy (J) | 342.11 | 353.05 | 357.62 | -1.3% |
| Energy/Inference (mJ) | 57.018 | 58.841 | 59.604 | -1.3% |
| FPS/Watt | 17.80 | 17.24 | 17.03 | -1.3% |
| Violation Rate (%) | 4.83 | 2.80 | 3.30 | +17.9% |

**Adaptive Statistics:**
- Mode Switches: 23
- Low Power Time: 1.0%
- Avg Switch Time: 25.2 ms

## Variable Workload

| Metric | Static 7W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 9.06 | 10.94 | 10.88 | -0.5% |
| Avg Power (W) | 5.75 | 6.11 | 6.15 | -0.7% |
| Total Energy (J) | 371.12 | 372.00 | 374.57 | -0.7% |
| Energy/Inference (mJ) | 49.483 | 49.600 | 49.943 | -0.7% |
| FPS/Watt | 21.74 | 20.46 | 20.32 | -0.7% |
| Violation Rate (%) | 3.63 | 13.77 | 12.77 | -7.3% |

**Adaptive Statistics:**
- Mode Switches: 23
- Low Power Time: 0.9%
- Avg Switch Time: 23.9 ms

## Periodic Workload

| Metric | Static 7W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 11.19 | 10.74 | 10.55 | -1.8% |
| Avg Power (W) | 5.26 | 5.33 | 5.34 | -0.2% |
| Total Energy (J) | 263.04 | 266.21 | 266.84 | -0.2% |
| Energy/Inference (mJ) | 146.135 | 147.894 | 148.246 | -0.2% |
| FPS/Watt | 5.70 | 5.63 | 5.62 | -0.2% |
| Violation Rate (%) | 8.56 | 6.39 | 5.61 | -12.2% |

**Adaptive Statistics:**
- Mode Switches: 7
- Low Power Time: 0.4%
- Avg Switch Time: 22.6 ms

