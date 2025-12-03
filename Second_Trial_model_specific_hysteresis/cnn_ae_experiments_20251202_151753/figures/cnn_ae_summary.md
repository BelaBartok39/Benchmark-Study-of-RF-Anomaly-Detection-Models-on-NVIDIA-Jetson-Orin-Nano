# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 6.85 | 5.85 | 7.65 | 7.58 | -0.8% |
| Avg Power (W) | 5.64 | 5.49 | 6.53 | 6.11 | +6.4% |
| Total Energy (J) | 338.33 | 329.25 | 391.58 | 366.85 | +6.3% |
| Energy/Inference (mJ) | 140.969 | 137.188 | 163.159 | 152.853 | +6.3% |
| FPS/Watt | 7.09 | 7.29 | 6.13 | 6.55 | +6.9% |
| Violation Rate (%) | 0.25 | 0.17 | 1.21 | 0.04 | -96.6% |

**Adaptive Statistics:**
- Mode Switches: 8
- Low Power Time (15W): 45.2%
- Avg Switch Time: 35.2 ms

## Continuous Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 5.04 | 5.08 | 5.11 | 4.92 | -3.8% |
| Avg Power (W) | 5.58 | 5.53 | 5.52 | 5.56 | -0.8% |
| Total Energy (J) | 339.92 | 331.75 | 336.01 | 338.70 | -0.8% |
| Energy/Inference (mJ) | 56.653 | 55.292 | 56.002 | 56.451 | -0.8% |
| FPS/Watt | 17.91 | 18.08 | 18.12 | 17.98 | -0.8% |
| Violation Rate (%) | 0.03 | 0.02 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time (15W): 100.0%
- Avg Switch Time: 0.0 ms

## Variable Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 4.81 | 4.74 | 4.86 | 4.85 | -0.0% |
| Avg Power (W) | 5.61 | 5.66 | 5.59 | 5.59 | -0.1% |
| Total Energy (J) | 341.75 | 344.75 | 340.14 | 340.37 | -0.1% |
| Energy/Inference (mJ) | 45.567 | 45.967 | 45.352 | 45.382 | -0.1% |
| FPS/Watt | 22.27 | 22.08 | 22.38 | 22.37 | -0.1% |
| Violation Rate (%) | 0.01 | 0.01 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time (15W): 100.0%
- Avg Switch Time: 0.0 ms

## Periodic Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 5.14 | 4.89 | 5.02 | 5.09 | +1.3% |
| Avg Power (W) | 5.23 | 5.30 | 5.24 | 5.23 | +0.3% |
| Total Energy (J) | 261.58 | 264.80 | 262.02 | 261.27 | +0.3% |
| Energy/Inference (mJ) | 145.325 | 147.112 | 145.565 | 145.148 | +0.3% |
| FPS/Watt | 5.73 | 5.66 | 5.72 | 5.74 | +0.3% |
| Violation Rate (%) | 0.06 | 0.00 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time (15W): 100.0%
- Avg Switch Time: 0.0 ms

