# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 14.75 | 14.76 | 14.81 | 14.89 | +0.6% |
| Avg Power (W) | 5.26 | 5.23 | 5.28 | 5.27 | +0.1% |
| Total Energy (J) | 324.89 | 323.19 | 325.81 | 320.74 | +1.6% |
| Energy/Inference (mJ) | 45.123 | 44.887 | 45.251 | 44.547 | +1.6% |
| FPS/Watt | 22.81 | 22.93 | 22.74 | 22.77 | +0.1% |
| Violation Rate (%) | 0.00 | 0.01 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 7
- Low Power Time (15W): 81.6%
- Medium Power Time (25W): 18.4%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 22.4 ms

## Continuous Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 9.63 | 9.60 | 9.60 | 9.84 | +2.5% |
| Avg Power (W) | 6.21 | 6.13 | 6.22 | 6.23 | -0.2% |
| Total Energy (J) | 383.16 | 378.53 | 384.15 | 379.11 | +1.3% |
| Energy/Inference (mJ) | 21.286 | 21.029 | 21.342 | 21.061 | +1.3% |
| FPS/Watt | 48.34 | 48.93 | 48.22 | 48.14 | -0.2% |
| Violation Rate (%) | 0.00 | 0.00 | 0.00 | 0.01 | N/A |

**Adaptive Statistics:**
- Mode Switches: 6
- Low Power Time (15W): 85.1%
- Medium Power Time (25W): 9.9%
- High Power Time (MAXN): 5.0%
- Avg Switch Time: 21.2 ms

## Variable Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 8.70 | 9.47 | 7.73 | 8.93 | +15.5% |
| Avg Power (W) | 6.42 | 6.18 | 7.04 | 6.41 | +9.0% |
| Total Energy (J) | 442.64 | 460.44 | 434.47 | 447.77 | -3.1% |
| Energy/Inference (mJ) | 19.673 | 20.464 | 19.310 | 19.901 | -3.1% |
| FPS/Watt | 58.43 | 60.64 | 53.24 | 58.52 | +9.9% |
| Violation Rate (%) | 0.01 | 0.00 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 3
- Low Power Time (15W): 95.7%
- Medium Power Time (25W): 4.3%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 21.3 ms

## Periodic Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 10.79 | 10.16 | 10.73 | 11.03 | +2.7% |
| Avg Power (W) | 5.21 | 5.18 | 5.21 | 5.13 | +1.6% |
| Total Energy (J) | 264.74 | 263.21 | 264.95 | 260.72 | +1.6% |
| Energy/Inference (mJ) | 49.027 | 48.743 | 49.065 | 48.282 | +1.6% |
| FPS/Watt | 17.28 | 17.38 | 17.27 | 17.55 | +1.7% |
| Violation Rate (%) | 0.06 | 0.00 | 0.00 | 0.09 | N/A |

**Adaptive Statistics:**
- Mode Switches: 8
- Low Power Time (15W): 71.8%
- Medium Power Time (25W): 15.1%
- High Power Time (MAXN): 13.0%
- Avg Switch Time: 21.7 ms

