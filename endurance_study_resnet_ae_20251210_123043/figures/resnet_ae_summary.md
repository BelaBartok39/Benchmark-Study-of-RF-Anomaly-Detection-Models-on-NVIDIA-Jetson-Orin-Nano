# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 10.59 | 10.56 | 10.59 | 10.60 | +0.1% |
| Avg Power (W) | 5.32 | 5.30 | 5.40 | 5.35 | +0.9% |
| Total Energy (J) | 19166.18 | 19098.78 | 19451.05 | 19280.33 | +0.9% |
| Energy/Inference (mJ) | 365.070 | 363.786 | 370.496 | 367.244 | +0.9% |
| FPS/Watt | 2.74 | 2.75 | 2.70 | 2.72 | +0.9% |
| Violation Rate (%) | 9.86 | 9.64 | 9.41 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 248
- Low Power Time (15W): 76.3%
- Medium Power Time (25W): 23.7%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 23.0 ms

## Periodic Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 9.04 | 9.02 | 9.02 | 9.06 | +0.4% |
| Avg Power (W) | 5.27 | 5.23 | 5.29 | 5.23 | +1.0% |
| Total Energy (J) | 18198.35 | 18072.24 | 18273.28 | 18088.36 | +1.0% |
| Energy/Inference (mJ) | 1516.529 | 1506.020 | 1522.773 | 1507.363 | +1.0% |
| FPS/Watt | 0.63 | 0.64 | 0.63 | 0.64 | +1.0% |
| Violation Rate (%) | 1.01 | 0.91 | 0.84 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 7
- Low Power Time (15W): 99.1%
- Medium Power Time (25W): 0.9%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 22.2 ms

