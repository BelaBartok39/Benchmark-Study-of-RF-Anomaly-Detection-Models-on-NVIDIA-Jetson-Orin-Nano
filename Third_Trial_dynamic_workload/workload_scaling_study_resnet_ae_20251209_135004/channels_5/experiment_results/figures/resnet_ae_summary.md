# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 24.99 | 24.85 | 24.73 | 25.45 | +2.9% |
| Avg Power (W) | 6.00 | 5.99 | 6.01 | 6.05 | -0.8% |
| Total Energy (J) | 371.97 | 371.10 | 372.21 | 369.67 | +0.7% |
| Energy/Inference (mJ) | 30.998 | 30.925 | 31.017 | 30.806 | +0.7% |
| FPS/Watt | 33.32 | 33.42 | 33.30 | 33.05 | -0.8% |
| Violation Rate (%) | 0.00 | 0.00 | 0.00 | 0.02 | N/A |

**Adaptive Statistics:**
- Mode Switches: 19
- Low Power Time (15W): 37.1%
- Medium Power Time (25W): 50.6%
- High Power Time (MAXN): 12.3%
- Avg Switch Time: 23.8 ms

## Continuous Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 24.72 | 24.90 | 24.78 | 25.57 | +3.2% |
| Avg Power (W) | 6.10 | 6.09 | 6.12 | 6.09 | +0.4% |
| Total Energy (J) | 845.88 | 849.28 | 842.41 | 894.51 | -6.2% |
| Energy/Inference (mJ) | 28.196 | 28.309 | 28.080 | 29.817 | -6.2% |
| FPS/Watt | 81.95 | 82.17 | 81.75 | 82.08 | +0.4% |
| Violation Rate (%) | 0.00 | 0.00 | 0.00 | 0.01 | N/A |

**Adaptive Statistics:**
- Mode Switches: 10
- Low Power Time (15W): 89.5%
- Medium Power Time (25W): 6.3%
- High Power Time (MAXN): 4.2%
- Avg Switch Time: 23.1 ms

## Variable Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 25.01 | 25.01 | 24.94 | 25.44 | +2.0% |
| Avg Power (W) | 6.10 | 6.08 | 6.11 | 6.09 | +0.3% |
| Total Energy (J) | 1062.31 | 1065.48 | 1059.32 | 1095.22 | -3.4% |
| Energy/Inference (mJ) | 28.328 | 28.413 | 28.249 | 29.206 | -3.4% |
| FPS/Watt | 102.51 | 102.72 | 102.27 | 102.55 | +0.3% |
| Violation Rate (%) | 0.01 | 0.00 | 0.00 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 21.7 ms

## Periodic Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 25.15 | 25.19 | 25.22 | 26.01 | +3.2% |
| Avg Power (W) | 5.86 | 5.85 | 5.85 | 5.85 | +0.2% |
| Total Energy (J) | 331.04 | 330.51 | 330.66 | 335.50 | -1.5% |
| Energy/Inference (mJ) | 36.782 | 36.723 | 36.740 | 37.278 | -1.5% |
| FPS/Watt | 25.60 | 25.64 | 25.62 | 25.66 | +0.2% |
| Violation Rate (%) | 0.02 | 0.03 | 0.00 | 0.01 | N/A |

**Adaptive Statistics:**
- Mode Switches: 9
- Low Power Time (15W): 6.8%
- Medium Power Time (25W): 69.5%
- High Power Time (MAXN): 23.7%
- Avg Switch Time: 23.2 ms

