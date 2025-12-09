# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 25.17 | 25.29 | 25.32 | 25.62 | +1.2% |
| Avg Power (W) | 5.66 | 5.65 | 5.67 | 5.69 | -0.4% |
| Total Energy (J) | 351.00 | 350.35 | 351.07 | 347.31 | +1.1% |
| Energy/Inference (mJ) | 29.250 | 29.196 | 29.256 | 28.942 | +1.1% |
| FPS/Watt | 35.31 | 35.38 | 35.30 | 35.17 | -0.4% |
| Violation Rate (%) | 0.12 | 0.15 | 0.15 | 0.23 | +55.6% |

**Adaptive Statistics:**
- Mode Switches: 3
- Low Power Time (15W): 0.3%
- Medium Power Time (25W): 0.2%
- High Power Time (MAXN): 99.6%
- Avg Switch Time: 23.4 ms

## Continuous Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 25.34 | 25.43 | 25.39 | 25.71 | +1.3% |
| Avg Power (W) | 5.73 | 5.72 | 5.75 | 5.74 | +0.2% |
| Total Energy (J) | 819.95 | 824.60 | 817.91 | 821.66 | -0.5% |
| Energy/Inference (mJ) | 27.332 | 27.487 | 27.264 | 27.389 | -0.5% |
| FPS/Watt | 87.30 | 87.38 | 86.98 | 87.13 | +0.2% |
| Violation Rate (%) | 0.00 | 0.00 | 0.00 | 0.02 | +500.0% |

**Adaptive Statistics:**
- Mode Switches: 10
- Low Power Time (15W): 89.2%
- Medium Power Time (25W): 4.5%
- High Power Time (MAXN): 6.3%
- Avg Switch Time: 23.2 ms

## Variable Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 25.11 | 25.18 | 24.95 | 25.11 | +0.6% |
| Avg Power (W) | 5.75 | 5.74 | 5.77 | 5.74 | +0.5% |
| Total Energy (J) | 1011.73 | 1015.36 | 999.60 | 1010.84 | -1.1% |
| Energy/Inference (mJ) | 26.979 | 27.076 | 26.656 | 26.956 | -1.1% |
| FPS/Watt | 108.72 | 108.89 | 108.33 | 108.84 | +0.5% |
| Violation Rate (%) | 0.00 | 0.01 | 0.01 | 0.01 | +100.0% |

**Adaptive Statistics:**
- Mode Switches: 4
- Low Power Time (15W): 96.6%
- Medium Power Time (25W): 1.7%
- High Power Time (MAXN): 1.7%
- Avg Switch Time: 21.6 ms

## Periodic Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 24.63 | 24.75 | 24.64 | 25.04 | +1.6% |
| Avg Power (W) | 5.47 | 5.46 | 5.47 | 5.42 | +1.0% |
| Total Energy (J) | 308.98 | 308.39 | 309.06 | 305.93 | +1.0% |
| Energy/Inference (mJ) | 34.331 | 34.265 | 34.341 | 33.993 | +1.0% |
| FPS/Watt | 27.41 | 27.47 | 27.40 | 27.69 | +1.1% |
| Violation Rate (%) | 0.00 | 0.00 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 8
- Low Power Time (15W): 11.9%
- Medium Power Time (25W): 88.1%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 22.2 ms

