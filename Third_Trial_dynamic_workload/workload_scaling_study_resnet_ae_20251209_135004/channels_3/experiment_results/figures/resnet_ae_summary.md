# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 14.77 | 14.79 | 14.74 | 14.99 | +1.7% |
| Avg Power (W) | 5.66 | 5.63 | 5.68 | 5.69 | -0.2% |
| Total Energy (J) | 349.68 | 347.77 | 345.46 | 346.19 | -0.2% |
| Energy/Inference (mJ) | 48.567 | 48.302 | 47.980 | 48.083 | -0.2% |
| FPS/Watt | 21.20 | 21.32 | 21.14 | 21.10 | -0.2% |
| Violation Rate (%) | 0.01 | 0.01 | 0.01 | 0.03 | +100.0% |

**Adaptive Statistics:**
- Mode Switches: 18
- Low Power Time (15W): 2.0%
- Medium Power Time (25W): 81.1%
- High Power Time (MAXN): 16.9%
- Avg Switch Time: 21.8 ms

## Continuous Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 9.69 | 9.81 | 9.64 | 12.60 | +30.7% |
| Avg Power (W) | 6.54 | 6.48 | 6.55 | 6.44 | +1.7% |
| Total Energy (J) | 403.72 | 400.21 | 404.53 | 438.83 | -8.5% |
| Energy/Inference (mJ) | 22.429 | 22.234 | 22.474 | 24.379 | -8.5% |
| FPS/Watt | 45.90 | 46.30 | 45.80 | 46.58 | +1.7% |
| Violation Rate (%) | 0.01 | 0.00 | 0.00 | 0.03 | N/A |

**Adaptive Statistics:**
- Mode Switches: 31
- Low Power Time (15W): 2.6%
- Medium Power Time (25W): 81.5%
- High Power Time (MAXN): 16.0%
- Avg Switch Time: 21.9 ms

## Variable Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 8.89 | 9.47 | 7.84 | 13.10 | +67.1% |
| Avg Power (W) | 6.73 | 6.52 | 7.40 | 6.40 | +13.5% |
| Total Energy (J) | 470.21 | 485.58 | 456.31 | 558.01 | -22.3% |
| Energy/Inference (mJ) | 20.898 | 21.581 | 20.280 | 24.801 | -22.3% |
| FPS/Watt | 55.74 | 57.51 | 50.70 | 58.63 | +15.6% |
| Violation Rate (%) | 0.00 | 0.00 | 0.00 | 0.01 | +200.0% |

**Adaptive Statistics:**
- Mode Switches: 44
- Low Power Time (15W): 4.9%
- Medium Power Time (25W): 84.1%
- High Power Time (MAXN): 11.0%
- Avg Switch Time: 22.6 ms

## Periodic Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 10.86 | 10.53 | 10.89 | 13.39 | +23.0% |
| Avg Power (W) | 5.61 | 5.58 | 5.61 | 5.57 | +0.7% |
| Total Energy (J) | 285.51 | 283.81 | 285.09 | 288.43 | -1.2% |
| Energy/Inference (mJ) | 52.872 | 52.558 | 52.794 | 53.413 | -1.2% |
| FPS/Watt | 16.03 | 16.13 | 16.05 | 16.16 | +0.7% |
| Violation Rate (%) | 0.02 | 0.00 | 0.02 | 0.02 | +0.0% |

**Adaptive Statistics:**
- Mode Switches: 16
- Low Power Time (15W): 41.7%
- Medium Power Time (25W): 52.4%
- High Power Time (MAXN): 5.9%
- Avg Switch Time: 22.9 ms

