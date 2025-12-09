# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 52.04 | 53.08 | 51.89 | 54.09 | +4.2% |
| Avg Power (W) | 6.04 | 5.99 | 6.10 | 5.98 | +2.0% |
| Total Energy (J) | 660.27 | 670.96 | 649.72 | 680.94 | -4.8% |
| Energy/Inference (mJ) | 27.511 | 27.957 | 27.072 | 28.373 | -4.8% |
| FPS/Watt | 66.20 | 66.77 | 65.56 | 66.92 | +2.1% |
| Violation Rate (%) | 0.00 | 0.00 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 2
- Low Power Time (15W): 0.5%
- Medium Power Time (25W): 99.5%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 23.4 ms

## Continuous Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 52.48 | 53.17 | 51.85 | 54.35 | +4.8% |
| Avg Power (W) | 6.04 | 5.98 | 6.09 | 5.97 | +1.9% |
| Total Energy (J) | 1642.11 | 1665.46 | 1609.89 | 1679.45 | -4.3% |
| Energy/Inference (mJ) | 27.368 | 27.758 | 26.831 | 27.991 | -4.3% |
| FPS/Watt | 165.59 | 167.15 | 164.26 | 167.38 | +1.9% |
| Violation Rate (%) | 0.00 | 0.00 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 2
- Low Power Time (15W): 0.2%
- Medium Power Time (25W): 99.8%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 23.2 ms

## Variable Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 52.21 | 53.16 | 52.39 | 53.97 | +3.0% |
| Avg Power (W) | 6.05 | 6.00 | 6.09 | 6.01 | +1.3% |
| Total Energy (J) | 2032.80 | 2076.64 | 2024.44 | 2064.82 | -2.0% |
| Energy/Inference (mJ) | 27.104 | 27.689 | 26.993 | 27.531 | -2.0% |
| FPS/Watt | 206.67 | 208.50 | 205.36 | 208.05 | +1.3% |
| Violation Rate (%) | 0.00 | 0.00 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 21.3 ms

## Periodic Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 51.55 | 52.83 | 51.07 | 53.38 | +4.5% |
| Avg Power (W) | 6.05 | 5.98 | 6.10 | 6.00 | +1.7% |
| Total Energy (J) | 488.57 | 499.89 | 481.97 | 501.38 | -4.0% |
| Energy/Inference (mJ) | 27.143 | 27.772 | 26.776 | 27.855 | -4.0% |
| FPS/Watt | 49.62 | 50.16 | 49.16 | 50.01 | +1.7% |
| Violation Rate (%) | 0.00 | 0.00 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 21.5 ms

