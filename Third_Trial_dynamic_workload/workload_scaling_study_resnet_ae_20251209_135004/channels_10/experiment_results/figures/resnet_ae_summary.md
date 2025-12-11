# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 51.81 | 52.94 | 51.38 | 53.80 | +4.7% |
| Avg Power (W) | 6.40 | 6.32 | 6.44 | 6.28 | +2.5% |
| Total Energy (J) | 688.22 | 708.59 | 680.41 | 744.57 | -9.4% |
| Energy/Inference (mJ) | 28.676 | 29.524 | 28.350 | 31.024 | -9.4% |
| FPS/Watt | 62.47 | 63.29 | 62.08 | 63.66 | +2.5% |
| Violation Rate (%) | 0.00 | 0.00 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 21.7 ms

## Continuous Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 53.09 | 54.14 | 52.93 | 54.23 | +2.5% |
| Avg Power (W) | 6.36 | 6.32 | 6.41 | 6.22 | +2.9% |
| Total Energy (J) | 1742.36 | 1782.62 | 1725.04 | 1961.07 | -13.7% |
| Energy/Inference (mJ) | 29.039 | 29.710 | 28.751 | 32.685 | -13.7% |
| FPS/Watt | 157.15 | 158.25 | 156.04 | 160.70 | +3.0% |
| Violation Rate (%) | 0.00 | 0.00 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 20.4 ms

## Variable Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 53.21 | 53.66 | 52.72 | 53.37 | +1.2% |
| Avg Power (W) | 6.36 | 6.31 | 6.41 | 6.22 | +2.9% |
| Total Energy (J) | 2185.79 | 2208.80 | 2142.20 | 2434.95 | -13.7% |
| Energy/Inference (mJ) | 29.144 | 29.451 | 28.563 | 32.466 | -13.7% |
| FPS/Watt | 196.55 | 198.12 | 195.10 | 200.87 | +3.0% |
| Violation Rate (%) | 0.00 | 0.00 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 20.0 ms

## Periodic Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 51.50 | 52.30 | 50.79 | 53.09 | +4.5% |
| Avg Power (W) | 6.38 | 6.32 | 6.44 | 6.24 | +3.1% |
| Total Energy (J) | 515.73 | 527.55 | 508.22 | 572.84 | -12.7% |
| Energy/Inference (mJ) | 28.651 | 29.308 | 28.235 | 31.824 | -12.7% |
| FPS/Watt | 47.01 | 47.50 | 46.61 | 48.11 | +3.2% |
| Violation Rate (%) | 0.00 | 0.00 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 21.9 ms

