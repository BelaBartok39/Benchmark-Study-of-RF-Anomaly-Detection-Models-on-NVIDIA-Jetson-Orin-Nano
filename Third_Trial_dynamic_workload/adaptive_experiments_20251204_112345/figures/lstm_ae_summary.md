# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 52.94 | 37.90 | 39.86 | 42.80 | +7.4% |
| Avg Power (W) | 5.77 | 6.87 | 6.88 | 6.87 | +0.3% |
| Total Energy (J) | 943.92 | 693.53 | 701.23 | 724.27 | -3.3% |
| Energy/Inference (mJ) | 39.330 | 28.897 | 29.218 | 30.178 | -3.3% |
| FPS/Watt | 69.34 | 58.21 | 58.11 | 58.26 | +0.3% |
| Violation Rate (%) | 100.00 | 99.86 | 99.82 | 99.61 | -0.2% |

**Adaptive Statistics:**
- Mode Switches: 27
- Low Power Time (15W): 0.4%
- Avg Switch Time: 24.3 ms

## Continuous Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 56.36 | 40.93 | 42.40 | 43.62 | +2.9% |
| Avg Power (W) | 5.78 | 6.64 | 6.83 | 6.90 | -1.0% |
| Total Energy (J) | 2368.68 | 1673.32 | 1764.96 | 1820.43 | -3.1% |
| Energy/Inference (mJ) | 39.478 | 27.889 | 29.416 | 30.340 | -3.1% |
| FPS/Watt | 173.13 | 150.56 | 146.37 | 144.95 | -1.0% |
| Violation Rate (%) | 100.00 | 99.72 | 99.59 | 99.44 | -0.2% |

**Adaptive Statistics:**
- Mode Switches: 75
- Low Power Time (15W): 0.4%
- Avg Switch Time: 24.6 ms

## Variable Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 57.68 | 42.94 | 43.03 | 43.46 | +1.0% |
| Avg Power (W) | 5.73 | 6.93 | 6.91 | 6.92 | -0.2% |
| Total Energy (J) | 2942.80 | 2254.94 | 2254.88 | 2266.13 | -0.5% |
| Energy/Inference (mJ) | 39.237 | 30.066 | 30.065 | 30.215 | -0.5% |
| FPS/Watt | 218.21 | 180.49 | 181.01 | 180.65 | -0.2% |
| Violation Rate (%) | 100.00 | 99.57 | 99.51 | 99.60 | +0.1% |

**Adaptive Statistics:**
- Mode Switches: 77
- Low Power Time (15W): 0.3%
- Avg Switch Time: 24.8 ms

## Periodic Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 53.59 | 40.50 | 41.06 | 42.98 | +4.7% |
| Avg Power (W) | 5.79 | 6.90 | 6.89 | 6.88 | +0.1% |
| Total Energy (J) | 721.30 | 527.24 | 526.36 | 544.98 | -3.5% |
| Energy/Inference (mJ) | 40.072 | 29.291 | 29.242 | 30.276 | -3.5% |
| FPS/Watt | 51.80 | 43.47 | 43.54 | 43.58 | +0.1% |
| Violation Rate (%) | 100.00 | 99.83 | 99.67 | 99.53 | -0.1% |

**Adaptive Statistics:**
- Mode Switches: 19
- Low Power Time (15W): 0.4%
- Avg Switch Time: 24.7 ms

