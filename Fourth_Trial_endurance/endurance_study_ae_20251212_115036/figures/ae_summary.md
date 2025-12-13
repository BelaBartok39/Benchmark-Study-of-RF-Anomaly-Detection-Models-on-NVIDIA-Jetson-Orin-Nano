# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 2.54 | 2.57 | 2.53 | 2.55 | +0.6% |
| Avg Power (W) | 5.06 | 5.07 | 5.08 | 5.08 | -0.1% |
| Total Energy (J) | 18234.89 | 18250.80 | 18278.33 | 18287.62 | -0.1% |
| Energy/Inference (mJ) | 347.331 | 347.634 | 348.159 | 348.336 | -0.1% |
| FPS/Watt | 2.88 | 2.88 | 2.87 | 2.87 | -0.1% |
| Violation Rate (%) | 0.00 | 0.00 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 22.3 ms

## Continuous Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 2.53 | 2.52 | 2.52 | 2.55 | +0.9% |
| Avg Power (W) | 5.35 | 5.35 | 5.36 | 5.35 | +0.1% |
| Total Energy (J) | 19255.34 | 19248.14 | 19289.73 | 19263.44 | +0.1% |
| Energy/Inference (mJ) | 53.487 | 53.467 | 53.583 | 53.510 | +0.1% |
| FPS/Watt | 18.70 | 18.71 | 18.67 | 18.69 | +0.1% |
| Violation Rate (%) | 0.00 | 0.00 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 9
- Low Power Time (15W): 99.7%
- Medium Power Time (25W): 0.3%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 22.2 ms

## Periodic Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 2.51 | 2.50 | 2.49 | 2.48 | -0.4% |
| Avg Power (W) | 5.03 | 5.03 | 5.03 | 4.99 | +0.7% |
| Total Energy (J) | 17384.31 | 17368.56 | 17365.27 | 17250.40 | +0.7% |
| Energy/Inference (mJ) | 1448.693 | 1447.380 | 1447.106 | 1437.533 | +0.7% |
| FPS/Watt | 0.66 | 0.66 | 0.66 | 0.67 | +0.7% |
| Violation Rate (%) | 0.01 | 0.00 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 23.4 ms

## Variable Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 2.57 | 2.57 | 2.59 | 2.74 | +6.1% |
| Avg Power (W) | 5.38 | 5.38 | 5.40 | 5.39 | +0.0% |
| Total Energy (J) | 19369.51 | 19371.12 | 19431.69 | 19424.40 | +0.0% |
| Energy/Inference (mJ) | 45.899 | 45.903 | 46.047 | 46.029 | +0.0% |
| FPS/Watt | 21.79 | 21.79 | 21.72 | 21.73 | +0.0% |
| Violation Rate (%) | 0.00 | 0.00 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 5
- Low Power Time (15W): 99.8%
- Medium Power Time (25W): 0.2%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 22.3 ms

