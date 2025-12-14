# Adaptive Power Management Summary

Comparison of Static MAXN (performance baseline) vs Adaptive Power Management

## Bursty Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 3.75 | 3.73 | -0.4% |
| Avg Power (W) | 5.31 | 5.28 | +0.7% |
| Total Energy (J) | 9568.38 | 9501.15 | +0.7% |
| Energy/Inference (mJ) | 289.951 | 287.914 | +0.7% |
| FPS/Watt | 3.45 | 3.47 | +0.7% |
| Violation Rate (%) | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 4
- Low Power Time (15W): 96.7%
- Medium Power Time (25W): 1.7%
- High Power Time (MAXN): 1.7%
- Avg Switch Time: 22.4 ms

## Periodic Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 3.76 | 3.80 | +1.0% |
| Avg Power (W) | 5.29 | 5.22 | +1.3% |
| Total Energy (J) | 9149.15 | 9032.72 | +1.3% |
| Energy/Inference (mJ) | 762.429 | 752.727 | +1.3% |
| FPS/Watt | 1.26 | 1.28 | +1.3% |
| Violation Rate (%) | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 22.2 ms

## Continuous Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 3.72 | 3.77 | +1.4% |
| Avg Power (W) | 5.57 | 5.52 | +0.8% |
| Total Energy (J) | 10030.03 | 9946.60 | +0.8% |
| Energy/Inference (mJ) | 55.722 | 55.259 | +0.8% |
| FPS/Watt | 17.95 | 18.10 | +0.8% |
| Violation Rate (%) | 0.00 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 24.4 ms

## Variable Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 3.73 | 3.78 | +1.3% |
| Avg Power (W) | 5.62 | 5.60 | +0.5% |
| Total Energy (J) | 10123.07 | 10078.08 | +0.4% |
| Energy/Inference (mJ) | 46.650 | 46.443 | +0.4% |
| FPS/Watt | 21.44 | 21.54 | +0.5% |
| Violation Rate (%) | 0.00 | 0.00 | +0.0% |

**Adaptive Statistics:**
- Mode Switches: 6
- Low Power Time (15W): 95.0%
- Medium Power Time (25W): 3.3%
- High Power Time (MAXN): 1.7%
- Avg Switch Time: 22.7 ms

