# Adaptive Power Management Summary

Comparison of Static MAXN (performance baseline) vs Adaptive Power Management

## Bursty Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 14.43 | 14.49 | +0.4% |
| Avg Power (W) | 5.10 | 5.10 | +0.0% |
| Total Energy (J) | 18372.40 | 18368.95 | +0.0% |
| Energy/Inference (mJ) | 349.951 | 349.885 | +0.0% |
| FPS/Watt | 2.86 | 2.86 | +0.0% |
| Violation Rate (%) | 68.21 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 398
- Low Power Time (15W): 74.6%
- Medium Power Time (25W): 25.4%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 22.3 ms

## Continuous Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 7.60 | 8.87 | +16.7% |
| Avg Power (W) | 5.75 | 5.58 | +3.0% |
| Total Energy (J) | 20718.79 | 20106.11 | +3.0% |
| Energy/Inference (mJ) | 57.552 | 55.850 | +3.0% |
| FPS/Watt | 17.38 | 17.91 | +3.1% |
| Violation Rate (%) | 1.85 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 3
- Low Power Time (15W): 99.9%
- Medium Power Time (25W): 0.1%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 22.0 ms

## Periodic Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 9.49 | 10.82 | +14.0% |
| Avg Power (W) | 5.04 | 5.00 | +0.7% |
| Total Energy (J) | 17397.73 | 17269.36 | +0.7% |
| Energy/Inference (mJ) | 1449.811 | 1439.114 | +0.7% |
| FPS/Watt | 0.66 | 0.67 | +0.7% |
| Violation Rate (%) | 4.47 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 5
- Low Power Time (15W): 99.8%
- Medium Power Time (25W): 0.2%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 22.6 ms

## Variable Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 10.85 | 8.38 | -22.7% |
| Avg Power (W) | 6.02 | 5.61 | +6.8% |
| Total Energy (J) | 21689.20 | 21313.82 | +1.7% |
| Energy/Inference (mJ) | 51.396 | 50.507 | +1.7% |
| FPS/Watt | 19.46 | 20.88 | +7.3% |
| Violation Rate (%) | 11.29 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 3
- Low Power Time (15W): 99.9%
- Medium Power Time (25W): 0.1%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 22.1 ms

