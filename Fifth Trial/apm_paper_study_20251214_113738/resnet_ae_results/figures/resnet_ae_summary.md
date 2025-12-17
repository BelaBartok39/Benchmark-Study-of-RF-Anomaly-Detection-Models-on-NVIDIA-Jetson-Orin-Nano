# Adaptive Power Management Summary

Comparison of Static MAXN (performance baseline) vs Adaptive Power Management

## Bursty Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 9.67 | 9.53 | -1.5% |
| Avg Power (W) | 5.09 | 5.07 | +0.5% |
| Total Energy (J) | 9165.04 | 9119.10 | +0.5% |
| Energy/Inference (mJ) | 277.728 | 276.336 | +0.5% |
| FPS/Watt | 3.60 | 3.62 | +0.5% |
| Violation Rate (%) | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 28.1 ms

## Periodic Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 7.57 | 7.46 | -1.5% |
| Avg Power (W) | 5.01 | 4.99 | +0.4% |
| Total Energy (J) | 8673.97 | 8641.43 | +0.4% |
| Energy/Inference (mJ) | 722.831 | 720.119 | +0.4% |
| FPS/Watt | 1.33 | 1.33 | +0.4% |
| Violation Rate (%) | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 21.2 ms

## Continuous Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 7.49 | 7.45 | -0.5% |
| Avg Power (W) | 5.48 | 5.48 | -0.0% |
| Total Energy (J) | 9863.56 | 9866.16 | -0.0% |
| Energy/Inference (mJ) | 54.798 | 54.812 | -0.0% |
| FPS/Watt | 18.25 | 18.25 | -0.0% |
| Violation Rate (%) | 0.00 | 0.00 | +0.0% |

**Adaptive Statistics:**
- Mode Switches: 6
- Low Power Time (15W): 95.0%
- Medium Power Time (25W): 3.3%
- High Power Time (MAXN): 1.7%
- Avg Switch Time: 21.9 ms

## Variable Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 9.44 | 9.42 | -0.3% |
| Avg Power (W) | 5.73 | 5.65 | +1.4% |
| Total Energy (J) | 10309.00 | 10166.75 | +1.4% |
| Energy/Inference (mJ) | 47.507 | 46.851 | +1.4% |
| FPS/Watt | 21.06 | 21.35 | +1.4% |
| Violation Rate (%) | 0.00 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 23.9 ms

