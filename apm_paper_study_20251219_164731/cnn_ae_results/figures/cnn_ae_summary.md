# Adaptive Power Management Summary

Comparison of Static MAXN (performance baseline) vs Adaptive Power Management

## Bursty Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 4.77 | 4.80 | +0.6% |
| Avg Power (W) | 5.05 | 5.05 | +0.0% |
| Total Energy (J) | 9098.61 | 9098.42 | +0.0% |
| Energy/Inference (mJ) | 275.715 | 275.710 | +0.0% |
| FPS/Watt | 3.63 | 3.63 | +0.0% |
| Violation Rate (%) | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 12
- Low Power Time (15W): 89.7%
- Medium Power Time (25W): 10.3%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 22.5 ms

## Periodic Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 4.70 | 4.68 | -0.6% |
| Avg Power (W) | 5.01 | 4.99 | +0.4% |
| Total Energy (J) | 8665.41 | 8628.15 | +0.4% |
| Energy/Inference (mJ) | 722.117 | 719.013 | +0.4% |
| FPS/Watt | 1.33 | 1.34 | +0.4% |
| Violation Rate (%) | 0.00 | 0.01 | N/A |

**Adaptive Statistics:**
- Mode Switches: 7
- Low Power Time (15W): 94.2%
- Medium Power Time (25W): 4.0%
- High Power Time (MAXN): 1.9%
- Avg Switch Time: 22.1 ms

## Continuous Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 4.67 | 4.70 | +0.7% |
| Avg Power (W) | 5.39 | 5.40 | -0.0% |
| Total Energy (J) | 9709.80 | 9715.99 | -0.1% |
| Energy/Inference (mJ) | 53.943 | 53.978 | -0.1% |
| FPS/Watt | 18.54 | 18.53 | -0.0% |
| Violation Rate (%) | 0.00 | 0.00 | +0.0% |

**Adaptive Statistics:**
- Mode Switches: 37
- Low Power Time (15W): 9.1%
- Medium Power Time (25W): 65.8%
- High Power Time (MAXN): 25.1%
- Avg Switch Time: 22.0 ms

## Variable Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 4.69 | 4.66 | -0.7% |
| Avg Power (W) | 5.47 | 5.47 | -0.1% |
| Total Energy (J) | 9841.13 | 9850.86 | -0.1% |
| Energy/Inference (mJ) | 45.351 | 45.396 | -0.1% |
| FPS/Watt | 22.06 | 22.04 | -0.1% |
| Violation Rate (%) | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 42
- Low Power Time (15W): 18.9%
- Medium Power Time (25W): 61.7%
- High Power Time (MAXN): 19.4%
- Avg Switch Time: 22.3 ms

