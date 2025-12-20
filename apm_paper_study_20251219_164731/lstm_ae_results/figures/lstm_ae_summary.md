# Adaptive Power Management Summary

Comparison of Static MAXN (performance baseline) vs Adaptive Power Management

## Bursty Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 14.25 | 14.17 | -0.5% |
| Avg Power (W) | 5.07 | 5.08 | -0.0% |
| Total Energy (J) | 9138.12 | 9139.64 | -0.0% |
| Energy/Inference (mJ) | 276.913 | 276.959 | -0.0% |
| FPS/Watt | 3.61 | 3.61 | -0.0% |
| Violation Rate (%) | 51.50 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 2
- Low Power Time (15W): 0.0%
- Medium Power Time (25W): 100.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 21.1 ms

## Periodic Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 8.87 | 8.88 | +0.1% |
| Avg Power (W) | 5.03 | 5.02 | +0.2% |
| Total Energy (J) | 8700.86 | 8683.34 | +0.2% |
| Energy/Inference (mJ) | 725.072 | 723.612 | +0.2% |
| FPS/Watt | 1.33 | 1.33 | +0.2% |
| Violation Rate (%) | 0.19 | 0.26 | +34.8% |

**Adaptive Statistics:**
- Mode Switches: 49
- Low Power Time (15W): 53.1%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 46.9%
- Avg Switch Time: 22.0 ms

## Continuous Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 6.74 | 6.75 | +0.1% |
| Avg Power (W) | 5.70 | 5.70 | +0.0% |
| Total Energy (J) | 10258.05 | 10258.27 | -0.0% |
| Energy/Inference (mJ) | 56.989 | 56.990 | -0.0% |
| FPS/Watt | 17.55 | 17.55 | +0.0% |
| Violation Rate (%) | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 6
- Low Power Time (15W): 0.0%
- Medium Power Time (25W): 100.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 22.0 ms

## Variable Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 10.88 | 10.83 | -0.4% |
| Avg Power (W) | 5.99 | 6.01 | -0.4% |
| Total Energy (J) | 10788.91 | 10829.81 | -0.4% |
| Energy/Inference (mJ) | 49.718 | 49.907 | -0.4% |
| FPS/Watt | 20.12 | 20.04 | -0.4% |
| Violation Rate (%) | 11.07 | 10.88 | -1.7% |

**Adaptive Statistics:**
- Mode Switches: 2
- Low Power Time (15W): 0.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 100.0%
- Avg Switch Time: 21.1 ms

