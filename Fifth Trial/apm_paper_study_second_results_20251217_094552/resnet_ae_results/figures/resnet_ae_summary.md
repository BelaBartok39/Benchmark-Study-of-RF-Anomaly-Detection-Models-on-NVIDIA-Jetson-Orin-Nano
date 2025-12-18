# Adaptive Power Management Summary

Comparison of Static MAXN (performance baseline) vs Adaptive Power Management

## Bursty Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 9.45 | 9.41 | -0.5% |
| Avg Power (W) | 5.07 | 5.07 | +0.0% |
| Total Energy (J) | 9129.68 | 9127.70 | +0.0% |
| Energy/Inference (mJ) | 276.657 | 276.597 | +0.0% |
| FPS/Watt | 3.62 | 3.62 | +0.0% |
| Violation Rate (%) | 47.96 | 48.72 | +1.6% |

**Adaptive Statistics:**
- Mode Switches: 2
- Low Power Time (15W): 0.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 100.0%
- Avg Switch Time: 20.5 ms

## Periodic Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 7.50 | 7.46 | -0.5% |
| Avg Power (W) | 5.01 | 4.99 | +0.5% |
| Total Energy (J) | 8670.32 | 8629.02 | +0.5% |
| Energy/Inference (mJ) | 722.527 | 719.085 | +0.5% |
| FPS/Watt | 1.33 | 1.34 | +0.5% |
| Violation Rate (%) | 1.77 | 1.60 | -9.4% |

**Adaptive Statistics:**
- Mode Switches: 50
- Low Power Time (15W): 53.1%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 46.9%
- Avg Switch Time: 21.8 ms

## Continuous Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 7.49 | 7.47 | -0.3% |
| Avg Power (W) | 5.46 | 5.47 | -0.1% |
| Total Energy (J) | 9837.96 | 9851.53 | -0.1% |
| Energy/Inference (mJ) | 54.655 | 54.731 | -0.1% |
| FPS/Watt | 18.30 | 18.28 | -0.1% |
| Violation Rate (%) | 0.59 | 0.61 | +4.7% |

**Adaptive Statistics:**
- Mode Switches: 3
- Low Power Time (15W): 0.1%
- Medium Power Time (25W): 0.1%
- High Power Time (MAXN): 99.8%
- Avg Switch Time: 21.6 ms

## Variable Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 9.38 | 9.24 | -1.5% |
| Avg Power (W) | 5.70 | 5.70 | +0.0% |
| Total Energy (J) | 10265.43 | 10265.56 | -0.0% |
| Energy/Inference (mJ) | 47.306 | 47.307 | -0.0% |
| FPS/Watt | 21.14 | 21.14 | +0.0% |
| Violation Rate (%) | 11.83 | 11.86 | +0.3% |

**Adaptive Statistics:**
- Mode Switches: 2
- Low Power Time (15W): 0.7%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 99.3%
- Avg Switch Time: 22.0 ms

