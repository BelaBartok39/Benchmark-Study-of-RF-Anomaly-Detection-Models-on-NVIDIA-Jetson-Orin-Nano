# Adaptive Power Management Summary

Comparison of Static MAXN (performance baseline) vs Adaptive Power Management

## Bursty Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 14.31 | 14.32 | +0.1% |
| Avg Power (W) | 5.07 | 5.07 | +0.0% |
| Total Energy (J) | 9131.59 | 9130.97 | +0.0% |
| Energy/Inference (mJ) | 276.715 | 276.696 | +0.0% |
| FPS/Watt | 3.61 | 3.62 | +0.0% |
| Violation Rate (%) | 51.66 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 2
- Low Power Time (15W): 0.0%
- Medium Power Time (25W): 100.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 20.8 ms

## Periodic Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 8.89 | 8.92 | +0.4% |
| Avg Power (W) | 5.02 | 4.99 | +0.5% |
| Total Energy (J) | 8687.62 | 8638.73 | +0.6% |
| Energy/Inference (mJ) | 723.969 | 719.894 | +0.6% |
| FPS/Watt | 1.33 | 1.34 | +0.5% |
| Violation Rate (%) | 3.01 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 48
- Low Power Time (15W): 53.1%
- Medium Power Time (25W): 46.9%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 21.9 ms

## Continuous Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 6.88 | 6.90 | +0.3% |
| Avg Power (W) | 5.69 | 5.69 | -0.1% |
| Total Energy (J) | 10242.32 | 10250.70 | -0.1% |
| Energy/Inference (mJ) | 56.902 | 56.948 | -0.1% |
| FPS/Watt | 17.57 | 17.56 | -0.1% |
| Violation Rate (%) | 0.48 | 0.64 | +33.6% |

**Adaptive Statistics:**
- Mode Switches: 2
- Low Power Time (15W): 0.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 100.0%
- Avg Switch Time: 20.3 ms

## Variable Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 10.85 | 10.91 | +0.6% |
| Avg Power (W) | 6.09 | 6.30 | -3.5% |
| Total Energy (J) | 10964.73 | 11343.60 | -3.5% |
| Energy/Inference (mJ) | 50.529 | 52.275 | -3.5% |
| FPS/Watt | 19.80 | 19.13 | -3.4% |
| Violation Rate (%) | 10.77 | 9.55 | -11.3% |

**Adaptive Statistics:**
- Mode Switches: 18
- Low Power Time (15W): 0.0%
- Medium Power Time (25W): 1.5%
- High Power Time (MAXN): 98.5%
- Avg Switch Time: 22.7 ms

