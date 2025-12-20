# Adaptive Power Management Summary

Comparison of Static MAXN (performance baseline) vs Adaptive Power Management

## Bursty Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 1.79 | 1.72 | -3.6% |
| Avg Power (W) | 5.03 | 5.03 | +0.0% |
| Total Energy (J) | 9048.32 | 9047.92 | +0.0% |
| Energy/Inference (mJ) | 274.191 | 274.179 | +0.0% |
| FPS/Watt | 3.65 | 3.65 | +0.0% |
| Violation Rate (%) | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 22.6 ms

## Periodic Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 1.65 | 1.64 | -0.4% |
| Avg Power (W) | 4.99 | 4.98 | +0.3% |
| Total Energy (J) | 8639.85 | 8611.00 | +0.3% |
| Energy/Inference (mJ) | 719.988 | 717.584 | +0.3% |
| FPS/Watt | 1.33 | 1.34 | +0.3% |
| Violation Rate (%) | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 22.7 ms

## Continuous Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 1.64 | 1.64 | +0.0% |
| Avg Power (W) | 5.17 | 5.17 | +0.0% |
| Total Energy (J) | 9312.08 | 9312.65 | -0.0% |
| Energy/Inference (mJ) | 51.734 | 51.737 | -0.0% |
| FPS/Watt | 19.33 | 19.33 | +0.0% |
| Violation Rate (%) | 0.00 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 21.0 ms

## Variable Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 1.62 | 1.64 | +0.7% |
| Avg Power (W) | 5.19 | 5.19 | -0.1% |
| Total Energy (J) | 9341.86 | 9353.51 | -0.1% |
| Energy/Inference (mJ) | 43.050 | 43.104 | -0.1% |
| FPS/Watt | 23.23 | 23.21 | -0.1% |
| Violation Rate (%) | 0.00 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 22.3 ms

