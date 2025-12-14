# Adaptive Power Management Summary

Comparison of Static MAXN (performance baseline) vs Adaptive Power Management

## Bursty Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 6.02 | 5.89 | -2.3% |
| Avg Power (W) | 5.32 | 5.28 | +0.8% |
| Total Energy (J) | 9584.05 | 9503.83 | +0.8% |
| Energy/Inference (mJ) | 290.426 | 287.995 | +0.8% |
| FPS/Watt | 3.44 | 3.47 | +0.9% |
| Violation Rate (%) | 0.01 | 0.00 | -66.7% |

**Adaptive Statistics:**
- Mode Switches: 21
- Low Power Time (15W): 81.9%
- Medium Power Time (25W): 16.4%
- High Power Time (MAXN): 1.7%
- Avg Switch Time: 22.2 ms

## Periodic Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 5.68 | 5.79 | +1.9% |
| Avg Power (W) | 5.26 | 5.20 | +1.1% |
| Total Energy (J) | 9104.84 | 9004.06 | +1.1% |
| Energy/Inference (mJ) | 758.737 | 750.338 | +1.1% |
| FPS/Watt | 1.27 | 1.28 | +1.1% |
| Violation Rate (%) | 0.03 | 0.01 | -66.7% |

**Adaptive Statistics:**
- Mode Switches: 9
- Low Power Time (15W): 92.4%
- Medium Power Time (25W): 5.6%
- High Power Time (MAXN): 2.0%
- Avg Switch Time: 22.1 ms

## Continuous Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 6.01 | 6.02 | +0.2% |
| Avg Power (W) | 5.67 | 5.65 | +0.4% |
| Total Energy (J) | 10215.72 | 10179.73 | +0.4% |
| Energy/Inference (mJ) | 56.754 | 56.554 | +0.4% |
| FPS/Watt | 17.63 | 17.69 | +0.4% |
| Violation Rate (%) | 0.03 | 0.02 | -19.2% |

**Adaptive Statistics:**
- Mode Switches: 34
- Low Power Time (15W): 4.5%
- Medium Power Time (25W): 35.2%
- High Power Time (MAXN): 60.3%
- Avg Switch Time: 22.9 ms

## Variable Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 5.53 | 5.54 | +0.1% |
| Avg Power (W) | 5.75 | 5.71 | +0.7% |
| Total Energy (J) | 10348.29 | 10280.39 | +0.7% |
| Energy/Inference (mJ) | 47.688 | 47.375 | +0.7% |
| FPS/Watt | 20.98 | 21.12 | +0.7% |
| Violation Rate (%) | 0.00 | 0.01 | +100.0% |

**Adaptive Statistics:**
- Mode Switches: 42
- Low Power Time (15W): 10.7%
- Medium Power Time (25W): 48.0%
- High Power Time (MAXN): 41.3%
- Avg Switch Time: 22.6 ms

