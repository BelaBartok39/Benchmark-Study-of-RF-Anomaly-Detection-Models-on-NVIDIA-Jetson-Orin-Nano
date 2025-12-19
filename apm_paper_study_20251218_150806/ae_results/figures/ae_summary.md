# Adaptive Power Management Summary

Comparison of Static MAXN (performance baseline) vs Adaptive Power Management

## Bursty Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 2.55 | 2.53 | -0.9% |
| Avg Power (W) | 5.06 | 5.06 | +0.1% |
| Total Energy (J) | 9113.37 | 9106.70 | +0.1% |
| Energy/Inference (mJ) | 276.163 | 275.960 | +0.1% |
| FPS/Watt | 3.62 | 3.62 | +0.1% |
| Violation Rate (%) | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 22.8 ms

## Periodic Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 2.58 | 2.56 | -0.7% |
| Avg Power (W) | 5.03 | 5.00 | +0.6% |
| Total Energy (J) | 8704.15 | 8654.97 | +0.6% |
| Energy/Inference (mJ) | 725.346 | 721.247 | +0.6% |
| FPS/Watt | 1.33 | 1.33 | +0.6% |
| Violation Rate (%) | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 3
- Low Power Time (15W): 98.2%
- Medium Power Time (25W): 1.8%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 22.6 ms

## Continuous Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 2.46 | 2.47 | +0.3% |
| Avg Power (W) | 5.32 | 5.33 | -0.1% |
| Total Energy (J) | 9588.83 | 9591.23 | -0.0% |
| Energy/Inference (mJ) | 53.271 | 53.285 | -0.0% |
| FPS/Watt | 18.78 | 18.77 | -0.1% |
| Violation Rate (%) | 0.00 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 20.7 ms

## Variable Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 2.51 | 2.51 | -0.0% |
| Avg Power (W) | 5.37 | 5.38 | -0.1% |
| Total Energy (J) | 9677.80 | 9683.02 | -0.1% |
| Energy/Inference (mJ) | 44.598 | 44.622 | -0.1% |
| FPS/Watt | 22.43 | 22.41 | -0.1% |
| Violation Rate (%) | 0.00 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 22.4 ms

