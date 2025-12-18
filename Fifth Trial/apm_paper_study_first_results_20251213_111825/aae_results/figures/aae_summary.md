# Adaptive Power Management Summary

Comparison of Static MAXN (performance baseline) vs Adaptive Power Management

## Bursty Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 2.93 | 2.95 | +0.9% |
| Avg Power (W) | 5.28 | 5.24 | +0.8% |
| Total Energy (J) | 9513.23 | 9440.66 | +0.8% |
| Energy/Inference (mJ) | 288.280 | 286.081 | +0.8% |
| FPS/Watt | 3.47 | 3.50 | +0.8% |
| Violation Rate (%) | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 21.3 ms

## Periodic Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 2.88 | 2.84 | -1.4% |
| Avg Power (W) | 5.26 | 5.20 | +1.2% |
| Total Energy (J) | 9101.70 | 8995.33 | +1.2% |
| Energy/Inference (mJ) | 758.475 | 749.611 | +1.2% |
| FPS/Watt | 1.27 | 1.28 | +1.2% |
| Violation Rate (%) | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 21.9 ms

## Continuous Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 2.95 | 2.98 | +0.9% |
| Avg Power (W) | 5.42 | 5.38 | +0.8% |
| Total Energy (J) | 9761.26 | 9687.84 | +0.8% |
| Energy/Inference (mJ) | 54.229 | 53.821 | +0.8% |
| FPS/Watt | 18.44 | 18.59 | +0.8% |
| Violation Rate (%) | 0.00 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 3
- Low Power Time (15W): 98.3%
- Medium Power Time (25W): 1.7%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 22.3 ms

## Variable Workload

| Metric | Static MAXN | Adaptive | Improvement |
|--------|-------------|----------|-------------|
| P95 Latency (ms) | 2.98 | 2.99 | +0.5% |
| Avg Power (W) | 5.44 | 5.40 | +0.8% |
| Total Energy (J) | 9802.85 | 9729.32 | +0.8% |
| Energy/Inference (mJ) | 45.174 | 44.836 | +0.8% |
| FPS/Watt | 22.14 | 22.31 | +0.8% |
| Violation Rate (%) | 0.00 | 0.00 | +0.0% |

**Adaptive Statistics:**
- Mode Switches: 4
- Low Power Time (15W): 94.4%
- Medium Power Time (25W): 1.7%
- High Power Time (MAXN): 3.9%
- Avg Switch Time: 23.0 ms

