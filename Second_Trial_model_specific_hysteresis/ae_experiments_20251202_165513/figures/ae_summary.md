# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 2.65 | 2.62 | 2.59 | 2.53 | -2.6% |
| Avg Power (W) | 5.27 | 5.19 | 5.19 | 5.26 | -1.3% |
| Total Energy (J) | 315.77 | 311.48 | 311.50 | 315.38 | -1.2% |
| Energy/Inference (mJ) | 131.572 | 129.783 | 129.791 | 131.410 | -1.2% |
| FPS/Watt | 7.60 | 7.70 | 7.70 | 7.60 | -1.2% |
| Violation Rate (%) | 0.04 | 0.00 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time (15W): 100.0%
- Avg Switch Time: 0.0 ms

## Continuous Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 2.57 | 2.58 | 2.57 | 2.55 | -1.0% |
| Avg Power (W) | 5.42 | 5.45 | 5.39 | 5.39 | +0.0% |
| Total Energy (J) | 329.83 | 331.84 | 327.98 | 327.87 | +0.0% |
| Energy/Inference (mJ) | 54.972 | 55.306 | 54.664 | 54.646 | +0.0% |
| FPS/Watt | 18.45 | 18.35 | 18.56 | 18.57 | +0.0% |
| Violation Rate (%) | 0.02 | 0.00 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time (15W): 100.0%
- Avg Switch Time: 0.0 ms

## Variable Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 2.76 | 2.76 | 2.76 | 2.77 | +0.4% |
| Avg Power (W) | 5.43 | 5.52 | 5.51 | 5.48 | +0.5% |
| Total Energy (J) | 330.58 | 336.06 | 335.27 | 333.59 | +0.5% |
| Energy/Inference (mJ) | 44.077 | 44.808 | 44.703 | 44.478 | +0.5% |
| FPS/Watt | 23.02 | 22.64 | 22.70 | 22.82 | +0.5% |
| Violation Rate (%) | 0.01 | 0.00 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time (15W): 100.0%
- Avg Switch Time: 0.0 ms

## Periodic Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 3.96 | 3.91 | 4.13 | 3.41 | -17.5% |
| Avg Power (W) | 5.43 | 5.48 | 5.72 | 5.44 | +5.0% |
| Total Energy (J) | 271.51 | 274.16 | 285.85 | 271.71 | +4.9% |
| Energy/Inference (mJ) | 150.837 | 152.312 | 158.806 | 150.950 | +4.9% |
| FPS/Watt | 5.52 | 5.47 | 5.25 | 5.52 | +5.2% |
| Violation Rate (%) | 0.06 | 0.00 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 0
- Low Power Time (15W): 100.0%
- Avg Switch Time: 0.0 ms

