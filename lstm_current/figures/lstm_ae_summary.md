# Adaptive Power Management Summary

## Bursty Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 14.27 | 14.31 | 14.25 | 14.36 | +0.8% |
| Avg Power (W) | 5.31 | 5.37 | 5.37 | 5.43 | -1.0% |
| Total Energy (J) | 318.59 | 321.94 | 322.08 | 325.39 | -1.0% |
| Energy/Inference (mJ) | 132.745 | 134.143 | 134.199 | 135.578 | -1.0% |
| FPS/Watt | 7.53 | 7.45 | 7.45 | 7.37 | -1.0% |
| Violation Rate (%) | 21.04 | 20.50 | 22.21 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 9
- Low Power Time (15W): 21.4%
- Avg Switch Time: 22.6 ms

## Continuous Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 9.05 | 7.08 | 7.10 | 8.09 | +14.0% |
| Avg Power (W) | 5.66 | 5.82 | 5.82 | 5.74 | +1.4% |
| Total Energy (J) | 344.57 | 354.16 | 354.17 | 349.17 | +1.4% |
| Energy/Inference (mJ) | 57.428 | 59.026 | 59.029 | 58.195 | +1.4% |
| FPS/Watt | 17.67 | 17.19 | 17.19 | 17.43 | +1.4% |
| Violation Rate (%) | 3.08 | 1.23 | 1.30 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 8
- Low Power Time (15W): 47.1%
- Avg Switch Time: 24.2 ms

## Variable Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 8.24 | 10.89 | 10.91 | 10.81 | -0.9% |
| Avg Power (W) | 5.78 | 6.19 | 6.20 | 6.03 | +2.8% |
| Total Energy (J) | 372.70 | 376.79 | 377.48 | 366.86 | +2.8% |
| Energy/Inference (mJ) | 49.693 | 50.239 | 50.330 | 48.915 | +2.8% |
| FPS/Watt | 21.64 | 20.20 | 20.16 | 20.75 | +2.9% |
| Violation Rate (%) | 2.68 | 14.21 | 14.69 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 7
- Low Power Time (15W): 40.3%
- Avg Switch Time: 24.3 ms

## Periodic Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 10.84 | 9.20 | 9.30 | 10.47 | +12.7% |
| Avg Power (W) | 5.35 | 5.36 | 5.36 | 5.34 | +0.4% |
| Total Energy (J) | 267.37 | 267.91 | 267.96 | 266.76 | +0.4% |
| Energy/Inference (mJ) | 148.537 | 148.836 | 148.865 | 148.200 | +0.4% |
| FPS/Watt | 5.61 | 5.60 | 5.60 | 5.62 | +0.4% |
| Violation Rate (%) | 7.00 | 4.28 | 4.33 | 0.00 | -100.0% |

**Adaptive Statistics:**
- Mode Switches: 5
- Low Power Time (15W): 0.3%
- Avg Switch Time: 22.7 ms

