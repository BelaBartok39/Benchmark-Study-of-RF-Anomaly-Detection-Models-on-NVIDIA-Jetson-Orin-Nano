# Adaptive Power Management Summary

## Variable Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 35.95 | 21.97 | 22.19 | 22.26 | +0.3% |
| Avg Power (W) | 5.71 | 6.78 | 6.79 | 6.79 | +0.0% |
| Total Energy (J) | 472.40 | 338.83 | 345.47 | 351.66 | -1.8% |
| Energy/Inference (mJ) | 41.991 | 30.118 | 30.708 | 31.259 | -1.8% |
| FPS/Watt | 131.30 | 110.64 | 110.47 | 110.48 | +0.0% |
| Violation Rate (%) | 59.61 | 0.27 | 0.40 | 0.52 | +28.9% |

**Adaptive Statistics:**
- Mode Switches: 2
- Low Power Time (15W): 0.1%
- Avg Switch Time: 21.6 ms

