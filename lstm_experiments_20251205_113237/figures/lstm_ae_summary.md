# Adaptive Power Management Summary

## Periodic Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 60.55 | 36.95 | 35.71 | 42.15 | +18.0% |
| Avg Power (W) | 5.75 | 6.87 | 6.87 | 6.87 | +0.0% |
| Total Energy (J) | 715.92 | 518.27 | 518.24 | 543.30 | -4.8% |
| Energy/Inference (mJ) | 39.773 | 28.793 | 28.791 | 30.183 | -4.8% |
| FPS/Watt | 52.19 | 43.68 | 43.68 | 43.69 | +0.0% |
| Violation Rate (%) | 35.05 | 0.28 | 0.29 | 2.08 | +605.7% |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 0.1%
- Avg Switch Time: 19.7 ms

