# Adaptive Power Management Summary

## Periodic Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 27.91 | 28.21 | 27.83 | 29.29 | +5.2% |
| Avg Power (W) | 6.63 | 6.59 | 6.64 | 6.52 | +1.8% |
| Total Energy (J) | 1969.12 | 1957.24 | 1971.21 | 1936.56 | +1.8% |
| Energy/Inference (mJ) | 21.879 | 21.747 | 21.902 | 21.517 | +1.8% |
| FPS/Watt | 45.23 | 45.51 | 45.17 | 46.01 | +1.9% |
| Violation Rate (%) | 0.00 | 0.00 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 40
- Low Power Time (15W): 11.9%
- Avg Switch Time: 24.2 ms

