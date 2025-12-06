# Adaptive Power Management Summary

## Periodic Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 27.27 | 28.16 | 26.43 | 27.88 | +5.5% |
| Avg Power (W) | 6.69 | 6.67 | 6.77 | 6.60 | +2.6% |
| Total Energy (J) | 379.48 | 377.79 | 383.44 | 373.93 | +2.5% |
| Energy/Inference (mJ) | 21.082 | 20.988 | 21.302 | 20.774 | +2.5% |
| FPS/Watt | 44.82 | 45.01 | 44.29 | 45.47 | +2.7% |
| Violation Rate (%) | 0.00 | 0.00 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 1
- Low Power Time (15W): 100.0%
- Medium Power Time (25W): 0.0%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 22.1 ms

