# Adaptive Power Management Summary

## Periodic Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 55.75 | 42.79 | 31.89 | 47.43 | +48.7% |
| Avg Power (W) | 5.79 | 6.54 | 6.90 | 6.60 | +4.4% |
| Total Energy (J) | 716.09 | 547.18 | 520.56 | 576.06 | -10.7% |
| Energy/Inference (mJ) | 39.783 | 30.399 | 28.920 | 32.004 | -10.7% |
| FPS/Watt | 51.79 | 45.86 | 43.48 | 45.47 | +4.6% |
| Violation Rate (%) | 0.14 | 0.00 | 0.00 | 0.07 | N/A |

**Adaptive Statistics:**
- Mode Switches: 31
- Low Power Time (15W): 2.0%
- Avg Switch Time: 24.7 ms

