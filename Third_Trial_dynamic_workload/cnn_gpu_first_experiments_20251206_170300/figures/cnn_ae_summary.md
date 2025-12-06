# Adaptive Power Management Summary

## Periodic Workload

| Metric | Static 15W | Static 25W | Static MAXN | Adaptive | Improvement |
|--------|-----------|-----------|-------------|----------|-------------|
| P95 Latency (ms) | 27.45 | 28.82 | 26.91 | 25.80 | -4.1% |
| Avg Power (W) | 6.71 | 6.72 | 6.78 | 6.55 | +3.4% |
| Total Energy (J) | 379.91 | 380.86 | 383.82 | 383.16 | +0.2% |
| Energy/Inference (mJ) | 21.106 | 21.159 | 21.323 | 21.286 | +0.2% |
| FPS/Watt | 44.73 | 44.63 | 44.26 | 45.80 | +3.5% |
| Violation Rate (%) | 0.00 | 0.00 | 0.00 | 0.00 | N/A |

**Adaptive Statistics:**
- Mode Switches: 8
- Low Power Time (15W): 4.3%
- Medium Power Time (25W): 95.7%
- High Power Time (MAXN): 0.0%
- Avg Switch Time: 24.5 ms

