# Updated Experiment Launcher - Multi-Channel & Batched Support

## ✅ What's New

The `run_adaptive_experiments.sh` script now supports multi-channel and batched inference through environment variables!

## 🚀 Quick Usage

### Default Mode (Backward Compatible)
```bash
./run_adaptive_experiments.sh lstm_ae false true
```
- Single-channel, single-sample
- Same behavior as before

### Batched Inference Mode
```bash
BATCH_SIZE=8 ./run_adaptive_experiments.sh lstm_ae false true
```
- Processes 8 samples per batch
- Better GPU utilization
- Lower per-sample latency (amortized)

### Multi-Channel Mode
```bash
NUM_CHANNELS=10 ./run_adaptive_experiments.sh lstm_ae false true
```
- Simulates 10 concurrent RF channels
- Realistic deployment scenario
- 10× more inferences in same duration

### Hybrid Mode (Recommended for Production Testing)
```bash
BATCH_SIZE=8 NUM_CHANNELS=10 ./run_adaptive_experiments.sh lstm_ae false true
```
- 10 concurrent channels
- Each channel processes batches of 8 samples
- Maximum GPU utilization
- Most realistic production scenario

## 📋 Complete Examples

### Example 1: Baseline Experiments (Your Current Workflow)
```bash
# Run standard single-channel experiments
./run_adaptive_experiments.sh lstm_ae false true
```
**Output:** `adaptive_experiments_YYYYMMDD_HHMMSS/`
- 4 workloads (bursty, continuous, variable, periodic)
- Single-channel, single-sample
- All baselines (15W, 25W, MAXN, Adaptive)

### Example 2: Multi-Channel Scalability Study
```bash
# Test 5 concurrent channels
NUM_CHANNELS=5 ./run_adaptive_experiments.sh lstm_ae false true
```
**Use case:** Show that adaptive power management scales to multi-channel deployments

### Example 3: Batching Optimization Study
```bash
# Test different batch sizes
BATCH_SIZE=4 ./run_adaptive_experiments.sh lstm_ae false true
BATCH_SIZE=8 ./run_adaptive_experiments.sh lstm_ae false true
BATCH_SIZE=16 ./run_adaptive_experiments.sh lstm_ae false true
```
**Use case:** Find optimal batch size for energy efficiency

### Example 4: Production Deployment Simulation
```bash
# 10-channel system with batched processing
BATCH_SIZE=8 NUM_CHANNELS=10 ./run_adaptive_experiments.sh lstm_ae false true
```
**Use case:** Validate adaptive power management under realistic production load

## 📊 What Gets Generated

The experiment will create a timestamped directory with:

```
adaptive_experiments_20250101_120000/
├── switching_overhead/
│   └── characterization_results.json
├── results/
│   ├── lstm_ae_static_low_bursty_results.json
│   ├── lstm_ae_static_medium_bursty_results.json
│   ├── lstm_ae_static_high_bursty_results.json
│   ├── lstm_ae_adaptive_bursty_results.json
│   └── ... (same for continuous, variable, periodic)
├── figures/
│   ├── lstm_ae_bursty_energy_latency.png
│   ├── lstm_ae_efficiency_comparison.png
│   └── lstm_ae_summary.md
└── EXPERIMENT_SUMMARY.md
```

The `EXPERIMENT_SUMMARY.md` will include your configuration:
```markdown
## Experiment Configuration

- **Power Management**: Three-tier adaptive (15W/25W/MAXN)
- **Threshold Mode**: Model-specific (auto-configured)
- **Batch Size**: 8 (batched inference)
- **Channels**: 10 (multi-channel concurrent)
- **Duration per Workload**: 60 seconds
- **Workload Patterns**: bursty continuous variable periodic
- **Test Samples**: 200
```

## 🔍 How Results Change

### Single-Channel (default)
```json
{
  "total_inferences": 6000,
  "batch_size": 1,
  "num_channels": 1,
  "avg_latency_ms": 8.5
}
```

### Batched (BATCH_SIZE=8)
```json
{
  "total_inferences": 6000,
  "batch_size": 8,
  "num_channels": 1,
  "avg_latency_ms": 3.2  // Lower due to batching
}
```

### Multi-Channel (NUM_CHANNELS=10)
```json
{
  "total_inferences": 60000,  // 10× more!
  "batch_size": 1,
  "num_channels": 10,
  "avg_latency_ms": 8.5
}
```

### Hybrid (BATCH_SIZE=8 NUM_CHANNELS=10)
```json
{
  "total_inferences": 60000,
  "batch_size": 8,
  "num_channels": 10,
  "avg_latency_ms": 3.2
}
```

## ⚙️ Parameters Reference

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `BATCH_SIZE` | 1 | Number of samples per batch |
| `NUM_CHANNELS` | 1 | Number of concurrent channels |

| Positional Argument | Default | Description |
|--------------------|---------|-------------|
| `$1` (MODEL) | `ae` | Model name (ae, aae, cnn_ae, lstm_ae, resnet_ae, ff) |
| `$2` (USE_TENSORRT) | `false` | Use TensorRT engine |
| `$3` (USE_MODEL_DEFAULTS) | `true` | Use model-specific thresholds |

## 💡 Recommendations

### For Paper Baseline
Use default single-channel mode to match original academic paper:
```bash
./run_adaptive_experiments.sh lstm_ae false true
```

### For Production Validation
Use multi-channel hybrid mode:
```bash
BATCH_SIZE=8 NUM_CHANNELS=10 ./run_adaptive_experiments.sh lstm_ae false true
```

### For Batch Size Optimization
Run multiple experiments:
```bash
for batch_size in 1 4 8 16; do
    BATCH_SIZE=$batch_size ./run_adaptive_experiments.sh lstm_ae false true
done
```

### For Channel Scaling Study
Run multiple experiments:
```bash
for channels in 1 5 10 20; do
    NUM_CHANNELS=$channels ./run_adaptive_experiments.sh lstm_ae false true
done
```

## 🐛 Troubleshooting

### GPU Out of Memory (OOM)
If you get CUDA OOM errors with multi-channel + batching:

**Solution 1:** Reduce batch size
```bash
BATCH_SIZE=4 NUM_CHANNELS=10 ./run_adaptive_experiments.sh lstm_ae
```

**Solution 2:** Reduce number of channels
```bash
BATCH_SIZE=8 NUM_CHANNELS=5 ./run_adaptive_experiments.sh lstm_ae
```

### Thermal Throttling
The script includes 60s cooldown between workloads. For multi-channel high-load experiments, you may need longer cooldowns.

Edit the script to increase cooldown:
```bash
# Line 133: Change from 60 to 120 seconds
sleep 120
```

## 📈 Expected Performance

### GPU Utilization
- Single-sample: ~30-40%
- Batched (8): ~60-80%
- Multi-channel (10): ~70-90%
- Hybrid (8×10): ~90-95%

### Throughput Scaling
- Single-channel: ~100 FPS
- 10-channel: ~1000 FPS (10× scaling)
- Batched: ~250 FPS (2.5× from batching)
- Hybrid (8×10): ~2500 FPS (25× total)

### Energy Efficiency
- Batching: 10-20% energy savings per inference
- Multi-channel: Sublinear energy growth (good!)
- Hybrid: Best energy per inference

## 🎯 Next Steps

1. **Run baseline:** `./run_adaptive_experiments.sh lstm_ae false true`
2. **Run multi-channel:** `NUM_CHANNELS=10 ./run_adaptive_experiments.sh lstm_ae false true`
3. **Compare results** using the generated figures
4. **Include in paper** as "realistic deployment scenario"

## 📚 Related Documentation

- `MULTI_CHANNEL_BATCHED_INFERENCE.md` - Technical deep dive
- `QUICK_START_MULTI_CHANNEL.md` - Quick reference guide
- `ADAPTIVE_POWER_MANAGEMENT.md` - Overall system documentation

---

**Note:** The script is fully backward compatible. Running without environment variables produces the same results as before the update.
