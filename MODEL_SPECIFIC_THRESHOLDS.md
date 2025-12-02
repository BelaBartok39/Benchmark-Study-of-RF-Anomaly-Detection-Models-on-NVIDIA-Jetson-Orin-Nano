# Model-Specific Thresholds for Adaptive Power Management

## Quick Fix Implementation

Based on analysis of initial experimental results (see `ADAPTIVE_POWER_ANALYSIS.md`), we've implemented **model-specific latency thresholds and hysteresis times** to address the limitations of the fixed-threshold approach.

## The Problem

Initial experiments showed that a fixed 10ms threshold doesn't work for all models:
- **Fast models (AE)**: Never switch (latency ~2.5ms, always below 10ms)
- **Slow models (LSTM-AE)**: Never switch back (latency ~10ms, can't stay below threshold for 5s)
- Result: No energy savings for fast models, increased energy for slow models due to switching overhead

## The Solution

Model-specific thresholds based on empirical baseline performance:

| Model | P95 Latency (15W) | Threshold | Hysteresis | Rationale |
|-------|------------------|-----------|------------|-----------|
| ae | ~2.8ms | 15ms | 3s | Fast, stable - loose threshold allows burst handling |
| ff | ~3.0ms | 15ms | 3s | Fast feedforward - similar to AE |
| aae | ~4.5ms | 12ms | 4s | Medium-fast - moderate threshold |
| cnn_ae | ~6.0ms | 18ms | 5s | Medium complexity - balanced approach |
| resnet_ae | ~8.0ms | 20ms | 6s | Medium-slow - loose threshold |
| lstm_ae | ~12ms | 25ms | 8s | Slow, high variance - very loose threshold |

## Usage

### Method 1: Automatic Configuration (Recommended)

Use the `--use-model-defaults` flag to automatically select model-specific parameters:

```bash
# Run with automatic model-specific configuration
python src/adaptive_benchmark.py \
    --model lstm_ae \
    --model-path src/output/weights/lstm_ae_best.pth \
    --workload bursty \
    --use-model-defaults  # This flag enables model-specific thresholds
```

The system will automatically use:
- LSTM-AE: 25ms threshold, 8s hysteresis
- AE: 15ms threshold, 3s hysteresis
- etc.

### Method 2: Using Shell Script

The `run_adaptive_experiments.sh` script now supports model defaults:

```bash
# Usage: ./run_adaptive_experiments.sh MODEL [USE_TENSORRT] [USE_MODEL_DEFAULTS]

# Enable model defaults (default behavior)
./run_adaptive_experiments.sh lstm_ae false true

# Disable model defaults (use fixed 10ms threshold)
./run_adaptive_experiments.sh lstm_ae false false

# Shorthand (defaults to model-specific thresholds)
./run_adaptive_experiments.sh lstm_ae
```

### Method 3: Manual Override

You can still manually specify thresholds if you want to experiment:

```bash
# Custom threshold for testing
python src/adaptive_benchmark.py \
    --model lstm_ae \
    --model-path src/output/weights/lstm_ae_best.pth \
    --workload bursty \
    --latency-threshold 20.0 \
    --hysteresis-time 7.0
    # No --use-model-defaults flag
```

## Implementation Details

The model-specific thresholds are defined in `src/adaptive_power_manager.py`:

```python
MODEL_THRESHOLDS = {
    'ae': 15.0,
    'ff': 15.0,
    'aae': 12.0,
    'cnn_ae': 18.0,
    'resnet_ae': 20.0,
    'lstm_ae': 25.0
}

MODEL_HYSTERESIS = {
    'ae': 3.0,
    'ff': 3.0,
    'aae': 4.0,
    'cnn_ae': 5.0,
    'resnet_ae': 6.0,
    'lstm_ae': 8.0
}
```

When `use_model_defaults=True`, the `AdaptivePowerManager` automatically looks up the appropriate values.

## Expected Results After Fix

### For LSTM-AE (Previously: 98% MAXN, MORE energy than baseline)

**With 25ms threshold:**
- Should stay in 15W mode more often (target: 40-60% in 15W)
- Fewer mode switches (target: 3-8 switches per minute)
- Energy savings: 10-25% vs static MAXN for bursty/periodic workloads
- Still switches to MAXN when needed for latency-sensitive periods

### For AE (Previously: 100% 15W, no adaptive benefit)

**With 15ms threshold:**
- Still mostly in 15W (latency is very stable at ~2.5ms)
- May occasionally switch to MAXN for burst handling
- Energy: Similar to before, but better latency guarantees

### For Medium Models (CNN-AE, ResNet-AE)

**With balanced thresholds (18-20ms):**
- Expected to show the "sweet spot" behavior
- 20-40% energy savings
- Dynamic switching between power modes
- Validates the adaptive approach

## Re-Running Experiments

To generate new results with model-specific thresholds:

```bash
# Test LSTM-AE with optimized thresholds
./run_adaptive_experiments.sh lstm_ae false true

# Compare with old fixed-threshold results
# Old results are in lstm_ae_experiments_20251202_095317/

# Test all models
for model in ae ff aae cnn_ae resnet_ae lstm_ae; do
    ./run_adaptive_experiments.sh $model false true
    sleep 120  # Thermal cooldown between models
done
```

## Validation

After re-running experiments, check:

1. **Mode switching behavior**:
   ```bash
   cat NEW_RESULTS/figures/lstm_ae_summary.md
   ```
   Look for "Low Power Time: XX%" - should be >30% for bursty/periodic workloads

2. **Energy savings**:
   ```
   | Total Energy (J) | Static 15W | Static MAXN | Adaptive | Improvement |
   ```
   "Improvement" should now be positive (energy savings)

3. **Latency violations**:
   ```
   | Violation Rate (%) | ... | ... | Adaptive | ... |
   ```
   Should be similar to or better than static baselines

## Tuning Guidelines

If you want to fine-tune thresholds for your specific workload:

1. **Profile your model** in 15W mode:
   ```bash
   python src/adaptive_benchmark.py \
       --model YOUR_MODEL \
       --workload continuous \
       --duration 60 \
       # Observe P95 latency
   ```

2. **Set threshold** to P95 + 50%:
   ```
   Threshold = P95_latency * 1.5
   ```

3. **Set hysteresis** based on variance:
   - Low variance (stable latency): 2-4s
   - Medium variance: 4-6s
   - High variance: 6-10s

4. **Test and iterate**:
   ```bash
   python src/adaptive_benchmark.py \
       --model YOUR_MODEL \
       --latency-threshold YOUR_THRESHOLD \
       --hysteresis-time YOUR_HYSTERESIS \
       --workload bursty
   ```

## Next Steps

1. ✅ Model-specific thresholds implemented
2. 🔄 Re-run experiments with optimized configuration
3. 📊 Generate new visualizations with corrected labels
4. 📝 Update paper with findings:
   - Fixed-threshold limitations
   - Model-specific optimization results
   - "Sweet spot" identification

## Questions?

If the new thresholds still don't work well:
- Check that model is loaded correctly (right window_size, etc.)
- Verify power modes are actually switching (check `nvpmodel -q`)
- Profile actual latency in both 15W and MAXN modes
- Consider testing on other workload patterns
