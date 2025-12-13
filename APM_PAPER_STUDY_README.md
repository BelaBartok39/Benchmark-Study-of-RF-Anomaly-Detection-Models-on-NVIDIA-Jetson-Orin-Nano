# APM Paper Study Branch - Changes Summary

This branch (`apm_paper_study_results`) contains modifications to streamline the endurance study for the Adaptive Power Management (APM) paper.

## Key Changes

### 1. Modified Benchmark Script (`src/adaptive_benchmark.py`)

**Change**: Removed 15W and 25W static power mode baselines from experiments.

**Rationale**: For the APM paper, we only need to compare:
- **Static MAXN** (performance baseline)
- **Adaptive** (our proposed solution)

**Modified Lines**: 1106-1128
- Removed `static_low` (15W) baseline tests
- Removed `static_medium` (25W) baseline tests
- Kept only `static_high` (MAXN) baseline tests

### 2. Updated Visualization Script (`src/visualize_adaptive_results.py`)

**Changes**:
- `load_results()` - Only loads MAXN and Adaptive results
- `plot_energy_latency_tradeoff()` - Compares only MAXN vs Adaptive (2 data points)
- `plot_benchmark_summary()` - Bar charts show only 2 bars per workload (MAXN vs Adaptive)
- `create_summary_table()` - Table format changed to 3 columns: Metric | Static MAXN | Adaptive | Improvement

**Rationale**: Clean, focused visualizations showing direct comparison between baseline and adaptive approaches.

### 3. New Automated Script (`run_apm_paper_study.sh`)

**Purpose**: Fully automated endurance study that runs all models sequentially.

**Features**:
- Runs all 5 models: `ae`, `aae`, `cnn_ae`, `resnet_ae`, `lstm_ae`
- For each model:
  - Tests 4 workloads: bursty, periodic, continuous, variable
  - Runs 60-minute tests for MAXN static mode
  - Runs 60-minute tests for Adaptive mode
  - Auto-generates visualizations after completion
- Includes automatic thermal cooldowns
- Comprehensive logging to `study_log.txt`
- Estimated total runtime: ~8 hours

**Usage**:
```bash
./run_apm_paper_study.sh
```

**Output Structure**:
```
apm_paper_study_YYYYMMDD_HHMMSS/
├── study_log.txt
├── ae_results/
│   ├── results/              # JSON files
│   ├── figures/              # PNG visualizations
│   └── calibration/          # Calibration data
├── aae_results/
│   └── ...
├── cnn_ae_results/
│   └── ...
├── resnet_ae_results/
│   └── ...
└── lstm_ae_results/
    └── ...
```

## Comparison: Old vs New

### Old Workflow (main branch):
- Run `run_endurance_study.sh` manually for each model
- Tests 3 static modes (15W, 25W, MAXN) + Adaptive
- Manually run visualization script after each model
- Visualizations show 4 power modes

### New Workflow (this branch):
- Run `run_apm_paper_study.sh` once
- Tests 1 static mode (MAXN) + Adaptive
- Visualizations auto-generated after each model
- Visualizations show 2 power modes (cleaner, more focused)
- **50% reduction in test time** (2 modes vs 4 modes)

## Time Savings

**Per Model**:
- Old: 4 workloads × 4 modes × 60 min = 960 min (~16 hours)
- New: 4 workloads × 2 modes × 60 min = 480 min (~8 hours)
- **Savings: 8 hours per model**

**Total Study** (5 models):
- Old: ~80 hours
- New: ~40 hours
- **Total savings: 40 hours**

## Running the Study

1. **Transfer to Jetson Orin Nano**:
   ```bash
   git checkout apm_paper_study_results
   ```

2. **Execute automated study**:
   ```bash
   ./run_apm_paper_study.sh
   ```

3. **Walk away**: The script will run for ~8 hours and complete all tests + visualizations automatically.

4. **Collect results**: All results and visualizations will be in `apm_paper_study_YYYYMMDD_HHMMSS/`

## Visualization Changes

### Energy-Latency Trade-off Plot
- **Before**: 4 points (15W, 25W, MAXN, Adaptive)
- **After**: 2 points (MAXN, Adaptive) with error bars
- **Error bars**: ±1 standard deviation for both energy and latency
- **Title**: "MAXN vs Adaptive Power Management"
- **Annotation**: "Energy Savings vs MAXN: X.X%"

### Benchmark Summary (3-panel plot)
- **Before**: 4 bars per workload
- **After**: 2 bars per workload with error bars
- **Bar width**: Increased from 0.2 to 0.35 for better visibility
- **Error bars**: ±1 standard deviation from raw data
- **Mean markers**: Horizontal black lines showing mean values (when different from bar metric)
  - For latency: Shows mean latency (bar shows P95)
  - For energy/efficiency: Estimated variance

### Summary Table
- **Before**: 6 columns (Metric | 15W | 25W | MAXN | Adaptive | Improvement)
- **After**: 4 columns (Metric | MAXN | Adaptive | Improvement)
- **Improvement**: Calculated relative to MAXN baseline

## SLA and Threshold Configuration

### Auto-Calibration Approach (Used in This Study)

This study uses **automatic hardware-based calibration** to determine optimal thresholds for each model, eliminating experimenter bias and ensuring academic rigor.

**Why Auto-Calibration?**
- ✅ **Eliminates experimenter bias**: No hand-tuning or cherry-picking thresholds
- ✅ **Reproducible methodology**: Other researchers can replicate the process on their hardware
- ✅ **Hardware-adaptive**: Automatically adjusts to different devices/thermal conditions
- ✅ **Scientifically defensible**: Clear, documented algorithm for threshold selection
- ✅ **Demonstrates autonomy**: System works without manual intervention

**Calibration Process** (30 seconds per workload):

1. **Profile each power mode** (15W, 25W, MAXN):
   - Set power mode via `nvpmodel`
   - Run 100 inference samples with 10-sample warmup
   - Collect latency statistics: mean, median, P50, P75, P90, P95, P99
   - Use single-threaded or multi-threaded profiling (matches test configuration)

2. **Determine thresholds** using SLA-based strategy:
   ```
   Target SLA: 10.0ms (P95 latency)
   Safety Margin: 0.9 (90% of measured capacity)

   For each power mode:
     sustainable_throughput = 1000ms / (P95_latency * safety_margin)

   Thresholds are set to:
     - 15W → 25W: When 15W cannot sustain load within SLA
     - 25W → MAXN: When 25W cannot sustain load within SLA

   NOTE: Hysteresis is NOT auto-calibrated. User-specified hysteresis
         (--hysteresis-time) always takes precedence to allow workload-specific
         tuning (e.g., 60s for guard duty cycles to prevent thrashing)
   ```

3. **Apply calibrated thresholds**: Immediately used for the 60-minute endurance tests

4. **Document results**: Calibration data saved to `{model}_calibration_results.json`

**Example Calibration Output**:
```json
{
  "target_sla_ms": 10.0,
  "safety_margin": 0.9,
  "profiles": {
    "15W": {"mean_ms": 2.5, "p95_ms": 3.2},
    "25W": {"mean_ms": 2.3, "p95_ms": 2.9},
    "MAXN": {"mean_ms": 2.1, "p95_ms": 2.6}
  },
  "medium_threshold_ms": 8.5,
  "high_threshold_ms": 18.0,
  "hysteresis_time_s": 60.0
}
```

**Configuration Used**:
```bash
--auto-calibrate              # Enable automatic calibration
--target-sla 10.0             # Target P95 latency: 10ms
--hysteresis-time 60.0        # Override: 60s for endurance study
--sparsity-factor 10.0        # 10x longer idle times
--duration 3600               # 60 minutes per test
```

**Academic Justification**:
> "Power mode switching thresholds were automatically determined via hardware profiling with a target Service Level Agreement (SLA) of 10ms P95 latency and 90% safety margin. This approach eliminates manual tuning of performance thresholds and ensures the adaptive system operates autonomously across different model architectures. The 60-second hysteresis parameter was explicitly chosen to prevent mode thrashing in guard duty workloads with sparse, bursty activity patterns—avoiding energy waste from rapid power mode transitions."

### Three-Tier Power Management

**Power Modes**:
- **15W (Low Power)**: Default mode, most energy efficient
- **25W (Medium Power)**: Intermediate performance/efficiency
- **MAXN (High Power)**: Maximum performance

**Switching Logic**:
1. **Upshift (15W → 25W)**: When moving average latency > `medium_threshold_ms`
2. **Upshift (25W → MAXN)**: When moving average latency > `high_threshold_ms`
3. **Downshift**: When latency stays below threshold for `hysteresis_time_s` (60s)

**Hysteresis Rationale**:
- Prevents mode thrashing during temporary load fluctuations
- 60-second hysteresis chosen for endurance study to simulate realistic "guard duty" cycles
- Allows system to confirm stable low-load conditions before downshifting

## Notes

- All improvements (energy savings, efficiency gains) are calculated relative to **Static MAXN** as the performance baseline
- Latency should remain comparable or better with Adaptive vs MAXN
- Energy consumption should be significantly lower with Adaptive vs MAXN
- This comparison directly demonstrates the value of adaptive power management
- **SLA target**: 10.0ms P95 latency across all models and workloads
- **Threshold determination**: Automatic hardware-based calibration with 90% safety margin
