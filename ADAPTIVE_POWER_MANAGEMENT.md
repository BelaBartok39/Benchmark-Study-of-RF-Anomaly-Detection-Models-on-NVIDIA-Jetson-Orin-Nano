# Adaptive Power Management for RF Anomaly Detection

**Extension to:** "A Benchmark Study of RF Anomaly Detection Models on NVIDIA Jetson Orin Nano"

## Overview

This extension addresses a key limitation of the original benchmark study: all models were evaluated using the MAXN SUPER power mode, which maximizes performance but can lead to excessive power consumption under light workloads and wasted energy.

We propose and evaluate a **simple yet effective adaptive power adjustment technique** based on real-time latency feedback that balances performance and energy efficiency for RF anomaly detection workloads.

## Motivation

The original benchmark demonstrated that all models achieve excellent performance in MAXN SUPER mode:
- Latency < 10ms (well below real-time requirements)
- Power consumption: 4-7W
- High throughput (>100 FPS for most models)

However, real-world RF spectrum monitoring exhibits **variable workload patterns**:
- **Bursty traffic**: Periods of high activity followed by idle/low activity
- **Continuous monitoring**: Sustained high-throughput processing
- **Variable complexity**: Different models triggered based on detection results

Running continuously in MAXN SUPER mode wastes energy during light workloads, while operating in low power mode (7W) may miss latency targets during bursts.

## Proposed Solution: Adaptive Power Management

### Algorithm

Our adaptive power management system implements a latency-driven mode switching strategy with hysteresis:

```
1. Initialize in LOW_POWER mode (15W)
2. For each inference:
   a. Measure inference latency
   b. If latency > threshold:
      - Switch to HIGH_POWER mode (MAXN SUPER)
   c. If latency < threshold for X consecutive seconds:
      - Switch to LOW_POWER mode (15W)
3. Track mode switches, energy consumption, and performance
```

**Note**: JetPack 6.1 supports three power modes: 15W (mode 0), 25W (mode 1), and MAXN SUPER (mode 2). The 7W mode requires a reboot and cannot be dynamically switched, so we use 15W as the low power mode.

### Key Parameters

- **Latency Threshold (T_lat)**: Maximum acceptable latency (default: 10ms)
- **Hysteresis Time (T_hyst)**: Time latency must remain below threshold before switching to low power (default: 5s)

These parameters can be tuned based on application requirements:
- **Strict latency requirements**: Lower T_lat, shorter T_hyst
- **Energy optimization**: Higher T_lat, longer T_hyst
- **Balanced approach**: Moderate values for both

### Implementation Details

The implementation consists of four main components:

1. **Adaptive Power Manager** (`src/adaptive_power_manager.py`)
   - Tracks latency history and makes switching decisions
   - Interfaces with Jetson's `nvpmodel` for power mode control
   - Implements hysteresis mechanism to prevent thrashing
   - Records switching overhead and statistics

2. **Workload Generator** (`src/workload_generator.py`)
   - Simulates realistic RF monitoring scenarios:
     - **Bursty**: Spectrum scanning with intermittent activity
     - **Continuous**: Real-time monitoring of busy bands
     - **Variable**: Multi-model pipelines with changing complexity
     - **Periodic**: Scheduled scanning operations
     - **Random**: Poisson arrival process

3. **Adaptive Benchmark** (`src/adaptive_benchmark.py`)
   - Compares three power management strategies:
     - Static Low Power (7W) - maximum energy efficiency
     - Static High Power (MAXN SUPER) - maximum performance
     - Adaptive - dynamic switching
   - Measures comprehensive metrics:
     - Latency (average, P95, P99, max)
     - Energy consumption (total, per-inference)
     - Throughput and efficiency (FPS/Watt)
     - Mode switching overhead
     - Violation rate (latency exceeding threshold)

4. **Visualization Tools** (`src/visualize_adaptive_results.py`)
   - Energy-latency Pareto frontier
   - Latency timeline with power mode transitions
   - Cross-workload comparison charts
   - Summary tables with improvement metrics

## Experimental Methodology

### Phase 1: Mode Switching Overhead Characterization

**Objective**: Measure the cost of transitioning between power modes

**Experiments**:
1. Measure switching latency (15W → MAXN and MAXN → 15W)
2. Test with multiple trials (20+) for statistical significance
3. Allow stabilization time (2s) after each switch
4. Measure impact on inference performance during transitions

**Expected Results**:
- Switching latency: 200-1000ms (to be measured)
- Performance impact during transition
- Transient behavior characterization

**Script**: `src/characterize_switching_overhead.py`

```bash
python src/characterize_switching_overhead.py \
    --trials 20 \
    --stabilization-time 2.0 \
    --performance-test \
    --output-dir switching_overhead_results
```

### Phase 2: Workload Pattern Evaluation

**Objective**: Evaluate adaptive power management across realistic workload patterns

**Workload Patterns**:
1. **Bursty** (5s bursts, 10s idle)
   - Simulates: Spectrum scanning, intermittent signal detection
   - Expected: High energy savings with adaptive approach

2. **Continuous** (sustained high throughput)
   - Simulates: Real-time monitoring of busy frequency bands
   - Expected: Similar to static MAXN (minimal switching)

3. **Variable** (changing complexity levels)
   - Simulates: Multi-model pipelines, preliminary detection
   - Expected: Moderate energy savings, adaptive switching

4. **Periodic** (15s intervals)
   - Simulates: Scheduled scanning operations
   - Expected: Predictable switching patterns, good energy savings

**Baselines**:
- **Static 15W**: Energy efficiency baseline (may violate latency targets)
- **Static MAXN SUPER**: Performance baseline (wastes energy during idle)

**Per-Workload Experiments**:
```bash
# Run complete benchmark for a model
python src/adaptive_benchmark.py \
    --model ae \
    --model-path src/output/weights/ae_best.pth \
    --engine-path src/output/engines/ae.engine \
    --use-tensorrt \
    --workload all \
    --duration 60 \
    --latency-threshold 10.0 \
    --hysteresis-time 5.0 \
    --run-baselines \
    --output-dir adaptive_results/ae
```

### Phase 3: Parameter Sensitivity Analysis

**Objective**: Study impact of threshold and hysteresis parameter selection

**Experiments**:
- Sweep latency thresholds: [5ms, 10ms, 15ms, 20ms]
- Sweep hysteresis times: [2s, 5s, 10s, 15s]
- Test on bursty workload (most sensitive to parameters)

**Metrics**:
- Energy-delay product (combined metric)
- Violation rate vs energy savings trade-off
- Mode switch frequency

### Phase 4: Multi-Model Comparison

**Objective**: Evaluate adaptive power management across different model architectures

**Models**:
- AE (baseline autoencoder)
- AE-TRT (TensorRT optimized)
- LSTM-AE (highest latency)
- FF-TRT (lowest latency)

**Expected Findings**:
- Models with higher baseline latency (LSTM-AE) benefit less from adaptive approach
- Models with low latency (FF-TRT) have more opportunities for energy savings
- TensorRT optimization reduces need for high-power mode

## Metrics and Evaluation

### Primary Metrics

1. **Energy Efficiency**
   - Total energy consumption (Joules)
   - Energy per inference (mJ)
   - FPS/Watt (throughput per watt)

2. **Performance**
   - Average latency (ms)
   - P95 and P99 latency (ms)
   - Violation rate (% inferences exceeding threshold)

3. **Adaptive Behavior**
   - Total mode switches
   - Average switching time
   - Time in each power mode (%)
   - Switching frequency

### Derived Metrics

- **Energy Savings**: Compared to static MAXN baseline
- **Energy-Delay Product**: Combined efficiency metric (lower is better)
- **Latency Penalty**: P95 latency increase compared to static MAXN
- **Efficiency Ratio**: (Energy savings) / (Latency penalty)

## Expected Results

### Hypothesis

Adaptive power management will:
1. **Reduce energy consumption** by 20-40% compared to static MAXN for bursty workloads
2. **Maintain performance** with <5% latency increase (P95)
3. **Show diminishing returns** for continuous workloads (few switches)
4. **Demonstrate parameter sensitivity** - optimal values vary by workload

### Energy-Performance Trade-off

We expect to demonstrate a **Pareto frontier** showing:
- Static 15W: Best energy, worst latency (may violate constraints)
- Static MAXN: Best latency, worst energy
- **Adaptive: Near-optimal balance** - close to MAXN latency with significantly lower energy

### Workload-Specific Findings

| Workload | Expected Energy Savings | Expected Mode Switches | Key Finding |
|----------|------------------------|------------------------|-------------|
| Bursty | 30-50% | 10-20 per minute | Maximum benefit for adaptive |
| Continuous | 0-10% | 0-2 per minute | Minimal benefit (stays in MAXN) |
| Variable | 15-30% | 5-15 per minute | Moderate benefit, adapts to changes |
| Periodic | 25-40% | 4-8 per minute | Predictable energy savings |

## Limitations and Future Work

### Current Limitations

1. **Switching Overhead**
   - Mode transitions take 200-1000ms
   - Energy cost of switching not explicitly modeled
   - Frequent switches may reduce net benefit

2. **Simple Decision Logic**
   - Reactive (not predictive)
   - Binary power modes only (no intermediate levels)
   - Fixed hysteresis - could be adaptive

3. **Workload Assumptions**
   - Synthetic workload patterns
   - May not capture all real-world scenarios
   - No consideration of multi-model pipelines

4. **Real-Time Guarantees**
   - Provides soft real-time behavior (average case)
   - No worst-case latency guarantees
   - Critical applications may require static MAXN

### Future Directions

1. **Predictive Switching**
   - Use short-term history to anticipate load changes
   - Machine learning for workload prediction
   - Proactive mode switching before latency violations

2. **Multi-Level Power Management**
   - Utilize intermediate power modes (10W, 15W)
   - Finer-grained control for better efficiency
   - Dynamic voltage and frequency scaling (DVFS)

3. **Model-Aware Policies**
   - Different switching policies per model
   - Ensemble approaches with staged inference
   - Power-aware model selection

4. **Temperature-Aware Switching**
   - Consider thermal constraints
   - Prevent thermal throttling through proactive switching
   - Extended operation in constrained environments

5. **Real-World Validation**
   - Test with actual RF spectrum traces
   - Deploy in spectrum monitoring systems
   - Long-term energy and reliability studies

## Quick Start

### 1. Run Mode Switching Characterization

```bash
# Characterize switching overhead (run first)
python src/characterize_switching_overhead.py \
    --trials 20 \
    --performance-test \
    --output-dir switching_overhead_results
```

### 2. Run Adaptive Benchmark

```bash
# Benchmark with adaptive power management
python src/adaptive_benchmark.py \
    --model ae \
    --model-path src/output/weights/ae_best.pth \
    --workload bursty \
    --duration 60 \
    --latency-threshold 10.0 \
    --hysteresis-time 5.0 \
    --run-baselines \
    --output-dir adaptive_results
```

### 3. Visualize Results

```bash
# Generate figures and summary
python src/visualize_adaptive_results.py \
    --results-dir adaptive_results \
    --model ae \
    --workloads bursty continuous variable periodic \
    --output-dir figures
```

## Repository Structure (New Files)

```
src/
├── adaptive_power_manager.py          # Core adaptive power management logic
├── workload_generator.py              # Realistic workload pattern generator
├── adaptive_benchmark.py              # Comprehensive benchmarking script
├── characterize_switching_overhead.py # Mode switching characterization
└── visualize_adaptive_results.py     # Visualization and analysis tools

adaptive_results/                      # Experimental results
├── ae_adaptive_bursty_results.json
├── ae_static_low_bursty_results.json
├── ae_static_high_bursty_results.json
└── ...

figures/                               # Publication-quality figures
├── ae_bursty_energy_latency.png
├── ae_bursty_timeline.png
├── ae_efficiency_comparison.png
└── ae_summary.md

ADAPTIVE_POWER_MANAGEMENT.md          # This documentation
```

## Citation

If you use this adaptive power management extension in your research, please cite:

```bibtex
@inproceedings{redmond2025benchmark,
  title={A Benchmark Study of RF Anomaly Detection Models on NVIDIA Jetson Orin Nano},
  author={Redmond, Nicholas D. and Ali, Mohd Hasan and Dasgupta, Dipankar and Won, Myounggyu},
  booktitle={IEEE Consumer Communications \& Networking Conference (CCNC)},
  year={2025},
  organization={IEEE}
}
```

## Contact

For questions about this extension:
- Nicholas D. Redmond - ndrdmond@memphis.edu
- Myounggyu Won - mwon@memphis.edu

## Acknowledgments

This work extends the original benchmark study and applies insights from the critical review document (`claude_context.md`) to address the energy efficiency limitations of static power mode operation.
