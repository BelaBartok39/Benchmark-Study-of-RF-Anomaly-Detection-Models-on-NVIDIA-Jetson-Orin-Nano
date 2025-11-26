# Reproducing Paper Results

This document provides detailed instructions for reproducing the exact results reported in the paper **"A Benchmark Study of RF Anomaly Detection Models on NVIDIA Jetson Orin Nano"** presented at IEEE CCNC 2025.

## Paper Results Location

The original experimental results used in the paper are preserved in the `paper_results/` directory:

```
paper_results/
├── experiment_1/              # Initial benchmark run
├── experiment_2/              # Replicate benchmark run
├── experiment_3(sust)/        # Sustained inference - Run 1
├── experiment_4(sust)/        # Sustained inference - Run 2
├── experiment_5(sust)/        # Sustained inference - Run 3
├── experiment_6(sust)/        # Sustained inference - Run 4
├── experiment_7(sust)/        # Sustained inference - Run 5
├── experiment_8(sust)/        # Sustained inference - Run 6
├── experiment_9(sust)/        # Sustained inference - Run 7
├── experiment_10(sust)/       # Sustained inference - Run 8
├── experiment_11(sust)/       # Sustained inference - Run 9
├── experiment_12(sust)/       # Sustained inference - Run 10
├── sustained_results/         # Aggregated sustained results
└── sustained_experiment_details.md  # Methodology documentation
```

Each experiment directory contains:
- `weights/` - Trained model weights (.pth files)
- `engines/` - TensorRT engine files (.onnx, .trt files)
- `figures/` - Generated visualizations
- `benchmark_results.json` - Raw benchmark metrics
- `benchmark_results.csv` - Summary metrics table
- `sustained_measurements.json` - Time-series data (sustained experiments only)

## Hardware Configuration

### Required Hardware

1. **NVIDIA Jetson Orin Nano Developer Kit (8GB)**
   - Product: Jetson Orin Nano 8GB
   - Part Number: 945-13766-0000-000

2. **Power Supply**
   - 14V DC power adapter (included with developer kit)
   - Ensure stable power delivery for sustained benchmarks

3. **Storage**
   - 64GB+ microSD card (Class 10 or UHS-I recommended)
   - Or NVMe SSD for better I/O performance

4. **Cooling** (Optional but recommended)
   - Active cooling fan to prevent thermal throttling
   - Heatsink (included with developer kit)

### Software Configuration

1. **Operating System**
   - Jetson Linux 36.4.3 (or later)
   - Ubuntu 22.04 LTS base

2. **JetPack SDK**
   - JetPack 6.1 (used in paper)
   - CUDA 12.2+
   - cuDNN 8.9+
   - TensorRT 8.6+

3. **Power Mode**
   - **CRITICAL:** MAXN SUPER mode must be enabled
   - This provides up to 70% performance increase
   - Verify with: `sudo nvpmodel -q`
   - Set with: `sudo nvpmodel -m 0` (MAXN mode)

## Step-by-Step Reproduction

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/rf-anomaly-jetson-benchmark.git
cd rf-anomaly-jetson-benchmark

# Verify Jetson configuration
echo "Checking Jetson configuration..."
jetson_release  # Should show JetPack 6.1+

# Enable MAXN SUPER power mode
sudo nvpmodel -m 2
sudo jetson_clocks  # Maximize clock frequencies

# Verify power mode
sudo nvpmodel -q  # Should show "MAXN"

# Install Python dependencies
pip3 install -r requirements.txt
```

### 2. Download Datasets

The datasets used in the paper are hosted on Hugging Face:

```bash
# Option 1: Use the download script
./download_datasets.sh

# Option 2: Manual download
# Visit: https://huggingface.co/datasets/b4byn1cky/RF_Anomaly_Detection
# Download: clean_5g_dataset.h5 and jammed_5g_dataset.h5
# Place in repository root
```

Verify datasets:
```bash
ls -lh *.h5
# Should show:
# clean_5g_dataset.h5  (785M)
# jammed_5g_dataset.h5 (785M)
```

### 3. Training Models (Optional)

The paper used pre-trained models, but you can retrain them:

```bash
cd src

# Train all models (as done in the paper)
python train_all.py \
    --clean ../clean_5g_dataset.h5 \
    --jammed ../jammed_5g_dataset.h5 \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-3 \
    --out-dir output

# This will create:
# - output/weights/*.pth (model weights)
```

**Note:** The paper results used specific random seeds. For exact reproduction, use the pre-trained weights in `paper_results/*/weights/`.

### 4. Running Sustained Benchmark (Paper Configuration)

The key experiments in the paper used 300-second (5-minute) sustained inference tests.

#### Quick Test (30 seconds)

First, verify everything works with a quick test:

```bash
# From repository root
python run_sustained_benchmark.py test

# This runs a 30-second test with 1 model
# Expected completion time: ~2 minutes
```

#### Full Paper Benchmark (300 seconds)

**WARNING:** This takes significant time (~2+ hours for all models).

```bash
# Edit run_sustained_benchmark.py if needed to select specific models
# The paper tested: AE, AAE, CNN-AE, LSTM-AE, ResNet-AE, FF

python run_sustained_benchmark.py

# Or run comprehensive benchmark directly:
cd src
python comprehensive_benchmark.py \
    --models ae aae cnn_ae lstm_ae resnet_ae ff \
    --sustained \
    --sustained-duration 300 \
    --thermal-cooldown 60 \
    --moving-window 30 \
    --convert-tensorrt \
    --max-samples 2000 \
    --batch-size 1 \
    --output-dir ../my_results
```

### 5. Parameters Used in Paper

The exact configuration used in the paper:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sustained Duration | 300 seconds | 5 minutes continuous inference |
| Thermal Cooldown | 60 seconds | Between model evaluations |
| Moving Window | 30 seconds | For temporal analysis |
| Max Samples | 2,000 | 1,000 clean + 1,000 jammed |
| Batch Size | 1 | Single-sample inference |
| Power Sampling | 10 Hz | Power monitoring frequency |
| Repetitions | 10 | Statistical robustness |
| Power Mode | MAXN SUPER | Maximum performance |

### 6. Expected Results

Your results should be close to these values (±5% variation is normal):

#### Latency (milliseconds)

| Model | Paper Result | Expected Range |
|-------|--------------|----------------|
| AE-TRT | 0.76 ms | 0.72 - 0.80 ms |
| AE | 2.2 ms | 2.1 - 2.3 ms |
| AAE | 2.4 ms | 2.3 - 2.5 ms |
| CNN-AE | 2.7 ms | 2.6 - 2.8 ms |
| FF-TRT | 0.4 ms | 0.38 - 0.42 ms |
| FF | 1.8 ms | 1.7 - 1.9 ms |
| LSTM-AE | 7.1 ms | 6.7 - 7.5 ms |
| ResNet-AE | 3.9 ms | 3.7 - 4.1 ms |

#### Throughput (FPS)

| Model | Paper Result | Expected Range |
|-------|--------------|----------------|
| AE-TRT | 397.2 FPS | 380 - 415 FPS |
| AE | 253.8 FPS | 240 - 265 FPS |
| AAE | 276.4 FPS | 260 - 290 FPS |
| FF-TRT | 531.7 FPS | 510 - 550 FPS |
| LSTM-AE | 111.4 FPS | 105 - 120 FPS |

#### Power Consumption (Watts)

| Model | Paper Result | Expected Range |
|-------|--------------|----------------|
| AAE | 4.1 W | 3.9 - 4.3 W |
| AE-TRT | 4.8 W | 4.6 - 5.0 W |
| FF-TRT | 3.8 W | 3.6 - 4.0 W |
| All models | < 5.5 W | Should stay under 6W |

### 7. Analyzing Results

After running the benchmark, you'll find:

```bash
my_results/
├── sustained_measurements.json    # Time-series data
├── benchmark_results.json         # Raw metrics
├── benchmark_results.csv          # Summary table
└── figures/                       # Visualizations
    ├── sustained_latency.png
    ├── sustained_throughput.png
    ├── sustained_power.png
    ├── sustained_energy_efficiency.png
    └── ...
```

Compare with paper results:

```bash
# View your results
cat my_results/benchmark_results.csv

# Compare with paper
cat paper_results/sustained_results/benchmark_results.csv
```

### 8. Troubleshooting

#### Performance Lower Than Expected

1. **Check power mode:**
   ```bash
   sudo nvpmodel -q  # Must show MAXN
   sudo jetson_clocks
   ```

2. **Check thermal throttling:**
   ```bash
   # Monitor temperature during benchmark
   watch -n 1 tegrastats
   ```
   Temperature should stay below 80°C. If higher, add cooling.

3. **Check JetPack version:**
   ```bash
   jetson_release
   ```
   Should be JetPack 6.1+. Earlier versions will have lower performance.

4. **Check memory:**
   ```bash
   free -h
   ```
   Ensure sufficient free memory (at least 4GB).

#### TensorRT Conversion Fails

- Not all models support TensorRT conversion
- Paper only converted AE and FF models
- LSTM models typically don't convert well to TensorRT

#### Dataset Issues

```bash
# Verify dataset integrity
python3 -c "
import h5py
f = h5py.File('clean_5g_dataset.h5', 'r')
print(f'Clean dataset: {len(f[\"signals\"])} samples')
f.close()
f = h5py.File('jammed_5g_dataset.h5', 'r')
print(f'Jammed dataset: {len(f[\"signals\"])} samples')
f.close()
"
# Should show 1,000 samples each
```

### 9. Differences from Paper Results

Minor differences are expected due to:

1. **Thermal variations** - Ambient temperature affects performance
2. **System load** - Background processes consume resources
3. **Random initialization** - Neural network training variability
4. **Power supply variations** - Voltage fluctuations
5. **JetPack version** - Minor SDK differences
6. **Hardware tolerances** - Manufacturing variations

Differences should be within ±5-10% for most metrics.

### 10. Citation

If you use these results in your research, please cite:

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

For questions about reproducing results:
- Open an issue on GitHub
- Email: ndrdmond@memphis.edu

## Validation Checklist

Before running the full benchmark, verify:

- [ ] Jetson Orin Nano 8GB model
- [ ] JetPack 6.1+ installed
- [ ] MAXN SUPER power mode enabled
- [ ] `jetson_clocks` executed
- [ ] Datasets downloaded (1.6GB total)
- [ ] Python dependencies installed
- [ ] Sufficient storage space (10GB+)
- [ ] Adequate cooling (fan recommended)
- [ ] Stable power supply
- [ ] No other intensive processes running

Once all items are checked, you're ready to reproduce the paper results!
