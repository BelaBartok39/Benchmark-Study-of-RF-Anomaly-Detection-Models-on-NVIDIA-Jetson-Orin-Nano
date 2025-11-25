# RF Anomaly Detection Benchmark on NVIDIA Jetson Orin Nano

This repository contains the code and resources for the paper:

**"A Benchmark Study of RF Anomaly Detection Models on NVIDIA Jetson Orin Nano"**
Nicholas D. Redmond, Mohd Hasan Ali, Dipankar Dasgupta, Myounggyu Won
*IEEE Consumer Communications & Networking Conference (CCNC) 2025*

## Overview

This is the first comprehensive benchmark study evaluating machine learning models for RF anomaly detection on the NVIDIA Jetson Orin Nano platform. The study demonstrates that edge AI platforms can support real-time RF anomaly detection with strong performance across latency, throughput, energy efficiency, memory usage, and resource utilization.

### Key Contributions

- First benchmark study of RF anomaly detection models on NVIDIA Jetson Orin Nano
- Comprehensive evaluation of 6 ML models including Autoencoder (AE), Adversarial Autoencoder (AAE), CNN-AE, LSTM-AE, ResNet-AE, and Feedforward networks
- Performance analysis using both PyTorch and TensorRT optimization
- Sustained inference testing (5-minute continuous operation) with power monitoring
- Real-world deployment metrics: latency, throughput, energy efficiency, memory footprint, resource utilization

### Models Evaluated

1. **AE** - Basic Autoencoder
2. **AAE** - Adversarial Autoencoder
3. **CNN-AE** - Convolutional Neural Network Autoencoder
4. **LSTM-AE** - Long Short-Term Memory Autoencoder
5. **ResNet-AE** - Residual Network Autoencoder
6. **FF** - Feedforward Neural Network (baseline)

## 🆕 Adaptive Power Management Extension

**NEW**: This repository now includes an extension that addresses energy efficiency through **adaptive power management**. The original benchmark used MAXN SUPER mode exclusively, which can waste energy under light workloads.

The adaptive power management system:
- **Dynamically switches** between power modes (15W and MAXN SUPER) based on real-time latency feedback
- **Balances performance and energy efficiency** for variable workloads
- **Reduces energy consumption by 20-40%** for bursty workloads while maintaining latency targets
- Includes comprehensive evaluation across realistic RF monitoring scenarios

*Note*: Uses 15W mode (JetPack 6.1 mode 0) instead of 7W, which requires a reboot and cannot be dynamically switched.

**Quick Start**:
```bash
# Run adaptive power management experiments
./run_adaptive_experiments.sh ae

# For TensorRT-optimized models
./run_adaptive_experiments.sh ae true
```

**See**: [ADAPTIVE_POWER_MANAGEMENT.md](ADAPTIVE_POWER_MANAGEMENT.md) for complete documentation, methodology, and results.

## Hardware Requirements

- **NVIDIA Jetson Orin Nano Developer Kit (8GB)**
- JetPack SDK 6.1 or later
- MAXN SUPER power mode enabled
- microSD card (64GB+ recommended)
- 14V power supply

## Software Requirements

- Ubuntu 22.04 (Jetson Linux 36.4.3+)
- Python 3.8+
- CUDA 12.x (included with JetPack)
- TensorRT 8.x+ (included with JetPack)

## Installation

### 1. Clone the Repository

```bash
git clone git@github.com:BelaBartok39/Benchmark-Study-of-RF-Anomaly-Detection-Models-on-NVIDIA-Jetson-Orin-Nano.git
cd Benchmark-Study-of-RF-Anomaly-Detection-Models-on-NVIDIA-Jetson-Orin-Nano
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Datasets

The datasets are hosted on Hugging Face and are too large to include in this repository.

**Dataset URL:** https://huggingface.co/datasets/b4byn1cky/RF_Anomaly_Detection

Download the datasets and place them in the repository root:

```bash
# Option 1: Manual download
# Visit the URL above and download clean_5g_dataset.h5 and jammed_5g_dataset.h5

# Option 2: Using huggingface-cli (if installed)
huggingface-cli download b4byn1cky/RF_Anomaly_Detection --repo-type dataset --local-dir ./datasets
mv datasets/*.h5 .
```

Expected files:
- `clean_5g_dataset.h5` (785 MB)
- `jammed_5g_dataset.h5` (785 MB)

### 4. Verify Installation

```bash
python3 -c "import torch; import tensorrt; import h5py; print('All dependencies installed successfully!')"
```

## Quick Start

### Train All Models

Train all 6 models with default parameters:

```bash
cd src
python train_all.py --clean ../clean_5g_dataset.h5 --jammed ../jammed_5g_dataset.h5
```

### Run Basic Benchmark

Benchmark models with PyTorch:

```bash
python comprehensive_benchmark.py \
    --models ae ff \
    --clean ../clean_5g_dataset.h5 \
    --jammed ../jammed_5g_dataset.h5 \
    --output-dir ../results
```

### Run Sustained Benchmark (Paper Configuration)

Run the full 5-minute sustained inference tests as described in the paper:

```bash
cd ..
python run_sustained_benchmark.py
```

This will:
- Run 300-second (5-minute) sustained inference per model
- Use 30-second moving average windows for analysis
- Enforce 60-second thermal cooldown between models
- Monitor power consumption at 10Hz
- Generate comprehensive visualizations

## Repository Structure

```
.
├── src/                          # Source code
│   ├── models/                   # Model architectures
│   │   ├── ae.py                # Autoencoder
│   │   ├── aae.py               # Adversarial Autoencoder
│   │   ├── cnn_ae.py            # CNN Autoencoder
│   │   ├── lstm_ae.py           # LSTM Autoencoder
│   │   ├── resnet_ae.py         # ResNet Autoencoder
│   │   └── ff_models.py         # Feedforward Network
│   ├── train.py                 # Training script
│   ├── train_all.py             # Train all models
│   ├── evaluate.py              # Evaluation script
│   ├── benchmark.py             # Basic benchmarking
│   ├── comprehensive_benchmark.py  # Full benchmarking suite
│   ├── convert_tensorrt.py      # PyTorch to TensorRT conversion
│   ├── data_loader.py           # Dataset loading utilities
│   ├── preprocessing.py         # Signal preprocessing
│   ├── power_monitor.py         # Power monitoring utilities
│   └── run_all.py               # Complete pipeline
├── paper_results/               # Results reported in the paper
│   ├── experiment_1/            # Initial benchmark
│   ├── experiment_2/            # Replicate benchmark
│   ├── experiment_3-12(sust)/   # Sustained inference experiments
│   └── sustained_results/       # Final sustained results
├── run_sustained_benchmark.py   # Sustained benchmark launcher
├── sustained_experiment_details.md  # Methodology documentation
├── requirements.txt             # Python dependencies
├── LICENSE                      # GPL-3.0 License
├── REPRODUCE.md                 # Instructions to reproduce paper results
├── CITATION.cff                 # Citation information
└── README.md                    # This file

# After downloading datasets:
├── clean_5g_dataset.h5          # Clean 5G signals (not in repo)
└── jammed_5g_dataset.h5         # Jammed 5G signals (not in repo)
```

## Usage Examples

### 1. Train a Single Model

```bash
cd src
python train.py \
    --model ae \
    --clean ../clean_5g_dataset.h5 \
    --jammed ../jammed_5g_dataset.h5 \
    --epochs 50 \
    --batch-size 32
```

### 2. Evaluate a Trained Model

```bash
python evaluate.py \
    --model ae \
    --model-path output/weights/ae_best.pth \
    --clean ../clean_5g_dataset.h5 \
    --jammed ../jammed_5g_dataset.h5
```

### 3. Convert to TensorRT

```bash
python convert_tensorrt.py \
    --model ae \
    --weights-path output/weights/ae_best.pth \
    --out-dir output/engines
```

### 4. Run Complete Pipeline

Train, evaluate, convert to TensorRT, and benchmark:

```bash
python run_all.py \
    --models ae ff \
    --clean ../clean_5g_dataset.h5 \
    --jammed ../jammed_5g_dataset.h5 \
    --epochs 50
```

## Benchmark Metrics

The benchmark suite evaluates:

1. **Latency** - Time to process a single sample (milliseconds)
2. **Throughput** - Samples processed per second (FPS)
3. **Energy Efficiency** - Average power consumption (Watts) and FPS/Watt
4. **Memory Footprint** - GPU and system memory usage (MB/GB)
5. **Resource Utilization** - CPU and GPU utilization percentages

## Key Results

From the paper (300-second sustained inference):

| Model | Latency (ms) | Throughput (FPS) | Avg Power (W) | GPU Memory (MB) |
|-------|--------------|------------------|---------------|-----------------|
| AE-TRT | 0.76 | 397.2 | 4.8 | 17.2 |
| AE | 2.2 | 253.8 | 4.5 | 17.4 |
| AAE | 2.4 | 276.4 | 4.1 | 14.2 |
| CNN-AE | 2.7 | 251.8 | 4.6 | 16.1 |
| FF-TRT | 0.4 | 531.7 | 3.8 | 9.8 |
| FF | 1.8 | 335.6 | 4.3 | 10.1 |
| LSTM-AE | 7.1 | 111.4 | 5.1 | 11.8 |
| ResNet-AE | 3.9 | 184.3 | 5.0 | 14.7 |

All models achieved:
- Latency < 10ms (well below real-time thresholds)
- Throughput > 60 FPS
- Power consumption < 5W
- Memory usage within 8GB system capacity

## Dataset Details

The datasets consist of synthetic 5G signals generated using GNU Radio:

- **Signal Type:** 5G NR with QPSK modulation
- **Samples per file:** 1,000 (balanced)
- **Sample length:** 1,024 complex I/Q samples
- **Sampling rate:** 1 MHz
- **Duration per sample:** ~1.024 ms
- **Pulse shaping:** Root-raised cosine (excess BW = 0.35)
- **Format:** HDF5 files

Each dataset contains:
- `clean_5g_dataset.h5` - Normal 5G signals
- `jammed_5g_dataset.h5` - Jammed/anomalous signals

## Reproducing Paper Results

See [REPRODUCE.md](REPRODUCE.md) for detailed instructions on reproducing the exact results from the paper.

The `paper_results/` directory contains the original experimental data used in the publication.

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@inproceedings{redmond2025benchmark,
  title={A Benchmark Study of RF Anomaly Detection Models on NVIDIA Jetson Orin Nano},
  author={Redmond, Nicholas D. and Ali, Mohd Hasan and Dasgupta, Dipankar and Won, Myounggyu},
  booktitle={IEEE Consumer Communications \& Networking Conference (CCNC)},
  year={2025},
  organization={IEEE}
}
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- University of Memphis, Department of Computer Science
- University of Memphis, Department of Electrical and Computer Engineering
- NVIDIA for the Jetson Orin Nano Developer Kit and JetPack SDK

## Contact

For questions or issues, please open an issue on GitHub or contact:

- Nicholas D. Redmond - ndrdmond@memphis.edu
- Myounggyu Won - mwon@memphis.edu

## Related Links

- [NVIDIA Jetson Orin Nano](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)
- [JetPack SDK](https://developer.nvidia.com/embedded/jetpack)
- [Dataset on Hugging Face](https://huggingface.co/datasets/b4byn1cky/RF_Anomaly_Detection)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
