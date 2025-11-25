#!/usr/bin/env python3
"""
Adaptive Power Management Benchmark for RF Anomaly Detection Models

Evaluates the effectiveness of adaptive power management for balancing
performance and energy efficiency on NVIDIA Jetson Orin Nano.

Compares three power management strategies:
1. Static Low Power (7W) - Maximum energy efficiency
2. Static High Power (MAXN SUPER) - Maximum performance
3. Adaptive - Dynamic switching based on latency feedback
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Import local modules
from adaptive_power_manager import AdaptivePowerManager, PowerMode
from workload_generator import WorkloadGenerator, WorkloadPattern
from power_monitor import JetsonPowerMonitor, SystemResourceMonitor
from data_loader import get_dataloaders
from train import get_model

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    print("Warning: TensorRT not available. Will use PyTorch models only.")
    TENSORRT_AVAILABLE = False


class AdaptiveBenchmark:
    """
    Comprehensive benchmark for adaptive power management evaluation.
    """

    def __init__(self,
                 model_name: str,
                 model_path: str,
                 engine_path: Optional[str] = None,
                 dataset_clean: str = '../clean_5g_dataset.h5',
                 dataset_jammed: str = '../jammed_5g_dataset.h5',
                 window_size: int = 128,
                 output_dir: str = 'adaptive_results',
                 verbose: bool = True):
        """
        Initialize adaptive benchmark.

        Args:
            model_name: Name of model to benchmark
            model_path: Path to PyTorch model weights
            engine_path: Optional path to TensorRT engine
            dataset_clean: Path to clean dataset
            dataset_jammed: Path to jammed dataset
            window_size: Input window size
            output_dir: Output directory for results
            verbose: Print detailed progress
        """
        self.model_name = model_name
        self.model_path = model_path
        self.engine_path = engine_path
        self.dataset_clean = dataset_clean
        self.dataset_jammed = dataset_jammed
        self.window_size = window_size
        self.output_dir = Path(output_dir)
        self.verbose = verbose

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.trt_engine = None
        self.use_tensorrt = False

        # Load dataset
        self.test_data = None

        if self.verbose:
            print(f"📊 Initializing Adaptive Benchmark for {model_name}")
            print(f"   Output directory: {self.output_dir}")

    def load_model(self, use_tensorrt: bool = False):
        """Load PyTorch or TensorRT model."""
        if use_tensorrt and self.engine_path and TENSORRT_AVAILABLE:
            if self.verbose:
                print(f"🚀 Loading TensorRT engine from {self.engine_path}")

            # Load TensorRT engine
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(TRT_LOGGER)

            with open(self.engine_path, 'rb') as f:
                self.trt_engine = runtime.deserialize_cuda_engine(f.read())

            self.trt_context = self.trt_engine.create_execution_context()

            # Allocate buffers
            self.trt_bindings = []
            for binding in self.trt_engine:
                size = trt.volume(self.trt_engine.get_binding_shape(binding))
                dtype = trt.nptype(self.trt_engine.get_binding_dtype(binding))
                alloc = cuda.mem_alloc(size * dtype().nbytes)
                self.trt_bindings.append(int(alloc))

            self.trt_stream = cuda.Stream()
            self.use_tensorrt = True

        else:
            if self.verbose:
                print(f"🔥 Loading PyTorch model from {self.model_path}")

            self.model = get_model(self.model_name, self.window_size)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            self.use_tensorrt = False

    def load_test_data(self, max_samples: int = 1000):
        """Load test dataset."""
        if self.verbose:
            print(f"📁 Loading test data (max {max_samples} samples)")

        _, val_loader = get_dataloaders(
            self.dataset_clean,
            self.dataset_jammed,
            window_size=self.window_size,
            batch_size=1
        )

        # Extract samples
        self.test_data = []
        for i, (x, y) in enumerate(val_loader):
            if i >= max_samples:
                break

            # Flatten input if needed
            if 'ae' in self.model_name or self.model_name == 'ff':
                x = x.view(x.size(0), -1)

            self.test_data.append(x)

        if self.verbose:
            print(f"   Loaded {len(self.test_data)} samples")

    def run_inference(self, sample_idx: int) -> float:
        """
        Run single inference and return latency.

        Args:
            sample_idx: Index of sample in test_data

        Returns:
            Inference latency in milliseconds
        """
        sample = self.test_data[sample_idx % len(self.test_data)]

        if self.use_tensorrt:
            # TensorRT inference
            start = time.time()
            self.trt_context.execute_async_v2(self.trt_bindings, self.trt_stream.handle, None)
            self.trt_stream.synchronize()
            latency = (time.time() - start) * 1000  # Convert to ms
        else:
            # PyTorch inference
            sample = sample.to(self.device)
            with torch.no_grad():
                start = time.time()
                _ = self.model(sample)[0]
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                latency = (time.time() - start) * 1000  # Convert to ms

        return latency

    def run_static_baseline(self,
                           power_mode: PowerMode,
                           workload_pattern: WorkloadPattern,
                           duration_s: float = 60.0) -> Dict:
        """
        Run baseline experiment with static power mode.

        Args:
            power_mode: Static power mode to use
            workload_pattern: Workload pattern
            duration_s: Duration of experiment

        Returns:
            Dictionary with experiment results
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"STATIC BASELINE: {power_mode.value} - {workload_pattern.value}")
            print(f"{'='*60}")

        # Set static power mode
        mode_num = 1 if power_mode == PowerMode.LOW_POWER else 0
        try:
            import subprocess
            subprocess.run(['sudo', 'nvpmodel', '-m', str(mode_num)],
                         check=True, capture_output=True, timeout=5.0)
            if self.verbose:
                print(f"✓ Power mode set to {power_mode.value}")
        except:
            if self.verbose:
                print(f"⚠️  Warning: Could not set power mode (simulation mode)")

        # Generate workload
        workload = WorkloadGenerator(
            pattern=workload_pattern,
            duration_s=duration_s,
            base_rate_fps=100.0,
            seed=42
        )
        schedule = workload.generate_schedule()

        # Start power monitoring
        power_monitor = JetsonPowerMonitor(sample_interval_ms=100)
        power_monitor.start_monitoring()

        # Run inference following workload schedule
        latencies = []
        actual_timestamps = []
        violations = 0
        latency_threshold = 10.0  # ms

        start_time = time.time()
        sample_idx = 0

        for scheduled_time, rate in zip(schedule['timestamps'], schedule['rates']):
            # Wait until scheduled time
            while (time.time() - start_time) < scheduled_time:
                time.sleep(0.0001)  # 0.1ms sleep

            # Run inference
            latency = self.run_inference(sample_idx)
            latencies.append(latency)
            actual_timestamps.append(time.time() - start_time)

            if latency > latency_threshold:
                violations += 1

            sample_idx += 1

            # Progress update
            if self.verbose and sample_idx % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {sample_idx} inferences, {elapsed:.1f}s elapsed, "
                      f"avg latency: {np.mean(latencies):.2f}ms")

        # Stop power monitoring
        power_metrics = power_monitor.stop_monitoring()

        # Calculate results
        latencies = np.array(latencies)

        results = {
            'experiment_type': 'static_baseline',
            'power_mode': power_mode.value,
            'workload_pattern': workload_pattern.value,
            'model_name': self.model_name,
            'use_tensorrt': self.use_tensorrt,

            # Latency metrics
            'total_inferences': len(latencies),
            'avg_latency_ms': float(np.mean(latencies)),
            'median_latency_ms': float(np.median(latencies)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
            'max_latency_ms': float(np.max(latencies)),
            'min_latency_ms': float(np.min(latencies)),

            # Violations
            'threshold_violations': int(violations),
            'violation_rate': float(violations / len(latencies)),

            # Power metrics
            'avg_power_w': power_metrics.get('avg_power_w', 0),
            'peak_power_w': power_metrics.get('peak_power_w', 0),
            'total_energy_j': power_metrics.get('total_energy_j', 0),

            # Efficiency
            'throughput_fps': len(latencies) / duration_s,
            'energy_per_inference_j': power_metrics.get('total_energy_j', 0) / len(latencies),
            'fps_per_watt': (len(latencies) / duration_s) / power_metrics.get('avg_power_w', 1),

            # Raw data
            'latencies': latencies.tolist(),
            'timestamps': actual_timestamps,
            'power_metrics': power_metrics
        }

        if self.verbose:
            print(f"\n📊 Results:")
            print(f"   Avg Latency: {results['avg_latency_ms']:.2f} ms")
            print(f"   P95 Latency: {results['p95_latency_ms']:.2f} ms")
            print(f"   Violations: {results['threshold_violations']} ({results['violation_rate']*100:.2f}%)")
            print(f"   Avg Power: {results['avg_power_w']:.2f} W")
            print(f"   Total Energy: {results['total_energy_j']:.2f} J")
            print(f"   Energy/Inference: {results['energy_per_inference_j']:.4f} J")

        return results

    def run_adaptive_experiment(self,
                               workload_pattern: WorkloadPattern,
                               duration_s: float = 60.0,
                               latency_threshold_ms: float = 10.0,
                               hysteresis_time_s: float = 5.0) -> Dict:
        """
        Run experiment with adaptive power management.

        Args:
            workload_pattern: Workload pattern
            duration_s: Duration of experiment
            latency_threshold_ms: Latency threshold for mode switching
            hysteresis_time_s: Hysteresis time before switching back to low power

        Returns:
            Dictionary with experiment results
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"ADAPTIVE: {workload_pattern.value}")
            print(f"  Threshold: {latency_threshold_ms}ms, Hysteresis: {hysteresis_time_s}s")
            print(f"{'='*60}")

        # Initialize adaptive power manager
        apm = AdaptivePowerManager(
            latency_threshold_ms=latency_threshold_ms,
            hysteresis_time_s=hysteresis_time_s,
            initial_mode=PowerMode.LOW_POWER,
            enable_switching=True,
            verbose=self.verbose
        )

        # Generate workload
        workload = WorkloadGenerator(
            pattern=workload_pattern,
            duration_s=duration_s,
            base_rate_fps=100.0,
            seed=42
        )
        schedule = workload.generate_schedule()

        # Start power monitoring
        power_monitor = JetsonPowerMonitor(sample_interval_ms=100)
        power_monitor.start_monitoring()

        # Run inference following workload schedule
        latencies = []
        actual_timestamps = []
        power_modes = []  # Track power mode at each inference

        start_time = time.time()
        sample_idx = 0

        for scheduled_time, rate in zip(schedule['timestamps'], schedule['rates']):
            # Wait until scheduled time
            while (time.time() - start_time) < scheduled_time:
                time.sleep(0.0001)  # 0.1ms sleep

            # Run inference
            latency = self.run_inference(sample_idx)
            latencies.append(latency)
            actual_timestamps.append(time.time() - start_time)
            power_modes.append(apm.get_current_mode().value)

            # Record with adaptive power manager
            apm.record_inference(latency)

            sample_idx += 1

            # Progress update
            if self.verbose and sample_idx % 100 == 0:
                elapsed = time.time() - start_time
                stats = apm.get_statistics()
                print(f"  Progress: {sample_idx} inferences, {elapsed:.1f}s elapsed, "
                      f"mode: {apm.get_current_mode().value}, "
                      f"switches: {stats['total_mode_switches']}, "
                      f"avg latency: {np.mean(latencies):.2f}ms")

        # Stop power monitoring
        power_metrics = power_monitor.stop_monitoring()

        # Get adaptive power manager statistics
        apm_stats = apm.get_statistics()

        # Calculate results
        latencies = np.array(latencies)

        results = {
            'experiment_type': 'adaptive',
            'workload_pattern': workload_pattern.value,
            'model_name': self.model_name,
            'use_tensorrt': self.use_tensorrt,
            'latency_threshold_ms': latency_threshold_ms,
            'hysteresis_time_s': hysteresis_time_s,

            # Latency metrics
            'total_inferences': len(latencies),
            'avg_latency_ms': float(np.mean(latencies)),
            'median_latency_ms': float(np.median(latencies)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
            'max_latency_ms': float(np.max(latencies)),
            'min_latency_ms': float(np.min(latencies)),

            # Violations
            'threshold_violations': apm_stats['threshold_violations'],
            'violation_rate': apm_stats['violation_rate'],

            # Power metrics
            'avg_power_w': power_metrics.get('avg_power_w', 0),
            'peak_power_w': power_metrics.get('peak_power_w', 0),
            'total_energy_j': power_metrics.get('total_energy_j', 0),

            # Adaptive power management metrics
            'total_mode_switches': apm_stats['total_mode_switches'],
            'avg_switch_time_ms': apm_stats['avg_switch_time_ms'],
            'time_in_low_power_s': apm_stats['time_in_low_power_s'],
            'time_in_high_power_s': apm_stats['time_in_high_power_s'],
            'low_power_percentage': apm_stats['low_power_percentage'],

            # Efficiency
            'throughput_fps': len(latencies) / duration_s,
            'energy_per_inference_j': power_metrics.get('total_energy_j', 0) / len(latencies),
            'fps_per_watt': (len(latencies) / duration_s) / power_metrics.get('avg_power_w', 1),

            # Raw data
            'latencies': latencies.tolist(),
            'timestamps': actual_timestamps,
            'power_modes': power_modes,
            'mode_switches': apm_stats['mode_switches'],
            'power_metrics': power_metrics,
            'apm_statistics': apm_stats
        }

        if self.verbose:
            print(f"\n📊 Results:")
            print(f"   Avg Latency: {results['avg_latency_ms']:.2f} ms")
            print(f"   P95 Latency: {results['p95_latency_ms']:.2f} ms")
            print(f"   Violations: {results['threshold_violations']} ({results['violation_rate']*100:.2f}%)")
            print(f"   Mode Switches: {results['total_mode_switches']}")
            print(f"   Low Power Time: {results['low_power_percentage']:.1f}%")
            print(f"   Avg Power: {results['avg_power_w']:.2f} W")
            print(f"   Total Energy: {results['total_energy_j']:.2f} J")
            print(f"   Energy/Inference: {results['energy_per_inference_j']:.4f} J")

        return results

    def save_results(self, results: Dict, filename: str):
        """Save results to JSON file."""
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        if self.verbose:
            print(f"💾 Results saved to {output_path}")


def main():
    """Main function for adaptive benchmark."""
    parser = argparse.ArgumentParser(description='Adaptive Power Management Benchmark')

    parser.add_argument('--model', type=str, required=True,
                       choices=['ae', 'aae', 'cnn_ae', 'lstm_ae', 'resnet_ae', 'ff'],
                       help='Model to benchmark')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to PyTorch model weights')
    parser.add_argument('--engine-path', type=str, default=None,
                       help='Path to TensorRT engine (optional)')
    parser.add_argument('--use-tensorrt', action='store_true',
                       help='Use TensorRT engine if available')

    parser.add_argument('--clean', type=str, default='../clean_5g_dataset.h5',
                       help='Path to clean dataset')
    parser.add_argument('--jammed', type=str, default='../jammed_5g_dataset.h5',
                       help='Path to jammed dataset')
    parser.add_argument('--window-size', type=int, default=128,
                       help='Input window size')
    parser.add_argument('--max-samples', type=int, default=1000,
                       help='Maximum number of test samples')

    parser.add_argument('--workload', type=str, default='all',
                       choices=['bursty', 'continuous', 'variable', 'periodic', 'random', 'all'],
                       help='Workload pattern(s) to test')
    parser.add_argument('--duration', type=float, default=60.0,
                       help='Duration of each experiment in seconds')

    parser.add_argument('--latency-threshold', type=float, default=10.0,
                       help='Latency threshold in milliseconds')
    parser.add_argument('--hysteresis-time', type=float, default=5.0,
                       help='Hysteresis time in seconds')

    parser.add_argument('--output-dir', type=str, default='adaptive_results',
                       help='Output directory for results')
    parser.add_argument('--run-baselines', action='store_true',
                       help='Run static power mode baselines')

    args = parser.parse_args()

    # Initialize benchmark
    benchmark = AdaptiveBenchmark(
        model_name=args.model,
        model_path=args.model_path,
        engine_path=args.engine_path,
        dataset_clean=args.clean,
        dataset_jammed=args.jammed,
        window_size=args.window_size,
        output_dir=args.output_dir,
        verbose=True
    )

    # Load model and data
    benchmark.load_model(use_tensorrt=args.use_tensorrt)
    benchmark.load_test_data(max_samples=args.max_samples)

    # Determine workload patterns to test
    if args.workload == 'all':
        patterns = [WorkloadPattern.BURSTY, WorkloadPattern.CONTINUOUS,
                   WorkloadPattern.VARIABLE, WorkloadPattern.PERIODIC]
    else:
        patterns = [WorkloadPattern(args.workload)]

    all_results = []

    # Run experiments for each workload pattern
    for pattern in patterns:
        # Run baselines if requested
        if args.run_baselines:
            # Low power baseline
            low_power_results = benchmark.run_static_baseline(
                power_mode=PowerMode.LOW_POWER,
                workload_pattern=pattern,
                duration_s=args.duration
            )
            all_results.append(low_power_results)
            benchmark.save_results(
                low_power_results,
                f'{args.model}_static_low_{pattern.value}_results.json'
            )

            # Cooldown
            print("\n⏳ Thermal cooldown: 30 seconds...")
            time.sleep(30)

            # High power baseline
            high_power_results = benchmark.run_static_baseline(
                power_mode=PowerMode.HIGH_POWER,
                workload_pattern=pattern,
                duration_s=args.duration
            )
            all_results.append(high_power_results)
            benchmark.save_results(
                high_power_results,
                f'{args.model}_static_high_{pattern.value}_results.json'
            )

            # Cooldown
            print("\n⏳ Thermal cooldown: 30 seconds...")
            time.sleep(30)

        # Run adaptive experiment
        adaptive_results = benchmark.run_adaptive_experiment(
            workload_pattern=pattern,
            duration_s=args.duration,
            latency_threshold_ms=args.latency_threshold,
            hysteresis_time_s=args.hysteresis_time
        )
        all_results.append(adaptive_results)
        benchmark.save_results(
            adaptive_results,
            f'{args.model}_adaptive_{pattern.value}_results.json'
        )

        # Cooldown between patterns
        if pattern != patterns[-1]:
            print("\n⏳ Thermal cooldown: 30 seconds...")
            time.sleep(30)

    # Save summary
    summary = {
        'model': args.model,
        'use_tensorrt': args.use_tensorrt,
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'latency_threshold_ms': args.latency_threshold,
            'hysteresis_time_s': args.hysteresis_time,
            'duration_s': args.duration
        },
        'results': all_results
    }
    benchmark.save_results(summary, f'{args.model}_adaptive_summary.json')

    print("\n✅ Adaptive benchmark completed!")
    print(f"📊 Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
