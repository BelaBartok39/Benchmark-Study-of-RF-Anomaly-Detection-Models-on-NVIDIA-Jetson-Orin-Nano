#!/usr/bin/env python3
"""
Adaptive Power Management Benchmark for RF Anomaly Detection Models

Evaluates the effectiveness of adaptive power management for balancing
performance and energy efficiency on NVIDIA Jetson Orin Nano.

Compares three power management strategies:
1. Static Low Power (15W) - Maximum energy efficiency
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
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from queue import Queue

# Import local modules
from adaptive_power_manager import AdaptivePowerManager, PowerMode
from workload_generator import WorkloadGenerator, WorkloadPattern
from power_monitor import JetsonPowerMonitor, SystemResourceMonitor
from data_loader import get_dataloaders
from train import get_model
from threshold_calibrator import ThresholdCalibrator

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
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=False))
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    print(f"\n❌ Error loading model: Shape mismatch detected.")
                    print(f"   This usually means the window_size doesn't match the trained model.")
                    print(f"   Current window_size: {self.window_size}")
                    print(f"   Try: --window-size 1024 or --window-size 2048")
                    print(f"\n   Error details: {e}\n")
                raise
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
            batch_size=1,
            max_samples=max_samples  # Limit data loading to save memory
        )

        # Extract samples
        self.test_data = []
        for i, (x, y) in enumerate(val_loader):
            if i >= max_samples:
                break

            # Flatten input only for dense models (ae, aae, ff)
            # CNN/LSTM models need 3D input: (batch, channels, seq_len)
            if self.model_name in ['ae', 'aae', 'ff']:
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

    def run_inference_batch(self, sample_indices: List[int]) -> Tuple[float, List[float]]:
        """
        Run batched inference and return aggregate latency.

        Args:
            sample_indices: List of sample indices to process as batch

        Returns:
            Tuple of (total_batch_latency_ms, per_sample_latencies_ms)
        """
        # Gather batch - concatenate along batch dimension
        batch_samples = []
        for idx in sample_indices:
            sample = self.test_data[idx % len(self.test_data)]
            batch_samples.append(sample)

        # Concatenate into batch tensor (each sample is [1, C, S], result is [N, C, S])
        batch = torch.cat(batch_samples, dim=0)

        if self.use_tensorrt:
            # TensorRT batched inference
            start = time.time()
            self.trt_context.execute_async_v2(self.trt_bindings, self.trt_stream.handle, None)
            self.trt_stream.synchronize()
            total_latency = (time.time() - start) * 1000  # Convert to ms
        else:
            # PyTorch batched inference
            batch = batch.to(self.device)
            with torch.no_grad():
                start = time.time()
                _ = self.model(batch)[0]
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                total_latency = (time.time() - start) * 1000  # Convert to ms

        # Per-sample latency (amortized)
        per_sample_latency = total_latency / len(sample_indices)
        per_sample_latencies = [per_sample_latency] * len(sample_indices)

        return total_latency, per_sample_latencies

    def _run_single_channel_adaptive(self,
                                     apm: AdaptivePowerManager,
                                     schedule: Dict,
                                     batch_size: int) -> Dict:
        """
        Run single-channel adaptive experiment with optional batching.

        Args:
            apm: Adaptive power manager instance
            schedule: Workload schedule from generator
            batch_size: Batch size (1 for single-sample)

        Returns:
            Dictionary with latencies, timestamps, and power modes
        """
        latencies = []
        actual_timestamps = []
        power_modes = []

        start_time = time.time()
        sample_idx = 0
        schedule_idx = 0

        if batch_size == 1:
            # Single-sample mode (backward compatible)
            for scheduled_time, rate in zip(schedule['timestamps'], schedule['rates']):
                # Wait until scheduled time
                while (time.time() - start_time) < scheduled_time:
                    time.sleep(0.0001)  # 0.1ms sleep

                # Run single inference
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
        else:
            # Batched mode
            batch_indices = []

            for scheduled_time, rate in zip(schedule['timestamps'], schedule['rates']):
                # Wait until scheduled time
                while (time.time() - start_time) < scheduled_time:
                    time.sleep(0.0001)  # 0.1ms sleep

                # Accumulate samples for batch
                batch_indices.append(sample_idx)
                sample_idx += 1

                # Process batch when full or at end
                if len(batch_indices) >= batch_size or schedule_idx == len(schedule['timestamps']) - 1:
                    # Run batched inference
                    batch_latency, per_sample_lats = self.run_inference_batch(batch_indices)

                    # Record each sample's latency
                    for lat in per_sample_lats:
                        latencies.append(lat)
                        actual_timestamps.append(time.time() - start_time)
                        power_modes.append(apm.get_current_mode().value)

                        # Record with adaptive power manager (using amortized latency)
                        apm.record_inference(lat)

                    batch_indices = []

                    # Progress update
                    if self.verbose and len(latencies) % 100 == 0:
                        elapsed = time.time() - start_time
                        stats = apm.get_statistics()
                        print(f"  Progress: {len(latencies)} inferences, {elapsed:.1f}s elapsed, "
                              f"mode: {apm.get_current_mode().value}, "
                              f"switches: {stats['total_mode_switches']}, "
                              f"avg latency: {np.mean(latencies):.2f}ms")

                schedule_idx += 1

        return {
            'latencies': latencies,
            'timestamps': actual_timestamps,
            'power_modes': power_modes
        }

    def _run_multi_channel_adaptive(self,
                                    apm: AdaptivePowerManager,
                                    schedule: Dict,
                                    num_channels: int,
                                    batch_size: int,
                                    duration_s: float) -> Dict:
        """
        Run multi-channel adaptive experiment with optional batching per channel.

        Args:
            apm: Adaptive power manager instance
            schedule: Workload schedule from generator (will be replicated per channel)
            num_channels: Number of concurrent channels
            batch_size: Batch size per channel (1 for single-sample)
            duration_s: Duration of experiment

        Returns:
            Dictionary with aggregated latencies, timestamps, and power modes
        """
        # Shared data structures (thread-safe)
        all_latencies = []
        all_timestamps = []
        all_power_modes = []
        latency_lock = threading.Lock()

        # Global start time
        global_start_time = time.time()

        def channel_worker(channel_id: int):
            """Worker function for each channel thread."""
            sample_idx = channel_id * 10000  # Offset to avoid sample overlap
            schedule_idx = 0

            if batch_size == 1:
                # Single-sample per channel
                for scheduled_time, rate in zip(schedule['timestamps'], schedule['rates']):
                    # Wait until scheduled time
                    while (time.time() - global_start_time) < scheduled_time:
                        time.sleep(0.0001)

                    # Run inference
                    latency = self.run_inference(sample_idx)
                    current_time = time.time() - global_start_time
                    current_mode = apm.get_current_mode().value

                    # Thread-safe append
                    with latency_lock:
                        all_latencies.append(latency)
                        all_timestamps.append(current_time)
                        all_power_modes.append(current_mode)

                        # Record with adaptive power manager
                        apm.record_inference(latency)

                    sample_idx += 1
            else:
                # Batched per channel
                batch_indices = []

                for scheduled_time, rate in zip(schedule['timestamps'], schedule['rates']):
                    # Wait until scheduled time
                    while (time.time() - global_start_time) < scheduled_time:
                        time.sleep(0.0001)

                    # Accumulate samples for batch
                    batch_indices.append(sample_idx)
                    sample_idx += 1

                    # Process batch when full or at end
                    if len(batch_indices) >= batch_size or schedule_idx == len(schedule['timestamps']) - 1:
                        # Run batched inference
                        batch_latency, per_sample_lats = self.run_inference_batch(batch_indices)
                        current_time = time.time() - global_start_time
                        current_mode = apm.get_current_mode().value

                        # Thread-safe append
                        with latency_lock:
                            for lat in per_sample_lats:
                                all_latencies.append(lat)
                                all_timestamps.append(current_time)
                                all_power_modes.append(current_mode)

                                # Record with adaptive power manager
                                apm.record_inference(lat)

                        batch_indices = []

                    schedule_idx += 1

        # Create and start channel threads
        threads = []
        for channel_id in range(num_channels):
            thread = threading.Thread(target=channel_worker, args=(channel_id,))
            thread.start()
            threads.append(thread)

        # Progress monitoring in main thread
        while any(t.is_alive() for t in threads):
            time.sleep(1.0)
            if self.verbose:
                with latency_lock:
                    if len(all_latencies) > 0:
                        elapsed = time.time() - global_start_time
                        stats = apm.get_statistics()
                        print(f"  Progress: {len(all_latencies)} inferences, {elapsed:.1f}s elapsed, "
                              f"mode: {apm.get_current_mode().value}, "
                              f"switches: {stats['total_mode_switches']}, "
                              f"avg latency: {np.mean(all_latencies):.2f}ms")

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        return {
            'latencies': all_latencies,
            'timestamps': all_timestamps,
            'power_modes': all_power_modes
        }

    def _run_single_channel_static(self,
                                   schedule: Dict,
                                   batch_size: int) -> Dict:
        """
        Run single-channel static baseline with optional batching.

        Args:
            schedule: Workload schedule from generator
            batch_size: Batch size (1 for single-sample)

        Returns:
            Dictionary with latencies, timestamps, and violations
        """
        latencies = []
        actual_timestamps = []
        violations = 0
        latency_threshold = 10.0  # ms

        start_time = time.time()
        sample_idx = 0
        schedule_idx = 0

        if batch_size == 1:
            # Single-sample mode
            for scheduled_time, rate in zip(schedule['timestamps'], schedule['rates']):
                while (time.time() - start_time) < scheduled_time:
                    time.sleep(0.0001)

                latency = self.run_inference(sample_idx)
                latencies.append(latency)
                actual_timestamps.append(time.time() - start_time)

                if latency > latency_threshold:
                    violations += 1

                sample_idx += 1

                if self.verbose and sample_idx % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Progress: {sample_idx} inferences, {elapsed:.1f}s elapsed, "
                          f"avg latency: {np.mean(latencies):.2f}ms")
        else:
            # Batched mode
            batch_indices = []

            for scheduled_time, rate in zip(schedule['timestamps'], schedule['rates']):
                while (time.time() - start_time) < scheduled_time:
                    time.sleep(0.0001)

                batch_indices.append(sample_idx)
                sample_idx += 1

                if len(batch_indices) >= batch_size or schedule_idx == len(schedule['timestamps']) - 1:
                    batch_latency, per_sample_lats = self.run_inference_batch(batch_indices)

                    for lat in per_sample_lats:
                        latencies.append(lat)
                        actual_timestamps.append(time.time() - start_time)

                        if lat > latency_threshold:
                            violations += 1

                    batch_indices = []

                    if self.verbose and len(latencies) % 100 == 0:
                        elapsed = time.time() - start_time
                        print(f"  Progress: {len(latencies)} inferences, {elapsed:.1f}s elapsed, "
                              f"avg latency: {np.mean(latencies):.2f}ms")

                schedule_idx += 1

        return {
            'latencies': latencies,
            'timestamps': actual_timestamps,
            'violations': violations
        }

    def _run_multi_channel_static(self,
                                  schedule: Dict,
                                  num_channels: int,
                                  batch_size: int,
                                  duration_s: float) -> Dict:
        """
        Run multi-channel static baseline with optional batching per channel.

        Args:
            schedule: Workload schedule from generator (replicated per channel)
            num_channels: Number of concurrent channels
            batch_size: Batch size per channel (1 for single-sample)
            duration_s: Duration of experiment

        Returns:
            Dictionary with aggregated latencies, timestamps, and violations
        """
        all_latencies = []
        all_timestamps = []
        all_violations = [0]  # List to allow mutation in thread
        latency_lock = threading.Lock()
        latency_threshold = 10.0  # ms

        global_start_time = time.time()

        def channel_worker(channel_id: int):
            """Worker function for each channel thread."""
            sample_idx = channel_id * 10000
            schedule_idx = 0

            if batch_size == 1:
                # Single-sample per channel
                for scheduled_time, rate in zip(schedule['timestamps'], schedule['rates']):
                    while (time.time() - global_start_time) < scheduled_time:
                        time.sleep(0.0001)

                    latency = self.run_inference(sample_idx)
                    current_time = time.time() - global_start_time

                    with latency_lock:
                        all_latencies.append(latency)
                        all_timestamps.append(current_time)

                        if latency > latency_threshold:
                            all_violations[0] += 1

                    sample_idx += 1
            else:
                # Batched per channel
                batch_indices = []

                for scheduled_time, rate in zip(schedule['timestamps'], schedule['rates']):
                    while (time.time() - global_start_time) < scheduled_time:
                        time.sleep(0.0001)

                    batch_indices.append(sample_idx)
                    sample_idx += 1

                    if len(batch_indices) >= batch_size or schedule_idx == len(schedule['timestamps']) - 1:
                        batch_latency, per_sample_lats = self.run_inference_batch(batch_indices)
                        current_time = time.time() - global_start_time

                        with latency_lock:
                            for lat in per_sample_lats:
                                all_latencies.append(lat)
                                all_timestamps.append(current_time)

                                if lat > latency_threshold:
                                    all_violations[0] += 1

                        batch_indices = []

                    schedule_idx += 1

        # Create and start channel threads
        threads = []
        for channel_id in range(num_channels):
            thread = threading.Thread(target=channel_worker, args=(channel_id,))
            thread.start()
            threads.append(thread)

        # Progress monitoring
        while any(t.is_alive() for t in threads):
            time.sleep(1.0)
            if self.verbose:
                with latency_lock:
                    if len(all_latencies) > 0:
                        elapsed = time.time() - global_start_time
                        print(f"  Progress: {len(all_latencies)} inferences, {elapsed:.1f}s elapsed, "
                              f"avg latency: {np.mean(all_latencies):.2f}ms")

        # Wait for all threads
        for thread in threads:
            thread.join()

        return {
            'latencies': all_latencies,
            'timestamps': all_timestamps,
            'violations': all_violations[0]
        }

    def run_static_baseline(self,
                           power_mode: PowerMode,
                           workload_pattern: WorkloadPattern,
                           duration_s: float = 60.0,
                           batch_size: int = 1,
                           num_channels: int = 1) -> Dict:
        """
        Run baseline experiment with static power mode.

        Args:
            power_mode: Static power mode to use
            workload_pattern: Workload pattern
            duration_s: Duration of experiment
            batch_size: Batch size for batched inference (default: 1)
            num_channels: Number of concurrent channels (default: 1)

        Returns:
            Dictionary with experiment results
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"STATIC BASELINE: {power_mode.value} - {workload_pattern.value}")
            if batch_size > 1:
                print(f"  Batch Size: {batch_size}")
            if num_channels > 1:
                print(f"  Channels: {num_channels}")
            print(f"{'='*60}")

        # Set static power mode (JetPack 6.1: 0=15W, 1=25W, 2=MAXN)
        mode_num = 0 if power_mode == PowerMode.LOW_POWER else 2
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

        # Dispatch to appropriate execution mode
        if num_channels == 1:
            # Single-channel mode (with or without batching)
            results_data = self._run_single_channel_static(schedule, batch_size)
        else:
            # Multi-channel mode (with or without batching)
            results_data = self._run_multi_channel_static(schedule, num_channels, batch_size, duration_s)

        latencies = results_data['latencies']
        actual_timestamps = results_data['timestamps']
        violations = results_data['violations']

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
            'batch_size': batch_size,
            'num_channels': num_channels,

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
                               hysteresis_time_s: float = 5.0,
                               use_model_defaults: bool = False,
                               enable_three_tier: bool = True,
                               batch_size: int = 1,
                               num_channels: int = 1) -> Dict:
        """
        Run experiment with adaptive power management.

        Args:
            workload_pattern: Workload pattern
            duration_s: Duration of experiment
            latency_threshold_ms: Latency threshold for mode switching (ignored if use_model_defaults=True)
            hysteresis_time_s: Hysteresis time before switching back to low power (ignored if use_model_defaults=True)
            use_model_defaults: If True, use model-specific thresholds and hysteresis
            enable_three_tier: Enable three-tier power management (15W/25W/MAXN)
            batch_size: Batch size for batched inference (default: 1 for single-sample)
            num_channels: Number of concurrent channels to simulate (default: 1 for single-channel)

        Returns:
            Dictionary with experiment results
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"ADAPTIVE: {workload_pattern.value}")
            print(f"  Threshold: {latency_threshold_ms}ms, Hysteresis: {hysteresis_time_s}s")
            if batch_size > 1:
                print(f"  Batch Size: {batch_size}")
            if num_channels > 1:
                print(f"  Channels: {num_channels}")
            print(f"{'='*60}")

        # Initialize adaptive power manager
        apm = AdaptivePowerManager(
            latency_threshold_ms=latency_threshold_ms,
            hysteresis_time_s=hysteresis_time_s,
            initial_mode=PowerMode.LOW_POWER,
            enable_switching=True,
            enable_three_tier=enable_three_tier,
            verbose=self.verbose,
            model_name=self.model_name,
            use_model_defaults=use_model_defaults
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

        # Dispatch to appropriate execution mode
        if num_channels == 1:
            # Single-channel mode (with or without batching)
            results_data = self._run_single_channel_adaptive(
                apm, schedule, batch_size
            )
        else:
            # Multi-channel mode (with or without batching)
            results_data = self._run_multi_channel_adaptive(
                apm, schedule, num_channels, batch_size, duration_s
            )

        latencies = results_data['latencies']
        actual_timestamps = results_data['timestamps']
        power_modes = results_data['power_modes']

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
            'batch_size': batch_size,
            'num_channels': num_channels,

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

    parser.add_argument('--clean', type=str, default='clean_5g_dataset.h5',
                       help='Path to clean dataset')
    parser.add_argument('--jammed', type=str, default='jammed_5g_dataset.h5',
                       help='Path to jammed dataset')
    parser.add_argument('--window-size', type=int, default=1024,
                       help='Input window size (must match training)')
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
    parser.add_argument('--use-model-defaults', action='store_true',
                       help='Use model-specific thresholds and hysteresis (overrides --latency-threshold and --hysteresis-time)')
    parser.add_argument('--auto-calibrate', action='store_true',
                       help='Automatically calibrate thresholds based on hardware profiling (overrides --use-model-defaults)')
    parser.add_argument('--target-sla', type=float, default=10.0,
                       help='Target SLA (latency) in milliseconds for auto-calibration (default: 10.0)')
    parser.add_argument('--auto-adjust-sla', action='store_true',
                       help='Automatically adjust SLA if target is unreachable based on hardware profiling (requires --auto-calibrate)')
    parser.add_argument('--enable-three-tier', action='store_true', default=True,
                       help='Enable three-tier power management (15W/25W/MAXN) instead of two-tier (15W/MAXN)')
    parser.add_argument('--disable-three-tier', dest='enable_three_tier', action='store_false',
                       help='Disable three-tier mode and use two-tier (15W/MAXN) only')

    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for batched inference (default: 1 for single-sample)')
    parser.add_argument('--num-channels', type=int, default=1,
                       help='Number of concurrent channels to simulate multi-channel RF monitoring (default: 1)')

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

    # Auto-calibrate thresholds if requested
    calibration_results = None
    if args.auto_calibrate:
        print("\n" + "="*60)
        print("AUTOMATIC THRESHOLD CALIBRATION")
        print("="*60)
        print("Running hardware profiling to determine optimal thresholds...")
        print("")

        calibrator = ThresholdCalibrator(
            benchmark=benchmark,
            target_sla_ms=args.target_sla,
            safety_margin=0.9,
            verbose=True,
            auto_adjust_sla=args.auto_adjust_sla
        )

        try:
            calibration_results = calibrator.calibrate(strategy='sla_based')

            # Override thresholds with calibrated values
            args.latency_threshold = calibration_results['high_threshold_ms']
            args.hysteresis_time = calibration_results['hysteresis_time_s']

            # For three-tier, we need the medium threshold too
            # We'll store it in a way that adaptive_power_manager can access
            # For now, pass high_threshold (25W→MAXN threshold) to latency_threshold

            print(f"\n✅ Calibration complete!")
            print(f"   Using calibrated thresholds:")
            print(f"   - 15W → 25W: {calibration_results['medium_threshold_ms']:.2f}ms")
            print(f"   - 25W → MAXN: {calibration_results['high_threshold_ms']:.2f}ms")
            print(f"   - Hysteresis: {calibration_results['hysteresis_time_s']:.1f}s")
            print("")

            # Save calibration results
            benchmark.save_results(
                calibration_results,
                f'{args.model}_calibration_results.json'
            )

        except ValueError as e:
            print(f"\n❌ Calibration failed: {e}")
            print("   Falling back to default thresholds")
            args.auto_calibrate = False

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
            # Low power baseline (15W)
            low_power_results = benchmark.run_static_baseline(
                power_mode=PowerMode.LOW_POWER,
                workload_pattern=pattern,
                duration_s=args.duration,
                batch_size=args.batch_size,
                num_channels=args.num_channels
            )
            all_results.append(low_power_results)
            benchmark.save_results(
                low_power_results,
                f'{args.model}_static_low_{pattern.value}_results.json'
            )

            # Cooldown
            print("\n⏳ Thermal cooldown: 30 seconds...")
            time.sleep(30)

            # Medium power baseline (25W)
            medium_power_results = benchmark.run_static_baseline(
                power_mode=PowerMode.MEDIUM_POWER,
                workload_pattern=pattern,
                duration_s=args.duration,
                batch_size=args.batch_size,
                num_channels=args.num_channels
            )
            all_results.append(medium_power_results)
            benchmark.save_results(
                medium_power_results,
                f'{args.model}_static_medium_{pattern.value}_results.json'
            )

            # Cooldown
            print("\n⏳ Thermal cooldown: 30 seconds...")
            time.sleep(30)

            # High power baseline (MAXN)
            high_power_results = benchmark.run_static_baseline(
                power_mode=PowerMode.HIGH_POWER,
                workload_pattern=pattern,
                duration_s=args.duration,
                batch_size=args.batch_size,
                num_channels=args.num_channels
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
            hysteresis_time_s=args.hysteresis_time,
            use_model_defaults=args.use_model_defaults,
            enable_three_tier=args.enable_three_tier,
            batch_size=args.batch_size,
            num_channels=args.num_channels
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
            'duration_s': args.duration,
            'batch_size': args.batch_size,
            'num_channels': args.num_channels,
            'enable_three_tier': args.enable_three_tier,
            'use_model_defaults': args.use_model_defaults,
            'auto_calibrate': args.auto_calibrate,
            'target_sla_ms': args.target_sla if args.auto_calibrate else None
        },
        'calibration': calibration_results if args.auto_calibrate else None,
        'results': all_results
    }
    benchmark.save_results(summary, f'{args.model}_adaptive_summary.json')

    print("\n✅ Adaptive benchmark completed!")
    print(f"📊 Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
