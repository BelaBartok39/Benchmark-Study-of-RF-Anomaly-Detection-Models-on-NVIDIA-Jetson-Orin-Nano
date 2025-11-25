#!/usr/bin/env python3
"""
Comprehensive benchmarking script for RF signal anomaly detection models.
Benchmarks PyTorch models, converts to TensorRT, and provides visual comparisons.
Enhanced with academic-grade power and energy efficiency monitoring.
"""

import os
import time
import argparse
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

# TensorRT imports with fallback
try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    print("Warning: pycuda and/or tensorrt not available. TensorRT benchmarking will be skipped.")
    TENSORRT_AVAILABLE = False

# Power monitoring imports
try:
    from power_monitor import JetsonPowerMonitor, SystemResourceMonitor
    POWER_MONITORING_AVAILABLE = True
except ImportError:
    print("Warning: Power monitoring not available. Running without power metrics.")
    POWER_MONITORING_AVAILABLE = False

from data_loader import get_dataloaders, RadioDataset
from train import get_model
from convert_tensorrt import export_onnx, build_engine

class BenchmarkResults:
    """Class to store and manage benchmark results with academic-grade metrics."""
    
    def __init__(self):
        self.results = {
            # Core performance metrics
            'models': [],
            'framework': [],
            'inference_latency_ms': [],
            'inference_throughput_fps': [],
            'preprocessing_latency_ms': [],
            'total_latency_ms': [],
            'accuracy_auc': [],
            'model_size_mb': [],
            'timestamp': [],
            
            # Enhanced memory metrics (academic standard)
            'gpu_memory_allocated_mb': [],
            'gpu_memory_peak_mb': [],
            'system_memory_used_mb': [],
            'system_memory_peak_mb': [],
            
            # Power and energy metrics (edge computing standard)
            'avg_power_w': [],
            'peak_power_w': [],
            'total_energy_j': [],
            'energy_per_inference_j': [],
            
            # Energy efficiency metrics (Jetson paper standard)
            'fps_per_watt': [],
            'efficiency_score': [],
            'power_efficiency_ratio': [],
            
            # System utilization metrics
            'avg_cpu_usage': [],
            'avg_gpu_usage': [],
            'monitoring_duration_s': []
        }
    
    def add_result(self, model_name, framework, inf_lat, inf_thr, prep_lat, 
                   total_lat, auc, size_mb, power_metrics=None, memory_metrics=None):
        """Add a benchmark result with enhanced academic metrics."""
        self.results['models'].append(model_name)
        self.results['framework'].append(framework)
        self.results['inference_latency_ms'].append(inf_lat)
        self.results['inference_throughput_fps'].append(inf_thr)
        self.results['preprocessing_latency_ms'].append(prep_lat)
        self.results['total_latency_ms'].append(total_lat)
        self.results['accuracy_auc'].append(auc)
        self.results['model_size_mb'].append(size_mb)
        self.results['timestamp'].append(datetime.now().isoformat())
        
        # Enhanced memory metrics
        if memory_metrics:
            self.results['gpu_memory_allocated_mb'].append(memory_metrics.get('gpu_memory_allocated_mb', 0))
            self.results['gpu_memory_peak_mb'].append(memory_metrics.get('gpu_memory_peak_mb', 0))
            self.results['system_memory_used_mb'].append(memory_metrics.get('system_memory_used_mb', 0))
            self.results['system_memory_peak_mb'].append(memory_metrics.get('system_memory_peak_mb', 0))
        else:
            self.results['gpu_memory_allocated_mb'].append(0)
            self.results['gpu_memory_peak_mb'].append(0)
            self.results['system_memory_used_mb'].append(0)
            self.results['system_memory_peak_mb'].append(0)
        
        # Power and energy metrics
        if power_metrics:
            avg_power = power_metrics.get('avg_power_w', 0)
            self.results['avg_power_w'].append(avg_power)
            self.results['peak_power_w'].append(power_metrics.get('peak_power_w', 0))
            self.results['total_energy_j'].append(power_metrics.get('total_energy_j', 0))
            self.results['energy_per_inference_j'].append(power_metrics.get('energy_per_inference_j', 0))
            self.results['avg_cpu_usage'].append(power_metrics.get('avg_cpu_usage', 0))
            self.results['avg_gpu_usage'].append(power_metrics.get('avg_gpu_usage', 0))
            self.results['monitoring_duration_s'].append(power_metrics.get('monitoring_duration_s', 0))
            
            # Calculate energy efficiency metrics
            efficiency = self._calculate_energy_efficiency(inf_thr, avg_power)
            self.results['fps_per_watt'].append(efficiency['fps_per_watt'])
            self.results['efficiency_score'].append(efficiency['efficiency_score'])
            self.results['power_efficiency_ratio'].append(efficiency['power_efficiency_ratio'])
        else:
            # Zero values for missing power metrics
            self.results['avg_power_w'].append(0)
            self.results['peak_power_w'].append(0)
            self.results['total_energy_j'].append(0)
            self.results['energy_per_inference_j'].append(0)
            self.results['fps_per_watt'].append(0)
            self.results['efficiency_score'].append(0)
            self.results['power_efficiency_ratio'].append(0)
            self.results['avg_cpu_usage'].append(0)
            self.results['avg_gpu_usage'].append(0)
            self.results['monitoring_duration_s'].append(0)
    
    def _calculate_energy_efficiency(self, throughput_fps, avg_power_w):
        """Calculate energy efficiency metrics."""
        if avg_power_w == 0:
            return {'fps_per_watt': 0, 'efficiency_score': 0, 'power_efficiency_ratio': 0}
            
        fps_per_watt = throughput_fps / avg_power_w
        efficiency_score = fps_per_watt * 100  # Scaled for readability
        power_efficiency_ratio = throughput_fps / avg_power_w
        
        return {
            'fps_per_watt': float(fps_per_watt),
            'efficiency_score': float(efficiency_score),
            'power_efficiency_ratio': float(power_efficiency_ratio)
        }
    
    def to_dataframe(self):
        """Convert results to pandas DataFrame."""
        return pd.DataFrame(self.results)
    
    def save_json(self, filepath):
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def load_json(self, filepath):
        """Load results from JSON file."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.results = json.load(f)

def measure_preprocessing_latency(dataset, num_samples=100):
    """Measure data preprocessing latency."""
    times = []
    
    # Time data loading and preprocessing
    for i in range(min(num_samples, len(dataset))):
        start_time = time.time()
        _ = dataset[i]  # This includes all preprocessing
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to ms
    
    return np.mean(times), np.std(times)

def measure_pytorch_inference(model, data_loader, device, num_batches=100):
    """
    Measure PyTorch model inference performance with academic-grade metrics.
    Includes power monitoring, detailed memory tracking, and energy efficiency.
    """
    model.to(device)
    model.eval()
    
    latencies = []
    gpu_memory_usage = []
    total_samples = 0
    
    # Initialize power monitoring if available
    power_monitor = None
    if POWER_MONITORING_AVAILABLE:
        power_monitor = JetsonPowerMonitor(sample_interval_ms=50)  # High-frequency sampling
        power_monitor.start_monitoring()
        print("🔋 Power monitoring started for PyTorch inference")
    
    # Get initial system state
    initial_memory = SystemResourceMonitor.get_system_memory_info() if POWER_MONITORING_AVAILABLE else {}
    initial_gpu_memory = SystemResourceMonitor.get_gpu_memory_info() if POWER_MONITORING_AVAILABLE else {}
    
    with torch.no_grad():
        for i, (x, _) in enumerate(data_loader):
            if i >= num_batches:
                break
                
            x = x.to(device)
            
            # Memory measurement
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            # Prepare input based on model type
            if hasattr(model, '__class__'):
                class_name = model.__class__.__name__
                # Flatten for models that use Linear layers (expect 1D input)
                if class_name in ['Autoencoder', 'AdversarialAutoencoder', 'FeedForwardNet']:
                    inp = x.view(x.size(0), -1)  # Flatten for feedforward models
                else:
                    inp = x  # Keep tensor format for CNN, LSTM, ResNet models (expect 3D input)
            else:
                inp = x.view(x.size(0), -1)  # Default to flatten
            
            # Measure inference time
            start_time = time.time()
            
            # Handle different model return patterns
            try:
                if hasattr(model, 'forward'):
                    # Check if model returns tuple or single value
                    output = model(inp)
                    if isinstance(output, tuple):
                        _ = output[0]  # Take first element if tuple
                    else:
                        _ = output
                else:
                    _ = model(inp)
            except Exception as e:
                print(f"Error during inference: {e}")
                continue
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            # Enhanced memory measurement
            if device.type == 'cuda':
                memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                gpu_memory_usage.append(memory_mb)
            
            total_samples += x.size(0)
    
    # Calculate performance metrics
    avg_latency = np.mean(latencies)
    throughput = 1000 / avg_latency if avg_latency > 0 else 0
    
    # Stop power monitoring and get comprehensive metrics
    power_metrics = {}
    memory_metrics = {}
    
    if power_monitor:
        power_metrics = power_monitor.stop_monitoring()
        print(f"🔋 PyTorch power monitoring completed: {power_metrics.get('avg_power_w', 0):.2f}W avg")
        
        # Get final system state for memory analysis
        final_memory = SystemResourceMonitor.get_system_memory_info()
        final_gpu_memory = SystemResourceMonitor.get_gpu_memory_info()
        
        memory_metrics = {
            'gpu_memory_allocated_mb': np.mean(gpu_memory_usage) if gpu_memory_usage else 0,
            'gpu_memory_peak_mb': np.max(gpu_memory_usage) if gpu_memory_usage else 0,
            'system_memory_used_mb': final_memory.get('system_memory_used_mb', 0),
            'system_memory_peak_mb': max(initial_memory.get('system_memory_used_mb', 0), 
                                       final_memory.get('system_memory_used_mb', 0))
        }
    else:
        # Fallback to basic GPU memory measurement
        memory_metrics = {
            'gpu_memory_allocated_mb': np.mean(gpu_memory_usage) if gpu_memory_usage else 0,
            'gpu_memory_peak_mb': np.max(gpu_memory_usage) if gpu_memory_usage else 0,
            'system_memory_used_mb': 0,
            'system_memory_peak_mb': 0
        }
    
    return avg_latency, throughput, power_metrics, memory_metrics
    
    avg_latency = np.mean(latencies)
    throughput = total_samples / (np.sum(latencies) / 1000)  # samples per second
    avg_memory = np.mean(memory_usage) if memory_usage else 0
    
    return avg_latency, throughput, avg_memory

def measure_tensorrt_inference(engine_path, data_loader, num_batches=100):
    """
    Measure TensorRT engine inference performance with academic-grade metrics.
    TensorRT 10.x compatible with power monitoring and energy efficiency.
    """
    if not TENSORRT_AVAILABLE:
        print("TensorRT not available. Skipping TensorRT benchmarking.")
        return 0.0, 0.0, {}, {}
    
    try:
        import pycuda.driver as cuda
        
        print(f"Loading TensorRT engine: {engine_path}")
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        
        with open(engine_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        if engine is None:
            print("Failed to load TensorRT engine")
            return 0.0, 0.0, {}, {}
        
        context = engine.create_execution_context()
        
        # Get TensorRT version
        trt_major = int(trt.__version__.split('.')[0])
        print(f"Using TensorRT {trt.__version__}")
        
        # Initialize power monitoring if available
        power_monitor = None
        if POWER_MONITORING_AVAILABLE:
            power_monitor = JetsonPowerMonitor(sample_interval_ms=50)
            power_monitor.start_monitoring()
            print("🔋 Power monitoring started for TensorRT inference")
        
        # Get initial system state
        initial_memory = SystemResourceMonitor.get_system_memory_info() if POWER_MONITORING_AVAILABLE else {}
        initial_gpu_memory = SystemResourceMonitor.get_gpu_memory_info() if POWER_MONITORING_AVAILABLE else {}
        
        # For TensorRT 10.x, use new API
        if trt_major >= 10:
            # Get input/output tensor names
            input_names = []
            output_names = []
            
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    input_names.append(name)
                else:
                    output_names.append(name)
            
            if not input_names:
                print("No input tensors found")
                return 0.0, 0.0, 0.0
                
            input_name = input_names[0]
            input_shape = engine.get_tensor_shape(input_name)
            input_dtype = trt.nptype(engine.get_tensor_dtype(input_name))
            
            print(f"Input: {input_name}, shape: {input_shape}, dtype: {input_dtype}")
            
            # Handle dynamic batch size
            if input_shape[0] == -1:
                input_shape = (1,) + input_shape[1:]
                context.set_input_shape(input_name, input_shape)
            
            # Allocate GPU memory
            input_size = np.prod(input_shape)
            input_nbytes = int(input_size * input_dtype().itemsize)  # Convert to Python int
            d_input = cuda.mem_alloc(input_nbytes)
            
            # Set tensor address
            context.set_tensor_address(input_name, int(d_input))
            
            if output_names:
                output_name = output_names[0]
                output_shape = engine.get_tensor_shape(output_name)
                output_dtype = trt.nptype(engine.get_tensor_dtype(output_name))
                output_size = np.prod(output_shape) if output_shape[0] != -1 else np.prod(output_shape[1:])
                output_nbytes = int(output_size * output_dtype().itemsize)  # Convert to Python int
                d_output = cuda.mem_alloc(output_nbytes)
                context.set_tensor_address(output_name, int(d_output))
        
        else:
            print("Legacy TensorRT versions not fully supported in this fix")
            return 0.0, 0.0, 0.0
        
        # Create CUDA stream
        stream = cuda.Stream()
        latencies = []
        total_samples = 0
        
        print(f"Running TensorRT inference on {min(num_batches, len(data_loader))} batches...")
        
        for batch_idx, (x, _) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
            
            try:
                # Process each sample in the batch
                for sample_idx in range(x.shape[0]):
                    sample = x[sample_idx:sample_idx+1]  # Keep batch dimension
                    
                    # Prepare input based on expected shape
                    if len(input_shape) == 2:  # Flattened input expected
                        sample_np = sample.view(1, -1).cpu().numpy().astype(input_dtype)
                    else:  # Keep original shape
                        sample_np = sample.cpu().numpy().astype(input_dtype)
                    
                    # Verify size matches
                    if sample_np.size != input_size:
                        print(f"Size mismatch: expected {input_size}, got {sample_np.size}")
                        continue
                    
                    # Measure inference time
                    start_time = time.time()
                    
                    # Copy input to GPU
                    cuda.memcpy_htod_async(d_input, sample_np.ravel(), stream)
                    
                    # Execute inference
                    context.execute_async_v3(stream.handle)
                    
                    # Wait for completion
                    stream.synchronize()
                    
                    end_time = time.time()
                    
                    latency_ms = (end_time - start_time) * 1000
                    latencies.append(latency_ms)
                    total_samples += 1
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        if not latencies:
            print("No successful TensorRT inferences")
            return 0.0, 0.0, {}, {}
        
        avg_latency = np.mean(latencies)
        throughput = total_samples / (np.sum(latencies) / 1000)
        
        print(f"TensorRT Results: {avg_latency:.2f}ms avg, {throughput:.2f} samples/sec")
        
        # Stop power monitoring and get comprehensive metrics
        power_metrics = {}
        memory_metrics = {}
        
        if power_monitor:
            power_metrics = power_monitor.stop_monitoring()
            print(f"🔋 TensorRT power monitoring completed: {power_metrics.get('avg_power_w', 0):.2f}W avg")
            
            # Get final system state for memory analysis
            final_memory = SystemResourceMonitor.get_system_memory_info()
            final_gpu_memory = SystemResourceMonitor.get_gpu_memory_info()
            
            memory_metrics = {
                'gpu_memory_allocated_mb': final_gpu_memory.get('gpu_memory_allocated_mb', 0),
                'gpu_memory_peak_mb': max(initial_gpu_memory.get('gpu_memory_allocated_mb', 0),
                                        final_gpu_memory.get('gpu_memory_allocated_mb', 0)),
                'system_memory_used_mb': final_memory.get('system_memory_used_mb', 0),
                'system_memory_peak_mb': max(initial_memory.get('system_memory_used_mb', 0),
                                           final_memory.get('system_memory_used_mb', 0))
            }
        else:
            # Fallback metrics
            memory_metrics = {
                'gpu_memory_allocated_mb': 0,
                'gpu_memory_peak_mb': 0,
                'system_memory_used_mb': 0,
                'system_memory_peak_mb': 0
            }
        
        # Cleanup
        if 'd_input' in locals():
            d_input.free()
        if 'd_output' in locals():
            d_output.free()
        
        return avg_latency, throughput, power_metrics, memory_metrics
        
    except Exception as e:
        print(f"TensorRT benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, 0.0, {}, {}

def measure_sustained_pytorch_inference(model, data_loader, duration_seconds=300, moving_window_seconds=30, device='cuda'):
    """
    Measure sustained PyTorch inference for academic rigor.
    
    Args:
        model: PyTorch model
        data_loader: Data loader for continuous inference
        duration_seconds: Total measurement duration (default: 5 minutes)
        moving_window_seconds: Moving average window (default: 30 seconds)
        device: Computing device
    
    Returns:
        dict: Comprehensive sustained measurements with time series data
    """
    print(f"🔬 Starting sustained PyTorch inference ({duration_seconds}s duration, {moving_window_seconds}s windows)")
    
    # Initialize tracking
    start_time = time.time()
    measurements = []
    power_monitor = None
    
    # Convert device string to torch.device if needed
    if isinstance(device, str):
        device = torch.device(device)
    
    try:
        # Start power monitoring with higher frequency for sustained measurements
        power_monitor = JetsonPowerMonitor(sample_interval_ms=100)  # 10 Hz sampling
        initial_memory = SystemResourceMonitor.get_system_memory_info()
        initial_gpu_memory = SystemResourceMonitor.get_gpu_memory_info()
        power_monitor.start_monitoring()
        
        # Ensure model is on the correct device
        model = model.to(device)
        model.eval()
        data_iter = iter(data_loader)
        
        # Continuous inference loop
        inference_count = 0
        window_start_time = start_time
        window_latencies = []
        window_power_samples = []
        
        while time.time() - start_time < duration_seconds:
            try:
                # Get next sample (cycle through dataset)
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(data_loader)  # Reset iterator
                    x, y = next(data_iter)
                
                x, y = x.to(device), y.to(device)
                
                # Prepare input based on model type (similar to regular PyTorch inference)
                if hasattr(model, '__class__'):
                    class_name = model.__class__.__name__
                    # Flatten for models that use Linear layers (expect 1D input)
                    if class_name in ['Autoencoder', 'AdversarialAutoencoder', 'FeedForwardNet']:
                        inp = x.view(x.size(0), -1)  # Flatten for feedforward models
                    else:
                        inp = x  # Keep tensor format for CNN, LSTM, ResNet models (expect 3D input)
                else:
                    inp = x.view(x.size(0), -1)  # Default to flatten
                
                # Single inference measurement
                inference_start = time.time()
                with torch.no_grad():
                    # Handle different model return patterns
                    try:
                        output = model(inp)
                        if isinstance(output, tuple):
                            _ = output[0]  # Take first element if tuple
                        else:
                            _ = output
                    except Exception as model_error:
                        print(f"Model inference error: {model_error}")
                        continue
                torch.cuda.synchronize() if str(device) == 'cuda' or 'cuda' in str(device) else None
                inference_end = time.time()
                
                latency_ms = (inference_end - inference_start) * 1000
                window_latencies.append(latency_ms)
                inference_count += 1
                
                # Check if we completed a moving window
                current_time = time.time()
                if current_time - window_start_time >= moving_window_seconds:
                    # Calculate window metrics
                    window_duration = current_time - window_start_time
                    window_avg_latency = np.mean(window_latencies)
                    window_throughput = len(window_latencies) / window_duration
                    
                    # Get current power reading (use most recent or estimate)
                    current_power = power_monitor.power_data[-1] if power_monitor.power_data else 5.0  # Fallback estimate
                    
                    # Get current memory usage
                    current_gpu_memory = SystemResourceMonitor.get_gpu_memory_info()
                    current_system_memory = SystemResourceMonitor.get_system_memory_info()
                    
                    # Store window measurement
                    measurement = {
                        'timestamp': current_time - start_time,
                        'window_duration_s': window_duration,
                        'avg_latency_ms': window_avg_latency,
                        'throughput_fps': window_throughput,
                        'power_w': current_power,
                        'fps_per_watt': window_throughput / current_power if current_power > 0 else 0,
                        'gpu_memory_mb': current_gpu_memory.get('gpu_memory_allocated_mb', 0),
                        'system_memory_mb': current_system_memory.get('system_memory_used_mb', 0),
                        'inferences_in_window': len(window_latencies)
                    }
                    measurements.append(measurement)
                    
                    print(f"⏱️  Window {len(measurements)}: {window_avg_latency:.2f}ms, {window_throughput:.1f}fps, {current_power:.2f}W")
                    
                    # Reset for next window
                    window_start_time = current_time
                    window_latencies = []
                
                # Brief pause to prevent overwhelming the system
                time.sleep(0.001)  # 1ms pause
                
            except Exception as e:
                print(f"Error in sustained inference: {e}")
                continue
        
        # Stop power monitoring
        final_power_metrics = power_monitor.stop_monitoring()
        
        # Calculate overall statistics
        if measurements:
            total_duration = time.time() - start_time
            avg_latency = np.mean([m['avg_latency_ms'] for m in measurements])
            avg_throughput = np.mean([m['throughput_fps'] for m in measurements])
            avg_power = np.mean([m['power_w'] for m in measurements])
            avg_fps_per_watt = np.mean([m['fps_per_watt'] for m in measurements if m['fps_per_watt'] > 0])
            
            result = {
                'sustained_duration_s': total_duration,
                'total_inferences': inference_count,
                'avg_latency_ms': avg_latency,
                'avg_throughput_fps': avg_throughput,
                'avg_power_w': avg_power,
                'avg_fps_per_watt': avg_fps_per_watt,
                'measurements': measurements,
                'power_metrics': final_power_metrics,
                'memory_metrics': {
                    'gpu_memory_allocated_mb': np.mean([m['gpu_memory_mb'] for m in measurements]),
                    'system_memory_used_mb': np.mean([m['system_memory_mb'] for m in measurements])
                }
            }
            
            print(f"✅ Sustained PyTorch completed: {avg_latency:.2f}ms avg, {avg_throughput:.1f}fps, {avg_power:.2f}W")
            return result
        else:
            print("❌ No sustained measurements collected")
            return {}
    
    except Exception as e:
        print(f"Sustained PyTorch benchmark failed: {e}")
        if power_monitor:
            power_monitor.stop_monitoring()
        return {}

def measure_sustained_tensorrt_inference(engine_path, data_loader, duration_seconds=300, moving_window_seconds=30):
    """
    Measure sustained TensorRT inference for academic rigor.
    
    Args:
        engine_path: Path to TensorRT engine
        data_loader: Data loader for continuous inference
        duration_seconds: Total measurement duration (default: 5 minutes)
        moving_window_seconds: Moving average window (default: 30 seconds)
    
    Returns:
        dict: Comprehensive sustained measurements with time series data
    """
    print(f"🔬 Starting sustained TensorRT inference ({duration_seconds}s duration, {moving_window_seconds}s windows)")
    
    # Initialize tracking
    start_time = time.time()
    measurements = []
    power_monitor = None
    
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        # Load TensorRT engine
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(engine_data)
        context = engine.create_execution_context()
        
        # Get input/output specifications using new TensorRT API
        input_names = []
        output_names = []
        
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_names.append(name)
            else:
                output_names.append(name)
        
        if not input_names:
            print("No input tensors found")
            return {}
            
        input_name = input_names[0]
        input_shape = engine.get_tensor_shape(input_name)
        input_dtype = trt.nptype(engine.get_tensor_dtype(input_name))
        
        print(f"Input: {input_name}, shape: {input_shape}, dtype: {input_dtype}")
        
        # Handle dynamic batch size
        if input_shape[0] == -1:
            input_shape = (1,) + input_shape[1:]
            context.set_input_shape(input_name, input_shape)
        
        # Allocate GPU memory
        input_size = np.prod(input_shape)
        input_nbytes = int(input_size * input_dtype().itemsize)
        d_input = cuda.mem_alloc(input_nbytes)
        
        # Set tensor address
        context.set_tensor_address(input_name, int(d_input))
        
        # Handle output tensors - must set ALL output tensors
        if not output_names:
            print("No output tensors found - this shouldn't happen")
            return {}
            
        for output_name in output_names:
            output_shape = engine.get_tensor_shape(output_name)
            output_dtype = trt.nptype(engine.get_tensor_dtype(output_name))
            print(f"Output: {output_name}, shape: {output_shape}, dtype: {output_dtype}")
            
            # Handle dynamic batch size for output
            if output_shape[0] == -1:
                output_shape = (1,) + output_shape[1:]
            
            output_size = np.prod(output_shape)
            output_nbytes = int(output_size * output_dtype().itemsize)
            d_output = cuda.mem_alloc(output_nbytes)
            context.set_tensor_address(output_name, int(d_output))
        
        # Create CUDA stream
        stream = cuda.Stream()
        
        # Start power monitoring
        power_monitor = JetsonPowerMonitor(sample_interval_ms=100)
        initial_memory = SystemResourceMonitor.get_system_memory_info()
        initial_gpu_memory = SystemResourceMonitor.get_gpu_memory_info()
        power_monitor.start_monitoring()
        
        data_iter = iter(data_loader)
        
        # Continuous inference loop
        inference_count = 0
        window_start_time = start_time
        window_latencies = []
        
        while time.time() - start_time < duration_seconds:
            try:
                # Get next sample
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(data_loader)
                    x, y = next(data_iter)
                
                # Prepare single sample for TensorRT
                sample = x[0] if x.size(0) > 0 else x
                
                # Prepare input based on expected shape
                if len(input_shape) == 2:
                    sample_np = sample.view(1, -1).cpu().numpy().astype(input_dtype)
                else:
                    sample_np = sample.cpu().numpy().astype(input_dtype)
                
                if sample_np.size != input_size:
                    continue
                
                # Single inference measurement
                inference_start = time.time()
                
                # Copy input to GPU
                cuda.memcpy_htod_async(d_input, sample_np.ravel(), stream)
                
                # Execute inference
                context.execute_async_v3(stream.handle)
                
                # Wait for completion
                stream.synchronize()
                
                inference_end = time.time()
                
                latency_ms = (inference_end - inference_start) * 1000
                window_latencies.append(latency_ms)
                inference_count += 1
                
                # Check if we completed a moving window
                current_time = time.time()
                if current_time - window_start_time >= moving_window_seconds:
                    # Calculate window metrics
                    window_duration = current_time - window_start_time
                    window_avg_latency = np.mean(window_latencies)
                    window_throughput = len(window_latencies) / window_duration
                    
                    # Get current power reading (use most recent or estimate)
                    current_power = power_monitor.power_data[-1] if power_monitor.power_data else 5.0  # Fallback estimate
                    
                    # Get current memory usage
                    current_gpu_memory = SystemResourceMonitor.get_gpu_memory_info()
                    current_system_memory = SystemResourceMonitor.get_system_memory_info()
                    
                    # Store window measurement
                    measurement = {
                        'timestamp': current_time - start_time,
                        'window_duration_s': window_duration,
                        'avg_latency_ms': window_avg_latency,
                        'throughput_fps': window_throughput,
                        'power_w': current_power,
                        'fps_per_watt': window_throughput / current_power if current_power > 0 else 0,
                        'gpu_memory_mb': current_gpu_memory.get('gpu_memory_allocated_mb', 0),
                        'system_memory_mb': current_system_memory.get('system_memory_used_mb', 0),
                        'inferences_in_window': len(window_latencies)
                    }
                    measurements.append(measurement)
                    
                    print(f"⏱️  Window {len(measurements)}: {window_avg_latency:.2f}ms, {window_throughput:.1f}fps, {current_power:.2f}W")
                    
                    # Reset for next window
                    window_start_time = current_time
                    window_latencies = []
                
                # Brief pause
                time.sleep(0.001)
                
            except Exception as e:
                print(f"Error in sustained TensorRT inference: {e}")
                continue
        
        # Cleanup
        d_input.free()
        d_output.free()
        
        # Stop power monitoring
        final_power_metrics = power_monitor.stop_monitoring()
        
        # Calculate overall statistics
        if measurements:
            total_duration = time.time() - start_time
            avg_latency = np.mean([m['avg_latency_ms'] for m in measurements])
            avg_throughput = np.mean([m['throughput_fps'] for m in measurements])
            avg_power = np.mean([m['power_w'] for m in measurements])
            avg_fps_per_watt = np.mean([m['fps_per_watt'] for m in measurements if m['fps_per_watt'] > 0])
            
            result = {
                'sustained_duration_s': total_duration,
                'total_inferences': inference_count,
                'avg_latency_ms': avg_latency,
                'avg_throughput_fps': avg_throughput,
                'avg_power_w': avg_power,
                'avg_fps_per_watt': avg_fps_per_watt,
                'measurements': measurements,
                'power_metrics': final_power_metrics,
                'memory_metrics': {
                    'gpu_memory_allocated_mb': np.mean([m['gpu_memory_mb'] for m in measurements]),
                    'system_memory_used_mb': np.mean([m['system_memory_mb'] for m in measurements])
                }
            }
            
            print(f"✅ Sustained TensorRT completed: {avg_latency:.2f}ms avg, {avg_throughput:.1f}fps, {avg_power:.2f}W")
            return result
        else:
            print("❌ No sustained measurements collected")
            return {}
    
    except Exception as e:
        print(f"Sustained TensorRT benchmark failed: {e}")
        if power_monitor:
            power_monitor.stop_monitoring()
        return {}

def thermal_cooldown(duration_seconds=60):
    """
    Thermal cooldown period between model benchmarks.
    Monitors temperature and power to ensure thermal equilibrium.
    """
    print(f"🌡️  Thermal cooldown period ({duration_seconds}s)...")
    
    start_time = time.time()
    power_monitor = JetsonPowerMonitor(sample_interval_ms=1000)  # 1 Hz for cooldown
    power_monitor.start_monitoring()
    
    try:
        while time.time() - start_time < duration_seconds:
            current_power = power_monitor.power_data[-1] if power_monitor.power_data else 5.0  # Fallback estimate
            elapsed = time.time() - start_time
            remaining = duration_seconds - elapsed
            
            print(f"     Cooldown: {remaining:.0f}s remaining, Power: {current_power:.2f}W", end='\r')
            time.sleep(5)  # Update every 5 seconds
        
        final_metrics = power_monitor.stop_monitoring()
        print(f"\n✅ Thermal cooldown complete. Idle power: {final_metrics.get('avg_power_w', 0):.2f}W")
        
    except Exception as e:
        print(f"Error during thermal cooldown: {e}")
        if power_monitor:
            power_monitor.stop_monitoring()

def get_model_size_mb(model_path):
    """Get model file size in MB."""
    if os.path.exists(model_path):
        return os.path.getsize(model_path) / (1024 * 1024)
    return 0.0

def get_model_accuracy_fast(model_name, model_path, window_size, device, val_loader):
    """Get model accuracy (AUC score) using pre-loaded dataset to avoid memory issues."""
    try:
        import torch
        from train import get_model
        from sklearn.metrics import roc_auc_score
        import numpy as np
        
        # Clear GPU memory before evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load model
        model = get_model(model_name, window_size)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        all_labels = []
        all_scores = []
        
        print(f"Evaluating {model_name} accuracy on validation set...")
        
        with torch.no_grad():
            for i, (data, labels) in enumerate(val_loader):
                if i > 200:  # Limit to first 200 batches to save memory
                    break
                    
                data = data.to(device)
                labels = labels.to(device)
                
                # Get reconstruction
                if model_name in ['aae']:
                    recon, _, _ = model(data)
                else:
                    recon = model(data)
                
                # Calculate reconstruction error
                mse = torch.mean((data - recon) ** 2, dim=(1, 2, 3))
                
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(mse.cpu().numpy())
        
        # Calculate AUC
        auc = roc_auc_score(all_labels, all_scores)
        
        # Cleanup
        del model, all_labels, all_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return auc
        
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        return 0.0

def convert_to_tensorrt(model_name, model_path, window_size, output_dir):
    """Convert PyTorch model to TensorRT engine."""
    if not TENSORRT_AVAILABLE:
        print(f"TensorRT not available. Skipping conversion for {model_name}")
        return None
    
    # Only certain models support TensorRT conversion
    supported_models = ['ae', 'ff']
    if model_name not in supported_models:
        print(f"TensorRT conversion not supported for {model_name}")
        return None
    
    try:
        from convert_tensorrt import main as convert_main
        from types import SimpleNamespace
        
        # Create conversion args
        convert_args = SimpleNamespace(
            model=model_name,
            weights_path=model_path,
            window_size=window_size,
            out_dir=output_dir
        )
        
        convert_main(convert_args)
        
        engine_path = os.path.join(output_dir, f"{model_name}.trt")
        if os.path.exists(engine_path):
            return engine_path
        else:
            print(f"TensorRT engine not found at {engine_path}")
            return None
            
    except Exception as e:
        print(f"TensorRT conversion failed for {model_name}: {e}")
        return None

def create_visualizations(results_df, output_dir):
    """Create comprehensive visualization plots."""
    # Fix: Use a matplotlib style that exists on all systems
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        try:
            plt.style.use('seaborn')  # Fallback to older seaborn style
        except OSError:
            plt.style.use('default')  # Ultimate fallback
            print("Using default matplotlib style (seaborn not available)")
    
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Inference Latency Comparison
    plt.figure(figsize=(12, 8))
    
    # Separate PyTorch and TensorRT results
    pytorch_df = results_df[results_df['framework'] == 'PyTorch']
    tensorrt_df = results_df[results_df['framework'] == 'TensorRT']
    
    x_pos = np.arange(len(pytorch_df))
    width = 0.35
    
    plt.bar(x_pos - width/2, pytorch_df['inference_latency_ms'], width, 
            label='PyTorch', alpha=0.8, color='skyblue')
    
    if not tensorrt_df.empty:
        # Match TensorRT results with PyTorch models
        trt_latencies = []
        for model in pytorch_df['models']:
            trt_row = tensorrt_df[tensorrt_df['models'] == model]
            if not trt_row.empty:
                trt_latencies.append(trt_row['inference_latency_ms'].iloc[0])
            else:
                trt_latencies.append(0)
        
        plt.bar(x_pos + width/2, trt_latencies, width, 
                label='TensorRT', alpha=0.8, color='lightcoral')
    
    plt.xlabel('Models')
    plt.ylabel('Inference Latency (ms)')
    plt.title('Inference Latency Comparison: PyTorch vs TensorRT')
    plt.xticks(x_pos, pytorch_df['models'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'inference_latency_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Throughput Comparison
    plt.figure(figsize=(12, 8))
    plt.bar(x_pos - width/2, pytorch_df['inference_throughput_fps'], width, 
            label='PyTorch', alpha=0.8, color='lightgreen')
    
    if not tensorrt_df.empty:
        trt_throughputs = []
        for model in pytorch_df['models']:
            trt_row = tensorrt_df[tensorrt_df['models'] == model]
            if not trt_row.empty:
                trt_throughputs.append(trt_row['inference_throughput_fps'].iloc[0])
            else:
                trt_throughputs.append(0)
        
        plt.bar(x_pos + width/2, trt_throughputs, width, 
                label='TensorRT', alpha=0.8, color='gold')
    
    plt.xlabel('Models')
    plt.ylabel('Throughput (samples/sec)')
    plt.title('Inference Throughput Comparison: PyTorch vs TensorRT')
    plt.xticks(x_pos, pytorch_df['models'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'throughput_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Accuracy vs Latency Trade-off
    plt.figure(figsize=(10, 8))
    for framework in results_df['framework'].unique():
        df_subset = results_df[results_df['framework'] == framework]
        plt.scatter(df_subset['inference_latency_ms'], df_subset['accuracy_auc'], 
                   label=framework, s=100, alpha=0.7)
        
        # Add model labels
        for i, row in df_subset.iterrows():
            plt.annotate(row['models'], 
                        (row['inference_latency_ms'], row['accuracy_auc']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('Inference Latency (ms)')
    plt.ylabel('Accuracy (AUC)')
    plt.title('Accuracy vs Latency Trade-off')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'accuracy_vs_latency.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Model Size vs Performance
    plt.figure(figsize=(12, 8))
    pytorch_only = results_df[results_df['framework'] == 'PyTorch']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Size vs Latency
    ax1.scatter(pytorch_only['model_size_mb'], pytorch_only['inference_latency_ms'], 
               s=100, alpha=0.7, color='purple')
    for i, row in pytorch_only.iterrows():
        ax1.annotate(row['models'], 
                    (row['model_size_mb'], row['inference_latency_ms']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax1.set_xlabel('Model Size (MB)')
    ax1.set_ylabel('Inference Latency (ms)')
    ax1.set_title('Model Size vs Inference Latency')
    ax1.grid(True, alpha=0.3)
    
    # Size vs Accuracy
    ax2.scatter(pytorch_only['model_size_mb'], pytorch_only['accuracy_auc'], 
               s=100, alpha=0.7, color='orange')
    for i, row in pytorch_only.iterrows():
        ax2.annotate(row['models'], 
                    (row['model_size_mb'], row['accuracy_auc']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax2.set_xlabel('Model Size (MB)')
    ax2.set_ylabel('Accuracy (AUC)')
    ax2.set_title('Model Size vs Accuracy')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'model_size_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Speedup Analysis (if TensorRT results available)
    if not tensorrt_df.empty:
        plt.figure(figsize=(10, 6))
        speedups = []
        model_names = []
        
        for model in pytorch_df['models']:
            pytorch_row = pytorch_df[pytorch_df['models'] == model]
            tensorrt_row = tensorrt_df[tensorrt_df['models'] == model]
            
            if not pytorch_row.empty and not tensorrt_row.empty:
                pytorch_lat = pytorch_row['inference_latency_ms'].iloc[0]
                tensorrt_lat = tensorrt_row['inference_latency_ms'].iloc[0]
                
                if tensorrt_lat > 0:
                    speedup = pytorch_lat / tensorrt_lat
                    speedups.append(speedup)
                    model_names.append(model)
        
        if speedups:
            plt.bar(model_names, speedups, color='lightblue', alpha=0.8)
            plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
            plt.xlabel('Models')
            plt.ylabel('Speedup (PyTorch Latency / TensorRT Latency)')
            plt.title('TensorRT Speedup over PyTorch')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'tensorrt_speedup.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # NEW ACADEMIC VISUALIZATIONS FOR POWER AND ENERGY EFFICIENCY
    
    # 6. Power Consumption Analysis
    power_df = results_df[results_df['avg_power_w'] > 0]  # Only rows with power data
    if not power_df.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Average Power Consumption by Model and Framework
        pytorch_power = power_df[power_df['framework'] == 'PyTorch']
        tensorrt_power = power_df[power_df['framework'] == 'TensorRT']
        
        if not pytorch_power.empty:
            # Get all unique models that have power data
            all_models = power_df['models'].unique()
            x_pos = np.arange(len(all_models))
            width = 0.35
            
            # Create arrays for PyTorch and TensorRT power data, matching model order
            pytorch_powers = []
            tensorrt_powers = []
            
            for model in all_models:
                # PyTorch power for this model
                pytorch_model_data = pytorch_power[pytorch_power['models'] == model]
                if not pytorch_model_data.empty:
                    pytorch_powers.append(pytorch_model_data['avg_power_w'].iloc[0])
                else:
                    pytorch_powers.append(0)
                
                # TensorRT power for this model
                tensorrt_model_data = tensorrt_power[tensorrt_power['models'] == model]
                if not tensorrt_model_data.empty:
                    tensorrt_powers.append(tensorrt_model_data['avg_power_w'].iloc[0])
                else:
                    tensorrt_powers.append(0)
            
            # Plot PyTorch bars
            ax1.bar(x_pos - width/2, pytorch_powers, width, 
                   label='PyTorch', alpha=0.8, color='steelblue')
            
            # Plot TensorRT bars (only for models that have TensorRT data)
            tensorrt_powers_filtered = [p for p in tensorrt_powers if p > 0]
            tensorrt_x_pos = [i for i, p in enumerate(tensorrt_powers) if p > 0]
            
            if tensorrt_powers_filtered:
                ax1.bar([x_pos[i] + width/2 for i in tensorrt_x_pos], tensorrt_powers_filtered, width, 
                       label='TensorRT', alpha=0.8, color='gold')
            
            ax1.set_xlabel('Models')
            ax1.set_ylabel('Average Power Consumption (W)')
            ax1.set_title('Power Consumption: PyTorch vs TensorRT')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(all_models, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Peak vs Average Power
        if not power_df.empty:
            ax2.scatter(power_df['avg_power_w'], power_df['peak_power_w'], 
                       c=power_df['framework'].map({'PyTorch': 'steelblue', 'TensorRT': 'gold'}),
                       s=100, alpha=0.7)
            
            # Add model labels
            for i, row in power_df.iterrows():
                ax2.annotate(f"{row['models']}\n({row['framework']})", 
                           (row['avg_power_w'], row['peak_power_w']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # Add diagonal line (peak = average)
            min_power = min(power_df['avg_power_w'].min(), power_df['peak_power_w'].min())
            max_power = max(power_df['avg_power_w'].max(), power_df['peak_power_w'].max())
            ax2.plot([min_power, max_power], [min_power, max_power], 
                    'k--', alpha=0.5, label='Peak = Average')
            
            ax2.set_xlabel('Average Power (W)')
            ax2.set_ylabel('Peak Power (W)')
            ax2.set_title('Peak vs Average Power Consumption')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'power_consumption_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 7. Energy Efficiency Analysis
    efficiency_df = results_df[results_df['fps_per_watt'] > 0]  # Only rows with efficiency data
    if not efficiency_df.empty:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Energy Efficiency (FPS/W) Comparison
        pytorch_eff = efficiency_df[efficiency_df['framework'] == 'PyTorch']
        tensorrt_eff = efficiency_df[efficiency_df['framework'] == 'TensorRT']
        
        if not pytorch_eff.empty:
            # Get all unique models that have efficiency data
            all_models = efficiency_df['models'].unique()
            x_pos = np.arange(len(all_models))
            width = 0.35
            
            # Create arrays for PyTorch and TensorRT efficiency data, matching model order
            pytorch_fps_per_w = []
            tensorrt_fps_per_w = []
            
            for model in all_models:
                # PyTorch efficiency for this model
                pytorch_model_data = pytorch_eff[pytorch_eff['models'] == model]
                if not pytorch_model_data.empty:
                    pytorch_fps_per_w.append(pytorch_model_data['fps_per_watt'].iloc[0])
                else:
                    pytorch_fps_per_w.append(0)
                
                # TensorRT efficiency for this model
                tensorrt_model_data = tensorrt_eff[tensorrt_eff['models'] == model]
                if not tensorrt_model_data.empty:
                    tensorrt_fps_per_w.append(tensorrt_model_data['fps_per_watt'].iloc[0])
                else:
                    tensorrt_fps_per_w.append(0)
            
            # Plot PyTorch bars
            ax1.bar(x_pos - width/2, pytorch_fps_per_w, width, 
                   label='PyTorch', alpha=0.8, color='steelblue')
            
            # Plot TensorRT bars (only for models that have TensorRT data)
            tensorrt_fps_filtered = [p for p in tensorrt_fps_per_w if p > 0]
            tensorrt_x_pos = [i for i, p in enumerate(tensorrt_fps_per_w) if p > 0]
            
            if tensorrt_fps_filtered:
                ax1.bar([x_pos[i] + width/2 for i in tensorrt_x_pos], tensorrt_fps_filtered, width, 
                       label='TensorRT', alpha=0.8, color='gold')
            
            ax1.set_xlabel('Models')
            ax1.set_ylabel('Energy Efficiency (FPS/W)')
            ax1.set_title('Energy Efficiency: PyTorch vs TensorRT')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(all_models, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Energy per Inference
        if not efficiency_df.empty:
            pytorch_energy = efficiency_df[efficiency_df['framework'] == 'PyTorch']
            tensorrt_energy = efficiency_df[efficiency_df['framework'] == 'TensorRT']
            
            if not pytorch_energy.empty:
                # Get all unique models that have energy data
                all_models = efficiency_df['models'].unique()
                x_pos = np.arange(len(all_models))
                width = 0.35
                
                # Create arrays for PyTorch and TensorRT energy data, matching model order
                pytorch_energy_per_inf = []
                tensorrt_energy_per_inf = []
                
                for model in all_models:
                    # PyTorch energy for this model
                    pytorch_model_data = pytorch_energy[pytorch_energy['models'] == model]
                    if not pytorch_model_data.empty:
                        # Calculate energy per inference from power and throughput
                        power_w = pytorch_model_data['avg_power_w'].iloc[0]
                        throughput_fps = pytorch_model_data['inference_throughput_fps'].iloc[0]
                        energy_per_inf = (power_w / throughput_fps * 1000) if throughput_fps > 0 else 0  # Convert to mJ
                        pytorch_energy_per_inf.append(energy_per_inf)
                    else:
                        pytorch_energy_per_inf.append(0)
                    
                    # TensorRT energy for this model
                    tensorrt_model_data = tensorrt_energy[tensorrt_energy['models'] == model]
                    if not tensorrt_model_data.empty:
                        power_w = tensorrt_model_data['avg_power_w'].iloc[0]
                        throughput_fps = tensorrt_model_data['inference_throughput_fps'].iloc[0]
                        energy_per_inf = (power_w / throughput_fps * 1000) if throughput_fps > 0 else 0  # Convert to mJ
                        tensorrt_energy_per_inf.append(energy_per_inf)
                    else:
                        tensorrt_energy_per_inf.append(0)
                
                # Plot PyTorch bars
                ax2.bar(x_pos - width/2, pytorch_energy_per_inf, width, 
                       label='PyTorch', alpha=0.8, color='steelblue')
                
                # Plot TensorRT bars (only for models that have TensorRT data)
                tensorrt_energy_filtered = [e for e in tensorrt_energy_per_inf if e > 0]
                tensorrt_x_pos = [i for i, e in enumerate(tensorrt_energy_per_inf) if e > 0]
                
                if tensorrt_energy_filtered:
                    ax2.bar([x_pos[i] + width/2 for i in tensorrt_x_pos], tensorrt_energy_filtered, width, 
                           label='TensorRT', alpha=0.8, color='gold')
                
                ax2.set_xlabel('Models')
                ax2.set_ylabel('Energy per Inference (mJ)')
                ax2.set_title('Energy Consumption per Inference')
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(all_models, rotation=45)
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        # Performance vs Power Trade-off
        ax3.scatter(efficiency_df[efficiency_df['framework'] == 'PyTorch']['avg_power_w'], 
                   efficiency_df[efficiency_df['framework'] == 'PyTorch']['inference_throughput_fps'],
                   label='PyTorch', s=100, alpha=0.7, color='steelblue')
        
        tensorrt_efficiency = efficiency_df[efficiency_df['framework'] == 'TensorRT']
        if not tensorrt_efficiency.empty:
            ax3.scatter(tensorrt_efficiency['avg_power_w'], 
                       tensorrt_efficiency['inference_throughput_fps'],
                       label='TensorRT', s=100, alpha=0.7, color='gold')
        
        # Add model labels
        for i, row in efficiency_df.iterrows():
            ax3.annotate(f"{row['models']}", 
                        (row['avg_power_w'], row['inference_throughput_fps']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax3.set_xlabel('Average Power Consumption (W)')
        ax3.set_ylabel('Inference Throughput (FPS)')
        ax3.set_title('Performance vs Power Trade-off')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # System Utilization Heatmap
        utilization_data = efficiency_df[['models', 'framework', 'avg_cpu_usage', 'avg_gpu_usage']].copy()
        if not utilization_data.empty:
            # Create pivot table for heatmap
            utilization_pivot = utilization_data.pivot_table(
                index='models', 
                columns='framework', 
                values=['avg_cpu_usage', 'avg_gpu_usage'],
                aggfunc='mean'
            )
            
            # Plot CPU and GPU utilization as grouped bars
            if len(utilization_data['models'].unique()) > 0:
                models = utilization_data['models'].unique()
                x_pos = np.arange(len(models))
                width = 0.2
                
                for i, framework in enumerate(['PyTorch', 'TensorRT']):
                    framework_data = utilization_data[utilization_data['framework'] == framework]
                    if not framework_data.empty:
                        cpu_usage = [framework_data[framework_data['models'] == model]['avg_cpu_usage'].values[0] 
                                   if len(framework_data[framework_data['models'] == model]) > 0 else 0 
                                   for model in models]
                        gpu_usage = [framework_data[framework_data['models'] == model]['avg_gpu_usage'].values[0] 
                                   if len(framework_data[framework_data['models'] == model]) > 0 else 0 
                                   for model in models]
                        
                        offset = (i - 0.5) * width
                        ax4.bar(x_pos + offset, cpu_usage, width/2, 
                               label=f'{framework} CPU', alpha=0.8, 
                               color='lightblue' if framework == 'PyTorch' else 'lightyellow')
                        ax4.bar(x_pos + offset + width/2, gpu_usage, width/2, 
                               label=f'{framework} GPU', alpha=0.8,
                               color='darkblue' if framework == 'PyTorch' else 'orange')
                
                ax4.set_xlabel('Models')
                ax4.set_ylabel('Utilization (%)')
                ax4.set_title('CPU and GPU Utilization')
                ax4.set_xticks(x_pos)
                ax4.set_xticklabels(models, rotation=45)
                ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'energy_efficiency_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 8. Academic Summary Dashboard
    if not power_df.empty:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Comprehensive Performance Overview
        pytorch_summary = results_df[results_df['framework'] == 'PyTorch']
        if not pytorch_summary.empty:
            models = pytorch_summary['models'].values
            
            # Radar chart data preparation
            metrics = ['Latency (inv)', 'Throughput', 'Power Eff', 'Memory Eff']
            
            # Normalize metrics for radar chart (0-1 scale)
            latency_norm = 1 - (pytorch_summary['inference_latency_ms'] / pytorch_summary['inference_latency_ms'].max())
            throughput_norm = pytorch_summary['inference_throughput_fps'] / pytorch_summary['inference_throughput_fps'].max()
            power_eff_norm = pytorch_summary['fps_per_watt'] / pytorch_summary['fps_per_watt'].max() if pytorch_summary['fps_per_watt'].max() > 0 else 0
            memory_eff_norm = 1 - (pytorch_summary['gpu_memory_allocated_mb'] / pytorch_summary['gpu_memory_allocated_mb'].max()) if pytorch_summary['gpu_memory_allocated_mb'].max() > 0 else 0
            
            # Multi-metric scatter plot
            ax1.scatter(pytorch_summary['inference_latency_ms'], pytorch_summary['fps_per_watt'], 
                       s=100, alpha=0.7, color='steelblue', label='PyTorch')
            
            for i, model in enumerate(models):
                ax1.annotate(model, 
                           (pytorch_summary.iloc[i]['inference_latency_ms'], 
                            pytorch_summary.iloc[i]['fps_per_watt']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax1.set_xlabel('Inference Latency (ms) - Lower is Better')
            ax1.set_ylabel('Energy Efficiency (FPS/W) - Higher is Better')
            ax1.set_title('Latency vs Energy Efficiency Trade-off')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        # TensorRT Acceleration Benefits
        tensorrt_summary = results_df[results_df['framework'] == 'TensorRT']
        if not tensorrt_summary.empty and not pytorch_summary.empty:
            # Calculate improvements
            improvements = []
            for model in tensorrt_summary['models'].unique():
                pytorch_row = pytorch_summary[pytorch_summary['models'] == model]
                tensorrt_row = tensorrt_summary[tensorrt_summary['models'] == model]
                
                if not pytorch_row.empty and not tensorrt_row.empty:
                    latency_improvement = (pytorch_row['inference_latency_ms'].iloc[0] - 
                                         tensorrt_row['inference_latency_ms'].iloc[0]) / pytorch_row['inference_latency_ms'].iloc[0] * 100
                    power_improvement = (tensorrt_row['fps_per_watt'].iloc[0] - 
                                       pytorch_row['fps_per_watt'].iloc[0]) / pytorch_row['fps_per_watt'].iloc[0] * 100 if pytorch_row['fps_per_watt'].iloc[0] > 0 else 0
                    
                    improvements.append({
                        'model': model,
                        'latency_improvement': latency_improvement,
                        'power_improvement': power_improvement
                    })
            
            if improvements:
                imp_df = pd.DataFrame(improvements)
                x_pos = np.arange(len(imp_df))
                width = 0.35
                
                ax2.bar(x_pos - width/2, imp_df['latency_improvement'], width, 
                       label='Latency Improvement (%)', alpha=0.8, color='green')
                ax2.bar(x_pos + width/2, imp_df['power_improvement'], width, 
                       label='Energy Efficiency Improvement (%)', alpha=0.8, color='orange')
                
                ax2.set_xlabel('Models')
                ax2.set_ylabel('Improvement (%)')
                ax2.set_title('TensorRT Acceleration Benefits')
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(imp_df['model'], rotation=45)
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Memory Usage Analysis
        memory_summary = results_df[results_df['gpu_memory_allocated_mb'] > 0]
        if not memory_summary.empty:
            pytorch_mem = memory_summary[memory_summary['framework'] == 'PyTorch']
            tensorrt_mem = memory_summary[memory_summary['framework'] == 'TensorRT']
            
            if not pytorch_mem.empty:
                # Get all unique models that have memory data
                all_models = memory_summary['models'].unique()
                x_pos = np.arange(len(all_models))
                width = 0.35
                
                # Create arrays for PyTorch memory data, matching model order
                pytorch_gpu_mem = []
                pytorch_sys_mem = []
                
                for model in all_models:
                    pytorch_model_data = pytorch_mem[pytorch_mem['models'] == model]
                    if not pytorch_model_data.empty:
                        pytorch_gpu_mem.append(pytorch_model_data['gpu_memory_allocated_mb'].iloc[0])
                        pytorch_sys_mem.append(pytorch_model_data['system_memory_used_mb'].iloc[0])
                    else:
                        pytorch_gpu_mem.append(0)
                        pytorch_sys_mem.append(0)
                
                ax3.bar(x_pos - width/2, pytorch_gpu_mem, width, 
                       label='PyTorch GPU Memory', alpha=0.8, color='steelblue')
                ax3.bar(x_pos - width/2, pytorch_sys_mem, width, 
                       bottom=pytorch_gpu_mem, label='PyTorch System Memory', alpha=0.6, color='lightblue')
                
                if not tensorrt_mem.empty:
                    # Create arrays for TensorRT memory data, matching model order
                    tensorrt_gpu_mem = []
                    tensorrt_sys_mem = []
                    
                    for model in all_models:
                        tensorrt_model_data = tensorrt_mem[tensorrt_mem['models'] == model]
                        if not tensorrt_model_data.empty:
                            tensorrt_gpu_mem.append(tensorrt_model_data['gpu_memory_allocated_mb'].iloc[0])
                            tensorrt_sys_mem.append(tensorrt_model_data['system_memory_used_mb'].iloc[0])
                        else:
                            tensorrt_gpu_mem.append(0)
                            tensorrt_sys_mem.append(0)
                    
                    # Only plot TensorRT bars for models that have TensorRT data
                    tensorrt_gpu_filtered = []
                    tensorrt_sys_filtered = []
                    tensorrt_x_pos = []
                    
                    for i, (gpu_mem, sys_mem) in enumerate(zip(tensorrt_gpu_mem, tensorrt_sys_mem)):
                        if gpu_mem > 0 or sys_mem > 0:  # Model has TensorRT data
                            tensorrt_gpu_filtered.append(gpu_mem)
                            tensorrt_sys_filtered.append(sys_mem)
                            tensorrt_x_pos.append(i)
                    
                    if tensorrt_gpu_filtered:
                        tensorrt_x_positions = [x_pos[i] + width/2 for i in tensorrt_x_pos]
                        ax3.bar(tensorrt_x_positions, tensorrt_gpu_filtered, width, 
                               label='TensorRT GPU Memory', alpha=0.8, color='gold')
                        ax3.bar(tensorrt_x_positions, tensorrt_sys_filtered, width, 
                               bottom=tensorrt_gpu_filtered, label='TensorRT System Memory', alpha=0.6, color='lightyellow')
                
                ax3.set_xlabel('Models')
                ax3.set_ylabel('Memory Usage (MB)')
                ax3.set_title('Memory Usage Comparison')
                ax3.set_xticks(x_pos)
                ax3.set_xticklabels(all_models, rotation=45)
                ax3.legend()
                ax3.grid(True, alpha=0.3)
        
        # Model Complexity vs Efficiency
        complexity_df = results_df[results_df['model_size_mb'] > 0]
        if not complexity_df.empty:
            # Create bubble chart: x=model_size, y=fps_per_watt, bubble_size=throughput
            pytorch_complexity = complexity_df[complexity_df['framework'] == 'PyTorch']
            tensorrt_complexity = complexity_df[complexity_df['framework'] == 'TensorRT']
            
            if not pytorch_complexity.empty:
                ax4.scatter(pytorch_complexity['model_size_mb'], 
                           pytorch_complexity['fps_per_watt'],
                           s=pytorch_complexity['inference_throughput_fps'] * 2,  # Size proportional to throughput
                           alpha=0.6, color='steelblue', label='PyTorch')
            
            if not tensorrt_complexity.empty:
                ax4.scatter(tensorrt_complexity['model_size_mb'], 
                           tensorrt_complexity['fps_per_watt'],
                           s=tensorrt_complexity['inference_throughput_fps'] * 2,
                           alpha=0.6, color='gold', label='TensorRT')
            
            # Add model labels
            for i, row in complexity_df.iterrows():
                ax4.annotate(row['models'], 
                           (row['model_size_mb'], row['fps_per_watt']),
                           xytext=(5, 5), textcoords='offset points', fontsize=7)
            
            ax4.set_xlabel('Model Size (MB)')
            ax4.set_ylabel('Energy Efficiency (FPS/W)')
            ax4.set_title('Model Complexity vs Energy Efficiency\n(Bubble size = Throughput)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Academic Benchmark Dashboard: Edge AI Performance Analysis', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'academic_dashboard.png'), dpi=300, bbox_inches='tight')
        plt.close()

def create_sustained_visualizations(sustained_results, output_dir):
    """
    Create visualizations specifically for sustained inference measurements.
    Shows time series data, thermal stability, and moving averages.
    """
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    if not sustained_results:
        print("No sustained measurement data available for visualization")
        return
    
    print("Creating sustained measurement visualizations...")
    
    # Create distinct color palette for each model/framework combination
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    # Generate distinct colors for each model/framework combination
    all_combinations = []
    for framework, models_data in sustained_results.items():
        for model_name in models_data.keys():
            all_combinations.append(f"{model_name}_{framework}")
    
    # Use a colormap that provides visually distinct colors
    if len(all_combinations) <= 10:
        # Use tab10 for up to 10 combinations (most distinct)
        base_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    elif len(all_combinations) <= 20:
        # Use tab20 for up to 20 combinations
        base_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    else:
        # Use hsv for larger numbers
        base_colors = plt.cm.hsv(np.linspace(0, 1, len(all_combinations)))
    
    # Create color mapping
    color_map = {}
    for i, combo in enumerate(all_combinations):
        color_map[combo] = base_colors[i % len(base_colors)]
    
    # 1. Time Series Analysis Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    line_styles = {'PyTorch': '-', 'TensorRT': '--'}  # Different line styles for frameworks
    markers = {'PyTorch': 'o', 'TensorRT': 's'}  # Different markers for frameworks
    
    for framework, models_data in sustained_results.items():
        for model_name, data in models_data.items():
            if 'measurements' not in data or not data['measurements']:
                continue
                
            measurements = data['measurements']
            timestamps = [m['timestamp'] for m in measurements]
            latencies = [m['avg_latency_ms'] for m in measurements]
            throughputs = [m['throughput_fps'] for m in measurements]
            powers = [m['power_w'] for m in measurements]
            fps_per_watts = [m['fps_per_watt'] for m in measurements]
            
            combo_key = f"{model_name}_{framework}"
            color = color_map.get(combo_key, 'gray')
            line_style = line_styles.get(framework, '-')
            marker = markers.get(framework, 'o')
            label = f"{model_name} ({framework})"
            
            # Latency over time
            axes[0,0].plot(timestamps, latencies, marker=marker, alpha=0.8, 
                          color=color, label=label, markersize=4, linestyle=line_style, linewidth=2)
            
            # Throughput over time
            axes[0,1].plot(timestamps, throughputs, marker=marker, alpha=0.8, 
                          color=color, label=label, markersize=4, linestyle=line_style, linewidth=2)
            
            # Power consumption over time
            axes[1,0].plot(timestamps, powers, marker=marker, alpha=0.8, 
                          color=color, label=label, markersize=4, linestyle=line_style, linewidth=2)
            
            # Energy efficiency over time
            axes[1,1].plot(timestamps, fps_per_watts, marker=marker, alpha=0.8, 
                          color=color, label=label, markersize=4, linestyle=line_style, linewidth=2)
    
    # Configure subplots
    axes[0,0].set_title('Inference Latency Over Time')
    axes[0,0].set_xlabel('Time (seconds)')
    axes[0,0].set_ylabel('Latency (ms)')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    axes[0,1].set_title('Throughput Over Time')
    axes[0,1].set_xlabel('Time (seconds)')
    axes[0,1].set_ylabel('Throughput (FPS)')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    axes[1,0].set_title('Power Consumption Over Time')
    axes[1,0].set_xlabel('Time (seconds)')
    axes[1,0].set_ylabel('Power (W)')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    axes[1,1].set_title('Energy Efficiency Over Time')
    axes[1,1].set_xlabel('Time (seconds)')
    axes[1,1].set_ylabel('FPS per Watt')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle('Sustained Inference Time Series Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'sustained_time_series.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Thermal Stability Analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Power stability analysis
    for framework, models_data in sustained_results.items():
        for model_name, data in models_data.items():
            if 'measurements' not in data or not data['measurements']:
                continue
                
            measurements = data['measurements']
            powers = [m['power_w'] for m in measurements]
            timestamps = [m['timestamp'] for m in measurements]
            
            if len(powers) < 2:
                continue
                
            combo_key = f"{model_name}_{framework}"
            color = color_map.get(combo_key, 'gray')
            line_style = line_styles.get(framework, '-')
            marker = markers.get(framework, 'o')
            
            # Power variation over time
            ax1.plot(timestamps, powers, alpha=0.8, color=color, linestyle=line_style, linewidth=2,
                    label=f"{model_name} ({framework})")
            
            # Power distribution histogram
            ax2.hist(powers, bins=20, alpha=0.6, color=color, 
                    label=f"{model_name} ({framework})", density=True)
            
            # Power stability metrics
            power_mean = np.mean(powers)
            power_std = np.std(powers)
            power_cv = power_std / power_mean * 100  # Coefficient of variation
            
            ax3.bar([f"{model_name}\n({framework})"], [power_cv], 
                   color=color, alpha=0.8)
            
            # Rolling statistics
            window_size = max(3, len(powers) // 10)
            if len(powers) >= window_size:
                rolling_mean = np.convolve(powers, np.ones(window_size)/window_size, mode='valid')
                rolling_timestamps = timestamps[window_size-1:]
                
                # Ensure arrays have same length
                min_length = min(len(rolling_timestamps), len(rolling_mean))
                rolling_timestamps = rolling_timestamps[:min_length]
                rolling_mean = rolling_mean[:min_length]
                
                if len(rolling_timestamps) > 0 and len(rolling_mean) > 0:
                    ax4.plot(rolling_timestamps, rolling_mean, color=color, linewidth=3,
                            linestyle=line_style, alpha=0.9,
                            label=f"{model_name} ({framework}) Rolling Mean")
    
    ax1.set_title('Power Consumption Stability')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Power (W)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Power Distribution')
    ax2.set_xlabel('Power (W)')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3.set_title('Power Coefficient of Variation (%)')
    ax3.set_ylabel('CV (%)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    ax4.set_title('Rolling Average Power')
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Power (W)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Thermal Stability and Power Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'sustained_thermal_stability.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Performance Consistency Analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Performance variability metrics
    performance_stats = []
    
    for framework, models_data in sustained_results.items():
        for model_name, data in models_data.items():
            if 'measurements' not in data or not data['measurements']:
                continue
                
            measurements = data['measurements']
            latencies = [m['avg_latency_ms'] for m in measurements]
            throughputs = [m['throughput_fps'] for m in measurements]
            fps_per_watts = [m['fps_per_watt'] for m in measurements]
            
            if len(latencies) < 2:
                continue
            
            stats = {
                'model': model_name,
                'framework': framework,
                'latency_mean': np.mean(latencies),
                'latency_std': np.std(latencies),
                'latency_cv': np.std(latencies) / np.mean(latencies) * 100,
                'throughput_mean': np.mean(throughputs),
                'throughput_std': np.std(throughputs),
                'throughput_cv': np.std(throughputs) / np.mean(throughputs) * 100,
                'efficiency_mean': np.mean(fps_per_watts),
                'efficiency_std': np.std(fps_per_watts),
                'efficiency_cv': np.std(fps_per_watts) / np.mean(fps_per_watts) * 100 if np.mean(fps_per_watts) > 0 else 0
            }
            performance_stats.append(stats)
    
    if performance_stats:
        stats_df = pd.DataFrame(performance_stats)
        
        # Latency consistency
        pytorch_stats = stats_df[stats_df['framework'] == 'PyTorch']
        tensorrt_stats = stats_df[stats_df['framework'] == 'TensorRT']
        
        if not pytorch_stats.empty:
            ax1.bar(pytorch_stats['model'], pytorch_stats['latency_cv'], 
                   alpha=0.8, color='steelblue', label='PyTorch')
        if not tensorrt_stats.empty:
            ax1.bar(tensorrt_stats['model'], tensorrt_stats['latency_cv'], 
                   alpha=0.8, color='gold', label='TensorRT')
        
        ax1.set_title('Latency Consistency (Lower CV = More Consistent)')
        ax1.set_ylabel('Coefficient of Variation (%)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Throughput consistency
        if not pytorch_stats.empty:
            ax2.bar(pytorch_stats['model'], pytorch_stats['throughput_cv'], 
                   alpha=0.8, color='steelblue', label='PyTorch')
        if not tensorrt_stats.empty:
            ax2.bar(tensorrt_stats['model'], tensorrt_stats['throughput_cv'], 
                   alpha=0.8, color='gold', label='TensorRT')
        
        ax2.set_title('Throughput Consistency (Lower CV = More Consistent)')
        ax2.set_ylabel('Coefficient of Variation (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Efficiency consistency
        if not pytorch_stats.empty:
            ax3.bar(pytorch_stats['model'], pytorch_stats['efficiency_cv'], 
                   alpha=0.8, color='steelblue', label='PyTorch')
        if not tensorrt_stats.empty:
            ax3.bar(tensorrt_stats['model'], tensorrt_stats['efficiency_cv'], 
                   alpha=0.8, color='gold', label='TensorRT')
        
        ax3.set_title('Energy Efficiency Consistency')
        ax3.set_ylabel('Coefficient of Variation (%)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Mean vs Std scatter plot
        if not stats_df.empty:
            for framework in ['PyTorch', 'TensorRT']:
                framework_data = stats_df[stats_df['framework'] == framework]
                if not framework_data.empty:
                    # Use framework-specific colors for scatter plot
                    framework_color = 'steelblue' if framework == 'PyTorch' else 'gold'
                    ax4.scatter(framework_data['latency_mean'], framework_data['latency_std'],
                              s=100, alpha=0.7, color=framework_color, label=framework)
                    
                    for _, row in framework_data.iterrows():
                        ax4.annotate(row['model'], 
                                   (row['latency_mean'], row['latency_std']),
                                   xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax4.set_xlabel('Mean Latency (ms)')
            ax4.set_ylabel('Latency Std Dev (ms)')
            ax4.set_title('Latency Mean vs Standard Deviation')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Performance Consistency Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'sustained_consistency_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Sustained vs Standard Benchmark Comparison (if standard results available)
    # This would compare sustained measurements with the quick benchmark results
    # Implementation depends on having both sustained and standard results available
    
    print(f"Sustained measurement visualizations saved to {fig_dir}")
    
    print(f"Visualizations saved to {fig_dir}")

def benchmark_all_models(args):
    """Main benchmarking function with support for sustained academic measurements."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on device: {device}")
    
    # Initialize results
    results = BenchmarkResults()
    sustained_results = {'PyTorch': {}, 'TensorRT': {}} if args.sustained else None
    
    # Output directories
    engine_dir = os.path.join(args.output_dir, 'engines')
    weights_dir = args.weights_dir if args.weights_dir else os.path.join(args.output_dir, 'weights')
    os.makedirs(engine_dir, exist_ok=True)
    if not args.weights_dir:
        os.makedirs(weights_dir, exist_ok=True)
    
    # Models to benchmark
    models_to_test = args.models
    
    # Configure benchmark mode
    if args.sustained:
        print(f"🔬 SUSTAINED ACADEMIC BENCHMARKING MODE")
        print(f"   Duration per model: {args.sustained_duration}s ({args.sustained_duration/60:.1f} minutes)")
        print(f"   Moving window: {args.moving_window}s")
        print(f"   Thermal cooldown: {args.thermal_cooldown}s")
        print(f"   Total estimated time: {len(models_to_test) * (args.sustained_duration * 2 + args.thermal_cooldown) / 60:.1f} minutes")
        print(f"   This will provide rigorous academic-grade measurements!")
        
        # Confirm with user for long benchmarks
        total_time_hours = len(models_to_test) * (args.sustained_duration * 2 + args.thermal_cooldown) / 3600
        if total_time_hours > 1:
            print(f"⚠️  WARNING: This benchmark will take approximately {total_time_hours:.1f} hours")
    else:
        print(f"📊 STANDARD BENCHMARKING MODE")
        print(f"   Batches per model: {args.num_batches}")
    
    print("📊 Loading dataset ONCE for all benchmarks...")
    
    # Load data ONCE and reuse for all operations
    # This prevents multiple dataset loading which causes OOM on Jetson
    train_loader, val_loader = get_dataloaders(
        args.clean, args.jammed,
        window_size=args.window_size,
        batch_size=args.batch_size,  # Use configurable batch size
        num_workers=0,  # No multiprocessing for consistent timing
        max_samples=args.max_samples  # Limit dataset size to prevent OOM
    )
    
    # Use the validation dataset for preprocessing measurement too
    print("Measuring preprocessing latency...")
    prep_latency, prep_std = measure_preprocessing_latency(val_loader.dataset, num_samples=100)
    print(f"Preprocessing: {prep_latency:.2f} ± {prep_std:.2f} ms")
    
    for i, model_name in enumerate(models_to_test):
        print(f"\n{'='*60}")
        print(f"Benchmarking {model_name.upper()} ({i+1}/{len(models_to_test)})")
        print(f"{'='*60}")
        
        model_path = os.path.join(weights_dir, f"{model_name}_best.pth")
        
        if not os.path.exists(model_path):
            print(f"Model weights not found: {model_path}")
            continue
        
        try:
            # Load model
            model = get_model(model_name, args.window_size)
            model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
            
            # Get model accuracy (skip if requested to save memory or if fewer than 50 batches - likely Jetson mode)
            if hasattr(args, 'skip_accuracy') and args.skip_accuracy:
                print("Skipping accuracy evaluation (memory saving mode)")
                auc_score = 0.0
            elif args.num_batches < 50:  # Auto-skip for Jetson mode (small batch counts)
                print("Skipping accuracy evaluation (Jetson memory saving mode)")
                auc_score = 0.0
            else:
                print("Getting model accuracy...")
                auc_score = get_model_accuracy_fast(model_name, model_path, args.window_size, device, val_loader)
                if auc_score is None:
                    auc_score = 0.0
            
            # Get model size
            model_size_mb = get_model_size_mb(model_path)
            
            # PYTORCH BENCHMARKING
            print(f"\n🐍 PyTorch Benchmarking...")
            
            if args.sustained:
                # Sustained PyTorch measurement
                pt_sustained_data = measure_sustained_pytorch_inference(
                    model, val_loader,
                    duration_seconds=args.sustained_duration,
                    moving_window_seconds=args.moving_window,
                    device=str(device)
                )
                
                if pt_sustained_data:
                    sustained_results['PyTorch'][model_name] = pt_sustained_data
                    
                    # Extract summary metrics for compatibility with standard results
                    pt_latency = pt_sustained_data['avg_latency_ms']
                    pt_throughput = pt_sustained_data['avg_throughput_fps']
                    pt_power_metrics = {
                        'avg_power_w': pt_sustained_data['avg_power_w'],
                        'peak_power_w': max([m['power_w'] for m in pt_sustained_data['measurements']]) if pt_sustained_data['measurements'] else 0,
                        'fps_per_watt': pt_sustained_data['avg_fps_per_watt']
                    }
                    pt_memory_metrics = pt_sustained_data['memory_metrics']
                    
                    print(f"   ✅ Sustained PyTorch Results:")
                    print(f"      Duration: {pt_sustained_data['sustained_duration_s']:.1f}s")
                    print(f"      Total Inferences: {pt_sustained_data['total_inferences']}")
                    print(f"      Avg Latency: {pt_latency:.2f} ms")
                    print(f"      Avg Throughput: {pt_throughput:.2f} FPS")
                    print(f"      Avg Power: {pt_sustained_data['avg_power_w']:.2f} W")
                    print(f"      Energy Efficiency: {pt_sustained_data['avg_fps_per_watt']:.2f} FPS/W")
                else:
                    print("   ❌ Sustained PyTorch measurement failed")
                    continue
            else:
                # Standard PyTorch measurement
                pt_latency, pt_throughput, pt_power_metrics, pt_memory_metrics = measure_pytorch_inference(
                    model, val_loader, device, num_batches=args.num_batches
                )
                
                print(f"   PyTorch Results:")
                print(f"      Inference Latency: {pt_latency:.2f} ms")
                print(f"      Throughput: {pt_throughput:.2f} samples/sec")
                if pt_power_metrics.get('avg_power_w', 0) > 0:
                    print(f"      Power Consumption: {pt_power_metrics.get('avg_power_w', 0):.2f} W")
                    print(f"      Energy Efficiency: {pt_throughput / pt_power_metrics.get('avg_power_w', 1):.2f} FPS/W")
            
            total_latency_pt = prep_latency + pt_latency
            
            # Add PyTorch results with enhanced metrics
            results.add_result(
                model_name, 'PyTorch', pt_latency, pt_throughput, prep_latency,
                total_latency_pt, auc_score, model_size_mb, pt_power_metrics, pt_memory_metrics
            )
            
            # TENSORRT BENCHMARKING
            if args.convert_tensorrt:
                print(f"\n🚀 TensorRT Benchmarking...")
                
                # Thermal cooldown before TensorRT if in sustained mode
                if args.sustained:
                    thermal_cooldown(args.thermal_cooldown)
                
                print("   Converting to TensorRT...")
                engine_path = convert_to_tensorrt(model_name, model_path, args.window_size, engine_dir)
                
                if engine_path and os.path.exists(engine_path):
                    if args.sustained:
                        # Sustained TensorRT measurement
                        trt_sustained_data = measure_sustained_tensorrt_inference(
                            engine_path, val_loader,
                            duration_seconds=args.sustained_duration,
                            moving_window_seconds=args.moving_window
                        )
                        
                        if trt_sustained_data:
                            sustained_results['TensorRT'][model_name] = trt_sustained_data
                            
                            # Extract summary metrics
                            trt_latency = trt_sustained_data['avg_latency_ms']
                            trt_throughput = trt_sustained_data['avg_throughput_fps']
                            trt_power_metrics = {
                                'avg_power_w': trt_sustained_data['avg_power_w'],
                                'peak_power_w': max([m['power_w'] for m in trt_sustained_data['measurements']]) if trt_sustained_data['measurements'] else 0,
                                'fps_per_watt': trt_sustained_data['avg_fps_per_watt']
                            }
                            trt_memory_metrics = trt_sustained_data['memory_metrics']
                            
                            speedup = pt_latency / trt_latency if trt_latency > 0 else 0
                            
                            print(f"   ✅ Sustained TensorRT Results:")
                            print(f"      Duration: {trt_sustained_data['sustained_duration_s']:.1f}s")
                            print(f"      Total Inferences: {trt_sustained_data['total_inferences']}")
                            print(f"      Avg Latency: {trt_latency:.2f} ms")
                            print(f"      Avg Throughput: {trt_throughput:.2f} FPS")
                            print(f"      Avg Power: {trt_sustained_data['avg_power_w']:.2f} W")
                            print(f"      Energy Efficiency: {trt_sustained_data['avg_fps_per_watt']:.2f} FPS/W")
                            print(f"      Speedup: {speedup:.2f}x")
                            
                            # Get TensorRT engine size
                            engine_size_mb = get_model_size_mb(engine_path)
                            total_latency_trt = prep_latency + trt_latency
                            
                            # Add TensorRT results
                            results.add_result(
                                model_name, 'TensorRT', trt_latency, trt_throughput, prep_latency,
                                total_latency_trt, auc_score, engine_size_mb, trt_power_metrics, trt_memory_metrics
                            )
                        else:
                            print("   ❌ Sustained TensorRT measurement failed")
                    else:
                        # Standard TensorRT measurement
                        print("   Benchmarking TensorRT inference...")
                        trt_latency, trt_throughput, trt_power_metrics, trt_memory_metrics = measure_tensorrt_inference(
                            engine_path, val_loader, num_batches=args.num_batches
                        )
                        
                        if trt_latency > 0:
                            total_latency_trt = prep_latency + trt_latency
                            speedup = pt_latency / trt_latency if trt_latency > 0 else 0
                            
                            print(f"   TensorRT Results:")
                            print(f"      Inference Latency: {trt_latency:.2f} ms")
                            print(f"      Throughput: {trt_throughput:.2f} samples/sec")
                            print(f"      Speedup: {speedup:.2f}x")
                            if trt_power_metrics.get('avg_power_w', 0) > 0:
                                print(f"      Power Consumption: {trt_power_metrics.get('avg_power_w', 0):.2f} W")
                                print(f"      Energy Efficiency: {trt_throughput / trt_power_metrics.get('avg_power_w', 1):.2f} FPS/W")
                            
                            # Get TensorRT engine size
                            engine_size_mb = get_model_size_mb(engine_path)
                            
                            # Add TensorRT results
                            results.add_result(
                                model_name, 'TensorRT', trt_latency, trt_throughput, prep_latency,
                                total_latency_trt, auc_score, engine_size_mb, trt_power_metrics, trt_memory_metrics
                            )
                        else:
                            print("   TensorRT benchmarking failed")
                else:
                    print("   TensorRT conversion failed")
            
            # Thermal cooldown after each model in sustained mode
            if args.sustained and i < len(models_to_test) - 1:  # Don't cooldown after last model
                thermal_cooldown(args.thermal_cooldown)
            
        except Exception as e:
            print(f"Error benchmarking {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
        finally:
            # Memory cleanup after each model
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print(f"   Memory cleaned up after {model_name}")
    
    # Save results
    results_df = results.to_dataframe()
    results_path = os.path.join(args.output_dir, 'benchmark_results.json')
    csv_path = os.path.join(args.output_dir, 'benchmark_results.csv')
    
    results.save_json(results_path)
    results_df.to_csv(csv_path, index=False)
    
    # Save sustained results if available
    if args.sustained and sustained_results:
        sustained_path = os.path.join(args.output_dir, 'sustained_measurements.json')
        import json
        with open(sustained_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                return obj
            
            json.dump(sustained_results, f, indent=2, default=convert_numpy)
        print(f"Sustained measurements saved to: {sustained_path}")
    
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))
    
    # Create visualizations
    if not results_df.empty:
        print("\nGenerating standard visualizations...")
        create_visualizations(results_df, args.output_dir)
        
        # Create sustained visualizations if available
        if args.sustained and sustained_results:
            print("Generating sustained measurement visualizations...")
            create_sustained_visualizations(sustained_results, args.output_dir)
    
    print(f"\nResults saved to:")
    print(f"  JSON: {results_path}")
    print(f"  CSV: {csv_path}")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description='Comprehensive RF Signal Model Benchmarking')
    parser.add_argument('--models', nargs='+', 
                       default=['ae', 'cnn_ae', 'lstm_ae', 'resnet_ae', 'aae', 'ff'],
                       choices=['ae', 'cnn_ae', 'lstm_ae', 'resnet_ae', 'aae', 'ff'],
                       help='Models to benchmark')
    parser.add_argument('--clean', type=str, default='clean_5g_dataset.h5',
                       help='Clean dataset path')
    parser.add_argument('--jammed', type=str, default='jammed_5g_dataset.h5',
                       help='Jammed dataset path')
    parser.add_argument('--window-size', type=int, default=1024,
                       help='Window size for models')
    parser.add_argument('--num-batches', type=int, default=100,
                       help='Number of batches for benchmarking')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for data loading (default: 1 for latency measurement)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to load per dataset (for memory saving)')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                       help='Output directory for results')
    parser.add_argument('--weights-dir', type=str, default=None,
                       help='Directory containing model weights (default: <output-dir>/weights)')
    parser.add_argument('--convert-tensorrt', action='store_true',
                       help='Convert models to TensorRT and benchmark')
    parser.add_argument('--skip-accuracy', action='store_true',
                       help='Skip accuracy evaluation to save memory')
    parser.add_argument('--sustained', action='store_true',
                       help='Enable sustained inference mode for rigorous academic benchmarking')
    parser.add_argument('--sustained-duration', type=int, default=300,
                       help='Duration in seconds for sustained inference per model (default: 300s = 5 minutes)')
    parser.add_argument('--thermal-cooldown', type=int, default=60,
                       help='Thermal cooldown period between models in seconds (default: 60s)')
    parser.add_argument('--moving-window', type=int, default=30,
                       help='Moving average window size in seconds for sustained measurements (default: 30s)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run benchmarking
    benchmark_all_models(args)

if __name__ == '__main__':
    main()
