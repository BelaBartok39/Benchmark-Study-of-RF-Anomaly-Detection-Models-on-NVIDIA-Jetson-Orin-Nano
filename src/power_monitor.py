#!/usr/bin/env python3
"""
Power monitoring utilities for Jetson devices.
Provides academic-grade power consumption and energy efficiency metrics.
"""

import os
import time
import threading
import subprocess
import psutil
import numpy as np
from typing import Dict, List, Optional, Tuple

class JetsonPowerMonitor:
    """
    Power monitoring for Jetson devices using tegrastats.
    Provides real-time power consumption, memory usage, and energy efficiency metrics.
    """
    
    def __init__(self, sample_interval_ms: int = 100):
        """
        Initialize power monitor.
        
        Args:
            sample_interval_ms: Sampling interval in milliseconds (default: 100ms)
        """
        self.sample_interval_ms = sample_interval_ms
        self.sample_interval_s = sample_interval_ms / 1000.0
        self.is_monitoring = False
        self.power_data = []
        self.memory_data = []
        self.cpu_data = []
        self.gpu_data = []
        self.timestamps = []
        self.monitor_thread = None
        self.start_time = None
        
    def start_monitoring(self):
        """Start power monitoring in background thread."""
        if self.is_monitoring:
            print("Warning: Power monitoring already running")
            return
            
        self.is_monitoring = True
        self.power_data.clear()
        self.memory_data.clear()
        self.cpu_data.clear()
        self.gpu_data.clear()
        self.timestamps.clear()
        self.start_time = time.time()
        
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        print(f"🔋 Power monitoring started (sampling every {self.sample_interval_ms}ms)")
        
    def stop_monitoring(self) -> Dict:
        """
        Stop power monitoring and return comprehensive metrics.
        
        Returns:
            Dict containing power, memory, and efficiency metrics
        """
        if not self.is_monitoring:
            print("Warning: Power monitoring not running")
            return {}
            
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            
        total_time = time.time() - self.start_time
        
        # Calculate comprehensive metrics
        metrics = self._calculate_metrics(total_time)
        
        print(f"🔋 Power monitoring stopped. Duration: {total_time:.2f}s")
        print(f"📊 Avg Power: {metrics.get('avg_power_w', 0):.2f}W, "
              f"Peak Power: {metrics.get('peak_power_w', 0):.2f}W")
        
        return metrics
        
    def _monitoring_loop(self):
        """Background monitoring loop."""
        sample_count = 0
        while self.is_monitoring:
            try:
                # Get tegrastats data
                tegra_data = self._get_tegrastats_data()
                
                # Get system memory data using psutil
                memory_info = psutil.virtual_memory()
                
                current_time = time.time() - (self.start_time or 0)
                
                self.timestamps.append(current_time)
                self.power_data.append(tegra_data.get('power_w', 0))
                self.memory_data.append(memory_info.used / 1024 / 1024)  # MB
                self.cpu_data.append(tegra_data.get('cpu_usage', 0))
                self.gpu_data.append(tegra_data.get('gpu_usage', 0))
                
                sample_count += 1
                # Debug: print every few samples
                if sample_count % 5 == 1:  # Print first sample and every 5th sample
                    power_w = tegra_data.get('power_w', 0)
                    cpu_pct = tegra_data.get('cpu_usage', 0)
                    gpu_pct = tegra_data.get('gpu_usage', 0)
                    print(f"🔋 Sample {sample_count}: {power_w:.1f}W, CPU: {cpu_pct:.1f}%, GPU: {gpu_pct:.1f}%")
                
                time.sleep(self.sample_interval_s)
                
            except Exception as e:
                print(f"Power monitoring error: {e}")
                time.sleep(self.sample_interval_s)
                
    def _get_tegrastats_data(self) -> Dict:
        """Parse single tegrastats output with improved reliability."""
        try:
            # Use a simpler approach: get 2 samples quickly and use the second one
            result = subprocess.run(
                ['bash', '-c', 'timeout 0.8 tegrastats --interval 200 | tail -1'],
                capture_output=True, text=True, timeout=1.0
            )
            
            if result.stdout and result.stdout.strip():
                return self._parse_tegrastats_line(result.stdout.strip())
            
            # Fallback: Single quick sample with shorter timeout
            result = subprocess.run(
                ['tegrastats', '--interval', '100'],
                capture_output=True, text=True, timeout=0.3
            )
            
            if result.stdout:
                lines = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
                if lines:
                    # Use the last line which should be the most recent
                    parsed_data = self._parse_tegrastats_line(lines[-1])
                    # Debug: print successful parsing
                    if parsed_data.get('power_w', 0) > 0:
                        print(f"📊 Power sample: {parsed_data.get('power_w', 0):.1f}W, "
                              f"CPU: {parsed_data.get('cpu_usage', 0):.1f}%, "
                              f"GPU: {parsed_data.get('gpu_usage', 0):.1f}%")
                    return parsed_data
                    
            return {'power_w': 0.0, 'cpu_usage': 0.0, 'gpu_usage': 0.0}
                
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, Exception) as e:
            # Fallback: estimate power based on CPU/GPU activity from psutil
            cpu_percent = psutil.cpu_percent(interval=0)
            # Rough power estimation for Jetson Orin Nano: 4-6W idle, up to 15W under load
            estimated_power = 4.5 + (cpu_percent / 100.0) * 3.0  # 4.5W base + up to 3W for CPU load
            
            return {
                'power_w': estimated_power,
                'cpu_usage': cpu_percent,
                'gpu_usage': 0.0
            }
            
    def _parse_tegrastats_line(self, line: str) -> Dict:
        """
        Parse a line from tegrastats output.
        Jetson Orin Nano format: "08-02-2025 22:11:48 RAM 3880/7620MB ... VDD_IN 5280mW/5280mW VDD_CPU_GPU_CV 1280mW/1280mW ..."
        """
        data = {'power_w': 0.0, 'cpu_usage': 0.0, 'gpu_usage': 0.0, 'memory_mb': 0.0}
        
        try:
            # Extract power consumption - VDD_IN is total system power for Jetson Orin Nano
            if 'VDD_IN' in line:
                # Look for power readings like "VDD_IN 5280mW/5280mW"
                vdd_section = line.split('VDD_IN')[1].split()[0]  # Get "5280mW/5280mW"
                power_reading = vdd_section.split('/')[0]  # Get first value "5280mW"
                if 'mW' in power_reading:
                    data['power_w'] = float(power_reading.replace('mW', '')) / 1000.0
                elif 'W' in power_reading:
                    data['power_w'] = float(power_reading.replace('W', ''))
                    
            # Extract GPU usage - look for GR3D_FREQ (GPU frequency usage)
            if 'GR3D_FREQ' in line:
                gpu_section = line.split('GR3D_FREQ')[1].split()[0]  # Get "45%"
                if '%' in gpu_section:
                    data['gpu_usage'] = float(gpu_section.replace('%', ''))
                    
            # Extract CPU usage (average of all cores)
            if 'CPU [' in line:
                cpu_section = line.split('CPU [')[1].split(']')[0]  # Get "31%@729,18%@729,24%@729,25%@729,30%@729,18%@729"
                cpu_values = []
                for part in cpu_section.split(','):
                    if '%@' in part:
                        cpu_percent = part.split('%@')[0]
                        cpu_values.append(float(cpu_percent))
                if cpu_values:
                    data['cpu_usage'] = float(np.mean(cpu_values))
                    
            # Extract memory usage - format "RAM 3880/7620MB"
            if 'RAM ' in line:
                ram_section = line.split('RAM ')[1].split('MB')[0]  # Get "3880/7620"
                if '/' in ram_section:
                    used_mem = float(ram_section.split('/')[0])
                    data['memory_mb'] = used_mem
                    
        except (ValueError, IndexError, AttributeError) as e:
            # Debug: print parsing errors for troubleshooting
            # print(f"Tegrastats parsing error: {e}, line: {line[:100]}...")
            pass
            
        return data
        
    def _calculate_metrics(self, total_time: float) -> Dict:
        """Calculate comprehensive power and efficiency metrics."""
        if not self.power_data:
            return {
                'avg_power_w': 0, 'peak_power_w': 0, 'min_power_w': 0,
                'total_energy_j': 0, 'avg_memory_mb': 0, 'peak_memory_mb': 0,
                'avg_cpu_usage': 0, 'peak_cpu_usage': 0,
                'avg_gpu_usage': 0, 'peak_gpu_usage': 0,
                'monitoring_duration_s': total_time
            }
            
        power_array = np.array(self.power_data)
        memory_array = np.array(self.memory_data)
        cpu_array = np.array(self.cpu_data)
        gpu_array = np.array(self.gpu_data)
        
        # Power metrics
        avg_power = np.mean(power_array)
        peak_power = np.max(power_array)
        min_power = np.min(power_array)
        
        # Energy calculation (Power × Time)
        total_energy = avg_power * total_time  # Joules (W × s)
        
        # Memory metrics (system RAM)
        avg_memory = np.mean(memory_array)
        peak_memory = np.max(memory_array)
        
        # CPU/GPU utilization
        avg_cpu = np.mean(cpu_array)
        peak_cpu = np.max(cpu_array)
        avg_gpu = np.mean(gpu_array)
        peak_gpu = np.max(gpu_array)
        
        return {
            # Power metrics
            'avg_power_w': float(avg_power),
            'peak_power_w': float(peak_power),
            'min_power_w': float(min_power),
            'power_std_w': float(np.std(power_array)),
            
            # Energy metrics
            'total_energy_j': float(total_energy),
            'energy_per_sample_j': float(total_energy / len(self.power_data)) if self.power_data else 0,
            
            # Memory metrics (system RAM)
            'avg_memory_mb': float(avg_memory),
            'peak_memory_mb': float(peak_memory),
            'memory_std_mb': float(np.std(memory_array)),
            
            # Utilization metrics
            'avg_cpu_usage': float(avg_cpu),
            'peak_cpu_usage': float(peak_cpu),
            'avg_gpu_usage': float(avg_gpu),
            'peak_gpu_usage': float(peak_gpu),
            
            # Monitoring metadata
            'monitoring_duration_s': total_time,
            'sample_count': len(self.power_data),
            'sample_interval_ms': self.sample_interval_ms
        }
        
    def calculate_energy_efficiency(self, throughput_fps: float, avg_power_w: float) -> Dict:
        """
        Calculate energy efficiency metrics for academic papers.
        
        Args:
            throughput_fps: Model inference throughput in FPS
            avg_power_w: Average power consumption in Watts
            
        Returns:
            Dict with energy efficiency metrics
        """
        if avg_power_w == 0:
            return {
                'fps_per_watt': 0,
                'energy_per_inference_j': 0,
                'efficiency_score': 0
            }
            
        fps_per_watt = throughput_fps / avg_power_w
        energy_per_inference = avg_power_w / throughput_fps  # Watts per FPS = Joules per inference
        
        # Efficiency score: normalized metric (higher is better)
        efficiency_score = fps_per_watt * 100  # Scale for readability
        
        return {
            'fps_per_watt': float(fps_per_watt),
            'energy_per_inference_j': float(energy_per_inference),
            'efficiency_score': float(efficiency_score),
            'power_efficiency_ratio': float(throughput_fps / avg_power_w) if avg_power_w > 0 else 0
        }

class SystemResourceMonitor:
    """Enhanced system resource monitoring with academic-grade metrics."""
    
    @staticmethod
    def get_gpu_memory_info() -> Dict:
        """Get detailed GPU memory information."""
        try:
            import torch
            if not torch.cuda.is_available():
                return {'gpu_memory_allocated_mb': 0, 'gpu_memory_reserved_mb': 0, 'gpu_memory_free_mb': 0}
                
            allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            reserved = torch.cuda.memory_reserved() / 1024 / 1024    # MB
            total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024  # MB
            free = total - reserved
            
            return {
                'gpu_memory_allocated_mb': float(allocated),
                'gpu_memory_reserved_mb': float(reserved),
                'gpu_memory_free_mb': float(free),
                'gpu_memory_total_mb': float(total),
                'gpu_memory_utilization': float(reserved / total * 100) if total > 0 else 0
            }
        except ImportError:
            return {'gpu_memory_allocated_mb': 0, 'gpu_memory_reserved_mb': 0, 'gpu_memory_free_mb': 0}
        
    @staticmethod
    def get_system_memory_info() -> Dict:
        """Get detailed system RAM information."""
        memory = psutil.virtual_memory()
        
        return {
            'system_memory_total_mb': float(memory.total / 1024 / 1024),
            'system_memory_used_mb': float(memory.used / 1024 / 1024),
            'system_memory_free_mb': float(memory.free / 1024 / 1024),
            'system_memory_available_mb': float(memory.available / 1024 / 1024),
            'system_memory_utilization': float(memory.percent)
        }
        
    @staticmethod
    def get_cpu_info() -> Dict:
        """Get CPU utilization and frequency information."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_freq = psutil.cpu_freq()
        
        return {
            'cpu_utilization': float(cpu_percent),
            'cpu_frequency_mhz': float(cpu_freq.current) if cpu_freq else 0,
            'cpu_core_count': psutil.cpu_count(),
            'cpu_core_count_physical': psutil.cpu_count(logical=False)
        }

# Test function for development
if __name__ == "__main__":
    print("🔋 Testing Jetson Power Monitor")
    
    monitor = JetsonPowerMonitor(sample_interval_ms=200)
    
    print("Starting 5-second test...")
    monitor.start_monitoring()
    
    # Simulate some work
    time.sleep(5)
    
    metrics = monitor.stop_monitoring()
    
    print("\n📊 Power Monitoring Results:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
            
    # Test energy efficiency calculation
    test_fps = 100.0
    test_power = 15.0
    efficiency = monitor.calculate_energy_efficiency(test_fps, test_power)
    
    print(f"\n⚡ Energy Efficiency (100 FPS @ 15W):")
    for key, value in efficiency.items():
        print(f"  {key}: {value:.3f}")
