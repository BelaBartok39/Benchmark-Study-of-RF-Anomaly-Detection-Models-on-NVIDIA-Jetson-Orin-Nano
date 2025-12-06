#!/usr/bin/env python3
"""
GPU Frequency Manager for NVIDIA Jetson Orin Nano.
Allows fine-grained control of GPU frequencies within nvpmodel limits.
"""

import os
import subprocess
from typing import List, Optional

class GPUFrequencyManager:
    """
    Manages GPU frequency scaling on Jetson Orin Nano via sysfs.
    Path: /sys/devices/17000000.ga10b/devfreq/17000000.ga10b/
    """
    
    # Common path for Orin series (GA10B GPU)
    SYSFS_PATH = "/sys/devices/17000000.ga10b/devfreq/17000000.ga10b"
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.available_freqs = self._get_available_frequencies()
        if self.verbose:
            print(f"GPU Frequencies available: {[f'{f/1e6:.0f}MHz' for f in self.available_freqs]}")

    def _get_available_frequencies(self) -> List[int]:
        """Read available frequencies from sysfs."""
        try:
            path = os.path.join(self.SYSFS_PATH, "available_frequencies")
            if not os.path.exists(path):
                # Fallback for non-Orin or different paths
                if self.verbose:
                    print(f"Warning: sysfs path {path} not found.")
                return []
            
            with open(path, 'r') as f:
                # Frequencies are in Hz
                freqs = [int(x) for x in f.read().strip().split()]
                return sorted(freqs)
        except Exception as e:
            if self.verbose:
                print(f"Error reading frequencies: {e}")
            return []

    def get_current_frequency(self) -> int:
        """Get current GPU frequency in Hz."""
        try:
            path = os.path.join(self.SYSFS_PATH, "cur_freq")
            if not os.path.exists(path):
                return 0
            with open(path, 'r') as f:
                return int(f.read().strip())
        except:
            return 0

    def set_frequency(self, freq_hz: int) -> bool:
        """
        Set GPU frequency.
        Note: This sets both min and max to the same value to 'pin' it.
        Requires sudo privileges.
        """
        # Snap to nearest available frequency
        if not self.available_freqs:
            return False
            
        target = min(self.available_freqs, key=lambda x: abs(x - freq_hz))
        
        try:
            # We need to use userspace governor to manually set freq, 
            # or set min/max limits on the current governor.
            # Setting min_freq and max_freq is the safest way on Jetson.
            
            cmd_min = f"echo {target} > {os.path.join(self.SYSFS_PATH, 'min_freq')}"
            cmd_max = f"echo {target} > {os.path.join(self.SYSFS_PATH, 'max_freq')}"
            
            # Execute with sudo
            subprocess.run(f"sudo bash -c '{cmd_min}'", shell=True, check=True)
            subprocess.run(f"sudo bash -c '{cmd_max}'", shell=True, check=True)
            
            if self.verbose:
                print(f"⚡ GPU Frequency set to {target/1e6:.0f} MHz")
            return True
            
        except subprocess.CalledProcessError as e:
            if self.verbose:
                print(f"❌ Failed to set frequency: {e}")
            return False

    def reset_to_auto(self):
        """Unpin frequency (allow OS to scale it within power mode limits)."""
        if not self.available_freqs:
            return
            
        min_f = self.available_freqs[0]
        max_f = self.available_freqs[-1]
        
        try:
            cmd_min = f"echo {min_f} > {os.path.join(self.SYSFS_PATH, 'min_freq')}"
            cmd_max = f"echo {max_f} > {os.path.join(self.SYSFS_PATH, 'max_freq')}"
            
            subprocess.run(f"sudo bash -c '{cmd_min}'", shell=True, check=True)
            subprocess.run(f"sudo bash -c '{cmd_max}'", shell=True, check=True)
            
            if self.verbose:
                print("🔄 GPU Frequency reset to Auto")
        except:
            pass

if __name__ == "__main__":
    # Simple test
    gpu = GPUFrequencyManager(verbose=True)
    current = gpu.get_current_frequency()
    print(f"Current Freq: {current/1e6:.0f} MHz")
