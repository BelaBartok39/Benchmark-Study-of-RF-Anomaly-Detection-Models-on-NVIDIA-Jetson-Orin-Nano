#!/usr/bin/env python3
"""
Adaptive Power Management for NVIDIA Jetson Orin Nano
Dynamically switches between power modes based on real-time latency feedback
to balance performance and energy efficiency.
"""

import time
import subprocess
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from enum import Enum


# Model-specific latency thresholds (ms) for three-tier power management
# PRIMARY threshold: 15W → 25W transition
# SECONDARY threshold: 25W → MAXN transition
MODEL_THRESHOLDS_MEDIUM = {
    'ae': 8.0,          # Fast model - rarely needs 25W
    'ff': 8.0,          # Fast feedforward model
    'aae': 7.0,         # Medium-fast
    'cnn_ae': 10.0,     # Medium complexity
    'resnet_ae': 12.0,  # Medium-slow
    'lstm_ae': 20.0     # Slow model - adjusted to 20ms to match 14ms baseline
}

MODEL_THRESHOLDS_HIGH = {
    'ae': 20.0,         # Fast model - very loose for MAXN
    'ff': 20.0,         # Fast feedforward
    'aae': 15.0,        # Medium-fast
    'cnn_ae': 25.0,     # Medium complexity
    'resnet_ae': 30.0,  # Medium-slow
    'lstm_ae': 40.0     # Slow model - adjusted high threshold
}

# Legacy single-threshold support (maps to HIGH threshold)
MODEL_THRESHOLDS = MODEL_THRESHOLDS_HIGH

# Model-specific hysteresis times (seconds)
# Longer hysteresis for models with higher latency variance
MODEL_HYSTERESIS = {
    'ae': 3.0,          # Stable, quick downshift
    'ff': 3.0,          # Stable
    'aae': 4.0,         # Moderate variance
    'cnn_ae': 5.0,      # Moderate variance
    'resnet_ae': 6.0,   # Higher variance
    'lstm_ae': 3.0      # High variance, adjusted for efficiency
}


class PowerMode(Enum):
    """Jetson Orin Nano power modes (JetPack 6.1)."""
    LOW_POWER = "15W"       # nvpmodel mode 0 (15W)
    MEDIUM_POWER = "25W"    # nvpmodel mode 1 (25W)
    HIGH_POWER = "MAXN"     # nvpmodel mode 2 (MAXN SUPER)


class AdaptivePowerManager:
    """
    Adaptive power management system for ML inference on Jetson platforms.

    Implements a latency-driven power mode switching algorithm with hysteresis
    to prevent oscillation and optimize energy efficiency while maintaining
    performance targets.

    Algorithm:
    1. Start in LOW_POWER mode (15W) by default
    2. Monitor inference latency continuously
    3. If latency > threshold: switch to HIGH_POWER mode (MAXN SUPER)
    4. If latency < threshold for hysteresis_time seconds: switch to LOW_POWER mode (15W)
    5. Track switching overhead and energy consumption
    """

    def __init__(self,
                 latency_threshold_ms: float = 10.0,
                 latency_threshold_medium_ms: Optional[float] = None,
                 latency_threshold_high_ms: Optional[float] = None,
                 hysteresis_time_s: float = 5.0,
                 initial_mode: PowerMode = PowerMode.LOW_POWER,
                 enable_switching: bool = True,
                 enable_three_tier: bool = True,
                 verbose: bool = True,
                 model_name: Optional[str] = None,
                 use_model_defaults: bool = False):
        """
        Initialize adaptive power manager.

        Args:
            latency_threshold_ms: Legacy single threshold (for backward compatibility)
            latency_threshold_medium_ms: Threshold for 15W → 25W transition (three-tier mode)
            latency_threshold_high_ms: Threshold for 25W → MAXN transition (three-tier mode)
            hysteresis_time_s: Time in seconds latency must remain below threshold before downshifting
            initial_mode: Starting power mode (default: LOW_POWER)
            enable_switching: Enable automatic power mode switching (default: True)
            enable_three_tier: Use three-tier mode (15W/25W/MAXN) vs two-tier (15W/MAXN) (default: True)
            verbose: Print detailed status messages (default: True)
            model_name: Model name for auto-configuration (e.g., 'ae', 'lstm_ae')
            use_model_defaults: If True, use model-specific thresholds and hysteresis (default: False)
        """
        self.enable_three_tier = enable_three_tier

        # Auto-configure based on model if requested
        if use_model_defaults and model_name is not None:
            if enable_three_tier:
                self.latency_threshold_medium_ms = MODEL_THRESHOLDS_MEDIUM.get(model_name, latency_threshold_ms)
                self.latency_threshold_high_ms = MODEL_THRESHOLDS_HIGH.get(model_name, latency_threshold_ms * 2)
            else:
                # Two-tier mode: use single threshold for 15W → MAXN
                self.latency_threshold_medium_ms = None
                self.latency_threshold_high_ms = MODEL_THRESHOLDS_HIGH.get(model_name, latency_threshold_ms)
            self.hysteresis_time_s = MODEL_HYSTERESIS.get(model_name, hysteresis_time_s)
            if verbose:
                print(f"🎯 Using model-specific configuration for '{model_name}':")
                if enable_three_tier:
                    print(f"   Three-tier mode: 15W → 25W → MAXN")
                    print(f"   Medium threshold (15W→25W): {self.latency_threshold_medium_ms}ms")
                    print(f"   High threshold (25W→MAXN): {self.latency_threshold_high_ms}ms")
                else:
                    print(f"   Two-tier mode: 15W → MAXN")
                    print(f"   Threshold: {self.latency_threshold_high_ms}ms")
                print(f"   Hysteresis: {self.hysteresis_time_s}s")
        else:
            # Manual configuration
            if enable_three_tier:
                if latency_threshold_medium_ms is not None and latency_threshold_high_ms is not None:
                    # Explicit three-tier thresholds provided
                    self.latency_threshold_medium_ms = latency_threshold_medium_ms
                    self.latency_threshold_high_ms = latency_threshold_high_ms
                else:
                    # Three-tier requested but only single threshold provided
                    # Derive medium/high thresholds from the single threshold
                    # Strategy: medium = threshold, high = threshold * 2.5
                    self.latency_threshold_medium_ms = latency_threshold_ms
                    self.latency_threshold_high_ms = latency_threshold_ms * 2.5
                    if verbose:
                        print(f"ℹ️  Three-tier mode: Derived thresholds from {latency_threshold_ms}ms")
                        print(f"   Medium (15W→25W): {self.latency_threshold_medium_ms}ms")
                        print(f"   High (25W→MAXN): {self.latency_threshold_high_ms}ms")
            else:
                # Two-tier mode explicitly requested
                self.latency_threshold_medium_ms = None
                self.latency_threshold_high_ms = latency_threshold_ms
                self.enable_three_tier = False
            self.hysteresis_time_s = hysteresis_time_s

        self.model_name = model_name
        self.current_mode = None
        self.enable_switching = enable_switching
        self.verbose = verbose

        # Latency tracking
        self.latency_history = deque(maxlen=1000)  # Keep last 1000 latency measurements
        self.below_threshold_start_time = None

        # Mode switching tracking
        self.mode_switches = []  # List of (timestamp, from_mode, to_mode, reason)
        self.mode_switch_times = []  # Switching overhead measurements
        self.time_in_modes = {
            PowerMode.LOW_POWER: 0.0,
            PowerMode.MEDIUM_POWER: 0.0,
            PowerMode.HIGH_POWER: 0.0
        }
        self.last_mode_change_time = time.time()

        # Statistics
        self.total_inferences = 0
        self.violations = 0  # Number of times latency exceeded threshold

        # Thread safety
        self.lock = threading.Lock()

        # Initialize to desired mode
        if enable_switching:
            self._set_power_mode(initial_mode, reason="initialization")
        else:
            self.current_mode = initial_mode

    def record_inference(self, latency_ms: float) -> Optional[str]:
        """
        Record an inference latency measurement and potentially trigger mode switch.

        Args:
            latency_ms: Inference latency in milliseconds

        Returns:
            Status message if mode was changed, None otherwise
        """
        with self.lock:
            self.latency_history.append(latency_ms)
            self.total_inferences += 1

            if not self.enable_switching:
                return None

            if self.enable_three_tier:
                return self._record_inference_three_tier(latency_ms)
            else:
                return self._record_inference_two_tier(latency_ms)

    def _record_inference_three_tier(self, latency_ms: float) -> Optional[str]:
        """Three-tier power management: 15W → 25W → MAXN."""
        # Priority 1: If latency exceeds high threshold, immediately switch to MAXN
        if latency_ms > self.latency_threshold_high_ms:
            self.violations += 1
            if self.current_mode != PowerMode.HIGH_POWER:
                msg = self._set_power_mode(
                    PowerMode.HIGH_POWER,
                    reason=f"latency {latency_ms:.2f}ms > high threshold {self.latency_threshold_high_ms}ms"
                )
                self.below_threshold_start_time = None
                return msg

        # Priority 2: If latency exceeds medium threshold, switch to 25W (if currently at 15W)
        elif latency_ms > self.latency_threshold_medium_ms:
            if self.current_mode == PowerMode.LOW_POWER:
                msg = self._set_power_mode(
                    PowerMode.MEDIUM_POWER,
                    reason=f"latency {latency_ms:.2f}ms > medium threshold {self.latency_threshold_medium_ms}ms"
                )
                self.below_threshold_start_time = None
                return msg
            # If already at MAXN, stay there (don't downshift yet)
            self.below_threshold_start_time = None

        # Latency is below medium threshold - consider downshifting
        else:
            # Start or continue tracking time below threshold
            if self.below_threshold_start_time is None:
                self.below_threshold_start_time = time.time()

            time_below = time.time() - self.below_threshold_start_time

            # Gradual downshift after hysteresis period
            if time_below >= self.hysteresis_time_s:
                # Step down one level at a time
                if self.current_mode == PowerMode.HIGH_POWER:
                    # MAXN → 25W
                    msg = self._set_power_mode(
                        PowerMode.MEDIUM_POWER,
                        reason=f"latency below high threshold for {time_below:.1f}s"
                    )
                    self.below_threshold_start_time = time.time()  # Restart timer for next downshift
                    return msg
                elif self.current_mode == PowerMode.MEDIUM_POWER and latency_ms < self.latency_threshold_medium_ms:
                    # 25W → 15W
                    msg = self._set_power_mode(
                        PowerMode.LOW_POWER,
                        reason=f"latency below medium threshold for {time_below:.1f}s"
                    )
                    self.below_threshold_start_time = None
                    return msg

        return None

    def _record_inference_two_tier(self, latency_ms: float) -> Optional[str]:
        """Legacy two-tier power management: 15W → MAXN."""
        # Check if latency exceeds threshold
        if latency_ms > self.latency_threshold_high_ms:
            self.violations += 1

            # If in low power mode and threshold violated, switch to high power immediately
            if self.current_mode == PowerMode.LOW_POWER:
                msg = self._set_power_mode(
                    PowerMode.HIGH_POWER,
                    reason=f"latency {latency_ms:.2f}ms > threshold {self.latency_threshold_high_ms}ms"
                )
                self.below_threshold_start_time = None
                return msg

        else:
            # Latency is below threshold
            if self.current_mode == PowerMode.HIGH_POWER:
                # Start or continue tracking time below threshold
                if self.below_threshold_start_time is None:
                    self.below_threshold_start_time = time.time()

                # Check if we've been below threshold long enough (hysteresis)
                time_below = time.time() - self.below_threshold_start_time
                if time_below >= self.hysteresis_time_s:
                    msg = self._set_power_mode(
                        PowerMode.LOW_POWER,
                        reason=f"latency below threshold for {time_below:.1f}s"
                    )
                    self.below_threshold_start_time = None
                    return msg

        return None

    def _set_power_mode(self, mode: PowerMode, reason: str = "") -> str:
        """
        Set the Jetson power mode.

        Args:
            mode: Target power mode
            reason: Reason for mode change (for logging)

        Returns:
            Status message
        """
        if self.current_mode is not None and mode == self.current_mode:
            return f"Already in {mode.value} mode"

        old_mode = self.current_mode
        start_time = time.time()

        try:
            # Map PowerMode to nvpmodel mode number (JetPack 6.1)
            # Mode 0 = 15W, Mode 1 = 25W, Mode 2 = MAXN SUPER
            if mode == PowerMode.LOW_POWER:
                mode_number = 0
            elif mode == PowerMode.MEDIUM_POWER:
                mode_number = 1
            else:  # HIGH_POWER
                mode_number = 2

            # Set power mode using nvpmodel
            # Note: This requires sudo privileges. For testing without Jetson, we'll simulate.
            try:
                subprocess.run(
                    ['sudo', 'nvpmodel', '-m', str(mode_number)],
                    check=True,
                    capture_output=True,
                    timeout=5.0
                )
                actual_switch = True
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                # Not on Jetson or no sudo access - simulate mode switch
                if self.verbose:
                    print(f"⚠️  Warning: Could not execute nvpmodel (simulation mode)")
                actual_switch = False
                time.sleep(0.5)  # Simulate switching delay

            # Measure switching time
            switch_time = time.time() - start_time
            self.mode_switch_times.append(switch_time)

            # Update time spent in previous mode
            if old_mode is not None:
                time_in_previous = time.time() - self.last_mode_change_time
                self.time_in_modes[old_mode] += time_in_previous
            self.last_mode_change_time = time.time()

            # Record switch
            self.mode_switches.append({
                'timestamp': time.time(),
                'from_mode': old_mode.value if old_mode else "UNKNOWN",
                'to_mode': mode.value,
                'reason': reason,
                'switch_time_s': switch_time,
                'actual_switch': actual_switch
            })

            self.current_mode = mode

            from_mode_str = old_mode.value if old_mode else "UNKNOWN"
            msg = f"🔄 Power mode: {from_mode_str} → {mode.value} ({reason}) [switch time: {switch_time*1000:.0f}ms]" 
            if self.verbose:
                print(msg)

            return msg

        except Exception as e:
            error_msg = f"❌ Failed to set power mode: {e}"
            if self.verbose:
                print(error_msg)
            return error_msg

    def get_current_mode(self) -> PowerMode:
        """Get current power mode."""
        return self.current_mode

    def force_mode(self, mode: PowerMode, reason: str = "manual override") -> str:
        """
        Manually force a power mode change.

        Args:
            mode: Target power mode
            reason: Reason for change

        Returns:
            Status message
        """
        with self.lock:
            return self._set_power_mode(mode, reason=reason)

    def get_statistics(self) -> Dict:
        """
        Get comprehensive statistics about power management.

        Returns:
            Dictionary with power management statistics
        """
        with self.lock:
            # Update time in current mode
            time_in_current = time.time() - self.last_mode_change_time
            total_time = sum(self.time_in_modes.values()) + time_in_current

            latency_array = np.array(list(self.latency_history)) if self.latency_history else np.array([0])

            return {
                # Current state
                'current_mode': self.current_mode.value,
                'latency_threshold_ms': self.latency_threshold_high_ms if hasattr(self, 'latency_threshold_high_ms') else self.latency_threshold_ms,
                'latency_threshold_medium_ms': self.latency_threshold_medium_ms if hasattr(self, 'latency_threshold_medium_ms') else None,
                'latency_threshold_high_ms': self.latency_threshold_high_ms if hasattr(self, 'latency_threshold_high_ms') else None,
                'hysteresis_time_s': self.hysteresis_time_s,
                'three_tier_enabled': self.enable_three_tier,

                # Latency statistics
                'total_inferences': self.total_inferences,
                'avg_latency_ms': float(np.mean(latency_array)),
                'median_latency_ms': float(np.median(latency_array)),
                'p95_latency_ms': float(np.percentile(latency_array, 95)) if len(latency_array) > 0 else 0,
                'p99_latency_ms': float(np.percentile(latency_array, 99)) if len(latency_array) > 0 else 0,
                'max_latency_ms': float(np.max(latency_array)),
                'min_latency_ms': float(np.min(latency_array)),

                # Violations
                'threshold_violations': self.violations,
                'violation_rate': float(self.violations / self.total_inferences) if self.total_inferences > 0 else 0,

                # Mode switching
                'total_mode_switches': len(self.mode_switches),
                'avg_switch_time_ms': float(np.mean(self.mode_switch_times) * 1000) if self.mode_switch_times else 0,
                'max_switch_time_ms': float(np.max(self.mode_switch_times) * 1000) if self.mode_switch_times else 0,

                # Time in modes
                'time_in_low_power_s': self.time_in_modes[PowerMode.LOW_POWER] +
                                        (time_in_current if self.current_mode == PowerMode.LOW_POWER else 0),
                'time_in_medium_power_s': self.time_in_modes[PowerMode.MEDIUM_POWER] +
                                          (time_in_current if self.current_mode == PowerMode.MEDIUM_POWER else 0),
                'time_in_high_power_s': self.time_in_modes[PowerMode.HIGH_POWER] +
                                         (time_in_current if self.current_mode == PowerMode.HIGH_POWER else 0),
                'total_runtime_s': total_time,
                'low_power_percentage': float((self.time_in_modes[PowerMode.LOW_POWER] +
                                               (time_in_current if self.current_mode == PowerMode.LOW_POWER else 0)) /
                                              total_time * 100) if total_time > 0 else 0,
                'medium_power_percentage': float((self.time_in_modes[PowerMode.MEDIUM_POWER] +
                                                  (time_in_current if self.current_mode == PowerMode.MEDIUM_POWER else 0)) /
                                                 total_time * 100) if total_time > 0 else 0,
                'high_power_percentage': float((self.time_in_modes[PowerMode.HIGH_POWER] +
                                                (time_in_current if self.current_mode == PowerMode.HIGH_POWER else 0)) /
                                               total_time * 100) if total_time > 0 else 0,

                # Mode switch history
                'mode_switches': self.mode_switches.copy()
            }

    def reset_statistics(self):
        """Reset all statistics and history."""
        with self.lock:
            self.latency_history.clear()
            self.mode_switches.clear()
            self.mode_switch_times.clear()
            self.time_in_modes = {PowerMode.LOW_POWER: 0.0, PowerMode.HIGH_POWER: 0.0}
            self.last_mode_change_time = time.time()
            self.total_inferences = 0
            self.violations = 0
            self.below_threshold_start_time = None

    def print_summary(self):
        """Print a summary of power management statistics."""
        stats = self.get_statistics()

        print("\n" + "="*60)
        print("ADAPTIVE POWER MANAGEMENT SUMMARY")
        print("="*60)
        print(f"Current Mode: {stats['current_mode']}")
        print(f"Latency Threshold: {stats['latency_threshold_ms']:.2f} ms")
        print(f"Hysteresis Time: {stats['hysteresis_time_s']:.1f} s")
        print()
        print(f"Total Inferences: {stats['total_inferences']}")
        print(f"Avg Latency: {stats['avg_latency_ms']:.2f} ms")
        print(f"P95 Latency: {stats['p95_latency_ms']:.2f} ms")
        print(f"P99 Latency: {stats['p99_latency_ms']:.2f} ms")
        print()
        print(f"Threshold Violations: {stats['threshold_violations']} ({stats['violation_rate']*100:.2f}%)")
        print(f"Mode Switches: {stats['total_mode_switches']}")
        print(f"Avg Switch Time: {stats['avg_switch_time_ms']:.1f} ms")
        print()
        print(f"Time in Low Power (15W): {stats['time_in_low_power_s']:.1f}s ({stats['low_power_percentage']:.1f}%)")
        print(f"Time in High Power (MAXN): {stats['time_in_high_power_s']:.1f}s ({100-stats['low_power_percentage']:.1f}%)")
        print(f"Total Runtime: {stats['total_runtime_s']:.1f}s")
        print("="*60 + "\n")


# Utility functions for power mode management
def get_current_power_mode() -> Optional[int]:
    """
    Get current nvpmodel power mode (JetPack 6.1).

    Returns:
        Power mode number (0=15W, 1=25W, 2=MAXN) or None if unable to determine
    """
    try:
        result = subprocess.run(
            ['nvpmodel', '-q'],
            capture_output=True,
            text=True,
            timeout=2.0
        )

        # Parse output to find current mode
        for line in result.stdout.split('\n'):
            if 'NV Power Mode' in line or 'mode' in line.lower():
                # Example: "NV Power Mode: MAXN" or similar
                if 'MAXN' in line.upper():
                    return 2
                elif '25W' in line or '25 W' in line:
                    return 1
                elif '15W' in line or '15 W' in line:
                    return 0
                # Fallback to mode number in line
                elif 'Mode: 2' in line or 'mode 2' in line.lower():
                    return 2
                elif 'Mode: 1' in line or 'mode 1' in line.lower():
                    return 1
                elif 'Mode: 0' in line or 'mode 0' in line.lower():
                    return 0

        return None

    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def set_power_mode(mode: int) -> bool:
    """
    Set Jetson power mode (JetPack 6.1).

    Args:
        mode: Power mode number (0=15W, 1=25W, 2=MAXN)

    Returns:
        True if successful, False otherwise
    """
    try:
        subprocess.run(
            ['sudo', 'nvpmodel', '-m', str(mode)],
            check=True,
            capture_output=True,
            timeout=5.0
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


# Test function for development
if __name__ == "__main__":
    print("🔋 Testing Adaptive Power Manager\n")

    # Create adaptive power manager with low threshold to trigger switches
    apm = AdaptivePowerManager(
        latency_threshold_ms=5.0,
        hysteresis_time_s=2.0,
        initial_mode=PowerMode.LOW_POWER,
        enable_switching=True,
        verbose=True
    )

    print("Simulating inference workload with varying latency...\n")

    # Simulate inference workload
    np.random.seed(42)

    # Phase 1: Low latency (should stay in low power mode)
    print("Phase 1: Low latency workload")
    for i in range(20):
        latency = np.random.uniform(2.0, 4.0)  # Below threshold
        apm.record_inference(latency)
        time.sleep(0.1)

    # Phase 2: High latency spike (should switch to high power)
    print("\nPhase 2: High latency spike")
    for i in range(10):
        latency = np.random.uniform(8.0, 12.0)  # Above threshold
        apm.record_inference(latency)
        time.sleep(0.1)

    # Phase 3: Return to low latency (should switch back after hysteresis time)
    print("\nPhase 3: Return to low latency")
    for i in range(30):
        latency = np.random.uniform(2.0, 4.0)  # Below threshold
        apm.record_inference(latency)
        time.sleep(0.1)

    # Print summary
    apm.print_summary()

    # Get detailed statistics
    stats = apm.get_statistics()
    print("Mode switch history:")
    for switch in stats['mode_switches']:
        print(f"  {switch['from_mode']} → {switch['to_mode']}: {switch['reason']}")
