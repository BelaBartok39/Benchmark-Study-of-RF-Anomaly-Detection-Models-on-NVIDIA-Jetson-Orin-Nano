#!/usr/bin/env python3
"""
Automatic Threshold Calibration for Adaptive Power Management

Profiles model performance at each power mode to automatically determine
optimal switching thresholds that guarantee target SLA while maximizing
energy efficiency.
"""

import time
import numpy as np
import subprocess
import threading
from typing import Dict, List
from adaptive_power_manager import PowerMode


class ThresholdCalibrator:
    """
    Automatically calibrates optimal power mode switching thresholds
    based on actual hardware performance.
    """

    def __init__(self, benchmark, target_sla_ms: float = 10.0,
                 safety_margin: float = 0.9, verbose: bool = True,
                 auto_adjust_sla: bool = False, num_channels: int = 1):
        """
        Args:
            benchmark: AdaptiveBenchmark instance with loaded model
            target_sla_ms: Target latency SLA to maintain (e.g., 10ms)
            safety_margin: Safety factor (0.9 = use 90% of measured capacity)
            verbose: Print calibration progress
            auto_adjust_sla: Automatically adjust SLA if target is unreachable
            num_channels: Number of concurrent channels to simulate during profiling
        """
        self.benchmark = benchmark
        self.target_sla_ms = target_sla_ms
        self.safety_margin = safety_margin
        self.verbose = verbose
        self.auto_adjust_sla = auto_adjust_sla
        self.num_channels = num_channels
        self.original_target_sla_ms = target_sla_ms  # Keep original for reporting

    def profile_power_mode(self, power_mode: PowerMode,
                          num_samples: int = 100) -> Dict:
        """
        Profile inference latency at a specific power mode.

        Uses multi-threaded concurrent profiling if num_channels > 1 to simulate
        realistic GPU contention that will occur during actual benchmarking.

        Args:
            power_mode: Power mode to profile
            num_samples: Number of inference samples to collect (total across all channels)

        Returns:
            Dict with latency statistics
        """
        if self.verbose:
            print(f"\n🔍 Profiling {power_mode.value}...")
            if self.num_channels > 1:
                print(f"   Using {self.num_channels} concurrent channels to simulate realistic load")

        # Set power mode
        mode_num = {
            PowerMode.LOW_POWER: 0,      # 15W
            PowerMode.MEDIUM_POWER: 1,   # 25W
            PowerMode.HIGH_POWER: 2      # MAXN
        }[power_mode]

        try:
            subprocess.run(['sudo', 'nvpmodel', '-m', str(mode_num)],
                         check=True, capture_output=True, timeout=5.0)
            time.sleep(2.0)  # Stabilization time
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Warning: Could not set power mode: {e}")

        # Warmup
        for i in range(10):
            self.benchmark.run_inference(i)

        # Collect latency samples with concurrent threads if num_channels > 1
        if self.num_channels == 1:
            # Single-threaded sequential profiling
            latencies = []
            for i in range(num_samples):
                latency = self.benchmark.run_inference(i)
                latencies.append(latency)

                if self.verbose and (i + 1) % 25 == 0:
                    print(f"  {i+1}/{num_samples} samples, "
                          f"avg: {np.mean(latencies):.2f}ms")
        else:
            # Multi-threaded concurrent profiling (simulates realistic GPU contention)
            latencies = []
            latency_lock = threading.Lock()
            samples_per_channel = num_samples // self.num_channels
            completed_samples = [0]  # Mutable list for closure

            def channel_worker(channel_id: int):
                """Worker function for each channel thread."""
                sample_idx = channel_id * 10000  # Offset to avoid sample overlap

                for i in range(samples_per_channel):
                    latency = self.benchmark.run_inference(sample_idx)

                    # Thread-safe append
                    with latency_lock:
                        latencies.append(latency)
                        completed_samples[0] += 1

                        # Progress reporting
                        if self.verbose and completed_samples[0] % 25 == 0:
                            print(f"  {completed_samples[0]}/{num_samples} samples, "
                                  f"avg: {np.mean(latencies):.2f}ms")

                    sample_idx += 1

            # Launch concurrent channel threads
            threads = []
            for channel_id in range(self.num_channels):
                thread = threading.Thread(target=channel_worker, args=(channel_id,))
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

        latencies = np.array(latencies)

        stats = {
            'power_mode': power_mode.value,
            'num_samples': num_samples,
            'mean_ms': float(np.mean(latencies)),
            'median_ms': float(np.median(latencies)),
            'std_ms': float(np.std(latencies)),
            'p50_ms': float(np.percentile(latencies, 50)),
            'p75_ms': float(np.percentile(latencies, 75)),
            'p90_ms': float(np.percentile(latencies, 90)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99)),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies))
        }

        if self.verbose:
            print(f"  ✓ Mean: {stats['mean_ms']:.2f}ms, "
                  f"P95: {stats['p95_ms']:.2f}ms, "
                  f"P99: {stats['p99_ms']:.2f}ms")

        return stats

    def calibrate(self, strategy: str = 'sla_based') -> Dict:
        """
        Run full calibration and determine optimal thresholds.

        Args:
            strategy: Calibration strategy
                'sla_based' - Work backwards from target SLA (recommended)
                'performance_gap' - Use performance gaps between modes

        Returns:
            Dict with calibrated thresholds and profiling data
        """
        if self.verbose:
            print("\n" + "="*60)
            print("AUTOMATIC THRESHOLD CALIBRATION")
            print("="*60)
            print(f"Target SLA: {self.target_sla_ms}ms")
            print(f"Safety Margin: {self.safety_margin * 100:.0f}%")
            print(f"Strategy: {strategy}")
            print("")

        # Profile each power mode
        low_stats = self.profile_power_mode(PowerMode.LOW_POWER)
        medium_stats = self.profile_power_mode(PowerMode.MEDIUM_POWER)
        high_stats = self.profile_power_mode(PowerMode.HIGH_POWER)

        # Check if SLA is achievable and handle intelligently
        sla_achievable = high_stats['p95_ms'] * self.safety_margin <= self.target_sla_ms

        if not sla_achievable:
            # Calculate minimum achievable SLA based on MAXN performance
            # Use P99 with 10% margin for realistic minimum
            min_achievable_sla = high_stats['p99_ms'] * 1.1

            if self.verbose:
                print("\n" + "⚠️ "*30)
                print(f"⚠️  TARGET SLA UNREACHABLE")
                print("⚠️ "*30)
                print(f"\n📊 Hardware Analysis:")
                print(f"   Target SLA:        {self.target_sla_ms:.2f}ms")
                print(f"   MAXN P95 latency:  {high_stats['p95_ms']:.2f}ms")
                print(f"   MAXN P99 latency:  {high_stats['p99_ms']:.2f}ms")
                print(f"   With safety margin: {high_stats['p95_ms'] * self.safety_margin:.2f}ms")

                print(f"\n💡 Recommended Actions:")
                print(f"   1. ✅ Use TARGET_SLA={min_achievable_sla:.1f} (minimum achievable)")
                print(f"   2. 🔧 Reduce workload (fewer channels, smaller batch)")
                print(f"   3. ⚡ Use TensorRT optimization")
                print(f"   4. 🎯 Run best-effort mode (track SLA violations)")

            if self.auto_adjust_sla:
                # Auto-adjust to achievable SLA
                self.target_sla_ms = min_achievable_sla
                if self.verbose:
                    print(f"\n✓ AUTO-ADJUSTED: Using SLA={min_achievable_sla:.1f}ms")
                    print(f"  (Original target: {self.original_target_sla_ms:.1f}ms)")
                    print("")
            else:
                # Provide suggestion and abort
                if self.verbose:
                    print(f"\n❌ Cannot proceed with {self.target_sla_ms}ms SLA")
                    print(f"   Run with: AUTO_ADJUST_SLA=true TARGET_SLA={self.target_sla_ms}")
                    print(f"   Or use:   TARGET_SLA={min_achievable_sla:.1f}")
                    print("")

                raise ValueError(
                    f"SLA {self.target_sla_ms}ms unreachable. "
                    f"Minimum achievable: {min_achievable_sla:.1f}ms. "
                    f"Use --auto-adjust-sla flag to auto-adjust."
                )

        if strategy == 'sla_based':
            result = self._calibrate_sla_based(low_stats, medium_stats, high_stats)
        elif strategy == 'performance_gap':
            result = self._calibrate_performance_gap(low_stats, medium_stats, high_stats)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Add common metadata
        result.update({
            'calibration_timestamp': time.time(),
            'target_sla_ms': self.target_sla_ms,
            'original_target_sla_ms': self.original_target_sla_ms,
            'sla_was_adjusted': self.target_sla_ms != self.original_target_sla_ms,
            'safety_margin': self.safety_margin,
            'strategy': strategy,
            'low_power_stats': low_stats,
            'medium_power_stats': medium_stats,
            'high_power_stats': high_stats,
            'can_meet_sla': sla_achievable,
            'auto_adjust_sla': self.auto_adjust_sla
        })

        if self.verbose:
            self._print_calibration_results(result)

        return result

    def _calibrate_sla_based(self, low_stats, medium_stats, high_stats) -> Dict:
        """
        Calibrate thresholds based on target SLA.

        Strategy:
        - Determine which modes can meet SLA
        - Set thresholds to switch before violating SLA
        - Use P95 latencies with safety margin
        """
        # Determine mode capabilities
        low_can_meet_sla = low_stats['p95_ms'] * self.safety_margin <= self.target_sla_ms
        medium_can_meet_sla = medium_stats['p95_ms'] * self.safety_margin <= self.target_sla_ms

        # Calculate thresholds
        if low_can_meet_sla:
            # 15W can meet SLA - stay there most of the time
            # Threshold should be high enough to avoid false positives during normal operation
            # Use max of (P99 * 1.05) and (SLA * 0.8) to ensure we don't switch on normal noise
            medium_threshold = max(low_stats['p99_ms'] * 1.05, self.target_sla_ms * 0.8)
            
            # High threshold (25W -> MAXN)
            # If 25W is also good, use its P99; otherwise use SLA limit
            if medium_can_meet_sla:
                high_threshold = max(medium_stats['p99_ms'] * 1.05, self.target_sla_ms * 0.9)
            else:
                high_threshold = self.target_sla_ms * 0.95

        elif medium_can_meet_sla:
            # Need 25W for SLA - switch from 15W early
            # Use P75 of 15W to switch to 25W proactively
            medium_threshold = low_stats['p75_ms'] * self.safety_margin
            
            # High threshold can be higher since 25W is safe
            high_threshold = max(medium_stats['p99_ms'] * 1.05, self.target_sla_ms * 0.9)
            
        else:
            # Need MAXN for SLA - switch aggressively
            medium_threshold = low_stats['p75_ms'] * 0.8  # Very conservative
            high_threshold = self.target_sla_ms * 0.9

        # Ensure threshold ordering
        if medium_threshold >= high_threshold:
            medium_threshold = high_threshold * 0.75

        # Calculate hysteresis (time to stay in high power before downshifting)
        # Optimized for energy efficiency: recover quickly when load drops
        if low_can_meet_sla:
            hysteresis = 3.0  # Fast recovery to 15W since it's capable
        elif medium_can_meet_sla:
            hysteresis = 3.0   # Fast recovery
        else:
            hysteresis = 2.0   # Very fast recovery, try to save power where possible

        return {
            'medium_threshold_ms': medium_threshold,
            'high_threshold_ms': high_threshold,
            'hysteresis_time_s': hysteresis,
            'low_can_meet_sla': low_can_meet_sla,
            'medium_can_meet_sla': medium_can_meet_sla,
            'expected_primary_mode': '15W' if low_can_meet_sla else ('25W' if medium_can_meet_sla else 'MAXN')
        }

    def _calibrate_performance_gap(self, low_stats, medium_stats, high_stats) -> Dict:
        """
        Calibrate thresholds based on performance gaps between modes.

        Strategy:
        - Use P75 of lower mode as threshold to switch up
        - Ensures switching provides meaningful benefit
        """
        medium_threshold = low_stats['p75_ms']
        high_threshold = medium_stats['p75_ms']

        # Ensure thresholds don't exceed SLA
        medium_threshold = min(medium_threshold, self.target_sla_ms * 0.8)
        high_threshold = min(high_threshold, self.target_sla_ms * 0.95)

        # Adaptive hysteresis based on performance gaps
        gap_15w_to_25w = (low_stats['mean_ms'] - medium_stats['mean_ms']) / low_stats['mean_ms']
        gap_25w_to_maxn = (medium_stats['mean_ms'] - high_stats['mean_ms']) / medium_stats['mean_ms']

        # Smaller gaps = shorter hysteresis (not worth staying in high power)
        if gap_15w_to_25w < 0.2:  # < 20% improvement
            hysteresis = 3.0
        elif gap_15w_to_25w < 0.4:
            hysteresis = 5.0
        else:
            hysteresis = 8.0

        return {
            'medium_threshold_ms': medium_threshold,
            'high_threshold_ms': high_threshold,
            'hysteresis_time_s': hysteresis,
            'performance_gap_15w_to_25w_pct': gap_15w_to_25w * 100,
            'performance_gap_25w_to_maxn_pct': gap_25w_to_maxn * 100
        }

    def _print_calibration_results(self, result: Dict):
        """Print calibration results in a readable format."""
        print("\n" + "="*60)
        print("CALIBRATION RESULTS")
        print("="*60)

        # Show SLA adjustment if it occurred
        if result.get('sla_was_adjusted', False):
            print(f"\n🔄 SLA Adjustment:")
            print(f"   Original target: {result['original_target_sla_ms']:.1f}ms")
            print(f"   Adjusted to:     {result['target_sla_ms']:.1f}ms ✓")
            print(f"   Reason: Hardware cannot guarantee {result['original_target_sla_ms']:.1f}ms")

        # Show calibration workload
        print(f"\n🔧 Calibration Configuration:")
        print(f"   Concurrent channels: {self.num_channels}")
        if self.num_channels > 1:
            print(f"   Note: Using multi-threaded profiling to simulate realistic GPU contention")

        print(f"\n📊 Profiled Performance:")
        low = result['low_power_stats']
        medium = result['medium_power_stats']
        high = result['high_power_stats']

        print(f"   15W:  Mean={low['mean_ms']:.2f}ms, "
              f"P95={low['p95_ms']:.2f}ms, P99={low['p99_ms']:.2f}ms")
        print(f"   25W:  Mean={medium['mean_ms']:.2f}ms, "
              f"P95={medium['p95_ms']:.2f}ms, P99={medium['p99_ms']:.2f}ms")
        print(f"   MAXN: Mean={high['mean_ms']:.2f}ms, "
              f"P95={high['p95_ms']:.2f}ms, P99={high['p99_ms']:.2f}ms")

        print(f"\n🎯 Calibrated Thresholds:")
        print(f"   15W → 25W:  {result['medium_threshold_ms']:.2f}ms")
        print(f"   25W → MAXN: {result['high_threshold_ms']:.2f}ms")
        print(f"   Hysteresis: {result['hysteresis_time_s']:.1f}s")

        print(f"\n💡 Expected Behavior:")
        if 'low_can_meet_sla' in result:
            if result['low_can_meet_sla']:
                print(f"   ✓ 15W mode can meet {self.target_sla_ms}ms SLA")
                print(f"   → Will use 15W primarily, 25W for bursts")
            elif result.get('medium_can_meet_sla', False):
                print(f"   ⚠ 15W cannot meet {self.target_sla_ms}ms SLA")
                print(f"   ✓ 25W mode can meet SLA")
                print(f"   → Will use 15W/25W primarily, MAXN for bursts")
            else:
                print(f"   ⚠ Need MAXN to reliably meet {self.target_sla_ms}ms SLA")
                print(f"   → Will use 25W/MAXN primarily")

        if 'performance_gap_15w_to_25w_pct' in result:
            print(f"\n📈 Performance Improvements:")
            print(f"   15W → 25W:  {result['performance_gap_15w_to_25w_pct']:.1f}% faster")
            if 'performance_gap_25w_to_maxn_pct' in result:
                print(f"   25W → MAXN: {result['performance_gap_25w_to_maxn_pct']:.1f}% faster")

        print("="*60 + "\n")
