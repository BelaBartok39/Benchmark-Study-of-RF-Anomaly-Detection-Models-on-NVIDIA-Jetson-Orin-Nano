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
from typing import Dict, List
from adaptive_power_manager import PowerMode


class ThresholdCalibrator:
    """
    Automatically calibrates optimal power mode switching thresholds
    based on actual hardware performance.
    """

    def __init__(self, benchmark, target_sla_ms: float = 10.0,
                 safety_margin: float = 0.9, verbose: bool = True):
        """
        Args:
            benchmark: AdaptiveBenchmark instance with loaded model
            target_sla_ms: Target latency SLA to maintain (e.g., 10ms)
            safety_margin: Safety factor (0.9 = use 90% of measured capacity)
            verbose: Print calibration progress
        """
        self.benchmark = benchmark
        self.target_sla_ms = target_sla_ms
        self.safety_margin = safety_margin
        self.verbose = verbose

    def profile_power_mode(self, power_mode: PowerMode,
                          num_samples: int = 100) -> Dict:
        """
        Profile inference latency at a specific power mode.

        Args:
            power_mode: Power mode to profile
            num_samples: Number of inference samples to collect

        Returns:
            Dict with latency statistics
        """
        if self.verbose:
            print(f"\n🔍 Profiling {power_mode.value}...")

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

        # Collect latency samples
        latencies = []
        for i in range(num_samples):
            latency = self.benchmark.run_inference(i)
            latencies.append(latency)

            if self.verbose and (i + 1) % 25 == 0:
                print(f"  {i+1}/{num_samples} samples, "
                      f"avg: {np.mean(latencies):.2f}ms")

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

        # Check if SLA is achievable
        if high_stats['p95_ms'] * self.safety_margin > self.target_sla_ms:
            raise ValueError(
                f"❌ Cannot meet {self.target_sla_ms}ms SLA!\n"
                f"   Even MAXN P95 latency is {high_stats['p95_ms']:.2f}ms\n"
                f"   Suggestion: Increase target SLA or use TensorRT optimization"
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
            'safety_margin': self.safety_margin,
            'strategy': strategy,
            'low_power_stats': low_stats,
            'medium_power_stats': medium_stats,
            'high_power_stats': high_stats,
            'can_meet_sla': True
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
            # Switch to 25W conservatively (e.g., at P90)
            medium_threshold = low_stats['p90_ms'] * self.safety_margin
            high_threshold = medium_stats['p90_ms'] * self.safety_margin if medium_can_meet_sla else self.target_sla_ms * 0.95
        elif medium_can_meet_sla:
            # Need 25W for SLA - switch from 15W early
            # Use P75 of 15W to switch to 25W proactively
            medium_threshold = low_stats['p75_ms'] * self.safety_margin
            high_threshold = medium_stats['p95_ms'] * self.safety_margin
        else:
            # Need MAXN for SLA - switch aggressively
            medium_threshold = low_stats['p75_ms'] * 0.8  # Very conservative
            high_threshold = self.target_sla_ms * 0.9

        # Ensure threshold ordering
        if medium_threshold >= high_threshold:
            medium_threshold = high_threshold * 0.75

        # Calculate hysteresis (time to stay in high power before downshifting)
        # More aggressive switching = shorter hysteresis
        if low_can_meet_sla:
            hysteresis = 10.0  # Long hysteresis, 15W is fine
        elif medium_can_meet_sla:
            hysteresis = 5.0   # Medium hysteresis
        else:
            hysteresis = 3.0   # Short hysteresis, need MAXN often

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
