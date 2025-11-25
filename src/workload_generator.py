#!/usr/bin/env python3
"""
Workload Generator for Adaptive Power Management Experiments

Generates realistic RF anomaly detection workload patterns to evaluate
adaptive power management effectiveness across different scenarios.
"""

import numpy as np
import time
from typing import Iterator, Tuple, Dict, Optional
from enum import Enum


class WorkloadPattern(Enum):
    """Types of workload patterns for RF anomaly detection."""
    BURSTY = "bursty"              # Periods of high activity followed by idle/low activity
    CONTINUOUS = "continuous"       # Sustained high-throughput processing
    VARIABLE = "variable"          # Variable complexity with changing inference rates
    PERIODIC = "periodic"          # Periodic bursts of activity
    RANDOM = "random"              # Random arrival pattern (Poisson process)


class WorkloadGenerator:
    """
    Generate realistic workload patterns for RF anomaly detection scenarios.

    Simulates different spectrum monitoring scenarios:
    - Bursty: Spectrum scanning with intermittent signal detection
    - Continuous: Real-time monitoring of busy frequency bands
    - Variable: Multi-model pipeline with preliminary detection
    - Periodic: Scheduled scanning operations
    - Random: Random signal arrival pattern
    """

    def __init__(self,
                 pattern: WorkloadPattern,
                 duration_s: float = 60.0,
                 base_rate_fps: float = 100.0,
                 seed: Optional[int] = None):
        """
        Initialize workload generator.

        Args:
            pattern: Workload pattern type
            duration_s: Total duration of workload in seconds
            base_rate_fps: Base inference rate in frames per second
            seed: Random seed for reproducibility
        """
        self.pattern = pattern
        self.duration_s = duration_s
        self.base_rate_fps = base_rate_fps
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        # Pattern-specific parameters
        self.burst_duration_s = 5.0    # Duration of each burst
        self.idle_duration_s = 10.0     # Duration of idle period between bursts
        self.periodic_interval_s = 15.0  # Interval for periodic pattern
        self.variable_complexity_levels = [0.5, 1.0, 2.0]  # Relative complexity multipliers

    def generate_schedule(self) -> Dict:
        """
        Generate a complete workload schedule.

        Returns:
            Dictionary containing:
            - timestamps: Array of timestamps when inferences should occur
            - rates: Array of instantaneous rates (FPS) at each timestamp
            - pattern: Workload pattern name
            - statistics: Summary statistics
        """
        if self.pattern == WorkloadPattern.BURSTY:
            return self._generate_bursty()
        elif self.pattern == WorkloadPattern.CONTINUOUS:
            return self._generate_continuous()
        elif self.pattern == WorkloadPattern.VARIABLE:
            return self._generate_variable()
        elif self.pattern == WorkloadPattern.PERIODIC:
            return self._generate_periodic()
        elif self.pattern == WorkloadPattern.RANDOM:
            return self._generate_random()
        else:
            raise ValueError(f"Unknown workload pattern: {self.pattern}")

    def _generate_bursty(self) -> Dict:
        """
        Generate bursty workload pattern.

        Simulates spectrum monitoring where signals appear in bursts.
        Pattern: High activity for burst_duration_s, then idle for idle_duration_s.
        """
        timestamps = []
        rates = []
        current_time = 0.0

        cycle_duration = self.burst_duration_s + self.idle_duration_s
        num_cycles = int(self.duration_s / cycle_duration) + 1

        for cycle in range(num_cycles):
            # Burst period - high inference rate
            burst_start = cycle * cycle_duration
            burst_end = min(burst_start + self.burst_duration_s, self.duration_s)

            if burst_start < self.duration_s:
                num_inferences = int(self.base_rate_fps * self.burst_duration_s)
                burst_times = np.linspace(burst_start, burst_end, num_inferences, endpoint=False)
                timestamps.extend(burst_times)
                rates.extend([self.base_rate_fps] * num_inferences)

            # Idle period - very low or no inference
            idle_start = burst_end
            idle_end = min(idle_start + self.idle_duration_s, self.duration_s)

            if idle_start < self.duration_s and idle_end > idle_start:
                # Occasional low-rate inference during idle
                num_idle_inferences = max(1, int(self.base_rate_fps * 0.1 * self.idle_duration_s))
                idle_times = np.linspace(idle_start, idle_end, num_idle_inferences, endpoint=False)
                timestamps.extend(idle_times)
                rates.extend([self.base_rate_fps * 0.1] * num_idle_inferences)

        timestamps = np.array(timestamps)
        rates = np.array(rates)

        # Sort by timestamp
        sort_idx = np.argsort(timestamps)
        timestamps = timestamps[sort_idx]
        rates = rates[sort_idx]

        return {
            'timestamps': timestamps,
            'rates': rates,
            'pattern': self.pattern.value,
            'statistics': {
                'total_inferences': len(timestamps),
                'avg_rate_fps': np.mean(rates),
                'peak_rate_fps': np.max(rates),
                'min_rate_fps': np.min(rates),
                'duration_s': self.duration_s
            }
        }

    def _generate_continuous(self) -> Dict:
        """
        Generate continuous high-throughput workload.

        Simulates real-time monitoring of busy frequency bands with sustained inference.
        """
        num_inferences = int(self.base_rate_fps * self.duration_s)
        timestamps = np.linspace(0, self.duration_s, num_inferences, endpoint=False)
        rates = np.ones(num_inferences) * self.base_rate_fps

        # Add small random jitter to simulate realistic timing variations
        jitter = np.random.normal(0, 0.001, num_inferences)  # ±1ms jitter
        timestamps = timestamps + jitter
        timestamps = np.clip(timestamps, 0, self.duration_s)
        timestamps = np.sort(timestamps)

        return {
            'timestamps': timestamps,
            'rates': rates,
            'pattern': self.pattern.value,
            'statistics': {
                'total_inferences': len(timestamps),
                'avg_rate_fps': np.mean(rates),
                'peak_rate_fps': np.max(rates),
                'min_rate_fps': np.min(rates),
                'duration_s': self.duration_s
            }
        }

    def _generate_variable(self) -> Dict:
        """
        Generate variable complexity workload.

        Simulates multi-model pipeline where different models are triggered
        based on preliminary detection results.
        """
        timestamps = []
        rates = []
        current_time = 0.0

        # Divide duration into segments with different complexity levels
        segment_duration = 10.0  # 10 second segments
        num_segments = int(self.duration_s / segment_duration) + 1

        for seg in range(num_segments):
            seg_start = seg * segment_duration
            seg_end = min(seg_start + segment_duration, self.duration_s)

            if seg_start >= self.duration_s:
                break

            # Randomly select complexity level
            complexity = np.random.choice(self.variable_complexity_levels)
            seg_rate = self.base_rate_fps * complexity

            # Generate inferences for this segment
            num_inferences = int(seg_rate * (seg_end - seg_start))
            if num_inferences > 0:
                seg_times = np.linspace(seg_start, seg_end, num_inferences, endpoint=False)
                timestamps.extend(seg_times)
                rates.extend([seg_rate] * num_inferences)

        timestamps = np.array(timestamps)
        rates = np.array(rates)

        # Sort by timestamp
        sort_idx = np.argsort(timestamps)
        timestamps = timestamps[sort_idx]
        rates = rates[sort_idx]

        return {
            'timestamps': timestamps,
            'rates': rates,
            'pattern': self.pattern.value,
            'statistics': {
                'total_inferences': len(timestamps),
                'avg_rate_fps': np.mean(rates),
                'peak_rate_fps': np.max(rates),
                'min_rate_fps': np.min(rates),
                'duration_s': self.duration_s
            }
        }

    def _generate_periodic(self) -> Dict:
        """
        Generate periodic burst workload.

        Simulates scheduled spectrum scanning operations that occur at regular intervals.
        """
        timestamps = []
        rates = []

        # Calculate number of periods
        num_periods = int(self.duration_s / self.periodic_interval_s) + 1
        burst_duration = min(5.0, self.periodic_interval_s * 0.3)  # Burst is 30% of interval

        for period in range(num_periods):
            burst_start = period * self.periodic_interval_s

            if burst_start >= self.duration_s:
                break

            burst_end = min(burst_start + burst_duration, self.duration_s)

            # Generate inferences during burst
            num_inferences = int(self.base_rate_fps * (burst_end - burst_start))
            if num_inferences > 0:
                burst_times = np.linspace(burst_start, burst_end, num_inferences, endpoint=False)
                timestamps.extend(burst_times)
                rates.extend([self.base_rate_fps] * num_inferences)

        timestamps = np.array(timestamps)
        rates = np.array(rates)

        return {
            'timestamps': timestamps,
            'rates': rates,
            'pattern': self.pattern.value,
            'statistics': {
                'total_inferences': len(timestamps),
                'avg_rate_fps': np.mean(rates),
                'peak_rate_fps': np.max(rates),
                'min_rate_fps': np.min(rates),
                'duration_s': self.duration_s
            }
        }

    def _generate_random(self) -> Dict:
        """
        Generate random arrival pattern (Poisson process).

        Simulates random signal arrivals following a Poisson distribution.
        """
        # Generate inter-arrival times from exponential distribution
        avg_inter_arrival = 1.0 / self.base_rate_fps
        timestamps = []
        current_time = 0.0

        while current_time < self.duration_s:
            inter_arrival = np.random.exponential(avg_inter_arrival)
            current_time += inter_arrival
            if current_time < self.duration_s:
                timestamps.append(current_time)

        timestamps = np.array(timestamps)

        # Calculate instantaneous rates (using window of 1 second)
        rates = []
        window_size = 1.0
        for t in timestamps:
            window_start = max(0, t - window_size/2)
            window_end = min(self.duration_s, t + window_size/2)
            count = np.sum((timestamps >= window_start) & (timestamps <= window_end))
            rate = count / window_size
            rates.append(rate)

        rates = np.array(rates)

        return {
            'timestamps': timestamps,
            'rates': rates,
            'pattern': self.pattern.value,
            'statistics': {
                'total_inferences': len(timestamps),
                'avg_rate_fps': np.mean(rates),
                'peak_rate_fps': np.max(rates),
                'min_rate_fps': np.min(rates),
                'duration_s': self.duration_s
            }
        }

    def generate_iterator(self) -> Iterator[Tuple[float, float]]:
        """
        Generate an iterator that yields (timestamp, rate) pairs.

        Useful for real-time simulation where you want to wait for scheduled times.

        Yields:
            Tuple of (timestamp, rate_fps)
        """
        schedule = self.generate_schedule()
        for timestamp, rate in zip(schedule['timestamps'], schedule['rates']):
            yield (timestamp, rate)

    def print_schedule_summary(self):
        """Print a summary of the generated workload schedule."""
        schedule = self.generate_schedule()
        stats = schedule['statistics']

        print("\n" + "="*60)
        print(f"WORKLOAD SCHEDULE: {schedule['pattern'].upper()}")
        print("="*60)
        print(f"Duration: {stats['duration_s']:.1f} seconds")
        print(f"Total Inferences: {stats['total_inferences']}")
        print(f"Average Rate: {stats['avg_rate_fps']:.2f} FPS")
        print(f"Peak Rate: {stats['peak_rate_fps']:.2f} FPS")
        print(f"Min Rate: {stats['min_rate_fps']:.2f} FPS")
        print("="*60 + "\n")


# Test function for development
if __name__ == "__main__":
    print("📊 Testing Workload Generator\n")

    patterns = [
        WorkloadPattern.BURSTY,
        WorkloadPattern.CONTINUOUS,
        WorkloadPattern.VARIABLE,
        WorkloadPattern.PERIODIC,
        WorkloadPattern.RANDOM
    ]

    for pattern in patterns:
        generator = WorkloadGenerator(
            pattern=pattern,
            duration_s=60.0,
            base_rate_fps=100.0,
            seed=42
        )

        generator.print_schedule_summary()

        # Show first few timestamps
        schedule = generator.generate_schedule()
        print(f"First 10 timestamps for {pattern.value}:")
        for i, (t, r) in enumerate(zip(schedule['timestamps'][:10], schedule['rates'][:10])):
            print(f"  {i+1}. t={t:.3f}s, rate={r:.1f} FPS")
        print()
