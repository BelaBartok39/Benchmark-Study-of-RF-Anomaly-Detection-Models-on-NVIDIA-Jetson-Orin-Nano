#!/usr/bin/env python3
"""
Analyze thermal validation results to demonstrate temperature impact on power.
"""

import json
import sys
from pathlib import Path

def analyze_thermal_test(results_dir):
    """Analyze thermal validation results from two back-to-back runs."""

    results_dir = Path(results_dir)

    print("="*70)
    print("THERMAL VALIDATION ANALYSIS")
    print("="*70)
    print()

    # Load both runs
    runs = []
    for run_num in [1, 2]:
        run_dir = results_dir / f"run{run_num}"

        # Try to find the adaptive continuous results
        adaptive_file = list(run_dir.glob("*_adaptive_continuous_results.json"))
        static_file = list(run_dir.glob("*_static_high_continuous_results.json"))

        if not adaptive_file or not static_file:
            print(f"❌ Error: Could not find results files in {run_dir}")
            continue

        with open(adaptive_file[0]) as f:
            adaptive = json.load(f)
        with open(static_file[0]) as f:
            static = json.load(f)

        runs.append({'adaptive': adaptive, 'static': static, 'run': run_num})

    if len(runs) < 2:
        print("❌ Error: Need 2 complete runs for comparison")
        return

    # Compare runs
    print("STATIC MAXN Baseline Comparison:")
    print("-" * 70)
    print(f"{'Metric':<30} {'Run 1':<20} {'Run 2':<20}")
    print("-" * 70)

    r1_static = runs[0]['static']
    r2_static = runs[1]['static']

    metrics = [
        ('Avg Power (W)', 'avg_power_w'),
        ('Peak Power (W)', 'peak_power_w'),
        ('Total Energy (J)', 'total_energy_j'),
        ('Avg CPU Temp (°C)', 'avg_temp_cpu_c'),
        ('Peak CPU Temp (°C)', 'peak_temp_cpu_c'),
        ('Avg GPU Temp (°C)', 'avg_temp_gpu_c'),
        ('Peak GPU Temp (°C)', 'peak_temp_gpu_c'),
        ('Avg SOC Temp (°C)', 'avg_temp_soc_c'),
        ('Peak SOC Temp (°C)', 'peak_temp_soc_c'),
    ]

    for label, key in metrics:
        val1 = r1_static.get(key, 0)
        val2 = r2_static.get(key, 0)

        if val1 == 0 and val2 == 0:
            continue  # Skip if both are zero (temp not logged)

        diff = val2 - val1
        diff_pct = (100 * diff / val1) if val1 != 0 else 0

        marker = ""
        if abs(diff_pct) > 3:
            marker = " ⚠️"

        print(f"{label:<30} {val1:<20.2f} {val2:<20.2f} ({diff_pct:+.1f}%){marker}")

    print()
    print("ADAPTIVE Mode Comparison:")
    print("-" * 70)

    r1_adaptive = runs[0]['adaptive']
    r2_adaptive = runs[1]['adaptive']

    for label, key in metrics:
        val1 = r1_adaptive.get(key, 0)
        val2 = r2_adaptive.get(key, 0)

        if val1 == 0 and val2 == 0:
            continue

        diff = val2 - val1
        diff_pct = (100 * diff / val1) if val1 != 0 else 0

        marker = ""
        if abs(diff_pct) > 3:
            marker = " ⚠️"

        print(f"{label:<30} {val1:<20.2f} {val2:<20.2f} ({diff_pct:+.1f}%){marker}")

    print()
    print("="*70)
    print("KEY FINDINGS:")
    print("-" * 70)

    # Check if temperature data was captured
    if r1_static.get('avg_temp_cpu_c', 0) == 0:
        print("⚠️  WARNING: No temperature data captured!")
        print("   Check that tegrastats is working and temperature parsing is correct.")
    else:
        # Analyze thermal impact
        temp_increase = r2_static.get('avg_temp_cpu_c', 0) - r1_static.get('avg_temp_cpu_c', 0)
        power_increase = r2_static.get('avg_power_w', 0) - r1_static.get('avg_power_w', 0)

        print(f"✓ Temperature increase (Run 1 → Run 2): {temp_increase:+.1f}°C")
        print(f"✓ Power increase (Run 1 → Run 2): {power_increase:+.3f}W ({100*power_increase/r1_static.get('avg_power_w', 1):+.1f}%)")

        if temp_increase > 3:
            print(f"\n✓ THERMAL CARRYOVER DETECTED:")
            print(f"  Run 2 started {temp_increase:.1f}°C warmer than Run 1")

            if abs(power_increase) > 0.1:
                print(f"  This correlated with {abs(power_increase):.3f}W power difference")
                print(f"  Estimated: ~{abs(power_increase)/temp_increase:.3f}W per °C")
        else:
            print("\n✓ Minimal thermal carryover between runs")
            print(f"  30-second cooldown was sufficient for thermal equilibrium")

    print("="*70)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_thermal_results.py <results_directory>")
        print("Example: python analyze_thermal_results.py thermal_validation_20251219_103045")
        sys.exit(1)

    analyze_thermal_test(sys.argv[1])
