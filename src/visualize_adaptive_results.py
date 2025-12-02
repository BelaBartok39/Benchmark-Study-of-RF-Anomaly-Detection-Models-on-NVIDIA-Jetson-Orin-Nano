#!/usr/bin/env python3
"""
Visualization Tools for Adaptive Power Management Results

Creates publication-quality figures comparing static vs adaptive power management.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List
import argparse


# Set publication-quality plot style
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14


def load_results(results_dir: Path, model: str, workload: str) -> Dict:
    """
    Load all results for a given model and workload.

    Returns:
        Dictionary with keys: 'static_low', 'static_high', 'adaptive'
    """
    results = {}

    # Load static low power results
    low_file = results_dir / f'{model}_static_low_{workload}_results.json'
    if low_file.exists():
        with open(low_file, 'r') as f:
            results['static_low'] = json.load(f)

    # Load static high power results
    high_file = results_dir / f'{model}_static_high_{workload}_results.json'
    if high_file.exists():
        with open(high_file, 'r') as f:
            results['static_high'] = json.load(f)

    # Load adaptive results
    adaptive_file = results_dir / f'{model}_adaptive_{workload}_results.json'
    if adaptive_file.exists():
        with open(adaptive_file, 'r') as f:
            results['adaptive'] = json.load(f)

    return results


def plot_energy_latency_tradeoff(results: Dict, output_path: Path):
    """
    Create energy-latency Pareto frontier plot.

    Compares static low, static high, and adaptive approaches.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract metrics
    strategies = []
    energies = []
    latencies = []
    colors = []
    markers = []

    if 'static_low' in results:
        strategies.append('Static 15W')
        energies.append(results['static_low']['energy_per_inference_j'])
        latencies.append(results['static_low']['p95_latency_ms'])
        colors.append('#3498db')  # Blue
        markers.append('s')  # Square

    if 'static_high' in results:
        strategies.append('Static MAXN')
        energies.append(results['static_high']['energy_per_inference_j'])
        latencies.append(results['static_high']['p95_latency_ms'])
        colors.append('#e74c3c')  # Red
        markers.append('^')  # Triangle

    if 'adaptive' in results:
        strategies.append('Adaptive')
        energies.append(results['adaptive']['energy_per_inference_j'])
        latencies.append(results['adaptive']['p95_latency_ms'])
        colors.append('#2ecc71')  # Green
        markers.append('o')  # Circle

    # Plot points
    for i, (strat, energy, latency, color, marker) in enumerate(
            zip(strategies, energies, latencies, colors, markers)):
        ax.scatter(energy * 1000, latency, s=200, c=color, marker=marker,
                  label=strat, alpha=0.8, edgecolors='black', linewidth=1.5)

    ax.set_xlabel('Energy per Inference (mJ)', fontweight='bold')
    ax.set_ylabel('P95 Latency (ms)', fontweight='bold')
    ax.set_title('Energy-Latency Trade-off: Static vs Adaptive Power Management',
                fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', frameon=True, shadow=True)

    # Add annotations showing improvements
    if 'adaptive' in results and 'static_high' in results:
        adaptive_energy = results['adaptive']['energy_per_inference_j'] * 1000
        static_high_energy = results['static_high']['energy_per_inference_j'] * 1000
        energy_savings = (1 - adaptive_energy / static_high_energy) * 100

        # Add text annotation
        ax.text(0.05, 0.95, f'Energy Savings: {energy_savings:.1f}%',
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_latency_timeline(results: Dict, output_path: Path):
    """
    Plot latency over time showing adaptive behavior.
    """
    if 'adaptive' not in results:
        print("⚠️  No adaptive results available for timeline plot")
        return

    adaptive = results['adaptive']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})

    timestamps = np.array(adaptive['timestamps'])
    latencies = np.array(adaptive['latencies'])
    power_modes = adaptive['power_modes']

    # Plot 1: Latency over time
    ax1.plot(timestamps, latencies, 'b-', linewidth=1, alpha=0.6, label='Latency')

    # Mark threshold
    threshold = adaptive.get('latency_threshold_ms', 10.0)
    ax1.axhline(y=threshold, color='r', linestyle='--', linewidth=2,
               label=f'Threshold ({threshold}ms)')

    # Color background based on violations
    violations = latencies > threshold
    if violations.any():
        violation_regions = []
        in_violation = False
        start_idx = 0

        for i, v in enumerate(violations):
            if v and not in_violation:
                start_idx = i
                in_violation = True
            elif not v and in_violation:
                violation_regions.append((timestamps[start_idx], timestamps[i-1]))
                in_violation = False

        if in_violation:
            violation_regions.append((timestamps[start_idx], timestamps[-1]))

        for start, end in violation_regions:
            ax1.axvspan(start, end, alpha=0.2, color='red')

    ax1.set_ylabel('Latency (ms)', fontweight='bold')
    ax1.set_title('Latency Timeline with Adaptive Power Management', fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right')

    # Plot 2: Power mode over time
    mode_numeric = [1 if mode == 'MAXN' else 0 for mode in power_modes]
    ax2.fill_between(timestamps, mode_numeric, step='post', alpha=0.5, color='orange')
    ax2.set_ylabel('Power Mode', fontweight='bold')
    ax2.set_xlabel('Time (s)', fontweight='bold')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['15W', 'MAXN'])
    ax2.grid(True, alpha=0.3, linestyle='--', axis='x')

    # Mark mode switches
    mode_switches = adaptive.get('mode_switches', [])
    for switch in mode_switches:
        switch_time = switch['timestamp'] - (timestamps[0] if len(timestamps) > 0 else 0)
        ax2.axvline(x=switch_time, color='red', linestyle=':', alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_comparison_bars(results_all_workloads: Dict, metric: str,
                        output_path: Path, ylabel: str):
    """
    Create grouped bar chart comparing metrics across workloads.

    Args:
        results_all_workloads: Dict mapping workload -> results dict
        metric: Metric key to plot
        ylabel: Y-axis label
    """
    workloads = list(results_all_workloads.keys())
    n_workloads = len(workloads)

    if n_workloads == 0:
        print(f"⚠️  No workload results available for {metric} plot")
        return

    # Prepare data
    static_low_values = []
    static_high_values = []
    adaptive_values = []

    for workload in workloads:
        results = results_all_workloads[workload]
        static_low_values.append(results.get('static_low', {}).get(metric, 0))
        static_high_values.append(results.get('static_high', {}).get(metric, 0))
        adaptive_values.append(results.get('adaptive', {}).get(metric, 0))

    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(n_workloads)
    width = 0.25

    bars1 = ax.bar(x - width, static_low_values, width, label='Static 15W',
                   color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, static_high_values, width, label='Static MAXN',
                   color='#e74c3c', alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + width, adaptive_values, width, label='Adaptive',
                   color='#2ecc71', alpha=0.8, edgecolor='black')

    ax.set_xlabel('Workload Pattern', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(f'{ylabel} Comparison Across Workloads', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([w.capitalize() for w in workloads])
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_efficiency_comparison(results_all_workloads: Dict, output_path: Path):
    """
    Create efficiency comparison chart (FPS/Watt).
    """
    plot_comparison_bars(
        results_all_workloads,
        metric='fps_per_watt',
        output_path=output_path,
        ylabel='Energy Efficiency (FPS/Watt)'
    )


def plot_energy_comparison(results_all_workloads: Dict, output_path: Path):
    """
    Create energy consumption comparison chart.
    """
    plot_comparison_bars(
        results_all_workloads,
        metric='total_energy_j',
        output_path=output_path,
        ylabel='Total Energy Consumption (J)'
    )


def plot_latency_comparison(results_all_workloads: Dict, output_path: Path):
    """
    Create P95 latency comparison chart.
    """
    plot_comparison_bars(
        results_all_workloads,
        metric='p95_latency_ms',
        output_path=output_path,
        ylabel='P95 Latency (ms)'
    )


def create_summary_table(results_all_workloads: Dict, output_path: Path):
    """
    Create a summary table comparing all approaches.
    """
    with open(output_path, 'w') as f:
        f.write("# Adaptive Power Management Summary\n\n")

        for workload, results in results_all_workloads.items():
            f.write(f"## {workload.capitalize()} Workload\n\n")
            f.write("| Metric | Static 15W | Static MAXN | Adaptive | Improvement |\n")
            f.write("|--------|-----------|-------------|----------|-------------|\n")

            metrics = [
                ('P95 Latency (ms)', 'p95_latency_ms', '.2f'),
                ('Avg Power (W)', 'avg_power_w', '.2f'),
                ('Total Energy (J)', 'total_energy_j', '.2f'),
                ('Energy/Inference (mJ)', 'energy_per_inference_j', '.3f', 1000),
                ('FPS/Watt', 'fps_per_watt', '.2f'),
                ('Violation Rate (%)', 'violation_rate', '.2f', 100)
            ]

            for metric_name, metric_key, fmt, *scale in metrics:
                scale = scale[0] if scale else 1

                low_val = results.get('static_low', {}).get(metric_key, 0) * scale
                high_val = results.get('static_high', {}).get(metric_key, 0) * scale
                adap_val = results.get('adaptive', {}).get(metric_key, 0) * scale

                # Calculate improvement over static high (performance baseline)
                if high_val != 0:
                    if 'Energy' in metric_name or 'Power' in metric_name:
                        # Lower is better
                        improvement = (1 - adap_val / high_val) * 100
                        improvement_str = f"{improvement:+.1f}%"
                    else:
                        # Higher is better or neutral
                        improvement = ((adap_val - high_val) / high_val) * 100
                        improvement_str = f"{improvement:+.1f}%"
                else:
                    improvement_str = "N/A"

                f.write(f"| {metric_name} | {low_val:{fmt}} | {high_val:{fmt}} | "
                       f"{adap_val:{fmt}} | {improvement_str} |\n")

            # Add mode switching stats for adaptive
            if 'adaptive' in results:
                adap = results['adaptive']
                f.write(f"\n**Adaptive Statistics:**\n")
                f.write(f"- Mode Switches: {adap.get('total_mode_switches', 0)}\n")
                f.write(f"- Low Power Time: {adap.get('low_power_percentage', 0):.1f}%\n")
                f.write(f"- Avg Switch Time: {adap.get('avg_switch_time_ms', 0):.1f} ms\n")

            f.write("\n")

    print(f"📄 Saved: {output_path}")


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(
        description='Visualize adaptive power management results'
    )

    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing result JSON files')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name')
    parser.add_argument('--workloads', type=str, nargs='+',
                       default=['bursty', 'continuous', 'variable', 'periodic'],
                       help='Workload patterns to visualize')
    parser.add_argument('--output-dir', type=str, default='figures',
                       help='Output directory for figures')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📊 Generating visualizations for {args.model}")
    print(f"   Results: {results_dir}")
    print(f"   Output: {output_dir}\n")

    # Load results for all workloads
    results_all = {}
    for workload in args.workloads:
        results = load_results(results_dir, args.model, workload)
        if results:
            results_all[workload] = results
            print(f"✓ Loaded results for {workload} workload")

    if not results_all:
        print("❌ No results found!")
        return

    print(f"\n{'='*60}")
    print("GENERATING FIGURES")
    print(f"{'='*60}\n")

    # Generate plots for each workload
    for workload, results in results_all.items():
        print(f"Creating plots for {workload} workload...")

        # Energy-latency tradeoff
        plot_energy_latency_tradeoff(
            results,
            output_dir / f'{args.model}_{workload}_energy_latency.png'
        )

        # Latency timeline (adaptive only)
        plot_latency_timeline(
            results,
            output_dir / f'{args.model}_{workload}_timeline.png'
        )

    # Cross-workload comparison plots
    if len(results_all) > 1:
        print(f"\nCreating cross-workload comparison plots...")

        plot_efficiency_comparison(
            results_all,
            output_dir / f'{args.model}_efficiency_comparison.png'
        )

        plot_energy_comparison(
            results_all,
            output_dir / f'{args.model}_energy_comparison.png'
        )

        plot_latency_comparison(
            results_all,
            output_dir / f'{args.model}_latency_comparison.png'
        )

    # Create summary table
    create_summary_table(
        results_all,
        output_dir / f'{args.model}_summary.md'
    )

    print(f"\n{'='*60}")
    print("✅ Visualization complete!")
    print(f"📊 Figures saved to {output_dir}/")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
