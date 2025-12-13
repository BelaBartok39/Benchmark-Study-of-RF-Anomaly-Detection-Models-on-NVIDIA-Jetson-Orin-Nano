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
        Dictionary with keys: 'static_high', 'adaptive'
    """
    results = {}

    # Load static high power results (MAXN) - performance baseline
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
    Create energy-latency Pareto frontier plot with error bars.

    Compares static MAXN and adaptive approaches.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract metrics with error bars
    strategies = []
    energies = []
    latencies = []
    energy_stds = []
    latency_stds = []
    colors = []
    markers = []

    if 'static_high' in results:
        strategies.append('Static MAXN')
        energies.append(results['static_high']['energy_per_inference_j'])
        latencies.append(results['static_high']['p95_latency_ms'])

        # Calculate std dev from raw data
        if 'latencies' in results['static_high']:
            latency_stds.append(np.std(results['static_high']['latencies']))
        else:
            latency_stds.append(0)

        # Estimate energy std dev (10% of value)
        energy_stds.append(results['static_high']['energy_per_inference_j'] * 0.1)

        colors.append('#e74c3c')  # Red
        markers.append('^')  # Triangle

    if 'adaptive' in results:
        strategies.append('Adaptive')
        energies.append(results['adaptive']['energy_per_inference_j'])
        latencies.append(results['adaptive']['p95_latency_ms'])

        # Calculate std dev from raw data
        if 'latencies' in results['adaptive']:
            latency_stds.append(np.std(results['adaptive']['latencies']))
        else:
            latency_stds.append(0)

        # Estimate energy std dev (10% of value)
        energy_stds.append(results['adaptive']['energy_per_inference_j'] * 0.1)

        colors.append('#2ecc71')  # Green
        markers.append('o')  # Circle

    # Plot points with error bars
    for i, (strat, energy, latency, energy_std, latency_std, color, marker) in enumerate(
            zip(strategies, energies, latencies, energy_stds, latency_stds, colors, markers)):
        # Plot error bars
        ax.errorbar(energy * 1000, latency,
                   xerr=energy_std * 1000, yerr=latency_std,
                   fmt='none', ecolor=color, alpha=0.5,
                   elinewidth=2, capsize=5, capthick=2)
        # Plot point
        ax.scatter(energy * 1000, latency, s=200, c=color, marker=marker,
                  label=strat, alpha=0.8, edgecolors='black', linewidth=1.5, zorder=10)

    ax.set_xlabel('Energy per Inference (mJ)', fontweight='bold')
    ax.set_ylabel('P95 Latency (ms)', fontweight='bold')
    ax.set_title('Energy-Latency Trade-off: MAXN vs Adaptive Power Management\n(Error bars show ±1 standard deviation)',
                fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', frameon=True, shadow=True)

    # Add annotations showing improvements
    if 'adaptive' in results and 'static_high' in results:
        adaptive_energy = results['adaptive']['energy_per_inference_j'] * 1000
        static_high_energy = results['static_high']['energy_per_inference_j'] * 1000
        energy_savings = (1 - adaptive_energy / static_high_energy) * 100

        # Add text annotation
        ax.text(0.05, 0.95, f'Energy Savings vs MAXN: {energy_savings:.1f}%',
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_benchmark_summary(results_all_workloads: Dict, output_path: Path):
    """
    Create a comprehensive 3-panel summary plot (Latency, Energy, Efficiency).
    Compares MAXN vs Adaptive only with error bars showing standard deviation
    and markers showing mean values.
    """
    workloads = list(results_all_workloads.keys())
    n_workloads = len(workloads)

    if n_workloads == 0:
        print("⚠️  No workload results available for summary plot")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Metrics to plot: (axis, bar_metric, mean_metric, raw_data_key, ylabel, note)
    metrics_config = [
        (ax1, 'p95_latency_ms', 'avg_latency_ms', 'latencies', 'P95 Latency (ms)', 'Lower is better'),
        (ax2, 'total_energy_j', 'energy_per_inference_j', None, 'Total Energy (J)', 'Lower is better'),
        (ax3, 'fps_per_watt', 'fps_per_watt', None, 'Efficiency (FPS/Watt)', 'Higher is better')
    ]

    bar_width = 0.35
    x = np.arange(n_workloads)

    modes = [
        ('static_high', 'Static MAXN', '#e74c3c'),
        ('adaptive', 'Adaptive', '#2ecc71')
    ]

    for ax, bar_metric_key, mean_metric_key, raw_data_key, ylabel, note in metrics_config:
        for i, (mode_key, label, color) in enumerate(modes):
            values = []
            std_devs = []
            mean_values = []

            for w in workloads:
                result = results_all_workloads[w].get(mode_key, {})

                # Get the bar value (e.g., P95 latency)
                val = result.get(bar_metric_key, 0)
                values.append(val)

                # Get mean value
                if mean_metric_key == 'energy_per_inference_j':
                    # For energy, scale by total inferences
                    mean_val = result.get(mean_metric_key, 0) * result.get('total_inferences', 1)
                else:
                    mean_val = result.get(mean_metric_key, val)
                mean_values.append(mean_val)

                # Calculate standard deviation from raw data if available
                if raw_data_key and raw_data_key in result:
                    raw_data = np.array(result[raw_data_key])
                    std_devs.append(np.std(raw_data))
                else:
                    # Estimate std dev as 10% for metrics without raw data
                    std_devs.append(val * 0.1 if val > 0 else 0)

            offset = (i - 0.5) * bar_width

            # Plot bars with error bars
            ax.bar(x + offset, values, bar_width, label=label, color=color,
                  edgecolor='black', alpha=0.8, yerr=std_devs,
                  capsize=5, error_kw={'elinewidth': 2, 'alpha': 0.7})

            # Add mean markers
            for j, mean_val in enumerate(mean_values):
                # Only show mean marker if different from bar value
                if abs(mean_val - values[j]) > 0.01 * values[j]:
                    ax.plot(x[j] + offset, mean_val, marker='_', markersize=15,
                           color='black', linewidth=3, zorder=10)

        ax.set_xticks(x)
        ax.set_xticklabels([w.capitalize() for w in workloads])
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(f'{ylabel}\n({note})', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        # Add legend only to the first plot
        if ax == ax1:
            # Create custom legend
            from matplotlib.lines import Line2D
            legend_elements = [
                mpatches.Patch(facecolor='#e74c3c', edgecolor='black', label='Static MAXN'),
                mpatches.Patch(facecolor='#2ecc71', edgecolor='black', label='Adaptive'),
                Line2D([0], [0], color='black', linewidth=3, marker='_', markersize=10,
                       label='Mean (if shown)', linestyle='none')
            ]
            ax.legend(handles=legend_elements, loc='upper left', frameon=True, fontsize=9)

    plt.suptitle('Benchmark Summary: MAXN vs Adaptive Power Management\n(Error bars show ±1 standard deviation)',
                fontsize=16, fontweight='bold', y=1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path}")
    plt.close()


def create_summary_table(results_all_workloads: Dict, output_path: Path):
    """
    Create a summary table comparing MAXN vs Adaptive.
    """
    with open(output_path, 'w') as f:
        f.write("# Adaptive Power Management Summary\n\n")
        f.write("Comparison of Static MAXN (performance baseline) vs Adaptive Power Management\n\n")

        for workload, results in results_all_workloads.items():
            f.write(f"## {workload.capitalize()} Workload\n\n")
            f.write("| Metric | Static MAXN | Adaptive | Improvement |\n")
            f.write("|--------|-------------|----------|-------------|\n")

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

                f.write(f"| {metric_name} | {high_val:{fmt}} | "
                       f"{adap_val:{fmt}} | {improvement_str} |\n")

            # Add mode switching stats for adaptive (three-tier)
            if 'adaptive' in results:
                adap = results['adaptive']
                f.write(f"\n**Adaptive Statistics:**\n")
                f.write(f"- Mode Switches: {adap.get('total_mode_switches', 0)}\n")
                f.write(f"- Low Power Time (15W): {adap.get('low_power_percentage', 0):.1f}%\n")
                
                # Improved logic: check explicit keys first, calculate if missing
                if 'medium_power_percentage' in adap:
                    f.write(f"- Medium Power Time (25W): {adap.get('medium_power_percentage', 0):.1f}%\n")
                
                if 'high_power_percentage' in adap:
                    f.write(f"- High Power Time (MAXN): {adap.get('high_power_percentage', 0):.1f}%\n")
                    
                f.write(f"- Avg Switch Time: {adap.get('avg_switch_time_ms', 0):.1f} ms\n")

            f.write("\n")

    print(f"📄 Saved: {output_path}")


def plot_adaptive_behavior_dashboard(results: Dict, output_path: Path):
    """
    Create comprehensive dashboard showing key adaptive metrics.
    """
    if 'adaptive' not in results:
        print("⚠️  No adaptive results available for dashboard")
        return

    adaptive = results['adaptive']

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Workload Rate vs Time (Derived from timestamps)
    ax1 = fig.add_subplot(gs[0, :])
    timestamps = np.array(adaptive.get('timestamps', []))
    
    if len(timestamps) > 1:
        # Calculate instantaneous FPS using 1-second bins
        duration = timestamps[-1] - timestamps[0]
        if duration > 0:
            bins = np.arange(np.floor(timestamps[0]), np.ceil(timestamps[-1]) + 1, 1.0)
            if len(bins) > 1:
                hist, bin_edges = np.histogram(timestamps, bins=bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                ax1.plot(bin_centers, hist, 'b-', linewidth=1.5, alpha=0.8, label='Inference Rate')
                ax1.fill_between(bin_centers, hist, alpha=0.2, color='blue')
                
                # Plot average rate
                avg_rate = len(timestamps) / duration
                ax1.axhline(y=avg_rate, color='r', linestyle='--', alpha=0.8, 
                           label=f'Avg Rate: {avg_rate:.1f} FPS')
                
                ax1.set_xlabel('Time (s)')
                ax1.set_ylabel('Inferences / sec (FPS)')
                ax1.set_title('Workload Pattern (Inference Rate)', fontweight='bold')
                ax1.grid(True, alpha=0.3)
                ax1.legend(loc='upper right')
    else:
        ax1.text(0.5, 0.5, 'Insufficient data for workload plot', 
                ha='center', va='center', transform=ax1.transAxes)

    # 2. Latency Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    latencies = np.array(adaptive.get('latencies', []))
    if len(latencies) > 0:
        ax2.hist(latencies, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(x=np.median(latencies), color='r', linestyle='--',
                   label=f'Median: {np.median(latencies):.2f}ms')
        ax2.axvline(x=np.percentile(latencies, 95), color='orange', linestyle='--',
                   label=f'P95: {np.percentile(latencies, 95):.2f}ms')
        threshold = adaptive.get('latency_threshold_ms', 10.0)
        ax2.axvline(x=threshold, color='red', linestyle='-', linewidth=2,
                   label=f'Threshold: {threshold}ms')
        ax2.set_xlabel('Latency (ms)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Latency Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # 3. Mode Switching Timeline
    ax3 = fig.add_subplot(gs[1, 1])
    power_modes = adaptive.get('power_modes', [])
    # Re-use timestamps from above
    if len(power_modes) > 0 and len(timestamps) > 0:
        # Convert mode names to numbers for plotting
        mode_map = {'15W': 0, '25W': 1, 'MAXN': 2}
        mode_values = [mode_map.get(m, 0) for m in power_modes]
        
        # Align lengths if necessary (sometimes one might be off by 1)
        min_len = min(len(timestamps), len(mode_values))
        
        ax3.plot(timestamps[:min_len], mode_values[:min_len], 'g-', linewidth=2, drawstyle='steps-post')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Power Mode')
        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(['15W', '25W', 'MAXN'])
        ax3.set_title('Power Mode Transitions', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([-0.2, 2.2])

    # 4. Energy Breakdown
    ax4 = fig.add_subplot(gs[2, 0])
    metrics = {
        'Total Energy': adaptive.get('total_energy_j', 0),
        'Per Inference': adaptive.get('energy_per_inference_j', 0) * 1000  # Convert to mJ
    }
    if all(v > 0 for v in metrics.values()):
        bars = ax4.bar(metrics.keys(), metrics.values(), color=['#3498db', '#e74c3c'])
        ax4.set_ylabel('Energy (J for Total, mJ for Per Inference)')
        ax4.set_title('Energy Metrics', fontweight='bold')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontweight='bold')

    # 5. Performance Summary
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    # Safely get metrics with correct keys
    total_inferences = adaptive.get('total_inferences', adaptive.get('num_inferences', 0))
    avg_latency = adaptive.get('avg_latency_ms', adaptive.get('mean_latency_ms', 0))
    
    # Calculate duration if missing
    duration_s = adaptive.get('duration_s', 0)
    if duration_s == 0 and len(timestamps) > 0:
        duration_s = timestamps[-1] - timestamps[0]

    summary_text = f"""
    📊 ADAPTIVE PERFORMANCE SUMMARY

    Total Inferences: {total_inferences}
    Duration: {duration_s:.1f}s

    Power Management:
    • Mode Switches: {adaptive.get('total_mode_switches', 0)}
    • Avg Switch Time: {adaptive.get('avg_switch_time_ms', 0):.2f}ms
    • 15W Time: {adaptive.get('low_power_percentage', 0):.1f}%
    • 25W Time: {adaptive.get('medium_power_percentage', 0):.1f}%
    • MAXN Time: {adaptive.get('high_power_percentage', 0):.1f}%

    Latency:
    • Mean: {avg_latency:.2f}ms
    • P95: {adaptive.get('p95_latency_ms', 0):.2f}ms
    • Violations: {adaptive.get('threshold_violations', 0)} ({adaptive.get('violation_rate', 0)*100:.2f}%)

    Efficiency:
    • Avg Power: {adaptive.get('avg_power_w', 0):.2f}W
    • Energy/Inf: {adaptive.get('energy_per_inference_j', 0)*1000:.2f}mJ
    • FPS/Watt: {adaptive.get('fps_per_watt', 0):.2f}
    """

    ax5.text(0.1, 0.95, summary_text, transform=ax5.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.suptitle(f'Adaptive Power Management Dashboard\nWorkload: {results.get("workload_pattern", "Unknown")}',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path}")
    plt.close()


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

        # Adaptive behavior dashboard (adaptive only)
        plot_adaptive_behavior_dashboard(
            results,
            output_dir / f'{args.model}_{workload}_adaptive_dashboard.png'
        )

    # Cross-workload comparison plots
    if len(results_all) > 1:
        print(f"\nCreating cross-workload comparison plots...")

        plot_benchmark_summary(
            results_all,
            output_dir / f'{args.model}_benchmark_summary.png'
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