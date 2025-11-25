#!/usr/bin/env python3
"""
Sustained Academic Benchmarking Script
Run comprehensive 5-minute sustained inference tests for academic rigor.
"""

import subprocess
import sys
import os

def run_sustained_benchmark():
    """Run the sustained academic benchmarking with recommended settings."""
    
    print("🔬 SUSTAINED ACADEMIC BENCHMARKING")
    print("=" * 50)
    print("This will run rigorous 5-minute sustained inference tests")
    print("for each model and framework combination.")
    print()
    print("Features:")
    print("• 5 minutes sustained inference per model")
    print("• 30-second moving average windows") 
    print("• 60-second thermal cooldown between models")
    print("• Continuous inference (not batch processing)")
    print("• Power monitoring with 10Hz sampling")
    print("• Comprehensive time-series visualizations")
    print()
    
    # Estimate total time
    models = ['ae', 'aae', 'cnn_ae', 'lstm_ae', 'resnet_ae', 'ff']  # All 6 models
    total_time_minutes = len(models) * (5 * 2 + 1)  # 5 min PyTorch + 5 min TensorRT + 1 min cooldown
    
    print(f"Estimated time for {len(models)} models: {total_time_minutes} minutes")
    print()
    
    # Ask for confirmation
    response = input("Continue with sustained benchmarking? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("Aborted.")
        return
    
    # Construct command
    cmd = [
        sys.executable, 'src/comprehensive_benchmark.py',
        '--models'] + models + [
        '--sustained',
        '--sustained-duration', '300',  # 5 minutes
        '--thermal-cooldown', '60',     # 1 minute cooldown
        '--moving-window', '30',        # 30 second windows
        '--convert-tensorrt',
        '--max-samples', '2000',        # Limit dataset size for Jetson
        '--batch-size', '1',            # Single sample inference
        '--output-dir', 'sustained_results',
        '--weights-dir', 'src/output/weights'  # Use existing trained weights
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    # Run the benchmark
    # Use current directory instead of hardcoded path
    script_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        subprocess.run(cmd, check=True, cwd=script_dir)
        print("\n✅ Sustained benchmarking completed!")
        print("Check the 'sustained_results' directory for:")
        print("  • sustained_measurements.json - Raw time-series data")
        print("  • figures/sustained_*.png - Academic visualizations")
        print("  • benchmark_results.csv - Summary metrics")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Benchmarking failed with exit code {e.returncode}")
        return 1
    except KeyboardInterrupt:
        print("\n⚠️  Benchmarking interrupted by user")
        return 1
    
    return 0

def run_quick_test():
    """Run a quick 30-second test to verify the sustained mode works."""
    
    print("🧪 QUICK SUSTAINED MODE TEST")
    print("=" * 30)
    print("Running 30-second test with 1 model...")
    
    cmd = [
        sys.executable, 'src/comprehensive_benchmark.py',
        '--models', 'ae',
        '--sustained',
        '--sustained-duration', '30',   # 30 seconds for quick test
        '--thermal-cooldown', '10',     # 10 second cooldown
        '--moving-window', '10',        # 10 second windows
        '--max-samples', '500',         # Small dataset
        '--batch-size', '1',
        '--output-dir', 'test_sustained',
        '--weights-dir', 'src/output/weights'  # Use existing trained weights
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    # Use current directory instead of hardcoded path
    script_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        subprocess.run(cmd, check=True, cwd=script_dir)
        print("\n✅ Quick test completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Test failed with exit code {e.returncode}")
        return 1
    
    return 0

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        exit(run_quick_test())
    else:
        exit(run_sustained_benchmark())
