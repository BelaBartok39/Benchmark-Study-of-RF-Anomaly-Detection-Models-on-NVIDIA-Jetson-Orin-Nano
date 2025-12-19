#!/bin/bash
#
# Quick Test: Validate Temperature Logging
#
# This test runs ResNet continuous workload twice back-to-back with
# minimal cooldown to demonstrate thermal accumulation effects.
#
# Expected outcome:
# - Run 1: Should show thermal accumulation during 10-min test
# - Run 2: Should start warmer and potentially show higher avg power
#

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="thermal_validation_${TIMESTAMP}"

echo "========================================================================"
echo "THERMAL LOGGING VALIDATION TEST"
echo "========================================================================"
echo "Output directory: $OUTPUT_DIR"
echo "Test: ResNet continuous workload (2 runs, 10 min each)"
echo "Purpose: Validate temperature logging and observe thermal effects"
echo ""

mkdir -p "$OUTPUT_DIR"

echo "Run 1: First continuous run (cold start)"
echo "--------------------------------------------------------------------"
python src/adaptive_benchmark.py \
    --model resnet_ae \
    --model-path src/output/weights/resnet_ae_best.pth \
    --workload continuous \
    --duration 600 \
    --hysteresis-time 30.0 \
    --target-sla 10.0 \
    --auto-calibrate \
    --run-baselines \
    --batch-size 1 \
    --num-channels 1 \
    --output-dir "$OUTPUT_DIR/run1" \
    2>&1 | tee "$OUTPUT_DIR/run1_log.txt"

echo ""
echo "⏳ Brief cooldown (30 seconds)..."
sleep 30

echo ""
echo "Run 2: Second continuous run (thermal carryover)"
echo "--------------------------------------------------------------------"
python src/adaptive_benchmark.py \
    --model resnet_ae \
    --model-path src/output/weights/resnet_ae_best.pth \
    --workload continuous \
    --duration 600 \
    --hysteresis-time 30.0 \
    --target-sla 10.0 \
    --auto-calibrate \
    --run-baselines \
    --batch-size 1 \
    --num-channels 1 \
    --output-dir "$OUTPUT_DIR/run2" \
    2>&1 | tee "$OUTPUT_DIR/run2_log.txt"

echo ""
echo "========================================================================"
echo "✅ VALIDATION TEST COMPLETE"
echo "========================================================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Check temperature data in JSON results files"
echo "  2. Compare avg/peak temperatures between run1 and run2"
echo "  3. Look for correlation between temperature and power consumption"
echo ""
