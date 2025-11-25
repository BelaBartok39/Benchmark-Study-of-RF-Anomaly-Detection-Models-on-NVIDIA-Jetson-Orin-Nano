#!/bin/bash
# Download RF Anomaly Detection Datasets from Hugging Face
# Dataset: https://huggingface.co/datasets/b4byn1cky/RF_Anomaly_Detection

set -e  # Exit on error

echo "======================================"
echo "RF Anomaly Detection Dataset Download"
echo "======================================"
echo ""
echo "This script will download the 5G RF datasets from Hugging Face."
echo "Dataset URL: https://huggingface.co/datasets/b4byn1cky/RF_Anomaly_Detection"
echo ""
echo "Total size: ~1.6GB (2 files, 785MB each)"
echo "Files to download:"
echo "  - clean_5g_dataset.h5"
echo "  - jammed_5g_dataset.h5"
echo ""

# Check if datasets already exist
if [ -f "clean_5g_dataset.h5" ] && [ -f "jammed_5g_dataset.h5" ]; then
    echo "Datasets already exist in the current directory."
    read -p "Do you want to re-download? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping download."
        exit 0
    fi
fi

# Check for huggingface-cli
if command -v huggingface-cli &> /dev/null; then
    echo "Using huggingface-cli for download..."
    echo ""

    # Create temporary directory
    mkdir -p .tmp_datasets

    # Download using huggingface-cli
    huggingface-cli download b4byn1cky/RF_Anomaly_Detection \
        --repo-type dataset \
        --local-dir .tmp_datasets

    # Move files to current directory
    mv .tmp_datasets/*.h5 .

    # Clean up
    rm -rf .tmp_datasets

    echo ""
    echo "Download complete!"

elif command -v wget &> /dev/null; then
    echo "Huggingface-cli not found. Using wget for manual download..."
    echo ""
    echo "NOTE: You may need to manually download from:"
    echo "https://huggingface.co/datasets/b4byn1cky/RF_Anomaly_Detection"
    echo ""
    echo "For now, attempting direct download (may require authentication)..."

    # Attempt direct download (may not work without authentication)
    BASE_URL="https://huggingface.co/datasets/b4byn1cky/RF_Anomaly_Detection/resolve/main"

    wget -O clean_5g_dataset.h5 "${BASE_URL}/clean_5g_dataset.h5" || {
        echo "Direct download failed. Please visit the URL above and download manually."
        exit 1
    }

    wget -O jammed_5g_dataset.h5 "${BASE_URL}/jammed_5g_dataset.h5" || {
        echo "Direct download failed. Please visit the URL above and download manually."
        exit 1
    }

    echo ""
    echo "Download complete!"

else
    echo "ERROR: Neither huggingface-cli nor wget found."
    echo ""
    echo "Please install one of the following:"
    echo "  1. Install huggingface-hub: pip install huggingface-hub"
    echo "  2. Install wget: sudo apt-get install wget"
    echo ""
    echo "Or manually download from:"
    echo "https://huggingface.co/datasets/b4byn1cky/RF_Anomaly_Detection"
    exit 1
fi

# Verify downloads
echo ""
echo "Verifying downloads..."
if [ -f "clean_5g_dataset.h5" ] && [ -f "jammed_5g_dataset.h5" ]; then
    echo "✓ clean_5g_dataset.h5 ($(du -h clean_5g_dataset.h5 | cut -f1))"
    echo "✓ jammed_5g_dataset.h5 ($(du -h jammed_5g_dataset.h5 | cut -f1))"
    echo ""
    echo "Datasets successfully downloaded!"
    echo ""
    echo "You can now run the benchmark:"
    echo "  python run_sustained_benchmark.py"
else
    echo "ERROR: Download verification failed. Files not found."
    exit 1
fi
