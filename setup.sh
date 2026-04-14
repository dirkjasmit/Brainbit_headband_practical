#!/usr/bin/env bash
# BrainBit EEG Streamer — environment setup
# Run once after cloning:  bash setup.sh

set -e

echo "Creating conda environment brainbit_311..."
conda env create -f environment.yml

echo "Installing supabase..."
conda run -n brainbit_311 pip install supabase

echo "Replacing bundled neurosdk dylib with v1.0.23..."
SITE=$(conda run -n brainbit_311 python -c \
  "import neurosdk, os; print(os.path.dirname(neurosdk.__file__))")
cp sdk2_lib/libneurosdk2.dylib "$SITE/libs/macos/libneurosdk2.dylib"

echo ""
echo "Done. Activate with:  conda activate brainbit_311"
