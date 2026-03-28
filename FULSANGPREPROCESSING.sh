#!/bin/bash
#SBATCH --job-name=fulsang_preprocessing
#SBATCH --output=fulsang_preprocessing_%j.out
#SBATCH --error=fulsang_preprocessing_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=tier3
#SBATCH --account=neurosteer

# FULSANGPREPROCESSING - Process DATA_preproc for Fulsang Dataset
# This script processes the preprocessed EEG and audio data from preproc_script.m
# and creates TFRecord files compatible with FULCCA.py

echo "=========================================="
echo "FULSANG PREPROCESSING"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check for data directory
DATA_PATH="Data/Fulsang/DATA_preproc"
if [ ! -d "$DATA_PATH" ]; then
    # Try alternative path
    DATA_PATH="/home/py9363/telluride_decoding/Data/Fulsang/DATA_preproc"
    if [ ! -d "$DATA_PATH" ]; then
        echo "ERROR: Data directory not found!"
        echo "  Tried: Data/Fulsang/DATA_preproc"
        echo "  Tried: $DATA_PATH"
        echo "  Please check the path and update this script"
        exit 1
    fi
fi

echo "Using data path: $DATA_PATH"

# Check if MATLAB files exist
MAT_COUNT=$(find "$DATA_PATH" -name "*.mat" 2>/dev/null | wc -l)
if [ $MAT_COUNT -eq 0 ]; then
    echo "ERROR: No MATLAB files found in $DATA_PATH"
    exit 1
fi

echo "Found $MAT_COUNT MATLAB file(s)"

# Run preprocessing
echo "=========================================="
echo "Running FULSANGPREPROCESSING..."
echo "=========================================="

python FULSANGPREPROCESSING.py \
    --data_path "$DATA_PATH" \
    --output_dir fulsang_preprocessed

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "PREPROCESSING COMPLETE!"
    echo "=========================================="
    echo "End time: $(date)"
    
    # Check output
    if [ -d "fulsang_preprocessed/tfrecords" ]; then
        TF_COUNT=$(find fulsang_preprocessed/tfrecords -name "*.tfrecords" 2>/dev/null | wc -l)
        echo "Created $TF_COUNT TFRecord file(s)"
        echo "Output directory: fulsang_preprocessed/tfrecords"
        
        # Show disk usage
        echo "Disk usage:"
        du -sh fulsang_preprocessed/
    else
        echo "WARNING: Output directory not found!"
    fi
else
    echo "=========================================="
    echo "PREPROCESSING FAILED!"
    echo "=========================================="
    echo "Exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "FULSANG preprocessing job completed!"

