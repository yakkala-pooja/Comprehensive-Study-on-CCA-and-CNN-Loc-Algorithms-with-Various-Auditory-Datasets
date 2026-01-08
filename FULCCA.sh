#!/bin/bash
#SBATCH --job-name=fulcca_cca
#SBATCH --output=fulcca_cca_%j.out
#SBATCH --error=fulcca_cca_%j.err
#SBATCH --time=24:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=tier3
#SBATCH --account=neurosteer

# FULCCA - Canonical Correlation Analysis Algorithm for Fulsang Dataset
# This script runs the FULCCA implementation with comprehensive metrics evaluation
# 
# Features:
# - Uses Fulsang preprocessing data (66 EEG channels)
# - EEG + Audio envelope correlation (improved CCA performance)
# - Optimal hyperparameters: cca_dims=12, regularization=0.08, window_size=1280 (20s)
# - Automatic audio file mapping and envelope extraction
# - Comprehensive metrics evaluation

echo "=========================================="
echo "FULCCA - CCA Algorithm for Fulsang Dataset"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=========================================="

# Load necessary modules
echo "Loading modules..."
module load python/3.8
module load cuda/11.2
module load gcc/9.3.0

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d "env" ]; then
    echo "Activating environment..."
    source env/bin/activate
fi

# Check Python version and packages
echo "Python version: $(python --version)"
echo "TensorFlow version: $(python -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null || echo 'Not installed')"
echo "NumPy version: $(python -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo 'Not installed')"

# Set environment variables
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create output directory
mkdir -p fulcca_results

# Check for existing Fulsang preprocessing and run if needed
echo "=========================================="
echo "Checking for Existing Fulsang Preprocessing"
echo "=========================================="

if [ ! -d "fulsang_preprocessed/tfrecords" ]; then
    echo "⚠ Fulsang preprocessing not found!"
    echo "  Expected directory: fulsang_preprocessed/tfrecords"
    echo ""
    echo "  Running preprocessing automatically..."
    echo ""
    
    # Check for DATA_preproc directory or zip file
    DATA_PREPROC_PATH=""
    if [ -d "Data/Fulsang/DATA_preproc" ]; then
        DATA_PREPROC_PATH="Data/Fulsang/DATA_preproc"
        echo "  Found DATA_preproc directory: $DATA_PREPROC_PATH"
    elif [ -f "Data/Fulsang/DATA_preproc.zip" ]; then
        DATA_PREPROC_PATH="Data/Fulsang/DATA_preproc.zip"
        echo "  Found DATA_preproc.zip: $DATA_PREPROC_PATH"
    elif [ -d "/home/py9363/telluride_decoding/Data/Fulsang/DATA_preproc" ]; then
        DATA_PREPROC_PATH="/home/py9363/telluride_decoding/Data/Fulsang/DATA_preproc"
        echo "  Found DATA_preproc directory: $DATA_PREPROC_PATH"
    elif [ -f "/home/py9363/telluride_decoding/Data/Fulsang/DATA_preproc.zip" ]; then
        DATA_PREPROC_PATH="/home/py9363/telluride_decoding/Data/Fulsang/DATA_preproc.zip"
        echo "  Found DATA_preproc.zip: $DATA_PREPROC_PATH"
    else
        echo "  ✗ ERROR: Could not find DATA_preproc directory or zip file!"
        echo "  Please ensure one of the following exists:"
        echo "    - Data/Fulsang/DATA_preproc/"
        echo "    - Data/Fulsang/DATA_preproc.zip"
        echo "    - /home/py9363/telluride_decoding/Data/Fulsang/DATA_preproc/"
        echo "    - /home/py9363/telluride_decoding/Data/Fulsang/DATA_preproc.zip"
        exit 1
    fi
    
    # Run preprocessing
    echo "  Running FULSANGPREPROCESSING.py..."
    python FULSANGPREPROCESSING.py \
        --data_path "$DATA_PREPROC_PATH" \
        --output_dir "fulsang_preprocessed"
    
    PREPROC_EXIT_CODE=$?
    
    if [ $PREPROC_EXIT_CODE -ne 0 ]; then
        echo "  ✗ ERROR: Preprocessing failed with exit code: $PREPROC_EXIT_CODE"
        exit 1
    fi
    
    # Verify preprocessing was successful
    if [ ! -d "fulsang_preprocessed/tfrecords" ]; then
        echo "  ✗ ERROR: Preprocessing completed but tfrecords directory not found!"
        exit 1
    fi
    
    tfrecord_count=$(find fulsang_preprocessed/tfrecords -name "*.tfrecords" 2>/dev/null | wc -l)
    if [ "$tfrecord_count" -eq 0 ]; then
        echo "  ✗ ERROR: Preprocessing completed but no TFRecord files were created!"
        exit 1
    fi
    
    echo "  ✓ Preprocessing completed successfully!"
    echo "  Created $tfrecord_count TFRecord file(s)"
else
    echo "✓ Fulsang preprocessing found"
    echo "  Using existing TFRecord files from: fulsang_preprocessed/tfrecords"
    
    # Count TFRecord files
    tfrecord_count=$(find fulsang_preprocessed/tfrecords -name "*.tfrecords" 2>/dev/null | wc -l)
    echo "  Found $tfrecord_count TFRecord file(s)"
    
    if [ "$tfrecord_count" -eq 0 ]; then
        echo "  ⚠ WARNING: No TFRecord files found in the directory!"
        echo "  Re-running preprocessing..."
        
        # Try to find and run preprocessing
        DATA_PREPROC_PATH=""
        if [ -d "Data/Fulsang/DATA_preproc" ]; then
            DATA_PREPROC_PATH="Data/Fulsang/DATA_preproc"
        elif [ -f "Data/Fulsang/DATA_preproc.zip" ]; then
            DATA_PREPROC_PATH="Data/Fulsang/DATA_preproc.zip"
        elif [ -d "/home/py9363/telluride_decoding/Data/Fulsang/DATA_preproc" ]; then
            DATA_PREPROC_PATH="/home/py9363/telluride_decoding/Data/Fulsang/DATA_preproc"
        elif [ -f "/home/py9363/telluride_decoding/Data/Fulsang/DATA_preproc.zip" ]; then
            DATA_PREPROC_PATH="/home/py9363/telluride_decoding/Data/Fulsang/DATA_preproc.zip"
        fi
        
        if [ -n "$DATA_PREPROC_PATH" ]; then
            python FULSANGPREPROCESSING.py \
                --data_path "$DATA_PREPROC_PATH" \
                --output_dir "fulsang_preprocessed"
        else
            echo "  ✗ ERROR: Could not find DATA_preproc to re-run preprocessing"
            exit 1
        fi
    fi
fi

# Run FULCCA with optimal Fulsang configuration
echo "=========================================="
echo "Running FULCCA Analysis"
echo "=========================================="

# Configuration: Optimal Fulsang settings (from Optimal_FULCCA.py)
echo "Running Optimal Fulsang CCA Configuration..."
python FULCCA.py \
    --tfrecord_dir fulsang_preprocessed/tfrecords \
    --batch_size 6 \
    --cca_dims 12 \
    --regularization 0.08 \
    --window_size 1280 \
    --output_dir fulcca_results/optimal_fulcca \
    --load_audio \
    --max_files 100

# Generate summary report
echo "=========================================="
echo "Generating Summary Report"
echo "=========================================="

python -c "
import json
import os
from pathlib import Path

# Collect results
results_summary = []
config_dir = 'fulcca_results/optimal_fulcca'

if os.path.exists(f'{config_dir}/results.json'):
    with open(f'{config_dir}/results.json', 'r') as f:
        results = json.load(f)
    
    print('FULCCA Results Summary:')
    print('=' * 50)
    print(f'Accuracy: {results.get(\"accuracy\", \"N/A\")}')
    print(f'ROC-AUC: {results.get(\"roc_auc_metrics\", {}).get(\"roc_auc_score\", \"N/A\")}')
    print(f'MCC: {results.get(\"advanced_metrics\", {}).get(\"matthews_correlation_coefficient\", \"N/A\")}')
    print(f'Balanced Accuracy: {results.get(\"advanced_metrics\", {}).get(\"balanced_accuracy\", \"N/A\")}')
    print('=' * 50)
else:
    print('No results found to summarize')
"

# Clean up temporary files
echo "Cleaning up temporary files..."
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Display final results
echo "=========================================="
echo "FULCCA Analysis Complete"
echo "=========================================="
echo "End time: $(date)"
echo "Job duration: $SECONDS seconds"
echo "Results saved to: fulcca_results/"
echo "=========================================="

# Display disk usage
echo "Disk usage:"
du -sh fulcca_results/ 2>/dev/null || echo "Results directory not found"

# Display GPU memory usage if available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU memory usage:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
fi

echo "FULCCA job completed successfully!"
