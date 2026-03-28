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
# - Aggressively optimized hyperparameters: cca_dims=20, regularization=0.08, window_size=1920 (30s)
# - Optimized lag range: 150-400ms (speech tracking strongest range)
# - Optimized filter band: 1-8 Hz (delta-theta, low frequencies dominate envelope tracking)
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

# Check if TFRecord files exist (FULPRE.py format: fulsang_*.tfrecords)
tfrecord_files=$(find fulsang_preprocessed/tfrecords -name "fulsang_*.tfrecords" 2>/dev/null | wc -l)

if [ ! -d "fulsang_preprocessed/tfrecords" ] || [ "$tfrecord_files" -eq 0 ]; then
    echo "⚠ Fulsang preprocessing not found or incomplete!"
    echo "  Expected directory: fulsang_preprocessed/tfrecords"
    echo "  Expected files: fulsang_*.tfrecords (FULPRE.py format)"
    echo ""
    echo "  Running FULPRE.py automatically..."
    echo ""
    
    # Check for Data/Fulsang directory
    if [ ! -d "Data/Fulsang" ]; then
        echo "  ✗ ERROR: Could not find Data/Fulsang directory!"
        echo "  Please ensure the following exists:"
        echo "    - Data/Fulsang/DATA_preproc/ (containing S*_data_preproc.mat files)"
        exit 1
    fi
    
    if [ ! -d "Data/Fulsang/DATA_preproc" ]; then
        echo "  ✗ ERROR: Could not find Data/Fulsang/DATA_preproc directory!"
        echo "  FULPRE.py requires MATLAB preprocessed files in Data/Fulsang/DATA_preproc/"
        exit 1
    fi
    
    echo "  Found Data/Fulsang/DATA_preproc directory"
    echo "  Running FULPRE.py to create trial-level TFRecords..."
    
            # Run preprocessing using FULPRE.py
            # Explicitly specify EEG directory if it exists
            if [ -d "Data/Fulsang/EEG" ]; then
                python FULPRE.py \
                    --data_dir "Data/Fulsang" \
                    --output_dir "fulsang_preprocessed" \
                    --eeg_raw_dir "Data/Fulsang/EEG"
            elif [ -f "Data/Fulsang/EEG.zip" ]; then
                python FULPRE.py \
                    --data_dir "Data/Fulsang" \
                    --output_dir "fulsang_preprocessed" \
                    --eeg_raw_dir "Data/Fulsang/EEG.zip"
            else
                python FULPRE.py \
                    --data_dir "Data/Fulsang" \
        --output_dir "fulsang_preprocessed"
            fi
    
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
    
    # Check for FULPRE.py format files (fulsang_*.tfrecords)
    tfrecord_count=$(find fulsang_preprocessed/tfrecords -name "fulsang_*.tfrecords" 2>/dev/null | wc -l)
    if [ "$tfrecord_count" -eq 0 ]; then
        echo "  ✗ ERROR: Preprocessing completed but no FULPRE.py format TFRecord files were created!"
        echo "  Expected files: fulsang_*.tfrecords"
        echo "  Found files:"
        find fulsang_preprocessed/tfrecords -name "*.tfrecords" 2>/dev/null | head -5
        exit 1
    fi
    
    echo "  ✓ Preprocessing completed successfully!"
    echo "  Created $tfrecord_count FULPRE.py format TFRecord file(s)"
else
    echo "✓ Fulsang preprocessing found"
    echo "  Using existing TFRecord files from: fulsang_preprocessed/tfrecords"
    
    # Count FULPRE.py format TFRecord files (fulsang_*.tfrecords)
    tfrecord_count=$(find fulsang_preprocessed/tfrecords -name "fulsang_*.tfrecords" 2>/dev/null | wc -l)
    echo "  Found $tfrecord_count FULPRE.py format TFRecord file(s)"
    
    if [ "$tfrecord_count" -eq 0 ]; then
        echo "  ⚠ WARNING: No FULPRE.py format TFRecord files found!"
        echo "  Expected files: fulsang_*.tfrecords"
        echo "  Re-running FULPRE.py preprocessing..."
        
        # Check for Data/Fulsang directory
        if [ ! -d "Data/Fulsang/DATA_preproc" ]; then
            echo "  ✗ ERROR: Could not find Data/Fulsang/DATA_preproc directory to re-run preprocessing"
            exit 1
        fi
        
                echo "  Running FULPRE.py..."
                # Explicitly specify EEG directory if it exists
                if [ -d "Data/Fulsang/EEG" ]; then
                    python FULPRE.py \
                        --data_dir "Data/Fulsang" \
                        --output_dir "fulsang_preprocessed" \
                        --eeg_raw_dir "Data/Fulsang/EEG"
                elif [ -f "Data/Fulsang/EEG.zip" ]; then
                    python FULPRE.py \
                        --data_dir "Data/Fulsang" \
                        --output_dir "fulsang_preprocessed" \
                        --eeg_raw_dir "Data/Fulsang/EEG.zip"
                else
                    python FULPRE.py \
                        --data_dir "Data/Fulsang" \
                --output_dir "fulsang_preprocessed"
                fi
        
        # Verify again
        tfrecord_count=$(find fulsang_preprocessed/tfrecords -name "fulsang_*.tfrecords" 2>/dev/null | wc -l)
        if [ "$tfrecord_count" -eq 0 ]; then
            echo "  ✗ ERROR: Preprocessing failed - no TFRecord files created"
            exit 1
        fi
        echo "  ✓ Created $tfrecord_count TFRecord file(s)"
    fi
fi

# Run FULCCA with optimal Fulsang configuration
echo "=========================================="
echo "Running FULCCA Analysis"
echo "=========================================="

# Configuration: AGGRESSIVELY OPTIMIZED Fulsang settings
# - Increased CCA dimensions: 12 → 20 (max 30)
# - Optimized lag range: 150-400ms (strongest speech tracking range)
# - Optimized filter band: 1-8 Hz (delta-theta, low frequencies dominate)
# - Window size: 1920 (30s, best from temporal analysis)
echo "Running Aggressively Optimized Fulsang CCA Configuration..."
python FULCCA.py \
    --tfrecord_dir fulsang_preprocessed/tfrecords \
    --batch_size 6 \
    --cca_dims 25 \
    --regularization 0.08 \
    --window_size 1920 \
    --min_lag_ms 150.0 \
    --max_lag_ms 400.0 \
    --eeg_low_freq 1.0 \
    --eeg_high_freq 8.0 \
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
