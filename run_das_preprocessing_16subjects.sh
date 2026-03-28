#!/bin/bash
#SBATCH --job-name=das_preprocessing_16subjects
#SBATCH --partition=tier3
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=das_preprocessing_16subjects_%j.log
#SBATCH --error=das_preprocessing_16subjects_%j.err

# ==================================================================================
# DAS PREPROCESSING WITH 16 SUBJECTS SUPPORT
# ==================================================================================
# This script runs the DAS preprocessing pipeline for all 16 subjects
# ==================================================================================

echo "=================================================================================="
echo "DAS PREPROCESSING WITH 16 SUBJECTS SUPPORT"
echo "=================================================================================="
echo "Features:"
echo "- Processes all 16 DAS subjects (S1-S16)"
echo "- Downsampling from 1000 Hz to 64 Hz"
echo "- Bandpass filtering (1-40 Hz)"
echo "- Subject-wise train/val/test splitting"
echo "- Comprehensive preprocessing reports"
echo "=================================================================================="

# Set up environment
echo "Setting up environment..."
module load python/3.9

# Create virtual environment if it doesn't exist
if [ ! -d "venv_preprocessing" ]; then
    echo "Creating preprocessing virtual environment..."
    python -m venv venv_preprocessing
fi

# Activate virtual environment
source venv_preprocessing/bin/activate

# Install required packages
echo "Installing required packages..."
pip install --upgrade pip
pip install tensorflow==2.12.0
pip install scipy numpy pandas
pip install tqdm

# Check if preprocessing script exists
if [ ! -f "das_preprocessing_16subjects.py" ]; then
    echo "✗ das_preprocessing_16subjects.py not found!"
    echo "Please ensure the preprocessing script is available"
    exit 1
fi

# Check if data directory exists
if [ ! -d "Data/Das/4004271" ]; then
    echo "✗ Data directory not found!"
    echo "Expected: Data/Das/4004271"
    echo "Please ensure the DAS data is available"
    exit 1
fi

# Run preprocessing
echo "Running DAS preprocessing for 16 subjects..."
python das_preprocessing_16subjects.py \
    --data_dir "Data/Das/4004271" \
    --output_dir "das_16subjects_preprocessed" \
    --create_split \
    > das_preprocessing_16subjects.log 2>&1

# Check if preprocessing was successful
if [ $? -eq 0 ]; then
    echo "=================================================================================="
    echo "✓ DAS PREPROCESSING COMPLETED SUCCESSFULLY"
    echo "=================================================================================="
    
    # Check results
    if [ -d "das_16subjects_preprocessed" ]; then
        echo "Results saved to: das_16subjects_preprocessed/"
        
        # Count TFRecord files
        tfrecord_count=$(find das_16subjects_preprocessed/tfrecords -name "*.tfrecords" 2>/dev/null | wc -l)
        echo "✓ TFRecord files created: $tfrecord_count"
        
        # Check for train/test splits
        if [ -d "das_16subjects_preprocessed/tfrecords/train" ] && [ -d "das_16subjects_preprocessed/tfrecords/test" ]; then
            train_count=$(find das_16subjects_preprocessed/tfrecords/train -name "*.tfrecords" 2>/dev/null | wc -l)
            test_count=$(find das_16subjects_preprocessed/tfrecords/test -name "*.tfrecords" 2>/dev/null | wc -l)
            echo "✓ Train TFRecord files: $train_count"
            echo "✓ Test TFRecord files: $test_count"
            
            # Check for validation split
            if [ -d "das_16subjects_preprocessed/tfrecords/val" ]; then
                val_count=$(find das_16subjects_preprocessed/tfrecords/val -name "*.tfrecords" 2>/dev/null | wc -l)
                echo "✓ Val TFRecord files: $val_count"
            fi
        fi
        
        # Show preprocessing summary
        if [ -f "das_16subjects_preprocessed/preprocessing_summary.json" ]; then
            echo ""
            echo "Preprocessing Summary:"
            python -c "
import json
with open('das_16subjects_preprocessed/preprocessing_summary.json', 'r') as f:
    summary = json.load(f)
print(f'Total subjects: {summary[\"total_subjects\"]}')
print(f'Total trials: {summary[\"total_trials\"]}')
print(f'Original sampling rate: {summary[\"preprocessing_info\"][\"original_sampling_rate\"]} Hz')
print(f'Target sampling rate: {summary[\"preprocessing_info\"][\"target_sampling_rate\"]} Hz')
print(f'EEG channels: {summary[\"preprocessing_info\"][\"n_channels\"]}')
"
        fi
        
        echo ""
        echo "Sample TFRecord files:"
        find das_16subjects_preprocessed/tfrecords -name "*.tfrecords" | head -5 | while read file; do
            echo "  $file"
        done
        
    else
        echo "✗ Output directory not created"
    fi
    
    echo ""
    echo "Preprocessing log saved to: das_preprocessing_16subjects.log"
    echo "=================================================================================="
    
else
    echo "=================================================================================="
    echo "✗ DAS PREPROCESSING FAILED"
    echo "=================================================================================="
    echo "Check the error log: das_preprocessing_16subjects.log"
    echo "=================================================================================="
    exit 1
fi
