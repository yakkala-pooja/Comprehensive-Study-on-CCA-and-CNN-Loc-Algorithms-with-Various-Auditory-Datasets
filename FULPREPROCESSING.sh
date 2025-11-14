#!/bin/bash
#SBATCH --job-name=fulpreprocessing
#SBATCH --output=fulpreprocessing_%j.out
#SBATCH --error=fulpreprocessing_%j.err
#SBATCH --time=4:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=tier3
#SBATCH --account=neurosteer

# FULPREPROCESSING - Accurate Fulsang Dataset Preprocessing
# This script runs comprehensive preprocessing with quality control
# 
# Features:
# - Comprehensive attention label validation
# - Data leakage prevention
# - Robust MATLAB file extraction
# - Quality control and error detection
# - Subject-wise data organization


# Environment setup
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=8

# Timeout handler for job management
timeout_handler() {
    echo "WARNING: 90% of time limit reached, saving progress..."
    if [ -d "fulsang_preprocessed" ]; then
        cp -r fulsang_preprocessed fulsang_preprocessed_backup_$(date +%Y%m%d_%H%M%S) 2>/dev/null || true
    fi
    if [ -f "fulpreprocessing.log" ]; then
        cp fulpreprocessing.log fulpreprocessing_backup_$(date +%Y%m%d_%H%M%S).log 2>/dev/null || true
    fi
}

# Set up timeout handler
trap timeout_handler SIGUSR1

# Function to check Python environment
check_python_env() {
    python3 -c "
import sys
required_packages = ['numpy', 'scipy', 'matplotlib', 'tensorflow', 'tqdm', 'pathlib']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f'ERROR: Missing packages: {missing_packages}')
    print('Attempting to install missing packages...')
    import subprocess
    for package in missing_packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        except:
            print(f'ERROR: Failed to install {package}')
            exit(1)
"
}

# Function to check Fulsang data availability
check_fulsang_data() {
    if [ -d "Data/Fulsang/DATA_preproc" ]; then
        mat_count=$(find Data/Fulsang/DATA_preproc -name "S*_data_preproc.mat" 2>/dev/null | wc -l)
        if [ $mat_count -eq 0 ]; then
            echo "ERROR: No MATLAB files found in Data/Fulsang/DATA_preproc"
            return 1
        fi
        return 0
    else
        echo "ERROR: Fulsang data directory not found at Data/Fulsang/DATA_preproc"
        return 1
    fi
}

# Function to run FULPREPROCESSING
run_fulpreprocessing() {
    if [ ! -f "FULPREPROCESSING.py" ]; then
        echo "ERROR: FULPREPROCESSING.py not found"
        return 1
    fi
    
    if [ ! -d "Data/Fulsang/DATA_preproc" ]; then
        echo "ERROR: Fulsang data directory not found"
        return 1
    fi
    
    python3 FULPREPROCESSING.py \
        --data_dir Data/Fulsang \
        --output_dir fulsang_preprocessed > fulpreprocessing.log 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        echo "ERROR: FULPREPROCESSING failed with exit code: $exit_code"
        echo "Check the error log: fulpreprocessing.log"
        tail -20 fulpreprocessing.log
        return $exit_code
    fi
    
    return 0
}

# Function to validate preprocessing results
validate_preprocessing_results() {
    if [ ! -d "fulsang_preprocessed/tfrecords" ]; then
        echo "ERROR: TFRecord directory not found"
        return 1
    fi
    
    tfrecord_count=$(find fulsang_preprocessed/tfrecords -name "*.tfrecords" 2>/dev/null | wc -l)
    if [ $tfrecord_count -eq 0 ]; then
        echo "ERROR: No TFRecord files found"
        return 1
    fi
    
    python3 -c "
import tensorflow as tf
from pathlib import Path

tfrecord_dir = Path('fulsang_preprocessed/tfrecords')
tfrecord_files = list(tfrecord_dir.glob('*.tfrecords'))

if not tfrecord_files:
    print('ERROR: No TFRecord files found for validation')
    exit(1)

for i, tfrecord_file in enumerate(tfrecord_files[:3]):
    try:
        dataset = tf.data.TFRecordDataset(str(tfrecord_file))
        record_count = 0
        
        for record in dataset:
            example = tf.train.Example.FromString(record.numpy())
            features = example.features.feature
            
            required_features = ['eeg', 'envelope', 'attention_label', 'subject_id']
            if not all(key in features for key in required_features):
                print(f'ERROR: Missing required features in {tfrecord_file.name}')
                exit(1)
            
            eeg_values = features['eeg'].float_list.value
            envelope_values = features['envelope'].float_list.value
            attention_label = features['attention_label'].int64_list.value[0]
            
            if len(eeg_values) != 66:
                print(f'ERROR: EEG data has {len(eeg_values)} channels, expected 66')
                exit(1)
            
            if len(envelope_values) != 1:
                print(f'ERROR: Envelope data has {len(envelope_values)} values, expected 1')
                exit(1)
            
            if attention_label not in [0, 1]:
                print(f'ERROR: Invalid attention label: {attention_label}')
                exit(1)
            
            record_count += 1
            if record_count >= 10:
                break
        
    except Exception as e:
        print(f'ERROR: Error validating {tfrecord_file.name}: {e}')
        exit(1)
"
    
    return $?
}

# Function to create final summary report
create_final_summary() {
    if [ -d "fulsang_preprocessed/tfrecords" ]; then
        tfrecord_count=$(find fulsang_preprocessed/tfrecords -name "*.tfrecords" 2>/dev/null | wc -l)
        echo "Created $tfrecord_count TFRecord files"
    fi
}

# Main execution
main() {
    check_python_env || exit 1
    check_fulsang_data || exit 1
    run_fulpreprocessing || exit 1
    validate_preprocessing_results
    create_final_summary
    exit 0
}

# Run main function
main "$@"
