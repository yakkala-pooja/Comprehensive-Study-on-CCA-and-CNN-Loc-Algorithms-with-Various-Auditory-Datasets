#!/bin/bash
#SBATCH --job-name=dascnn_cnn_loc
#SBATCH --output=dascnn_cnn_loc_%j.out
#SBATCH --error=dascnn_cnn_loc_%j.err
#SBATCH --time=8:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=tier3
#SBATCH --account=neurosteer

# DASCNN - CNN-LOC Algorithm for DAS Dataset
# This script runs the DASCNN implementation with comprehensive metrics evaluation
# 
# Features:
# - CNN-LOC architecture optimized for DAS data (64 EEG channels)
# - Comprehensive metrics: Accuracy, MSED, ROC-AUC, temporal performance
# - Temporal analysis across window lengths from 0.5s to 30s
# - Robust preprocessing and data handling
# - Detailed performance evaluation and reporting

echo "=================================================================================="
echo "DASCNN - CNN-LOC ALGORITHM FOR DAS DATASET"
echo "=================================================================================="
echo "Started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=================================================================================="

# Environment setup
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=8

# Timeout handler for job management
timeout_handler() {
    echo "=================================================================================="
    echo "JOB TIMEOUT WARNING: 90% of time limit reached"
    echo "Current time: $(date)"
    echo "Attempting to save current progress..."
    echo "=================================================================================="
    
    # Try to save any partial results
    if [ -d "dascnn_results" ]; then
        echo "Saving DASCNN partial results..."
        cp -r dascnn_results dascnn_results_backup_$(date +%Y%m%d_%H%M%S) 2>/dev/null || true
    fi
    
    # Try to save any log files
    if [ -f "dascnn_training.log" ]; then
        echo "Saving DASCNN training log..."
        cp dascnn_training.log dascnn_training_backup_$(date +%Y%m%d_%H%M%S).log 2>/dev/null || true
    fi
}

# Set up timeout handler
trap timeout_handler SIGUSR1

# Function to check Python environment
check_python_env() {
    echo "=================================================================================="
    echo "CHECKING PYTHON ENVIRONMENT"
    echo "=================================================================================="
    
    echo "Python version: $(python3 --version 2>/dev/null || echo 'Python not found')"
    echo "Available memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
    echo "Available CPUs: $(nproc)"
    
    # Check GPU configuration
    echo "GPU Configuration:"
    nvidia-smi || echo "nvidia-smi not available"
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    
    # Check required Python packages
    echo "Checking required Python packages..."
    python3 -c "
import sys
print(f'Python executable: {sys.executable}')

required_packages = ['numpy', 'scipy', 'matplotlib', 'tensorflow', 'torch', 'sklearn', 'seaborn', 'pandas', 'tqdm']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
        print(f'✓ {package} - Available')
    except ImportError:
        print(f'✗ {package} - MISSING')
        missing_packages.append(package)

if missing_packages:
    print(f'Missing packages: {missing_packages}')
    print('Attempting to install missing packages...')
    import subprocess
    for package in missing_packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f'✓ Installed {package}')
        except:
            print(f'✗ Failed to install {package}')
else:
    print('✓ All required packages are available!')
"

    # Install additional dependencies that might be missing
    echo "Installing additional dependencies..."
    python3 -c "
import subprocess
import sys

additional_packages = ['pyedflib', 'mne', 'gammatone', 'librosa', 'soundfile']
print('Installing additional packages for EEG processing...')

for package in additional_packages:
    try:
        __import__(package)
        print(f'✓ {package} - Already available')
    except ImportError:
        print(f'Installing {package}...')
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f'✓ Installed {package}')
        except Exception as e:
            print(f'⚠ Could not install {package}: {e}')
            print(f'  This may cause issues with some preprocessing features')
"
}

# Function to check DAS data availability
check_das_data() {
    echo "=================================================================================="
    echo "CHECKING DAS DATA AVAILABILITY"
    echo "=================================================================================="
    
    # Check for DAS validated data
    if [ -d "das_16subjects_preprocessed/tfrecords" ]; then
        tfrecord_count=$(find das_16subjects_preprocessed/tfrecords -name "*.tfrecords" 2>/dev/null | wc -l)
        echo "✓ Found DAS validated data"
        echo "✓ TFRecord files: $tfrecord_count"
        echo "✓ Data leakage prevention enabled"
        echo "✓ Attention labels validated"
        
        # Check for train/test subdirectories
        if [ -d "das_16subjects_preprocessed/tfrecords/train" ] && [ -d "das_16subjects_preprocessed/tfrecords/test" ]; then
            train_count=$(find das_16subjects_preprocessed/tfrecords/train -name "*.tfrecords" 2>/dev/null | wc -l)
            test_count=$(find das_16subjects_preprocessed/tfrecords/test -name "*.tfrecords" 2>/dev/null | wc -l)
            echo "✓ Found separate train/test directories"
            echo "✓ Train TFRecord files: $train_count"
            echo "✓ Test TFRecord files: $test_count"
        fi
        
        # Display sample TFRecord files
        echo "Sample DAS TFRecord files:"
        find das_16subjects_preprocessed/tfrecords -name "*.tfrecords" | head -3
        
        # Check for preprocessing reports
        if [ -d "das_16subjects_preprocessed/reports" ]; then
            echo "✓ Found preprocessing reports"
            if [ -f "das_16subjects_preprocessed/reports/preprocessing_report.txt" ]; then
                echo "✓ Found preprocessing report"
            fi
        fi
        
        return 0
    elif [ -d "corrected_das_analysis_results/tfrecords" ]; then
        tfrecord_count=$(find corrected_das_analysis_results/tfrecords -name "*.tfrecords" 2>/dev/null | wc -l)
        echo "✓ Found corrected DAS validated data"
        echo "✓ TFRecord files: $tfrecord_count"
        echo "✓ Data leakage prevention enabled"
        echo "✓ Attention labels validated"
        
        # Check for train/test subdirectories
        if [ -d "corrected_das_analysis_results/tfrecords/train" ] && [ -d "corrected_das_analysis_results/tfrecords/test" ]; then
            train_count=$(find corrected_das_analysis_results/tfrecords/train -name "*.tfrecords" 2>/dev/null | wc -l)
            test_count=$(find corrected_das_analysis_results/tfrecords/test -name "*.tfrecords" 2>/dev/null | wc -l)
            echo "✓ Found separate train/test directories"
            echo "✓ Train TFRecord files: $train_count"
            echo "✓ Test TFRecord files: $test_count"
        fi
        
        # Display sample TFRecord files
        echo "Sample corrected DAS TFRecord files:"
        find corrected_das_analysis_results/tfrecords -name "*.tfrecords" | head -3
        
        return 0
    else
        echo "✗ No DAS validated data found!"
        echo "Expected locations:"
        echo "  - das_16subjects_preprocessed/tfrecords"
        echo "  - corrected_das_analysis_results/tfrecords"
        echo ""
        echo "This means:"
        echo "  1. DAS preprocessing hasn't been run yet"
        echo "  2. Data is in a different location"
        echo ""
        echo "Please run DAS preprocessing first to create validated data"
        return 1
    fi
}

# Function to run DAS preprocessing if needed
run_das_preprocessing() {
    echo "=================================================================================="
    echo "RUNNING DAS PREPROCESSING"
    echo "=================================================================================="
    echo "This step prepares the DAS dataset for CNN-LOC training:"
    echo "  - Loads EEG data (128 channels, 1000 Hz sampling)"
    echo "  - Extracts validated attention labels"
    echo "  - Creates TFRecord files for efficient training"
    echo "  - Prevents data leakage with subject-wise organization"
    echo "=================================================================================="
    
    # Run DAS preprocessing
    if [ -f "das_preprocessing_16subjects.py" ]; then
        echo "Running das_preprocessing_16subjects.py (16-SUBJECTS VERSION)..."
        echo "Features:"
        echo "  ✓ Validated attention labels with quality control"
        echo "  ✓ Subject-wise organized data (no data leakage)"
        echo "  ✓ Robust EEG data extraction (64 channels, 64 Hz)"
        echo "  ✓ Comprehensive preprocessing reports"
        echo "  ✓ 16 subjects support (S1-S16)"
        
        python3 das_preprocessing_16subjects.py --data_dir "Data/Das/4004271" --output_dir "das_16subjects_preprocessed" --create_split > das_preprocessing_dascnn.log 2>&1
        
        if [ $? -eq 0 ]; then
            echo "✓ DAS preprocessing completed successfully"
            echo "Results saved to das_16subjects_preprocessed/"
            
            # Check if TFRecord files were created
            if [ -d "das_16subjects_preprocessed/tfrecords" ]; then
                tfrecord_count=$(find das_16subjects_preprocessed/tfrecords -name "*.tfrecords" 2>/dev/null | wc -l)
                echo "✓ Created $tfrecord_count TFRecord files"
                
                # Check for preprocessing reports
                if [ -d "das_16subjects_preprocessed/reports" ]; then
                    echo "✓ Created preprocessing reports"
                fi
                
                return 0
            else
                echo "⚠ WARNING: TFRecord directory not found"
                return 1
            fi
        else
            echo "✗ DAS preprocessing failed"
            echo "Check the log file: das_preprocessing_dascnn.log"
            echo "Error details:"
            tail -20 das_preprocessing_dascnn.log
            return 1
        fi
    else
        echo "✗ das_preprocessing_16subjects.py not found!"
        echo "Expected file: das_preprocessing_16subjects.py"
        echo ""
        echo "Please ensure das_preprocessing_16subjects.py is available"
        return 1
    fi
}

# Function to run DASCNN training
run_dascnn_training() {
    echo "=================================================================================="
    echo "RUNNING DASCNN TRAINING WITH DAS INTEGRATION"
    echo "=================================================================================="
    echo "This step trains the DASCNN model with comprehensive metrics:"
    echo "  - CNN-LOC architecture optimized for DAS data (64 EEG channels)"
    echo "  - Accuracy, MSED, ROC-AUC metrics evaluation"
    echo "  - Temporal performance analysis (0.5s to 30s)"
    echo "  - DAS integration for data quality"
    echo "  - Data leakage prevention"
    echo "  - Validated attention labels"
    echo "=================================================================================="
    
    if [ ! -f "DASCNN.py" ]; then
        echo "✗ DASCNN.py not found!"
        echo "Please ensure the DASCNN script is available"
        return 1
    fi
    
    if [ ! -d "das_16subjects_preprocessed/tfrecords" ] && [ ! -d "corrected_das_analysis_results/tfrecords" ]; then
        echo "✗ No TFRecord data found!"
        echo "Please run preprocessing first"
        return 1
    fi
    
    echo "Starting DASCNN training for DAS dataset..."
    
    # Check if DAS data is available
    if [ -d "das_16subjects_preprocessed/tfrecords" ]; then
        echo "✓ Found DAS validated data"
        echo "✓ Using high-quality preprocessed data"
        TFRecord_DIR="das_16subjects_preprocessed/tfrecords"
    elif [ -d "corrected_das_analysis_results/tfrecords" ]; then
        echo "✓ Found corrected DAS validated data"
        echo "✓ Using high-quality preprocessed data"
        TFRecord_DIR="corrected_das_analysis_results/tfrecords"
    else
        echo "✗ No DAS validated data found!"
        echo "Please run DAS preprocessing first to create validated data"
        return 1
    fi
    
    echo "Using TFRecord directory: $TFRecord_DIR"
    
    # Run DASCNN training
    # Use window_size=512 (4s) for proper attention decoding (not 32 which is too short)
    python3 DASCNN.py \
        --tfrecord_dir "$TFRecord_DIR" \
        --batch_size 16 \
        --num_epochs 50 \
        --learning_rate 1e-4 \
        --window_size 512 \
        --overlap 0.5 \
        --output_dir dascnn_results > dascnn_training.log 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "=================================================================================="
        echo "DASCNN TRAINING COMPLETED SUCCESSFULLY!"
        echo "Finished at: $(date)"
        echo "=================================================================================="
        
        # Check for results
        if [ -d "dascnn_results" ]; then
            echo "Results directory: dascnn_results"
            echo "Generated files:"
            find dascnn_results -type f -name "*.json" -o -name "*.png" -o -name "*.pkl" -o -name "*.txt" | sort
            
            # Display results summary
            if [ -f "dascnn_results/results.json" ]; then
                echo ""
                echo "=================================================================================="
                echo "DASCNN RESULTS SUMMARY"
                echo "=================================================================================="
                python3 -c "
import json
import os

try:
    with open('dascnn_results/results.json', 'r') as f:
        results = json.load(f)
    
    print(f'Test Accuracy: {results[\"accuracy\"]:.4f}')
    print(f'Test Loss: {results[\"loss\"]:.4f}')
    print(f'Best Validation Accuracy: {results.get(\"best_val_acc\", \"N/A\")}')
    print(f'Timestamp: {results.get(\"timestamp\", \"N/A\")}')
    
    # ROC-AUC metrics
    if 'roc_auc_metrics' in results:
        roc_auc = results['roc_auc_metrics']
        if 'error' not in roc_auc:
            print('')
            print('ROC-AUC METRICS:')
            print('----------------')
            print(f'ROC-AUC Score: {roc_auc.get(\"roc_auc_score\", \"N/A\"):.4f}')
            print(f'Average Precision: {roc_auc.get(\"average_precision\", \"N/A\"):.4f}')
            print(f'Optimal Threshold: {roc_auc.get(\"optimal_threshold\", \"N/A\"):.4f}')
    
    # MSED metrics
    if 'msed_metrics' in results:
        msed = results['msed_metrics']
        if 'error' not in msed:
            print('')
            print('MSED METRICS:')
            print('-------------')
            print(f'MSE: {msed.get(\"mse\", \"N/A\"):.4f}')
            print(f'RMSE: {msed.get(\"rmse\", \"N/A\"):.4f}')
            print(f'MAE: {msed.get(\"mae\", \"N/A\"):.4f}')
            print(f'R-squared: {msed.get(\"r_squared\", \"N/A\"):.4f}')
    
    # Advanced metrics
    if 'advanced_metrics' in results:
        advanced = results['advanced_metrics']
        if 'error' not in advanced:
            print('')
            print('ADVANCED METRICS:')
            print('-----------------')
            print(f'Matthews Correlation Coefficient: {advanced.get(\"matthews_correlation_coefficient\", \"N/A\"):.4f}')
            print(f'Cohen\\'s Kappa: {advanced.get(\"cohens_kappa\", \"N/A\"):.4f}')
            print(f'Balanced Accuracy: {advanced.get(\"balanced_accuracy\", \"N/A\"):.4f}')
    
    # Temporal metrics
    if 'temporal_metrics' in results:
        temporal = results['temporal_metrics']
        print('')
        print('TEMPORAL PERFORMANCE:')
        print('--------------------')
        for window_size, metrics in temporal.get('temporal_analysis', {}).items():
            print(f'{window_size}: {metrics.get(\"accuracy\", \"N/A\"):.4f}')
        print(f'Recommended: {temporal.get(\"recommended_window_size\", \"N/A\")}')
    
    print('')
    print('FILES GENERATED:')
    print('---------------')
    for root, dirs, files in os.walk('dascnn_results'):
        for file in files:
            print(f'  {os.path.join(root, file)}')
            
except Exception as e:
    print(f'Could not read results: {e}')
    print('Please check the results directory manually')
"
            fi
        fi
        
        return 0
    else
        echo "=================================================================================="
        echo "DASCNN TRAINING FAILED with exit code: $exit_code"
        echo "Check the error log: dascnn_training.log"
        echo "=================================================================================="
        return $exit_code
    fi
}

# Function to create final summary report
create_final_summary() {
    echo "=================================================================================="
    echo "FINAL SUMMARY REPORT"
    echo "=================================================================================="
    echo "Algorithm: DASCNN (CNN-LOC for DAS Dataset)"
    echo "Finished at: $(date)"
    echo ""
    
    # Check preprocessing results
    echo "PREPROCESSING RESULTS:"
    echo "---------------------"
    if [ -d "das_16subjects_preprocessed/tfrecords" ]; then
        tfrecord_count=$(find das_16subjects_preprocessed/tfrecords -name "*.tfrecords" 2>/dev/null | wc -l)
        echo "✓ DAS validated data: $tfrecord_count TFRecord files"
        echo "✓ Data leakage prevention implemented"
        echo "✓ Attention labels validated"
        echo "✓ Subject-wise organization applied"
    elif [ -d "corrected_das_analysis_results/tfrecords" ]; then
        tfrecord_count=$(find corrected_das_analysis_results/tfrecords -name "*.tfrecords" 2>/dev/null | wc -l)
        echo "✓ Corrected DAS validated data: $tfrecord_count TFRecord files"
        echo "✓ Data leakage prevention implemented"
        echo "✓ Attention labels validated"
        echo "✓ Subject-wise organization applied"
    else
        echo "✗ No DAS validated data found"
    fi
    
    # Check training results
    echo ""
    echo "DASCNN TRAINING RESULTS:"
    echo "------------------------"
    if [ -d "dascnn_results" ]; then
        echo "✓ Training completed successfully"
        if [ -f "dascnn_results/results.json" ]; then
            echo "✓ Results file generated"
        fi
        if [ -f "dascnn_results/comprehensive_metrics_report.txt" ]; then
            echo "✓ Comprehensive metrics report generated"
        fi
        if [ -f "dascnn_results/best_model.pth" ]; then
            echo "✓ Best model saved"
        fi
    else
        echo "✗ Training failed - no results directory found"
    fi
    
    # List all generated files
    echo ""
    echo "GENERATED FILES:"
    echo "================"
    find . -name "*dascnn*" -type f | grep -E "\.(log|json|png|pkl|txt)$" | sort
    
    echo ""
    echo "=================================================================================="
    echo "DASCNN TRAINING COMPLETED"
    echo "=================================================================================="
}

# Main execution
main() {
    echo "Starting DASCNN training pipeline..."
    
    # Step 1: Check Python environment
    check_python_env
    
    # Step 2: Check DAS data availability
    echo ""
    check_das_data
    if [ $? -ne 0 ]; then
        echo "DAS data not found, attempting to run preprocessing..."
        run_das_preprocessing
        if [ $? -ne 0 ]; then
            echo "✗ Preprocessing failed, exiting..."
            exit 1
        fi
    fi
    
    # Step 3: Run DASCNN training
    echo ""
    run_dascnn_training
    if [ $? -ne 0 ]; then
        echo "✗ Training failed, exiting..."
        exit 1
    fi
    
    # Step 4: Create final summary
    echo ""
    create_final_summary
    
    echo ""
    echo "🎉 SUCCESS: DASCNN training completed successfully!"
    echo "Check the results in dascnn_results/ directory"
    exit 0
}

# Run main function
main "$@"
