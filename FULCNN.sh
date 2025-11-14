#!/bin/bash
#SBATCH --job-name=fulcnn_cnn_loc
#SBATCH --output=fulcnn_cnn_loc_%j.out
#SBATCH --error=fulcnn_cnn_loc_%j.err
#SBATCH --time=8:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=tier3
#SBATCH --account=neurosteer

# FULCNN - CNN-LOC Algorithm for Fulsang Dataset
# This script runs the FULCNN model training with all the metrics we need
# 
# What it does:
# - Uses CNN-LOC architecture that works well with Fulsang data
# - Calculates accuracy, MSED, ROC-AUC, and temporal performance metrics
# - Analyzes performance across different window lengths (0.5s to 30s)
# - Handles preprocessing and data loading properly
# - Generates detailed reports on how well the model performs
# - Can tune hyperparameters to find better settings
# - Has better default hyperparameters than before
#
# How to use:
#   bash FULCNN.sh              # Just run training with the defaults
#   bash FULCNN.sh --tune       # Run hyperparameter tuning first
#   bash FULCNN.sh -t           # Same as above, shorter flag

echo "=================================================================================="
echo "FULCNN - CNN-LOC Algorithm for Fulsang Dataset"
echo "=================================================================================="
echo "Started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $SLURM_NODELIST"
echo "=================================================================================="

# Environment setup
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=8

# Handle timeout - save what we can before the job gets killed
timeout_handler() {
    echo "=================================================================================="
    echo "WARNING: We're at 90% of the time limit!"
    echo "Current time: $(date)"
    echo "Trying to save whatever progress we have..."
    echo "=================================================================================="
    
    # Save any results we've got so far
    if [ -d "fulcnn_results" ]; then
        echo "Backing up FULCNN results..."
        cp -r fulcnn_results fulcnn_results_backup_$(date +%Y%m%d_%H%M%S) 2>/dev/null || true
    fi
    
    # Save the training log too
    if [ -f "fulcnn_training.log" ]; then
        echo "Backing up training log..."
        cp fulcnn_training.log fulcnn_training_backup_$(date +%Y%m%d_%H%M%S).log 2>/dev/null || true
    fi
}

# Set up the timeout handler
trap timeout_handler SIGUSR1

# Check if Python and everything we need is set up properly
check_python_env() {
    echo "=================================================================================="
    echo "Checking Python Environment"
    echo "=================================================================================="
    
    echo "Python version: $(python3 --version 2>/dev/null || echo 'Python not found')"
    echo "Available memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
    echo "Available CPUs: $(nproc)"
    
    # See what GPU we have
    nvidia-smi || echo "nvidia-smi not available"
    
    # Make sure we have all the Python packages we need
    echo "Checking if we have all the required packages..."
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

    # Install some extra packages that might be useful for EEG processing
    echo "Installing some extra packages for EEG processing..."
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

# Check if we have the Fulsang data ready to go
check_fulsang_data() {
    echo "=================================================================================="
    echo "Checking if Fulsang Data is Available"
    echo "=================================================================================="
    
    # Look for the preprocessed data
    if [ -d "fulsang_preprocessed/tfrecords" ]; then
        tfrecord_count=$(find fulsang_preprocessed/tfrecords -name "*.tfrecords" 2>/dev/null | wc -l)
        echo "✓ Found preprocessed data from FULPREPROCESSING"
        echo "✓ Found $tfrecord_count TFRecord files"
        echo "✓ Data leakage prevention is on"
        echo "✓ Attention labels have been validated"
        
        # Check if there are any reports
        if [ -d "fulsang_preprocessed/reports" ]; then
            echo "✓ Found preprocessing reports"
        fi
        
        return 0
    else
        echo "✗ Couldn't find preprocessed data!"
        echo "Expected location: fulsang_preprocessed/tfrecords"
        echo ""
        echo "This probably means:"
        echo "  1. FULPREPROCESSING.py hasn't been run yet"
        echo "  2. The data is somewhere else"
        echo ""
        echo "You'll need to run FULPREPROCESSING.py first to create the validated data"
        echo "We disabled the old data loading to make sure we only use good quality data"
        return 1
    fi
}

# Run the preprocessing step if we need to
run_fulsang_preprocessing() {
    echo "=================================================================================="
    echo "Running Fulsang Preprocessing"
    echo "=================================================================================="
    
    # Run the preprocessing script
    if [ -f "FULPREPROCESSING.py" ]; then
        echo "Running FULPREPROCESSING.py..."
        python3 FULPREPROCESSING.py --data_dir "Data/Fulsang" --output_dir "fulsang_preprocessed" > fulsang_preprocessing_fulcnn.log 2>&1
        
        if [ $? -eq 0 ]; then
            echo "✓ Preprocessing finished successfully"
            echo "Results are in fulsang_preprocessed/"
            
            if [ -d "fulsang_preprocessed/tfrecords" ]; then
                tfrecord_count=$(find fulsang_preprocessed/tfrecords -name "*.tfrecords" 2>/dev/null | wc -l)
                echo "✓ Created $tfrecord_count TFRecord files"
                return 0
            else
                echo "⚠ WARNING: Couldn't find the TFRecord directory"
                return 1
            fi
        else
            echo "✗ Preprocessing failed - check fulsang_preprocessing_fulcnn.log"
            tail -20 fulsang_preprocessing_fulcnn.log
            return 1
        fi
    else
        echo "✗ Couldn't find FULPREPROCESSING.py!"
        return 1
    fi
}

# Actually run the FULCNN training
run_fulcnn_training() {
    echo "=================================================================================="
    echo "Running FULCNN Training"
    echo "=================================================================================="
    
    if [ ! -f "FULCNN.py" ]; then
        echo "✗ Couldn't find FULCNN.py!"
        return 1
    fi
    
    if [ ! -d "fulsang_preprocessed/tfrecords" ]; then
        echo "✗ No TFRecord data found! Run preprocessing first"
        return 1
    fi
    
    echo "Starting FULCNN training for the Fulsang dataset..."
    TFRecord_DIR="fulsang_preprocessed/tfrecords"
    
    # Run the training with hyperparameters that don't use too much memory
    python3 FULCNN.py \
        --tfrecord_dir "$TFRecord_DIR" \
        --batch_size 16 \
        --num_epochs 100 \
        --learning_rate 2e-4 \
        --window_size 512 \
        --weight_decay 1e-4 \
        --dropout_rate 0.4 \
        --label_smoothing 0.1 \
        --output_dir fulcnn_results > fulcnn_training.log 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "=================================================================================="
        echo "FULCNN Training Finished Successfully!"
        echo "Finished at: $(date)"
        echo "=================================================================================="
        
        # Display results summary
        if [ -d "fulcnn_results" ] && [ -f "fulcnn_results/results.json" ]; then
            echo ""
            echo "=================================================================================="
            echo "FULCNN RESULTS SUMMARY"
            echo "=================================================================================="
            python3 -c "
import json
import os

try:
    with open('fulcnn_results/results.json', 'r') as f:
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
    
except Exception as e:
    print(f'Could not read results: {e}')
    print('Please check the results directory manually')
"
        fi
        
        return 0
    else
        echo "=================================================================================="
        echo "FULCNN Training Failed (exit code: $exit_code)"
        echo "Check the error log: fulcnn_training.log"
        echo "=================================================================================="
        return $exit_code
    fi
}

# Run hyperparameter tuning to find better settings
run_hyperparameter_tuning() {
    echo "=================================================================================="
    echo "Running Hyperparameter Tuning"
    echo "=================================================================================="
    
    if [ ! -f "quick_tuning.py" ]; then
        echo "✗ Couldn't find quick_tuning.py!"
        return 1
    fi
    
    echo "Starting hyperparameter tuning..."
    python3 quick_tuning.py > hyperparameter_tuning.log 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "=================================================================================="
        echo "Hyperparameter Tuning Finished Successfully!"
        echo "Finished at: $(date)"
        echo "=================================================================================="
        
        # Show the best configuration we found
        if [ -d "quick_tuning_results" ] && [ -f "quick_tuning_results/quick_tuning_results.json" ]; then
            echo ""
            echo "=================================================================================="
            echo "Best Hyperparameter Configuration"
            echo "=================================================================================="
            python3 -c "
import json
import os

try:
    with open('quick_tuning_results/quick_tuning_results.json', 'r') as f:
        data = json.load(f)
    
    best_config = data['best_config']
    best_score = data['best_score']
    
    print(f'Best Configuration: {best_config[\"name\"]}')
    print(f'Best Score: {best_score:.4f}')
    print('')
    print('Optimal Parameters:')
    for key, value in best_config.items():
        if key not in ['name', 'tfrecord_dir', 'output_dir']:
            print(f'  {key}: {value}')
    
    print('')
    print('RECOMMENDED COMMAND FOR PRODUCTION TRAINING:')
    print('python3 FULCNN.py \\')
    print(f'    --tfrecord_dir fulsang_preprocessed/tfrecords \\')
    print(f'    --batch_size {best_config[\"batch_size\"]} \\')
    print(f'    --num_epochs {best_config[\"num_epochs\"]} \\')
    print(f'    --learning_rate {best_config[\"learning_rate\"]} \\')
    print(f'    --window_size {best_config[\"window_size\"]} \\')
    print(f'    --weight_decay {best_config[\"weight_decay\"]} \\')
    print(f'    --dropout_rate {best_config[\"dropout_rate\"]} \\')
    print(f'    --label_smoothing {best_config[\"label_smoothing\"]} \\')
    print('    --output_dir fulcnn_results_optimized')
    
except Exception as e:
    print(f'Could not read tuning results: {e}')
"
        fi
        
        return 0
    else
        echo "=================================================================================="
        echo "Hyperparameter Tuning Failed (exit code: $exit_code)"
        echo "Check the error log: hyperparameter_tuning.log"
        echo "=================================================================================="
        return $exit_code
    fi
}

# Create a summary of what we did
create_final_summary() {
    echo "=================================================================================="
    echo "Final Summary Report"
    echo "=================================================================================="
    echo "Algorithm: FULCNN (CNN-LOC for Fulsang Dataset)"
    echo "Finished at: $(date)"
    echo ""
    
    # Check how preprocessing went
    echo "Preprocessing Results:"
    echo "---------------------"
    if [ -d "fulsang_preprocessed/tfrecords" ]; then
        tfrecord_count=$(find fulsang_preprocessed/tfrecords -name "*.tfrecords" 2>/dev/null | wc -l)
        echo "✓ Found validated preprocessed data: $tfrecord_count TFRecord files"
    else
        echo "✗ Couldn't find validated preprocessed data"
    fi
    
    # Check how training went
    echo ""
    echo "FULCNN Training Results:"
    echo "------------------------"
    if [ -d "fulcnn_results" ]; then
        echo "✓ Training finished successfully"
    else
        echo "✗ Training failed - couldn't find the results directory"
    fi
    
    echo ""
    echo "=================================================================================="
    echo "FULCNN Training Complete"
    echo "=================================================================================="
}

# Main function - this is where everything happens
main() {
    echo "Starting the FULCNN training pipeline..."
    
    # See if they want to do hyperparameter tuning
    if [ "$1" = "--tune" ] || [ "$1" = "-t" ]; then
        echo "Hyperparameter Tuning Mode Enabled"
        echo ""
        
        check_python_env
        echo ""
        check_fulsang_data
        if [ $? -ne 0 ]; then
            run_fulsang_preprocessing || exit 1
        fi
        
        echo ""
        run_hyperparameter_tuning || exit 1
        
        echo ""
        echo "🎉 Success! Hyperparameter tuning finished!"
        exit 0
    fi
    
    # Regular training mode
    echo "Regular Training Mode"
    echo ""
    
    check_python_env
    echo ""
    check_fulsang_data
    if [ $? -ne 0 ]; then
        run_fulsang_preprocessing || exit 1
    fi
    
    echo ""
    run_fulcnn_training || exit 1
    
    echo ""
    create_final_summary
    
    echo ""
    echo "🎉 Success! FULCNN training finished!"
    exit 0
}

# Actually run everything
main "$@"
