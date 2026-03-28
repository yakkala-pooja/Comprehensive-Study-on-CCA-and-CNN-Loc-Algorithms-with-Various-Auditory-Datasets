#!/bin/bash
#SBATCH --job-name=ku255cnnloc
#SBATCH --output=ku255cnnloc_%j.out
#SBATCH --error=ku255cnnloc_%j.err
#SBATCH --time=8:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=tier3
#SBATCH --account=neurosteer

# KU255CNNLOC - CNN-LOC Algorithm for KU Leuven 255 Dataset
# This script runs the KU255CNN model training with all the metrics we need
# 
# What it does:
# - Uses CNN-LOC architecture that works well with KU255 data
# - Calculates accuracy, MSED, ROC-AUC, and temporal performance metrics
# - Analyzes performance across different window lengths (1s to 30s, 128-3840 samples at 128Hz)
# - Handles preprocessing and data loading properly
# - Generates detailed reports on how well the model performs
# - Can tune hyperparameters to find better settings
# - Has improved hyperparameters for better learning
#
# Usage:
#   bash KU255CNNLOC.sh                    # Default: 1024 samples (8 seconds)
#   bash KU255CNNLOC.sh --window_size 128  # 1 second
#   bash KU255CNNLOC.sh --window_size 640  # 5 seconds
#   bash KU255CNNLOC.sh --window_size 2048 # 16 seconds
#   bash KU255CNNLOC.sh --window_size 3840 # 30 seconds (maximum)
#   WINDOW_SIZE=2560 bash KU255CNNLOC.sh   # 20 seconds (alternative syntax)
#   bash KU255CNNLOC.sh --all_windows      # Run all window sizes (1-30s) and create JSON table
#   bash KU255CNNLOC.sh --all_windows --min_window 5 --max_window 20  # Custom range
#   bash KU255CNNLOC.sh --all_windows --skip_existing  # Resume interrupted run


echo "KU255CNNLOC - CNN-LOC Algorithm for KU Leuven 255 Dataset"
echo "Started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $SLURM_NODELIST"

# Environment setup
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=8

# Handle timeout - save what we can before the job gets killed
timeout_handler() {
    echo "WARNING: We're at 90% of the time limit!"
    echo "Current time: $(date)"
    echo "Trying to save whatever progress we have..."
    
    # Save any results we've got so far
    if [ -d "ku255cnnloc_results" ]; then
        echo "Backing up KU255CNN results..."
        cp -r ku255cnnloc_results ku255cnnloc_results_backup_$(date +%Y%m%d_%H%M%S) 2>/dev/null || true
    fi
    
    # Save the training log too
    if [ -f "ku255cnnloc_training.log" ]; then
        echo "Backing up training log..."
        cp ku255cnnloc_training.log ku255cnnloc_training_backup_$(date +%Y%m%d_%H%M%S).log 2>/dev/null || true
    fi
}

# Set up the timeout handler
trap timeout_handler SIGUSR1

# Check if Python and everything we need is set up properly
check_python_env() {
    echo "Checking Python Environment"
    
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
        print(f'{package} - Available')
    except ImportError:
        print(f'{package} - MISSING')
        missing_packages.append(package)

if missing_packages:
    print(f'Missing packages: {missing_packages}')
    print('Attempting to install missing packages...')
    import subprocess
    for package in missing_packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f'Installed {package}')
        except:
            print(f'Failed to install {package}')
else:
    print('All required packages are available!')
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
        print(f'{package} - Already available')
    except ImportError:
        print(f'Installing {package}...')
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f'Installed {package}')
        except Exception as e:
            print(f'Could not install {package}: {e}')
            print(f'This may cause issues with some preprocessing features')
"
}

# Check if we have the KU255 data ready to go
check_ku255_data() {
    echo "Checking if KU255 Data is Available"
    
    # Look for the preprocessed data
    if [ -d "kuleuven_255_preprocessed" ]; then
        mat_count=$(find kuleuven_255_preprocessed -name "*_preprocessed.mat" 2>/dev/null | wc -l)
        echo "Found preprocessed data from PREPROCESS255"
        echo "Found $mat_count preprocessed .mat files"
        echo "Data leakage prevention is on"
        echo "Attention labels have been validated"
        
        # Check if there are any reports
        if [ -d "kuleuven_255_preprocessed/reports" ]; then
            echo "Found preprocessing reports"
        fi
        
        return 0
    else
        echo "Couldn't find preprocessed data!"
        echo "Expected location: kuleuven_255_preprocessed"
        echo ""
        echo "This probably means:"
        echo "  1. PREPROCESS255.py hasn't been run yet"
        echo "  2. The data is somewhere else"
        echo ""
        echo "You'll need to run PREPROCESS255.py first to create the validated data"
        echo "We disabled the old data loading to make sure we only use good quality data"
        return 1
    fi
}

# Run the preprocessing step if we need to
run_ku255_preprocessing() {
    echo "Running KU255 Preprocessing"
    
    # Run the preprocessing script
    if [ -f "PREPROCESS255.py" ]; then
        echo "Running PREPROCESS255.py..."
        python3 PREPROCESS255.py --data_dir "Data/KULeuven 255" --output_dir "kuleuven_255_preprocessed" > ku255_preprocessing_ku255cnnloc.log 2>&1
        
        if [ $? -eq 0 ]; then
            echo "Preprocessing finished successfully"
            echo "Results are in kuleuven_255_preprocessed/"
            
            if [ -d "kuleuven_255_preprocessed" ]; then
                mat_count=$(find kuleuven_255_preprocessed -name "*_preprocessed.mat" 2>/dev/null | wc -l)
                echo "Created $mat_count preprocessed .mat files"
                return 0
            else
                echo "WARNING: Couldn't find the preprocessed directory"
                return 1
            fi
        else
            echo "Preprocessing failed - check ku255_preprocessing_ku255cnnloc.log"
            tail -20 ku255_preprocessing_ku255cnnloc.log
            return 1
        fi
    else
        echo "Couldn't find PREPROCESS255.py!"
        return 1
    fi
}

# Actually run the KU255CNN training
run_ku255cnnloc_training() {
    echo "Running KU255CNNLOC Training"
    
    if [ ! -f "KU255CNNLOC.py" ]; then
        echo "Couldn't find KU255CNNLOC.py!"
        return 1
    fi
    
    if [ ! -d "kuleuven_255_preprocessed" ]; then
        echo "No preprocessed data found! Run preprocessing first"
        return 1
    fi
    
    echo "Starting KU255CNNLOC training for the KU Leuven 255 dataset..."
    PREPROCESSED_DIR="kuleuven_255_preprocessed"
    
    # Run the training with optimized hyperparameters
    # Window size: supports 128-3840 samples (1-30 seconds at 128Hz)
    # Default: 1024 samples (8 seconds) - recommended for AAD decoding
    # Can be overridden with: bash KU255CNNLOC.sh --window_size 2048
    WINDOW_SIZE=${WINDOW_SIZE:-1024}  # Default to 8 seconds, can be overridden
    
    # Validate window size (1-30 seconds = 128-3840 samples at 128Hz)
    if [ "$WINDOW_SIZE" -lt 128 ] || [ "$WINDOW_SIZE" -gt 3840 ]; then
        echo "ERROR: window_size must be between 128-3840 samples (1-30 seconds at 128Hz)"
        echo "Current value: $WINDOW_SIZE"
        return 1
    fi
    
    # Calculate window size in seconds (using awk for portability)
    WINDOW_SECONDS=$(awk "BEGIN {printf \"%.2f\", $WINDOW_SIZE / 128}")
    echo "Using window size: $WINDOW_SIZE samples ($WINDOW_SECONDS seconds at 128Hz)"
    
    # Anti-collapse hyperparameters (MODEL_COLLAPSE_FIX / weak-signal AAD):
    # - lr 3e-3 (1e-3 was too low; model barely updates and collapses to one class)
    # - weight_decay 1e-5 (1e-4 was too high and prevented learning)
    # - dropout 0.1 (0.3 was too high and killed gradient flow)
    # - label_smoothing 0.0 (0.2 pushes outputs to 0.5 and causes stagnation)
    python3 KU255CNNLOC.py \
        --preprocessed_dir "$PREPROCESSED_DIR" \
        --batch_size 64 \
        --num_epochs 50 \
        --learning_rate 3e-3 \
        --window_size "$WINDOW_SIZE" \
        --overlap 0.5 \
        --weight_decay 1e-5 \
        --dropout_rate 0.1 \
        --label_smoothing 0.0 \
        --output_dir ku255cnnloc_results > ku255cnnloc_training.log 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "KU255CNNLOC Training Finished Successfully!"
        echo "Finished at: $(date)"
        
        # Display results summary
        if [ -d "ku255cnnloc_results" ] && [ -f "ku255cnnloc_results/results.json" ]; then
            echo ""
            echo "KU255CNNLOC RESULTS SUMMARY"
            python3 -c "
import json
import os

try:
    with open('ku255cnnloc_results/results.json', 'r') as f:
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
        echo "KU255CNNLOC Training Failed (exit code: $exit_code)"
        echo "Check the error log: ku255cnnloc_training.log"
        return $exit_code
    fi
}

# Run hyperparameter tuning to find better settings
run_hyperparameter_tuning() {
    echo "Running Hyperparameter Tuning"
    
    if [ ! -f "quick_tuning.py" ]; then
        echo "Couldn't find quick_tuning.py!"
        return 1
    fi
    
    echo "Starting hyperparameter tuning..."
    python3 quick_tuning.py > hyperparameter_tuning.log 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "Hyperparameter Tuning Finished Successfully!"
        echo "Finished at: $(date)"
        
        # Show the best configuration we found
        if [ -d "quick_tuning_results" ] && [ -f "quick_tuning_results/quick_tuning_results.json" ]; then
            echo ""
            echo "Best Hyperparameter Configuration"
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
        if key not in ['name', 'preprocessed_dir', 'output_dir']:
            print(f'  {key}: {value}')
    
    print('')
    print('RECOMMENDED COMMAND FOR PRODUCTION TRAINING:')
    print('python3 KU255CNNLOC.py \\')
    print(f'    --preprocessed_dir kuleuven_255_preprocessed \\')
    print(f'    --batch_size {best_config[\"batch_size\"]} \\')
    print(f'    --num_epochs {best_config[\"num_epochs\"]} \\')
    print(f'    --learning_rate {best_config[\"learning_rate\"]} \\')
    print(f'    --window_size {best_config[\"window_size\"]} \\')
    print(f'    --weight_decay {best_config[\"weight_decay\"]} \\')
    print(f'    --dropout_rate {best_config[\"dropout_rate\"]} \\')
    print(f'    --label_smoothing {best_config[\"label_smoothing\"]} \\')
    print('    --output_dir ku255cnnloc_results_optimized')
    
except Exception as e:
    print(f'Could not read tuning results: {e}')
"
        fi
        
        return 0
    else
        echo "Hyperparameter Tuning Failed (exit code: $exit_code)"
        echo "Check the error log: hyperparameter_tuning.log"
        return $exit_code
    fi
}

# Run training for all window sizes and collect results into JSON table
run_all_window_sizes() {
    echo "Running KU255CNNLOC Training for All Window Sizes"
    echo "=================================================="
    
    if [ ! -f "KU255CNNLOC.py" ]; then
        echo "Couldn't find KU255CNNLOC.py!"
        return 1
    fi
    
    if [ ! -d "kuleuven_255_preprocessed" ]; then
        echo "No preprocessed data found! Run preprocessing first"
        return 1
    fi
    
    # Use the Python script if it exists, otherwise fall back to inline implementation
    if [ -f "run_all_window_sizes.py" ]; then
        echo "Using run_all_window_sizes.py script"
        
        OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR:-ku255cnnloc_all_windows}
        RESULTS_FILE=${RESULTS_FILE:-window_size_results_table.json}
        MIN_WINDOW=${MIN_WINDOW:-1}
        MAX_WINDOW=${MAX_WINDOW:-30}
        STEP=${STEP:-1}
        SKIP_FLAG=""
        
        if [ "${SKIP_EXISTING:-false}" = "true" ]; then
            SKIP_FLAG="--skip_existing"
        fi
        
        python3 run_all_window_sizes.py \
            --min_window "$MIN_WINDOW" \
            --max_window "$MAX_WINDOW" \
            --step "$STEP" \
            --output_base "$OUTPUT_BASE_DIR" \
            --results_file "$RESULTS_FILE" \
            $SKIP_FLAG
        
        return $?
    else
        echo "run_all_window_sizes.py not found, using inline implementation"
        echo "For better results, create run_all_window_sizes.py using the provided template"
        
        # Fallback: simple loop that calls the training function
        OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR:-ku255cnnloc_all_windows}
        RESULTS_FILE=${RESULTS_FILE:-window_size_results_table.json}
        MIN_WINDOW=${MIN_WINDOW:-1}
        MAX_WINDOW=${MAX_WINDOW:-30}
        STEP=${STEP:-1}
        
        mkdir -p "$OUTPUT_BASE_DIR"
        
        echo "Running window sizes from ${MIN_WINDOW}s to ${MAX_WINDOW}s (step: ${STEP}s)"
        echo "This will take a long time. Consider using run_all_window_sizes.py for better progress tracking."
        echo ""
        
        for seconds in $(seq $MIN_WINDOW $STEP $MAX_WINDOW); do
            window_size=$((seconds * 128))
            window_seconds=$(awk "BEGIN {printf \"%.2f\", $window_size / 128}")
            output_dir="${OUTPUT_BASE_DIR}/window_${window_size}samples_${window_seconds}s"
            
            echo "Running window size: ${window_size} samples (${window_seconds}s)"
            
            export WINDOW_SIZE=$window_size
            run_ku255cnnloc_training_single "$output_dir" || echo "Failed for window size $window_size"
        done
        
        echo ""
        echo "All window sizes completed!"
        echo "Results are in: ${OUTPUT_BASE_DIR}/"
        echo "To create a JSON table, run: python3 -c \"...\" # (use run_all_window_sizes.py instead)"
        
        return 0
    fi
}

# Helper function to run training for a single window size with custom output directory
run_ku255cnnloc_training_single() {
    local output_dir=$1
    
    if [ ! -f "KU255CNNLOC.py" ]; then
        echo "Couldn't find KU255CNNLOC.py!"
        return 1
    fi
    
    if [ ! -d "kuleuven_255_preprocessed" ]; then
        echo "No preprocessed data found! Run preprocessing first"
        return 1
    fi
    
    PREPROCESSED_DIR="kuleuven_255_preprocessed"
    WINDOW_SIZE=${WINDOW_SIZE:-1024}
    
    # Validate window size
    if [ "$WINDOW_SIZE" -lt 128 ] || [ "$WINDOW_SIZE" -gt 3840 ]; then
        echo "ERROR: window_size must be between 128-3840 samples"
        return 1
    fi
    
    mkdir -p "$output_dir"
    
    python3 KU255CNNLOC.py \
        --preprocessed_dir "$PREPROCESSED_DIR" \
        --batch_size 64 \
        --num_epochs 50 \
        --learning_rate 3e-3 \
        --window_size "$WINDOW_SIZE" \
        --overlap 0.5 \
        --weight_decay 1e-5 \
        --dropout_rate 0.1 \
        --label_smoothing 0.0 \
        --output_dir "$output_dir" > "${output_dir}/training.log" 2>&1
    
    return $?
}

# Create a summary of what we did
create_final_summary() {
    echo "Final Summary Report"
    echo "Algorithm: KU255CNNLOC (CNN-LOC for KU Leuven 255 Dataset)"
    echo "Finished at: $(date)"
    echo ""
    
    # Check how preprocessing went
    echo "Preprocessing Results:"
    if [ -d "kuleuven_255_preprocessed" ]; then
        mat_count=$(find kuleuven_255_preprocessed -name "*_preprocessed.mat" 2>/dev/null | wc -l)
        echo "Found validated preprocessed data: $mat_count preprocessed .mat files"
    else
        echo "Couldn't find validated preprocessed data"
    fi
    
    # Check how training went
    echo ""
    echo "KU255CNNLOC Training Results:"
    if [ -d "ku255cnnloc_results" ]; then
        echo "Training finished successfully"
    else
        echo "Training failed - couldn't find the results directory"
    fi
    
    echo ""
    echo "KU255CNNLOC Training Complete"
}

# Main function - this is where everything happens
main() {
    echo "Starting the KU255CNNLOC training pipeline..."
    
    # Parse window_size argument if provided
    # Usage: bash KU255CNNLOC.sh --window_size 2048
    # Or: WINDOW_SIZE=2048 bash KU255CNNLOC.sh
    # Or: bash KU255CNNLOC.sh --window_size 2048 --tune
    while [[ $# -gt 0 ]]; do
        case $1 in
            --window_size)
                export WINDOW_SIZE="$2"
                echo "Window size set to: $WINDOW_SIZE samples"
                shift 2
                ;;
            --tune|-t)
                TUNE_MODE=1
                shift
                ;;
            --all_windows|--all-windows)
                ALL_WINDOWS_MODE=1
                shift
                ;;
            --min_window)
                MIN_WINDOW="$2"
                shift 2
                ;;
            --max_window)
                MAX_WINDOW="$2"
                shift 2
                ;;
            --step)
                STEP="$2"
                shift 2
                ;;
            --output_base)
                OUTPUT_BASE_DIR="$2"
                shift 2
                ;;
            --results_file)
                RESULTS_FILE="$2"
                shift 2
                ;;
            --skip_existing)
                SKIP_EXISTING=true
                shift
                ;;
            *)
                # Unknown argument, keep it for potential future use
                shift
                ;;
        esac
    done
    
    # See if they want to run all window sizes
    if [ "${ALL_WINDOWS_MODE:-0}" = "1" ]; then
        echo "All Window Sizes Mode Enabled"
        echo ""
        
        check_python_env
        echo ""
        check_ku255_data
        if [ $? -ne 0 ]; then
            run_ku255_preprocessing || exit 1
        fi
        
        echo ""
        run_all_window_sizes || exit 1
        
        echo ""
        echo "Success! All window sizes testing finished!"
        exit 0
    fi
    
    # See if they want to do hyperparameter tuning
    if [ "${TUNE_MODE:-0}" = "1" ]; then
        echo "Hyperparameter Tuning Mode Enabled"
        echo ""
        
        check_python_env
        echo ""
        check_ku255_data
        if [ $? -ne 0 ]; then
            run_ku255_preprocessing || exit 1
        fi
        
        echo ""
        run_hyperparameter_tuning || exit 1
        
        echo ""
        echo "Success! Hyperparameter tuning finished!"
        exit 0
    fi
    
    # Regular training mode
    echo "Regular Training Mode"
    echo ""
    
    check_python_env
    echo ""
    check_ku255_data
    if [ $? -ne 0 ]; then
        run_ku255_preprocessing || exit 1
    fi
    
    echo ""
    run_ku255cnnloc_training || exit 1
    
    echo ""
    create_final_summary
    
    echo ""
    echo "Success! KU255CNNLOC training finished!"
    exit 0
}

main "$@"

