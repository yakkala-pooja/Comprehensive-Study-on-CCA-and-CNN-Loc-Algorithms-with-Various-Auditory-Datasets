#!/bin/bash
#SBATCH --job-name=fulcnnloc
#SBATCH --output=fulcnnloc_%j.out
#SBATCH --error=fulcnnloc_%j.err
#SBATCH --time=48:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=tier3
#SBATCH --account=neurosteer

# FULCNNLOC - CNN-LOC Algorithm for Fulsang Dataset
# This script runs the FULCNNLOC model training with CNN-LOC architecture
# 
# What it does:
# - Uses FULPRE.py to preprocess Fulsang data from MATLAB files
# - Creates TFRecord files for efficient data loading
# - Trains CNN-LOC model (similar to CombinedCNNLOC.py architecture)
# - Calculates accuracy, ROC-AUC, and other metrics
# - Handles preprocessing and data loading properly
# - Generates detailed reports on model performance

echo "FULCNNLOC - CNN-LOC Algorithm for Fulsang Dataset"
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
    if [ -d "fulcnnloc_results" ]; then
        echo "Backing up FULCNNLOC results..."
        cp -r fulcnnloc_results fulcnnloc_results_backup_$(date +%Y%m%d_%H%M%S) 2>/dev/null || true
    fi
    
    # Save the training log too
    if [ -f "fulcnnloc_training.log" ]; then
        echo "Backing up training log..."
        cp fulcnnloc_training.log fulcnnloc_training_backup_$(date +%Y%m%d_%H%M%S).log 2>/dev/null || true
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
}

# Check if we have the Fulsang data ready to go
check_fulsang_data() {
    echo "Checking if Fulsang Data is Available"
    
    # Look for the preprocessed TFRecord files
    if [ -d "fulsang_preprocessed/tfrecords" ]; then
        tfrecord_count=$(find fulsang_preprocessed/tfrecords -name "*.tfrecords" 2>/dev/null | wc -l)
        if [ "$tfrecord_count" -gt 0 ]; then
            echo "Found preprocessed TFRecord files from FULPRE.py"
            echo "Found $tfrecord_count TFRecord files"
            echo "Data is ready for training"
            return 0
        else
            echo "Directory exists but contains no TFRecord files!"
            echo "Expected location: fulsang_preprocessed/tfrecords"
            echo "This probably means FULPRE.py failed or hasn't been run yet"
            return 1
        fi
    else
        echo "Couldn't find preprocessed TFRecord files!"
        echo "Expected location: fulsang_preprocessed/tfrecords"
        echo ""
        echo "This probably means:"
        echo "  1. FULPRE.py hasn't been run yet"
        echo "  2. The data is somewhere else"
        echo ""
        echo "You'll need to run FULPRE.py first to create the TFRecord files"
        return 1
    fi
}

# Run the preprocessing step if we need to
run_fulsang_preprocessing() {
    echo "Running Fulsang Preprocessing with FULPRE.py"
    
    # Check if we have the raw MATLAB files
    if [ ! -d "Data/Fulsang/DATA_preproc" ]; then
        echo "ERROR: Could not find Data/Fulsang/DATA_preproc directory"
        echo "Please ensure the Fulsang preprocessed MATLAB files are available"
        return 1
    fi
    
    # Run the preprocessing script
    if [ -f "FULPRE.py" ]; then
        echo "Running FULPRE.py..."
        echo "This may take several minutes..."
        python3 FULPRE.py \
            --data_dir "Data/Fulsang" \
            --output_dir "fulsang_preprocessed" > fulpreprocessing.log 2>&1
        
        local exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            echo "Preprocessing finished successfully"
            echo "Results are in fulsang_preprocessed/"
            
            # Verify TFRecord files were created
            if [ -d "fulsang_preprocessed/tfrecords" ]; then
                tfrecord_count=$(find fulsang_preprocessed/tfrecords -name "*.tfrecords" 2>/dev/null | wc -l)
                if [ "$tfrecord_count" -gt 0 ]; then
                    echo "Created $tfrecord_count TFRecord files"
                    return 0
                else
                    echo "ERROR: TFRecord directory exists but contains no files!"
                    echo "Check fulpreprocessing.log for errors"
                    tail -30 fulpreprocessing.log
                    return 1
                fi
            else
                echo "ERROR: Couldn't find the TFRecord directory after preprocessing"
                echo "Check fulpreprocessing.log for errors"
                tail -30 fulpreprocessing.log
                return 1
            fi
        else
            echo "ERROR: Preprocessing failed with exit code $exit_code"
            echo "Check fulpreprocessing.log for details:"
            tail -50 fulpreprocessing.log
            return 1
        fi
    else
        echo "ERROR: Couldn't find FULPRE.py!"
        return 1
    fi
}

# Run comprehensive analysis: window size sweep + hyperparameter tuning
run_comprehensive_analysis() {
    echo "Running Comprehensive FULCNNLOC Analysis"
    echo "This includes:"
    echo "  1. Window size sweep (1s to 30s)"
    echo "  2. Hyperparameter tuning for best window size"
    echo ""
    
    if [ ! -f "FULCNNLOC.py" ]; then
        echo "Couldn't find FULCNNLOC.py!"
        return 1
    fi
    
    if [ ! -d "fulsang_preprocessed/tfrecords" ]; then
        echo "No preprocessed TFRecord files found! Run preprocessing first"
        return 1
    fi
    
    echo "Starting comprehensive analysis for the Fulsang dataset..."
    TFRECORD_DIR="fulsang_preprocessed/tfrecords"
    
    # Run comprehensive analysis (window sweep + hyperparameter tuning)
    python3 FULCNNLOC.py \
        --tfrecord_dir "$TFRECORD_DIR" \
        --run_all \
        --batch_size 32 \
        --num_epochs 30 \
        --learning_rate 1e-3 \
        --overlap 0.5 \
        --dropout_rate 0.3 \
        --output_dir fulcnnloc_results > fulcnnloc_comprehensive.log 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "Comprehensive Analysis Finished Successfully!"
        echo "Finished at: $(date)"
        
        # Display results summary
        if [ -f "fulcnnloc_results/window_size_sweep_results.json" ]; then
            echo ""
            echo "WINDOW SIZE SWEEP RESULTS SUMMARY"
            python3 -c "
import json
import os

try:
    with open('fulcnnloc_results/window_size_sweep_results.json', 'r') as f:
        results = json.load(f)
    
    valid_results = [r for r in results if 'error' not in r and 'accuracy' in r]
    if valid_results:
        best = max(valid_results, key=lambda x: x['accuracy'])
        print(f'Best Window Size: {best[\"window_size_seconds\"]:.1f}s ({best[\"window_size_samples\"]} samples)')
        print(f'Best Accuracy: {best[\"accuracy\"]:.4f}')
        print(f'Best ROC-AUC: {best.get(\"roc_auc\", \"N/A\")}')
        print(f'Best F1-Score: {best.get(\"f1_score\", \"N/A\")}')
    
except Exception as e:
    print(f'Could not read window size sweep results: {e}')
"
        fi
        
        if [ -f "fulcnnloc_results/hyperparameter_tuning_results.json" ]; then
            echo ""
            echo "HYPERPARAMETER TUNING RESULTS SUMMARY"
            python3 -c "
import json
import os

try:
    with open('fulcnnloc_results/hyperparameter_tuning_results.json', 'r') as f:
        results = json.load(f)
    
    valid_results = [r for r in results if 'error' not in r and 'accuracy' in r]
    if valid_results:
        best = max(valid_results, key=lambda x: x['accuracy'])
        print(f'Best Learning Rate: {best[\"learning_rate\"]:.0e}')
        print(f'Best Batch Size: {best[\"batch_size\"]}')
        print(f'Best Dropout Rate: {best[\"dropout_rate\"]:.2f}')
        print(f'Best Accuracy: {best[\"accuracy\"]:.4f}')
        print(f'Best ROC-AUC: {best.get(\"roc_auc\", \"N/A\")}')
        print(f'Best F1-Score: {best.get(\"f1_score\", \"N/A\")}')
    
except Exception as e:
    print(f'Could not read hyperparameter tuning results: {e}')
"
        fi
        
        return 0
    else
        echo "Comprehensive Analysis Failed (exit code: $exit_code)"
        echo "Check the error log: fulcnnloc_comprehensive.log"
        tail -50 fulcnnloc_comprehensive.log
        return $exit_code
    fi
}

# Actually run the FULCNNLOC training (single experiment)
run_fulcnnloc_training() {
    echo "Running FULCNNLOC Training (Single Experiment)"
    
    if [ ! -f "FULCNNLOC.py" ]; then
        echo "Couldn't find FULCNNLOC.py!"
        return 1
    fi
    
    if [ ! -d "fulsang_preprocessed/tfrecords" ]; then
        echo "No preprocessed TFRecord files found! Run preprocessing first"
        return 1
    fi
    
    echo "Starting FULCNNLOC training for the Fulsang dataset..."
    TFRECORD_DIR="fulsang_preprocessed/tfrecords"
    
    # Run the training with optimized hyperparameters
    python3 FULCNNLOC.py \
        --tfrecord_dir "$TFRECORD_DIR" \
        --batch_size 32 \
        --num_epochs 50 \
        --learning_rate 1e-3 \
        --window_size 512 \
        --overlap 0.5 \
        --dropout_rate 0.3 \
        --output_dir fulcnnloc_results > fulcnnloc_training.log 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "FULCNNLOC Training Finished Successfully!"
        echo "Finished at: $(date)"
        
        # Display results summary
        if [ -d "fulcnnloc_results" ] && [ -f "fulcnnloc_results/results.json" ]; then
            echo ""
            echo "FULCNNLOC RESULTS SUMMARY"
            python3 -c "
import json
import os

try:
    with open('fulcnnloc_results/results.json', 'r') as f:
        results = json.load(f)
    
    print(f'Test Accuracy: {results[\"accuracy\"]:.4f}')
    print(f'Test ROC-AUC: {results.get(\"roc_auc\", \"N/A\")}')
    print(f'Best Validation Accuracy: {results.get(\"best_val_acc\", \"N/A\")}')
    print(f'Timestamp: {results.get(\"timestamp\", \"N/A\")}')
    
except Exception as e:
    print(f'Could not read results: {e}')
    print('Please check the results directory manually')
"
        fi
        
        return 0
    else
        echo "FULCNNLOC Training Failed (exit code: $exit_code)"
        echo "Check the error log: fulcnnloc_training.log"
        tail -50 fulcnnloc_training.log
        return $exit_code
    fi
}

# Create a summary of what we did
create_final_summary() {
    echo ""
    echo "Final Summary Report"
    echo "Algorithm: FULCNNLOC (CNN-LOC for Fulsang Dataset)"
    echo "Finished at: $(date)"
    echo ""
    
    # Check how preprocessing went
    echo "Preprocessing Results:"
    if [ -d "fulsang_preprocessed/tfrecords" ]; then
        tfrecord_count=$(find fulsang_preprocessed/tfrecords -name "*.tfrecords" 2>/dev/null | wc -l)
        echo "Found preprocessed TFRecord files: $tfrecord_count files"
    else
        echo "Couldn't find preprocessed TFRecord files"
    fi
    
    # Check how training went
    echo ""
    echo "FULCNNLOC Analysis Results:"
    if [ -d "fulcnnloc_results" ]; then
        echo "Analysis finished successfully"
        if [ -f "fulcnnloc_results/window_size_sweep_results.json" ]; then
            echo "Window size sweep results found"
        fi
        if [ -f "fulcnnloc_results/hyperparameter_tuning_results.json" ]; then
            echo "Hyperparameter tuning results found"
        fi
        if [ -f "fulcnnloc_results/results.json" ]; then
            echo "Single experiment results found"
        fi
    else
        echo "Analysis failed - couldn't find the results directory"
    fi
    
    echo ""
    echo "FULCNNLOC Training Complete"
}

# Main function - this is where everything happens
main() {
    echo "Starting the FULCNNLOC training pipeline..."
    echo ""
    
    check_python_env
    echo ""
    
    check_fulsang_data
    if [ $? -ne 0 ]; then
        echo ""
        echo "Preprocessed data not found. Running FULPRE.py preprocessing..."
        run_fulsang_preprocessing || exit 1
        echo ""
    fi
    
    echo ""
    # Run comprehensive analysis (window sweep + hyperparameter tuning)
    run_comprehensive_analysis || exit 1
    
    echo ""
    create_final_summary
    
    echo ""
    echo "Success! FULCNNLOC training finished!"
    exit 0
}

main "$@"

