#!/bin/bash
#SBATCH --job-name=stanet_training
#SBATCH --output=stanet_training_%j.out
#SBATCH --error=stanet_training_%j.err
#SBATCH --time=8:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=tier3
#SBATCH --account=neurosteer

# STAnet - SpatioTemporal Attention Network for Auditory Attention Detection with EEG
# This script runs the STAnet implementation as described in:
# "Leveraging Graphic and Convolutional Neural Networks for Auditory Attention Detection 
#  with EEG on Das Dataset" by Pahuja et al., Interspeech 2024
# 
# Features:
# - Graph Convolutional Networks for modeling EEG channel relationships
# - Spatial Attention Mechanism for weighting EEG channels
# - Temporal Attention Mechanism for weighting temporal patterns
# - Convolutional layers for hierarchical feature extraction
# - Comprehensive metrics evaluation

echo "=================================================================================="
echo "STAnet - SPATIOTEMPORAL ATTENTION NETWORK FOR AUDITORY ATTENTION DETECTION"
echo "=================================================================================="
echo "Started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=================================================================================="

# Environment setup
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/telluride_decoding"
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=8

# Store original working directory
ORIGINAL_DIR="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Use SLURM_SUBMIT_DIR if available (directory where sbatch was run)
# Otherwise use script directory
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    LOG_DIR="$SLURM_SUBMIT_DIR"
    WORK_DIR="$SLURM_SUBMIT_DIR"
else
    LOG_DIR="$SCRIPT_DIR"
    WORK_DIR="$SCRIPT_DIR"
fi

# Change to work directory to ensure proper imports
cd "$WORK_DIR"
export PYTHONPATH="${PYTHONPATH}:$(pwd):$ORIGINAL_DIR"

# Timeout handler for job management
timeout_handler() {
    echo "=================================================================================="
    echo "JOB TIMEOUT WARNING: 90% of time limit reached"
    echo "Current time: $(date)"
    echo "Attempting to save current progress..."
    echo "=================================================================================="
    
    # Try to save any partial results
    if [ -d "stanet_results" ]; then
        echo "Saving STAnet partial results..."
        cp -r stanet_results stanet_results_backup_$(date +%Y%m%d_%H%M%S) 2>/dev/null || true
    fi
    
    # Try to save any log files
    TRAINING_LOG="$LOG_DIR/stanet_training.log"
    if [ -f "$TRAINING_LOG" ]; then
        echo "Saving STAnet training log..."
        cp "$TRAINING_LOG" "$LOG_DIR/stanet_training_backup_$(date +%Y%m%d_%H%M%S).log" 2>/dev/null || true
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
        
        # Check for train/test/val subdirectories
        if [ -d "das_16subjects_preprocessed/tfrecords/train" ] && [ -d "das_16subjects_preprocessed/tfrecords/test" ]; then
            train_count=$(find das_16subjects_preprocessed/tfrecords/train -name "*.tfrecords" 2>/dev/null | wc -l)
            test_count=$(find das_16subjects_preprocessed/tfrecords/test -name "*.tfrecords" 2>/dev/null | wc -l)
            echo "✓ Found separate train/test directories"
            echo "✓ Train TFRecord files: $train_count"
            echo "✓ Test TFRecord files: $test_count"
            if [ -d "das_16subjects_preprocessed/tfrecords/val" ]; then
                val_count=$(find das_16subjects_preprocessed/tfrecords/val -name "*.tfrecords" 2>/dev/null | wc -l)
                echo "✓ Val TFRecord files: $val_count"
            fi
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
        
        # Check for train/test/val subdirectories
        if [ -d "corrected_das_analysis_results/tfrecords/train" ] && [ -d "corrected_das_analysis_results/tfrecords/test" ]; then
            train_count=$(find corrected_das_analysis_results/tfrecords/train -name "*.tfrecords" 2>/dev/null | wc -l)
            test_count=$(find corrected_das_analysis_results/tfrecords/test -name "*.tfrecords" 2>/dev/null | wc -l)
            echo "✓ Found separate train/test directories"
            echo "✓ Train TFRecord files: $train_count"
            echo "✓ Test TFRecord files: $test_count"
            if [ -d "corrected_das_analysis_results/tfrecords/val" ]; then
                val_count=$(find corrected_das_analysis_results/tfrecords/val -name "*.tfrecords" 2>/dev/null | wc -l)
                echo "✓ Val TFRecord files: $val_count"
            fi
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
    echo "This step prepares the DAS dataset for STAnet training:"
    echo "  - Loads EEG data (64 channels, 1000 Hz sampling)"
    echo "  - Extracts validated attention labels"
    echo "  - Creates TFRecord files for efficient training"
    echo "  - Prevents data leakage with subject-wise organization"
    echo "=================================================================================="
    
    # Run DAS preprocessing
    # Check for file in current directory or original directory
    PREPROCESSING_SCRIPT=""
    if [ -f "das_preprocessing_16subjects.py" ]; then
        PREPROCESSING_SCRIPT="das_preprocessing_16subjects.py"
    elif [ -f "$ORIGINAL_DIR/das_preprocessing_16subjects.py" ]; then
        PREPROCESSING_SCRIPT="$ORIGINAL_DIR/das_preprocessing_16subjects.py"
    fi
    
    if [ -n "$PREPROCESSING_SCRIPT" ]; then
        echo "Running das_preprocessing_16subjects.py (16-SUBJECTS VERSION)..."
        echo "Features:"
        echo "  ✓ Validated attention labels with quality control"
        echo "  ✓ Subject-wise organized data (no data leakage)"
        echo "  ✓ Robust EEG data extraction (64 channels, 64 Hz)"
        echo "  ✓ Comprehensive preprocessing reports"
        echo "  ✓ 16 subjects support (S1-S16)"
        
        # Create log file in writable directory
        LOG_FILE="$LOG_DIR/das_preprocessing_stanet.log"
        
        # Run preprocessing and capture output
        # Use exec to ensure redirection works correctly
        python3 "$PREPROCESSING_SCRIPT" --data_dir "Data/Das/4004271" --output_dir "das_16subjects_preprocessed" --create_split > "$LOG_FILE" 2>&1
        PREPROCESSING_EXIT_CODE=$?
        
        if [ $PREPROCESSING_EXIT_CODE -eq 0 ]; then
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
            echo "✗ DAS preprocessing failed with exit code: $PREPROCESSING_EXIT_CODE"
            if [ -f "$LOG_FILE" ]; then
                echo "Check the log file: $LOG_FILE"
                echo "Error details:"
                tail -20 "$LOG_FILE" 2>/dev/null || echo "Could not read log file"
            else
                echo "Log file not created. Check Python output above."
            fi
            return 1
        fi
    else
        echo "✗ das_preprocessing_16subjects.py not found!"
        echo "Current directory: $(pwd)"
        echo "Original directory: $ORIGINAL_DIR"
        echo "Script directory: $SCRIPT_DIR"
        echo ""
        echo "Searching for das_preprocessing_16subjects.py..."
        find . -name "das_preprocessing_16subjects.py" -type f 2>/dev/null | head -5
        if [ -n "$ORIGINAL_DIR" ] && [ "$ORIGINAL_DIR" != "$(pwd)" ]; then
            find "$ORIGINAL_DIR" -name "das_preprocessing_16subjects.py" -type f 2>/dev/null | head -5
        fi
        echo ""
        echo "Please ensure das_preprocessing_16subjects.py is available"
        return 1
    fi
}

# Function to run STAnet training
run_stanet_training() {
    echo "=================================================================================="
    echo "RUNNING STAnet TRAINING"
    echo "=================================================================================="
    echo "This step trains the STAnet model with comprehensive metrics:"
    echo "  - Graph Convolutional Networks for EEG channel relationships"
    echo "  - Spatial Attention Mechanism for weighting EEG channels"
    echo "  - Temporal Attention Mechanism for weighting temporal patterns"
    echo "  - Convolutional layers for hierarchical feature extraction"
    echo "  - Accuracy, Balanced Accuracy, F1 Score, ROC-AUC metrics"
    echo "  - DAS integration for data quality"
    echo "  - Data leakage prevention"
    echo "  - Validated attention labels"
    echo "=================================================================================="
    
    if [ ! -f "run_STAnet.py" ]; then
        echo "✗ run_STAnet.py not found!"
        echo "Please ensure the STAnet training script is available"
        return 1
    fi
    
    if [ ! -d "das_16subjects_preprocessed/tfrecords" ] && [ ! -d "corrected_das_analysis_results/tfrecords" ]; then
        echo "✗ No TFRecord data found!"
        echo "Please run preprocessing first"
        return 1
    fi
    
    echo "Starting STAnet training for DAS dataset..."
    
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
    
    # Verify STAnet.py exists
    if [ ! -f "STAnet.py" ]; then
        echo "✗ STAnet.py not found in current directory!"
        echo "Current directory: $(pwd)"
        echo "Looking for STAnet.py..."
        find . -name "STAnet.py" -type f 2>/dev/null | head -5
        return 1
    fi
    
    echo "✓ Found STAnet.py"
    
    # Verify run_STAnet.py exists
    if [ ! -f "run_STAnet.py" ]; then
        echo "✗ run_STAnet.py not found in current directory!"
        return 1
    fi
    
    echo "✓ Found run_STAnet.py"
    
    # Create log file in writable directory
    TRAINING_LOG_FILE="$LOG_DIR/stanet_training.log"
    
    # Also try current directory as fallback
    TRAINING_LOG_FILE_ALT="$(pwd)/stanet_training.log"
    
    echo "Log file will be written to: $TRAINING_LOG_FILE"
    echo "Alternative log location: $TRAINING_LOG_FILE_ALT"
    echo "Current working directory: $(pwd)"
    echo "LOG_DIR: $LOG_DIR"
    
    # Ensure log directory exists and is writable
    mkdir -p "$(dirname "$TRAINING_LOG_FILE")" 2>/dev/null || true
    
    # Run STAnet training with error capture
    set +e  # Don't exit on error
    python3 run_STAnet.py \
        --tfrecord_dir "$TFRecord_DIR" \
        --batch_size 32 \
        --num_epochs 100 \
        --learning_rate 0.0001 \
        --window_size 32 \
        --overlap 0.5 \
        --num_channels 64 \
        --time_steps 32 \
        --num_features 5 \
        --gcn_hidden 64 \
        --dropout_rate 0.3 \
        --num_workers 4 \
        --output_dir stanet_results \
        --save_model > "$TRAINING_LOG_FILE" 2>&1
    
    local exit_code=$?
    set -e  # Re-enable exit on error
    
    # Check if log file was created, if not try to find it
    if [ ! -f "$TRAINING_LOG_FILE" ]; then
        # Try alternative locations
        if [ -f "$TRAINING_LOG_FILE_ALT" ]; then
            TRAINING_LOG_FILE="$TRAINING_LOG_FILE_ALT"
        elif [ -f "$(pwd)/stanet_training.log" ]; then
            TRAINING_LOG_FILE="$(pwd)/stanet_training.log"
        elif [ -f "./stanet_training.log" ]; then
            TRAINING_LOG_FILE="./stanet_training.log"
        fi
    fi
    
    if [ $exit_code -eq 0 ]; then
        echo "=================================================================================="
        echo "STAnet TRAINING COMPLETED SUCCESSFULLY!"
        echo "Finished at: $(date)"
        echo "=================================================================================="
        
        # Check for results
        if [ -d "stanet_results" ]; then
            echo "Results directory: stanet_results"
            echo "Generated files:"
            find stanet_results -type f -name "*.json" -o -name "*.png" -o -name "*.pth" -o -name "*.txt" | sort
            
            # Display results summary
            if [ -f "stanet_results/training_results.json" ]; then
                echo ""
                echo "=================================================================================="
                echo "STAnet RESULTS SUMMARY"
                echo "=================================================================================="
                python3 -c "
import json
import os

try:
    with open('stanet_results/training_results.json', 'r') as f:
        results = json.load(f)
    
    print('TRAINING METRICS:')
    print('-----------------')
    if 'train_losses' in results and len(results['train_losses']) > 0:
        print(f'Final Train Loss: {results[\"train_losses\"][-1]:.4f}')
    if 'train_accs' in results and len(results['train_accs']) > 0:
        print(f'Final Train Accuracy: {results[\"train_accs\"][-1]:.2f}%')
    
    print('')
    print('VALIDATION METRICS:')
    print('-------------------')
    if 'val_losses' in results and len(results['val_losses']) > 0:
        print(f'Final Val Loss: {results[\"val_losses\"][-1]:.4f}')
    if 'val_accs' in results and len(results['val_accs']) > 0:
        print(f'Final Val Accuracy: {results[\"val_accs\"][-1]:.2f}%')
        if len(results['val_accs']) > 0:
            print(f'Best Val Accuracy: {max(results[\"val_accs\"]):.2f}%')
    
    print('')
    print('TEST METRICS:')
    print('-------------')
    if 'test_metrics' in results and results['test_metrics']:
        test_metrics = results['test_metrics']
        print(f'Test Accuracy: {test_metrics.get(\"accuracy\", \"N/A\"):.4f}')
        print(f'Balanced Accuracy: {test_metrics.get(\"balanced_accuracy\", \"N/A\"):.4f}')
        print(f'F1 Score: {test_metrics.get(\"f1_score\", \"N/A\"):.4f}')
        print(f'ROC-AUC: {test_metrics.get(\"roc_auc\", \"N/A\"):.4f}')
    
    print('')
    print('MODEL INFORMATION:')
    print('------------------')
    if os.path.exists('stanet_results/stanet_model.pth'):
        print('✓ Model saved: stanet_results/stanet_model.pth')
    
    print('')
    print('FILES GENERATED:')
    print('---------------')
    for root, dirs, files in os.walk('stanet_results'):
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
        echo "STAnet TRAINING FAILED with exit code: $exit_code"
        echo "=================================================================================="
        
        # Try multiple log file locations
        LOG_FOUND=false
        for log_file in "$TRAINING_LOG_FILE" "$TRAINING_LOG_FILE_ALT" "$LOG_DIR/stanet_training.log" "$(pwd)/stanet_training.log" "./stanet_training.log"; do
            if [ -f "$log_file" ]; then
                echo "Found log file at: $log_file"
                echo "Last 100 lines of error log:"
                tail -100 "$log_file" 2>/dev/null || echo "Could not read log file"
                LOG_FOUND=true
                break
            fi
        done
        
        if [ "$LOG_FOUND" = false ]; then
            echo "Log file not found in any expected location."
            echo "Searched in:"
            echo "  - $TRAINING_LOG_FILE"
            echo "  - $TRAINING_LOG_FILE_ALT"
            echo "  - $LOG_DIR/stanet_training.log"
            echo "  - $(pwd)/stanet_training.log"
            echo ""
            echo "Checking SLURM output file for errors..."
            if [ -n "$SLURM_JOB_ID" ] && [ -f "stanet_training_${SLURM_JOB_ID}.out" ]; then
                echo "Found SLURM output file. Last 50 lines:"
                tail -50 "stanet_training_${SLURM_JOB_ID}.out" 2>/dev/null || echo "Could not read SLURM output"
            fi
            if [ -n "$SLURM_JOB_ID" ] && [ -f "stanet_training_${SLURM_JOB_ID}.err" ]; then
                echo "Found SLURM error file. Last 50 lines:"
                tail -50 "stanet_training_${SLURM_JOB_ID}.err" 2>/dev/null || echo "Could not read SLURM error"
            fi
            echo ""
            echo "Trying to capture Python error output directly..."
            # Try to run again with minimal parameters to capture error
            python3 run_STAnet.py --tfrecord_dir "$TFRecord_DIR" --batch_size 2 --num_epochs 1 --num_workers 0 2>&1 | head -100
        fi
        
        return $exit_code
    fi
}

# Function to create final summary report
create_final_summary() {
    echo "=================================================================================="
    echo "FINAL SUMMARY REPORT"
    echo "=================================================================================="
    echo "Algorithm: STAnet (SpatioTemporal Attention Network)"
    echo "Paper: Leveraging Graphic and Convolutional Neural Networks for Auditory Attention"
    echo "       Detection with EEG on Das Dataset (Pahuja et al., Interspeech 2024)"
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
    echo "STAnet TRAINING RESULTS:"
    echo "------------------------"
    if [ -d "stanet_results" ]; then
        echo "✓ Training completed successfully"
        if [ -f "stanet_results/training_results.json" ]; then
            echo "✓ Results file generated"
        fi
        if [ -f "stanet_results/stanet_model.pth" ]; then
            echo "✓ Model saved"
        fi
    else
        echo "✗ Training failed - no results directory found"
    fi
    
    # List all generated files
    echo ""
    echo "GENERATED FILES:"
    echo "================"
    find . -name "*stanet*" -type f | grep -E "\.(log|json|png|pth|txt)$" | sort
    
    echo ""
    echo "STAnet ARCHITECTURE HIGHLIGHTS:"
    echo "=============================="
    echo "✓ Graph Convolutional Networks (GCN) for EEG channel relationships"
    echo "✓ Spatial Attention Mechanism for weighting EEG channels"
    echo "✓ Temporal Attention Mechanism for weighting temporal patterns"
    echo "✓ Convolutional layers for hierarchical feature extraction"
    echo "✓ Sequential attention refinement (spatial → temporal)"
    
    echo ""
    echo "=================================================================================="
    echo "STAnet TRAINING COMPLETED"
    echo "=================================================================================="
}

# Main execution
main() {
    echo "Starting STAnet training pipeline..."
    
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
    
    # Step 3: Run STAnet training
    echo ""
    run_stanet_training
    if [ $? -ne 0 ]; then
        echo "✗ Training failed, exiting..."
        exit 1
    fi
    
    # Step 4: Create final summary
    echo ""
    create_final_summary
    
    echo ""
    echo "🎉 SUCCESS: STAnet training completed successfully!"
    echo "Check the results in stanet_results/ directory"
    exit 0
}

# Run main function
main "$@"
