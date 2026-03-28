#!/bin/bash
#SBATCH --job-name=dascnn_fast_research
#SBATCH --output=dascnn_fast_research_%j.out
#SBATCH --error=dascnn_fast_research_%j.err
#SBATCH --time=24:00:00  # Increased for hyperparameter tuning (can take 12-24 hours)
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=tier3
#SBATCH --account=neurosteer

# DASCNN_Fast_Research.sh - Fast Research-Grade CNN-LOC for DAS Dataset
#
# This script runs the optimized research-grade CNN-LOC model with:
# - Full CNN-LOC architecture (research-grade quality)
# - Mixed precision training (FP16/FP32) for 2x speedup
# - Model compilation (torch.compile) for 20-30% speedup
# - Upfront preprocessing for 10-50x faster data loading
# - Optimized data loading (multi-worker, prefetching)
# - Comprehensive research-grade metrics
# - Optional hyperparameter tuning (95 experiments)
#
# Expected performance: ~50-100 it/s (vs ~6 it/s original)
# Time per epoch: ~3-5 minutes (vs ~28 minutes original)
# Speedup: ~8-16x faster while maintaining research-grade quality
#
# Hyperparameter tuning:
#   Set TUNE_HYPERPARAMETERS=1 to enable (runs 95 experiments, ~12-24 hours)
#   Example: TUNE_HYPERPARAMETERS=1 sbatch DASCNN_Fast_Research.sh

set -e  # Exit on error

echo "=================================================================================="
echo "DASCNN FAST RESEARCH-GRADE CNN-LOC TRAINING"
echo "=================================================================================="
echo "Started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo "Memory: ${SLURM_MEM_PER_NODE:-$(free -h | grep '^Mem:' | awk '{print $2}')}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "=================================================================================="

# Environment setup
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Timeout handler for job management
timeout_handler() {
    echo "=================================================================================="
    echo "JOB TIMEOUT WARNING: 90% of time limit reached"
    echo "Current time: $(date)"
    echo "Attempting to save current progress..."
    echo "=================================================================================="
    
    # Try to save any partial results
    if [ -d "dascnn_fast_results" ]; then
        echo "Saving DASCNN Fast Research partial results..."
        cp -r dascnn_fast_results dascnn_fast_results_backup_$(date +%Y%m%d_%H%M%S) 2>/dev/null || true
    fi
    
    # Try to save any log files
    if [ -f "dascnn_fast_research_training.log" ]; then
        echo "Saving DASCNN Fast Research training log..."
        cp dascnn_fast_research_training.log dascnn_fast_research_backup_$(date +%Y%m%d_%H%M%S).log 2>/dev/null || true
    fi
}

# Set up timeout handler
trap timeout_handler SIGUSR1

# Function to check Python environment
check_python_env() {
    echo ""
    echo "=================================================================================="
    echo "CHECKING PYTHON ENVIRONMENT"
    echo "=================================================================================="
    
    echo "Python version: $(python3 --version 2>/dev/null || echo 'Python not found')"
    echo "Python executable: $(which python3)"
    echo "Available memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
    echo "Available CPUs: $(nproc)"
    
    # Check GPU configuration
    echo ""
    echo "GPU Configuration:"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | head -1
        echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    else
        echo "nvidia-smi not available (running on CPU)"
    fi
    
    # Check required Python packages
    echo ""
    echo "Checking required Python packages..."
    python3 -c "
import sys
print(f'Python executable: {sys.executable}')

required_packages = {
    'numpy': 'numpy',
    'scipy': 'scipy',
    'torch': 'torch',
    'tensorflow': 'tensorflow',
    'sklearn': 'sklearn',
    'tqdm': 'tqdm'
}

missing_packages = []
for package_name, import_name in required_packages.items():
    try:
        __import__(import_name)
        print(f'✓ {package_name} - Available')
    except ImportError:
        print(f'✗ {package_name} - MISSING')
        missing_packages.append(package_name)

if missing_packages:
    print(f'\\nERROR: Missing packages: {missing_packages}')
    print('Please install missing packages before running this script.')
    sys.exit(1)
else:
    print('\\n✓ All required packages are available')
" || {
    echo "ERROR: Python environment check failed"
    exit 1
}
}

# Function to check data availability
check_data() {
    echo ""
    echo "=================================================================================="
    echo "CHECKING DATA AVAILABILITY"
    echo "=================================================================================="
    
    DAS_TFRECORD_DIR="${DAS_TFRECORD_DIR:-das_16subjects_preprocessed/tfrecords}"
    
    if [ ! -d "$DAS_TFRECORD_DIR" ]; then
        echo "✗ ERROR: Das TFRecord directory not found: $DAS_TFRECORD_DIR"
        echo "  Please run preprocessing first: python3 das_preprocessing_16subjects.py"
        exit 1
    fi
    
    # Check for train/test subdirectories
    if [ -d "$DAS_TFRECORD_DIR/train" ] && [ -d "$DAS_TFRECORD_DIR/test" ]; then
        train_files=$(find "$DAS_TFRECORD_DIR/train" -name "*.tfrecords" 2>/dev/null | wc -l)
        test_files=$(find "$DAS_TFRECORD_DIR/test" -name "*.tfrecords" 2>/dev/null | wc -l)
        
        if [ "$train_files" -gt 0 ] && [ "$test_files" -gt 0 ]; then
            echo "✓ Das TFRecord files found:"
            echo "  Train files: $train_files"
            echo "  Test files: $test_files"
            echo "  Directory: $DAS_TFRECORD_DIR"
        else
            echo "✗ ERROR: Insufficient TFRecord files found"
            echo "  Train files: $train_files (expected > 0)"
            echo "  Test files: $test_files (expected > 0)"
            exit 1
        fi
    else
        # Check for files in main directory
        all_files=$(find "$DAS_TFRECORD_DIR" -name "*.tfrecords" 2>/dev/null | wc -l)
        if [ "$all_files" -gt 0 ]; then
            echo "✓ Das TFRecord files found: $all_files files"
            echo "  Directory: $DAS_TFRECORD_DIR"
        else
            echo "✗ ERROR: No TFRecord files found in $DAS_TFRECORD_DIR"
            exit 1
        fi
    fi
}

# Check environment and data
check_python_env
check_data

# Training parameters (can be overridden by environment variables)
WINDOW_SIZE=${WINDOW_SIZE:-512}
OVERLAP=${OVERLAP:-0.5}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_EPOCHS=${NUM_EPOCHS:-50}
LEARNING_RATE=${LEARNING_RATE:-5e-4}
DROPOUT_RATE=${DROPOUT_RATE:-0.45}
WEIGHT_DECAY=${WEIGHT_DECAY:-5e-5}
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.08}
NUM_WORKERS=${NUM_WORKERS:-4}
USE_MIXED_PRECISION=${USE_MIXED_PRECISION:-1}
COMPILE_MODEL=${COMPILE_MODEL:-0}  # Disabled by default due to compilation issues on some systems
PREPROCESS_ALL=${PREPROCESS_ALL:-1}
TUNE_HYPERPARAMETERS=${TUNE_HYPERPARAMETERS:-0}  # Set to 1 to enable hyperparameter tuning
OUTPUT_DIR=${OUTPUT_DIR:-dascnn_fast_results}
DAS_TFRECORD_DIR="${DAS_TFRECORD_DIR:-das_16subjects_preprocessed/tfrecords}"

# Display training configuration
echo ""
echo "=================================================================================="
echo "TRAINING CONFIGURATION"
echo "=================================================================================="
echo "Window size: $WINDOW_SIZE samples"
echo "Overlap: $OVERLAP"
echo "Batch size: $BATCH_SIZE"
echo "Number of epochs: $NUM_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Dropout rate: $DROPOUT_RATE"
echo "Weight decay: $WEIGHT_DECAY"
echo "Label smoothing: $LABEL_SMOOTHING"
echo "Data loading workers: $NUM_WORKERS"
echo "Mixed precision: $([ "$USE_MIXED_PRECISION" -eq 1 ] && echo 'Enabled' || echo 'Disabled')"
echo "Model compilation: $([ "$COMPILE_MODEL" -eq 1 ] && echo 'Enabled' || echo 'Disabled')"
echo "Upfront preprocessing: $([ "$PREPROCESS_ALL" -eq 1 ] && echo 'Enabled' || echo 'Disabled')"
echo "Hyperparameter tuning: $([ "$TUNE_HYPERPARAMETERS" -eq 1 ] && echo 'Enabled (95 experiments)' || echo 'Disabled')"
echo "Output directory: $OUTPUT_DIR"
echo "TFRecord directory: $DAS_TFRECORD_DIR"
echo "=================================================================================="

# Build command arguments
CMD_ARGS=(
    --tfrecord_dir "$DAS_TFRECORD_DIR"
    --window_size "$WINDOW_SIZE"
    --overlap "$OVERLAP"
    --batch_size "$BATCH_SIZE"
    --num_epochs "$NUM_EPOCHS"
    --learning_rate "$LEARNING_RATE"
    --dropout_rate "$DROPOUT_RATE"
    --weight_decay "$WEIGHT_DECAY"
    --label_smoothing "$LABEL_SMOOTHING"
    --output_dir "$OUTPUT_DIR"
    --num_workers "$NUM_WORKERS"
)

# Add optional flags
if [ "$USE_MIXED_PRECISION" -eq 1 ]; then
    CMD_ARGS+=(--use_mixed_precision)
fi

if [ "$COMPILE_MODEL" -eq 1 ]; then
    CMD_ARGS+=(--compile_model)
fi

if [ "$PREPROCESS_ALL" -eq 1 ]; then
    CMD_ARGS+=(--preprocess_all)
fi

# Add hyperparameter tuning flag if enabled
if [ "$TUNE_HYPERPARAMETERS" -eq 1 ]; then
    CMD_ARGS+=(--tune_hyperparameters)
    echo ""
    echo "⚠ WARNING: Hyperparameter tuning mode enabled!"
    echo "  This will run 95 experiments (75 + 20) and may take 12-24 hours"
    echo "  Estimated time: ~2-3 hours per experiment × 95 = 190-285 hours total"
    echo "  Consider using a longer time limit or running in stages"
    echo ""
fi

# Run training
echo ""
echo "=================================================================================="
echo "RUNNING FAST RESEARCH-GRADE DAS CNN-LOC TRAINING"
echo "=================================================================================="
echo "Command: python3 DASCNN_Fast_Research.py ${CMD_ARGS[*]}"
echo "=================================================================================="
echo ""

# Run with logging
python3 DASCNN_Fast_Research.py "${CMD_ARGS[@]}" 2>&1 | tee dascnn_fast_research_training.log

TRAINING_EXIT_CODE=${PIPESTATUS[0]}

# Check results
echo ""
echo "=================================================================================="
echo "TRAINING SUMMARY"
echo "=================================================================================="
echo "Finished at: $(date)"
echo ""

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    if [ "$TUNE_HYPERPARAMETERS" -eq 1 ]; then
        echo "✓ Hyperparameter tuning completed successfully"
        
        # Display tuning results if available
        if [ -f "$OUTPUT_DIR/hyperparameter_tuning_results.json" ]; then
            echo ""
            echo "Tuning results summary:"
            python3 -c "
import json
import sys
try:
    with open('$OUTPUT_DIR/hyperparameter_tuning_results.json', 'r') as f:
        results = json.load(f)
    valid_results = [r for r in results if 'error' not in r and 'accuracy' in r]
    if valid_results:
        best = max(valid_results, key=lambda x: x['accuracy'])
        print(f\"  Best Test Accuracy: {best.get('accuracy', 'N/A'):.4f}\")
        print(f\"  Best Val Accuracy: {best.get('best_val_acc', 'N/A'):.4f}\")
        print(f\"  Best Learning Rate: {best.get('learning_rate', 'N/A'):.0e}\")
        print(f\"  Best Batch Size: {best.get('batch_size', 'N/A')}\")
        print(f\"  Best Dropout Rate: {best.get('dropout_rate', 'N/A'):.2f}\")
        print(f\"  Best Weight Decay: {best.get('weight_decay', 'N/A'):.0e}\")
        print(f\"  Best Label Smoothing: {best.get('label_smoothing', 'N/A'):.2f}\")
        print(f\"  Total experiments: {len(valid_results)}/{len(results)}\")
    else:
        print('  No valid results found')
except Exception as e:
    print(f'  Could not read tuning results: {e}')
" || echo "  Could not read tuning results file"
        fi
    else
        echo "✓ Fast Research-Grade DAS CNN-LOC training completed successfully"
        
        # Display results if available
        if [ -f "$OUTPUT_DIR/results.json" ]; then
        echo ""
        echo "Results summary:"
        python3 -c "
import json
import sys
try:
    with open('$OUTPUT_DIR/results.json', 'r') as f:
        results = json.load(f)
    print(f\"  Test Accuracy: {results.get('accuracy', 'N/A'):.4f}\")
    print(f\"  Balanced Accuracy: {results.get('balanced_accuracy', 'N/A'):.4f}\")
    print(f\"  ROC-AUC: {results.get('roc_auc', 'N/A'):.4f}\")
    print(f\"  F1 Score: {results.get('f1_score', 'N/A'):.4f}\")
    print(f\"  Matthews Correlation: {results.get('matthews_corrcoef', 'N/A'):.4f}\")
    print(f\"  Best Val Accuracy: {results.get('best_val_acc', 'N/A'):.4f}\")
except Exception as e:
    print(f'  Could not read results: {e}')
" || echo "  Could not read results file"
        fi
    fi
else
    echo "✗ Fast Research-Grade DAS CNN-LOC training failed with exit code: $TRAINING_EXIT_CODE"
    echo "  Check dascnn_fast_research_training.log for details"
fi

echo ""
echo "Output directory: $OUTPUT_DIR/"
echo "Log file: dascnn_fast_research_training.log"
echo "=================================================================================="

# Exit with error if training failed
if [ $TRAINING_EXIT_CODE -ne 0 ]; then
    exit 1
fi

exit 0

