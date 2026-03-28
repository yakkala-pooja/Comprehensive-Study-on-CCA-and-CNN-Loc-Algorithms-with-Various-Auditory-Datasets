#!/bin/bash
#SBATCH --job-name=stanetcnn_training
#SBATCH --output=stanetcnn_training_%j.out
#SBATCH --error=stanetcnn_training_%j.err
#SBATCH --time=24:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=tier3
#SBATCH --account=neurosteer

# STANETCNN.sh - Dual-Branch Architecture Training Script
#
# This script trains STANETCNN model on three datasets:
# 1. Combined Dataset (Das + Fulsang)
# 2. Das Dataset
# 3. Fulsang Dataset
#
# Architecture:
# - STAtNet Branch: Spatial-Temporal Attention CNN
# - ST-GCN Branch: Spatio-Temporal Graph Convolution Network
# - Soft-Voting Fusion Layer

set -e  # Exit on error

echo "=================================================================================="
echo "STANETCNN - Dual-Branch Architecture for Auditory Attention Decoding"
echo "=================================================================================="
echo "Started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "=================================================================================="
echo ""
echo "GPU Memory Management:"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS"
echo "  Effective batch size: $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "  GCN hidden dimension: $GCN_HIDDEN"
echo "  PyTorch CUDA alloc config: $PYTORCH_CUDA_ALLOC_CONF"
echo "=================================================================================="

# Environment setup
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/telluride_decoding"
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=8

# GPU Memory Management
# Enable expandable segments to avoid fragmentation (as suggested in PyTorch docs)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Set memory fraction to prevent OOM (use 90% of GPU memory)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF},max_split_size_mb:512"

# Default paths
DAS_DATA_DIR="${DAS_DATA_DIR:-das_16subjects_preprocessed}"
DAS_TFRECORD_DIR="${DAS_TFRECORD_DIR:-das_16subjects_preprocessed/tfrecords}"
FULSANG_TFRECORD_DIR="${FULSANG_TFRECORD_DIR:-fulsang_preprocessed/tfrecords}"
FULSANG_RAW_DIR="${FULSANG_RAW_DIR:-}"
FULSANG_AUDIO_DIR="${FULSANG_AUDIO_DIR:-}"
FULSANG_MWF_DIR="${FULSANG_MWF_DIR:-/home/py9363/telluride_decoding/MWF_cleaned_Fuglsang}"

# Training parameters (optimized for memory efficiency)
WINDOW_SIZE=512
OVERLAP=0.5
BATCH_SIZE=8  # Reduced from 32 for dual-branch architecture memory efficiency
GRADIENT_ACCUMULATION_STEPS=2  # Effective batch size = 8 * 2 = 16
NUM_EPOCHS=50
LEARNING_RATE=1e-3
DROPOUT_RATE=0.3
GCN_HIDDEN=32  # Reduced from 64 for memory efficiency

# Results tracking
COMBINED_SUCCESS=0
DAS_SUCCESS=0
FULSANG_SUCCESS=0

# ============================================================================
# Function: Check GPU memory status
# ============================================================================
check_gpu_memory() {
    echo ""
    echo "GPU Memory Status:"
    python3 -c "
import torch
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    total_mem = props.total_memory / 1e9
    allocated = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    free = total_mem - reserved
    print(f'  Device: {device} ({props.name})')
    print(f'  Total: {total_mem:.2f} GB')
    print(f'  Allocated: {allocated:.2f} GB')
    print(f'  Reserved: {reserved:.2f} GB')
    print(f'  Free: {free:.2f} GB')
else:
    print('  CUDA not available')
" 2>/dev/null || echo "  Could not check GPU memory"
    echo ""
}

# Clear GPU memory before starting
echo "Initial GPU memory status:"
check_gpu_memory
echo "Clearing GPU memory..."
python3 -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None; print('GPU memory cleared')" 2>/dev/null || echo "GPU memory clearing attempted"

# ============================================================================
# Function: Run training on a dataset
# ============================================================================
run_training() {
    local dataset=$1
    local output_dir=$2
    local log_file=$3
    shift 3
    local extra_args="$@"
    
    echo ""
    echo "=================================================================================="
    echo "RUNNING STANETCNN TRAINING: $dataset"
    echo "=================================================================================="
    
    # Clear GPU memory before each training run
    echo "Clearing GPU memory before $dataset training..."
    check_gpu_memory
    python3 -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true
    
    # Build command with eval to properly handle quoted arguments
    eval python3 STANETCNN.py \
        --dataset \"$dataset\" \
        --window_size $WINDOW_SIZE \
        --overlap $OVERLAP \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --num_epochs $NUM_EPOCHS \
        --learning_rate $LEARNING_RATE \
        --dropout_rate $DROPOUT_RATE \
        --gcn_hidden $GCN_HIDDEN \
        --output_dir \"$output_dir\" \
        $extra_args > \"$log_file\" 2>&1
    
    # Clear GPU memory after training
    echo "Clearing GPU memory after $dataset training..."
    python3 -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true
    check_gpu_memory
    
    return $?
}

# ============================================================================
# 1. Combined Dataset Training
# ============================================================================
echo ""
echo "=================================================================================="
echo "CHECKING COMBINED DATASET"
echo "=================================================================================="
echo "Note: CombinedDataset loads data in memory and combines Das + Fulsang datasets."
echo "      Combined data is stored in memory (not saved to disk)."
echo "      MWF-processed Fulsang data is stored in: $FULSANG_MWF_DIR"
echo ""

if [ -d "$DAS_DATA_DIR" ]; then
    echo "✓ Das data directory found: $DAS_DATA_DIR"
    
    # Check for CombinedDataset requirements
    if [ -d "$DAS_DATA_DIR/tfrecords" ] || [ -f "$DAS_DATA_DIR"/*.mat ]; then
        echo "✓ Das data files found"
        
        # Build extra args for combined dataset
        COMBINED_EXTRA_ARGS="--das_data_dir \"$DAS_DATA_DIR\" --das_preprocessing_type \"16SUBJECTS\" --fulsang_mwf_dir \"$FULSANG_MWF_DIR\""
        
        # Only add optional Fulsang args if they're set and non-empty
        if [ -n "$FULSANG_RAW_DIR" ]; then
            COMBINED_EXTRA_ARGS="$COMBINED_EXTRA_ARGS --fulsang_raw_dir \"$FULSANG_RAW_DIR\""
        fi
        if [ -n "$FULSANG_AUDIO_DIR" ]; then
            COMBINED_EXTRA_ARGS="$COMBINED_EXTRA_ARGS --fulsang_audio_dir \"$FULSANG_AUDIO_DIR\""
        fi
        
        # Run Combined dataset training
        run_training "combined" \
            "stanetcnn_combined_results" \
            "stanetcnn_combined_training.log" \
            $COMBINED_EXTRA_ARGS
        
        if [ $? -eq 0 ]; then
            COMBINED_SUCCESS=1
            echo "✓ Combined dataset training completed successfully"
        else
            echo "✗ Combined dataset training failed"
            echo "  Check stanetcnn_combined_training.log for details"
        fi
    else
        echo "⚠ Warning: Das data files not found, skipping Combined dataset"
    fi
else
    echo "⚠ Warning: Das data directory not found: $DAS_DATA_DIR"
    echo "  Skipping Combined dataset training"
fi

# ============================================================================
# 2. Das Dataset Training
# ============================================================================
echo ""
echo "=================================================================================="
echo "CHECKING DAS DATASET"
echo "=================================================================================="

if [ -d "$DAS_TFRECORD_DIR" ]; then
    train_files=$(find "$DAS_TFRECORD_DIR/train" -name "*.tfrecords" 2>/dev/null | wc -l)
    test_files=$(find "$DAS_TFRECORD_DIR/test" -name "*.tfrecords" 2>/dev/null | wc -l)
    
    if [ "$train_files" -gt 0 ] && [ "$test_files" -gt 0 ]; then
        echo "✓ Das TFRecord files found:"
        echo "  Train files: $train_files"
        echo "  Test files: $test_files"
        
        # Run Das dataset training
        run_training "das" \
            "stanetcnn_das_results" \
            "stanetcnn_das_training.log" \
            --das_tfrecord_dir "$DAS_TFRECORD_DIR"
        
        if [ $? -eq 0 ]; then
            DAS_SUCCESS=1
            echo "✓ Das dataset training completed successfully"
        else
            echo "✗ Das dataset training failed"
            echo "  Check stanetcnn_das_training.log for details"
        fi
    else
        echo "⚠ Warning: Insufficient TFRecord files found"
        echo "  Train files: $train_files (expected > 0)"
        echo "  Test files: $test_files (expected > 0)"
        echo "  Skipping Das dataset training"
    fi
else
    echo "⚠ Warning: Das TFRecord directory not found: $DAS_TFRECORD_DIR"
    echo "  Skipping Das dataset training"
fi

# ============================================================================
# 3. Fulsang Dataset Training
# ============================================================================
echo ""
echo "=================================================================================="
echo "CHECKING FULSANG DATASET"
echo "=================================================================================="

# Check for Fulsang TFRecord directory (default: fulsang_preprocessed/tfrecords)
if [ -d "$FULSANG_TFRECORD_DIR" ]; then
    # Check if it has train/test subdirectories or files directly
    if [ -d "$FULSANG_TFRECORD_DIR/train" ] && [ -d "$FULSANG_TFRECORD_DIR/test" ]; then
        train_files=$(find "$FULSANG_TFRECORD_DIR/train" -name "*.tfrecords" 2>/dev/null | wc -l)
        test_files=$(find "$FULSANG_TFRECORD_DIR/test" -name "*.tfrecords" 2>/dev/null | wc -l)
    else
        # Check for files directly in the directory
        train_files=$(find "$FULSANG_TFRECORD_DIR" -name "*.tfrecords" 2>/dev/null | wc -l)
        test_files=0
    fi
    
    if [ "$train_files" -gt 0 ] || [ "$test_files" -gt 0 ]; then
        echo "✓ Fulsang TFRecord files found:"
        echo "  Directory: $FULSANG_TFRECORD_DIR"
        if [ "$train_files" -gt 0 ]; then
            echo "  Train files: $train_files"
        fi
        if [ "$test_files" -gt 0 ]; then
            echo "  Test files: $test_files"
        fi
        
        # Run Fulsang dataset training
        run_training "fulsang" \
            "stanetcnn_fulsang_results" \
            "stanetcnn_fulsang_training.log" \
            --fulsang_tfrecord_dir "$FULSANG_TFRECORD_DIR"
        
        if [ $? -eq 0 ]; then
            FULSANG_SUCCESS=1
            echo "✓ Fulsang dataset training completed successfully"
        else
            echo "✗ Fulsang dataset training failed"
            echo "  Check stanetcnn_fulsang_training.log for details"
        fi
    else
        echo "⚠ Warning: No TFRecord files found in $FULSANG_TFRECORD_DIR"
        echo "  Skipping Fulsang dataset training"
    fi
else
    echo "⚠ Warning: Fulsang TFRecord directory not found: $FULSANG_TFRECORD_DIR"
    echo "  Default path: fulsang_preprocessed/tfrecords"
    echo "  Set FULSANG_TFRECORD_DIR environment variable to use a different path"
    echo "  Skipping Fulsang dataset training"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=================================================================================="
echo "TRAINING SUMMARY"
echo "=================================================================================="
echo "Finished at: $(date)"
echo ""
echo "Results:"
echo "  Combined Dataset: $([ $COMBINED_SUCCESS -eq 1 ] && echo '✓ SUCCESS' || echo '✗ FAILED/SKIPPED')"
echo "  Das Dataset:      $([ $DAS_SUCCESS -eq 1 ] && echo '✓ SUCCESS' || echo '✗ FAILED/SKIPPED')"
echo "  Fulsang Dataset:  $([ $FULSANG_SUCCESS -eq 1 ] && echo '✓ SUCCESS' || echo '✗ FAILED/SKIPPED')"
echo ""
echo "Output directories:"
[ $COMBINED_SUCCESS -eq 1 ] && echo "  stanetcnn_combined_results/"
[ $DAS_SUCCESS -eq 1 ] && echo "  stanetcnn_das_results/"
[ $FULSANG_SUCCESS -eq 1 ] && echo "  stanetcnn_fulsang_results/"
echo ""
echo "Log files:"
[ $COMBINED_SUCCESS -eq 1 ] && echo "  stanetcnn_combined_training.log"
[ $DAS_SUCCESS -eq 1 ] && echo "  stanetcnn_das_training.log"
[ $FULSANG_SUCCESS -eq 1 ] && echo "  stanetcnn_fulsang_training.log"
echo "=================================================================================="

# Exit with error if all trainings failed
if [ $COMBINED_SUCCESS -eq 0 ] && [ $DAS_SUCCESS -eq 0 ] && [ $FULSANG_SUCCESS -eq 0 ]; then
    echo ""
    echo "✗ ERROR: All dataset trainings failed or were skipped"
    exit 1
fi

exit 0

