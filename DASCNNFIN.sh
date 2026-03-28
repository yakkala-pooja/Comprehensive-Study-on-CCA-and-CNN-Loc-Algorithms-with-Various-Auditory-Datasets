#!/bin/bash
#SBATCH --job-name=dascnnfin_training
#SBATCH --output=dascnnfin_training_%j.out
#SBATCH --error=dascnnfin_training_%j.err
#SBATCH --time=24:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=tier3
#SBATCH --account=neurosteer

# DASCNNFIN.sh - Das Dataset CNN-LOC Training Script
#
# This script trains CNN-LOC model on Das dataset using TFRecord files
# from das_16subjects_preprocessed/tfrecords/train/ and /test/

set -e  # Exit on error

echo "=================================================================================="
echo "DAS CNN-LOC Training"
echo "=================================================================================="
echo "Started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "=================================================================================="

# Default paths
DAS_TFRECORD_DIR="${DAS_TFRECORD_DIR:-das_16subjects_preprocessed/tfrecords}"

# Check for Das TFRecord files
echo ""
echo "=================================================================================="
echo "CHECKING DAS TFRECORD FILES"
echo "=================================================================================="
if [ -d "$DAS_TFRECORD_DIR" ]; then
    train_files=$(find "$DAS_TFRECORD_DIR/train" -name "*.tfrecords" 2>/dev/null | wc -l)
    test_files=$(find "$DAS_TFRECORD_DIR/test" -name "*.tfrecords" 2>/dev/null | wc -l)
    
    if [ "$train_files" -gt 0 ] && [ "$test_files" -gt 0 ]; then
        echo "✓ Das TFRecord files found:"
        echo "  Train files: $train_files"
        echo "  Test files: $test_files"
    else
        echo "✗ ERROR: Insufficient TFRecord files found"
        echo "  Train files: $train_files (expected > 0)"
        echo "  Test files: $test_files (expected > 0)"
        echo "  Directory: $DAS_TFRECORD_DIR"
        exit 1
    fi
else
    echo "✗ ERROR: Das TFRecord directory not found: $DAS_TFRECORD_DIR"
    echo "  Please run preprocessing first: python3 das_preprocessing_16subjects.py"
    exit 1
fi

# Training parameters
WINDOW_SIZE=512
OVERLAP=0.5
BATCH_SIZE=32
NUM_EPOCHS=50
LEARNING_RATE=1e-3
DROPOUT_RATE=0.3

# Run CNN-LOC training
echo ""
echo "=================================================================================="
echo "RUNNING DAS CNN-LOC TRAINING"
echo "=================================================================================="
python3 DASCNNFIN.py \
    --tfrecord_dir "$DAS_TFRECORD_DIR" \
    --window_size $WINDOW_SIZE \
    --overlap $OVERLAP \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --dropout_rate $DROPOUT_RATE \
    --output_dir dascnnfin_results > dascnnfin_training.log 2>&1

TRAINING_EXIT_CODE=$?

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✓ Das CNN-LOC training completed successfully"
else
    echo "✗ Das CNN-LOC training failed with exit code: $TRAINING_EXIT_CODE"
    echo "  Check dascnnfin_training.log for details"
fi

# Summary
echo ""
echo "=================================================================================="
echo "TRAINING SUMMARY"
echo "=================================================================================="
echo "Finished at: $(date)"
echo ""
echo "Results:"
echo "  Das CNN-LOC Training: $([ $TRAINING_EXIT_CODE -eq 0 ] && echo '✓ SUCCESS' || echo '✗ FAILED')"
echo ""
echo "Output directory:"
echo "  dascnnfin_results/"
echo ""
echo "Log file:"
echo "  dascnnfin_training.log"
echo "=================================================================================="

# Exit with error if training failed
if [ $TRAINING_EXIT_CODE -ne 0 ]; then
    exit 1
fi

exit 0

