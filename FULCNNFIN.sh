#!/bin/bash
#SBATCH --job-name=fulcnnfin_training
#SBATCH --output=fulcnnfin_training_%j.out
#SBATCH --error=fulcnnfin_training_%j.err
#SBATCH --time=24:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=tier3
#SBATCH --account=neurosteer

# FULCNNFIN.sh - Fulsang Dataset CNN-LOC Training Script
#
# This script trains CNN-LOC model on Fulsang dataset using TFRecord files
# from fulsang_preprocessed/tfrecords/

set -e  # Exit on error

echo "=================================================================================="
echo "FULSANG CNN-LOC Training"
echo "=================================================================================="
echo "Started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "=================================================================================="

# Default paths
FULSANG_TFRECORD_DIR="${FULSANG_TFRECORD_DIR:-fulsang_preprocessed/tfrecords}"

# Check for Fulsang TFRecord files
echo ""
echo "=================================================================================="
echo "CHECKING FULSANG TFRECORD FILES"
echo "=================================================================================="
if [ -d "$FULSANG_TFRECORD_DIR" ]; then
    tfrecord_files=$(find "$FULSANG_TFRECORD_DIR" -name "*.tfrecords" 2>/dev/null | wc -l)
    
    if [ "$tfrecord_files" -gt 0 ]; then
        echo "✓ Fulsang TFRecord files found: $tfrecord_files files"
        echo "  Directory: $FULSANG_TFRECORD_DIR"
    else
        echo "✗ ERROR: No TFRecord files found in $FULSANG_TFRECORD_DIR"
        echo "  Please run preprocessing first to generate TFRecord files"
        exit 1
    fi
else
    echo "✗ ERROR: Fulsang TFRecord directory not found: $FULSANG_TFRECORD_DIR"
    echo "  Please run preprocessing first to generate TFRecord files"
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
echo "RUNNING FULSANG CNN-LOC TRAINING"
echo "=================================================================================="
python3 FULCNNFIN.py \
    --tfrecord_dir "$FULSANG_TFRECORD_DIR" \
    --window_size $WINDOW_SIZE \
    --overlap $OVERLAP \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --dropout_rate $DROPOUT_RATE \
    --output_dir fulcnnfin_results > fulcnnfin_training.log 2>&1

TRAINING_EXIT_CODE=$?

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✓ Fulsang CNN-LOC training completed successfully"
else
    echo "✗ Fulsang CNN-LOC training failed with exit code: $TRAINING_EXIT_CODE"
    echo "  Check fulcnnfin_training.log for details"
fi

# Summary
echo ""
echo "=================================================================================="
echo "TRAINING SUMMARY"
echo "=================================================================================="
echo "Finished at: $(date)"
echo ""
echo "Results:"
echo "  Fulsang CNN-LOC Training: $([ $TRAINING_EXIT_CODE -eq 0 ] && echo '✓ SUCCESS' || echo '✗ FAILED')"
echo ""
echo "Output directory:"
echo "  fulcnnfin_results/"
echo ""
echo "Log file:"
echo "  fulcnnfin_training.log"
echo "=================================================================================="

# Exit with error if training failed
if [ $TRAINING_EXIT_CODE -ne 0 ]; then
    exit 1
fi

exit 0

