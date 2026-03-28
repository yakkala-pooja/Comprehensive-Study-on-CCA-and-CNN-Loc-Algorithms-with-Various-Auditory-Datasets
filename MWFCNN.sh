#!/bin/bash
#SBATCH --job-name=mwfcnn
#SBATCH --output=mwfcnn_%j.out
#SBATCH --error=mwfcnn_%j.err
#SBATCH --time=8:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=tier3
#SBATCH --account=neurosteer

# MWFCNN - CNN-LOC Algorithm for Combined Das (DASPREPROCESS) and Fuglsang (MWF) Datasets
# This script trains a CNN model on combined preprocessed Das and MWF-cleaned Fuglsang data.

echo "=================================================================================="
echo "MWFCNN - COMBINED DAS (DASPREPROCESS) AND FUGLSANG (MWF) DATA"
echo "=================================================================================="
echo "Started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=================================================================================="

# Environment setup
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=8

# Timeout handler
timeout_handler() {
    echo "=================================================================================="
    echo "JOB TIMEOUT WARNING: 90% of time limit reached"
    echo "Current time: $(date)"
    echo "Attempting to save current progress..."
    echo "=================================================================================="
    
    if [ -f "mwfcnn_best_model.pth" ]; then
        echo "Saving MWFCNN model..."
        cp mwfcnn_best_model.pth mwfcnn_best_model_backup_$(date +%Y%m%d_%H%M%S).pth 2>/dev/null || true
    fi
}

trap timeout_handler SIGUSR1

# Check data availability
echo "=================================================================================="
echo "CHECKING DATA AVAILABILITY"
echo "=================================================================================="

DAS_DATA_DIR="/home/py9363/telluride_decoding/Data/Das/4004271"
DAS_PREPROCESSED_DIR="preprocessed_Das"
FUGLSANG_MWF_DIR="MWF_cleaned_Fuglsang"

# Check for raw Das data
if [ -d "$DAS_DATA_DIR" ]; then
    das_raw_files=$(find "$DAS_DATA_DIR" -name "S*.mat" -not -name "*_MWF.mat" -not -name "*_preprocessed.mat" 2>/dev/null | wc -l)
    echo "✓ Das raw data directory found: $DAS_DATA_DIR"
    echo "  Found $das_raw_files raw Das files"
    
    if [ "$das_raw_files" -eq 0 ]; then
        echo "✗ ERROR: No raw Das files found in $DAS_DATA_DIR"
        echo "  Expected files: S1.mat, S2.mat, etc."
        exit 1
    fi
else
    echo "✗ Das data directory not found at $DAS_DATA_DIR"
    exit 1
fi

# Check for preprocessed Das data or run preprocessing
if [ -d "$DAS_PREPROCESSED_DIR" ]; then
    das_files=$(find "$DAS_PREPROCESSED_DIR" -name "S*_preprocessed.mat" 2>/dev/null | wc -l)
    echo "✓ Das preprocessed data directory found: $DAS_PREPROCESSED_DIR"
    echo "  Found $das_files preprocessed files"
    
    if [ "$das_files" -eq 0 ]; then
        echo "⚠ WARNING: No preprocessed Das files found!"
        echo "  Expected files: S1_preprocessed.mat, S2_preprocessed.mat, etc."
        echo ""
        echo "=================================================================================="
        echo "AUTOMATICALLY RUNNING DASPREPROCESS FOR DAS DATASET"
        echo "=================================================================================="
        python3 unified_preprocessing.py --dataset das --das_dir "$DAS_DATA_DIR" > das_preprocessing.log 2>&1
        if [ $? -eq 0 ]; then
            das_files=$(find "$DAS_PREPROCESSED_DIR" -name "S*_preprocessed.mat" 2>/dev/null | wc -l)
            echo "✓ Das preprocessing completed: $das_files files created"
        else
            echo "✗ Das preprocessing failed. Check das_preprocessing.log"
            exit 1
        fi
    fi
else
    echo "✗ Das preprocessed data directory not found at $DAS_PREPROCESSED_DIR"
    echo ""
    echo "=================================================================================="
    echo "AUTOMATICALLY RUNNING DASPREPROCESS FOR DAS DATASET"
    echo "=================================================================================="
    mkdir -p "$DAS_PREPROCESSED_DIR"
    python3 unified_preprocessing.py --dataset das --das_dir "$DAS_DATA_DIR" > das_preprocessing.log 2>&1
    if [ $? -eq 0 ]; then
        das_files=$(find "$DAS_PREPROCESSED_DIR" -name "S*_preprocessed.mat" 2>/dev/null | wc -l)
        echo "✓ Das preprocessing completed: $das_files files created"
    else
        echo "✗ Das preprocessing failed. Check das_preprocessing.log"
        exit 1
    fi
fi

if [ -d "$FUGLSANG_MWF_DIR" ]; then
    fuglsang_files=$(find "$FUGLSANG_MWF_DIR" -name "sub*_MWF.mat" 2>/dev/null | wc -l)
    echo "✓ Fuglsang MWF-cleaned data directory found: $FUGLSANG_MWF_DIR"
    echo "  Found $fuglsang_files MWF-cleaned files"
    
    if [ "$fuglsang_files" -eq 0 ]; then
        echo "⚠ WARNING: No MWF-cleaned Fuglsang files found!"
        echo "  Expected files: sub01_MWF.mat, sub02_MWF.mat, etc."
        echo ""
        echo "=================================================================================="
        echo "AUTOMATICALLY RUNNING MWF PROCESSING FOR FUGLSANG DATASET"
        echo "=================================================================================="
        FUGLSANG_EEG_DIR="${FUGLSANG_EEG_DIR:-/home/py9363/telluride_decoding/Data/Fulsang/EEG}"
        echo "Using Fuglsang EEG directory: $FUGLSANG_EEG_DIR"
        python3 mwf_artifact_removal.py --dataset fuglsang --fuglsang_eeg_dir "$FUGLSANG_EEG_DIR" --unified > fuglsang_mwf_processing.log 2>&1
        if [ $? -eq 0 ]; then
            fuglsang_files=$(find "$FUGLSANG_MWF_DIR" -name "sub*_MWF.mat" 2>/dev/null | wc -l)
            echo "✓ Fuglsang MWF processing completed: $fuglsang_files files created"
        else
            echo "✗ Fuglsang MWF processing failed. Check fuglsang_mwf_processing.log"
            exit 1
        fi
    fi
else
    echo "✗ Fuglsang MWF-cleaned data directory not found at $FUGLSANG_MWF_DIR"
    echo ""
    echo "=================================================================================="
    echo "AUTOMATICALLY RUNNING MWF PROCESSING FOR FUGLSANG DATASET"
    echo "=================================================================================="
    mkdir -p "$FUGLSANG_MWF_DIR"
    FUGLSANG_EEG_DIR="${FUGLSANG_EEG_DIR:-/home/py9363/telluride_decoding/Data/Fulsang/EEG}"
    echo "Using Fuglsang EEG directory: $FUGLSANG_EEG_DIR"
    python3 mwf_artifact_removal.py --dataset fuglsang --fuglsang_eeg_dir "$FUGLSANG_EEG_DIR" --unified > fuglsang_mwf_processing.log 2>&1
    if [ $? -eq 0 ]; then
        fuglsang_files=$(find "$FUGLSANG_MWF_DIR" -name "sub*_MWF.mat" 2>/dev/null | wc -l)
        echo "✓ Fuglsang MWF processing completed: $fuglsang_files files created"
    else
        echo "✗ Fuglsang MWF processing failed. Check fuglsang_mwf_processing.log"
        exit 1
    fi
fi

# Final check
das_files=$(find "$DAS_PREPROCESSED_DIR" -name "S*_preprocessed.mat" 2>/dev/null | wc -l)
fuglsang_files=$(find "$FUGLSANG_MWF_DIR" -name "sub*_MWF.mat" 2>/dev/null | wc -l)

if [ "$das_files" -eq 0 ] || [ "$fuglsang_files" -eq 0 ]; then
    echo ""
    echo "✗ ERROR: Insufficient data found"
    echo "  Das preprocessed files: $das_files"
    echo "  Fuglsang MWF files: $fuglsang_files"
    echo ""
    echo "Please ensure preprocessing has completed successfully before running MWFCNN"
    exit 1
fi

echo ""
echo "✓ All data available:"
echo "  Das preprocessed: $das_files files"
echo "  Fuglsang MWF: $fuglsang_files files"

# Run MWFCNN training
echo ""
echo "=================================================================================="
echo "RUNNING MWFCNN TRAINING"
echo "=================================================================================="

if [ ! -f "MWFCNN.py" ]; then
    echo "✗ MWFCNN.py not found!"
    exit 1
fi

python3 MWFCNN.py \
    --das_preprocessed_dir preprocessed_Das \
    --fuglsang_mwf_dir MWF_cleaned_Fuglsang \
    --batch_size 32 \
    --num_epochs 50 \
    --learning_rate 1e-3 \
    --window_size 512 > mwfcnn_training.log 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "=================================================================================="
    echo "MWFCNN TRAINING COMPLETED SUCCESSFULLY!"
    echo "Finished at: $(date)"
    echo "=================================================================================="
    
    if [ -f "mwfcnn_best_model.pth" ]; then
        echo "✓ Best model saved: mwfcnn_best_model.pth"
    fi
    
    echo ""
    echo "🎉 SUCCESS: MWFCNN training completed!"
    exit 0
else
    echo "=================================================================================="
    echo "MWFCNN TRAINING FAILED with exit code: $EXIT_CODE"
    echo "Check the error log: mwfcnn_training.log"
    echo "=================================================================================="
    tail -20 mwfcnn_training.log
    exit $EXIT_CODE
fi

