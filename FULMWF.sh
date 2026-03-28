#!/bin/bash
#SBATCH --job-name=fuglsang_mwf
#SBATCH --output=fuglsang_mwf_%j.out
#SBATCH --error=fuglsang_mwf_%j.err
#SBATCH --time=8:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=tier3
#SBATCH --account=neurosteer

# MWF Artifact Removal for Fuglsang Dataset Only
# This script applies Multi-channel Wiener Filtering (MWF) to remove artifacts
# from Fuglsang EEG recordings.
#
# Usage:
#   bash FULMWF.sh              # Process Fuglsang dataset
#   bash FULMWF.sh --visualize  # Include visualization

echo "=================================================================================="
echo "MWF ARTIFACT REMOVAL FOR FUGLSANG DATASET"
echo "=================================================================================="
echo "Started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=================================================================================="

# Environment setup
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
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
    
    if [ -d "MWF_cleaned_Fuglsang" ]; then
        echo "Saving Fuglsang MWF partial results..."
        cp -r MWF_cleaned_Fuglsang MWF_cleaned_Fuglsang_backup_$(date +%Y%m%d_%H%M%S) 2>/dev/null || true
    fi
}

trap timeout_handler SIGUSR1

# Check Python environment
echo "=================================================================================="
echo "CHECKING PYTHON ENVIRONMENT"
echo "=================================================================================="
python3 --version

# Check data availability
echo ""
echo "=================================================================================="
echo "CHECKING DATA AVAILABILITY"
echo "=================================================================================="
FUGLSANG_EEG_DIR="${FUGLSANG_EEG_DIR:-/home/py9363/telluride_decoding/Data/Fulsang/EEG}"

if [ -d "$FUGLSANG_EEG_DIR" ]; then
    fuglsang_files=$(find "$FUGLSANG_EEG_DIR" -name "S*.mat" 2>/dev/null | wc -l)
    echo "✓ Fuglsang dataset found: $fuglsang_files subject files"
    echo "  Path: $FUGLSANG_EEG_DIR"
else
    echo "✗ Fuglsang dataset not found at $FUGLSANG_EEG_DIR"
    exit 1
fi

# Run MWF processing
echo ""
echo "=================================================================================="
echo "RUNNING MWF ARTIFACT REMOVAL: FUGLSANG DATASET"
echo "=================================================================================="

if [ ! -f "mwf_artifact_removal.py" ]; then
    echo "✗ mwf_artifact_removal.py not found!"
    exit 1
fi

VISUALIZE_FLAG=""
if [[ "$*" == *"--visualize"* ]]; then
    VISUALIZE_FLAG="--visualize"
fi

python3 mwf_artifact_removal.py --dataset fuglsang --fuglsang_eeg_dir "$FUGLSANG_EEG_DIR" --unified $VISUALIZE_FLAG > fuglsang_mwf_processing.log 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "=================================================================================="
    echo "MWF PROCESSING COMPLETED SUCCESSFULLY!"
    echo "Finished at: $(date)"
    echo "=================================================================================="
    
    if [ -d "MWF_cleaned_Fuglsang" ]; then
        echo "Results directory: MWF_cleaned_Fuglsang"
        echo "Generated files:"
        find MWF_cleaned_Fuglsang -name "*.mat" | head -5
        echo "..."
    fi
    
    echo ""
    echo "🎉 SUCCESS: Fuglsang MWF processing completed!"
    exit 0
else
    echo "=================================================================================="
    echo "MWF PROCESSING FAILED with exit code: $EXIT_CODE"
    echo "Check the error log: fuglsang_mwf_processing.log"
    echo "=================================================================================="
    tail -20 fuglsang_mwf_processing.log
    exit $EXIT_CODE
fi

