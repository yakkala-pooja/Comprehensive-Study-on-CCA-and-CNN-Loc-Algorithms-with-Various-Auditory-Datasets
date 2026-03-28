#!/bin/bash
#SBATCH --job-name=mwfcca
#SBATCH --output=mwfcca_%j.out
#SBATCH --error=mwfcca_%j.err
#SBATCH --time=8:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=tier3
#SBATCH --account=neurosteer

# MWFCCA - Canonical Correlation Analysis for Combined Das and Fuglsang MWF-Cleaned Datasets
# This script runs CCA analysis on combined MWF-cleaned data from both datasets.

echo "=================================================================================="
echo "MWFCCA - COMBINED DAS AND FUGLSANG MWF-CLEANED DATA"
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
}

trap timeout_handler SIGUSR1

# Check data availability
echo "=================================================================================="
echo "CHECKING MWF-CLEANED DATA AVAILABILITY"
echo "=================================================================================="

DAS_MWF_DIR="MWF_cleaned_DAS"
FUGLSANG_MWF_DIR="MWF_cleaned_Fuglsang"

if [ -d "$DAS_MWF_DIR" ]; then
    das_files=$(find "$DAS_MWF_DIR" -name "S*_MWF.mat" 2>/dev/null | wc -l)
    echo "✓ Das MWF-cleaned data directory found: $DAS_MWF_DIR"
    echo "  Found $das_files MWF-cleaned files"
    
    if [ "$das_files" -eq 0 ]; then
        echo "⚠ WARNING: No MWF-cleaned Das files found!"
        echo "  Expected files: S1_MWF.mat, S2_MWF.mat, etc."
        echo ""
        echo "=================================================================================="
        echo "AUTOMATICALLY RUNNING MWF PROCESSING FOR DAS DATASET"
        echo "=================================================================================="
        python3 mwf_artifact_removal.py --dataset das --unified > das_mwf_processing.log 2>&1
        if [ $? -eq 0 ]; then
            das_files=$(find "$DAS_MWF_DIR" -name "S*_MWF.mat" 2>/dev/null | wc -l)
            echo "✓ Das MWF processing completed: $das_files files created"
        else
            echo "✗ Das MWF processing failed. Check das_mwf_processing.log"
            exit 1
        fi
    fi
else
    echo "✗ Das MWF-cleaned data directory not found at $DAS_MWF_DIR"
    echo ""
    echo "=================================================================================="
    echo "AUTOMATICALLY RUNNING MWF PROCESSING FOR DAS DATASET"
    echo "=================================================================================="
    mkdir -p "$DAS_MWF_DIR"
    python3 mwf_artifact_removal.py --dataset das --unified > das_mwf_processing.log 2>&1
    if [ $? -eq 0 ]; then
        das_files=$(find "$DAS_MWF_DIR" -name "S*_MWF.mat" 2>/dev/null | wc -l)
        echo "✓ Das MWF processing completed: $das_files files created"
    else
        echo "✗ Das MWF processing failed. Check das_mwf_processing.log"
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
das_files=$(find "$DAS_MWF_DIR" -name "S*_MWF.mat" 2>/dev/null | wc -l)
fuglsang_files=$(find "$FUGLSANG_MWF_DIR" -name "sub*_MWF.mat" 2>/dev/null | wc -l)

if [ "$das_files" -eq 0 ] || [ "$fuglsang_files" -eq 0 ]; then
    echo ""
    echo "✗ ERROR: Insufficient MWF-cleaned data found"
    echo "  Das files: $das_files"
    echo "  Fuglsang files: $fuglsang_files"
    echo ""
    echo "Please ensure MWF processing has completed successfully before running MWFCCA"
    exit 1
fi

echo ""
echo "✓ All MWF-cleaned data available:"
echo "  Das: $das_files files"
echo "  Fuglsang: $fuglsang_files files"

# Run MWFCCA analysis
echo ""
echo "=================================================================================="
echo "RUNNING MWFCCA ANALYSIS"
echo "=================================================================================="

if [ ! -f "MWFCCA.py" ]; then
    echo "✗ MWFCCA.py not found!"
    exit 1
fi

python3 MWFCCA.py \
    --das_mwf_dir MWF_cleaned_DAS \
    --fuglsang_mwf_dir MWF_cleaned_Fuglsang \
    --n_components 10 \
    --output_dir mwfcca_results > mwfcca_analysis.log 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "=================================================================================="
    echo "MWFCCA ANALYSIS COMPLETED SUCCESSFULLY!"
    echo "Finished at: $(date)"
    echo "=================================================================================="
    
    if [ -d "mwfcca_results" ]; then
        echo "✓ Results saved to: mwfcca_results/"
        if [ -f "mwfcca_results/mwfcca_results.json" ]; then
            echo "✓ Results file: mwfcca_results.json"
        fi
    fi
    
    echo ""
    echo "🎉 SUCCESS: MWFCCA analysis completed!"
    exit 0
else
    echo "=================================================================================="
    echo "MWFCCA ANALYSIS FAILED with exit code: $EXIT_CODE"
    echo "Check the error log: mwfcca_analysis.log"
    echo "=================================================================================="
    tail -20 mwfcca_analysis.log
    exit $EXIT_CODE
fi

