#!/bin/bash
# CombinedCCA.sh - Run CombinedCCA.py (Combined Das + Fulsang CCA)
#
# This script automatically runs preprocessing before CCA training:
#   1. Das 16-subject preprocessing (creates TFRecord files with audio mapping)
#   2. Das MWF audio file mapping (adds audio paths to existing MWF files)
#   3. Fulsang MWF processing (creates MWF-cleaned files)
#
# Usage (SLURM cluster):
#   sbatch CombinedCCA.sh
#
# Usage (local bash, no SLURM):
#   bash CombinedCCA.sh
#
# You can override the default data locations by exporting these
# environment variables before running:
#   DAS_DATA_DIR, DAS_ORIGINAL_DIR, DAS_AUDIO_DIR, DAS_PREPROCESSING_TYPE,
#   FULSANG_RAW_DIR, FULSANG_AUDIO_DIR, FULSANG_MWF_DIR

# -------- Optional SLURM directives (ignored when run locally) --------
#SBATCH --job-name=combined_cca
#SBATCH --output=combined_cca_%j.out
#SBATCH --error=combined_cca_%j.err
#SBATCH --time=24:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=tier3
#SBATCH --account=neurosteer

set -e

echo "==============================================================================="
echo "RUNNING CombinedCCA.py (Combined Das + Fulsang CCA)"
echo "Started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "==============================================================================="

# Load modules on the cluster (safe to ignore locally)
MODULES_LOADED=false
if command -v module >/dev/null 2>&1; then
    if [ -n "${MODULES_TO_LOAD:-}" ]; then
        for mod in $MODULES_TO_LOAD; do
            if module load "$mod"; then
                MODULES_LOADED=true
            else
                echo "Warning: module '$mod' not found"
            fi
        done
    else
        echo "Warning: MODULES_TO_LOAD not set. No modules will be loaded."
    fi
fi

# NOTE: We intentionally do NOT activate any local virtualenv here.
# If cluster modules are loaded, we also prevent user-level site-packages
# from overriding them; otherwise we leave user packages enabled.
if [ "$MODULES_LOADED" = true ]; then
    export PYTHONNOUSERSITE=1
    unset PYTHONPATH
else
    unset PYTHONNOUSERSITE
fi

# Default paths (can be overridden via environment variables)
DAS_DATA_DIR="${DAS_DATA_DIR:-das_16subjects_preprocessed}"
DAS_ORIGINAL_DIR="${DAS_ORIGINAL_DIR:-Data/Das/4004271}"
DAS_AUDIO_DIR="${DAS_AUDIO_DIR:-Data/Das/4004271/stimuli/stimuli}"
DAS_PREPROCESSING_TYPE="${DAS_PREPROCESSING_TYPE:-16SUBJECTS}"
FULSANG_RAW_DIR="${FULSANG_RAW_DIR:-/home/py9363/telluride_decoding/Data/Fulsang/EEG}"
FULSANG_AUDIO_DIR="${FULSANG_AUDIO_DIR:-/home/py9363/telluride_decoding/Data/Fulsang/AUDIO}"
FULSANG_MWF_DIR="${FULSANG_MWF_DIR:-MWF_cleaned_Fuglsang}"

# Use optimal configuration values (recommended)
WINDOW_SIZE=512
OVERLAP=0.5
CCA_DIMS=16
REGULARIZATION=0.02
BATCH_SIZE=6
OUTPUT_DIR="combined_cca_results"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/combined_cca_$(date +%Y%m%d_%H%M%S).log"

echo "Using configuration:"
echo "  DAS_DATA_DIR          = $DAS_DATA_DIR"
echo "  DAS_ORIGINAL_DIR      = $DAS_ORIGINAL_DIR"
echo "  DAS_AUDIO_DIR         = $DAS_AUDIO_DIR"
echo "  DAS_PREPROCESSING_TYPE= $DAS_PREPROCESSING_TYPE"
echo "  FULSANG_RAW_DIR       = $FULSANG_RAW_DIR"
echo "  FULSANG_AUDIO_DIR     = $FULSANG_AUDIO_DIR"
echo "  FULSANG_MWF_DIR       = $FULSANG_MWF_DIR"
echo "  WINDOW_SIZE           = $WINDOW_SIZE"
echo "  OVERLAP               = $OVERLAP"
echo "  CCA_DIMS              = $CCA_DIMS"
echo "  REGULARIZATION        = $REGULARIZATION"
echo "  BATCH_SIZE            = $BATCH_SIZE"
echo "  OUTPUT_DIR            = $OUTPUT_DIR"
echo "  LOG_FILE              = $LOG_FILE"
echo "==============================================================================="

# Run and capture both stdout and stderr to a log file (and still show on console)
python CombinedCCA.py \
    --das_data_dir "$DAS_DATA_DIR" \
    --das_original_dir "$DAS_ORIGINAL_DIR" \
    --das_audio_dir "$DAS_AUDIO_DIR" \
    --das_preprocessing_type "$DAS_PREPROCESSING_TYPE" \
    --fulsang_raw_dir "$FULSANG_RAW_DIR" \
    --fulsang_audio_dir "$FULSANG_AUDIO_DIR" \
    --fulsang_mwf_dir "$FULSANG_MWF_DIR" \
    --window_size "$WINDOW_SIZE" \
    --overlap "$OVERLAP" \
    --cca_dims "$CCA_DIMS" \
    --regularization "$REGULARIZATION" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_FILE"

# Get real Python exit code from the tee pipeline
EXIT_CODE=${PIPESTATUS[0]}

echo "==============================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ CombinedCCA.py completed successfully"
else
    echo "✗ CombinedCCA.py failed with exit code: $EXIT_CODE"
fi
echo "Finished at: $(date)"
echo "Results directory (if run completed): $OUTPUT_DIR"
echo "Full log written to: $LOG_FILE"
echo "==============================================================================="

exit $EXIT_CODE


