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
#   FULSANG_RAW_DIR, FULSANG_AUDIO_DIR, FULSANG_MWF_DIR, FULSANG_EXPINFO_DIR,
#   COMBINED_DATASET_DIR
#
# Dataset subset (matches CombinedCCA.py --dataset_subset):
#   DATASET_SUBSET=full|das|fulsang   default full; use das for Das-only CCA
#
# Preprocessing / envelope (optional overrides):
#   BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ, BANDPASS_ORDER  (defaults 2, 8, 1)
#   MIN_LAG_MS, MAX_LAG_MS, EEG_LAG_TAPS
#   ENVELOPE_NORMALIZE=zscore|scale_only
#   CCA_BOTH_ENVELOPES=1            -> --cca_both_envelopes
#   NO_USE_HILBERT_ENVELOPE=1       -> --no_use_hilbert_envelope
#   NO_USE_GAMMATONE_FILTER=1       -> --no_use_gammatone_filter
#   NO_BALANCE_ENVELOPE_ENERGY=1    -> --no_balance_envelope_energy
#
# Optional training window subsample (reduces sliding-window redundancy):
#   MAX_TRAIN_WINDOWS=N   -> --max_train_windows N (unset = use all train windows)

# -------- Optional SLURM directives (ignored when run locally) --------
#SBATCH --job-name=combined_cca
#SBATCH --output=combined_cca_%j.out
#SBATCH --error=combined_cca_%j.err
#SBATCH --time=05:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Do not request GPU (CombinedCCA.py uses CPU only); avoids ReqNodeNotAvail when GPU nodes are down
# #SBATCH --gres=gpu:1
#SBATCH --partition=tier3
#SBATCH --account=neurosteer

set -e

echo "==============================================================================="
echo "RUNNING CombinedCCA.py (Combined Das + Fulsang CCA)"
echo "Started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "==============================================================================="

# Ensure gammatone is available for envelope extraction (installed only if missing)
# Set SKIP_GAMMATONE_INSTALL=1 to skip this step.
if [ "${SKIP_GAMMATONE_INSTALL:-0}" = "1" ] || [ "${SKIP_GAMMATONE_INSTALL:-0}" = "true" ]; then
    echo "Skipping gammatone installation check (SKIP_GAMMATONE_INSTALL=${SKIP_GAMMATONE_INSTALL})"
else
    if python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('gammatone') else 1)" >/dev/null 2>&1; then
        echo "✓ Python package 'gammatone' is already installed."
    else
        echo "⚠ Python package 'gammatone' not found. Installing..."
        if python -m pip install --user gammatone; then
            echo "✓ Installed gammatone."
        else
            echo "✗ Failed to install gammatone. Continuing without it (fallback envelope extraction will be used)."
        fi
    fi
fi

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
DAS_PREPROCESSING_TYPE="${DAS_PREPROCESSING_TYPE:-COMBINED_DAS}"
# Use relative paths so same script works on any machine (run from repo root)
FULSANG_RAW_DIR="${FULSANG_RAW_DIR:-Data/Fulsang}"
FULSANG_AUDIO_DIR="${FULSANG_AUDIO_DIR:-Data/Fulsang/AUDIO}"
FULSANG_MWF_DIR="${FULSANG_MWF_DIR:-MWF_cleaned_Fuglsang}"
FULSANG_EXPINFO_DIR="${FULSANG_EXPINFO_DIR:-Exp_Info}"
COMBINED_DATASET_DIR="${COMBINED_DATASET_DIR:-combined_dataset}"

# Paper-style defaults: 64 Hz, 8 s window, bandpass 2–8 Hz (speech-brain), gammatone+Hilbert in Python by default
TARGET_SAMPLING_RATE="${TARGET_SAMPLING_RATE:-64}"
WINDOW_SEC="${WINDOW_SEC:-8}"
WINDOW_SIZE="${WINDOW_SIZE:-512}"
OVERLAP="${OVERLAP:-0.5}"
BANDPASS_LOW_HZ="${BANDPASS_LOW_HZ:-2.0}"
BANDPASS_HIGH_HZ="${BANDPASS_HIGH_HZ:-8.0}"
BANDPASS_ORDER="${BANDPASS_ORDER:-1}"
MIN_LAG_MS="${MIN_LAG_MS:-0.0}"
MAX_LAG_MS="${MAX_LAG_MS:-300.0}"
EEG_LAG_TAPS="${EEG_LAG_TAPS:-12}"
DATASET_SUBSET="${DATASET_SUBSET:-full}"
ENVELOPE_NORMALIZE="${ENVELOPE_NORMALIZE:-scale_only}"
CCA_DIMS=16
REGULARIZATION=0.01
BATCH_SIZE=6
OUTPUT_DIR="combined_cca_results"
MAX_TRAIN_WINDOWS="${MAX_TRAIN_WINDOWS:-}"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/combined_cca_$(date +%Y%m%d_%H%M%S).log"

MAX_TRAIN_WIN_ARGS=""
if [ -n "$MAX_TRAIN_WINDOWS" ]; then
    MAX_TRAIN_WIN_ARGS="--max_train_windows $MAX_TRAIN_WINDOWS"
fi

# Optional flags (set env to 1/true/yes to enable)
CCA_EXTRA_ARGS=""
if [ "${CCA_BOTH_ENVELOPES:-0}" = "1" ] || [ "${CCA_BOTH_ENVELOPES:-0}" = "true" ] || [ "${CCA_BOTH_ENVELOPES:-0}" = "yes" ]; then
    CCA_EXTRA_ARGS="$CCA_EXTRA_ARGS --cca_both_envelopes"
fi
if [ "${NO_USE_HILBERT_ENVELOPE:-0}" = "1" ] || [ "${NO_USE_HILBERT_ENVELOPE:-0}" = "true" ] || [ "${NO_USE_HILBERT_ENVELOPE:-0}" = "yes" ]; then
    CCA_EXTRA_ARGS="$CCA_EXTRA_ARGS --no_use_hilbert_envelope"
fi
if [ "${NO_USE_GAMMATONE_FILTER:-0}" = "1" ] || [ "${NO_USE_GAMMATONE_FILTER:-0}" = "true" ] || [ "${NO_USE_GAMMATONE_FILTER:-0}" = "yes" ]; then
    CCA_EXTRA_ARGS="$CCA_EXTRA_ARGS --no_use_gammatone_filter"
fi
if [ "${NO_BALANCE_ENVELOPE_ENERGY:-0}" = "1" ] || [ "${NO_BALANCE_ENVELOPE_ENERGY:-0}" = "true" ] || [ "${NO_BALANCE_ENVELOPE_ENERGY:-0}" = "yes" ]; then
    CCA_EXTRA_ARGS="$CCA_EXTRA_ARGS --no_balance_envelope_energy"
fi

echo "Using configuration:"
echo "  DAS_DATA_DIR          = $DAS_DATA_DIR"
echo "  DAS_ORIGINAL_DIR      = $DAS_ORIGINAL_DIR"
echo "  DAS_AUDIO_DIR         = $DAS_AUDIO_DIR"
echo "  DAS_PREPROCESSING_TYPE= $DAS_PREPROCESSING_TYPE"
echo "  FULSANG_RAW_DIR       = $FULSANG_RAW_DIR"
echo "  FULSANG_AUDIO_DIR     = $FULSANG_AUDIO_DIR"
echo "  FULSANG_MWF_DIR       = $FULSANG_MWF_DIR"
echo "  FULSANG_EXPINFO_DIR   = $FULSANG_EXPINFO_DIR"
echo "  COMBINED_DATASET_DIR  = $COMBINED_DATASET_DIR"
echo "  DATASET_SUBSET        = $DATASET_SUBSET"
echo "  TARGET_SAMPLING_RATE  = $TARGET_SAMPLING_RATE Hz"
echo "  WINDOW_SEC            = $WINDOW_SEC s"
echo "  WINDOW_SIZE           = $WINDOW_SIZE samples"
echo "  OVERLAP               = $OVERLAP"
echo "  BANDPASS              = ${BANDPASS_LOW_HZ}-${BANDPASS_HIGH_HZ} Hz (order $BANDPASS_ORDER)"
echo "  LAG_MS                = ${MIN_LAG_MS}-${MAX_LAG_MS}, EEG_LAG_TAPS=$EEG_LAG_TAPS"
echo "  ENVELOPE_NORMALIZE    = $ENVELOPE_NORMALIZE"
echo "  CCA_DIMS              = $CCA_DIMS"
echo "  REGULARIZATION        = $REGULARIZATION"
echo "  BATCH_SIZE            = $BATCH_SIZE"
echo "  OUTPUT_DIR            = $OUTPUT_DIR"
echo "  MAX_TRAIN_WINDOWS     = ${MAX_TRAIN_WINDOWS:-<unset, all train windows>}"
echo "  CCA_EXTRA_ARGS        = ${CCA_EXTRA_ARGS:-<none>}"
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
    --fulsang_expinfo_dir "$FULSANG_EXPINFO_DIR" \
    --fulsang_mwf_dir "$FULSANG_MWF_DIR" \
    --combined_dataset_dir "$COMBINED_DATASET_DIR" \
    --target_sampling_rate "$TARGET_SAMPLING_RATE" \
    --window_sec "$WINDOW_SEC" \
    --window_size "$WINDOW_SIZE" \
    --overlap "$OVERLAP" \
    --bandpass_low_hz "$BANDPASS_LOW_HZ" \
    --bandpass_high_hz "$BANDPASS_HIGH_HZ" \
    --bandpass_order "$BANDPASS_ORDER" \
    --min_lag_ms "$MIN_LAG_MS" \
    --max_lag_ms "$MAX_LAG_MS" \
    --eeg_lag_taps "$EEG_LAG_TAPS" \
    --envelope_normalize "$ENVELOPE_NORMALIZE" \
    --dataset_subset "$DATASET_SUBSET" \
    --cca_dims "$CCA_DIMS" \
    --regularization "$REGULARIZATION" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$OUTPUT_DIR" \
    $MAX_TRAIN_WIN_ARGS \
    $CCA_EXTRA_ARGS \
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


