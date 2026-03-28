#!/bin/bash
#SBATCH --job-name=paper_cca
#SBATCH --output=paper_cca_%j.out
#SBATCH --error=paper_cca_%j.err
#SBATCH --time=24:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=tier3
#SBATCH --account=neurosteer
# Start in the real repo — SLURM runs a *copy* of this script under /var/spool/slurmd/...,
# so $(dirname "$0") there is NOT your checkout. Comment out if your Slurm is too old for --chdir.
#SBATCH --chdir=/home/py9363/telluride_decoding

# PaperCCA.sh — SLURM job for PaperCCA.py (Das-only by default; see env vars below).
# Submit from repo:  cd /path/to/telluride_decoding && sbatch PaperCCA.sh
# Or rely on #SBATCH --chdir above. Override checkout: export REPO_ROOT=/path/to/repo
# Local:              bash PaperCCA.sh

set -euo pipefail

# Resolve project root (Slurm spool breaks dirname "$0" — see above).
DEFAULT_REPO="/home/py9363/telluride_decoding"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

if [ -n "${REPO_ROOT:-}" ]; then
  cd "$REPO_ROOT"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/PaperCCA.py" ]; then
  cd "$SLURM_SUBMIT_DIR"
elif [ -f "${DEFAULT_REPO}/PaperCCA.py" ]; then
  cd "$DEFAULT_REPO"
else
  cd "$SCRIPT_DIR"
fi

if [ ! -f "./PaperCCA.py" ]; then
  echo "ERROR: PaperCCA.py not found in $(pwd)"
  echo "  Set REPO_ROOT to your telluride_decoding clone, or submit with: cd /path/to/telluride_decoding && sbatch PaperCCA.sh"
  echo "  If #SBATCH --chdir is wrong for this cluster, edit PaperCCA.sh or unset a bad SLURM_SUBMIT_DIR."
  exit 1
fi

echo "Working directory: $(pwd)"

echo "=================================================================================="
echo "PaperCCA — Das-only CCA + LDA"
echo "=================================================================================="
echo "Started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "=================================================================================="

export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

# Das source: same priority as DASCCA — das_16subjects_preprocessed/tfrecords first, then
# das_combined_preprocessed/tfrecords, then MWF_cleaned_DAS.
# Override: export DAS_DATA_DIR / DAS_PREPROCESSING_TYPE / DAS_MWF_DIR before sbatch.
DAS_RAW_DIR="${DAS_RAW_DIR:-/home/py9363/telluride_decoding/Data/Das/4004271}"
DAS_AUDIO_DIR="${DAS_AUDIO_DIR:-$DAS_RAW_DIR/stimuli/stimuli}"

# NOTE: Do not use `find ... | head -1` with `set -o pipefail`: find exits 141 on SIGPIPE and the
# script aborts under `set -e` before any error message. Use find -print -quit (GNU find) instead.
echo "Detecting Das data source..."

if [ -z "${DAS_PREPROCESSING_TYPE+x}" ]; then
  _first_16=""
  if [ -d das_16subjects_preprocessed/tfrecords ]; then
    _first_16=$(find das_16subjects_preprocessed/tfrecords \( -name "*.tfrecords" -o -name "*.tfrecord" \) -print -quit 2>/dev/null || true)
  fi
  _first_comb=""
  if [ -d das_combined_preprocessed/tfrecords ]; then
    _first_comb=$(find das_combined_preprocessed/tfrecords \( -name "*.tfrecords" -o -name "*.tfrecord" \) -print -quit 2>/dev/null || true)
  fi
  _mwf_root="${DAS_MWF_DIR:-MWF_cleaned_DAS}"
  _first_mwf=""
  if [ -d "$_mwf_root" ]; then
    _first_mwf=$(find "$_mwf_root" -maxdepth 1 -name 'S*_MWF.mat' -print -quit 2>/dev/null || true)
  fi
  if [ -n "$_first_16" ]; then
    DAS_PREPROCESSING_TYPE=COMBINED_DAS
    DAS_DATA_DIR="${DAS_DATA_DIR:-das_16subjects_preprocessed}"
  elif [ -n "$_first_comb" ]; then
    DAS_PREPROCESSING_TYPE=COMBINED_DAS
    DAS_DATA_DIR="${DAS_DATA_DIR:-das_combined_preprocessed}"
  elif [ -n "$_first_mwf" ]; then
    DAS_PREPROCESSING_TYPE=MWF
    DAS_MWF_DIR="$_mwf_root"
    DAS_DATA_DIR="${DAS_DATA_DIR:-das_16subjects_preprocessed}"
  else
    echo "ERROR: No Das data found."
    echo "  Expected (in order):"
    echo "    - das_16subjects_preprocessed/tfrecords/*.tfrecords  (same as DASCCA.py default)"
    echo "    - das_combined_preprocessed/tfrecords/"
    echo "    - S*_MWF.mat under ${DAS_MWF_DIR:-MWF_cleaned_DAS}"
    echo "  Run: python3 das_preprocessing_16subjects.py ...  OR  das_preprocessing_combined.py  OR  mwf_artifact_removal.py --dataset das --unified"
    exit 1
  fi
else
  DAS_DATA_DIR="${DAS_DATA_DIR:-das_16subjects_preprocessed}"
  DAS_MWF_DIR="${DAS_MWF_DIR:-MWF_cleaned_DAS}"
fi

MWF_ARG=()
if [ "$DAS_PREPROCESSING_TYPE" = "MWF" ] && [ -n "${DAS_MWF_DIR:-}" ]; then
  MWF_ARG=( --das-mwf-dir "$DAS_MWF_DIR" )
fi

echo ""
echo "Paths:"
echo "  DAS_DATA_DIR           = $DAS_DATA_DIR"
echo "  DAS_PREPROCESSING_TYPE = $DAS_PREPROCESSING_TYPE"
echo "  DAS_MWF_DIR            = ${DAS_MWF_DIR:-}"
echo "  DAS_RAW_DIR (original) = $DAS_RAW_DIR"
echo "  DAS_AUDIO_DIR          = $DAS_AUDIO_DIR"
echo ""

python3 PaperCCA.py \
  --das-only \
  --das-data-dir "$DAS_DATA_DIR" \
  --das-preprocessing-type "$DAS_PREPROCESSING_TYPE" \
  --das-original-dir "$DAS_RAW_DIR" \
  --das-audio-dir "$DAS_AUDIO_DIR" \
  "${MWF_ARG[@]}" \
  --use-gammatone \
  --fs-intermediate 128 \
  --fs-cca 20 \
  --eeg-bp-general-low 1 \
  --eeg-bp-general-high 32 \
  --eeg-bp-linear-low 1 \
  --eeg-bp-linear-high 9 \
  --compress-power 0.6 \
  --eeg-lag-ms 350 \
  --encoder-sec 1.25 \
  --pca-components 0 \
  --j-candidates "2,3,4,5,6,8,10" \
  --window-seconds "1,5,10,30" \
  --outer-mode trial \
  --inner-val-frac 0.2 \
  --output-json paper_cca_results.json \
  "$@"

echo ""
echo "Finished at: $(date)"
