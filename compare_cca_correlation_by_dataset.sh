#!/bin/bash
# compare_cca_correlation_by_dataset.sh - Run CCA rho1 comparison (Das only vs Fulsang only vs Combined)
#
# Usage (local):
#   bash compare_cca_correlation_by_dataset.sh
#
# Usage (SLURM):
#   sbatch compare_cca_correlation_by_dataset.sh
#
# Override paths via environment (optional):
#   COMBINED_DATASET_DIR, DAS_DATA_DIR, FULSANG_RAW_DIR, FULSANG_AUDIO_DIR, FULSANG_MWF_DIR, EXPINFO_DIR
# Override run size:
#   MAX_WINDOWS_PER_GROUP  (default 2000; more data helps CCA)
#   MAX_WINDOWS_TOTAL     (default 5000)

#SBATCH --job-name=cca_compare
#SBATCH --output=cca_compare_%j.out
#SBATCH --error=cca_compare_%j.err
#SBATCH --time=02:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=tier3
#SBATCH --account=neurosteer

set -e

echo "==============================================================================="
echo "CCA rho1 comparison: Das only | Fulsang only | Combined"
echo "Started at: $(date)"
echo "==============================================================================="

COMBINED_DATASET_DIR="${COMBINED_DATASET_DIR:-combined_dataset}"
DAS_DATA_DIR="${DAS_DATA_DIR:-das_16subjects_preprocessed}"
DAS_ORIGINAL_DIR="${DAS_ORIGINAL_DIR:-Data/Das/4004271}"
DAS_AUDIO_DIR="${DAS_AUDIO_DIR:-Data/Das/4004271/stimuli/stimuli}"
FULSANG_RAW_DIR="${FULSANG_RAW_DIR:-Data/Fulsang}"
FULSANG_AUDIO_DIR="${FULSANG_AUDIO_DIR:-Data/Fulsang/AUDIO}"
FULSANG_MWF_DIR="${FULSANG_MWF_DIR:-MWF_cleaned_Fuglsang}"
EXPINFO_DIR="${EXPINFO_DIR:-Exp_Info}"
MAX_WINDOWS_PER_GROUP="${MAX_WINDOWS_PER_GROUP:-2000}"
MAX_WINDOWS_TOTAL="${MAX_WINDOWS_TOTAL:-5000}"

python compare_cca_correlation_by_dataset.py \
    --use_hilbert_envelope \
    --combined_dataset_dir "$COMBINED_DATASET_DIR" \
    --das_data_dir "$DAS_DATA_DIR" \
    --das_original_dir "$DAS_ORIGINAL_DIR" \
    --das_audio_dir "$DAS_AUDIO_DIR" \
    --fulsang_raw_dir "$FULSANG_RAW_DIR" \
    --fulsang_audio_dir "$FULSANG_AUDIO_DIR" \
    --fulsang_mwf_dir "$FULSANG_MWF_DIR" \
    --fulsang_expinfo_dir "$EXPINFO_DIR" \
    --max_windows_per_group "$MAX_WINDOWS_PER_GROUP" \
    --max_windows_total "$MAX_WINDOWS_TOTAL"

EXIT_CODE=$?
echo "==============================================================================="
echo "Finished at: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Done. Check the rho1 table above."
else
    echo "Exit code: $EXIT_CODE"
fi
echo "==============================================================================="
exit $EXIT_CODE
