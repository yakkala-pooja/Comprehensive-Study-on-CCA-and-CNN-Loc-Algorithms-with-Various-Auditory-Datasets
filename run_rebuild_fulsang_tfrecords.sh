#!/bin/bash
#SBATCH --job-name=fulsang_rebuild_tfrecords
#SBATCH --output=fulsang_rebuild_tfrecords_%j.out
#SBATCH --error=fulsang_rebuild_tfrecords_%j.err
#SBATCH --time=04:00:00
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
echo "Rebuild Fulsang TFRecords from MWF (aligned trials/labels)"
echo "Started at: $(date)"
echo "===============================================================================" 

python rebuild_fulsang_tfrecords_from_mwf.py \
  --mwf_dir "MWF_cleaned_Fuglsang" \
  --audio_dir "Data/Fulsang/AUDIO" \
  --output_dir "fulsang_preprocessed_rebuilt" \
  --sampling_rate 64

EXIT_CODE=$?
echo "==============================================================================="
echo "Finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "==============================================================================="
exit $EXIT_CODE

