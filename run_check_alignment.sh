#!/bin/bash
#SBATCH --job-name=check_alignment
#SBATCH --output=check_alignment_%j.out
#SBATCH --error=check_alignment_%j.err
#SBATCH --time=02:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=tier3
#SBATCH --account=neurosteer

set -e

echo "==============================================================================="
echo "EEG–envelope alignment / mismatch check"
echo "Started at: $(date)"
echo "==============================================================================="

python check_alignment.py

EXIT_CODE=$?
echo "==============================================================================="
echo "Finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "==============================================================================="
exit $EXIT_CODE

