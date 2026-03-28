#!/bin/bash
#SBATCH --job-name=eeg_trial_plots
#SBATCH --output=eeg_trial_plots_%j.out
#SBATCH --error=eeg_trial_plots_%j.err
#SBATCH --time=24:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=tier3
#SBATCH --account=neurosteer

# Run plot_all_subjects_eeg_trials.py from the repo root (Das + Fulsang raw + preproc).
#
# Usage (run sbatch from the repo directory so SLURM_SUBMIT_DIR points here):
#   cd /path/to/telluride_decoding && sbatch run_plot_all_subjects_eeg_trials.sh
#   bash run_plot_all_subjects_eeg_trials.sh
#   bash run_plot_all_subjects_eeg_trials.sh --max-trials 2 --verbose
#   PLOT_OUT=/path/to/plots sbatch run_plot_all_subjects_eeg_trials.sh
#
# Optional environment variables:
#   PLOT_OUT       Output directory for PNGs (default: <repo>/eeg_trial_figures_3stage)
#   PYTHON         Python interpreter (default: python3)

set -e

# Slurm runs a *copy* of this script under /var/spool/slurmd/... so BASH_SOURCE is not the repo.
# SLURM_SUBMIT_DIR is the cwd where `sbatch` was run (your checkout).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  REPO_DIR="${SLURM_SUBMIT_DIR}"
else
  REPO_DIR="${SCRIPT_DIR}"
fi
cd "${REPO_DIR}"

if [ ! -f "${REPO_DIR}/plot_all_subjects_eeg_trials.py" ]; then
  echo "ERROR: plot_all_subjects_eeg_trials.py not found in: ${REPO_DIR}"
  echo "  cd to the telluride_decoding repo, then: sbatch run_plot_all_subjects_eeg_trials.sh"
  exit 1
fi

PYTHON="${PYTHON:-python3}"
PLOT_OUT="${PLOT_OUT:-${REPO_DIR}/eeg_trial_figures_3stage}"

echo "Repo:     ${REPO_DIR}"
echo "Python:   ${PYTHON}"
echo "Plot out: ${PLOT_OUT}"
echo "Started:  $(date)"
echo "Job ID:   ${SLURM_JOB_ID:-N/A}"
echo "Node:     ${SLURM_NODELIST:-$(hostname)}"
echo "================================================================================"

"${PYTHON}" plot_all_subjects_eeg_trials.py --plot-out "${PLOT_OUT}" "$@"

code=$?
echo "================================================================================"
echo "Finished: $(date)"
echo "Exit code: ${code}"
exit "${code}"
