#!/bin/bash
#SBATCH --job-name=optimal_fulcca
#SBATCH --output=optimal_fulcca_%j.out
#SBATCH --error=optimal_fulcca_%j.err
#SBATCH --time=4:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=tier3
#SBATCH --account=neurosteer

echo "Job $SLURM_JOB_ID started at $(date)"

module load python/3.8
module load cuda/11.2
module load gcc/9.3.0

python Optimal_FULCCA.py \
    --tfrecord_dir "/home/py9363/telluride_decoding/fulsang_preprocessed/tfrecords" \
    --output_dir "optimal_fulcca_results"

echo "Job completed at $(date)"
du -sh optimal_fulcca_results/
