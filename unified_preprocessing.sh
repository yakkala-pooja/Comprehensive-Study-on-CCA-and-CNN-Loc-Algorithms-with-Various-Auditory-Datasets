#!/bin/bash
#SBATCH --job-name=fulcnn_cnn_loc
#SBATCH --output=fulcnn_cnn_loc_%j.out
#SBATCH --error=fulcnn_cnn_loc_%j.err
#SBATCH --time=8:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=tier3
#SBATCH --account=neurosteer

