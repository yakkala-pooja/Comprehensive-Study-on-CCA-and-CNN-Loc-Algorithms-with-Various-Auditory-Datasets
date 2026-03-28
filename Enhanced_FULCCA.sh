#!/bin/bash
#SBATCH --job-name=enhanced_fulcca
#SBATCH --output=enhanced_fulcca_%j.out
#SBATCH --error=enhanced_fulcca_%j.err
#SBATCH --time=12:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=tier3
#SBATCH --account=neurosteer

echo "=========================================="
echo "Enhanced FULCCA - Optimized for 68%+ Accuracy"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=========================================="

# Load modules
echo "Loading modules..."
module load python/3.8
module load cuda/11.2
module load gcc/9.3.0

# Check Python and TensorFlow versions
echo "Python version: $(python --version)"
echo "TensorFlow version: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo "NumPy version: $(python -c 'import numpy as np; print(np.__version__)')"

echo "=========================================="
echo "Running Enhanced FULCCA Analysis"
echo "=========================================="

# Set TFRecord directory
TFRECORD_DIR="/home/py9363/telluride_decoding/fulsang_preprocessed/tfrecords"

# Run enhanced FULCCA analysis
echo "Starting enhanced FULCCA analysis..."
python Enhanced_FULCCA.py \
    --tfrecord_dir "$TFRECORD_DIR" \
    --output_dir "enhanced_fulcca_results"

echo "=========================================="
echo "Enhanced FULCCA Analysis Complete"
echo "=========================================="
echo "End time: $(date)"
echo "Job duration: $SECONDS seconds"
echo "Results saved to: enhanced_fulcca_results/"
echo "=========================================="

# Check disk usage
echo "Disk usage:"
du -sh enhanced_fulcca_results/

# Check GPU memory usage
echo "GPU memory usage:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits

echo "Enhanced FULCCA job completed successfully!"
