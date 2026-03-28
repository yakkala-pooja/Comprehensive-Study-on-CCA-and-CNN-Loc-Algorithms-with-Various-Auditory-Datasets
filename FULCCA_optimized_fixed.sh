#!/bin/bash
#SBATCH --job-name=fulcca_optimized
#SBATCH --output=fulcca_optimized_%j.out
#SBATCH --error=fulcca_optimized_%j.err
#SBATCH --time=10:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=tier3
#SBATCH --account=neurosteer

echo "=========================================="
echo "FULCCA Optimized - Target: 68%+ Accuracy"
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

# Check versions
echo "Python version: $(python --version)"
echo "TensorFlow version: $(python -c 'import tensorflow as tf; print(tf.__version__)')"

echo "=========================================="
echo "Running Optimized FULCCA Analysis"
echo "=========================================="

# Set TFRecord directory
TFRECORD_DIR="/home/py9363/telluride_decoding/fulsang_preprocessed/tfrecords"

# Run optimized FULCCA analysis
echo "Starting optimized FULCCA analysis..."
python FULCCA_optimized.py \
    --tfrecord_dir "$TFRECORD_DIR" \
    --output_dir "fulcca_optimized_results" \
    --cca_dims 8 \
    --regularization 0.05 \
    --window_size 512 \
    --batch_size 16

echo "=========================================="
echo "Optimized FULCCA Analysis Complete"
echo "=========================================="
echo "End time: $(date)"
echo "Job duration: $SECONDS seconds"
echo "Results saved to: fulcca_optimized_results/"
echo "=========================================="

# Check results
echo "Disk usage:"
du -sh fulcca_optimized_results/

echo "GPU memory usage:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits

echo "Optimized FULCCA job completed successfully!"
