#!/bin/bash
#SBATCH --job-name=fine_tuning_fulcca
#SBATCH --output=fine_tuning_fulcca_%j.out
#SBATCH --error=fine_tuning_fulcca_%j.err
#SBATCH --time=6:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=tier3
#SBATCH --account=neurosteer

echo "=========================================="
echo "FULCCA Fine-tuning - Target: 68%+ Accuracy"
echo "Based on best: 66.67% (extended_window)"
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
echo "Running FULCCA Fine-tuning Analysis"
echo "=========================================="

# Run fine-tuning analysis
echo "Starting FULCCA fine-tuning analysis..."
python Fine_Tuning_FULCCA.py \
    --tfrecord_dir "/home/py9363/telluride_decoding/fulsang_preprocessed/tfrecords" \
    --output_dir "fine_tuning_results"

echo "=========================================="
echo "FULCCA Fine-tuning Analysis Complete"
echo "=========================================="
echo "End time: $(date)"
echo "Job duration: $SECONDS seconds"
echo "Results saved to: fine_tuning_results/"
echo "=========================================="

# Check results
echo "Disk usage:"
du -sh fine_tuning_results/

echo "GPU memory usage:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits

echo "FULCCA fine-tuning analysis completed successfully!"
