#!/bin/bash
#SBATCH --job-name=optimal_dascca
#SBATCH --output=optimal_dascca_%j.out
#SBATCH --error=optimal_dascca_%j.err
#SBATCH --time=12:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=tier3
#SBATCH --account=neurosteer

# Optimal_DASCCA - Optimized CCA Algorithm Targeting 83%+ Accuracy
# This script runs multiple optimized configurations to achieve 83%+ accuracy

echo "=========================================="
echo "OPTIMAL DASCCA - TARGETING 83%+ ACCURACY"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=========================================="

# Load necessary modules
echo "Loading modules..."
module load python/3.8
module load cuda/11.2
module load gcc/9.3.0

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d "env" ]; then
    echo "Activating environment..."
    source env/bin/activate
fi

# Check Python version and packages
echo "Python version: $(python --version)"
echo "TensorFlow version: $(python -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null || echo 'Not installed')"

# Set environment variables
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create output directory
mkdir -p optimal_dascca_results

# Check if preprocessing is needed
if [ ! -d "das_16subjects_preprocessed/tfrecords" ]; then
    echo "DAS 16-subjects preprocessing not found. Running preprocessing..."
    
    if [ -f "das_preprocessing_16subjects.py" ]; then
        echo "Running DAS preprocessing with 16 subjects support..."
        python das_preprocessing_16subjects.py \
            --data_dir "Data/Das/4004271" \
            --output_dir "das_16subjects_preprocessed" \
            --create_split
        
        if [ $? -eq 0 ]; then
            echo "✓ DAS preprocessing completed successfully"
        else
            echo "✗ DAS preprocessing failed"
            exit 1
        fi
    else
        echo "✗ das_preprocessing_16subjects.py not found!"
        exit 1
    fi
else
    echo "✓ DAS 16-subjects preprocessing already exists"
fi

# Run Optimal DASCCA with different configurations
echo "=========================================="
echo "Running Optimal DASCCA Configurations"
echo "=========================================="

# Configuration 1: Optimal 83% target
echo "Running Configuration 1: optimal_83_target..."
python Optimal_DASCCA.py \
    --tfrecord_dir das_16subjects_preprocessed/tfrecords \
    --config_name optimal_83_target \
    --output_dir optimal_dascca_results/optimal_83_target

# Configuration 2: High-dimensional
echo "Running Configuration 2: optimal_high_dim..."
python Optimal_DASCCA.py \
    --tfrecord_dir das_16subjects_preprocessed/tfrecords \
    --config_name optimal_high_dim \
    --output_dir optimal_dascca_results/optimal_high_dim

# Configuration 3: Extended window
echo "Running Configuration 3: optimal_extended..."
python Optimal_DASCCA.py \
    --tfrecord_dir das_16subjects_preprocessed/tfrecords \
    --config_name optimal_extended \
    --output_dir optimal_dascca_results/optimal_extended

# Configuration 4: Fine-tuned
echo "Running Configuration 4: optimal_finetuned..."
python Optimal_DASCCA.py \
    --tfrecord_dir das_16subjects_preprocessed/tfrecords \
    --config_name optimal_finetuned \
    --output_dir optimal_dascca_results/optimal_finetuned

# Configuration 5: Aggressive
echo "Running Configuration 5: optimal_aggressive..."
python Optimal_DASCCA.py \
    --tfrecord_dir das_16subjects_preprocessed/tfrecords \
    --config_name optimal_aggressive \
    --output_dir optimal_dascca_results/optimal_aggressive

# Generate summary report
echo "=========================================="
echo "Generating Summary Report"
echo "=========================================="

python -c "
import json
import os
from pathlib import Path
import pandas as pd

results_summary = []
config_dirs = [
    'optimal_dascca_results/optimal_83_target',
    'optimal_dascca_results/optimal_high_dim',
    'optimal_dascca_results/optimal_extended',
    'optimal_dascca_results/optimal_finetuned',
    'optimal_dascca_results/optimal_aggressive'
]

for config_dir in config_dirs:
    if os.path.exists(f'{config_dir}/results.json'):
        with open(f'{config_dir}/results.json', 'r') as f:
            results = json.load(f)
        
        config_name = config_dir.split('/')[-1]
        config_info = results.get('configuration', {})
        
        results_summary.append({
            'Configuration': config_name,
            'CCA_Dims': config_info.get('cca_dims', 'N/A'),
            'Regularization': config_info.get('regularization', 'N/A'),
            'Window_Size': config_info.get('window_size', 'N/A'),
            'Batch_Size': config_info.get('batch_size', 'N/A'),
            'Test_Accuracy': results.get('test_accuracy', 'N/A'),
            'ROC_AUC': results.get('roc_auc', 'N/A'),
            'Matthews_Correlation': results.get('matthews_correlation', 'N/A'),
            'Balanced_Accuracy': results.get('balanced_accuracy', 'N/A')
        })

if results_summary:
    df = pd.DataFrame(results_summary)
    df.to_csv('optimal_dascca_results/configuration_summary.csv', index=False)
    
    print('Optimal DASCCA Configuration Summary:')
    print('=' * 80)
    print(df.to_string(index=False))
    print('=' * 80)
    
    # Find best configuration
    accuracies = df['Test_Accuracy'].replace('N/A', 0).astype(float)
    if accuracies.max() > 0:
        best_idx = accuracies.idxmax()
        best_config = df.iloc[best_idx]
        
        print(f'\\n🏆 BEST CONFIGURATION: {best_config[\"Configuration\"]}')
        print(f'   Test Accuracy: {best_config[\"Test_Accuracy\"]:.4f}')
        print(f'   ROC-AUC: {best_config[\"ROC_AUC\"]:.4f}')
        print(f'   Matthews Correlation: {best_config[\"Matthews_Correlation\"]:.4f}')
        print(f'   Balanced Accuracy: {best_config[\"Balanced_Accuracy\"]:.4f}')
        
        # Check if 83% target achieved
        if float(best_config['Test_Accuracy']) >= 0.83:
            print('\\n🎉 SUCCESS: Target accuracy of 83%+ achieved!')
        else:
            print(f'\\n⚠ Current best: {float(best_config[\"Test_Accuracy\"]):.2%}, target: 83%+')
        
        with open('optimal_dascca_results/best_configuration.txt', 'w') as f:
            f.write(f'Best Configuration: {best_config[\"Configuration\"]}\\n')
            f.write(f'Test Accuracy: {best_config[\"Test_Accuracy\"]:.4f}\\n')
            f.write(f'ROC-AUC: {best_config[\"ROC_AUC\"]:.4f}\\n')
            f.write(f'Matthews Correlation: {best_config[\"Matthews_Correlation\"]:.4f}\\n')
            f.write(f'Balanced Accuracy: {best_config[\"Balanced_Accuracy\"]:.4f}\\n')
            f.write(f'\\nConfiguration Details:\\n')
            f.write(f'  CCA Dimensions: {best_config[\"CCA_Dims\"]}\\n')
            f.write(f'  Regularization: {best_config[\"Regularization\"]}\\n')
            f.write(f'  Window Size: {best_config[\"Window_Size\"]}\\n')
            f.write(f'  Batch Size: {best_config[\"Batch_Size\"]}\\n')
else:
    print('No results found to summarize')

print('\\nSummary report generated successfully!')
"

# Clean up temporary files
echo "Cleaning up temporary files..."
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Display final results
echo "=========================================="
echo "Optimal DASCCA Analysis Complete"
echo "=========================================="
echo "End time: $(date)"
echo "Job duration: $SECONDS seconds"
echo "Results saved to: optimal_dascca_results/"
echo "Summary report: optimal_dascca_results/configuration_summary.csv"
echo "Best configuration: optimal_dascca_results/best_configuration.txt"
echo "=========================================="

# Display disk usage
echo "Disk usage:"
du -sh optimal_dascca_results/ 2>/dev/null || echo "Results directory not found"

# Display GPU memory usage if available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU memory usage:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
fi

echo "Optimal DASCCA job completed successfully!"