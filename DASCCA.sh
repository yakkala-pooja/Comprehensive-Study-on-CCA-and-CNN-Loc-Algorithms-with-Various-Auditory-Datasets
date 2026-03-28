#!/bin/bash
#SBATCH --job-name=dascca_cca
#SBATCH --output=dascca_cca_%j.out
#SBATCH --error=dascca_cca_%j.err
#SBATCH --time=24:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=tier3
#SBATCH --account=neurosteer

# DASCCA - Canonical Correlation Analysis Algorithm for DAS Dataset
# This script runs the DASCCA implementation with comprehensive metrics evaluation
# 
# Features:
# - Uses DAS 16-subjects preprocessing data
# - EEG + Audio envelope correlation (improved CCA performance)
# - Automatic audio file mapping and envelope extraction
# - Comprehensive metrics evaluation
# - Backward model: Time-lagging on EEG (spatiotemporal features)
# - Forward model: Causal audio filtering (stimulus precedes response)
# - PCA regularization (optional, for EEG dimensionality reduction)
# - LDA classifier (combines canonical correlation coefficients)

echo "=========================================="
echo "DASCCA - CCA Algorithm for DAS Dataset"
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
echo "NumPy version: $(python -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo 'Not installed')"

# Set environment variables
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create output directory
mkdir -p dascca_results

# Check for existing DAS 16-subjects preprocessing (DO NOT RUN PREPROCESSING)
echo "=========================================="
echo "Checking for Existing DAS Preprocessing"
echo "=========================================="

if [ ! -d "das_16subjects_preprocessed/tfrecords" ]; then
    echo "✗ ERROR: DAS 16-subjects preprocessing not found!"
    echo "  Expected directory: das_16subjects_preprocessed/tfrecords"
    echo ""
    echo "  Please run preprocessing separately using:"
    echo "    python das_preprocessing_16subjects.py \\"
    echo "        --data_dir \"Data/Das/4004271\" \\"
    echo "        --output_dir \"das_16subjects_preprocessed\" \\"
    echo "        --create_split"
    echo ""
    echo "  This script uses EXISTING preprocessing data only."
    exit 1
else
    echo "✓ DAS 16-subjects preprocessing found"
    echo "  Using existing TFRecord files from: das_16subjects_preprocessed/tfrecords"
    
    # Count TFRecord files
    tfrecord_count=$(find das_16subjects_preprocessed/tfrecords -name "*.tfrecords" 2>/dev/null | wc -l)
    echo "  Found $tfrecord_count TFRecord file(s)"
    
    if [ "$tfrecord_count" -eq 0 ]; then
        echo "  ⚠ WARNING: No TFRecord files found in the directory!"
        exit 1
    fi
fi

# Run DASCCA with different configurations
echo "=========================================="
echo "Running DASCCA Analysis"
echo "=========================================="

# Configuration 1: Standard CCA with Backward/Forward Model
# Audio loading ENABLED for proper CCA performance
# Full dataset (no max_files limit). Window 8s (1024 samples at 128 Hz).
# LDA: Enabled by default for better classification
echo "Running Standard CCA Configuration with Backward/Forward Model..."
python DASCCA.py \
    --tfrecord_dir das_16subjects_preprocessed/tfrecords \
    --batch_size 16 \
    --cca_dims 40 \
    --regularization 0.08 \
    --window_size 1024 \
    --output_dir dascca_results/standard_cca \
    --load_audio \
    --audio_base_dir /home/py9363/telluride_decoding/Data/Das/4004271/stimuli/stimuli \
    --eeg_lag_samples 5 \
    --use_lda

# OPTIMIZATION: Reduced to single configuration for faster completion
# Uncomment additional configurations if needed, but they will take longer

# Configuration 2: High-dimensional CCA (4 dimensions max due to audio features)
# echo "Running High-dimensional CCA Configuration..."
# python DASCCA.py \
#     --tfrecord_dir das_16subjects_preprocessed/tfrecords \
#     --batch_size 16 \
#     --cca_dims 4 \
#     --regularization 0.005 \
#     --window_size 512 \
#     --output_dir dascca_results/high_dim_cca \
#     --no_load_audio \
#     --max_files 100

# Configuration 3: Low-dimensional CCA (2 dimensions)
# echo "Running Low-dimensional CCA Configuration..."
# python DASCCA.py \
#     --tfrecord_dir das_16subjects_preprocessed/tfrecords \
#     --batch_size 16 \
#     --cca_dims 2 \
#     --regularization 0.05 \
#     --window_size 512 \
#     --output_dir dascca_results/low_dim_cca \
#     --no_load_audio \
#     --max_files 100

# Generate summary report
echo "=========================================="
echo "Generating Summary Report"
echo "=========================================="

python -c "
import json
import os
from pathlib import Path
import pandas as pd

# Collect results from all configurations
results_summary = []
config_dirs = [
    'dascca_results/standard_cca',
    'dascca_results/high_dim_cca', 
    'dascca_results/low_dim_cca',
    'dascca_results/short_window_cca',
    'dascca_results/long_window_cca',
    'dascca_results/high_reg_cca',
    'dascca_results/low_reg_cca',
    'dascca_results/large_batch_cca',
    'dascca_results/ultra_high_dim_cca',
    'dascca_results/very_short_window_cca',
    'dascca_results/very_long_window_cca',
    'dascca_results/extreme_reg_cca'
]

for config_dir in config_dirs:
    if os.path.exists(f'{config_dir}/results.json'):
        with open(f'{config_dir}/results.json', 'r') as f:
            results = json.load(f)
        
        config_name = config_dir.split('/')[-1]
        results_summary.append({
            'Configuration': config_name,
            'Accuracy': results.get('accuracy', 'N/A'),
            'ROC-AUC': results.get('roc_auc_metrics', {}).get('roc_auc_score', 'N/A'),
            'RMSE': results.get('msed_metrics', {}).get('rmse', 'N/A'),
            'R-squared': results.get('msed_metrics', {}).get('r_squared', 'N/A'),
            'MCC': results.get('advanced_metrics', {}).get('matthews_correlation_coefficient', 'N/A'),
            'Balanced_Accuracy': results.get('advanced_metrics', {}).get('balanced_accuracy', 'N/A')
        })

# Save summary
if results_summary:
    df = pd.DataFrame(results_summary)
    df.to_csv('dascca_results/configuration_summary.csv', index=False)
    
    print('Configuration Summary:')
    print(df.to_string(index=False))
    
    # Find best configuration
    best_acc_idx = df['Accuracy'].astype(float).idxmax()
    best_config = df.iloc[best_acc_idx]['Configuration']
    best_accuracy = df.iloc[best_acc_idx]['Accuracy']
    
    print(f'\\nBest Configuration: {best_config}')
    print(f'Best Accuracy: {best_accuracy}')
    
    # Find best ROC-AUC
    roc_auc_values = df['ROC-AUC'].replace('N/A', 0).astype(float)
    best_roc_config = 'N/A'
    best_roc_auc = 'N/A'
    if roc_auc_values.max() > 0:
        best_roc_idx = roc_auc_values.idxmax()
        best_roc_config = df.iloc[best_roc_idx]['Configuration']
        best_roc_auc = df.iloc[best_roc_idx]['ROC-AUC']
        print(f'Best ROC-AUC Configuration: {best_roc_config}')
        print(f'Best ROC-AUC: {best_roc_auc}')
    
    # Find best MCC
    mcc_values = df['MCC'].replace('N/A', 0).astype(float)
    best_mcc_config = 'N/A'
    best_mcc = 'N/A'
    if mcc_values.max() > 0:
        best_mcc_idx = mcc_values.idxmax()
        best_mcc_config = df.iloc[best_mcc_idx]['Configuration']
        best_mcc = df.iloc[best_mcc_idx]['MCC']
        print(f'Best MCC Configuration: {best_mcc_config}')
        print(f'Best MCC: {best_mcc}')
    
    with open('dascca_results/best_configuration.txt', 'w') as f:
        f.write(f'Best Configuration: {best_config}\\n')
        f.write(f'Best Accuracy: {best_accuracy}\\n')
        f.write(f'Best ROC-AUC Configuration: {best_roc_config}\\n')
        f.write(f'Best ROC-AUC: {best_roc_auc}\\n')
        f.write(f'Best MCC Configuration: {best_mcc_config}\\n')
        f.write(f'Best MCC: {best_mcc}\\n')
        f.write(f'Summary saved to: dascca_results/configuration_summary.csv\\n')
else:
    print('No results found to summarize')

print('Summary report generated successfully!')
"

# Generate detailed analysis report
echo "=========================================="
echo "Generating Detailed Analysis Report"
echo "=========================================="

python -c "
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Create detailed analysis report
report_content = []
report_content.append('=' * 80)
report_content.append('DASCCA DETAILED ANALYSIS REPORT')
report_content.append('=' * 80)
report_content.append('')

config_dirs = [
    'dascca_results/standard_cca',
    'dascca_results/high_dim_cca', 
    'dascca_results/low_dim_cca',
    'dascca_results/short_window_cca',
    'dascca_results/long_window_cca',
    'dascca_results/high_reg_cca',
    'dascca_results/low_reg_cca',
    'dascca_results/large_batch_cca',
    'dascca_results/ultra_high_dim_cca',
    'dascca_results/very_short_window_cca',
    'dascca_results/very_long_window_cca',
    'dascca_results/extreme_reg_cca'
]

for config_dir in config_dirs:
    if os.path.exists(f'{config_dir}/results.json'):
        with open(f'{config_dir}/results.json', 'r') as f:
            results = json.load(f)
        
        config_name = config_dir.split('/')[-1].replace('_', ' ').title()
        report_content.append(f'{config_name}:')
        report_content.append('-' * 40)
        
        # Basic metrics
        accuracy = results.get('accuracy', 'N/A')
        report_content.append(f'Accuracy: {accuracy}')
        
        # ROC-AUC metrics
        roc_auc = results.get('roc_auc_metrics', {})
        if 'error' not in roc_auc:
            roc_score = roc_auc.get('roc_auc_score', 'N/A')
            avg_precision = roc_auc.get('average_precision', 'N/A')
            report_content.append(f'ROC-AUC Score: {roc_score}')
            report_content.append(f'Average Precision: {avg_precision}')
        
        # MSED metrics
        msed = results.get('msed_metrics', {})
        if 'error' not in msed:
            rmse = msed.get('rmse', 'N/A')
            r_squared = msed.get('r_squared', 'N/A')
            report_content.append(f'RMSE: {rmse}')
            report_content.append(f'R-squared: {r_squared}')
        
        # Advanced metrics
        advanced = results.get('advanced_metrics', {})
        if 'error' not in advanced:
            mcc = advanced.get('matthews_correlation_coefficient', 'N/A')
            balanced_acc = advanced.get('balanced_accuracy', 'N/A')
            report_content.append(f'Matthews Correlation Coefficient: {mcc}')
            report_content.append(f'Balanced Accuracy: {balanced_acc}')
        
        # Temporal metrics
        temporal = results.get('temporal_metrics', {})
        if temporal:
            report_content.append('Temporal Performance:')
            for key, value in temporal.items():
                report_content.append(f'  {key}: {value:.4f}')
        
        report_content.append('')

# Save detailed report
# Ensure directory exists
from pathlib import Path
Path('dascca_results').mkdir(parents=True, exist_ok=True)
with open('dascca_results/detailed_analysis_report.txt', 'w') as f:
    f.write('\\n'.join(report_content))

print('Detailed analysis report generated successfully!')
"

# Clean up temporary files
echo "Cleaning up temporary files..."
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Display final results
echo "=========================================="
echo "DASCCA Analysis Complete"
echo "=========================================="
echo "End time: $(date)"
echo "Job duration: $SECONDS seconds"
echo "Results saved to: dascca_results/"
echo "Summary report: dascca_results/configuration_summary.csv"
echo "Detailed report: dascca_results/detailed_analysis_report.txt"
echo "Best configuration: dascca_results/best_configuration.txt"
echo "=========================================="

# Display disk usage
echo "Disk usage:"
du -sh dascca_results/ 2>/dev/null || echo "Results directory not found"

# Display GPU memory usage if available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU memory usage:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
fi

echo "DASCCA job completed successfully!"
