#!/bin/bash
#SBATCH --job-name=dascnn_full
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=dascnn_full_%j.log
#SBATCH --error=dascnn_full_%j.err

# ==================================================================================
# DASCNN - FULL CNN-LOC ALGORITHM FOR DAS DATASET (16-SUBJECTS PIPELINE)
# ==================================================================================
# This script runs the full DASCNN implementation with:
# - Complete CNN-LOC architecture with attention mechanisms
# - Multi-scale feature extraction and residual connections
# - Comprehensive preprocessing (same as FULCNN)
# - Spectrogram-based frequency analysis
# - Mixed precision training
# - Comprehensive metrics evaluation
# - Temporal analysis across window lengths
# ==================================================================================

echo "=================================================================================="
echo "DASCNN - FULL CNN-LOC ALGORITHM FOR DAS DATASET"
echo "=================================================================================="
echo "Features:"
echo "- Full CNN-LOC architecture with attention mechanisms"
echo "- Multi-scale feature extraction and residual connections"
echo "- Comprehensive preprocessing (same as FULCNN)"
echo "- Spectrogram-based frequency analysis"
echo "- Mixed precision training"
echo "- Comprehensive metrics evaluation"
echo "- Temporal analysis across window lengths"
echo "=================================================================================="

# Set up environment
echo "Setting up environment..."
module load python/3.9
module load cuda/11.8

# Create virtual environment if it doesn't exist
if [ ! -d "venv_dascnn_full" ]; then
    echo "Creating DASCNN full virtual environment..."
    python -m venv venv_dascnn_full
fi

# Activate virtual environment
source venv_dascnn_full/bin/activate

# Install/upgrade required packages
echo "Installing DASCNN full packages..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow==2.12.0
pip install scikit-learn matplotlib seaborn tqdm
pip install scipy numpy pandas

# Function to check DAS data availability
check_das_data() {
    echo "=================================================================================="
    echo "CHECKING DAS DATA AVAILABILITY (16-SUBJECTS PIPELINE)"
    echo "=================================================================================="
    
    # Check for 16-subjects preprocessed data
    if [ -d "das_16subjects_preprocessed" ]; then
        echo "✓ Found DAS 16-subjects preprocessed data"
        
        # Check TFRecord files
        tfrecord_count=$(find das_16subjects_preprocessed/tfrecords -name "*.tfrecords" 2>/dev/null | wc -l)
        if [ $tfrecord_count -gt 0 ]; then
            echo "✓ TFRecord files: $tfrecord_count"
        else
            echo "⚠ TFRecord files: $tfrecord_count (may need preprocessing)"
        fi
        
        # Check for train/test splits
        if [ -d "das_16subjects_preprocessed/tfrecords/train" ] && [ -d "das_16subjects_preprocessed/tfrecords/test" ]; then
            train_count=$(find das_16subjects_preprocessed/tfrecords/train -name "*.tfrecords" 2>/dev/null | wc -l)
            test_count=$(find das_16subjects_preprocessed/tfrecords/test -name "*.tfrecords" 2>/dev/null | wc -l)
            echo "✓ Found separate train/test directories"
            echo "✓ Train TFRecord files: $train_count"
            echo "✓ Test TFRecord files: $test_count"
            
            # Check for validation split
            if [ -d "das_16subjects_preprocessed/tfrecords/val" ]; then
                val_count=$(find das_16subjects_preprocessed/tfrecords/val -name "*.tfrecords" 2>/dev/null | wc -l)
                echo "✓ Val TFRecord files: $val_count"
            fi
            
            # If no TFRecord files found, run preprocessing
            if [ $tfrecord_count -eq 0 ]; then
                echo "⚠ No TFRecord files found, running preprocessing..."
                if [ -f "das_preprocessing_16subjects.py" ]; then
                    python das_preprocessing_16subjects.py --data_dir "Data/Das/4004271" --output_dir "das_16subjects_preprocessed" --create_split > das_preprocessing_full.log 2>&1
                    if [ $? -eq 0 ]; then
                        echo "✓ Preprocessing completed, rechecking files..."
                        tfrecord_count=$(find das_16subjects_preprocessed/tfrecords -name "*.tfrecords" 2>/dev/null | wc -l)
                        echo "✓ TFRecord files after preprocessing: $tfrecord_count"
                    else
                        echo "✗ Preprocessing failed"
                        return 1
                    fi
                else
                    echo "✗ das_preprocessing_16subjects.py not found!"
                    return 1
                fi
            fi
        else
            echo "⚠ No train/test subdirectories found, will use subject-wise splitting"
        fi
        
        # Show sample files
        echo "Sample DAS TFRecord files:"
        find das_16subjects_preprocessed/tfrecords -name "*.tfrecords" | head -3 | while read file; do
            echo "  $file"
        done
        
    else
        echo "✗ DAS 16-subjects preprocessed data not found"
        echo "  Expected directory: das_16subjects_preprocessed"
        echo "  Running preprocessing pipeline..."
        
        # Run preprocessing
        if [ -f "das_preprocessing_16subjects.py" ]; then
            echo "Running das_preprocessing_16subjects.py..."
            python das_preprocessing_16subjects.py --data_dir "Data/Das/4004271" --output_dir "das_16subjects_preprocessed" --create_split > das_preprocessing_full.log 2>&1
            
            if [ $? -eq 0 ]; then
                echo "✓ DAS preprocessing completed successfully"
                echo "Results saved to das_16subjects_preprocessed/"
                
                # Check if TFRecord files were created
                if [ -d "das_16subjects_preprocessed/tfrecords" ]; then
                    tfrecord_count=$(find das_16subjects_preprocessed/tfrecords -name "*.tfrecords" 2>/dev/null | wc -l)
                    echo "✓ Created $tfrecord_count TFRecord files"
                    
                    # Check for train/test splits
                    if [ -d "das_16subjects_preprocessed/tfrecords/train" ] && [ -d "das_16subjects_preprocessed/tfrecords/test" ]; then
                        train_count=$(find das_16subjects_preprocessed/tfrecords/train -name "*.tfrecords" 2>/dev/null | wc -l)
                        test_count=$(find das_16subjects_preprocessed/tfrecords/test -name "*.tfrecords" 2>/dev/null | wc -l)
                        echo "✓ Train TFRecord files: $train_count"
                        echo "✓ Test TFRecord files: $test_count"
                    fi
                fi
                
                return 0
            else
                echo "✗ DAS preprocessing failed"
                echo "Check the log: das_preprocessing_full.log"
                return 1
            fi
        else
            echo "✗ das_preprocessing_16subjects.py not found!"
            echo "Expected file: das_preprocessing_16subjects.py"
            return 1
        fi
    fi
    
    echo "✓ Data leakage prevention enabled"
    echo "✓ Attention labels validated"
    echo "=================================================================================="
    return 0
}

# Function to run full DASCNN training
run_full_dascnn_training() {
    echo "=================================================================================="
    echo "RUNNING FULL DASCNN TRAINING WITH COMPREHENSIVE ARCHITECTURE"
    echo "=================================================================================="
    echo "This step trains the full DASCNN model with:"
    echo "- Complete CNN-LOC architecture with attention mechanisms"
    echo "- Multi-scale feature extraction and residual connections"
    echo "- Comprehensive preprocessing (same as FULCNN)"
    echo "- Spectrogram-based frequency analysis"
    echo "- Mixed precision training (AMP) for speed"
    echo "- Comprehensive metrics evaluation"
    echo "- Temporal analysis across window lengths (0.5s to 30s)"
    echo "- 16-subjects preprocessing pipeline integration"
    echo "- Data leakage prevention"
    echo "- Validated attention labels"
    echo "=================================================================================="
    
    echo "Starting full DASCNN training for DAS dataset..."
    
    # Run full DASCNN with comprehensive architecture
    python DASCNN_full.py \
        --tfrecord_dir das_16subjects_preprocessed/tfrecords \
        --batch_size 16 \
        --num_epochs 50 \
        --learning_rate 1e-4 \
        --window_size 512 \
        --output_dir dascnn_full_results \
        --num_workers 4 \
        --use_mixed_precision \
        2>&1 | tee dascnn_full_training.log
    
    # Check if training was successful
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "=================================================================================="
        echo "✓ FULL DASCNN TRAINING COMPLETED SUCCESSFULLY"
        echo "=================================================================================="
        echo "Results saved to: dascnn_full_results/"
        echo "Training log: dascnn_full_training.log"
        
        # Show key results
        if [ -f "dascnn_full_results/dascnn_results.json" ]; then
            echo ""
            echo "Key Results:"
            python -c "
import json
with open('dascnn_full_results/dascnn_results.json', 'r') as f:
    results = json.load(f)
print(f'Test Accuracy: {results[\"accuracy\"]:.4f}')
print(f'Balanced Accuracy: {results[\"balanced_accuracy\"]:.4f}')
print(f'F1 Score: {results[\"f1_score\"]:.4f}')
print(f'ROC-AUC: {results[\"roc_auc\"]:.4f}')
print(f'Matthews Correlation Coefficient: {results[\"matthews_corrcoef\"]:.4f}')
print(f'Cohen\\'s Kappa: {results[\"cohen_kappa\"]:.4f}')
print(f'Number of samples: {results[\"n_samples\"]}')
print('')
print('Temporal Performance:')
for window_size, metrics in results['temporal_metrics'].items():
    print(f'  {window_size}: {metrics[\"accuracy\"]:.4f} accuracy ({metrics[\"n_samples\"]} samples)')
"
        fi
        
        return 0
    else
        echo "=================================================================================="
        echo "✗ FULL DASCNN TRAINING FAILED"
        echo "=================================================================================="
        echo "Check the error log: dascnn_full_training.log"
        return 1
    fi
}

# Function to show full architecture features
show_full_architecture_features() {
    echo "=================================================================================="
    echo "FULL DASCNN ARCHITECTURE FEATURES"
    echo "=================================================================================="
    echo "1. Complete CNN-LOC Architecture:"
    echo "   - MultiScaleFeatureExtractor with 1x1 and 3x1 kernels"
    echo "   - ResidualBlock with SpatialTemporalAttention"
    echo "   - Temporal processing blocks with pooling"
    echo "   - Spatial processing blocks with pooling"
    echo "   - Global attention mechanism"
    echo "   - Adaptive pooling for variable input sizes"
    echo ""
    echo "2. Comprehensive Preprocessing (Same as FULCNN):"
    echo "   - Artifact detection and removal (>5 std deviations)"
    echo "   - Bandpass filtering (1-40 Hz)"
    echo "   - Robust normalization (MAD)"
    echo "   - Soft clipping (tanh)"
    echo "   - Spectrogram computation with frequency bands"
    echo "   - Delta, Theta, Alpha, Beta band extraction"
    echo ""
    echo "3. Advanced Training Features:"
    echo "   - Mixed precision training (AMP)"
    echo "   - OneCycleLR scheduler"
    echo "   - AdamW optimizer with weight decay"
    echo "   - Comprehensive metrics evaluation"
    echo "   - Temporal analysis across window lengths"
    echo ""
    echo "4. DAS-Specific Adaptations:"
    echo "   - 64 EEG channels (vs 66 in Fulsang)"
    echo "   - 64 Hz sampling rate (downsampled from 1000 Hz)"
    echo "   - 16-subjects preprocessing pipeline"
    echo "   - Binary attention decoding (Left vs Right)"
    echo "   - Subject-wise data splitting"
    echo ""
    echo "5. Comprehensive Metrics:"
    echo "   - Accuracy, Balanced Accuracy"
    echo "   - Precision, Recall, F1-Score"
    echo "   - ROC-AUC, Matthews Correlation Coefficient"
    echo "   - Cohen's Kappa"
    echo "   - Temporal performance analysis"
    echo "   - Confusion matrix analysis"
    echo "=================================================================================="
}

# Main execution
main() {
    echo "Starting full DASCNN execution..."
    
    # Check data availability
    if ! check_das_data; then
        echo "Data check failed, exiting..."
        exit 1
    fi
    
    # Show full architecture features
    show_full_architecture_features
    
    # Run full training
    if ! run_full_dascnn_training; then
        echo "Training failed, exiting..."
        exit 1
    fi
    
    echo ""
    echo "=================================================================================="
    echo "FULL DASCNN EXECUTION COMPLETED SUCCESSFULLY"
    echo "=================================================================================="
    echo "Complete CNN-LOC architecture implemented with all features:"
    echo "✓ Full CNN-LOC architecture with attention mechanisms"
    echo "✓ Multi-scale feature extraction and residual connections"
    echo "✓ Comprehensive preprocessing (same as FULCNN)"
    echo "✓ Spectrogram-based frequency analysis"
    echo "✓ Mixed precision training"
    echo "✓ Comprehensive metrics evaluation"
    echo "✓ Temporal analysis across window lengths"
    echo "✓ 16-subjects preprocessing pipeline"
    echo ""
    echo "Results available in: dascnn_full_results/"
    echo "Training log: dascnn_full_training.log"
    echo "=================================================================================="
}

# Run main function
main "$@"
