#!/bin/bash
#SBATCH --job-name=combined_training
#SBATCH --output=combined_training_%j.out
#SBATCH --error=combined_training_%j.err
#SBATCH --time=24:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=tier3
#SBATCH --account=neurosteer

# Combined.sh - Combined Das and Fulsang Training Script
#
# This script:
# 1. Checks for Das MWF data (already processed)
# 2. Applies MWF filtering to Fulsang raw data
# 3. Trains CNN-LOC model using FULCNN architecture (CombinedCNNLOC.py) - RUNS FIRST
# 4. (Optional) CCA training is handled by a separate script: CombinedCCA.sh
#
# To run ONLY the Combined CCA analysis:
#   - On a SLURM cluster:  sbatch CombinedCCA.sh
#   - Locally (bash):      bash CombinedCCA.sh

set -e  # Exit on error

echo "=================================================================================="
echo "COMBINED - Das (MWF) + Fulsang (MWF) Training"
echo "=================================================================================="
echo "Started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "=================================================================================="

# Default paths
DAS_MWF_DIR="MWF_cleaned_DAS"
DAS_16SUBJECTS_DIR="das_16subjects_preprocessed"
DAS_RAW_DIR="${DAS_RAW_DIR:-/home/py9363/telluride_decoding/Data/Das/4004271}"
FULSANG_RAW_DIR="${FULSANG_RAW_DIR:-/home/py9363/telluride_decoding/Data/Fulsang/EEG}"
FULSANG_AUDIO_DIR="${FULSANG_AUDIO_DIR:-/home/py9363/telluride_decoding/Data/Fulsang/AUDIO}"
FULSANG_MWF_DIR="${FULSANG_MWF_DIR:-/home/py9363/telluride_decoding/MWF_cleaned_Fuglsang}"

# Check for Das data (MWF or 16-subjects preprocessing)
echo ""
echo "=================================================================================="
echo "CHECKING DAS DATA"
echo "=================================================================================="
DAS_DATA_DIR=""
DAS_PREPROCESSING_TYPE=""

if [ -d "$DAS_MWF_DIR" ]; then
    das_files=$(find "$DAS_MWF_DIR" -name "S*_MWF.mat" 2>/dev/null | wc -l)
    if [ "$das_files" -gt 0 ]; then
        DAS_DATA_DIR="$DAS_MWF_DIR"
        DAS_PREPROCESSING_TYPE="MWF"
        echo "✓ Das MWF-cleaned data directory found: $DAS_MWF_DIR"
        echo "  Found $das_files MWF-cleaned files"
    fi
fi

if [ -z "$DAS_DATA_DIR" ] && [ -d "$DAS_16SUBJECTS_DIR" ]; then
    das_tfrecord_files=$(find "$DAS_16SUBJECTS_DIR/tfrecords" -name "*.tfrecords" 2>/dev/null | wc -l)
    if [ "$das_tfrecord_files" -gt 0 ]; then
        DAS_DATA_DIR="$DAS_16SUBJECTS_DIR"
        DAS_PREPROCESSING_TYPE="16SUBJECTS"
        echo "✓ Das 16-subjects preprocessed data directory found: $DAS_16SUBJECTS_DIR"
        echo "  Found $das_tfrecord_files TFRecord files"
    fi
fi

if [ -z "$DAS_DATA_DIR" ]; then
    echo "⚠ WARNING: No Das preprocessed data found!"
    echo "  Checked:"
    echo "    - MWF_cleaned_DAS (MWF processing)"
    echo "    - das_16subjects_preprocessed (16-subjects preprocessing)"
    echo ""
    echo "=================================================================================="
    echo "AUTOMATICALLY RUNNING DAS 16-SUBJECTS PREPROCESSING"
    echo "=================================================================================="
    
    # Check for raw Das data
    if [ -d "$DAS_RAW_DIR" ]; then
        das_raw_files=$(find "$DAS_RAW_DIR" -name "S*.mat" 2>/dev/null | wc -l)
        echo "✓ Das raw data directory found: $DAS_RAW_DIR"
        echo "  Found $das_raw_files raw Das files"
        
        if [ "$das_raw_files" -eq 0 ]; then
            echo "✗ ERROR: No raw Das files found in $DAS_RAW_DIR"
            exit 1
        fi
        
        # Run 16-subjects preprocessing
        if [ ! -f "das_preprocessing_16subjects.py" ]; then
            echo "✗ ERROR: das_preprocessing_16subjects.py not found!"
            exit 1
        fi
        
        python3 das_preprocessing_16subjects.py --data_dir "$DAS_RAW_DIR" --output_dir "$DAS_16SUBJECTS_DIR" --create_split > das_16subjects_preprocessing.log 2>&1
        
        if [ $? -eq 0 ]; then
            das_tfrecord_files=$(find "$DAS_16SUBJECTS_DIR/tfrecords" -name "*.tfrecords" 2>/dev/null | wc -l)
            if [ "$das_tfrecord_files" -gt 0 ]; then
                DAS_DATA_DIR="$DAS_16SUBJECTS_DIR"
                DAS_PREPROCESSING_TYPE="16SUBJECTS"
                echo "✓ Das 16-subjects preprocessing completed: $das_tfrecord_files TFRecord files created"
            else
                echo "✗ Das preprocessing failed. Check das_16subjects_preprocessing.log"
                exit 1
            fi
        else
            echo "✗ Das preprocessing failed. Check das_16subjects_preprocessing.log"
            exit 1
        fi
    else
        echo "✗ Das raw data directory not found at $DAS_RAW_DIR"
        echo "  Please run preprocessing first:"
        echo "    MWF: python3 mwf_artifact_removal.py --dataset das --unified"
        echo "    16-subjects: python3 das_preprocessing_16subjects.py --data_dir $DAS_RAW_DIR --output_dir $DAS_16SUBJECTS_DIR --create_split"
        exit 1
    fi
fi

echo "  Using Das preprocessing: $DAS_PREPROCESSING_TYPE"

# Check for Fulsang raw data and apply MWF if needed
echo ""
echo "=================================================================================="
echo "CHECKING FULSANG DATA"
echo "=================================================================================="
if [ -d "$FULSANG_RAW_DIR" ]; then
    fulsang_raw_files=$(find "$FULSANG_RAW_DIR" -name "S*.mat" 2>/dev/null | wc -l)
    echo "✓ Fulsang raw data directory found: $FULSANG_RAW_DIR"
    echo "  Found $fulsang_raw_files raw files"
    
    if [ "$fulsang_raw_files" -eq 0 ]; then
        echo "✗ ERROR: No raw Fulsang files found in $FULSANG_RAW_DIR"
        exit 1
    fi
else
    echo "✗ Fulsang raw data directory not found at $FULSANG_RAW_DIR"
    exit 1
fi

# Check for Fulsang MWF data or apply MWF
if [ -d "$FULSANG_MWF_DIR" ]; then
    fulsang_mwf_files=$(find "$FULSANG_MWF_DIR" -name "sub*_MWF.mat" 2>/dev/null | wc -l)
    echo "✓ Fulsang MWF directory found: $FULSANG_MWF_DIR"
    echo "  Found $fulsang_mwf_files MWF-cleaned files"
    
    if [ "$fulsang_mwf_files" -lt "$fulsang_raw_files" ]; then
        echo "⚠ WARNING: Not all Fulsang files have been MWF processed"
        echo "  Raw files: $fulsang_raw_files, MWF files: $fulsang_mwf_files"
        echo "  MWF processing will be applied automatically during training"
    fi
else
    echo "⚠ Fulsang MWF directory not found at $FULSANG_MWF_DIR"
    echo "  MWF processing will be applied automatically during training"
    mkdir -p "$FULSANG_MWF_DIR"
fi

# Training parameters
WINDOW_SIZE=512
OVERLAP=0.5
BATCH_SIZE=32
NUM_EPOCHS=50
LEARNING_RATE=1e-3

# Run CNN-LOC training FIRST (CCA is commented out for now)
echo ""
echo "=================================================================================="
echo "RUNNING CNN-LOC TRAINING (FULCNN ARCHITECTURE) - RANDOM SPLIT"
echo "=================================================================================="
python3 CombinedCNNLOC.py \
    --das_data_dir "$DAS_DATA_DIR" \
    --das_preprocessing_type "$DAS_PREPROCESSING_TYPE" \
    --fulsang_raw_dir "$FULSANG_RAW_DIR" \
    --fulsang_audio_dir "$FULSANG_AUDIO_DIR" \
    --fulsang_mwf_dir "$FULSANG_MWF_DIR" \
    --window_size $WINDOW_SIZE \
    --overlap $OVERLAP \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --output_dir combined_cnnloc_results > combined_cnnloc_training.log 2>&1

CNN_EXIT_CODE=$?

if [ $CNN_EXIT_CODE -eq 0 ]; then
    echo "✓ CNN-LOC training (random split) completed successfully"
else
    echo "✗ CNN-LOC training (random split) failed with exit code: $CNN_EXIT_CODE"
    echo "  Check combined_cnnloc_training.log for details"
fi

# Run CNN-LOC training with Subject-Level Splitting
echo ""
echo "=================================================================================="
echo "RUNNING CNN-LOC TRAINING (FULCNN ARCHITECTURE) - SUBJECT-LEVEL SPLIT"
echo "=================================================================================="
python3 CombinedCNNLOCSub.py \
    --das_data_dir "$DAS_DATA_DIR" \
    --das_preprocessing_type "$DAS_PREPROCESSING_TYPE" \
    --fulsang_raw_dir "$FULSANG_RAW_DIR" \
    --fulsang_audio_dir "$FULSANG_AUDIO_DIR" \
    --fulsang_mwf_dir "$FULSANG_MWF_DIR" \
    --window_size $WINDOW_SIZE \
    --overlap $OVERLAP \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --output_dir combined_cnnloc_sub_results > combined_cnnloc_sub_training.log 2>&1

CNN_SUB_EXIT_CODE=$?

if [ $CNN_SUB_EXIT_CODE -eq 0 ]; then
    echo "✓ CNN-LOC training (subject-level split) completed successfully"
else
    echo "✗ CNN-LOC training (subject-level split) failed with exit code: $CNN_SUB_EXIT_CODE"
    echo "  Check combined_cnnloc_sub_training.log for details"
fi

# CCA training is commented out for now (taking too long)
# Uncomment this section when ready to run CCA training
# ================================================================================
# echo ""
# echo "=================================================================================="
# echo "RUNNING CCA TRAINING (OPTIMAL_FULCCA ARCHITECTURE)"
# echo "=================================================================================="
# python3 CombinedCCA.py \
#     --das_data_dir "$DAS_DATA_DIR" \
#     --das_preprocessing_type "$DAS_PREPROCESSING_TYPE" \
#     --fulsang_raw_dir "$FULSANG_RAW_DIR" \
#     --fulsang_audio_dir "$FULSANG_AUDIO_DIR" \
#     --fulsang_mwf_dir "$FULSANG_MWF_DIR" \
#     --window_size 1280 \
#     --overlap $OVERLAP \
#     --batch_size 6 \
#     --cca_dims 12 \
#     --regularization 0.08 \
#     --output_dir combined_cca_results > combined_cca_training.log 2>&1
# 
# CCA_EXIT_CODE=$?
# 
# if [ $CCA_EXIT_CODE -eq 0 ]; then
#     echo "✓ CCA training completed successfully"
# else
#     echo "✗ CCA training failed with exit code: $CCA_EXIT_CODE"
#     echo "  Check combined_cca_training.log for details"
# fi
# ================================================================================
CCA_EXIT_CODE=0  # Set to 0 since CCA is skipped

# Summary
echo ""
echo "=================================================================================="
echo "TRAINING SUMMARY"
echo "=================================================================================="
echo "Finished at: $(date)"
echo ""
echo "Results:"
echo "  CNN-LOC Training (Random Split): $([ $CNN_EXIT_CODE -eq 0 ] && echo '✓ SUCCESS' || echo '✗ FAILED')"
echo "  CNN-LOC Training (Subject-Level Split): $([ $CNN_SUB_EXIT_CODE -eq 0 ] && echo '✓ SUCCESS' || echo '✗ FAILED')"
echo "  CCA Training: SKIPPED (commented out - taking too long)"
echo ""
echo "Output directories:"
echo "  CNN-LOC (Random): combined_cnnloc_results/"
echo "  CNN-LOC (Subject-Level): combined_cnnloc_sub_results/"
echo "  CCA: combined_cca_results/ (not generated - CCA skipped)"
echo ""
echo "Log files:"
echo "  CNN-LOC (Random): combined_cnnloc_training.log"
echo "  CNN-LOC (Subject-Level): combined_cnnloc_sub_training.log"
echo "  CCA: combined_cca_training.log (not generated - CCA skipped)"
echo "=================================================================================="

# Exit with error if both CNN-LOC trainings failed (CCA is skipped)
if [ $CNN_EXIT_CODE -ne 0 ] && [ $CNN_SUB_EXIT_CODE -ne 0 ]; then
    exit 1
fi

exit 0

