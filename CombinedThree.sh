#!/bin/bash
#SBATCH --job-name=combined_three_training
#SBATCH --output=combined_three_training_%j.out
#SBATCH --error=combined_three_training_%j.err
#SBATCH --time=24:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=tier3
#SBATCH --account=neurosteer

set -e

echo "=================================================================================="
echo "COMBINED THREE DATASETS - Das + Fulsang + KU Leuven 255 Training"
echo "=================================================================================="
echo "Started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "=================================================================================="

DAS_MWF_DIR="MWF_cleaned_DAS"
DAS_16SUBJECTS_DIR="das_16subjects_preprocessed"
DAS_RAW_DIR="${DAS_RAW_DIR:-/home/py9363/telluride_decoding/Data/Das/4004271}"
FULSANG_RAW_DIR="${FULSANG_RAW_DIR:-/home/py9363/telluride_decoding/Data/Fulsang/EEG}"
FULSANG_AUDIO_DIR="${FULSANG_AUDIO_DIR:-/home/py9363/telluride_decoding/Data/Fulsang/AUDIO}"
FULSANG_MWF_DIR="${FULSANG_MWF_DIR:-/home/py9363/telluride_decoding/MWF_cleaned_Fuglsang}"
KULEUVEN_RAW_DIR="${KULEUVEN_RAW_DIR:-/home/py9363/telluride_decoding/Data/KULeuven 255}"
KULEUVEN_STIMULI_DIR="${KULEUVEN_STIMULI_DIR:-/home/py9363/telluride_decoding/Data/KULeuven 255/stimuli/stimuli}"
KULEUVEN_PREPROCESSED_DIR="${KULEUVEN_PREPROCESSED_DIR:-kuleuven_255_preprocessed}"

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
    
    if [ -d "$DAS_RAW_DIR" ]; then
        das_raw_files=$(find "$DAS_RAW_DIR" -name "S*.mat" 2>/dev/null | wc -l)
        echo "✓ Das raw data directory found: $DAS_RAW_DIR"
        echo "  Found $das_raw_files raw Das files"
        
        if [ "$das_raw_files" -eq 0 ]; then
            echo "✗ ERROR: No raw Das files found in $DAS_RAW_DIR"
            exit 1
        fi
        
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
        exit 1
    fi
fi

echo "  Using Das preprocessing: $DAS_PREPROCESSING_TYPE"

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

echo ""
echo "=================================================================================="
echo "CHECKING KU LEUVEN 255 DATA"
echo "=================================================================================="
KULEUVEN_NEEDS_PREPROCESSING=0

if [ -d "$KULEUVEN_PREPROCESSED_DIR" ]; then
    kuleuven_preprocessed_files=$(find "$KULEUVEN_PREPROCESSED_DIR" -name "S*_preprocessed.mat" 2>/dev/null | wc -l)
    echo "✓ KU Leuven preprocessed directory found: $KULEUVEN_PREPROCESSED_DIR"
    echo "  Found $kuleuven_preprocessed_files preprocessed files"
    
    if [ "$kuleuven_preprocessed_files" -eq 0 ]; then
        echo "⚠ WARNING: No preprocessed KU Leuven files found"
        echo "  Will run PREPROCESS255.py to create preprocessed data"
        KULEUVEN_NEEDS_PREPROCESSING=1
    fi
else
    echo "⚠ KU Leuven preprocessed directory not found at $KULEUVEN_PREPROCESSED_DIR"
    echo "  Will run PREPROCESS255.py to create preprocessed data"
    mkdir -p "$KULEUVEN_PREPROCESSED_DIR"
    KULEUVEN_NEEDS_PREPROCESSING=1
fi

if [ "$KULEUVEN_NEEDS_PREPROCESSING" -eq 1 ]; then
    echo ""
    echo "=================================================================================="
    echo "RUNNING KU LEUVEN 255 PREPROCESSING"
    echo "=================================================================================="
    
    if [ ! -d "$KULEUVEN_RAW_DIR" ]; then
        echo "✗ ERROR: KU Leuven raw data directory not found at $KULEUVEN_RAW_DIR"
        exit 1
    fi
    
    kuleuven_raw_dirs=$(find "$KULEUVEN_RAW_DIR" -type d -name "S*" 2>/dev/null | wc -l)
    echo "✓ KU Leuven raw data directory found: $KULEUVEN_RAW_DIR"
    echo "  Found $kuleuven_raw_dirs subject directories"
    
    if [ "$kuleuven_raw_dirs" -eq 0 ]; then
        echo "✗ ERROR: No KU Leuven subject directories found in $KULEUVEN_RAW_DIR"
        exit 1
    fi
    
    if [ ! -f "PREPROCESS255.py" ]; then
        echo "✗ ERROR: PREPROCESS255.py not found!"
        exit 1
    fi
    
    python3 PREPROCESS255.py \
        --data_dir "$KULEUVEN_RAW_DIR" \
        --stimuli_dir "$KULEUVEN_STIMULI_DIR" \
        --output_dir "$KULEUVEN_PREPROCESSED_DIR" \
        --target_sampling_rate 128 \
        --target_channels 64 > kuleuven_preprocessing.log 2>&1
    
    if [ $? -eq 0 ]; then
        kuleuven_preprocessed_files=$(find "$KULEUVEN_PREPROCESSED_DIR" -name "S*_preprocessed.mat" 2>/dev/null | wc -l)
        if [ "$kuleuven_preprocessed_files" -gt 0 ]; then
            echo "✓ KU Leuven preprocessing completed: $kuleuven_preprocessed_files preprocessed files created"
        else
            echo "✗ KU Leuven preprocessing failed. Check kuleuven_preprocessing.log"
            exit 1
        fi
    else
        echo "✗ KU Leuven preprocessing failed. Check kuleuven_preprocessing.log"
        exit 1
    fi
fi

WINDOW_SIZE=512
OVERLAP=0.5
BATCH_SIZE=64
NUM_EPOCHS=50
LEARNING_RATE=5e-4
DROPOUT_RATE=0.5
WEIGHT_DECAY=1e-4
USE_FOCAL_LOSS="--use_focal_loss"
FOCAL_GAMMA=2.0
USE_AUGMENTATION="--use_augmentation"

echo ""
echo "=================================================================================="
echo "RUNNING COMBINED THREE DATASETS CNN-LOC TRAINING"
echo "=================================================================================="
python3 CombinedThreeCNN.py \
    --das_data_dir "$DAS_DATA_DIR" \
    --das_preprocessing_type "$DAS_PREPROCESSING_TYPE" \
    --fulsang_raw_dir "$FULSANG_RAW_DIR" \
    --fulsang_audio_dir "$FULSANG_AUDIO_DIR" \
    --fulsang_mwf_dir "$FULSANG_MWF_DIR" \
    --kuleuven_preprocessed_dir "$KULEUVEN_PREPROCESSED_DIR" \
    --window_size $WINDOW_SIZE \
    --overlap $OVERLAP \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --dropout_rate $DROPOUT_RATE \
    --weight_decay $WEIGHT_DECAY \
    $USE_FOCAL_LOSS \
    --focal_gamma $FOCAL_GAMMA \
    $USE_AUGMENTATION \
    --output_dir combined_three_cnn_results > combined_three_cnn_training.log 2>&1

THREE_CNN_EXIT_CODE=$?

if [ $THREE_CNN_EXIT_CODE -eq 0 ]; then
    echo "✓ Combined Three Datasets CNN-LOC training completed successfully"
else
    echo "✗ Combined Three Datasets CNN-LOC training failed with exit code: $THREE_CNN_EXIT_CODE"
    echo "  Check combined_three_cnn_training.log for details"
fi

echo ""
echo "=================================================================================="
echo "TRAINING SUMMARY"
echo "=================================================================================="
echo "Finished at: $(date)"
echo ""
echo "Results:"
echo "  Combined Three Datasets CNN-LOC Training: $([ $THREE_CNN_EXIT_CODE -eq 0 ] && echo '✓ SUCCESS' || echo '✗ FAILED')"
echo ""
echo "Output directory:"
echo "  Combined Three CNN: combined_three_cnn_results/"
echo ""
echo "Log files:"
echo "  KU Leuven Preprocessing: kuleuven_preprocessing.log"
echo "  Combined Three CNN: combined_three_cnn_training.log"
echo "=================================================================================="

if [ $THREE_CNN_EXIT_CODE -ne 0 ]; then
    exit 1
fi

exit 0

