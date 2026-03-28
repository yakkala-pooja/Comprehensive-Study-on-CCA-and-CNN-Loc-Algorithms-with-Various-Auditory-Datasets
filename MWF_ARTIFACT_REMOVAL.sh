#!/bin/bash
#SBATCH --job-name=mwf_artifact_removal
#SBATCH --output=mwf_artifact_removal_%j.out
#SBATCH --error=mwf_artifact_removal_%j.err
#SBATCH --time=8:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=tier3
#SBATCH --account=neurosteer

# MWF Artifact Removal Script for Das and Fuglsang Datasets
# This script applies Multi-channel Wiener Filtering (MWF) to remove artifacts
# from EEG recordings in both Das and Fuglsang datasets.
#
# Features:
# - MWF artifact removal using EOG reference channels
# - Downsampling for Fuglsang dataset (512 Hz -> 128 Hz)
# - Visualization of before/after filtering
# - Unified preprocessing function
#
# Usage:
#   bash MWF_ARTIFACT_REMOVAL.sh              # Process both datasets
#   bash MWF_ARTIFACT_REMOVAL.sh --das        # Process only Das dataset
#   bash MWF_ARTIFACT_REMOVAL.sh --fuglsang   # Process only Fuglsang dataset
#   bash MWF_ARTIFACT_REMOVAL.sh --visualize  # Include visualization

echo "=================================================================================="
echo "MWF ARTIFACT REMOVAL FOR DAS AND FUGLSANG DATASETS"
echo "=================================================================================="
echo "Started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=================================================================================="

# Environment setup
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=8

# Timeout handler for job management
timeout_handler() {
    echo "=================================================================================="
    echo "JOB TIMEOUT WARNING: 90% of time limit reached"
    echo "Current time: $(date)"
    echo "Attempting to save current progress..."
    echo "=================================================================================="
    
    # Try to save any partial results
    if [ -d "MWF_cleaned_DAS" ]; then
        echo "Saving Das MWF partial results..."
        cp -r MWF_cleaned_DAS MWF_cleaned_DAS_backup_$(date +%Y%m%d_%H%M%S) 2>/dev/null || true
    fi
    
    if [ -d "MWF_cleaned_Fuglsang" ]; then
        echo "Saving Fuglsang MWF partial results..."
        cp -r MWF_cleaned_Fuglsang MWF_cleaned_Fuglsang_backup_$(date +%Y%m%d_%H%M%S) 2>/dev/null || true
    fi
    
    # Try to save any log files
    if [ -f "mwf_processing.log" ]; then
        echo "Saving MWF processing log..."
        cp mwf_processing.log mwf_processing_backup_$(date +%Y%m%d_%H%M%S).log 2>/dev/null || true
    fi
}

# Set up timeout handler
trap timeout_handler SIGUSR1

# Function to check Python environment
check_python_env() {
    echo "=================================================================================="
    echo "CHECKING PYTHON ENVIRONMENT"
    echo "=================================================================================="
    
    echo "Python version: $(python3 --version 2>/dev/null || echo 'Python not found')"
    echo "Available memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
    echo "Available CPUs: $(nproc)"
    
    # Check required Python packages
    echo "Checking required Python packages..."
    python3 -c "
import sys
print(f'Python executable: {sys.executable}')

required_packages = ['numpy', 'scipy', 'matplotlib', 'tqdm', 'pathlib']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
        print(f'✓ {package} - Available')
    except ImportError:
        print(f'✗ {package} - MISSING')
        missing_packages.append(package)

if missing_packages:
    print(f'Missing packages: {missing_packages}')
    print('Attempting to install missing packages...')
    import subprocess
    for package in missing_packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f'✓ Installed {package}')
        except:
            print(f'✗ Failed to install {package}')
else:
    print('✓ All required packages are available!')
"
}

# Function to check data availability
check_data_availability() {
    echo "=================================================================================="
    echo "CHECKING DATA AVAILABILITY"
    echo "=================================================================================="
    
    # Check Das dataset
    if [ -d "Data/Das/4004271" ]; then
        das_files=$(find Data/Das/4004271 -name "S*.mat" 2>/dev/null | wc -l)
        echo "✓ Das dataset found: $das_files subject files"
    else
        echo "✗ Das dataset not found at Data/Das/4004271"
    fi
    
    # Check Fuglsang dataset
    FUGLSANG_EEG_DIR="${FUGLSANG_EEG_DIR:-/home/py9363/telluride_decoding/Data/Fulsang/EEG}"
    if [ -d "$FUGLSANG_EEG_DIR" ]; then
        fuglsang_files=$(find "$FUGLSANG_EEG_DIR" -name "S*.mat" 2>/dev/null | wc -l)
        echo "✓ Fuglsang dataset found: $fuglsang_files subject files"
        echo "  Path: $FUGLSANG_EEG_DIR"
    else
        echo "✗ Fuglsang dataset not found at $FUGLSANG_EEG_DIR"
    fi
}

# Function to run MWF processing
run_mwf_processing() {
    local dataset=$1
    local visualize=$2
    
    echo "=================================================================================="
    echo "RUNNING MWF ARTIFACT REMOVAL: $dataset DATASET"
    echo "=================================================================================="
    
    if [ ! -f "mwf_artifact_removal.py" ]; then
        echo "✗ mwf_artifact_removal.py not found!"
        echo "Please ensure the MWF script is available"
        return 1
    fi
    
    local cmd="python3 mwf_artifact_removal.py --dataset $dataset"
    
    # Add Fuglsang EEG directory if processing Fuglsang
    if [ "$dataset" = "fuglsang" ]; then
        FUGLSANG_EEG_DIR="${FUGLSANG_EEG_DIR:-/home/py9363/telluride_decoding/Data/Fulsang/EEG}"
        cmd="$cmd --fuglsang_eeg_dir $FUGLSANG_EEG_DIR"
    fi
    
    if [ "$visualize" = "true" ]; then
        cmd="$cmd --visualize"
    fi
    
    cmd="$cmd --unified"
    
    echo "Running command: $cmd"
    echo "=================================================================================="
    
    $cmd > mwf_processing_${dataset}.log 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "=================================================================================="
        echo "MWF PROCESSING COMPLETED SUCCESSFULLY FOR $dataset DATASET!"
        echo "Finished at: $(date)"
        echo "=================================================================================="
        
        # Check for results
        if [ "$dataset" = "das" ]; then
            if [ -d "MWF_cleaned_DAS" ]; then
                echo "Results directory: MWF_cleaned_DAS"
                echo "Generated files:"
                find MWF_cleaned_DAS -name "*.mat" | head -5
                echo "..."
            fi
        elif [ "$dataset" = "fuglsang" ]; then
            if [ -d "MWF_cleaned_Fuglsang" ]; then
                echo "Results directory: MWF_cleaned_Fuglsang"
                echo "Generated files:"
                find MWF_cleaned_Fuglsang -name "*.mat" | head -5
                echo "..."
            fi
        fi
        
        return 0
    else
        echo "=================================================================================="
        echo "MWF PROCESSING FAILED for $dataset dataset with exit code: $exit_code"
        echo "Check the error log: mwf_processing_${dataset}.log"
        echo "=================================================================================="
        tail -20 mwf_processing_${dataset}.log
        return $exit_code
    fi
}

# Function to create visualizations
create_visualizations() {
    echo "=================================================================================="
    echo "CREATING MWF VISUALIZATIONS"
    echo "=================================================================================="
    
    if [ ! -f "visualize_mwf_results.py" ]; then
        echo "✗ visualize_mwf_results.py not found!"
        echo "Skipping visualization step"
        return 1
    fi
    
    # Create visualizations for one subject from each dataset
    if [ -d "MWF_cleaned_DAS" ]; then
        das_subjects=$(find MWF_cleaned_DAS -name "S*_MWF.mat" | head -1)
        if [ -n "$das_subjects" ]; then
            subject_id=$(basename "$das_subjects" | sed 's/_MWF.mat//')
            echo "Creating visualization for Das dataset: $subject_id"
            python3 visualize_mwf_results.py --dataset das --subject "$subject_id" --trial 0 \
                --mwf_dir MWF_cleaned_DAS --output_dir Results/MWF_verification
        fi
    fi
    
    if [ -d "MWF_cleaned_Fuglsang" ]; then
        fuglsang_subjects=$(find MWF_cleaned_Fuglsang -name "sub*_MWF.mat" | head -1)
        if [ -n "$fuglsang_subjects" ]; then
            subject_id=$(basename "$fuglsang_subjects" | sed 's/_MWF.mat//')
            echo "Creating visualization for Fuglsang dataset: $subject_id"
            python3 visualize_mwf_results.py --dataset fuglsang --subject "$subject_id" --trial 0 \
                --mwf_dir MWF_cleaned_Fuglsang --output_dir Results/MWF_verification
        fi
    fi
    
    echo "Visualization complete!"
}

# Function to create final summary
create_final_summary() {
    echo "=================================================================================="
    echo "FINAL SUMMARY REPORT"
    echo "=================================================================================="
    echo "Algorithm: Multi-channel Wiener Filtering (MWF) Artifact Removal"
    echo "Finished at: $(date)"
    echo ""
    
    # Check Das results
    echo "DAS DATASET RESULTS:"
    echo "-------------------"
    if [ -d "MWF_cleaned_DAS" ]; then
        das_count=$(find MWF_cleaned_DAS -name "*.mat" | wc -l)
        echo "✓ Processing completed successfully"
        echo "✓ Generated $das_count MWF-cleaned files"
        if [ -f "MWF_cleaned_DAS/processing_summary.txt" ]; then
            echo "✓ Processing summary available"
        fi
    else
        echo "✗ No Das results found"
    fi
    
    # Check Fuglsang results
    echo ""
    echo "FUGLSANG DATASET RESULTS:"
    echo "-------------------------"
    if [ -d "MWF_cleaned_Fuglsang" ]; then
        fuglsang_count=$(find MWF_cleaned_Fuglsang -name "*.mat" | wc -l)
        echo "✓ Processing completed successfully"
        echo "✓ Generated $fuglsang_count MWF-cleaned files"
        if [ -f "MWF_cleaned_Fuglsang/processing_summary.txt" ]; then
            echo "✓ Processing summary available"
        fi
    else
        echo "✗ No Fuglsang results found"
    fi
    
    # Check visualizations
    echo ""
    echo "VISUALIZATIONS:"
    echo "---------------"
    if [ -d "Results/MWF_verification" ]; then
        vis_count=$(find Results/MWF_verification -name "*.png" | wc -l)
        echo "✓ Generated $vis_count visualization files"
    else
        echo "✗ No visualizations found"
    fi
    
    # List all generated files
    echo ""
    echo "GENERATED FILES:"
    echo "================"
    find . -name "*MWF*" -type f | grep -E "\.(log|mat|png|txt)$" | sort | head -20
    
    echo ""
    echo "=================================================================================="
    echo "MWF ARTIFACT REMOVAL COMPLETED"
    echo "=================================================================================="
}

# Main execution
main() {
    echo "Starting MWF artifact removal pipeline..."
    
    # Parse command line arguments
    PROCESS_DAS=false
    PROCESS_FUGLSANG=false
    VISUALIZE=false
    
    for arg in "$@"; do
        case $arg in
            --das)
                PROCESS_DAS=true
                ;;
            --fuglsang)
                PROCESS_FUGLSANG=true
                ;;
            --visualize)
                VISUALIZE=true
                ;;
            --both)
                PROCESS_DAS=true
                PROCESS_FUGLSANG=true
                ;;
        esac
    done
    
    # Default: process both if no specific dataset specified
    if [ "$PROCESS_DAS" = false ] && [ "$PROCESS_FUGLSANG" = false ]; then
        PROCESS_DAS=true
        PROCESS_FUGLSANG=true
    fi
    
    # Step 1: Check Python environment
    check_python_env
    
    # Step 2: Check data availability
    echo ""
    check_data_availability
    
    # Step 3: Run MWF processing
    echo ""
    DAS_SUCCESS=true
    FUGLSANG_SUCCESS=true
    
    if [ "$PROCESS_DAS" = true ]; then
        run_mwf_processing "das" "$VISUALIZE"
        DAS_SUCCESS=$?
    fi
    
    if [ "$PROCESS_FUGLSANG" = true ]; then
        echo ""
        run_mwf_processing "fuglsang" "$VISUALIZE"
        FUGLSANG_SUCCESS=$?
    fi
    
    # Step 4: Create additional visualizations if requested
    if [ "$VISUALIZE" = true ]; then
        echo ""
        create_visualizations
    fi
    
    # Step 5: Create final summary
    echo ""
    create_final_summary
    
    # Final status
    if [ $DAS_SUCCESS -eq 0 ] && [ $FUGLSANG_SUCCESS -eq 0 ]; then
        echo ""
        echo "🎉 SUCCESS: MWF artifact removal completed successfully!"
        echo "Check the results in MWF_cleaned_DAS/ and MWF_cleaned_Fuglsang/ directories"
        exit 0
    else
        echo ""
        echo "⚠ WARNING: Some processing steps failed"
        echo "Check the log files for details"
        exit 1
    fi
}

# Run main function
main "$@"

