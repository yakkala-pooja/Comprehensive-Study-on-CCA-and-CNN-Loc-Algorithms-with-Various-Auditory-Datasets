#!/bin/bash
# Quick Hyperparameter Tuning Script for FULCNN

echo "=================================================================================="
echo "FULCNN QUICK HYPERPARAMETER TUNING"
echo "=================================================================================="
echo "Started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=================================================================================="

# Set up environment
export CUDA_VISIBLE_DEVICES=0
cd /home/py9363/telluride_decoding

echo "Starting quick hyperparameter tuning..."
echo "This will test 6 key configurations focusing on the most impactful parameters"
echo ""

# Run quick tuning
python3 quick_tuning.py 2>&1 | tee quick_tuning.log

# Check if tuning completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================================================="
    echo "QUICK TUNING COMPLETED SUCCESSFULLY!"
    echo "Finished at: $(date)"
    echo "=================================================================================="
    
    # Display best configuration
    if [ -f "quick_tuning_results/quick_tuning_results.json" ]; then
        echo "Best configuration found:"
        python3 -c "
import json
with open('quick_tuning_results/quick_tuning_results.json', 'r') as f:
    data = json.load(f)
best_config = data['best_config']
best_score = data['best_score']
print(f'Configuration: {best_config[\"name\"]}')
print(f'Score: {best_score:.4f}')
print('Parameters:')
for key, value in best_config.items():
    if key not in ['name', 'tfrecord_dir', 'output_dir']:
        print(f'  {key}: {value}')
"
    fi
    
    echo ""
    echo "Files generated:"
    echo "  quick_tuning_results/quick_tuning_results.json"
    echo "  quick_tuning_results/config_*/ (individual run results)"
    echo "  quick_tuning.log"
    
else
    echo ""
    echo "=================================================================================="
    echo "QUICK TUNING FAILED!"
    echo "Check the log file: quick_tuning.log"
    echo "=================================================================================="
    exit 1
fi

echo ""
echo "=================================================================================="
echo "NEXT STEPS"
echo "=================================================================================="
echo "1. Review the best configuration in quick_tuning_results/"
echo "2. Run production training with the best parameters:"
echo "   python3 FULCNN.py --batch_size <best_batch> --learning_rate <best_lr> ..."
echo "3. For more comprehensive tuning, run: python3 hyperparameter_tuning.py"
echo "4. If results are still poor, consider architecture changes or data augmentation"
echo ""
echo "🎉 Quick tuning completed!"
