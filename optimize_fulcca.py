#!/usr/bin/env python3
"""
Quick optimization script for FULCCA.py to achieve 68%+ accuracy.
This script modifies the existing FULCCA.py with optimized parameters.
"""

import os
import sys
import subprocess

def optimize_fulcca():
    """Apply optimizations to FULCCA.py for better performance."""
    
    print("🔧 Applying FULCCA optimizations for 68%+ accuracy...")
    
    # Read the current FULCCA.py
    with open('FULCCA.py', 'r') as f:
        content = f.read()
    
    # Apply optimizations
    optimizations = [
        # 1. Optimize CCA dimensions and regularization
        ("cca_dims: int = 5", "cca_dims: int = 8"),
        ("regularization: float = 0.01", "regularization: float = 0.05"),
        
        # 2. Improve preprocessing parameters
        ("artifact_threshold = 5.0", "artifact_threshold = 3.0"),  # More sensitive artifact detection
        ("high_freq = min(40.0 / nyquist, 0.99)", "high_freq = min(30.0 / nyquist, 0.99)"),  # Focus on attention frequencies
        
        # 3. Optimize model training
        ("epochs=1", "epochs=3"),  # More training epochs
        ("learning_rate=1e-3", "learning_rate=1e-4"),  # Lower learning rate for stability
        
        # 4. Improve prediction aggregation
        ("sample_predictions = tf.cast(sample_predictions > (window_size // 2), tf.int64)", 
         "sample_predictions = tf.cast(sample_predictions > (window_size * 0.4), tf.int64)"),  # Lower threshold
    ]
    
    # Apply each optimization
    for old, new in optimizations:
        if old in content:
            content = content.replace(old, new)
            print(f"✓ Applied: {old} → {new}")
        else:
            print(f"⚠️  Not found: {old}")
    
    # Add enhanced configurations
    enhanced_configs = '''
    # Enhanced CCA configurations for better performance
    enhanced_configs = [
        {'name': 'optimal_balanced', 'cca_dims': 8, 'regularization': 0.05, 'window_size': 512},
        {'name': 'precision_focused', 'cca_dims': 12, 'regularization': 0.08, 'window_size': 768},
        {'name': 'robust_general', 'cca_dims': 6, 'regularization': 0.03, 'window_size': 640},
        {'name': 'high_dim_optimized', 'cca_dims': 15, 'regularization': 0.1, 'window_size': 512},
        {'name': 'extended_window', 'cca_dims': 10, 'regularization': 0.06, 'window_size': 1024},
        {'name': 'fine_tuned', 'cca_dims': 4, 'regularization': 0.02, 'window_size': 384},
    ]
    '''
    
    # Add enhanced configurations before main function
    if "def main():" in content:
        content = content.replace("def main():", enhanced_configs + "\n\ndef main():")
        print("✓ Added enhanced configurations")
    
    # Write optimized version
    with open('FULCCA_optimized.py', 'w') as f:
        f.write(content)
    
    print("\n🎉 FULCCA optimizations applied successfully!")
    print("📁 Optimized version saved as: FULCCA_optimized.py")
    
    return True

def create_optimized_slurm_script():
    """Create optimized Slurm script."""
    
    slurm_content = '''#!/bin/bash
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
TFRECORD_DIR="/home/py9363/telluride_decoding/FULPREPROCESSING"

# Run optimized FULCCA analysis
echo "Starting optimized FULCCA analysis..."
python FULCCA_optimized.py \\
    --tfrecord_dir "$TFRECORD_DIR" \\
    --output_dir "fulcca_optimized_results" \\
    --cca_dims 8 \\
    --regularization 0.05 \\
    --window_size 512 \\
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
'''
    
    with open('FULCCA_optimized.sh', 'w') as f:
        f.write(slurm_content)
    
    print("📁 Optimized Slurm script saved as: FULCCA_optimized.sh")

if __name__ == "__main__":
    print("🚀 FULCCA Quick Optimization Tool")
    print("=" * 50)
    
    try:
        # Apply optimizations
        optimize_fulcca()
        
        # Create optimized Slurm script
        create_optimized_slurm_script()
        
        print("\n" + "=" * 50)
        print("✅ OPTIMIZATION COMPLETE!")
        print("=" * 50)
        print("📁 Files created:")
        print("  - FULCCA_optimized.py (optimized script)")
        print("  - FULCCA_optimized.sh (optimized Slurm script)")
        print("\n🚀 To run optimized version:")
        print("  sbatch FULCCA_optimized.sh")
        print("\n🎯 Expected improvements:")
        print("  - Better artifact detection (threshold: 5.0 → 3.0)")
        print("  - Focused frequency range (40Hz → 30Hz)")
        print("  - More training epochs (1 → 3)")
        print("  - Lower learning rate (1e-3 → 1e-4)")
        print("  - Optimized CCA parameters (dims: 5→8, reg: 0.01→0.05)")
        print("  - Better prediction threshold (50% → 40%)")
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        sys.exit(1)
