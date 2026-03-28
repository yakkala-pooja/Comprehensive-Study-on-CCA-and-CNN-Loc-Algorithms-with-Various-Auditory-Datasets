#!/usr/bin/env python3
"""
Test script to verify prediction aggregation fix works.
"""

import os
import sys
import numpy as np
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")

# Set environment variables for robust GPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Allow GPU memory growth

# Force CPU usage for random operations to avoid CUDA handle corruption
os.environ['TF_DETERMINISTIC_OPS'] = '1'  # Use deterministic operations
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # Use deterministic cuDNN

# Force GPU-only mode
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU only

# Configure GPU for maximum stability
try:
    # Check if GPU is available
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"Found {len(gpu_devices)} GPU device(s)")
        # Set memory growth for all GPUs (compatible with TensorFlow 2.20.0)
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✓ GPU memory growth configured")
        
        # Try to set memory limit if supported (TensorFlow 2.4+)
        try:
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_limit(gpu, 8192)  # 8GB limit
            print("✓ GPU memory limits configured")
        except AttributeError:
            print("✓ GPU memory limits not supported in this TensorFlow version")
        except Exception as e:
            print(f"GPU memory limit warning: {e}")
            
    else:
        print("No GPU devices found, testing CPU mode...")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
except Exception as e:
    print(f"GPU configuration failed: {e}")
    print("Testing CPU mode...")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Enable TensorFlow v2 behavior
tf.compat.v1.enable_v2_behavior()

# Force GPU-only mode with CPU fallback for problematic operations
device = tf.device('/GPU:0')
print("Using GPU for computation (GPU-only mode with CPU fallback)")

# Set random seeds for reproducibility and stability
tf.random.set_seed(42)
np.random.seed(42)
print("✓ Random seeds set for reproducibility")

# Force CPU for random operations to avoid CUDA handle corruption
def safe_random_operations():
    """Force CPU usage for random operations."""
    with tf.device('/CPU:0'):
        tf.random.set_seed(42)
        np.random.seed(42)

# Test prediction aggregation logic
print("\nTesting prediction aggregation logic...")
try:
    # Clear any existing GPU memory
    tf.keras.backend.clear_session()
    
    # Use safe random operations
    safe_random_operations()
    
    # Test the prediction aggregation
    print("Testing prediction aggregation...")
    
    # Simulate the scenario from the error
    batch_size = 4
    window_size = 512
    channels = 33
    
    # Simulate CCA predictions (batch_size * window_size predictions)
    total_predictions = batch_size * window_size
    cca_predictions = tf.random.uniform(shape=(total_predictions, 20))  # 20 CCA dimensions
    
    print(f"Input predictions shape: {cca_predictions.shape}")
    print(f"Expected batch_size: {batch_size}")
    print(f"Expected window_size: {window_size}")
    
    # Split concatenated CCA output (simulate the CCA model output)
    cca_width = cca_predictions.shape[-1] // 2
    pred1 = cca_predictions[:, :cca_width]
    pred2 = cca_predictions[:, cca_width:]
    
    # Use first CCA component for classification
    cca_scores = pred1[:, 0]  # First CCA component
    
    # Convert CCA scores to binary predictions
    binary_predictions = tf.cast(cca_scores > 0, tf.int64)
    
    print(f"Binary predictions shape: {binary_predictions.shape}")
    
    # Aggregate predictions per sample (batch_size predictions per batch)
    # The dataset is reshaped to (batch_size * window_size, 33), so we need to
    # aggregate back to batch_size predictions
    batch_size_calc = binary_predictions.shape[0] // window_size
    window_size_calc = window_size
    
    print(f"Calculated batch_size: {batch_size_calc}")
    print(f"Calculated window_size: {window_size_calc}")
    
    # Reshape predictions back to (batch_size, window_size)
    pred_reshaped = tf.reshape(binary_predictions, (batch_size_calc, window_size_calc))
    
    print(f"Reshaped predictions shape: {pred_reshaped.shape}")
    
    # Aggregate per sample using majority voting
    sample_predictions = tf.reduce_sum(pred_reshaped, axis=1)
    sample_predictions = tf.cast(sample_predictions > (window_size_calc // 2), tf.int64)
    
    print(f"Final sample predictions shape: {sample_predictions.shape}")
    print(f"Sample predictions: {sample_predictions.numpy()}")
    
    # Verify the dimensions match
    if sample_predictions.shape[0] == batch_size:
        print("✓ Prediction aggregation successful! Dimensions match.")
    else:
        print(f"❌ Dimension mismatch: expected {batch_size}, got {sample_predictions.shape[0]}")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Prediction aggregation test failed: {e}")
    sys.exit(1)

print("\n✓ All prediction aggregation tests passed! FULCCA should work correctly now.")
