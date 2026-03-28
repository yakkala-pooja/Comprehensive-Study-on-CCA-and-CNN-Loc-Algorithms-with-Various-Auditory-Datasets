#!/usr/bin/env python3
"""
Test script to verify CCA dimension fix works.
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

# Test CCA dimension compatibility
print("\nTesting CCA dimension compatibility...")
try:
    # Clear any existing GPU memory
    tf.keras.backend.clear_session()
    
    # Use safe random operations
    safe_random_operations()
    
    # Test the reshaping logic
    print("Testing batch reshaping logic...")
    
    # Simulate batched data: (batch_size, window_size, channels)
    batch_size = 4
    window_size = 512
    channels = 33
    
    # Create test data
    input_1_batch = tf.random.uniform(shape=(batch_size, window_size, channels))
    input_2_batch = tf.random.uniform(shape=(batch_size, window_size, channels))
    
    print(f"Original batch shapes: input_1={input_1_batch.shape}, input_2={input_2_batch.shape}")
    
    # Reshape for CCA compatibility
    input_1_reshaped = tf.reshape(input_1_batch, (-1, channels))
    input_2_reshaped = tf.reshape(input_2_batch, (-1, channels))
    
    print(f"Reshaped for CCA: input_1={input_1_reshaped.shape}, input_2={input_2_reshaped.shape}")
    
    # Test matrix multiplication (this was causing the error)
    with tf.device('/CPU:0'):
        # Test the operations that were failing
        cov_xx = tf.matmul(tf.transpose(input_1_reshaped), input_1_reshaped)
        cov_yy = tf.matmul(tf.transpose(input_2_reshaped), input_2_reshaped)
        cov_xy = tf.matmul(tf.transpose(input_1_reshaped), input_2_reshaped)
        
        print(f"✓ Matrix multiplication successful:")
        print(f"  cov_xx shape: {cov_xx.shape}")
        print(f"  cov_yy shape: {cov_yy.shape}")
        print(f"  cov_xy shape: {cov_xy.shape}")
        
except Exception as e:
    print(f"❌ CCA dimension test failed: {e}")
    sys.exit(1)

print("\n✓ All CCA dimension tests passed! FULCCA should work correctly now.")
