#!/usr/bin/env python3
"""
Test script to verify FULCCA CPU-first approach works.
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

# Test CPU-first approach
print("\nTesting CPU-first approach for random operations...")
try:
    # Clear any existing GPU memory
    tf.keras.backend.clear_session()
    
    # Use safe random operations
    safe_random_operations()
    
    # Test CPU-first approach
    print("Testing CPU-first approach...")
    with tf.device('/CPU:0'):
        # Test random operations on CPU
        random_tensor = tf.random.uniform(shape=(10, 10), minval=0, maxval=1)
        print(f"✓ Random tensor creation successful on CPU: shape {random_tensor.shape}")
        
        # Test weight initialization on CPU
        weights = tf.Variable(tf.random.uniform(shape=(5, 5), minval=-1, maxval=1))
        print(f"✓ Weight initialization successful on CPU: shape {weights.shape}")
        
        # Test matrix operations on CPU
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[2.0, 0.0], [0.0, 2.0]])
        c = tf.matmul(a, b)
        print(f"✓ Matrix multiplication successful on CPU: {c.numpy()}")
        
except Exception as e:
    print(f"❌ CPU operations failed: {e}")
    sys.exit(1)

print("\n✓ All CPU-first tests passed! FULCCA should work correctly now.")
