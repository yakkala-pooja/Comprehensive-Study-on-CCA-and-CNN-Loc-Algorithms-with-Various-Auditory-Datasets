#!/usr/bin/env python3
"""
Test script to verify FULCCA GPU-only mode works correctly.
"""

import os
import sys
import numpy as np
import tensorflow as tf

# Set environment variables for robust GPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Allow GPU memory growth
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'  # Use private GPU threads
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # Use async GPU allocator

# Force GPU-only mode
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU only

# Configure GPU for maximum stability
try:
    # Check if GPU is available
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"Found {len(gpu_devices)} GPU device(s)")
        # Set memory growth and other GPU configurations
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
            # Set GPU memory limit to prevent OOM
            tf.config.experimental.set_memory_limit(gpu, 8192)  # 8GB limit
        print("✓ GPU memory growth and limits configured")
    else:
        raise RuntimeError("No GPU devices found! GPU-only mode requires GPU.")
except Exception as e:
    print(f"GPU configuration failed: {e}")
    raise RuntimeError("Cannot proceed without GPU. Please ensure GPU is available.")

# Enable TensorFlow v2 behavior
tf.compat.v1.enable_v2_behavior()

# Force GPU-only mode
device = tf.device('/GPU:0')
print("Using GPU for computation (GPU-only mode)")

# Set random seeds for reproducibility and stability
tf.random.set_seed(42)
np.random.seed(42)
print("✓ Random seeds set for reproducibility")

# Test basic TensorFlow operations
print("\nTesting TensorFlow GPU operations...")
try:
    with device:
        # Clear any existing GPU memory
        tf.keras.backend.clear_session()
        
        # Create a simple tensor
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[2.0, 0.0], [0.0, 2.0]])
        c = tf.matmul(a, b)
        print(f"✓ Matrix multiplication successful: {c.numpy()}")
        
        # Test random operations (this was causing the CUDA error)
        random_tensor = tf.random.uniform(shape=(10, 10), minval=0, maxval=1)
        print(f"✓ Random tensor creation successful: shape {random_tensor.shape}")
        
        # Test weight initialization (this was also causing issues)
        weights = tf.Variable(tf.random.uniform(shape=(5, 5), minval=-1, maxval=1))
        print(f"✓ Weight initialization successful: shape {weights.shape}")
        
except Exception as e:
    print(f"❌ TensorFlow GPU operation failed: {e}")
    sys.exit(1)

print("\n✓ All GPU tests passed! FULCCA GPU-only mode should work correctly now.")
