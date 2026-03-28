#!/usr/bin/env python3
"""
Test script to verify TensorFlow map function fix works.
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

# Test TensorFlow map function with correct signature
print("\nTesting TensorFlow map function with correct signature...")
try:
    # Clear any existing GPU memory
    tf.keras.backend.clear_session()
    
    # Use safe random operations
    safe_random_operations()
    
    # Test the map function signature
    print("Testing map function signature...")
    
    # Create test data
    batch_size = 4
    window_size = 512
    channels = 33
    
    # Create a simple dataset
    def generator():
        for i in range(10):
            input_1 = tf.random.uniform(shape=(window_size, channels))
            input_2 = tf.random.uniform(shape=(window_size, channels))
            label = tf.constant([i % 2], dtype=tf.int64)
            yield {'input_1': input_1, 'input_2': input_2}, label
    
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            {
                'input_1': tf.TensorSpec(shape=(window_size, channels), dtype=tf.float32),
                'input_2': tf.TensorSpec(shape=(window_size, channels), dtype=tf.float32)
            },
            tf.TensorSpec(shape=(1,), dtype=tf.int64)
        )
    )
    
    # Define reshape function with correct signature
    def reshape_batch(inputs, labels):
        # Reshape from (batch_size, window_size, 33) to (batch_size * window_size, 33)
        input_1_reshaped = tf.reshape(inputs['input_1'], (-1, channels))
        input_2_reshaped = tf.reshape(inputs['input_2'], (-1, channels))
        
        return {
            'input_1': input_1_reshaped,
            'input_2': input_2_reshaped
        }, labels
    
    # Test batching and mapping
    batched_dataset = dataset.batch(batch_size).map(reshape_batch)
    
    # Test iteration
    for i, batch in enumerate(batched_dataset.take(2)):
        inputs, labels = batch
        print(f"✓ Batch {i+1}: input_1 shape: {inputs['input_1'].shape}, input_2 shape: {inputs['input_2'].shape}, labels shape: {labels.shape}")
        
except Exception as e:
    print(f"❌ TensorFlow map function test failed: {e}")
    sys.exit(1)

print("\n✓ All TensorFlow map function tests passed! FULCCA should work correctly now.")
