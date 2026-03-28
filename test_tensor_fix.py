#!/usr/bin/env python3
"""
Quick test to verify the tensor type fix works.
"""

import tensorflow as tf
import numpy as np

print("Testing tensor type fix...")

# Simulate the scenario
window_size = 512
batch_size = 4

# Create test data
pred_reshaped = tf.random.uniform(shape=(batch_size, window_size), minval=0, maxval=2, dtype=tf.int32)
pred_reshaped = tf.cast(pred_reshaped, tf.int64)

print(f"pred_reshaped shape: {pred_reshaped.shape}")
print(f"pred_reshaped dtype: {pred_reshaped.dtype}")

# Test the fixed aggregation logic
sample_predictions = tf.reduce_sum(pred_reshaped, axis=1)
print(f"sample_predictions shape: {sample_predictions.shape}")
print(f"sample_predictions dtype: {sample_predictions.dtype}")

# This should work now
sample_predictions = tf.cast(sample_predictions > (window_size // 2), tf.int64)
print(f"Final sample_predictions shape: {sample_predictions.shape}")
print(f"Final sample_predictions dtype: {sample_predictions.dtype}")
print(f"Final sample_predictions values: {sample_predictions.numpy()}")

print("✅ Tensor type fix test passed!")
