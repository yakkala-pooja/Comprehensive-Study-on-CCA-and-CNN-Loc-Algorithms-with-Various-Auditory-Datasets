#!/usr/bin/env python3
"""
CombinedCCA - Optimal CCA Algorithm for Combined Das and Fulsang Dataset

This script implements CCA (Canonical Correlation Analysis) for the combined dataset
using the Optimal_FULCCA configuration, adapted for CombinedDataset.
"""

import os
# Force CPU usage to avoid CUDA compatibility issues
# Set this BEFORE importing tensorflow to ensure it takes effect
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
import subprocess
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add telluride_decoding to path
sys.path.append('telluride_decoding')
sys.path.append('.')

# Fixed CCA calculation function with progress logging and batch limiting
# This function must be defined BEFORE the import that uses it
def calculate_cca_parameters_fixed(dataset, dim, regularization=0.1, 
                                   mini_batch_count=0, eps_eig=1e-12):
    """Fixed version of calculate_cca_parameters_from_dataset with robust batch handling.

    Behavior:
    - If mini_batch_count > 0: use that many batches (or fewer if dataset is smaller).
    - If mini_batch_count == 0 or None:
        * If dataset has finite known cardinality: use ALL batches.
        * If dataset is infinite/unknown: cap at 10,000 batches to avoid hanging.
    """
    import logging
    import tensorflow as tf
    
    # Robust handling of mini_batch_count
    if mini_batch_count == 0 or mini_batch_count is None:
        try:
            card = tf.data.experimental.cardinality(dataset)
        except Exception:
            card = tf.data.experimental.UNKNOWN_CARDINALITY
        
        if card == tf.data.experimental.INFINITE_CARDINALITY:
            mini_batch_count = 10000
            logging.warning(
                'Dataset has infinite cardinality; capping CCA processing to %d batches '
                'to avoid an infinite loop.', mini_batch_count
            )
        elif card == tf.data.experimental.UNKNOWN_CARDINALITY:
            mini_batch_count = 10000
            logging.warning(
                'Dataset cardinality is unknown; capping CCA processing to %d batches. '
                'If you want fewer/more, pass mini_batch_count explicitly.',
                mini_batch_count
            )
        else:
            # Use the full finite dataset
            mini_batch_count = int(card.numpy())
            logging.info(
                'mini_batch_count was 0; detected finite dataset with %d batches. '
                'Processing ALL batches for CCA.', mini_batch_count
            )
    
    if not isinstance(dataset, tf.data.Dataset) and not isinstance(dataset, tf.data.DatasetV2):
        raise TypeError('dataset must be a tf.data.Dataset object')
    if regularization < 0.0:
        raise ValueError('regularization lambda must be >= 0')
    
    logging.info('Calculating CCA parameters from a dataset with %s.', dataset)
    logging.info(' Looking for %d output dimensions with regularization = %g.', dim, regularization)
    logging.info(' Processing up to %d batches (with progress updates every 100 batches)...', mini_batch_count)
    
    cov_xx = 0
    cov_yy = 0
    cov_xy = 0
    sum_x = 0
    sum_y = 0
    num_mini_batches = 0
    total_frames = 0
    
    # Use take() with the fixed batch count
    dataset_iter = dataset.take(mini_batch_count)
    
    for batch_idx, dataset_item in enumerate(dataset_iter):
        # Progress logging every 100 batches
        if batch_idx > 0 and batch_idx % 100 == 0:
            logging.info('  Processed %d/%d batches...', batch_idx, mini_batch_count)
        
        # Handle both tuple format (x_dict, y) and dict format (just x_dict)
        if isinstance(dataset_item, tuple):
            (x_dict, y) = dataset_item
        else:
            x_dict = dataset_item
            y = None
        
        if not isinstance(x_dict, dict):
            raise TypeError('X_dict is a %s, not a dict.' % type(x_dict))
        
        x = x_dict['input_1'].numpy()
        y = x_dict['input_2'].numpy()
        
        if x.shape[1] == 0:
            raise ValueError('First input to CCA estimator must have more than 0 columns.')
        if y.shape[1] == 0:
            raise ValueError('Second input to CCA estimator must have more than 0 columns.')
        
        n_row = x.shape[0]
        total_frames += x.shape[0]
        cov_xx += np.matmul(x.T, x)
        cov_yy += np.matmul(y.T, y)
        cov_xy += np.matmul(x.T, y)
        sum_x += np.sum(x, axis=0, keepdims=True)
        sum_y += np.sum(y, axis=0, keepdims=True)
        num_mini_batches += 1
        
        if mini_batch_count and num_mini_batches >= mini_batch_count:
            break
    
    logging.info('Calculating the CCA parameters from %d minibatches', num_mini_batches)
    if not num_mini_batches:
        raise ValueError('No minibatches in dataset, can\'t compute CCA model.')
    
    mean_x = sum_x / total_frames
    mean_y = sum_y / total_frames
    cov_xx = cov_xx / (num_mini_batches * n_row - 1) - np.matmul(mean_x.T, mean_x)
    cov_xx += regularization * np.eye(x.shape[1])
    cov_yy = cov_yy / (num_mini_batches * n_row - 1) - np.matmul(mean_y.T, mean_y)
    cov_yy += regularization * np.eye(y.shape[1])
    cov_xy = cov_xy / (num_mini_batches * n_row - 1) - np.matmul(mean_x.T, mean_y)
    
    logging.info('Computing eigendecomposition...')
    x_vals, x_vecs = np.linalg.eigh(cov_xx)
    y_vals, y_vecs = np.linalg.eigh(cov_yy)
    
    # For numerical stability
    idx1 = np.where(x_vals > eps_eig)[0]
    x_vals = x_vals[idx1]
    x_vecs = x_vecs[:, idx1]
    
    idx2 = np.where(y_vals > eps_eig)[0]
    y_vals = y_vals[idx2]
    y_vecs = y_vecs[:, idx2]
    
    logging.info('Computing CCA rotations...')
    k11 = np.matmul(np.matmul(x_vecs, np.diag(np.reciprocal(np.sqrt(x_vals)))),
                    x_vecs.transpose())
    k22 = np.matmul(np.matmul(y_vecs, np.diag(np.reciprocal(np.sqrt(y_vals)))),
                    y_vecs.transpose())
    t = np.matmul(np.matmul(k11, cov_xy), k22)
    u, e, v = np.linalg.svd(t, full_matrices=False)
    v = v.transpose()
    
    rot_x = np.matmul(k11, u[:, 0:dim])
    rot_y = np.matmul(k22, v[:, 0:dim])
    e = e[0:dim]
    
    logging.info('CCA calculation complete!')
    return rot_x, rot_y, mean_x, mean_y, e

try:
    from telluride_decoding.cca import BrainModelCCA, cca_pearson_correlation_first
    import telluride_decoding.cca as cca_module
    import logging
    # Monkey-patch the CCA calculation function to fix the infinite loop issue
    # This replaces the original function with our fixed version that handles mini_batch_count=0 properly
    cca_module.calculate_cca_parameters_from_dataset = calculate_cca_parameters_fixed
except ImportError as e:
    print(f"Error: Could not import telluride_decoding CCA modules: {e}")
    sys.exit(1)

# Import combined dataset
from CombinedDataset import CombinedDataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# Do NOT override CUDA_VISIBLE_DEVICES here; let the environment / modules
# decide which GPUs are visible so we can use the cluster's GPU-enabled tf.

tf.compat.v1.enable_v2_behavior()

# Force CPU usage to avoid CUDA compatibility issues
# The CUDA error (CUDA_ERROR_UNSUPPORTED_PTX_VERSION) indicates GPU driver/CUDA version mismatch
# Setting CUDA_VISIBLE_DEVICES before TensorFlow operations ensures CPU-only mode
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
device = tf.device('/CPU:0')
print("⚠ Using CPU for CCA training (GPU disabled due to CUDA compatibility issues)")

tf.random.set_seed(42)
np.random.seed(42)

# Optimal Configuration (based on Optimal_FULCCA opt_3)
# IMPROVED: Increased window size, reduced regularization, more CCA dimensions
# Can use 512 (4s) or 1024 (8s) for better signal capture
OPTIMAL_CONFIG = {
    'name': 'opt_3_combined_improved',
    'cca_dims': 16,  # Increased from 12 for better signal capture
    'regularization': 0.02,  # Reduced from 0.08 to allow stronger signal
    'window_size': 512,  # Can be 512 (4s) or 1024 (8s) at 128 Hz
    'batch_size': 6
}


class CombinedCCADataset:
    """
    TensorFlow dataset wrapper that pairs EEG windows with real stimulus envelopes.
    
    - input_1: flattened EEG window
    - input_2: attended stimulus envelope for that window
    
    The generator also returns auxiliary data (left/right envelopes + label)
    so evaluation can correlate EEG with both candidate envelopes.
    """
    
    def __init__(self, combined_dataset: CombinedDataset, mode: str = 'full'):
        self.combined_dataset = combined_dataset
        self.mode = mode
        self.window_size = combined_dataset.window_size
        self.sampling_rate = combined_dataset.sampling_rate
        self.n_channels = combined_dataset.n_channels
        
        # Get window indices
        self.window_indices = combined_dataset.get_window_indices()
        
        print(f"\nCombinedCCADataset initialized:")
        print(f"  Mode: {mode}")
        print(f"  Total windows: {len(self.window_indices)}")
        print(f"  Window size: {self.window_size} samples")
        print(f"  Sampling rate: {self.sampling_rate} Hz")
        print(f"  Channels: {self.n_channels}")
        
        # Validate data quality and alignment
        self._validate_data_quality()
        
        # Validate timing synchronization
        timing_validation = self.combined_dataset.validate_timing_synchronization(n_samples=50)
        if not timing_validation.get('valid', False):
            print(f"⚠️  Timing synchronization issues detected: {timing_validation.get('status', 'unknown')}")
        else:
            print(f"✓ Timing synchronization validated")
    
    def _validate_data_quality(self):
        """Validate data quality and alignment between Das and Fulsang datasets."""
        print("\n" + "="*60)
        print("DATA QUALITY VALIDATION")
        print("="*60)
        
        # Check EEG data quality
        eeg_data = self.combined_dataset.eeg_data
        print(f"\n1. EEG Data Quality:")
        print(f"   Shape: {eeg_data.shape}")
        print(f"   Mean: {np.mean(eeg_data):.6f}")
        print(f"   Std: {np.std(eeg_data):.6f}")
        print(f"   NaN count: {np.sum(np.isnan(eeg_data))}")
        print(f"   Inf count: {np.sum(np.isinf(eeg_data))}")
        
        # Check per-channel variance
        channel_vars = np.var(eeg_data, axis=0)
        zero_var_channels = np.sum(channel_vars == 0)
        print(f"   Channels with zero variance: {zero_var_channels}/{len(channel_vars)}")
        if zero_var_channels > 0:
            print(f"   ⚠️  WARNING: {zero_var_channels} channels have zero variance!")
        
        # Check label distribution
        labels = self.combined_dataset.labels
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\n2. Label Distribution:")
        for u, c in zip(unique, counts):
            print(f"   Label {u}: {c} samples ({100*c/len(labels):.1f}%)")
        
        # Check envelope availability
        sample_indices = np.random.choice(len(self.window_indices), min(10, len(self.window_indices)), replace=False)
        envelope_checks = []
        for idx in sample_indices:
            start_idx, end_idx, label = self.window_indices[idx]
            try:
                left_env, right_env = self.combined_dataset.get_envelope_window(start_idx, end_idx)
                if left_env is not None and right_env is not None:
                    envelope_checks.append(True)
                else:
                    envelope_checks.append(False)
            except Exception as e:
                envelope_checks.append(False)
        
        valid_envelopes = np.sum(envelope_checks)
        print(f"\n3. Envelope Availability:")
        print(f"   Valid envelopes: {valid_envelopes}/{len(envelope_checks)} sampled windows")
        if valid_envelopes < len(envelope_checks):
            print(f"   ⚠️  WARNING: Some windows have missing envelopes!")
        
        # Check envelope differences and quality
        if valid_envelopes > 0:
            left_envs = []
            right_envs = []
            envelope_stats = {'left': [], 'right': []}
            
            for idx in sample_indices[:valid_envelopes]:
                start_idx, end_idx, label = self.window_indices[idx]
                try:
                    left_env, right_env = self.combined_dataset.get_envelope_window(start_idx, end_idx)
                    if left_env is not None and right_env is not None:
                        left_flat = left_env.flatten()
                        right_flat = right_env.flatten()
                        left_envs.append(left_flat)
                        right_envs.append(right_flat)
                        
                        # Collect statistics
                        envelope_stats['left'].append({
                            'mean': np.mean(left_flat),
                            'std': np.std(left_flat),
                            'min': np.min(left_flat),
                            'max': np.max(left_flat),
                            'energy': np.sum(left_flat**2)
                        })
                        envelope_stats['right'].append({
                            'mean': np.mean(right_flat),
                            'std': np.std(right_flat),
                            'min': np.min(right_flat),
                            'max': np.max(right_flat),
                            'energy': np.sum(right_flat**2)
                        })
                except Exception as e:
                    pass
            
            if left_envs and right_envs:
                left_envs = np.array(left_envs)
                right_envs = np.array(right_envs)
                
                # Overall correlation
                env_correlation = np.corrcoef(left_envs.flatten(), right_envs.flatten())[0, 1]
                print(f"   Left-Right envelope correlation: {env_correlation:.4f}")
                if env_correlation > 0.9:
                    print(f"   ⚠️  WARNING: Envelopes are very similar (correlation > 0.9)")
                    print(f"      This may make it difficult to distinguish attended vs unattended")
                
                # Per-window correlation analysis
                per_window_corrs = []
                for i in range(len(left_envs)):
                    corr = np.corrcoef(left_envs[i], right_envs[i])[0, 1]
                    per_window_corrs.append(corr)
                
                print(f"   Per-window correlation: {np.mean(per_window_corrs):.4f} ± {np.std(per_window_corrs):.4f}")
                
                # Envelope quality metrics
                if envelope_stats['left'] and envelope_stats['right']:
                    left_means = [s['mean'] for s in envelope_stats['left']]
                    right_means = [s['mean'] for s in envelope_stats['right']]
                    left_energies = [s['energy'] for s in envelope_stats['left']]
                    right_energies = [s['energy'] for s in envelope_stats['right']]
                    
                    print(f"   Left envelope - Mean: {np.mean(left_means):.6f}, Energy: {np.mean(left_energies):.6f}")
                    print(f"   Right envelope - Mean: {np.mean(right_means):.6f}, Energy: {np.mean(right_energies):.6f}")
                    
                    # Check if envelopes have sufficient variation
                    left_variation = np.mean([s['std'] for s in envelope_stats['left']])
                    right_variation = np.mean([s['std'] for s in envelope_stats['right']])
                    
                    if left_variation < 0.01 or right_variation < 0.01:
                        print(f"   ⚠️  WARNING: Low envelope variation (left: {left_variation:.6f}, right: {right_variation:.6f})")
                        print(f"      Envelopes may be too flat/similar to decode attention")
                    
                    # Check energy difference
                    energy_diff = abs(np.mean(left_energies) - np.mean(right_energies))
                    if energy_diff < 0.001:
                        print(f"   ⚠️  WARNING: Very similar envelope energies (diff: {energy_diff:.6f})")
        
        # Check window size compatibility
        print(f"\n4. Window Size Validation:")
        if self.window_size < 256:
            print(f"   ⚠️  WARNING: Window size {self.window_size} may be too short for reliable decoding")
            print(f"      Recommended: >= 512 samples (4+ seconds at 128 Hz)")
        else:
            print(f"   ✓ Window size {self.window_size} is adequate")
        
        print("="*60 + "\n")
    
    def _preprocess_window(self, eeg_window: np.ndarray) -> np.ndarray:
        """Preprocess EEG window for CCA (per-channel standardization + gentle soft clip).
        
        IMPROVED: Reduced aggressive clipping to preserve more signal.
        """
        # Baseline correction
        eeg_window = eeg_window - np.mean(eeg_window, axis=0, keepdims=True)
        
        # Normalization
        std_vals = np.std(eeg_window, axis=0, keepdims=True)
        std_vals = np.where(std_vals == 0, 1.0, std_vals)
        eeg_window = eeg_window / std_vals
        
        # Gentle soft clipping (reduced from 0.5 to 1.0 to preserve more signal)
        # Only clips extreme outliers, preserves most of the signal
        eeg_window = np.tanh(eeg_window * 1.0)
        
        # Ensure valid values
        eeg_window = np.nan_to_num(eeg_window, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return eeg_window.astype(np.float32)
    
    def create_tf_dataset(self, batch_size: int = 6) -> tf.data.Dataset:
        """Create TensorFlow dataset from combined data for CCA."""
        feature_dim = self.window_size * self.n_channels
        
        def generator():
            for start_idx, end_idx, label in self.window_indices:
                eeg_window = self.combined_dataset.eeg_data[start_idx:end_idx]
                eeg_window = self._preprocess_window(eeg_window)
                
                left_env, right_env = self.combined_dataset.get_envelope_window(start_idx, end_idx)
                left_env = left_env.flatten().astype(np.float32)
                right_env = right_env.flatten().astype(np.float32)
                attended_env = left_env if label == 0 else right_env
                
                eeg_flat = eeg_window.flatten().astype(np.float32)
                
                yield (
                    {
                        'input_1': eeg_flat,
                        'input_2': attended_env
                    },
                    {
                        'left_env': left_env,
                        'right_env': right_env,
                        'label': np.array(label, dtype=np.float32)
                    }
                )
        
        input_signature = {
            'input_1': tf.TensorSpec(shape=(feature_dim,), dtype=tf.float32),
            'input_2': tf.TensorSpec(shape=(self.window_size,), dtype=tf.float32)
        }
        aux_signature = {
            'left_env': tf.TensorSpec(shape=(self.window_size,), dtype=tf.float32),
            'right_env': tf.TensorSpec(shape=(self.window_size,), dtype=tf.float32),
            'label': tf.TensorSpec(shape=(), dtype=tf.float32)
        }
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(input_signature, aux_signature)
        )
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


class OptimalCombinedCCAModel:
    """
    Optimal Combined CCA model with opt_3 configuration.
    Based on Optimal_FULCCA but adapted for CombinedDataset.
    """
    
    def __init__(self, cca_dims: int = 16, regularization: float = 0.02, window_size: int = 512):
        self.cca_dims = cca_dims
        self.regularization = regularization
        self.window_size = window_size
        self.model = None
        self.is_fitted = False
    
    def fit(self, dataset: tf.data.Dataset):
        """Fit the optimal CCA model with class balancing."""
        # Limit dataset size to prevent infinite loops (max 10000 batches for CCA calculation)
        MAX_BATCHES_FOR_CCA = 10000
        print(f"\nLimiting dataset to {MAX_BATCHES_FOR_CCA} batches for CCA calculation...")
        limited_dataset = dataset.take(MAX_BATCHES_FOR_CCA)
        
        try:
            class_weights = self._calculate_class_weights(limited_dataset)
        except Exception:
            class_weights = {0: 1.0, 1: 1.0}
        
        self.model = self._create_optimal_cca_model(limited_dataset)
        # Force CPU usage for model compilation to avoid CUDA errors
        with tf.device('/CPU:0'):
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss=self._create_weighted_loss(class_weights),
                metrics=[cca_pearson_correlation_first]
            )
        print("Starting CCA model fitting (this may take a while)...")
        # Force CPU usage for model fitting
        with tf.device('/CPU:0'):
            fit_results = self.model.fit(limited_dataset, epochs=2)
        self.is_fitted = True
        
        # Log CCA training diagnostics
        print("\n" + "="*80)
        print("CCA TRAINING DIAGNOSTICS")
        print("="*80)
        if hasattr(self.model, 'rot_x') and hasattr(self.model, 'rot_y'):
            # Extract eigenvalues from the model if available
            # The eigenvalues are logged during CCA calculation, but we can check them here
            print("✓ CCA model fitted successfully")
            print(f"  CCA dimensions: {self.cca_dims} (requested: {self.cca_dims})")
            print(f"  Regularization: {self.regularization} (requested: {self.regularization})")
            print(f"  Window size: {self.window_size} samples")
            print(f"  Input 1 (EEG) dimensions: {self.model._input1_width}")
            print(f"  Input 2 (Envelope) dimensions: {self.model._input2_width}")
            
            # Verify the model was created with correct parameters
            if hasattr(self.model, '_cca_dims'):
                actual_dims = self.model._cca_dims
                if actual_dims != self.cca_dims:
                    print(f"  ⚠️  WARNING: Model CCA dims ({actual_dims}) != requested ({self.cca_dims})")
        print("="*80 + "\n")
    
    def _create_optimal_cca_model(self, dataset: tf.data.Dataset):
        """Create optimal CCA model using telluride_decoding implementation."""
        with tf.device('/CPU:0'):
            cca_model = BrainModelCCA(
                input_dataset=dataset,
                cca_dims=self.cca_dims,
                regularization_lambda=self.regularization
            )
        return cca_model
    
    def _calculate_class_weights(self, dataset: tf.data.Dataset) -> Dict[int, float]:
        """Calculate class weights for imbalanced data.
        
        Uses the auxiliary labels from the dataset to properly count class distribution.
        """
        all_labels = []
        for batch in dataset:
            # Extract labels from auxiliary data (not from input_2 which is the envelope)
            if isinstance(batch, tuple):
                inputs, aux = batch
                if 'label' in aux:
                    labels = aux['label'].numpy()
                    if labels.ndim == 0:
                        all_labels.append(int(labels))
                    else:
                        all_labels.extend(labels.astype(int).tolist())
            elif isinstance(batch, dict) and 'label' in batch:
                labels = batch['label'].numpy()
                if labels.ndim == 0:
                    all_labels.append(int(labels))
                else:
                    all_labels.extend(labels.astype(int).tolist())
        
        if not all_labels:
            print("⚠️  Warning: Could not extract labels for class weighting, using equal weights")
            return {0: 1.0, 1: 1.0}
        
        all_labels = np.array(all_labels)
        unique_classes, class_counts = np.unique(all_labels, return_counts=True)
        total_samples = len(all_labels)
        n_classes = len(unique_classes)
        
        print(f"\nClass distribution for weighting:")
        for i, class_id in enumerate(unique_classes):
            count = class_counts[i]
            percentage = 100.0 * count / total_samples
            print(f"  Class {class_id}: {count} samples ({percentage:.1f}%)")
        
        class_weights = {}
        for i, class_id in enumerate(unique_classes):
            if class_counts[i] > 0:
                # Use balanced weighting: total_samples / (n_classes * class_count)
                weight = total_samples / (n_classes * class_counts[i])
                class_weights[int(class_id)] = float(weight)
                print(f"  Class {class_id} weight: {weight:.4f}")
            else:
                class_weights[int(class_id)] = 1.0
        
        if 0 not in class_weights:
            class_weights[0] = 1.0
        if 1 not in class_weights:
            class_weights[1] = 1.0
        
        # Check for severe imbalance
        if len(unique_classes) == 2:
            min_count = min(class_counts)
            max_count = max(class_counts)
            imbalance_ratio = min_count / max_count
            if imbalance_ratio < 0.3:
                print(f"⚠️  Warning: Severe class imbalance detected (ratio: {imbalance_ratio:.2f})")
                print(f"   Consider using stratified sampling or stronger class weights")
            
        return class_weights
    
    def _create_weighted_loss(self, class_weights: Dict[int, float]):
        """Create weighted loss function for class balancing."""
        def weighted_binary_crossentropy_loss(y_true, y_pred):
            cca_width = y_pred.shape[-1] // 2
            pred1 = y_pred[:, :cca_width]
            cca_scores = pred1[:, 0]
            y_pred_prob = (tf.nn.tanh(cca_scores) + 1.0) / 2.0
            
            weights = tf.where(y_true == 0, 
                             tf.constant(class_weights[0], dtype=tf.float32),
                             tf.constant(class_weights[1], dtype=tf.float32))
            
            bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred_prob)
            weighted_loss = bce_loss * weights
            return tf.reduce_mean(weighted_loss)
        
        return weighted_binary_crossentropy_loss
    
    def _compute_correlation_scores(self, predictions: tf.Tensor) -> np.ndarray:
        """Compute correlation scores from CCA projections.
        
        For attention decoding, we compute a score per sample indicating how well
        the EEG matches the envelope. We use the sum of squared dot products across
        all CCA dimensions, weighted by dimension importance (exponential decay).
        
        The CCA algorithm maximizes correlation, so the weighted sum of projections
        gives a robust similarity score. Higher values indicate better match between
        EEG and envelope.
        
        We use squared dot products to emphasize stronger correlations and make the
        difference between attended and unattended more pronounced.
        """
        preds = predictions.numpy()
        cca_width = preds.shape[-1] // 2
        proj_eeg = preds[:, :cca_width]
        proj_env = preds[:, cca_width:]
        
        # Normalize projections to unit length for each dimension
        # This ensures fair comparison across dimensions
        proj_eeg_norm = proj_eeg / (np.linalg.norm(proj_eeg, axis=0, keepdims=True) + 1e-8)
        proj_env_norm = proj_env / (np.linalg.norm(proj_env, axis=0, keepdims=True) + 1e-8)
        
        # Use weighted sum across all CCA dimensions for more robust scoring
        # Weight by dimension index (first dimension has highest correlation)
        # Exponential decay gives more weight to the most correlated dimensions
        weights = np.exp(-np.arange(cca_width) * 0.15)  # Exponential decay (steeper)
        weights = weights / np.sum(weights)  # Normalize
        
        # Compute weighted sum of normalized dot products (correlation-like)
        # Using squared values to emphasize stronger correlations
        dot_products = proj_eeg_norm * proj_env_norm
        scores = np.sum(dot_products * weights, axis=1)
        
        # Alternative: Use sum of squared dot products for even stronger emphasis
        # scores = np.sum((dot_products ** 2) * weights, axis=1)
        
        return scores
    
    def predict(self, dataset: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Predict attention by correlating EEG with both candidate envelopes."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        all_predictions = []
        all_targets = []
        all_left_scores = []
        all_right_scores = []
        
        with device:
            for batch in tqdm(dataset, desc="Predicting"):
                if isinstance(batch, tuple):
                    inputs, aux = batch
                else:
                    inputs = batch
                    aux = None
                
                if aux is None or 'left_env' not in aux or 'right_env' not in aux:
                    raise ValueError("Dataset must provide left/right envelopes for prediction.")
                
                eeg_view = inputs['input_1']
                left_env = aux['left_env']
                right_env = aux['right_env']
                
                left_inputs = {'input_1': eeg_view, 'input_2': left_env}
                right_inputs = {'input_1': eeg_view, 'input_2': right_env}
                
                left_scores = self._compute_correlation_scores(self.model(left_inputs))
                right_scores = self._compute_correlation_scores(self.model(right_inputs))
                
                all_left_scores.extend(left_scores)
                all_right_scores.extend(right_scores)
                
                window_predictions = (right_scores > left_scores).astype(np.int64)
                all_predictions.extend(window_predictions)
                
                if 'label' in aux:
                    all_targets.extend(aux['label'].numpy().astype(np.int64))
        
        # Diagnostic analysis
        all_left_scores = np.array(all_left_scores)
        all_right_scores = np.array(all_right_scores)
        all_targets_arr = np.array(all_targets) if all_targets else None
        
        print("\n" + "="*80)
        print("PREDICTION DIAGNOSTICS")
        print("="*80)
        print(f"\nCorrelation Score Statistics:")
        print(f"  Left envelope correlations:")
        print(f"    Mean: {np.mean(all_left_scores):.6f} ± {np.std(all_left_scores):.6f}")
        print(f"    Min: {np.min(all_left_scores):.6f}, Max: {np.max(all_left_scores):.6f}")
        print(f"  Right envelope correlations:")
        print(f"    Mean: {np.mean(all_right_scores):.6f} ± {np.std(all_right_scores):.6f}")
        print(f"    Min: {np.min(all_right_scores):.6f}, Max: {np.max(all_right_scores):.6f}")
        
        score_diff = all_right_scores - all_left_scores
        print(f"\nScore Difference (Right - Left):")
        print(f"  Mean: {np.mean(score_diff):.6f} ± {np.std(score_diff):.6f}")
        print(f"  Min: {np.min(score_diff):.6f}, Max: {np.max(score_diff):.6f}")
        print(f"  Samples where right > left: {np.sum(score_diff > 0)}/{len(score_diff)} ({100*np.sum(score_diff > 0)/len(score_diff):.1f}%)")
        
        if all_targets_arr is not None and len(all_targets_arr) > 0:
            # Analyze by true label
            left_attention_mask = all_targets_arr == 0
            right_attention_mask = all_targets_arr == 1
            
            print(f"\nCorrelation Scores by True Attention:")
            if np.sum(left_attention_mask) > 0:
                left_att_left_scores = all_left_scores[left_attention_mask]
                left_att_right_scores = all_right_scores[left_attention_mask]
                print(f"  When LEFT is attended (n={np.sum(left_attention_mask)}):")
                print(f"    Left envelope corr: {np.mean(left_att_left_scores):.6f} ± {np.std(left_att_left_scores):.6f}")
                print(f"    Right envelope corr: {np.mean(left_att_right_scores):.6f} ± {np.std(left_att_right_scores):.6f}")
                print(f"    Difference (attended - unattended): {np.mean(left_att_left_scores - left_att_right_scores):.6f}")
            
            if np.sum(right_attention_mask) > 0:
                right_att_left_scores = all_left_scores[right_attention_mask]
                right_att_right_scores = all_right_scores[right_attention_mask]
                print(f"  When RIGHT is attended (n={np.sum(right_attention_mask)}):")
                print(f"    Left envelope corr: {np.mean(right_att_left_scores):.6f} ± {np.std(right_att_left_scores):.6f}")
                print(f"    Right envelope corr: {np.mean(right_att_right_scores):.6f} ± {np.std(right_att_right_scores):.6f}")
                print(f"    Difference (attended - unattended): {np.mean(right_att_right_scores - right_att_left_scores):.6f}")
        
        print("="*80 + "\n")
        
        return np.array(all_predictions), np.array(all_targets)


class OptimalCombinedCCATrainer:
    """Optimal trainer with comprehensive analysis."""
    
    def __init__(self, model: OptimalCombinedCCAModel, output_dir: str):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def train(self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset) -> float:
        """Train the optimal model."""
        self.model.fit(train_dataset)
        val_predictions, val_targets = self.model.predict(val_dataset)
        val_accuracy = accuracy_score(val_targets, val_predictions)
        return val_accuracy
    
    def test(self, test_dataset: tf.data.Dataset) -> Dict:
        """Test with comprehensive metrics."""
        predictions, targets = self.model.predict(test_dataset)
        accuracy = accuracy_score(targets, predictions)
        
        try:
            roc_auc = roc_auc_score(targets, predictions)
            avg_precision = average_precision_score(targets, predictions)
        except ValueError:
            roc_auc = 0.5
            avg_precision = 0.5
        
        mcc = matthews_corrcoef(targets, predictions)
        balanced_acc = balanced_accuracy_score(targets, predictions)
        
        results = {
            'accuracy': accuracy,
            'roc_auc_metrics': {'roc_auc_score': roc_auc, 'average_precision': avg_precision},
            'advanced_metrics': {
                'matthews_correlation_coefficient': mcc,
                'balanced_accuracy': balanced_acc
            },
            'predictions': predictions,
            'targets': targets
        }
        
        return results


def calculate_detailed_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict:
    """Calculate comprehensive detailed metrics."""
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='binary')
    recall = recall_score(targets, predictions, average='binary')
    f1 = f1_score(targets, predictions, average='binary')
    
    # Advanced metrics
    mcc = matthews_corrcoef(targets, predictions)
    balanced_acc = balanced_accuracy_score(targets, predictions)
    
    # ROC-AUC metrics
    try:
        roc_auc = roc_auc_score(targets, predictions)
        avg_precision = average_precision_score(targets, predictions)
    except ValueError:
        roc_auc = 0.5
        avg_precision = 0.5
    
    # Confusion matrix
    cm = confusion_matrix(targets, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    class_report = classification_report(targets, predictions, 
                                       target_names=['Left Attention', 'Right Attention'], 
                                       labels=[0, 1],
                                       output_dict=True)
    
    detailed_metrics = {
        'basic_metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'npv': npv,
            'ppv': ppv
        },
        'advanced_metrics': {
            'matthews_correlation_coefficient': mcc,
            'balanced_accuracy': balanced_acc,
            'roc_auc_score': roc_auc,
            'average_precision': avg_precision
        },
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        },
        'classification_report': class_report
    }
    
    return detailed_metrics


def generate_comprehensive_report(results: Dict, detailed_metrics: Dict, 
                                val_accuracy: float, output_path: Path):
    """Generate comprehensive analysis report."""
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS - {OPTIMAL_CONFIG['name']}")
    print(f"{'='*80}")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"ROC-AUC: {results.get('roc_auc', 0):.4f}")
    
    # Performance analysis
    if results['accuracy'] < 0.55:
        print(f"\n⚠️  PERFORMANCE ANALYSIS:")
        print(f"   Accuracy is near chance (50%). Possible issues:")
        print(f"   1. Window size may be too short (current: {OPTIMAL_CONFIG['window_size']} samples)")
        print(f"      → Try: --window_size 512 or --window_size 1024")
        print(f"   2. CCA dimensions may be too low (current: {OPTIMAL_CONFIG['cca_dims']})")
        print(f"      → Try: --cca_dims 16 or --cca_dims 20")
        print(f"   3. Regularization may be too high (current: {OPTIMAL_CONFIG['regularization']})")
        print(f"      → Try: --regularization 0.02 or --regularization 0.01")
        print(f"   4. Left envelope may be too weak (check envelope statistics above)")
        print(f"      → Verify Das audio file mapping and Fulsang left/right assignment")
    
    print(f"{'='*80}\n")
    
    save_comprehensive_results(results, detailed_metrics, val_accuracy, output_path)


def save_comprehensive_results(results: Dict, detailed_metrics: Dict, 
                             val_accuracy: float, output_path: Path):
    """Save comprehensive results to files."""
    results_to_save = {
        'configuration': OPTIMAL_CONFIG,
        'validation_accuracy': val_accuracy,
        'test_accuracy': results['accuracy'],
        'roc_auc': detailed_metrics['advanced_metrics']['roc_auc_score'],
        'matthews_correlation': detailed_metrics['advanced_metrics']['matthews_correlation_coefficient'],
        'balanced_accuracy': detailed_metrics['advanced_metrics']['balanced_accuracy']
    }
    
    with open(output_path / "comprehensive_results.json", 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    with open(output_path / "detailed_metrics.json", 'w') as f:
        json.dump(detailed_metrics, f, indent=2)
    
    with open(output_path / "classification_report.txt", 'w') as f:
        f.write("Combined CCA Optimal Configuration Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Configuration: {OPTIMAL_CONFIG['name']}\n")
        f.write(f"Test Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Validation Accuracy: {val_accuracy:.4f}\n\n")
        
        class_report = detailed_metrics['classification_report']
        f.write("Per-Class Metrics:\n")
        
        # sklearn's classification_report with output_dict=True uses target_names as keys,
        # e.g., 'Left Attention' and 'Right Attention'. Fall back to '0'/'1' if needed.
        left_key = 'Left Attention'
        right_key = 'Right Attention'
        if left_key not in class_report and '0' in class_report:
            left_key = '0'
        if right_key not in class_report and '1' in class_report:
            right_key = '1'

        def safe_metric(report_dict, key, metric):
            try:
                return float(report_dict.get(key, {}).get(metric, 0.0))
            except Exception:
                return 0.0

        f.write(f"Left Attention (0):\n")
        f.write(f"  Precision: {safe_metric(class_report, left_key, 'precision'):.4f}\n")
        f.write(f"  Recall: {safe_metric(class_report, left_key, 'recall'):.4f}\n")
        f.write(f"  F1-Score: {safe_metric(class_report, left_key, 'f1-score'):.4f}\n")
        f.write(f"Right Attention (1):\n")
        f.write(f"  Precision: {safe_metric(class_report, right_key, 'precision'):.4f}\n")
        f.write(f"  Recall: {safe_metric(class_report, right_key, 'recall'):.4f}\n")
        f.write(f"  F1-Score: {safe_metric(class_report, right_key, 'f1-score'):.4f}\n")


def cleanup_gpu_memory():
    """Clean up GPU memory."""
    try:
        tf.keras.backend.clear_session()
        gc.collect()
    except Exception:
        pass


def split_dataset(dataset_wrapper: CombinedCCADataset, train_ratio: float = 0.60, 
                  val_ratio: float = 0.25) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Split dataset into train/val/test sets."""
    all_indices = list(range(len(dataset_wrapper.window_indices)))
    np.random.shuffle(all_indices)
    
    train_size = int(train_ratio * len(all_indices))
    val_size = int(val_ratio * len(all_indices))
    
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + val_size]
    test_indices = all_indices[train_size + val_size:]
    
    # Create separate datasets for each split
    class SplitDataset:
        def __init__(self, parent, indices):
            self.parent = parent
            self.indices = indices
            self.window_indices = [parent.window_indices[i] for i in indices]
        
        def create_tf_dataset(self, batch_size):
            def generator():
                for idx in self.indices:
                    start_idx, end_idx, label = self.parent.window_indices[idx]
                    eeg_window = self.parent.combined_dataset.eeg_data[start_idx:end_idx]
                    eeg_window = self.parent._preprocess_window(eeg_window)
                    left_env, right_env = self.parent.combined_dataset.get_envelope_window(start_idx, end_idx)
                    left_env = left_env.flatten().astype(np.float32)
                    right_env = right_env.flatten().astype(np.float32)
                    attended_env = left_env if label == 0 else right_env
                    eeg_flat = eeg_window.flatten().astype(np.float32)
                    yield (
                        {'input_1': eeg_flat, 'input_2': attended_env},
                        {'left_env': left_env, 'right_env': right_env, 'label': np.array(label, dtype=np.float32)}
                    )
            
            output_signature = (
                {
                    'input_1': tf.TensorSpec(shape=(self.parent.window_size * self.parent.n_channels,), dtype=tf.float32),
                    'input_2': tf.TensorSpec(shape=(self.parent.window_size,), dtype=tf.float32)
                },
                {
                    'left_env': tf.TensorSpec(shape=(self.parent.window_size,), dtype=tf.float32),
                    'right_env': tf.TensorSpec(shape=(self.parent.window_size,), dtype=tf.float32),
                    'label': tf.TensorSpec(shape=(), dtype=tf.float32)
                }
            )
            
            dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            return dataset
    
    train_dataset_wrapper = SplitDataset(dataset_wrapper, train_indices)
    val_dataset_wrapper = SplitDataset(dataset_wrapper, val_indices)
    test_dataset_wrapper = SplitDataset(dataset_wrapper, test_indices)
    
    train_dataset = train_dataset_wrapper.create_tf_dataset(OPTIMAL_CONFIG['batch_size'])
    val_dataset = val_dataset_wrapper.create_tf_dataset(OPTIMAL_CONFIG['batch_size'])
    test_dataset = test_dataset_wrapper.create_tf_dataset(OPTIMAL_CONFIG['batch_size'])
    
    return train_dataset, val_dataset, test_dataset


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Combined Das+Fulsang CCA using Optimal_FULCCA configuration')
    parser.add_argument('--das_data_dir', type=str, default='das_16subjects_preprocessed',
                       help='Directory containing Das preprocessed data')
    parser.add_argument('--das_preprocessing_type', type=str, default='16SUBJECTS',
                       choices=['MWF', 'DASPREPROCESS', '16SUBJECTS'],
                       help='Type of Das preprocessing')
    parser.add_argument('--fulsang_raw_dir', type=str, 
                       default='/home/py9363/telluride_decoding/Data/Fulsang/EEG',
                       help='Directory containing Fulsang raw EEG data')
    parser.add_argument('--fulsang_audio_dir', type=str,
                       default='/home/py9363/telluride_decoding/Data/Fulsang/AUDIO',
                       help='Directory containing Fulsang audio data')
    parser.add_argument('--fulsang_mwf_dir', type=str, default='MWF_cleaned_Fuglsang',
                       help='Output directory for Fulsang MWF processing')
    parser.add_argument('--das_original_dir', type=str, default='Data/Das/4004271',
                       help='Directory containing original Das .mat files (for envelope extraction)')
    parser.add_argument('--das_audio_dir', type=str, default='Data/Das/4004271/stimuli/stimuli',
                       help='Directory containing Das audio files (for envelope extraction)')
    parser.add_argument('--combined_dataset_dir', type=str, default='combined_dataset',
                       help='Centralized directory for all processed files (default: combined_dataset)')
    parser.add_argument('--window_size', type=int, default=OPTIMAL_CONFIG['window_size'],
                       help=f'Window size in samples (default: {OPTIMAL_CONFIG["window_size"]} ≈ {OPTIMAL_CONFIG["window_size"]//128}s, can use 1024 ≈ 8s at 128Hz)')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Window overlap fraction (default: 0.5)')
    parser.add_argument('--cca_dims', type=int, default=OPTIMAL_CONFIG['cca_dims'],
                       help=f'Number of CCA dimensions (default: {OPTIMAL_CONFIG["cca_dims"]})')
    parser.add_argument('--regularization', type=float, default=OPTIMAL_CONFIG['regularization'],
                       help=f'CCA regularization (default: {OPTIMAL_CONFIG["regularization"]})')
    parser.add_argument('--batch_size', type=int, default=OPTIMAL_CONFIG['batch_size'],
                       help=f'Batch size (default: {OPTIMAL_CONFIG["batch_size"]})')
    parser.add_argument('--output_dir', type=str, default='combined_cca_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # ========================================================================
    # RUN PREPROCESSING FIRST
    # ========================================================================
    print("="*80)
    print("RUNNING PREPROCESSING PIPELINE")
    print("="*80)
    
    # 1. Run Das 16-subject preprocessing
    print("\n[1/3] Running Das 16-subject preprocessing...")
    das_original_dir = getattr(args, 'das_original_dir', 'Data/Das/4004271')
    das_audio_dir = getattr(args, 'das_audio_dir', 'Data/Das/4004271/stimuli/stimuli')
    das_preprocessing_cmd = [
        sys.executable, "das_preprocessing_16subjects.py",
        "--data_dir", das_original_dir,
        "--output_dir", "das_16subjects_preprocessed",
        "--audio_dir", das_audio_dir,
        "--create_split"
    ]
    print(f"Command: {' '.join(das_preprocessing_cmd)}")
    result = subprocess.run(das_preprocessing_cmd, capture_output=False)
    if result.returncode != 0:
        print(f"⚠️  Warning: Das preprocessing returned exit code {result.returncode}")
        print("  Continuing anyway...")
    else:
        print("✓ Das 16-subject preprocessing completed")
    
    # 2. Map audio files to existing Das MWF files (Das MWF already processed)
    print("\n[2/3] Mapping audio files to existing Das MWF files...")
    combined_dataset_dir = getattr(args, 'combined_dataset_dir', 'combined_dataset')
    das_mwf_dir = os.path.join(combined_dataset_dir, 'das_mwf')
    os.makedirs(das_mwf_dir, exist_ok=True)
    
    # Run MWF script in map-only mode for Das (just adds audio file paths)
    print("  Adding audio file mapping to existing Das MWF files...")
    das_mwf_cmd = [
        sys.executable, "mwf_artifact_removal.py",
        "--dataset", "das",
        "--das_dir", das_original_dir,
        "--das_audio_dir", das_audio_dir
    ]
    print(f"  Command: {' '.join(das_mwf_cmd)}")
    result = subprocess.run(das_mwf_cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  ⚠️  Warning: Das MWF audio mapping returned exit code {result.returncode}")
    else:
        print("  ✓ Das MWF audio mapping completed")
        # Move files to centralized location if needed
        default_das_mwf = "MWF_cleaned_DAS"
        if os.path.exists(default_das_mwf):
            import shutil
            for mwf_file in Path(default_das_mwf).glob("S*_MWF.mat"):
                target = Path(das_mwf_dir) / mwf_file.name
                if not target.exists():
                    shutil.move(str(mwf_file), str(target))
                    print(f"    Moved {mwf_file.name} to centralized location")
    
    # 3. Run MWF processing for Fulsang only
    print("\n[3/3] Running MWF artifact removal for Fulsang...")
    fulsang_mwf_dir = os.path.join(combined_dataset_dir, 'fulsang_mwf')
    os.makedirs(fulsang_mwf_dir, exist_ok=True)
    
    fulsang_mwf_cmd = [
        sys.executable, "mwf_artifact_removal.py",
        "--dataset", "fuglsang",
        "--fuglsang_eeg_dir", getattr(args, 'fulsang_raw_dir', '/home/py9363/telluride_decoding/Data/Fulsang/EEG'),
        "--fuglsang_audio_dir", getattr(args, 'fulsang_audio_dir', '/home/py9363/telluride_decoding/Data/Fulsang/AUDIO')
    ]
    print(f"  Command: {' '.join(fulsang_mwf_cmd)}")
    result = subprocess.run(fulsang_mwf_cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  ⚠️  Warning: Fulsang MWF processing returned exit code {result.returncode}")
    else:
        print("  ✓ Fulsang MWF processing completed")
        # Move files to centralized location
        default_fulsang_mwf = "MWF_cleaned_Fuglsang"
        if os.path.exists(default_fulsang_mwf):
            import shutil
            for mwf_file in Path(default_fulsang_mwf).glob("sub*_MWF.mat"):
                target = Path(fulsang_mwf_dir) / mwf_file.name
                if not target.exists():
                    shutil.move(str(mwf_file), str(target))
                    print(f"    Moved {mwf_file.name} to centralized location")
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE - Starting CCA Training")
    print("="*80 + "\n")
    
    # Update optimal config with args
    OPTIMAL_CONFIG['window_size'] = args.window_size
    OPTIMAL_CONFIG['cca_dims'] = args.cca_dims
    OPTIMAL_CONFIG['regularization'] = args.regularization
    OPTIMAL_CONFIG['batch_size'] = args.batch_size
    
    print("="*80)
    print("COMBINED CCA - Das (MWF) + Fulsang (MWF) CCA Training")
    print("="*80)
    print(f"Using Optimal_FULCCA configuration: {OPTIMAL_CONFIG['name']}")
    print(f"\nConfiguration Values:")
    print(f"  CCA dimensions: {OPTIMAL_CONFIG['cca_dims']} (default: 16)")
    print(f"  Regularization: {OPTIMAL_CONFIG['regularization']} (default: 0.02)")
    print(f"  Window size: {OPTIMAL_CONFIG['window_size']} samples (default: 512)")
    print(f"  Batch size: {OPTIMAL_CONFIG['batch_size']} (default: 6)")
    print(f"\nCommand-line Arguments:")
    print(f"  --cca_dims: {args.cca_dims}")
    print(f"  --regularization: {args.regularization}")
    print(f"  --window_size: {args.window_size}")
    print(f"  --batch_size: {args.batch_size}")
    
    # Warn if using non-optimal defaults
    if args.window_size == 256:
        print(f"\n  ⚠️  WARNING: Using window_size=256 (suboptimal). Recommended: 512 or 1024")
    if args.cca_dims == 12:
        print(f"  ⚠️  WARNING: Using cca_dims=12 (suboptimal). Recommended: 16")
    if args.regularization == 0.08:
        print(f"  ⚠️  WARNING: Using regularization=0.08 (suboptimal). Recommended: 0.02")
    
    # Create combined dataset
    print("\n" + "="*80)
    print("LOADING COMBINED DATASET")
    print("="*80)
    combined_dataset = CombinedDataset(
        das_data_dir=args.das_data_dir,
        das_preprocessing_type=args.das_preprocessing_type,
        das_original_dir=getattr(args, 'das_original_dir', 'Data/Das/4004271'),
        das_audio_dir=getattr(args, 'das_audio_dir', 'Data/Das/4004271/stimuli/stimuli'),
        fulsang_raw_dir=args.fulsang_raw_dir,
        fulsang_audio_dir=args.fulsang_audio_dir,
        fulsang_mwf_output_dir=args.fulsang_mwf_dir,
        combined_dataset_dir=getattr(args, 'combined_dataset_dir', 'combined_dataset'),
        window_size=args.window_size,
        overlap=args.overlap
    )
    
    # Create TensorFlow dataset wrapper
    print("\n" + "="*80)
    print("CREATING TENSORFLOW DATASET")
    print("="*80)
    tf_dataset_wrapper = CombinedCCADataset(combined_dataset)
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(
        tf_dataset_wrapper,
        train_ratio=0.60,
        val_ratio=0.25
    )
    
    # Create CCA model
    print("\n" + "="*80)
    print("INITIALIZING OPTIMAL COMBINED CCA MODEL")
    print("="*80)
    cca_model = OptimalCombinedCCAModel(
        cca_dims=OPTIMAL_CONFIG['cca_dims'],
        regularization=OPTIMAL_CONFIG['regularization'],
        window_size=OPTIMAL_CONFIG['window_size']
    )
    
    # Create trainer
    trainer = OptimalCombinedCCATrainer(cca_model, args.output_dir)
    
    # Train model
    print("\n" + "="*80)
    print("TRAINING CCA MODEL")
    print("="*80)
    val_accuracy = trainer.train(train_dataset, val_dataset)
    
    # Test model
    print("\n" + "="*80)
    print("TESTING MODEL")
    print("="*80)
    results = trainer.test(test_dataset)
    
    # Calculate detailed metrics
    detailed_metrics = calculate_detailed_metrics(results['predictions'], results['targets'])
    
    # Generate report
    generate_comprehensive_report(results, detailed_metrics, val_accuracy, Path(args.output_dir))
    
    print(f"\n✓ Combined CCA Training Complete")
    print(f"  Test Accuracy: {results['accuracy']:.4f}")
    print(f"  Validation Accuracy: {val_accuracy:.4f}")
    print(f"  ROC-AUC: {detailed_metrics['advanced_metrics']['roc_auc_score']:.4f}")
    print(f"\n✓ Results saved to {args.output_dir}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        cleanup_gpu_memory()
