#!/usr/bin/env python3
"""
CombinedCCA - Optimal CCA Algorithm for Combined Das and Fulsang Dataset

This script implements CCA (Canonical Correlation Analysis) for the combined dataset
using the Optimal_FULCCA configuration, adapted for CombinedDataset.

Evaluation (paper-style): Single subject-level train/val/test split (no K-fold CV).
Accuracy: Window-level test accuracy (primary); trial-level accuracy (majority vote
per trial) is also reported. Keep evaluation and accuracy definitions unchanged to
match the paper.
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
import inspect
import json
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import re
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


def make_lagged_audio(audio: np.ndarray, lag_samples: np.ndarray, fs: float = 128.0) -> np.ndarray:
    """
    Create time-lagged audio features for CCA (forward model).
    Neural response to speech has a delay (typically 0-250ms). This function creates
    lagged copies of the audio envelope to account for this latency.
    Args:
        audio: Audio envelope of shape (T,) or (T, B); B=1 for 1-band combined envelope.
        lag_samples: Array of lag values in samples (e.g., 0-250ms at fs Hz).
        fs: Sampling rate in Hz (default: 128 Hz for combined dataset).
    Returns:
        Lagged audio features of shape (T, B * num_lags). For (T,1) input, shape (T, num_lags).
    """
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    T, B = audio.shape
    num_lags = len(lag_samples)
    lagged_features = []
    for lag in lag_samples:
        shifted = np.roll(audio, int(lag), axis=0)
        if lag > 0:
            shifted[: int(lag), :] = 0
        lagged_features.append(shifted)
    lagged_audio = np.concatenate(lagged_features, axis=1)
    return lagged_audio.astype(np.float32)


def make_lagged_eeg(eeg: np.ndarray, L: int) -> np.ndarray:
    """
    Create time-lagged EEG features for backward model (paper: spatiotemporal wx).
    x(t) = [eeg(t), eeg(t-1), ..., eeg(t-L+1)] per channel then flatten -> (T, C*L).
    Causal: only past and current; early t padded with zeros.
    Args:
        eeg: EEG of shape (T, C)
        L: Number of backward taps (lag order)
    Returns:
        (T, C*L) float32
    """
    T, C = eeg.shape
    if L <= 1:
        return np.asarray(eeg, dtype=np.float32)
    out = np.zeros((T, C * L), dtype=np.float32)
    for t in range(T):
        segs = []
        for lag in range(L):
            idx = t - lag
            if idx >= 0:
                segs.append(eeg[idx, :])
            else:
                segs.append(np.zeros(C, dtype=eeg.dtype))
        out[t, :] = np.concatenate(segs, axis=0)
    return out


# Default max batches when dataset cardinality is unknown (avoids ~7+ hour runs).
# 2000 batches @ ~2.5s/batch ≈ 1.4 hours. Increase if you need more data for CCA.
DEFAULT_CCA_MAX_BATCHES = 2000

# Fixed CCA calculation function with progress logging and batch limiting
# This function must be defined BEFORE the import that uses it
def calculate_cca_parameters_fixed(dataset, dim, regularization=0.1, 
                                   mini_batch_count=0, eps_eig=1e-12):
    """Fixed version of calculate_cca_parameters_from_dataset with robust batch handling.

    Behavior:
    - If mini_batch_count > 0: use that many batches (or fewer if dataset is smaller).
    - If mini_batch_count == 0 or None:
        * If dataset has finite known cardinality: use ALL batches.
        * If dataset is infinite/unknown: cap at DEFAULT_CCA_MAX_BATCHES to avoid long runs.
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
            mini_batch_count = DEFAULT_CCA_MAX_BATCHES
            logging.warning(
                'Dataset has infinite cardinality; capping CCA processing to %d batches '
                'to avoid an infinite loop. Tune DEFAULT_CCA_MAX_BATCHES or pass mini_batch_count if needed.',
                mini_batch_count
            )
        elif card == tf.data.experimental.UNKNOWN_CARDINALITY:
            mini_batch_count = DEFAULT_CCA_MAX_BATCHES
            logging.warning(
                'Dataset cardinality is unknown; capping CCA processing to %d batches. '
                'If you want fewer/more, pass mini_batch_count explicitly or set DEFAULT_CCA_MAX_BATCHES.',
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
    
    # Use total_frames for denominator (not num_mini_batches * n_row) so variable batch sizes are correct
    n_minus_1 = max(1, total_frames - 1)
    mean_x = sum_x / total_frames
    mean_y = sum_y / total_frames
    cov_xx = (cov_xx - total_frames * np.matmul(mean_x.T, mean_x)) / n_minus_1
    cov_xx += regularization * np.eye(x.shape[1])
    cov_yy = (cov_yy - total_frames * np.matmul(mean_y.T, mean_y)) / n_minus_1
    cov_yy += regularization * np.eye(y.shape[1])
    cov_xy = (cov_xy - total_frames * np.matmul(mean_x.T, mean_y)) / n_minus_1
    
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
    cca_module.calculate_cca_parameters_from_dataset = calculate_cca_parameters_fixed
except ImportError as e:
    print(f"Error: Could not import telluride_decoding CCA modules: {e}")
    sys.exit(1)

# Import combined dataset
from CombinedDataset import CombinedDataset

def _combined_dataset_kwargs(args, window_size_samples: int):
    """Build kwargs for CombinedDataset so older versions without bandpass_* still work."""
    fs = getattr(args, 'target_sampling_rate', PAPER_FS_HZ)
    kwargs = {
        'das_data_dir': args.das_data_dir,
        'das_preprocessing_type': args.das_preprocessing_type,
        'das_original_dir': args.das_original_dir,
        'das_audio_dir': args.das_audio_dir,
        'fulsang_raw_dir': args.fulsang_raw_dir,
        'fulsang_audio_dir': args.fulsang_audio_dir,
        'fulsang_mwf_output_dir': args.fulsang_mwf_dir,
    }
    expinfo_dir = getattr(args, 'fulsang_expinfo_dir', None)
    if expinfo_dir and str(expinfo_dir).strip():
        kwargs['fulsang_expinfo_dir'] = expinfo_dir.strip()
    kwargs.update({
        'combined_dataset_dir': getattr(args, 'combined_dataset_dir', 'combined_dataset'),
        'window_size': window_size_samples,
        'overlap': args.overlap,
        'target_sampling_rate': fs,
    })
    sig = inspect.signature(CombinedDataset.__init__)
    if 'bandpass_low_hz' in sig.parameters:
        kwargs['bandpass_low_hz'] = getattr(args, 'bandpass_low_hz', 2.0)
        kwargs['bandpass_high_hz'] = getattr(args, 'bandpass_high_hz', 8.0)
        kwargs['bandpass_order'] = getattr(args, 'bandpass_order', 1)
    if 'use_hilbert_envelope' in sig.parameters:
        kwargs['use_hilbert_envelope'] = getattr(args, 'use_hilbert_envelope', True)
    if 'envelope_normalize' in sig.parameters:
        kwargs['envelope_normalize'] = getattr(args, 'envelope_normalize', 'scale_only')
    if 'balance_envelope_energy' in sig.parameters:
        kwargs['balance_envelope_energy'] = getattr(args, 'balance_envelope_energy', True)
    if 'use_gammatone_filter' in sig.parameters:
        kwargs['use_gammatone_filter'] = getattr(args, 'use_gammatone_filter', True)
    return kwargs

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

# Sampling rate (Hz) for converting window seconds to samples
SAMPLING_RATE_HZ = 128

# Window sweep: 1s, 5s, 10s, 20s, 30s (paper: accuracy vs decision window length)
WINDOW_SWEEP_SECONDS = [1, 5, 10, 20, 30]

# Max EEG/envelope dimension for CCA without PCA. Above this, PCA reduces to PCA_CCA_COMPONENTS to avoid OOM (exit 137).
# Cap EEG dim to avoid OOM when stacking all time points (e.g. 640 dim * 5M samples).
# With eeg_lag_taps=10, 640-dim EEG is reduced via PCA to PCA_CCA_COMPONENTS before CCA.
MAX_EEG_DIM_CCA = 512
MAX_ENV_DIM_CCA = 512
PCA_CCA_COMPONENTS = 128  # retain more EEG variance for subtle speech-tracking components

# Paper-style defaults: 64 Hz, 8 s window, bandpass 2–8 Hz (speech-brain delta/theta), L=12 EEG taps
PAPER_WINDOW_SEC = 8.0
PAPER_FS_HZ = 64
OPTIMAL_CONFIG = {
    'name': 'opt_3_combined_improved',
    'cca_dims': 16,
    'regularization': 0.01,  # Lower than 0.02 for stronger CCA fit; use 0.02 if unstable
    'window_size': int(PAPER_FS_HZ * PAPER_WINDOW_SEC),  # 512 samples = 8 s @ 64 Hz
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
    
    def __init__(self, combined_dataset: CombinedDataset, mode: str = 'full',
                 use_time_lags: bool = True, min_lag_ms: float = 0.0, max_lag_ms: float = 300.0, fs: Optional[float] = None,
                 max_windows: Optional[int] = None, use_time_resolved: bool = True, eeg_lag_taps: int = 12):
        self.combined_dataset = combined_dataset
        self.mode = mode
        self.use_time_resolved = use_time_resolved  # classic: (T, channels) per window, correlation across time
        self.eeg_lag_taps = max(1, int(eeg_lag_taps))  # backward model: [eeg(t), eeg(t-1), ...] -> (T, n_channels*L)
        self.window_size = combined_dataset.window_size
        self.sampling_rate = combined_dataset.sampling_rate
        self.n_channels = combined_dataset.n_channels
        self.eeg_time_point_dim = self.n_channels * self.eeg_lag_taps  # per-time-point EEG dim when time-resolved
        self.envelope_bands = getattr(combined_dataset, 'envelope_bands', getattr(CombinedDataset, 'ENVELOPE_BANDS', 1))
        self.use_time_lags = use_time_lags
        self.fs = fs if fs is not None else float(self.sampling_rate)
        self.min_lag_ms = max(0.0, min_lag_ms)
        self.max_lag_ms = min(500.0, max_lag_ms)
        if use_time_lags:
            if self.max_lag_ms > 300:
                print(f"  ⚠ WARNING: max_lag_ms={self.max_lag_ms}ms — high lag range on CPU can explode dimensionality. Consider 0–300ms.")
            min_lag_samples = int(self.min_lag_ms * self.fs / 1000.0)
            max_lag_samples = int(self.max_lag_ms * self.fs / 1000.0)
            self.lag_samples = np.arange(min_lag_samples, max_lag_samples + 1)
            self.num_lags = len(self.lag_samples)
            self.envelope_dim = self.window_size * self.envelope_bands * self.num_lags
            print(f"  Time-lagged audio: {self.num_lags} lags ({self.min_lag_ms}-{self.max_lag_ms}ms at {self.fs} Hz)")
        else:
            self.lag_samples = np.array([0])
            self.num_lags = 1
            self.envelope_dim = self.window_size * self.envelope_bands
        
        # Get window indices (optionally cap to max_windows to match ~DAS + ~Fulsang standalone counts)
        self.window_indices = combined_dataset.get_window_indices(max_windows=max_windows)
        
        print(f"\nCombinedCCADataset initialized (1-band envelope, Fulsang-style):")
        print(f"  Mode: {mode}")
        print(f"  Time-resolved (classic): {self.use_time_resolved} (correlation across time per window)")
        if self.use_time_resolved and self.eeg_lag_taps > 1:
            print(f"  Time-lagged EEG (backward model): L={self.eeg_lag_taps} taps -> {self.eeg_time_point_dim} features per time point")
        print(f"  Total windows: {len(self.window_indices)}")
        print(f"  Window size: {self.window_size} samples")
        print(f"  Envelope bands: {self.envelope_bands} (input_2 dim: {self.envelope_dim})")
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
        
        # Check envelope availability (sample more windows for representative stats)
        n_sample = min(50, len(self.window_indices))
        sample_indices = np.random.choice(len(self.window_indices), n_sample, replace=False)
        envelope_checks = []
        for idx in sample_indices:
            win = self.window_indices[idx]
            start_idx, end_idx, label = win[0], win[1], win[2]
            try:
                left_env, right_env = self.combined_dataset.get_envelope_window(start_idx, end_idx)
                if left_env is not None and right_env is not None:
                    envelope_checks.append(True)
                else:
                    envelope_checks.append(False)
            except Exception as e:
                envelope_checks.append(False)
        
        valid_envelopes = np.sum(envelope_checks)
        # Indices that actually passed the envelope check (not just first N)
        passed_indices = [sample_indices[i] for i in range(len(sample_indices)) if envelope_checks[i]]
        print(f"\n3. Envelope Availability:")
        print(f"   Valid envelopes: {valid_envelopes}/{len(envelope_checks)} sampled windows")
        if valid_envelopes < len(envelope_checks):
            print(f"   ⚠️  WARNING: Some windows have missing envelopes!")
        
        # Check envelope differences and quality (only over windows that have valid envelopes)
        if passed_indices:
            left_envs = []
            right_envs = []
            envelope_stats = {'left': [], 'right': []}
            
            for idx in passed_indices:
                win = self.window_indices[idx]
                start_idx, end_idx, label = win[0], win[1], win[2]
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
                    
                    # Check left vs right energy imbalance (audio not read for one channel)
                    mean_left_energy = np.mean(left_energies)
                    mean_right_energy = np.mean(right_energies)
                    if mean_right_energy > 1e-6 and mean_left_energy < 0.01 * mean_right_energy:
                        print(f"   ⚠️  WARNING: Left envelope much weaker than right (ratio ~{mean_right_energy/max(1e-12, mean_left_energy):.0f}:1)")
                        print(f"      Left channel audio may not be read correctly. Check Das left_audio_file and Fulsang aske=left mapping.")
                    elif mean_left_energy > 1e-6 and mean_right_energy < 0.01 * mean_left_energy:
                        print(f"   ⚠️  WARNING: Right envelope much weaker than left. Check right_audio_file and Fulsang marianne=right mapping.")
                    
                    # Check if envelopes have sufficient variation
                    left_variation = np.mean([s['std'] for s in envelope_stats['left']])
                    right_variation = np.mean([s['std'] for s in envelope_stats['right']])
                    
                    if left_variation < 0.01 or right_variation < 0.01:
                        print(f"   ⚠️  WARNING: Low envelope variation (left: {left_variation:.6f}, right: {right_variation:.6f})")
                        print(f"      Envelopes may be too flat/similar to decode attention")
                    
                    # Check energy difference
                    energy_diff = abs(mean_left_energy - mean_right_energy)
                    if energy_diff < 0.001:
                        print(f"   ⚠️  WARNING: Very similar envelope energies (diff: {energy_diff:.6f})")
        
        # Check window size compatibility
        print(f"\n4. Window Size Validation:")
        if self.window_size < 256:
            print(f"   ⚠️  WARNING: Window size {self.window_size} may be too short for reliable decoding")
            print(f"      Recommended: >= 512 samples (4+ seconds at 128 Hz)")
        else:
            print(f"   ✓ Window size {self.window_size} is adequate")
        
        # Label and envelope verification (paper-style: left/right vs attend_lr, timing)
        print(f"\n5. Label and envelope verification:")
        print(f"   Labels: 0 = attend left, 1 = attend right (attend_lr 1→0, 2→1).")
        print(f"   Envelopes: left = speaker on left position, right = speaker on right position.")
        print(f"   Fulsang: when only attend_lr is available, left=Aske (male), right=Marianne (female).")
        print(f"   Envelope timing: resampled to EEG length per trial (no constant lag); CCA uses {self.min_lag_ms:.0f}–{self.max_lag_ms:.0f} ms lags.")
        print(f"   ✓ If decoding is poor, confirm experiment left/right matches this mapping.")
        
        print("="*60 + "\n")
    
    def _preprocess_window(self, eeg_window: np.ndarray) -> np.ndarray:
        """Preprocess EEG window for CCA: per-channel baseline + linear scaling only.

        No tanh: CCA is linear; nonlinear clipping compresses dynamic range and weakens correlations.
        """
        eeg_window = eeg_window - np.mean(eeg_window, axis=0, keepdims=True)
        std_vals = np.std(eeg_window, axis=0, keepdims=True)
        std_vals = np.where(std_vals == 0, 1.0, std_vals)
        eeg_window = eeg_window / std_vals
        eeg_window = np.nan_to_num(eeg_window, nan=0.0, posinf=0.0, neginf=0.0)
        return eeg_window.astype(np.float32)
    
    def create_tf_dataset(self, batch_size: int = 6) -> tf.data.Dataset:
        """Create TensorFlow dataset for CCA. If use_time_resolved (classic): (T, channels) and (T, num_lags) per window."""
        def generator():
            for win in self.window_indices:
                start_idx, end_idx, label = win[0], win[1], win[2]
                eeg_window = self.combined_dataset.eeg_data[start_idx:end_idx]
                eeg_window = self._preprocess_window(eeg_window)
                if self.use_time_resolved and self.eeg_lag_taps > 1:
                    eeg_window = make_lagged_eeg(eeg_window, self.eeg_lag_taps)
                left_env, right_env = self.combined_dataset.get_envelope_window(start_idx, end_idx)
                left_env = left_env.flatten().astype(np.float32)
                right_env = right_env.flatten().astype(np.float32)
                if self.use_time_lags:
                    left_env = make_lagged_audio(left_env.reshape(-1, 1), self.lag_samples, self.fs)
                    right_env = make_lagged_audio(right_env.reshape(-1, 1), self.lag_samples, self.fs)
                # Do not z-score envelopes per window (destroys slow structure and hurts CCA vs dataset-level norm)
                if self.use_time_resolved:
                    # Classic: keep (T, channels) and (T, num_lags); do not flatten
                    left_env = left_env.astype(np.float32)
                    right_env = right_env.astype(np.float32)
                    attended_env = left_env if label == 0 else right_env
                    yield (
                        {'input_1': eeg_window.astype(np.float32), 'input_2': attended_env},
                        {'left_env': left_env, 'right_env': right_env, 'label': np.array(label, dtype=np.float32)}
                    )
                else:
                    left_env = left_env.flatten().astype(np.float32)
                    right_env = right_env.flatten().astype(np.float32)
                    attended_env = left_env if label == 0 else right_env
                    eeg_flat = eeg_window.flatten().astype(np.float32)
                    yield (
                        {'input_1': eeg_flat, 'input_2': attended_env},
                        {'left_env': left_env, 'right_env': right_env, 'label': np.array(label, dtype=np.float32)}
                    )
        
        if self.use_time_resolved:
            eeg_cols = self.eeg_time_point_dim  # n_channels or n_channels * eeg_lag_taps
            input_signature = {
                'input_1': tf.TensorSpec(shape=(self.window_size, eeg_cols), dtype=tf.float32),
                'input_2': tf.TensorSpec(shape=(self.window_size, self.num_lags), dtype=tf.float32)
            }
            aux_signature = {
                'left_env': tf.TensorSpec(shape=(self.window_size, self.num_lags), dtype=tf.float32),
                'right_env': tf.TensorSpec(shape=(self.window_size, self.num_lags), dtype=tf.float32),
                'label': tf.TensorSpec(shape=(), dtype=tf.float32)
            }
        else:
            feature_dim = self.window_size * self.n_channels
            input_signature = {
                'input_1': tf.TensorSpec(shape=(feature_dim,), dtype=tf.float32),
                'input_2': tf.TensorSpec(shape=(self.envelope_dim,), dtype=tf.float32)
            }
            aux_signature = {
                'left_env': tf.TensorSpec(shape=(self.envelope_dim,), dtype=tf.float32),
                'right_env': tf.TensorSpec(shape=(self.envelope_dim,), dtype=tf.float32),
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
    CCA model for attention decoding on the combined dataset.
    
    Default (paper style): Trains CCA on (EEG, attended envelope only). At test:
    compute ρ_left, ρ_right with same CCA, f = ρ_left − ρ_right, LDA(f).
    Use --cca_both_envelopes to train on (EEG, both envelopes) instead.
    
    Scoring (per window):
    - use_first_cca_component_only=False (default): weighted sum over CCA dimensions,
      score = (rho * w).sum() with w = exp(-0.15*j). More dimensions contribute.
    - use_first_cca_component_only=True: use first canonical component only, score = rho[0].
      Many AAD papers use ρ₁ only for stability; try this if accuracy is low.
    """
    
    def __init__(self, cca_dims: int = 16, regularization: float = 0.02, window_size: int = 512,
                 envelope_dim: Optional[int] = None, eeg_flat_dim: Optional[int] = None,
                 use_lda: bool = True, use_first_cca_component_only: bool = False,
                 train_cca_on_both_envelopes: bool = False):
        self.window_size = window_size
        self.use_lda = use_lda
        self.train_cca_on_both_envelopes = bool(train_cca_on_both_envelopes)
        if envelope_dim is not None and eeg_flat_dim is not None:
            actual_max_cca_dims = min(eeg_flat_dim, envelope_dim)
            if cca_dims > actual_max_cca_dims:
                print(f"  CCA dims capped: requested {cca_dims}, max {actual_max_cca_dims} (min(EEG={eeg_flat_dim}, Audio={envelope_dim}))")
                cca_dims = actual_max_cca_dims
            elif cca_dims < 1:
                cca_dims = 1
        self.cca_dims = cca_dims
        self.regularization = regularization
        self.cca_params = None   # Single CCA: {rot_x, rot_y, mean_x, mean_y, eigenvalues}
        self.lda_model = None
        self.lda_scaler = None
        self._pca_eeg = None
        self._pca_env = None
        self.is_fitted = False
        self.use_first_cca_component_only = use_first_cca_component_only
    
    def fit(self, dataset: tf.data.Dataset):
        """Fit CCA on (EEG, attended envelope) by default (paper), or both envelopes if train_cca_on_both_envelopes. Then LDA on f = ρ_left − ρ_right."""
        MAX_BATCHES = 10000

        def _get_batch(b):
            if isinstance(b, tuple):
                inputs, aux = b
            else:
                inputs, aux = b, None
            if aux is None or 'left_env' not in aux or 'right_env' not in aux:
                return None, None, None, None
            eeg = inputs['input_1'].numpy() if hasattr(inputs['input_1'], 'numpy') else np.array(inputs['input_1'])
            left = aux['left_env'].numpy() if hasattr(aux['left_env'], 'numpy') else np.array(aux['left_env'])
            right = aux['right_env'].numpy() if hasattr(aux['right_env'], 'numpy') else np.array(aux['right_env'])
            lab = aux.get('label')
            if lab is not None:
                lab = lab.numpy() if hasattr(lab, 'numpy') else np.array(lab)
                lab = np.atleast_1d(lab).flatten()
            if eeg.ndim == 1:
                eeg = eeg[None, :]
                left = left[None, :]
                right = right[None, :]
            return eeg, left, right, lab

        # Peek first batch to decide streaming vs in-memory
        limited_peek = dataset.take(1)
        first_batch = next(iter(limited_peek), None)
        if first_batch is None:
            raise ValueError("No batches in dataset; cannot fit CCA.")
        eeg0, left0, right0, lab0 = _get_batch(first_batch)
        if eeg0 is None:
            raise ValueError("No batches with left/right envelopes found; cannot fit CCA.")
        # Batched time-resolved: (B, T, D) -> treat as time-resolved, eeg_dim = D. Single window: (T, D).
        if eeg0.ndim == 3:
            time_resolved = eeg0.shape[1] > 1
            eeg_dim = eeg0.shape[2]
        else:
            time_resolved = eeg0.ndim == 2 and eeg0.shape[0] > 1
            eeg_dim = eeg0.shape[1]
        # Envelope feature dim: per time point (e.g. 17 lags). For (B,T,D) or (T,D) use last dim.
        env_dim = int(left0.shape[-1]) if left0.ndim >= 2 else int(left0.size)
        use_streaming = time_resolved and eeg_dim > MAX_EEG_DIM_CCA

        if use_streaming:
            env_dim_cca = int(left0.shape[-1]) if left0.ndim >= 2 else 17
            n_components_eeg = min(PCA_CCA_COMPONENTS, eeg_dim)
            use_both = self.train_cca_on_both_envelopes
            if use_both:
                self._env_dim_stream = env_dim_cca
            else:
                self._env_dim_stream = None  # paper style: single stream (attended only)
            print(f"\n  Streaming path (EEG dim {eeg_dim} > {MAX_EEG_DIM_CCA}): IncrementalPCA -> CCA ({'both envelopes' if use_both else 'attended envelope only (paper)'}) -> LDA.")
            incremental_pca = IncrementalPCA(n_components=n_components_eeg)
            limited = dataset.take(MAX_BATCHES)
            for batch in tqdm(limited, desc="Pass 1: fit IncrementalPCA"):
                eeg, left, right, lab = _get_batch(batch)
                if eeg is None:
                    continue
                chunk = np.vstack(eeg).astype(np.float32)
                incremental_pca.partial_fit(chunk)
            self._pca_eeg = incremental_pca

            def _cca_stream_gen():
                for batch in dataset.take(MAX_BATCHES):
                    eeg, left, right, lab = _get_batch(batch)
                    if eeg is None:
                        continue
                    B = eeg.shape[0]
                    X_chunk = np.vstack(eeg).astype(np.float32)
                    X_chunk = self._pca_eeg.transform(X_chunk).astype(np.float32)
                    labels_b = np.array([int(lab[i]) if lab is not None and i < len(lab) else 0 for i in range(B)])
                    if use_both:
                        left_stacked = np.vstack([left[i] for i in range(B)]).astype(np.float32)
                        right_stacked = np.vstack([right[i] for i in range(B)]).astype(np.float32)
                        Y_chunk = np.hstack([left_stacked, right_stacked])
                    else:
                        Y_chunk = np.vstack([left[i] if labels_b[i] == 0 else right[i] for i in range(B)]).astype(np.float32)
                    yield {'input_1': X_chunk, 'input_2': Y_chunk}

            env_dim_y = 2 * env_dim_cca if use_both else env_dim_cca
            out_spec = (
                tf.TensorSpec(shape=(None, n_components_eeg), dtype=tf.float32),
                tf.TensorSpec(shape=(None, env_dim_y), dtype=tf.float32)
            )
            ds_cca = tf.data.Dataset.from_generator(
                _cca_stream_gen,
                output_signature={'input_1': out_spec[0], 'input_2': out_spec[1]}
            )
            max_cca = min(n_components_eeg, env_dim_y)
            cca_dims = min(self.cca_dims, max_cca)
            if cca_dims != self.cca_dims:
                print(f"  CCA dims capped: {self.cca_dims} -> {cca_dims}")
                self.cca_dims = cca_dims
            cca_batch_count = MAX_BATCHES
            print(f"  Training CCA (EEG ↔ {'both envelopes' if use_both else 'attended envelope'}), J={cca_dims} (up to {cca_batch_count} batches)...")
            rot_x, rot_y, mean_x, mean_y, e_vals = cca_module.calculate_cca_parameters_from_dataset(
                ds_cca, cca_dims, regularization=self.regularization, mini_batch_count=cca_batch_count)
            self.cca_params = {'rot_x': rot_x, 'rot_y': rot_y, 'mean_x': mean_x, 'mean_y': mean_y, 'eigenvalues': e_vals}
            first_cc = np.sqrt(np.clip(e_vals[0], 0, 1))
            print(f"    First canonical correlation: {first_cc:.6f}")
            if first_cc < 0.1:
                print(f"    ⚠ Low first canonical correlation ({first_cc:.3f}). Healthy range ~0.2–0.4. Check EEG–envelope alignment and envelope quality.")
            self.is_fitted = True
            if self.use_lda:
                print("  Fitting LDA on f = ρ_left − ρ_right (streaming)...")
                all_f, all_labels_lda = [], []
                for batch in tqdm(dataset.take(MAX_BATCHES), desc="Pass 3: compute rho for LDA"):
                    eeg, left, right, lab = _get_batch(batch)
                    if eeg is None:
                        continue
                    for i in range(eeg.shape[0]):
                        side = ('left', 'right') if use_both else (None, None)
                        rho_l = self._compute_rho(eeg[i], left[i], envelope_side=side[0])
                        rho_r = self._compute_rho(eeg[i], right[i], envelope_side=side[1])
                        all_f.append(rho_l - rho_r)
                        all_labels_lda.append(int(lab[i]) if lab is not None and i < len(lab) else 0)
                self._fit_lda_common(np.array(all_f, dtype=np.float32), np.array(all_labels_lda, dtype=np.int64))
        else:
            # In-memory path
            use_both = self.train_cca_on_both_envelopes
            all_eeg, all_left, all_right, all_labels = [], [], [], []
            print("\nCollecting training windows for CCA (EEG ↔ {})...".format("both envelopes" if use_both else "attended envelope (paper)"))
            for batch in tqdm(dataset.take(MAX_BATCHES), desc="Collecting data"):
                eeg, left, right, lab = _get_batch(batch)
                if eeg is None:
                    continue
                for i in range(eeg.shape[0]):
                    all_eeg.append(eeg[i])
                    all_left.append(left[i])
                    all_right.append(right[i])
                    all_labels.append(int(lab[i]) if lab is not None and i < len(lab) else 0)
            if not all_eeg:
                raise ValueError("No batches with left/right envelopes found; cannot fit CCA.")
            n_windows = len(all_eeg)
            time_resolved_collected = time_resolved or (all_eeg[0].ndim == 2 and all_eeg[0].shape[0] > 1)
            X = np.vstack(all_eeg).astype(np.float32)
            Y_left = np.vstack(all_left).astype(np.float32)
            Y_right = np.vstack(all_right).astype(np.float32)
            labels = np.array(all_labels, dtype=np.int64)
            if use_both:
                Y_cca = np.hstack([Y_left, Y_right]).astype(np.float32)
                self._env_dim_stream = Y_cca.shape[1] // 2
            else:
                if time_resolved_collected:
                    Y_cca = np.vstack([all_left[i] if labels[i] == 0 else all_right[i] for i in range(n_windows)]).astype(np.float32)
                else:
                    Y_cca = np.where(labels[:, None] == 0, Y_left, Y_right).astype(np.float32)
                self._env_dim_stream = None
            if time_resolved_collected:
                print(f"  Time-resolved: {n_windows} windows → {X.shape[0]} time points. EEG shape: {X.shape}, Y: {Y_cca.shape}")
            else:
                print(f"  Collected {X.shape[0]} windows. EEG shape: {X.shape}, Y: {Y_cca.shape}")
            n_components_eeg = min(PCA_CCA_COMPONENTS, X.shape[1], X.shape[0] - 1)
            env_dim_one = Y_left.shape[1]
            n_components_env = min(PCA_CCA_COMPONENTS, env_dim_one, X.shape[0] - 1)
            if X.shape[1] > MAX_EEG_DIM_CCA and n_components_eeg >= 16:
                print(f"  Reducing EEG dimension {X.shape[1]} -> {n_components_eeg} (PCA) to avoid OOM.")
                self._pca_eeg = PCA(n_components=n_components_eeg, random_state=42).fit(X)
                X = self._pca_eeg.transform(X).astype(np.float32)
            if use_both and env_dim_one > MAX_ENV_DIM_CCA and n_components_env >= 16:
                print(f"  Reducing envelope dimension per stream {env_dim_one} -> {n_components_env} (PCA) to avoid OOM.")
                Y_combined = np.vstack([Y_left, Y_right]).astype(np.float32)
                self._pca_env = PCA(n_components=n_components_env, random_state=42).fit(Y_combined)
                Y_left_pca = self._pca_env.transform(Y_left).astype(np.float32)
                Y_right_pca = self._pca_env.transform(Y_right).astype(np.float32)
                Y_cca = np.hstack([Y_left_pca, Y_right_pca]).astype(np.float32)
                self._env_dim_stream = n_components_env
            elif not use_both and env_dim_one > MAX_ENV_DIM_CCA and n_components_env >= 16:
                print(f"  Reducing envelope dimension {env_dim_one} -> {n_components_env} (PCA) to avoid OOM.")
                self._pca_env = PCA(n_components=n_components_env, random_state=42).fit(Y_cca)
                Y_cca = self._pca_env.transform(Y_cca).astype(np.float32)
            max_cca = min(X.shape[1], Y_cca.shape[1])
            cca_dims = min(self.cca_dims, max_cca)
            if cca_dims != self.cca_dims:
                print(f"  CCA dims capped: {self.cca_dims} -> {cca_dims}")
                self.cca_dims = cca_dims
            batch_size_cca = min(5000, X.shape[0])
            ds_cca = tf.data.Dataset.from_tensor_slices({'input_1': X, 'input_2': Y_cca}).batch(batch_size_cca)
            print(f"  Training CCA (EEG ↔ {'both envelopes' if use_both else 'attended envelope'}), J={cca_dims}...")
            rot_x, rot_y, mean_x, mean_y, e_vals = cca_module.calculate_cca_parameters_from_dataset(
                ds_cca, cca_dims, regularization=self.regularization, mini_batch_count=0)
            self.cca_params = {'rot_x': rot_x, 'rot_y': rot_y, 'mean_x': mean_x, 'mean_y': mean_y, 'eigenvalues': e_vals}
            first_cc = np.sqrt(np.clip(e_vals[0], 0, 1))
            print(f"    First canonical correlation: {first_cc:.6f}")
            if first_cc < 0.1:
                print(f"    ⚠ Low first canonical correlation ({first_cc:.3f}). Healthy range ~0.2–0.4. Check EEG–envelope alignment and envelope quality.")
            self.is_fitted = True
            if self.use_lda:
                print("  Fitting LDA on f = ρ_left − ρ_right...")
                if time_resolved_collected:
                    self._fit_lda_from_windows(all_eeg, all_left, all_right, labels)
                else:
                    self._fit_lda_from_arrays(X, Y_left, Y_right, labels)
            # Diagnostic: correlation on training sample (if poor on val/test but good here → generalization issue)
            side = (None, None) if getattr(self, '_env_dim_stream', None) is None else ('left', 'right')
            if time_resolved_collected and len(all_eeg) >= 50:
                n_sample = min(500, len(all_eeg))
                diff_left_att, diff_right_att = [], []
                for i in range(n_sample):
                    rho_l = self._compute_rho(all_eeg[i], all_left[i], envelope_side=side[0])
                    rho_r = self._compute_rho(all_eeg[i], all_right[i], envelope_side=side[1])
                    sl, sr = self._rho_to_score(rho_l), self._rho_to_score(rho_r)
                    if labels[i] == 0:
                        diff_left_att.append(sl - sr)
                    else:
                        diff_right_att.append(sr - sl)
                d_l = np.mean(diff_left_att) if diff_left_att else 0.0
                d_r = np.mean(diff_right_att) if diff_right_att else 0.0
                print(f"  On training sample (n={n_sample}): attended−unattended corr diff: LEFT={d_l:.4f}, RIGHT={d_r:.4f} (healthy ~0.05–0.15; if val/test ≈0, try more training subjects)")
        print("\n" + "="*80)
        print("CCA (EEG ↔ {}) TRAINING COMPLETE".format("BOTH ENVELOPES" if self.train_cca_on_both_envelopes else "ATTENDED ENVELOPE (PAPER)"))
        print("="*80)
        print(f"  CCA dimensions: {self.cca_dims}, regularization: {self.regularization}")
        print(f"  At test: {'LDA on f=ρ_left−ρ_right' if self.lda_model is not None else 'predict right if ρ_right > ρ_left'}")
        print("="*80 + "\n")
    
    def _compute_rho(self, X: np.ndarray, Y: np.ndarray, envelope_side: str = None) -> np.ndarray:
        """Per-dimension canonical correlation for (X, Y). Returns length-cca_dims vector.
        When CCA was trained on both envelopes, envelope_side must be 'left' or 'right' to use the correct half of rot_y/mean_y."""
        if self.cca_params is None:
            raise ValueError("Model not fitted.")
        rot_x = np.asarray(self.cca_params['rot_x'])
        rot_y = np.asarray(self.cca_params['rot_y'])
        mean_x = np.asarray(self.cca_params['mean_x']).reshape(1, -1)
        mean_y = np.asarray(self.cca_params['mean_y']).reshape(1, -1)
        env_dim_stream = getattr(self, '_env_dim_stream', None)
        if env_dim_stream is not None:
            if envelope_side not in ('left', 'right'):
                raise ValueError("CCA was trained on both envelopes; pass envelope_side='left' or 'right'.")
            d = env_dim_stream
            if envelope_side == 'left':
                mean_y = mean_y[:, :d]
                rot_y = rot_y[:d, :]
            else:
                mean_y = mean_y[:, d:]
                rot_y = rot_y[d:, :]
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
        if self._pca_eeg is not None:
            X = self._pca_eeg.transform(X).astype(np.float32)
        if self._pca_env is not None:
            Y = self._pca_env.transform(Y).astype(np.float32)
        U = (X - mean_x) @ rot_x
        V = (Y - mean_y) @ rot_y
        J = U.shape[1]
        n_samples = U.shape[0]
        if n_samples == 1:
            # Single window: Pearson across time undefined (one sample). Use vector-level normalized score
            # so scale is correlation-like ([-1, 1]) and LDA gets comparable features across windows.
            u, v = U[0], V[0]
            denom = (np.linalg.norm(u) * np.linalg.norm(v)) + 1e-8
            rho = ((u * v) / denom).astype(np.float32)
            return rho
        rho = np.zeros(J, dtype=np.float32)
        for j in range(J):
            u, v = U[:, j], V[:, j]
            u = u - np.mean(u)
            v = v - np.mean(v)
            d = np.sqrt(np.sum(u**2) * np.sum(v**2)) + 1e-8
            rho[j] = np.sum(u * v) / d
        return rho
    
    def _fit_lda_from_windows(self, all_eeg: list, all_left: list, all_right: list, labels: np.ndarray):
        """Fit LDA on f = ρ_left − ρ_right from per-window arrays (time-resolved: each window (T, C))."""
        side = (None, None) if getattr(self, '_env_dim_stream', None) is None else ('left', 'right')
        all_f = []
        for i in range(len(all_eeg)):
            rho_l = self._compute_rho(all_eeg[i], all_left[i], envelope_side=side[0])
            rho_r = self._compute_rho(all_eeg[i], all_right[i], envelope_side=side[1])
            if getattr(self, 'use_first_cca_component_only', False):
                all_f.append(np.array([rho_l[0] - rho_r[0]], dtype=np.float32))
            else:
                all_f.append(rho_l - rho_r)
        self._fit_lda_common(np.array(all_f, dtype=np.float32), labels)

    def _fit_lda_from_arrays(self, X: np.ndarray, Y_left: np.ndarray, Y_right: np.ndarray, labels: np.ndarray):
        """Fit LDA on f = ρ_left − ρ_right using already-transformed stacked arrays (flattened windows)."""
        n = X.shape[0]
        if self._pca_env is not None:
            Y_left = self._pca_env.transform(Y_left).astype(np.float32)
            Y_right = self._pca_env.transform(Y_right).astype(np.float32)
        rot_x = np.asarray(self.cca_params['rot_x'])
        rot_y = np.asarray(self.cca_params['rot_y'])
        mean_x = np.asarray(self.cca_params['mean_x']).reshape(1, -1)
        mean_y = np.asarray(self.cca_params['mean_y']).reshape(1, -1)
        env_dim_stream = getattr(self, '_env_dim_stream', None)
        if env_dim_stream is not None:
            d = env_dim_stream
            mean_y_left = mean_y[:, :d]
            mean_y_right = mean_y[:, d:]
            rot_y_left = rot_y[:d, :]
            rot_y_right = rot_y[d:, :]
        else:
            mean_y_left = mean_y_right = mean_y
            rot_y_left = rot_y_right = rot_y
        J = rot_x.shape[1]
        all_f = []
        for i in range(n):
            U = (X[i:i+1] - mean_x) @ rot_x
            V_l = (Y_left[i:i+1] - mean_y_left) @ rot_y_left
            V_r = (Y_right[i:i+1] - mean_y_right) @ rot_y_right
            u, v_l, v_r = U[0], V_l[0], V_r[0]
            denom_l = (np.linalg.norm(u) * np.linalg.norm(v_l)) + 1e-8
            denom_r = (np.linalg.norm(u) * np.linalg.norm(v_r)) + 1e-8
            rho_l = ((u * v_l) / denom_l).astype(np.float32)
            rho_r = ((u * v_r) / denom_r).astype(np.float32)
            if getattr(self, 'use_first_cca_component_only', False):
                all_f.append(np.array([rho_l[0] - rho_r[0]], dtype=np.float32))
            else:
                all_f.append(rho_l - rho_r)
        self._fit_lda_common(np.array(all_f, dtype=np.float32), labels)

    def _fit_lda_common(self, F: np.ndarray, labels: np.ndarray):
        """Fit LDA on feature matrix F and labels; set lda_model/lda_scaler or None on failure."""
        if len(F) < 2:
            print("  → LDA skipped: too few samples")
            return
        if np.any(~np.isfinite(F)):
            print("  → LDA skipped: non-finite values in f")
            return
        if len(np.unique(labels)) < 2:
            print("  → LDA skipped: need two classes")
            return
        if F.shape[0] < F.shape[1] + 2:
            print("  → LDA skipped: too few samples for feature dimension (use threshold instead)")
            return
        try:
            self.lda_scaler = StandardScaler()
            F_scaled = self.lda_scaler.fit_transform(F)
            n_classes = len(np.unique(labels))
            priors = np.ones(n_classes) / n_classes
            self.lda_model = LinearDiscriminantAnalysis(priors=priors, shrinkage='auto', solver='lsqr')
            self.lda_model.fit(F_scaled, labels)
            print(f"  ✓ LDA fitted on {len(labels)} windows (f = ρ_left − ρ_right), shrinkage=auto")
        except Exception as e:
            print(f"  → LDA failed ({e}); using threshold ρ_right > ρ_left")
            self.lda_model = None
            self.lda_scaler = None
    
    def _rho_to_score(self, rho: np.ndarray) -> float:
        """Convert per-dimension rho to scalar: first component only or weighted sum."""
        if getattr(self, 'use_first_cca_component_only', False) and len(rho) > 0:
            return float(rho[0])
        w = np.exp(-np.arange(len(rho)) * 0.15)
        w = w / w.sum()
        return float((rho * w).sum())

    def score_window(self, X: np.ndarray, Y: np.ndarray, use_cca_left: bool = None, envelope_side: str = None) -> float:
        """Score one window: CCA correlation of (X, Y). When CCA trained on both envelopes, pass envelope_side='left' or 'right'."""
        rho = self._compute_rho(X, Y, envelope_side=envelope_side)
        return self._rho_to_score(rho)
    
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
        
        # Normalize projections to unit length per sample (not per batch)
        # This ensures fair comparison across samples without batch statistics leakage
        # axis=1 normalizes each sample independently (per-sample normalization)
        proj_eeg_norm = proj_eeg / (np.linalg.norm(proj_eeg, axis=1, keepdims=True) + 1e-8)
        proj_env_norm = proj_env / (np.linalg.norm(proj_env, axis=1, keepdims=True) + 1e-8)
        
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
        """Predict attention using CCA (trained on both envelopes): ρ_left, ρ_right; LDA on f=ρ_left−ρ_right or threshold."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        all_predictions = []
        all_targets = []
        all_left_scores = []
        all_right_scores = []
        
        for batch in tqdm(dataset, desc="Predicting"):
            if isinstance(batch, tuple):
                inputs, aux = batch
            else:
                inputs = batch
                aux = None
            
            if aux is None or 'left_env' not in aux or 'right_env' not in aux:
                raise ValueError("Dataset must provide left/right envelopes for prediction.")
            
            eeg_np = inputs['input_1'].numpy() if hasattr(inputs['input_1'], 'numpy') else np.array(inputs['input_1'])
            left_np = aux['left_env'].numpy() if hasattr(aux['left_env'], 'numpy') else np.array(aux['left_env'])
            right_np = aux['right_env'].numpy() if hasattr(aux['right_env'], 'numpy') else np.array(aux['right_env'])
            if eeg_np.ndim == 1:
                eeg_np = eeg_np[None, :]
                left_np = left_np[None, :]
                right_np = right_np[None, :]
            B = eeg_np.shape[0]
            left_scores = np.empty(B, dtype=np.float32)
            right_scores = np.empty(B, dtype=np.float32)
            f_list = []
            side = (None, None) if getattr(self, '_env_dim_stream', None) is None else ('left', 'right')
            for i in range(B):
                rho_l = self._compute_rho(eeg_np[i], left_np[i], envelope_side=side[0])
                rho_r = self._compute_rho(eeg_np[i], right_np[i], envelope_side=side[1])
                left_scores[i] = self._rho_to_score(rho_l)
                right_scores[i] = self._rho_to_score(rho_r)
                # LDA uses full vector f or first component only to match scoring
                if getattr(self, 'use_first_cca_component_only', False):
                    f_list.append(np.array([rho_l[0] - rho_r[0]], dtype=np.float32))
                else:
                    f_list.append(rho_l - rho_r)
            
            all_left_scores.extend(left_scores)
            all_right_scores.extend(right_scores)
            # Paper-style: LDA on f = ρ_left − ρ_right (vector) or threshold on scalar
            if self.lda_model is not None and self.lda_scaler is not None:
                f_batch = np.array(f_list, dtype=np.float32)
                f_scaled = self.lda_scaler.transform(f_batch)
                window_predictions = self.lda_model.predict(f_scaled).astype(np.int64)
            else:
                window_predictions = (right_scores > left_scores).astype(np.int64)
            all_predictions.extend(window_predictions)
            
            if 'label' in aux:
                lab = aux['label'].numpy() if hasattr(aux['label'], 'numpy') else np.array(aux['label'])
                lab = np.atleast_1d(lab).flatten().astype(np.int64)
                if len(lab) == B:
                    all_targets.extend(lab)
                elif len(lab) == 1:
                    all_targets.extend([int(lab[0])] * B)
                else:
                    all_targets.extend(lab[:B])
        
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
        if len(val_targets) == 0:
            print("  ⚠️  Validation set is empty; validation accuracy set to NaN.")
            return float('nan')
        val_accuracy = accuracy_score(val_targets, val_predictions)
        return val_accuracy
    
    def test(self, test_dataset: tf.data.Dataset) -> Dict:
        """Test with comprehensive metrics."""
        predictions, targets = self.model.predict(test_dataset)
        if len(targets) == 0:
            raise ValueError("Test set is empty; cannot compute metrics.")
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


def compute_per_subject_accuracy(predictions: np.ndarray, targets: np.ndarray,
                                  test_indices: List[int], window_indices: List) -> Optional[Dict]:
    """Per-subject test accuracy and mean ± SEM (paper-style: mean error ± SEM across subjects)."""
    if not test_indices or len(test_indices) != len(predictions):
        return None
    from collections import defaultdict
    by_subj = defaultdict(lambda: {'pred': [], 'tar': []})
    for i, idx in enumerate(test_indices):
        if idx >= len(window_indices):
            continue
        win = window_indices[idx]
        if len(win) < 4:
            continue
        subj = win[3]
        by_subj[subj]['pred'].append(int(predictions[i]))
        by_subj[subj]['tar'].append(int(targets[i]))
    if not by_subj:
        return None
    per_subj_acc = {}
    for subj, d in by_subj.items():
        if len(d['pred']) > 0:
            per_subj_acc[subj] = float(accuracy_score(d['tar'], d['pred']))
    if not per_subj_acc:
        return None
    accs = np.array(list(per_subj_acc.values()))
    n = len(accs)
    sem = float(np.std(accs, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return {
        'per_subject_accuracy': per_subj_acc,
        'mean_accuracy': float(np.mean(accs)),
        'sem_accuracy': sem,
        'n_subjects': n
    }


def compute_trial_level_metrics(predictions: np.ndarray, targets: np.ndarray,
                                test_indices: List[int], window_indices: List) -> Optional[Dict]:
    """Aggregate window-level predictions to trial level by majority vote. Returns trial_accuracy and trial_balanced_accuracy or None."""
    if not test_indices or len(test_indices) != len(predictions):
        return None
    from collections import defaultdict
    agg_pred = defaultdict(list)
    agg_tar = defaultdict(lambda: None)
    for i, idx in enumerate(test_indices):
        if idx >= len(window_indices):
            continue
        win = window_indices[idx]
        if len(win) < 5:
            continue
        subj, trial_idx = win[3], win[4]
        agg_pred[(subj, trial_idx)].append(int(predictions[i]))
        agg_tar[(subj, trial_idx)] = int(targets[i])
    if not agg_pred:
        return None
    trial_pred_list = [int(round(np.mean(ps))) for ps in agg_pred.values()]
    trial_tar_list = [agg_tar[k] for k in agg_pred]
    return {
        'trial_accuracy': float(accuracy_score(trial_tar_list, trial_pred_list)),
        'trial_balanced_accuracy': float(balanced_accuracy_score(trial_tar_list, trial_pred_list))
    }


def calculate_detailed_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict:
    """Calculate comprehensive detailed metrics."""
    accuracy = accuracy_score(targets, predictions)
    try:
        precision = precision_score(targets, predictions, average='binary', zero_division=0)
        recall = recall_score(targets, predictions, average='binary', zero_division=0)
        f1 = f1_score(targets, predictions, average='binary', zero_division=0)
    except ValueError:
        precision = recall = f1 = 0.0
    
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
    
    # Confusion matrix (labels=[0,1] forces 2x2 so ravel() is always 4 elements)
    cm = confusion_matrix(targets, predictions, labels=[0, 1])
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
                                val_accuracy: float, output_path: Path,
                                per_subject_metrics: Optional[Dict] = None):
    """Generate comprehensive analysis report. Paper-style: report mean accuracy ± SEM across subjects when available."""
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS - {OPTIMAL_CONFIG['name']}")
    print(f"{'='*80}")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    if per_subject_metrics:
        print(f"Mean accuracy ± SEM (across {per_subject_metrics['n_subjects']} test subjects): {per_subject_metrics['mean_accuracy']:.4f} ± {per_subject_metrics['sem_accuracy']:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    roc_auc = detailed_metrics.get('advanced_metrics', {}).get('roc_auc_score', 0.5)
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    # Performance analysis
    if results['accuracy'] < 0.55:
        print(f"\n⚠️  PERFORMANCE ANALYSIS:")
        print(f"   Accuracy is near chance (50%). If validation/test correlation differences are ~0 but")
        print(f"   training first canonical correlation was >0.1, CCA is not generalizing to held-out subjects.")
        print(f"   Try in order:")
        print(f"   1. Default: CCA on attended envelope only (paper). Envelope: scale_only, Hilbert on, bandpass 2–8 Hz. Try --cca_both_envelopes to train on both.")
        print(f"   2. Sliding windows: optional --max_train_windows N to subsample train set and reduce temporal redundancy")
        print(f"   3. Use 8 s window: --window_size 512 @ 64 Hz (default)")
        print(f"   4. Lag range: --min_lag_ms 50 --max_lag_ms 400 if 0–300 ms is insufficient")
        print(f"   5. CCA: --cca_dims 16–20, --regularization 0.01–0.02")
        print(f"   6. Verify envelope mapping (left/right, Das audio paths, Fulsang attend_lr)")
    
    print(f"{'='*80}\n")
    
    save_comprehensive_results(results, detailed_metrics, val_accuracy, output_path)


def save_comprehensive_results(results: Dict, detailed_metrics: Dict, 
                             val_accuracy: float, output_path: Path):
    """Save comprehensive results to files. Paper-style: test_accuracy = window-level accuracy (same as paper)."""
    val_acc_serializable = val_accuracy if (isinstance(val_accuracy, (int, float)) and not np.isnan(val_accuracy)) else None
    results_to_save = {
        'configuration': OPTIMAL_CONFIG,
        'validation_accuracy': val_acc_serializable,
        'test_accuracy': results['accuracy'],  # Paper-style: window-level accuracy
        'roc_auc': detailed_metrics['advanced_metrics']['roc_auc_score'],
        'matthews_correlation': detailed_metrics['advanced_metrics']['matthews_correlation_coefficient'],
        'balanced_accuracy': detailed_metrics['advanced_metrics']['balanced_accuracy']
    }
    if 'trial_accuracy' in detailed_metrics.get('advanced_metrics', {}):
        results_to_save['trial_accuracy'] = detailed_metrics['advanced_metrics']['trial_accuracy']
        results_to_save['trial_balanced_accuracy'] = detailed_metrics['advanced_metrics']['trial_balanced_accuracy']
    if 'mean_accuracy_across_subjects' in detailed_metrics.get('advanced_metrics', {}):
        results_to_save['mean_accuracy_across_subjects'] = detailed_metrics['advanced_metrics']['mean_accuracy_across_subjects']
        results_to_save['sem_accuracy_across_subjects'] = detailed_metrics['advanced_metrics']['sem_accuracy_across_subjects']
        results_to_save['n_test_subjects'] = detailed_metrics['advanced_metrics'].get('n_test_subjects')
    
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
                  val_ratio: float = 0.25,
                  max_train_windows: Optional[int] = None):
    """Split dataset into train/val/test sets using subject-level splitting.
    
    Paper-style evaluation: single train/val/test split (no K-fold cross-validation).
    Primary metric: window-level test accuracy; trial-level accuracy (majority vote
    per trial) is also reported. Keep this behavior to match the paper.
    
    CRITICAL: This prevents data leakage by ensuring no subject appears in multiple splits.
    Random window-level splitting would mix windows from the same subject/session across
    splits, causing autocorrelation and inflated accuracy (10-20%).
    
    Window indices format: (start_idx, end_idx, label, subject_id, trial_idx, dataset)
    
    If max_train_windows is set, the training set is randomly subsampled to at most that
    many windows (reduces redundancy from overlapping sliding windows; optional).
    """
    print("\n" + "="*80)
    print("SUBJECT-LEVEL SPLITTING (Preventing Data Leakage)")
    print("="*80)
    
    # Group windows by subject to prevent data leakage
    subject_windows = {}  # {subject_id: [window_indices]}
    
    for window_idx, window_info in enumerate(dataset_wrapper.window_indices):
        # Extract subject_id from window info (index 3)
        # Format: (start_idx, end_idx, label, subject_id, trial_idx, dataset)
        if len(window_info) >= 4:
            subject_id = window_info[3]
        else:
            # Fallback for old format (shouldn't happen with new CombinedDataset)
            subject_id = 'unknown'
        
        if subject_id not in subject_windows:
            subject_windows[subject_id] = []
        subject_windows[subject_id].append(window_idx)
    
    print(f"Found {len(subject_windows)} unique subjects:")
    for subject_id, windows in sorted(subject_windows.items()):
        print(f"  {subject_id}: {len(windows)} windows")
    
    # Partition subjects by dataset (Das vs Fulsang) for optional per-dataset train cap
    # window_info[5] is 'Das' or 'Fulsang' (or 'COMBINED_DAS', 'Fulsang-MWF', etc.)
    def _classify_dataset(dataset_name: str, subject_id: str) -> str:
        """Return 'das' or 'fulsang'. Use subject_id pattern as fallback when dataset tag is unknown."""
        tag = str(dataset_name).strip().lower()
        if tag in ('das', 'das-mwf', 'combined_das', 'das-preprocessed', 'daspreprocess'):
            return 'das'
        if tag in ('fulsang', 'fulsang-mwf', 'fuglsang'):
            return 'fulsang'
        # Fallback: Das often uses S1-S9 or 1-9; Fulsang uses S01-S18 or 01-18
        sid = str(subject_id).strip()
        if not sid:
            return 'das'
        # Single digit after optional S (e.g. S1, S2, 1, 2, ..., 9) -> typical Das 9-subject ID
        if re.match(r'^S?\d{1}$', sid) or (sid.isdigit() and 1 <= int(sid) <= 9):
            return 'das'
        # Two digits or 10+ (e.g. S01, S18, 01, 18) -> typical Fulsang
        return 'fulsang'
    
    das_subjects = []
    fulsang_subjects = []
    for subject_id in subject_windows:
        win_info = dataset_wrapper.window_indices[subject_windows[subject_id][0]]
        dataset_name = win_info[5] if len(win_info) >= 6 else 'unknown'
        kind = _classify_dataset(dataset_name, subject_id)
        if kind == 'das':
            das_subjects.append(subject_id)
        else:
            fulsang_subjects.append(subject_id)
    
    print(f"Partitioned by dataset: {len(das_subjects)} Das ({', '.join(sorted(das_subjects))}), {len(fulsang_subjects)} Fulsang ({', '.join(sorted(fulsang_subjects))})")
    
    np.random.seed(42)  # Fixed seed for reproducibility
    
    # Subject-level split: train_ratio of subjects for train, rest split 50/50 val/test
    subjects = list(subject_windows.keys())
    np.random.shuffle(subjects)
    n_subjects = len(subjects)
    n_train_subjects = max(1, int(train_ratio * n_subjects))
    n_heldout = n_subjects - n_train_subjects
    n_val_subjects = n_heldout // 2
    n_test_subjects = n_heldout - n_val_subjects
    if n_subjects >= 3 and n_val_subjects < 1:
        n_val_subjects = 1
        n_test_subjects = n_heldout - 1
    if n_subjects < 3:
        print(f"  ⚠️  WARNING: Only {n_subjects} subject(s). Val or test may be empty.")
    train_subjects = subjects[:n_train_subjects]
    val_subjects = subjects[n_train_subjects:n_train_subjects + n_val_subjects]
    test_subjects = subjects[n_train_subjects + n_val_subjects:]
    print(f"\nSubject-level split: {n_train_subjects} train, {n_val_subjects} val, {n_test_subjects} test subjects (train_ratio={train_ratio}).")
    
    print(f"\nSubject-wise split:")
    print(f"  Train subjects: {len(train_subjects)} ({', '.join(sorted(train_subjects))})")
    print(f"  Val subjects:   {len(val_subjects)} ({', '.join(sorted(val_subjects))})")
    print(f"  Test subjects: {len(test_subjects)} ({', '.join(sorted(test_subjects))})")
    
    # Collect window indices for each split (train = only from train subjects)
    train_indices = []
    for subject_id in train_subjects:
        train_indices.extend(subject_windows[subject_id])
    
    if max_train_windows is not None and len(train_indices) > max_train_windows:
        np.random.seed(42)
        n_take = min(max_train_windows, len(train_indices))
        train_indices = list(np.random.choice(train_indices, size=n_take, replace=False))
        print(f"  Subsampled train windows to {n_take} (--max_train_windows; reduces sliding-window redundancy).")
    
    val_indices = []
    for subject_id in val_subjects:
        val_indices.extend(subject_windows[subject_id])
    
    test_indices = []
    for subject_id in test_subjects:
        test_indices.extend(subject_windows[subject_id])
    
    print(f"\nWindow-wise split:")
    print(f"  Train windows: {len(train_indices)} (from {len(train_subjects)} subjects; {100*len(train_indices)/len(dataset_wrapper.window_indices):.1f}% of all windows)")
    if len(train_indices) > 8000:
        print(f"  Note: many overlapping train windows; optional --max_train_windows N subsamples for less redundancy.")
    print(f"  Val windows: {len(val_indices)} ({100*len(val_indices)/len(dataset_wrapper.window_indices):.1f}%)")
    print(f"  Test windows: {len(test_indices)} ({100*len(test_indices)/len(dataset_wrapper.window_indices):.1f}%)")
    print("="*80 + "\n")
    
    # Create separate datasets for each split
    class SplitDataset:
        def __init__(self, parent, indices):
            self.parent = parent
            self.indices = indices
            self.window_indices = [parent.window_indices[i] for i in indices]
        
        def create_tf_dataset(self, batch_size):
            use_tr = getattr(self.parent, 'use_time_resolved', True)
            def generator():
                for idx in self.indices:
                    window_info = self.parent.window_indices[idx]
                    start_idx, end_idx, label = window_info[0], window_info[1], window_info[2]
                    eeg_window = self.parent.combined_dataset.eeg_data[start_idx:end_idx]
                    eeg_window = self.parent._preprocess_window(eeg_window)
                    if use_tr and getattr(self.parent, 'eeg_lag_taps', 1) > 1:
                        eeg_window = make_lagged_eeg(eeg_window, self.parent.eeg_lag_taps)
                    left_env, right_env = self.parent.combined_dataset.get_envelope_window(start_idx, end_idx)
                    left_env = left_env.flatten().astype(np.float32)
                    right_env = right_env.flatten().astype(np.float32)
                    if self.parent.use_time_lags:
                        left_env = make_lagged_audio(left_env.reshape(-1, 1), self.parent.lag_samples, self.parent.fs)
                        right_env = make_lagged_audio(right_env.reshape(-1, 1), self.parent.lag_samples, self.parent.fs)
                    if use_tr:
                        left_env = left_env.astype(np.float32)
                        right_env = right_env.astype(np.float32)
                        attended_env = left_env if label == 0 else right_env
                        yield (
                            {'input_1': eeg_window.astype(np.float32), 'input_2': attended_env},
                            {'left_env': left_env, 'right_env': right_env, 'label': np.array(label, dtype=np.float32)}
                        )
                    else:
                        left_env = left_env.flatten().astype(np.float32)
                        right_env = right_env.flatten().astype(np.float32)
                        attended_env = left_env if label == 0 else right_env
                        yield (
                            {'input_1': eeg_window.flatten().astype(np.float32), 'input_2': attended_env},
                            {'left_env': left_env, 'right_env': right_env, 'label': np.array(label, dtype=np.float32)}
                        )
            
            if use_tr:
                eeg_cols = getattr(self.parent, 'eeg_time_point_dim', self.parent.n_channels)
                output_signature = (
                    {
                        'input_1': tf.TensorSpec(shape=(self.parent.window_size, eeg_cols), dtype=tf.float32),
                        'input_2': tf.TensorSpec(shape=(self.parent.window_size, self.parent.num_lags), dtype=tf.float32)
                    },
                    {
                        'left_env': tf.TensorSpec(shape=(self.parent.window_size, self.parent.num_lags), dtype=tf.float32),
                        'right_env': tf.TensorSpec(shape=(self.parent.window_size, self.parent.num_lags), dtype=tf.float32),
                        'label': tf.TensorSpec(shape=(), dtype=tf.float32)
                    }
                )
            else:
                output_signature = (
                    {
                        'input_1': tf.TensorSpec(shape=(self.parent.window_size * self.parent.n_channels,), dtype=tf.float32),
                        'input_2': tf.TensorSpec(shape=(self.parent.envelope_dim,), dtype=tf.float32)
                    },
                    {
                        'left_env': tf.TensorSpec(shape=(self.parent.envelope_dim,), dtype=tf.float32),
                        'right_env': tf.TensorSpec(shape=(self.parent.envelope_dim,), dtype=tf.float32),
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
    
    return train_dataset, val_dataset, test_dataset, test_indices, dataset_wrapper


def _filter_window_indices_by_subset(window_indices: list, subset: str) -> list:
    """Filter windows by dataset subset: 'full', 'das', or 'fulsang'."""
    subset = str(subset or "full").strip().lower()
    if subset == "full":
        return window_indices

    def _classify_dataset(dataset_name: str, subject_id: str) -> str:
        tag = str(dataset_name).strip().lower()
        if tag in ('das', 'das-mwf', 'combined_das', 'das-preprocessed', 'daspreprocess'):
            return 'das'
        if tag in ('fulsang', 'fulsang-mwf', 'fuglsang'):
            return 'fulsang'
        sid = str(subject_id).strip()
        if not sid:
            return 'das'
        if re.match(r'^S?\d{1}$', sid) or (sid.isdigit() and 1 <= int(sid) <= 9):
            return 'das'
        return 'fulsang'

    out = []
    for win in window_indices:
        dataset_name = win[5] if len(win) >= 6 else 'unknown'
        subject_id = win[3] if len(win) >= 4 else ''
        kind = _classify_dataset(dataset_name, subject_id)
        if kind == subset:
            out.append(win)
    return out


def run_single_experiment(args, window_size_samples: int) -> Dict:
    """Run one train/val/test experiment for a given window size. Used for single run or window sweep.
    Returns dict with window_sec, window_samples, val_accuracy, test_accuracy, roc_auc, (and optional extras).
    """
    combined_dataset = CombinedDataset(**_combined_dataset_kwargs(args, window_size_samples))
    tf_dataset_wrapper = CombinedCCADataset(
        combined_dataset,
        use_time_lags=getattr(args, 'use_time_lags', True),
        min_lag_ms=getattr(args, 'min_lag_ms', 0.0),
        max_lag_ms=getattr(args, 'max_lag_ms', 300.0),
        fs=getattr(args, 'sampling_rate', None),
        max_windows=getattr(args, 'max_windows', None),
        eeg_lag_taps=getattr(args, 'eeg_lag_taps', 12)
    )
    subset = getattr(args, 'dataset_subset', 'full')
    if subset != 'full':
        before = len(tf_dataset_wrapper.window_indices)
        tf_dataset_wrapper.window_indices = _filter_window_indices_by_subset(tf_dataset_wrapper.window_indices, subset)
        after = len(tf_dataset_wrapper.window_indices)
        print(f"Dataset subset filter: {subset} ({before} -> {after} windows)")
    train_dataset, val_dataset, test_dataset, test_indices, _ = split_dataset(
        tf_dataset_wrapper,
        train_ratio=0.70,
        val_ratio=0.25,
        max_train_windows=getattr(args, 'max_train_windows', None)
    )
    OPTIMAL_CONFIG['window_size'] = window_size_samples
    if getattr(tf_dataset_wrapper, 'use_time_resolved', True):
        eeg_flat_dim = getattr(tf_dataset_wrapper, 'eeg_time_point_dim', tf_dataset_wrapper.n_channels)
        envelope_dim = tf_dataset_wrapper.num_lags
    else:
        envelope_dim = tf_dataset_wrapper.envelope_dim
        eeg_flat_dim = tf_dataset_wrapper.window_size * tf_dataset_wrapper.n_channels
    cca_model = OptimalCombinedCCAModel(
        cca_dims=OPTIMAL_CONFIG['cca_dims'],
        regularization=OPTIMAL_CONFIG['regularization'],
        window_size=window_size_samples,
        envelope_dim=envelope_dim,
        eeg_flat_dim=eeg_flat_dim,
        use_first_cca_component_only=getattr(args, 'use_first_cca_component_only', False),
        train_cca_on_both_envelopes=getattr(args, 'cca_both_envelopes', False)
    )
    output_dir = Path(args.output_dir)
    trainer = OptimalCombinedCCATrainer(cca_model, str(output_dir))
    val_accuracy = trainer.train(train_dataset, val_dataset)
    results = trainer.test(test_dataset)
    detailed_metrics = calculate_detailed_metrics(results['predictions'], results['targets'])
    window_sec = window_size_samples / getattr(args, 'target_sampling_rate', SAMPLING_RATE_HZ)
    out = {
        'window_sec': window_sec,
        'window_samples': window_size_samples,
        'val_accuracy': float(val_accuracy) if not np.isnan(val_accuracy) else None,
        'test_accuracy': results['accuracy'],
        'roc_auc': detailed_metrics['advanced_metrics']['roc_auc_score'],
    }
    trial_metrics = compute_trial_level_metrics(
        results['predictions'], results['targets'], test_indices, tf_dataset_wrapper.window_indices
    )
    if trial_metrics:
        out['trial_accuracy'] = trial_metrics['trial_accuracy']
        out['trial_balanced_accuracy'] = trial_metrics['trial_balanced_accuracy']
    return out


def run_window_sweep(args, window_seconds_list: Optional[List[float]] = None,
                    window_seconds_min: int = 1, window_seconds_max: int = 30) -> List[Dict]:
    """Run experiments for given window sizes in seconds.
    If window_seconds_list is provided, use it; else use range(window_seconds_min, window_seconds_max + 1).
    Default window_seconds_list is WINDOW_SWEEP_SECONDS = [1, 5, 10, 20, 30].
    Supports float seconds (e.g. 1.25 for paper encoder length).
    """
    fs = getattr(args, 'target_sampling_rate', SAMPLING_RATE_HZ)
    if window_seconds_list is None:
        window_seconds_list = list(range(window_seconds_min, window_seconds_max + 1))
    results = []
    for sec in window_seconds_list:
        samples = int(sec * fs)
        sec_str = f"{sec:.2f}" if isinstance(sec, float) else str(sec)
        print(f"\n{'='*80}\nWindow size: {sec_str}s ({samples} samples @ {fs} Hz)\n{'='*80}")
        try:
            r = run_single_experiment(args, samples)
            r['window_sec'] = sec
            results.append(r)
            test_acc_s = f"{r['test_accuracy']:.4f}" if r.get('test_accuracy') is not None else 'N/A'
            val_acc_s = f"{r['val_accuracy']:.4f}" if r.get('val_accuracy') is not None else 'N/A'
            roc_s = f"{r['roc_auc']:.4f}" if r.get('roc_auc') is not None else 'N/A'
            print(f"  → Test Acc: {test_acc_s}, Val Acc: {val_acc_s}, ROC-AUC: {roc_s}")
        except Exception as e:
            print(f"  → Failed: {e}")
            results.append({
                'window_sec': sec,
                'window_samples': samples,
                'val_accuracy': None,
                'test_accuracy': None,
                'roc_auc': None,
                'error': str(e)
            })
    return results


def save_window_sweep_results(results: List[Dict], output_dir: Path):
    """Save sweep results to CSV and optional accuracy-vs-window plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = output_dir / "window_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSweep results saved to {csv_path}")

    # Plot: test accuracy and ROC-AUC vs window (seconds)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    secs = [r['window_sec'] for r in results if r.get('test_accuracy') is not None]
    accs = [r['test_accuracy'] for r in results if r.get('test_accuracy') is not None]
    rocs = [r['roc_auc'] for r in results if r.get('roc_auc') is not None]
    if secs and accs:
        ax1.plot(secs, accs, 'b-o', label='Test accuracy', markersize=4)
    ax1.set_xlabel('Window size (s)')
    ax1.set_ylabel('Test accuracy', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim(0, 1.05)
    ax1.axhline(0.77, color='b', linestyle='--', alpha=0.5, label='77% target')
    ax1.legend(loc='upper left')
    if secs and rocs:
        ax2 = ax1.twinx()
        ax2.plot(secs, rocs, 'g-s', label='ROC-AUC', markersize=4)
        ax2.set_ylabel('ROC-AUC', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.set_ylim(0.5, 1.05)
        ax2.legend(loc='upper right')
    plt.title('Combined CCA: Test accuracy and ROC-AUC vs window size')
    plt.tight_layout()
    plot_path = output_dir / "window_sweep.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Sweep plot saved to {plot_path}")


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Combined Das+Fulsang CCA using Optimal_FULCCA configuration')
    parser.add_argument('--das_data_dir', type=str, default='das_combined_preprocessed',
                       help='Directory containing DAS data from das_preprocessing_combined.py')
    parser.add_argument('--no_auto_das_preprocess', action='store_true',
                       help='Do not run das_preprocessing_combined.py automatically when TFRecords are missing')
    parser.add_argument('--das_preprocessing_type', type=str, default='COMBINED_DAS',
                       choices=['COMBINED_DAS', 'MWF', 'DASPREPROCESS'],
                       help='Type of Das preprocessing')
    parser.add_argument('--fulsang_raw_dir', type=str, 
                       default='Data/Fulsang',
                       help='Fulsang data root (raw EEG in EEG/; put S*_expinfo.mat in Exp_Info/, EEG/, or here)')
    parser.add_argument('--fulsang_audio_dir', type=str,
                       default='Data/Fulsang/AUDIO',
                       help='Directory containing Fulsang audio data')
    parser.add_argument('--fulsang_expinfo_dir', type=str, default='Exp_Info',
                       help='Directory with S*_expinfo.mat (attend left/right). Default: Exp_Info (relative to cwd). Set to empty to disable.')
    parser.add_argument('--fulsang_mwf_dir', type=str, default='MWF_cleaned_Fuglsang',
                       help='Output directory for Fulsang MWF processing')
    parser.add_argument('--das_original_dir', type=str, default='Data/Das/4004271',
                       help='Directory containing original Das .mat files (for envelope extraction)')
    parser.add_argument('--das_audio_dir', type=str, default='Data/Das/4004271/stimuli/stimuli',
                       help='Directory containing Das audio files (for envelope extraction)')
    parser.add_argument('--combined_dataset_dir', type=str, default='combined_dataset',
                       help='Centralized directory for all processed files (default: combined_dataset)')
    parser.add_argument('--target_sampling_rate', type=int, default=PAPER_FS_HZ,
                       help=f'Target sampling rate in Hz (default: {PAPER_FS_HZ}, paper).')
    parser.add_argument('--window_sec', type=float, default=PAPER_WINDOW_SEC,
                       help=f'Window length in seconds (default: {PAPER_WINDOW_SEC}, paper CCA encoder length).')
    parser.add_argument('--window_size', type=int, default=OPTIMAL_CONFIG['window_size'],
                       help=f'Window size in samples (default: 8s @ 64 Hz = 512; used when --window_sec not set)')
    parser.add_argument('--overlap', type=float, default=0.25,
                       help='Window overlap fraction (default: 0.25)')
    parser.add_argument('--cca_dims', type=int, default=OPTIMAL_CONFIG['cca_dims'],
                       help=f'Number of CCA dimensions (default: {OPTIMAL_CONFIG["cca_dims"]})')
    parser.add_argument('--regularization', type=float, default=OPTIMAL_CONFIG['regularization'],
                       help=f'CCA regularization (default: {OPTIMAL_CONFIG["regularization"]})')
    parser.add_argument('--batch_size', type=int, default=OPTIMAL_CONFIG['batch_size'],
                       help=f'Batch size (default: {OPTIMAL_CONFIG["batch_size"]})')
    parser.add_argument('--output_dir', type=str, default='combined_cca_results',
                       help='Output directory for results')
    parser.add_argument('--single_run', action='store_true',
                       help='Explicit single-window run (default: single 8s run; no flag needed).')
    parser.add_argument('--window_seconds', type=str, default=None,
                       help='Accuracy vs window length: comma-separated seconds (e.g. 1,5,10,20,30). If set, runs sweep; else single 8s run.')
    parser.add_argument('--window_seconds_min', type=int, default=1,
                       help='Min window size in seconds for sweep when not using --window_seconds (default: 1)')
    parser.add_argument('--window_seconds_max', type=int, default=30,
                       help='Max window size in seconds for sweep when not using --window_seconds (default: 30)')
    parser.add_argument('--sampling_rate', type=float, default=128.0,
                       help='Sampling rate in Hz for time lags (default: 128, matches combined dataset)')
    parser.add_argument('--use_time_lags', action='store_true', default=True,
                       help='Use time-lagged envelope (default: 0–300 ms)')
    parser.add_argument('--no_time_lags', dest='use_time_lags', action='store_false',
                       help='Disable time-lagged envelope')
    parser.add_argument('--min_lag_ms', type=float, default=0.0,
                       help='Minimum lag in ms (default: 0)')
    parser.add_argument('--max_lag_ms', type=float, default=300.0,
                       help='Maximum lag in ms (default: 300; many studies use 0–300 or 50–400)')
    parser.add_argument('--eeg_lag_taps', type=int, default=12,
                       help='Backward model EEG taps per time point (L). Default 12 (paper 10–15).')
    parser.add_argument('--use_first_cca_component_only', action='store_true',
                       help='Use first canonical component ρ₁ only for scoring (default: weighted sum). Many AAD papers use ρ₁ only.')
    parser.add_argument('--bandpass_low_hz', type=float, default=2.0,
                       help='Butterworth bandpass low cutoff in Hz. Default 2 (speech-brain delta/theta). Set 0 to disable.')
    parser.add_argument('--bandpass_high_hz', type=float, default=8.0,
                       help='Butterworth bandpass high cutoff in Hz. Default 8 for speech-brain. Set <= low to disable.')
    parser.add_argument('--bandpass_order', type=int, default=1,
                       help='Butterworth bandpass filter order (default: 1, match Fulsang).')
    parser.add_argument('--no_use_hilbert_envelope', dest='use_hilbert_envelope', action='store_false', default=True,
                       help='Disable Hilbert envelope (default: Hilbert on for speech-brain).')
    parser.add_argument('--no_use_gammatone_filter', dest='use_gammatone_filter', action='store_false', default=True,
                       help='Disable gammatone filterbank for audio envelope extraction (default: gammatone on when available).')
    parser.add_argument('--cca_both_envelopes', action='store_true',
                       help='Train CCA on both envelopes (left|right). Default: paper style, train on attended envelope only.')
    parser.add_argument('--envelope_normalize', type=str, default='scale_only', choices=('zscore', 'scale_only'),
                       help="Envelope normalization: 'scale_only' preserves slow structure (default); 'zscore' removes mean.")
    parser.add_argument('--no_balance_envelope_energy', dest='balance_envelope_energy', action='store_false', default=True,
                       help='Do not scale right envelope to match left energy (default: balance on)')
    parser.add_argument('--max_windows', type=int, default=None,
                       help='Cap combined dataset to this many windows (keeps subject balance). e.g. 7000 for ~5k DAS + ~2k Fulsang.')
    parser.add_argument('--max_train_windows', type=int, default=None,
                       help='Randomly subsample training to at most N windows (optional; reduces overlap redundancy from sliding windows).')
    parser.add_argument('--dataset_subset', type=str, default='full', choices=('full', 'das', 'fulsang'),
                       help="Use only a dataset subset: 'full' (default), 'das', or 'fulsang'.")
    
    args = parser.parse_args()
    # Default: single 8s run (no flags). Use --window_seconds 1,5,10,20,30 for accuracy vs window length (paper figure).
    if getattr(args, 'window_seconds', None):
        print("Mode: Window sweep (accuracy vs decision window length).")
    else:
        print("Mode: Single run (8 s @ 64 Hz). Use --window_seconds 1,5,10,20,30 for sweep.")
    
    # Pre-check: COMBINED_DAS requires das_combined_preprocessed/tfrecords. Run preprocessing by default if missing.
    if args.das_preprocessing_type == "COMBINED_DAS":
        tfrecord_dir = Path(args.das_data_dir) / "tfrecords"
        tfrecord_files = list(tfrecord_dir.glob("*.tfrecords")) or list(tfrecord_dir.glob("*/*.tfrecords")) if tfrecord_dir.exists() else []
        if not tfrecord_files and not args.no_auto_das_preprocess:
            script_dir = Path(__file__).resolve().parent
            preprocess_script = script_dir / "das_preprocessing_combined.py"
            if preprocess_script.exists():
                print("\n" + "="*80)
                print("DAS combined TFRecords not found - running das_preprocessing_combined.py by default")
                print("="*80)
                cmd = [
                    sys.executable, str(preprocess_script),
                    "--data_dir", args.das_original_dir,
                    "--output_dir", args.das_data_dir,
                    "--audio_dir", args.das_audio_dir,
                ]
                ret = subprocess.run(cmd, cwd=str(script_dir))
                if ret.returncode != 0:
                    print(f"\nPreprocessing exited with code {ret.returncode}. Fix errors above or run manually: python das_preprocessing_combined.py")
                    sys.exit(ret.returncode)
                tfrecord_dir = Path(args.das_data_dir) / "tfrecords"
                tfrecord_files = list(tfrecord_dir.glob("*.tfrecords")) or list(tfrecord_dir.glob("*/*.tfrecords"))
        if not tfrecord_dir.exists():
            print("\n" + "="*80)
            print("ERROR: DAS combined TFRecord directory not found")
            print("="*80)
            print(f"  Directory: {tfrecord_dir.resolve()}")
            print("\n  Run: python das_preprocessing_combined.py")
            print("  Or ensure DAS .mat files (S*.mat) exist in:", args.das_original_dir)
            print("="*80)
            sys.exit(1)
        if not tfrecord_files:
            print("\n" + "="*80)
            print("ERROR: No TFRecord files in DAS combined directory")
            print("="*80)
            print(f"  Directory: {tfrecord_dir.resolve()}")
            print("  Run: python das_preprocessing_combined.py")
            print("="*80)
            sys.exit(1)
    
    # ========================================================================
    # RUN PREPROCESSING FIRST
    # ========================================================================
    print("="*80)
    print("RUNNING PREPROCESSING PIPELINE")
    print("="*80)
    
    das_original_dir = getattr(args, 'das_original_dir', 'Data/Das/4004271')
    das_audio_dir = getattr(args, 'das_audio_dir', 'Data/Das/4004271/stimuli/stimuli')
    
    # 1. Run Das preprocessing (only when not COMBINED_DAS or when using MWF/DASPREPROCESS and need extra setup)
    if args.das_preprocessing_type != "COMBINED_DAS":
        print("\n[1/3] Running Das preprocessing...")
        das_preprocessing_cmd = [
            sys.executable, "das_preprocessing_combined.py",
            "--data_dir", das_original_dir,
            "--output_dir", args.das_data_dir,
            "--audio_dir", das_audio_dir,
        ]
        print(f"Command: {' '.join(das_preprocessing_cmd)}")
        result = subprocess.run(das_preprocessing_cmd, capture_output=False, cwd=str(Path(__file__).resolve().parent))
        if result.returncode != 0:
            print(f"⚠️  Warning: Das preprocessing returned exit code {result.returncode}")
            print("  Continuing anyway...")
        else:
            print("✓ Das preprocessing completed")
    else:
        print("\n[1/3] COMBINED_DAS: Using existing DAS TFRecords from", args.das_data_dir)
    
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
    
    fulsang_raw = Path(getattr(args, 'fulsang_raw_dir', 'Data/Fulsang'))
    fulsang_eeg_dir = fulsang_raw / "EEG" if (fulsang_raw / "EEG").exists() else fulsang_raw
    fulsang_audio_dir = getattr(args, 'fulsang_audio_dir', 'Data/Fulsang/AUDIO')
    fulsang_mwf_cmd = [
        sys.executable, "mwf_artifact_removal.py",
        "--dataset", "fuglsang",
        "--fuglsang_eeg_dir", str(fulsang_eeg_dir),
        "--fuglsang_audio_dir", str(fulsang_audio_dir)
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
    print(f"  Regularization: {OPTIMAL_CONFIG['regularization']} (default: 0.01)")
    print(f"  Window size: {OPTIMAL_CONFIG['window_size']} samples (8 s @ 64 Hz)")
    print(f"  Batch size: {OPTIMAL_CONFIG['batch_size']} (default: 6)")
    print(f"\nCommand-line Arguments:")
    print(f"  --target_sampling_rate: {getattr(args, 'target_sampling_rate', PAPER_FS_HZ)}")
    print(f"  --window_sec: {getattr(args, 'window_sec', 'not set')}")
    print(f"  --window_size: {args.window_size}")
    print(f"  --cca_dims: {args.cca_dims}")
    print(f"  --regularization: {args.regularization}")
    print(f"  --batch_size: {args.batch_size}")
    
    # Warn if using non-optimal defaults
    effective_window = int(getattr(args, 'target_sampling_rate', PAPER_FS_HZ) * getattr(args, 'window_sec', 0)) if getattr(args, 'window_sec', None) is not None else args.window_size
    if effective_window == 256 or args.window_size == 256:
        print(f"\n  ⚠️  WARNING: Using window_size=256 (suboptimal). Recommended: 512 or 1024 for 4s/8s")
    if args.cca_dims == 12:
        print(f"  ⚠️  WARNING: Using cca_dims=12 (suboptimal). Recommended: 16")
    if args.regularization == 0.08:
        print(f"  ⚠️  WARNING: Using regularization=0.08 (suboptimal). Recommended: 0.01–0.02")
    
    # Accuracy vs decision window length (1s–30s): run sweep only when --window_seconds is set
    if getattr(args, 'window_seconds', None):
        try:
            window_sec_list = [float(x.strip()) for x in args.window_seconds.split(',')]
        except (ValueError, AttributeError):
            window_sec_list = WINDOW_SWEEP_SECONDS
        print("\n" + "="*80)
        print(f"WINDOW SWEEP: {', '.join(str(s) + 's' for s in window_sec_list)} @ {getattr(args, 'target_sampling_rate', PAPER_FS_HZ)} Hz")
        print("="*80)
        sweep_results = run_window_sweep(args, window_seconds_list=window_sec_list)
        save_window_sweep_results(sweep_results, Path(args.output_dir))
        print("\n" + "="*80)
        print("WINDOW SWEEP SUMMARY")
        print("="*80)
        for r in sweep_results:
            sec = r['window_sec']
            sec_str = f"{sec:.2f}" if isinstance(sec, float) else str(sec)
            if r.get('test_accuracy') is not None:
                va = r.get('val_accuracy')
                va_str = f"{va:.4f}" if va is not None else "N/A"
                print(f"  {sec_str}s: Test Acc = {r['test_accuracy']:.4f}, Val Acc = {va_str}, ROC-AUC = {r['roc_auc']:.4f}")
            else:
                print(f"  {sec_str}s: Failed ({r.get('error', 'unknown')})")
        print("="*80)
        print(f"\n✓ Sweep complete. Results in {args.output_dir}")
        return
    
    # Single run at 8s (default when no --window_seconds)
    # Create combined dataset
    print("\n" + "="*80)
    print("LOADING COMBINED DATASET")
    print("="*80)
    # Resolve window size: --window_sec overrides --window_size (default: 8 s @ 64 Hz)
    fs = getattr(args, 'target_sampling_rate', PAPER_FS_HZ)
    window_size_samples = int(fs * args.window_sec) if getattr(args, 'window_sec', None) is not None else args.window_size
    if getattr(args, 'window_sec', None) is not None:
        print(f"  Window: {args.window_sec}s @ {fs} Hz = {window_size_samples} samples")
    combined_dataset = CombinedDataset(**_combined_dataset_kwargs(args, window_size_samples))
    
    # Fulsang label reminder
    fulsang_raw = getattr(args, 'fulsang_raw_dir', None)
    if fulsang_raw and ('dummy' in str(fulsang_raw).lower() or 'temp' in str(fulsang_raw).lower()):
        print("\n  ⚠ For best accuracy, provide Fulsang raw EEG so true attention labels are used.")
        print("    Expected: Data/Fulsang/EEG/EEG.zip or Data/Fulsang/EEG/EEG (extracted).")
    
    # Create TensorFlow dataset wrapper
    print("\n" + "="*80)
    print("CREATING TENSORFLOW DATASET")
    print("="*80)
    tf_dataset_wrapper = CombinedCCADataset(
        combined_dataset,
        max_windows=getattr(args, 'max_windows', None),
        eeg_lag_taps=getattr(args, 'eeg_lag_taps', 12)
    )
    if getattr(args, 'dataset_subset', 'full') != 'full':
        subset = getattr(args, 'dataset_subset', 'full')
        before = len(tf_dataset_wrapper.window_indices)
        tf_dataset_wrapper.window_indices = _filter_window_indices_by_subset(tf_dataset_wrapper.window_indices, subset)
        after = len(tf_dataset_wrapper.window_indices)
        print(f"Dataset subset filter: {subset} ({before} -> {after} windows)")
    
    train_dataset, val_dataset, test_dataset, test_indices, tf_dataset_wrapper = split_dataset(
        tf_dataset_wrapper,
        train_ratio=0.70,
        val_ratio=0.25,
        max_train_windows=getattr(args, 'max_train_windows', None)
    )
    
    # Create CCA model
    print("\n" + "="*80)
    print("INITIALIZING OPTIMAL COMBINED CCA MODEL")
    print("="*80)
    if getattr(tf_dataset_wrapper, 'use_time_resolved', True):
        eeg_flat_dim = getattr(tf_dataset_wrapper, 'eeg_time_point_dim', tf_dataset_wrapper.n_channels)
        envelope_dim = tf_dataset_wrapper.num_lags
    else:
        envelope_dim = tf_dataset_wrapper.envelope_dim
        eeg_flat_dim = tf_dataset_wrapper.window_size * tf_dataset_wrapper.n_channels
    OPTIMAL_CONFIG['window_size'] = window_size_samples
    cca_model = OptimalCombinedCCAModel(
        cca_dims=OPTIMAL_CONFIG['cca_dims'],
        regularization=OPTIMAL_CONFIG['regularization'],
        window_size=window_size_samples,
        envelope_dim=envelope_dim,
        eeg_flat_dim=eeg_flat_dim,
        use_first_cca_component_only=getattr(args, 'use_first_cca_component_only', False),
        train_cca_on_both_envelopes=getattr(args, 'cca_both_envelopes', False)
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
    
    # Trial-level aggregation (majority vote per trial)
    trial_metrics = compute_trial_level_metrics(
        results['predictions'], results['targets'], test_indices, tf_dataset_wrapper.window_indices
    )
    if trial_metrics:
        detailed_metrics['advanced_metrics']['trial_accuracy'] = trial_metrics['trial_accuracy']
        detailed_metrics['advanced_metrics']['trial_balanced_accuracy'] = trial_metrics['trial_balanced_accuracy']
    
    # Paper-style: mean accuracy ± SEM across subjects
    per_subj = compute_per_subject_accuracy(
        results['predictions'], results['targets'], test_indices, tf_dataset_wrapper.window_indices
    )
    if per_subj:
        detailed_metrics['advanced_metrics']['mean_accuracy_across_subjects'] = per_subj['mean_accuracy']
        detailed_metrics['advanced_metrics']['sem_accuracy_across_subjects'] = per_subj['sem_accuracy']
        detailed_metrics['advanced_metrics']['per_subject_accuracy'] = per_subj['per_subject_accuracy']
        detailed_metrics['advanced_metrics']['n_test_subjects'] = per_subj['n_subjects']
    
    # Generate report
    generate_comprehensive_report(results, detailed_metrics, val_accuracy, Path(args.output_dir), per_subject_metrics=per_subj)
    
    print(f"\n✓ Combined CCA Training Complete")
    print(f"  Test Accuracy: {results['accuracy']:.4f}")
    if per_subj:
        print(f"  Mean accuracy ± SEM (across {per_subj['n_subjects']} test subjects): {per_subj['mean_accuracy']:.4f} ± {per_subj['sem_accuracy']:.4f}")
    print(f"  Test Balanced Accuracy: {detailed_metrics['advanced_metrics']['balanced_accuracy']:.4f}")
    print(f"  Validation Accuracy: {val_accuracy:.4f}")
    print(f"  ROC-AUC: {detailed_metrics['advanced_metrics']['roc_auc_score']:.4f}")
    if trial_metrics:
        print(f"  Trial Accuracy: {trial_metrics['trial_accuracy']:.4f}")
        print(f"  Trial Balanced Accuracy: {trial_metrics['trial_balanced_accuracy']:.4f}")
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
