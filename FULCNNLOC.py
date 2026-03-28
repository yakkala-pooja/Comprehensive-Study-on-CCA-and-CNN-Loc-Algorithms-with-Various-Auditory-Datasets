#!/usr/bin/env python3
"""
FULCNNLOC - CNN-LOC Algorithm for Fulsang Dataset

This script implements CNN-LOC (Convolutional Neural Network - Localization) for the 
Fulsang dataset using TFRecord files created by FULPRE.py preprocessing.

Architecture is based on CombinedCNNLOC.py with Fulsang-specific adaptations.
"""

import os
import sys
import re
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
import tensorflow as tf
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import json
import pickle
from datetime import datetime
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           precision_recall_fscore_support, roc_auc_score, roc_curve,
                           precision_recall_curve, average_precision_score,
                           matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
                           f1_score)
import warnings
warnings.filterwarnings('ignore')


def seed_everything(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================================
# CNN-LOC Architecture Components (from CombinedCNNLOC.py)
# ============================================================================

class SpatialTemporalAttention(nn.Module):
    """Channel attention for EEG data."""
    
    def __init__(self, channels: int, reduction: int = 8):
        super(SpatialTemporalAttention, self).__init__()
        
        self.channels = channels
        self.reduction = max(1, reduction)
        self.reduced_channels = max(1, channels // self.reduction)
        
        # Channel attention only
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, self.reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.reduced_channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention: learn which channels are important
        channel_att = self.channel_attention(x)
        # Add residual connection to prevent information loss
        out = x * channel_att + x * 0.1  # 10% residual to prevent complete suppression
        return out


class ResidualBlock(nn.Module):
    """Residual block with attention.
    
    Uses GroupNorm instead of BatchNorm for better cross-subject generalization.
    GroupNorm is independent of batch statistics and works better with small batches
    and subject-independent splits.
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, num_groups: int = 8):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        # Use GroupNorm instead of BatchNorm for cross-subject generalization
        self.norm1 = nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels)
        
        # Shortcut for residual connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels)
            )
        
        self.attention = SpatialTemporalAttention(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.attention(out)
        
        out += residual
        out = self.relu(out)
        
        return out


class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale features using different kernel sizes.
    
    Uses GroupNorm instead of BatchNorm for better cross-subject generalization.
    """
    
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8):
        super(MultiScaleFeatureExtractor, self).__init__()
        
        # Two scales: 1x1 and 3x1
        self.conv1x1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        self.conv3x1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=(3, 1), padding=(1, 0))
        
        # Use GroupNorm instead of BatchNorm
        self.norm = nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        feat1 = self.conv1x1(x)
        feat3 = self.conv3x1(x)
        
        # Concatenate
        out = torch.cat([feat1, feat3], dim=1)
        out = self.relu(self.norm(out))
        
        return out


class AdaptivePooling(nn.Module):
    """Adaptive pooling for variable input sizes."""
    
    def __init__(self, output_size: int = 1):
        super(AdaptivePooling, self).__init__()
        self.output_size = output_size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size)
    
    def forward(self, x):
        return self.adaptive_pool(x)


class CNNLOCBackbone(nn.Module):
    """Backbone network: attention, residual blocks, multi-scale features.
    
    Enhanced CNN-LOC (CNN-LOC++): re-balanced for (time=64/128, freq=8).
    Original CNN-LOC used (32/64 time, 5 freq); we restore temporal and spatial
    pooling so capacity matches resolution and features do not collapse.
    """
    
    def __init__(self, input_channels: int = 66, input_time: int = 32, input_freq: int = 5):
        super(CNNLOCBackbone, self).__init__()
        
        self.input_channels = input_channels
        self.input_time = input_time
        self.input_freq = input_freq
        
        print(f"Building CNN-LOC backbone (Enhanced): channels={input_channels}, time={input_time}, freq={input_freq}")
        
        # Initial multi-scale features (using GroupNorm for cross-subject generalization)
        self.initial_features = MultiScaleFeatureExtractor(input_channels, 32, num_groups=8)
        
        # Temporal blocks: two pools so time 64->32->16 or 128->64->32
        self.temporal_block1 = ResidualBlock(32, 32, stride=1, num_groups=8)
        self.temporal_pool1 = nn.MaxPool2d((2, 1), (2, 1))  # 64->32 or 128->64
        
        self.temporal_block2 = ResidualBlock(32, 64, stride=1, num_groups=8)
        self.temporal_pool2 = nn.MaxPool2d((2, 1), (2, 1))  # 32->16 or 64->32
        
        # Spatial blocks: two pools so freq 8->4->2
        self.spatial_block1 = ResidualBlock(64, 64, stride=1, num_groups=8)
        self.spatial_pool1 = nn.MaxPool2d((1, 2), (1, 2))  # 8->4 (or 5->2 for legacy 5-band)
        
        self.spatial_block2 = ResidualBlock(64, 128, stride=1, num_groups=8)
        self.spatial_pool2 = nn.MaxPool2d((1, 2), (1, 2))  # 4->2
        
        # Global attention
        self.global_attention = SpatialTemporalAttention(128)
        
        # Adaptive pooling
        self.adaptive_pooling = AdaptivePooling(output_size=1)
        
        # Calculate output size
        self._calculate_output_size()
    
    def _calculate_output_size(self):
        """Figure out output size by running a dummy input."""
        dummy_input = torch.randn(1, self.input_channels, self.input_time, self.input_freq, device='cpu')
        self.eval()
        with torch.no_grad():
            x = self.forward(dummy_input.cpu())
            self.output_size = x.numel()
    
    def forward(self, x):
        """Forward pass.
        
        Expected input shape: (batch, channels, time, freq) = (batch, 66, 32, 5)
        """
        # Verify input shape (only on first forward pass for debugging)
        if not hasattr(self, '_shape_verified'):
            expected_shape = (x.size(1), x.size(2), x.size(3))
            if expected_shape != (self.input_channels, self.input_time, self.input_freq):
                print(f"⚠ WARNING: Input shape mismatch! Expected (C, T, F)={self.input_channels, self.input_time, self.input_freq}, got {expected_shape}")
            self._shape_verified = True
        
        # Multi-scale features
        x = self.initial_features(x)
        
        # Temporal processing
        x = self.temporal_block1(x)
        x = self.temporal_pool1(x)  # 64->32 or 128->64
        
        x = self.temporal_block2(x)
        x = self.temporal_pool2(x)  # 32->16 or 64->32
        
        # Spatial processing
        x = self.spatial_block1(x)
        x = self.spatial_pool1(x)  # 8->4 (or 5->2)
        
        x = self.spatial_block2(x)
        x = self.spatial_pool2(x)  # 4->2
        
        # Attention (apply before final pooling to preserve more information)
        x = self.global_attention(x)
        
        # Pool and flatten
        x = self.adaptive_pooling(x)
        x = x.view(x.size(0), -1)
        
        return x


class CNNLOCModel(nn.Module):
    """Full CNN-LOC model: backbone + classifier for EEG attention decoding."""
    
    def __init__(self, input_channels: int = 66, input_time: int = 32, input_freq: int = 5,
                 num_classes: int = 2, dropout_rate: float = 0.3):
        super(CNNLOCModel, self).__init__()
        
        # Create backbone
        self.backbone = CNNLOCBackbone(input_channels, input_time, input_freq)
        
        # Classifier (using LayerNorm instead of BatchNorm1d for cross-subject generalization)
        # Even simpler classifier with higher dropout for better generalization
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),  # Increased dropout for regularization
            nn.Linear(self.backbone.output_size, 32),  # Further reduced capacity
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # High dropout before output
            nn.Linear(32, num_classes)  # Direct to output
        )
        
        self._initialize_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model created with {n_params:,} parameters")
    
    def _initialize_weights(self):
        """Initialize model weights with proper scaling."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Use Xavier/Kaiming for better initialization (especially for final layer)
                if m.out_features == 2:  # Final classification layer
                    # CRITICAL FIX: gain=0.1 was too small, causing near-zero logits
                    # Use gain=1.0 to produce reasonable initial logits (not too small, not too large)
                    # This allows the model to learn without starting collapsed
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                    # Small random bias to break symmetry and prevent immediate collapse
                    nn.init.uniform_(m.bias, -0.1, 0.1)  # Small random bias instead of zero
                else:
                    # Hidden layers: use Xavier for better gradient flow
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the model."""
        features = self.backbone(x)
        output = self.classifier(features)
        return output


# ============================================================================
# Dataset and Training
# ============================================================================

class FULCNNLOCDataset(Dataset):
    """
    PyTorch Dataset for Fulsang TFRecord files (from FULPRE.py).
    Loads TFRecord files and converts to time-frequency representation for CNN-LOC.
    
    Preserves trial boundaries - windows are created within trials only.
    """
    
    def __init__(self, tfrecord_dir: str, mode: str = 'train', 
                 window_size: int = 512, overlap: float = 0.5,
                 transform_eeg: bool = True, allow_cross_trial: bool = False,
                 subject_wise_normalization: bool = True,
                 global_normalization: bool = False,  # NEW: Use global stats instead of per-subject
                 augment: bool = False, cache_tf_max: int = 0):
        self.tfrecord_dir = Path(tfrecord_dir)
        self.mode = mode
        self.window_size = window_size
        self.overlap = overlap
        self.transform_eeg = transform_eeg
        self.allow_cross_trial = allow_cross_trial  # If False, windows stay within trials
        self.subject_wise_normalization = subject_wise_normalization
        self.global_normalization = global_normalization  # NEW: Use global stats across all training subjects
        # CRITICAL: Mutual exclusion - only one normalization mode
        if subject_wise_normalization and global_normalization:
            raise ValueError(
                "subject_wise_normalization and global_normalization are mutually exclusive. "
                "Use exactly one of: subject_wise_normalization=True (per-subject z-score) or "
                "global_normalization=True (global z-score across training subjects)."
            )
        norm_mode = "global" if global_normalization else ("subject-wise" if subject_wise_normalization else "window-wise")
        self.augment = augment and (mode == 'train')  # Only augment during training
        # Optional LRU-style cache for time-frequency transform (avoids recomputing in __getitem__)
        self._tf_cache_max = cache_tf_max if not (augment and mode == 'train') else 0
        self._tf_cache = OrderedDict() if self._tf_cache_max > 0 else None
        
        # Fulsang dataset parameters
        self.sampling_rate = 64  # Hz
        self.n_channels = 66  # EEG channels
        self.trial_length = 3200  # samples per trial (50 seconds at 64 Hz)
        
        # Warn about very short windows (1s is too short for AAD)
        window_seconds = window_size / self.sampling_rate
        if window_seconds < 4.0:
            print(f"\n⚠ WARNING: Window size {window_seconds:.1f}s is very short for AAD (attention decoding).")
            print(f"  AAD relies on slow cortical tracking (delta/theta timescales).")
            print(f"  Recommended minimum: 8s. Best performance typically at 16-30s.")
            print(f"  Short windows may lead to biased predictions and poor generalization.")
        
        # Load trials (preserving boundaries)
        self.trials = self._load_trials()
        
        # Compute normalization statistics if enabled
        # NOTE: For subject-independent splits, stats should be computed ONLY on training subjects
        # This is done in split_dataset() to prevent data leakage
        # Here we initialize empty - stats will be set after splitting
        if self.subject_wise_normalization or self.global_normalization:
            # For now, compute on all subjects (backward compatibility)
            # But this will be recomputed with only training subjects in split_dataset()
            self.subject_stats = {}  # Per-subject stats (for subject_wise_normalization)
            self.global_stats = {}  # Global stats (for global_normalization): {'mean': array, 'std': array}
            self._needs_subject_stats = True  # Flag that stats need to be computed after split
        else:
            self.subject_stats = {}
            self.global_stats = {}
            self._needs_subject_stats = False
        
        # Create windows from trials
        self.windows = self._create_windows_from_trials()
        
        print(f"\nFULCNNLOCDataset initialized:")
        print(f"  Mode: {mode}")
        print(f"  Normalization: {norm_mode}")
        print(f"  Total trials: {len(self.trials)}")
        print(f"  Total windows: {len(self.windows)}")
        print(f"  Window size: {self.window_size} samples ({self.window_size/self.sampling_rate:.1f}s)")
        print(f"  Overlap: {overlap}")
        print(f"  Allow cross-trial windows: {allow_cross_trial}")
        print(f"  Sampling rate: {self.sampling_rate} Hz")
        print(f"  Channels: {self.n_channels}")
        print(f"  Transform EEG: {transform_eeg}")
    
    def _load_trials(self) -> List[Dict]:
        """
        Load trials from TFRecord files, preserving trial boundaries.
        
        Returns:
            List of trial dictionaries, each containing:
            - eeg: (trial_length, n_channels) array
            - label: int (trial-level label)
            - subject_id: str
            - trial_idx: int
            - file: str
            - metadata: dict with additional fields
        """
        tfrecord_files = sorted(
            list(self.tfrecord_dir.glob("*.tfrecords")) + list(self.tfrecord_dir.glob("*.tfrecord"))
        )
        
        if not tfrecord_files:
            # Check if directory exists
            if not self.tfrecord_dir.exists():
                raise ValueError(
                    f"TFRecord directory does not exist: {self.tfrecord_dir}\n"
                    f"Please run FULPRE.py first to generate TFRecord files.\n"
                    f"Example: python FULPRE.py --data_dir Data/Fulsang --output_dir fulsang_preprocessed"
                )
            else:
                raise ValueError(
                    f"No TFRecord files (*.tfrecords) found in {self.tfrecord_dir}\n"
                    f"Please run FULPRE.py first to generate TFRecord files.\n"
                    f"Example: python FULPRE.py --data_dir Data/Fulsang --output_dir fulsang_preprocessed"
                )
        
        print(f"Loading trials from {len(tfrecord_files)} TFRecord files...")
        
        trials = []
        
        for tfrecord_file in tqdm(tfrecord_files, desc="Loading TFRecords"):
            try:
                dataset = tf.data.TFRecordDataset(str(tfrecord_file))
                for record in dataset:
                    try:
                        example = tf.train.Example.FromString(record.numpy())
                        features = example.features.feature
                        
                        # Check required features
                        if 'eeg' not in features or 'attention_label' not in features:
                            continue
                        
                        # Extract EEG data (trial-based: each record = one trial)
                        eeg_values = features['eeg'].float_list.value
                        if not eeg_values or len(eeg_values) == 0:
                            continue
                        
                        # Get n_channels and n_samples from TFRecord
                        n_channels = self.n_channels  # Default
                        n_samples = self.trial_length  # Default
                        
                        if 'n_channels' in features:
                            n_channels_list = features['n_channels'].int64_list.value
                            if n_channels_list and len(n_channels_list) > 0:
                                n_channels = int(n_channels_list[0])
                        
                        if 'n_samples' in features:
                            n_samples_list = features['n_samples'].int64_list.value
                            if n_samples_list and len(n_samples_list) > 0:
                                n_samples = int(n_samples_list[0])
                        
                        # Validate expected size (trial-based only)
                        expected_size = n_samples * n_channels
                        if len(eeg_values) != expected_size:
                            print(f"  WARNING: Skipping record with unexpected size: {len(eeg_values)} != {expected_size}")
                            continue
                        
                        # Explicit check: one record = one trial
                        if n_samples != self.trial_length:
                            print(f"  WARNING: n_samples={n_samples} differs from expected trial_length={self.trial_length}")
                            # Continue anyway, but warn that windowing assumptions may break
                        
                        # Reshape to (n_samples, n_channels)
                        eeg_data = np.array(eeg_values, dtype=np.float32).reshape(n_samples, n_channels)
                        
                        # Validate shape
                        if eeg_data.shape[0] != n_samples or eeg_data.shape[1] != n_channels:
                            print(f"  WARNING: Skipping record with shape mismatch: {eeg_data.shape} != ({n_samples}, {n_channels})")
                            continue
                        
                        # Extract attention label (trial-level)
                        label = 0
                        if 'attention_label' in features:
                            label_list = features['attention_label'].int64_list.value
                            if label_list and len(label_list) > 0:
                                label = int(label_list[0])
                        
                        # Extract metadata
                        subject_id = "unknown"
                        if 'subject_id' in features:
                            subject_list = features['subject_id'].bytes_list.value
                            if subject_list and len(subject_list) > 0:
                                try:
                                    subject_id = subject_list[0].decode('utf-8')
                                except:
                                    pass
                        
                        # Hard fail if subject_id is missing (required for subject-level splitting)
                        if subject_id == "unknown":
                            raise ValueError(
                                f"TFRecord record missing subject_id in file {tfrecord_file.name}. "
                                f"Re-run preprocessing; subject splits would be invalid."
                            )
                        
                        trial_idx = 0
                        if 'trial_idx' in features:
                            trial_list = features['trial_idx'].int64_list.value
                            if trial_list and len(trial_list) > 0:
                                trial_idx = int(trial_list[0])
                        
                        # Extract additional metadata fields
                        metadata = {
                            'subject_id': subject_id,
                            'trial_idx': trial_idx,
                            'file': tfrecord_file.name
                        }
                        
                        # Add optional expinfo fields if available
                        for field in ['attend_lr', 'acoustic_condition', 'n_speakers', 'attend_mf_raw', 'trigger']:
                            if field in features:
                                field_list = features[field].int64_list.value
                                if field_list and len(field_list) > 0:
                                    metadata[field] = int(field_list[0])
                        
                        trials.append({
                            'eeg': eeg_data,
                            'label': label,
                            'subject_id': subject_id,
                            'trial_idx': trial_idx,
                            'file': tfrecord_file.name,
                            'metadata': metadata
                        })
                        
                    except Exception as record_error:
                        print(f"  WARNING: Error reading record from {tfrecord_file.name}: {record_error}")
                        continue
                        
            except Exception as e:
                print(f"ERROR: Error reading {tfrecord_file}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not trials:
            raise ValueError("No valid trials found in TFRecord files")
        
        # Validate labels are binary (0/1)
        all_labels = [t['label'] for t in trials]
        unique_labels = set(all_labels)
        if not unique_labels.issubset({0, 1}):
            raise ValueError(
                f"Invalid labels found: {unique_labels}. Expected only 0 and 1. "
                f"This may indicate a problem with label encoding in TFRecord files."
            )
        
        # Print summary
        print(f"\nLoaded {len(trials)} trials")
        label_counts = np.bincount(all_labels, minlength=2)
        print(f"Label distribution: {dict(enumerate(label_counts))}")
        
        if len(label_counts) > 2 or any(count > 0 for count in label_counts[2:]):
            print(f"WARNING: Found labels beyond 0/1: {unique_labels}")
        
        # Count by subject
        subject_counts = {}
        for trial in trials:
            subj = trial['subject_id']
            subject_counts[subj] = subject_counts.get(subj, 0) + 1
        print(f"Trials per subject: {len(subject_counts)} subjects, {min(subject_counts.values())}-{max(subject_counts.values())} trials each")
        
        # Report trial length statistics
        if trials:
            trial_lengths = [len(t['eeg']) for t in trials]
            min_trial_len = min(trial_lengths)
            max_trial_len = max(trial_lengths)
            if min_trial_len != max_trial_len:
                print(f"  WARNING: Trial lengths vary: min={min_trial_len}, max={max_trial_len}, expected={self.trial_length}")
            elif min_trial_len != self.trial_length:
                print(f"  WARNING: Trial length={min_trial_len} differs from expected={self.trial_length}")
        
        return trials
    
    def _compute_subject_stats(self, train_subject_ids: Optional[set] = None) -> Dict[str, Dict]:
        """
        Compute per-subject, per-channel normalization statistics.
        
        This is critical for cross-subject generalization - it removes subject-specific
        baseline shifts and power spectrum differences.
        
        **CRITICAL**: Only computes stats for training subjects to prevent data leakage.
        If train_subject_ids is None, computes for all subjects (for backward compatibility,
        but this should be avoided for subject-independent splits).
        
        Args:
            train_subject_ids: Optional set of subject IDs to include. If None, uses all subjects.
        
        Returns:
            Dictionary mapping subject_id -> {'mean': (n_channels,), 'std': (n_channels,)}
        """
        subject_data = {}
        
        # Collect trials per subject, filtering by train_subject_ids if provided
        for trial in self.trials:
            subj = trial['subject_id']
            # Only include subjects in train_subject_ids if provided
            if train_subject_ids is not None and subj not in train_subject_ids:
                continue
            if subj not in subject_data:
                subject_data[subj] = []
            subject_data[subj].append(trial['eeg'])
        
        # Compute per-subject, per-channel statistics
        subject_stats = {}
        for subj, trial_arrays in subject_data.items():
            # Concatenate all trials for this subject
            all_data = np.concatenate(trial_arrays, axis=0)  # (total_samples, n_channels)
            
            # Compute mean and std per channel
            mean = np.mean(all_data, axis=0, keepdims=False)  # (n_channels,)
            std = np.std(all_data, axis=0, keepdims=False)  # (n_channels,)
            std = np.where(std == 0, 1.0, std)  # Avoid division by zero
            
            subject_stats[subj] = {
                'mean': mean.astype(np.float32),
                'std': std.astype(np.float32)
            }
        
        if train_subject_ids is not None:
            print(f"\nComputed subject-wise normalization stats for {len(subject_stats)} training subjects")
            print(f"  (Excluded test/val subjects to prevent data leakage)")
        else:
            print(f"\nComputed subject-wise normalization stats for {len(subject_stats)} subjects")
            print(f"  ⚠ WARNING: Stats computed on ALL subjects - may cause data leakage!")
        return subject_stats
    
    def _compute_global_stats(self, train_subject_ids: Optional[set] = None) -> Dict[str, np.ndarray]:
        """
        Compute global normalization statistics across all training subjects.
        
        This preserves task-relevant signal while removing subject-specific baselines.
        Unlike per-subject normalization, this keeps relative power differences that
        are consistent across subjects (e.g., attention-related alpha power changes).
        
        **CRITICAL**: Only computes stats for training subjects to prevent data leakage.
        
        Args:
            train_subject_ids: Optional set of subject IDs to include. If None, uses all subjects.
        
        Returns:
            Dictionary with 'mean' and 'std' arrays of shape (n_channels,)
        """
        all_data = []
        
        # Collect all trials from training subjects
        for trial in self.trials:
            subj = trial['subject_id']
            # Only include subjects in train_subject_ids if provided
            if train_subject_ids is not None and subj not in train_subject_ids:
                continue
            all_data.append(trial['eeg'])
        
        if not all_data:
            raise ValueError("No training data found for global stats computation!")
        
        # Concatenate all trials from all training subjects
        all_data_concat = np.concatenate(all_data, axis=0)  # (total_samples, n_channels)
        
        # Compute global mean and std per channel (across all training subjects)
        mean = np.mean(all_data_concat, axis=0, keepdims=False)  # (n_channels,)
        std = np.std(all_data_concat, axis=0, keepdims=False)  # (n_channels,)
        std = np.where(std == 0, 1.0, std)  # Avoid division by zero
        
        if train_subject_ids is not None:
            print(f"\nComputed global normalization stats from {len(train_subject_ids)} training subjects")
            print(f"  (Excluded test/val subjects to prevent data leakage)")
        else:
            print(f"\nComputed global normalization stats from all subjects")
            print(f"  ⚠ WARNING: Stats computed on ALL subjects - may cause data leakage!")
        
        return {
            'mean': mean.astype(np.float32),
            'std': std.astype(np.float32)
        }
    
    def _create_windows_from_trials(self) -> List[Dict]:
        """
        Create sliding windows from trials, preserving trial boundaries.
        
        Returns:
            List of window dictionaries, each containing:
            - trial_idx: index into self.trials
            - start_sample: start sample within trial
            - end_sample: end sample within trial
            - label: window label (same as trial label, since labels are constant within trials)
            - metadata: trial metadata
        """
        # Guard against overlap >= 1.0 which would cause step_size = 0
        if self.overlap >= 1.0:
            raise ValueError(f"overlap must be < 1.0, got {self.overlap}")
        step_size = max(1, int(self.window_size * (1 - self.overlap)))
        windows = []
        
        for trial_idx, trial in enumerate(self.trials):
            eeg = trial['eeg']
            trial_length = eeg.shape[0]
            label = trial['label']
            
            # Skip trials that are too short
            if trial_length < self.window_size:
                if not self.allow_cross_trial:
                    continue  # Skip short trials if not allowing cross-trial windows
                # If allowing cross-trial, we'd need to handle this differently
                # For now, skip short trials
                continue
            
            # Create windows within this trial
            num_windows = (trial_length - self.window_size) // step_size + 1
            
            for win_idx in range(num_windows):
                start_sample = win_idx * step_size
                end_sample = start_sample + self.window_size
                
                if end_sample > trial_length:
                    break
                
                # Label is constant within trial, so use trial label directly
                # (no need for majority vote since all samples in trial have same label)
                windows.append({
                    'trial_idx': trial_idx,
                    'start_sample': start_sample,
                    'end_sample': end_sample,
                    'label': label,
                    'metadata': trial['metadata'].copy()
                })
        
        if not windows:
            # Get actual trial lengths for better error message
            if self.trials:
                actual_trial_lengths = [len(t['eeg']) for t in self.trials]
                min_trial_len = min(actual_trial_lengths)
                max_trial_len = max(actual_trial_lengths)
                trial_len_str = f"{min_trial_len}" if min_trial_len == max_trial_len else f"{min_trial_len}-{max_trial_len}"
            else:
                trial_len_str = "unknown"
            
            raise ValueError(
                f"No windows created! Check window_size ({self.window_size}) vs actual trial lengths ({trial_len_str}). "
                f"Expected trial_length: {self.trial_length}. Total trials: {len(self.trials)}"
            )
        
        # Print summary
        window_labels = [w['label'] for w in windows]
        window_label_dist = np.bincount(window_labels, minlength=2)
        print(f"\nWindow creation summary:")
        print(f"  Total windows: {len(windows)}")
        print(f"  Window label distribution: {dict(enumerate(window_label_dist))}")
        
        # Count windows per trial
        windows_per_trial = {}
        for w in windows:
            tidx = w['trial_idx']
            windows_per_trial[tidx] = windows_per_trial.get(tidx, 0) + 1
        print(f"  Windows per trial: {min(windows_per_trial.values())}-{max(windows_per_trial.values())} (mean: {np.mean(list(windows_per_trial.values())):.1f})")
        
        return windows
    
    def _transform_eeg(self, eeg_window: np.ndarray) -> np.ndarray:
        """
        Transform EEG window to time-frequency representation using proper EEG frequency bands.
        
        Uses scipy.signal.spectrogram to compute power in standard EEG bands:
        - Delta (1-4 Hz)
        - Theta (4-8 Hz)
        - Alpha (8-13 Hz)
        - Beta (13-30 Hz)
        - Gamma (30-40 Hz)
        
        This provides a physiologically meaningful representation that should generalize
        better across subjects than arbitrary FFT bin divisions.
        
        Output shape: (n_channels, time_frames, n_freq_bands) where n_freq_bands=5.
        """
        from scipy import signal
        
        n_samples, n_channels = eeg_window.shape
        sampling_rate = 64.0  # Fulsang dataset sampling rate
        
        # Enhanced EEG frequency bands (Hz) - split into sub-bands for better resolution
        # More granular bands capture finer attention-relevant frequency structure
        freq_bands = [
            (1, 4),      # Delta
            (4, 8),      # Theta
            (8, 10),     # Alpha-low (attention-relevant)
            (10, 13),    # Alpha-high (attention-relevant)
            (13, 18),    # Beta-low
            (18, 25),    # Beta-mid (attention-relevant)
            (25, 30),    # Beta-high
            (30, 40)     # Gamma
        ]
        n_freq_bands = len(freq_bands)
        
        # Use more time frames for better temporal resolution
        # Increased from 32/64 to 64/128 to capture finer temporal dynamics
        window_seconds = n_samples / sampling_rate
        time_frames = 128 if window_seconds >= 8.0 else 64
        
        # Initialize output: (channels, time_frames, freq_bands)
        eeg_tf = np.zeros((n_channels, time_frames, n_freq_bands), dtype=np.float32)
        
        # Process each channel separately
        for ch_idx in range(n_channels):
            channel_data = eeg_window[:, ch_idx]
            
            # Compute spectrogram with adaptive parameters
            # nperseg: segment length for STFT (at least 16, no larger than window length)
            nperseg = max(16, min(64, n_samples // 2))
            noverlap = max(8, nperseg // 2)  # 50% overlap, but at least 8 samples
            
            try:
                # Compute spectrogram
                freqs, times, Sxx = signal.spectrogram(
                    channel_data,
                    fs=sampling_rate,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    window='hann',
                    mode='magnitude'  # Use magnitude, we'll square it for power
                )
                
                # Convert to power
                Sxx_power = Sxx ** 2
                
                # Extract power in each frequency band for each time point
                n_time_spectrogram = Sxx_power.shape[1]
                
                # Interpolate or downsample to desired time_frames
                if n_time_spectrogram >= time_frames:
                    # Downsample: average adjacent time bins
                    step = n_time_spectrogram / time_frames
                    time_indices = [int(i * step) for i in range(time_frames)]
                    Sxx_power_resampled = Sxx_power[:, time_indices]
                else:
                    # Upsample: repeat or interpolate
                    Sxx_power_resampled = np.zeros((Sxx_power.shape[0], time_frames))
                    step = n_time_spectrogram / time_frames
                    for t_idx in range(time_frames):
                        src_idx = min(int(t_idx * step), n_time_spectrogram - 1)
                        Sxx_power_resampled[:, t_idx] = Sxx_power[:, src_idx]
                
                # Extract band power for each time frame
                for band_idx, (low_freq, high_freq) in enumerate(freq_bands):
                    # Clamp high frequency to Nyquist
                    if high_freq >= sampling_rate / 2:
                        high_freq = sampling_rate / 2 - 0.1
                    
                    # Find frequency indices in this band
                    freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
                    
                    if np.any(freq_mask):
                        # Average power across frequencies in this band
                        band_power = np.mean(Sxx_power_resampled[freq_mask, :], axis=0)
                    else:
                        # No frequencies in this band (shouldn't happen, but handle gracefully)
                        band_power = np.zeros(time_frames)
                    
                    # Store in output (sqrt-power to preserve relative differences)
                    # CRITICAL FIX: log() was destroying discriminative variance
                    # sqrt() preserves relative power differences while compressing dynamic range
                    eeg_tf[ch_idx, :, band_idx] = np.sqrt(band_power + 1e-8)
                    
            except Exception as e:
                # Fallback: if spectrogram fails, use simple FFT-based approach
                print(f"Warning: spectrogram failed for channel {ch_idx}, using fallback: {e}")
                # Simple fallback: use mean power in band (very coarse)
                for band_idx in range(n_freq_bands):
                    eeg_tf[ch_idx, :, band_idx] = 0.0  # Zero padding
        
        return eeg_tf.astype(np.float32)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        trial = self.trials[window['trial_idx']]
        
        # Extract window from trial
        eeg_window = trial['eeg'][window['start_sample']:window['end_sample']].copy()
        
        # Validate shape
        assert eeg_window.shape == (self.window_size, self.n_channels), \
            f"Window shape mismatch: {eeg_window.shape} != ({self.window_size}, {self.n_channels})"
        
        # Data augmentation (only during training, helps with cross-subject generalization)
        # More aggressive augmentation to force model to learn robust, subject-invariant features
        if self.augment:
            # 1. Random noise (increased to better simulate inter-subject variability)
            noise_std = np.random.uniform(0.015, 0.03)
            noise = np.random.normal(0, noise_std, eeg_window.shape).astype(np.float32)
            eeg_window = eeg_window + noise
            
            # 2. Random channel dropout (increased to 40% chance, drop 15% of channels)
            # Forces model to not rely on specific channels (subject-specific patterns)
            if np.random.rand() < 0.4:
                n_drop = max(1, int(self.n_channels * 0.15))  # Drop 15% of channels
                drop_channels = np.random.choice(self.n_channels, n_drop, replace=False)
                eeg_window[:, drop_channels] = 0
            
            # 3. Random time shift (always apply, larger shifts)
            shift = np.random.randint(-self.window_size // 10, self.window_size // 10 + 1)
            if shift != 0:
                eeg_window = np.roll(eeg_window, shift, axis=0)
            
            # 4. Random scaling (always apply, wider range to simulate gain variations)
            scale = np.random.uniform(0.85, 1.15)  # Wider range
            eeg_window = eeg_window * scale
            
            # 5. Random channel scaling (increased to 50% chance, wider range)
            if np.random.rand() < 0.5:
                channel_scales = np.random.uniform(0.9, 1.1, (1, self.n_channels))  # Wider range
                eeg_window = eeg_window * channel_scales
            
            # 6. Random frequency masking (30% chance) - simulates frequency-specific artifacts
            if np.random.rand() < 0.3:
                # Mask a random time segment (simulates artifacts)
                mask_length = int(self.window_size * np.random.uniform(0.05, 0.15))
                mask_start = np.random.randint(0, max(1, self.window_size - mask_length))
                eeg_window[mask_start:mask_start + mask_length, :] *= np.random.uniform(0.5, 0.8)
        
        # Preprocess (baseline correction and normalization)
        subject_id = trial['subject_id']
        
        if self.global_normalization and self.global_stats:
            # Global normalization: use statistics computed across all training subjects
            # This preserves task-relevant signal (e.g., attention-related power changes)
            # while removing subject-specific baselines
            stats = self.global_stats
            eeg_window = (eeg_window - stats['mean']) / (stats['std'] + 1e-8)
        elif self.subject_wise_normalization and subject_id in self.subject_stats:
            # Subject-wise normalization: per-subject, per-channel z-score
            # This directly attacks "subject identity leakage" but may remove task signal
            # if attention-related changes are relative to each subject's baseline
            stats = self.subject_stats[subject_id]
            eeg_window = (eeg_window - stats['mean']) / (stats['std'] + 1e-8)
        else:
            # Fallback: window-wise normalization (less effective for cross-subject)
            # 1. Remove DC offset (baseline correction)
            eeg_window = eeg_window - np.mean(eeg_window, axis=0, keepdims=True)
            
            # 2. Robust normalization: use median absolute deviation (MAD) instead of std
            # This is less sensitive to outliers and subject-specific artifacts
            mad_vals = np.median(np.abs(eeg_window), axis=0, keepdims=True)
            mad_vals = np.where(mad_vals == 0, 1.0, mad_vals)
            # Scale MAD to approximate std (MAD * 1.4826 ≈ std for normal distribution)
            eeg_window = eeg_window / (mad_vals * 1.4826 + 1e-8)
        
        # 3. Clip extreme values to reduce subject-specific artifacts
        # CRITICAL FIX: Clipping was removing discriminative signal
        # DISABLED - re-enable only if absolutely necessary for numerical stability
        # eeg_window = np.clip(eeg_window, -5.0, 5.0)
        
        # Transform to time-frequency representation (cache when not augmenting to avoid recompute)
        if self.transform_eeg:
            cache_key = (window['trial_idx'], window['start_sample']) if self._tf_cache is not None else None
            if cache_key is not None and cache_key in self._tf_cache:
                eeg_tf = self._tf_cache.pop(cache_key).copy()
                self._tf_cache[cache_key] = eeg_tf  # re-insert at end (LRU)
            else:
                eeg_tf = self._transform_eeg(eeg_window)
                if self._tf_cache is not None:
                    self._tf_cache[cache_key] = eeg_tf.copy()
                    if len(self._tf_cache) > self._tf_cache_max:
                        self._tf_cache.popitem(last=False)
            eeg_tf = np.asarray(eeg_tf, dtype=np.float32)
        else:
            # Simple reshape
            eeg_tf = eeg_window.T[:, :, np.newaxis]
        
        # Convert to tensors
        # eeg_tf shape: (n_channels, time_frames, n_freq_bands) = (66, 32, 5)
        # PyTorch Conv2d expects: (batch, channels, height, width)
        # So we need to ensure shape is (channels, time, freq) which is correct
        eeg_tensor = torch.FloatTensor(eeg_tf)
        
        # Feature-level normalization (per channel, per sample)
        # CRITICAL: This normalization may be removing ALL signal!
        # If logits are near zero, this normalization is likely the culprit
        # DISABLED to prevent signal removal - re-enable only if needed
        # if eeg_tensor.numel() > 0:
        #     # Normalize each channel independently
        #     for ch in range(eeg_tensor.shape[0]):
        #         ch_data = eeg_tensor[ch]
        #         if ch_data.std() > 1e-6:  # Avoid division by zero
        #             eeg_tensor[ch] = (ch_data - ch_data.mean()) / (ch_data.std() + 1e-8)
        
        label_tensor = torch.tensor(window['label'], dtype=torch.long)
        
        # Return trial_idx for trial-level loss aggregation (if needed)
        trial_idx = window.get('trial_idx', -1)
        
        return eeg_tensor, label_tensor, trial_idx


def compute_class_weights_per_trial(dataset: 'FULCNNLOCDataset', 
                                    train_window_indices: List[int],
                                    num_classes: int = 2,
                                    cap: Tuple[float, float] = (0.5, 2.0)) -> torch.Tensor:
    """
    Compute class weights from trial-level counts (one label per trial), not per window.
    Use this when using trial-level loss or to avoid biasing the classifier by window count.
    """
    trial_ids = sorted(set(dataset.windows[i]['trial_idx'] for i in train_window_indices))
    trial_labels = [dataset.trials[t]['label'] for t in trial_ids]
    trial_label_counts = np.bincount(trial_labels, minlength=num_classes)
    total_trials = len(trial_ids)
    weights = [
        total_trials / (num_classes * trial_label_counts[c]) if trial_label_counts[c] > 0 else 1.0
        for c in range(num_classes)
    ]
    return torch.clamp(torch.FloatTensor(weights), cap[0], cap[1]), trial_label_counts, total_trials


def split_dataset(dataset: FULCNNLOCDataset, train_ratio: float = 0.7, 
                  val_ratio: float = 0.15, split_by: str = 'subject') -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train/val/test sets.
    
    Args:
        dataset: FULCNNLOCDataset instance
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        split_by: 'subject' (split by subject, subject-independent), 
                  'trial' (split by trial, prevents leakage but may leak across subjects),
                  or 'window' (random window split, allows leakage - not recommended)
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    if split_by == 'subject':
        # Split by subject for true subject-independent evaluation
        unique_subjects = set(t['subject_id'] for t in dataset.trials)
        unique_subjects = sorted(list(unique_subjects))
        
        # Guard: check for 'unknown' subjects and minimum subject count
        if "unknown" in unique_subjects:
            raise ValueError("Found subject_id='unknown'. Cannot do subject split safely.")
        if len(unique_subjects) < 3:
            raise ValueError(f"Need at least 3 subjects for train/val/test subject split, got {len(unique_subjects)}")
        
        # Random split of subjects
        np.random.seed(42)
        np.random.shuffle(unique_subjects)
        
        n_subjects = len(unique_subjects)
        # Improved split: more subjects in validation for more stable tuning
        # Use 12/4/2 instead of 12/2/4 for better validation stability
        if n_subjects >= 18:
            # For 18 subjects: 12 train / 4 val / 2 test (better than 12/2/4)
            train_subjects = 12
            val_subjects = 4
        else:
            # Fallback for smaller datasets
            train_subjects = int(train_ratio * n_subjects)
            val_subjects = int(val_ratio * n_subjects)
        
        train_subject_set = set(unique_subjects[:train_subjects])
        val_subject_set = set(unique_subjects[train_subjects:train_subjects + val_subjects])
        test_subject_set = set(unique_subjects[train_subjects + val_subjects:])
        
        # CRITICAL FIX: Compute normalization stats ONLY on training subjects
        # This prevents data leakage where test/val normalization uses information from test/val data
        if (dataset.subject_wise_normalization or dataset.global_normalization) and dataset._needs_subject_stats:
            if dataset.global_normalization:
                print(f"\nComputing global normalization stats ONLY on training subjects...")
                dataset.global_stats = dataset._compute_global_stats(train_subject_ids=train_subject_set)
            if dataset.subject_wise_normalization:
                print(f"\nComputing subject-wise normalization stats ONLY on training subjects...")
                dataset.subject_stats = dataset._compute_subject_stats(train_subject_ids=train_subject_set)
            dataset._needs_subject_stats = False  # Mark as computed
        
        # Create window indices for each split (all windows from subjects in each set)
        train_window_indices = [i for i, w in enumerate(dataset.windows) 
                               if dataset.trials[w['trial_idx']]['subject_id'] in train_subject_set]
        val_window_indices = [i for i, w in enumerate(dataset.windows) 
                             if dataset.trials[w['trial_idx']]['subject_id'] in val_subject_set]
        test_window_indices = [i for i, w in enumerate(dataset.windows) 
                              if dataset.trials[w['trial_idx']]['subject_id'] in test_subject_set]
        
        # Helper to compute trial-level label distribution
        def get_trial_label_dist(indices):
            trial_ids = sorted(set(dataset.windows[i]['trial_idx'] for i in indices))
            labels = [dataset.trials[t]['label'] for t in trial_ids]
            return len(trial_ids), dict(enumerate(np.bincount(labels, minlength=2)))
        
        train_n_trials, train_trial_labels = get_trial_label_dist(train_window_indices)
        val_n_trials, val_trial_labels = get_trial_label_dist(val_window_indices)
        test_n_trials, test_trial_labels = get_trial_label_dist(test_window_indices)
        
        # Window-level label distribution
        train_labels = [dataset.windows[idx]['label'] for idx in train_window_indices]
        val_labels = [dataset.windows[idx]['label'] for idx in val_window_indices]
        test_labels = [dataset.windows[idx]['label'] for idx in test_window_indices]
        
        print(f"\nSplitting by subject (subject-independent evaluation):")
        print(f"  Train: {len(train_subject_set)} subjects, {train_n_trials} trials, {len(train_window_indices)} windows")
        print(f"    Trial label dist: {train_trial_labels}, Window label dist: {dict(enumerate(np.bincount(train_labels, minlength=2)))}")
        print(f"  Val: {len(val_subject_set)} subjects, {val_n_trials} trials, {len(val_window_indices)} windows")
        print(f"    Trial label dist: {val_trial_labels}, Window label dist: {dict(enumerate(np.bincount(val_labels, minlength=2)))}")
        print(f"  Test: {len(test_subject_set)} subjects, {test_n_trials} trials, {len(test_window_indices)} windows")
        print(f"    Trial label dist: {test_trial_labels}, Window label dist: {dict(enumerate(np.bincount(test_labels, minlength=2)))}")
        
    elif split_by == 'trial':
        # Split by trial to prevent data leakage
        # All windows from a trial go to the same split
        unique_trials = set(w['trial_idx'] for w in dataset.windows)
        unique_trials = sorted(list(unique_trials))
        
        # Random split of trials
        np.random.seed(42)
        np.random.shuffle(unique_trials)
        
        n_trials = len(unique_trials)
        train_trials = int(train_ratio * n_trials)
        val_trials = int(val_ratio * n_trials)
        
        train_trial_set = set(unique_trials[:train_trials])
        val_trial_set = set(unique_trials[train_trials:train_trials + val_trials])
        test_trial_set = set(unique_trials[train_trials + val_trials:])
        
        # Create window indices for each split
        train_window_indices = [i for i, w in enumerate(dataset.windows) if w['trial_idx'] in train_trial_set]
        val_window_indices = [i for i, w in enumerate(dataset.windows) if w['trial_idx'] in val_trial_set]
        test_window_indices = [i for i, w in enumerate(dataset.windows) if w['trial_idx'] in test_trial_set]
        
        # Helper to compute trial-level label distribution
        def get_trial_label_dist(trial_set):
            labels = [dataset.trials[t]['label'] for t in trial_set]
            return dict(enumerate(np.bincount(labels, minlength=2)))
        
        train_trial_labels = get_trial_label_dist(train_trial_set)
        val_trial_labels = get_trial_label_dist(val_trial_set)
        test_trial_labels = get_trial_label_dist(test_trial_set)
        
        # Window-level label distribution
        train_labels = [dataset.windows[idx]['label'] for idx in train_window_indices]
        val_labels = [dataset.windows[idx]['label'] for idx in val_window_indices]
        test_labels = [dataset.windows[idx]['label'] for idx in test_window_indices]
        
        print(f"\nSplitting by trial:")
        print(f"  Train: {len(train_trial_set)} trials, {len(train_window_indices)} windows")
        print(f"    Trial label dist: {train_trial_labels}, Window label dist: {dict(enumerate(np.bincount(train_labels, minlength=2)))}")
        print(f"  Val: {len(val_trial_set)} trials, {len(val_window_indices)} windows")
        print(f"    Trial label dist: {val_trial_labels}, Window label dist: {dict(enumerate(np.bincount(val_labels, minlength=2)))}")
        print(f"  Test: {len(test_trial_set)} trials, {len(test_window_indices)} windows")
        print(f"    Trial label dist: {test_trial_labels}, Window label dist: {dict(enumerate(np.bincount(test_labels, minlength=2)))}")
        
    elif split_by == 'window':
        # Random split by window (allows leakage - not recommended)
        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        indices = list(range(total_size))
        np.random.seed(42)
        np.random.shuffle(indices)
        
        train_window_indices = indices[:train_size]
        val_window_indices = indices[train_size:train_size + val_size]
        test_window_indices = indices[train_size + val_size:]
        
        # Helper to compute trial-level label distribution
        def get_trial_label_dist(indices):
            trial_ids = sorted(set(dataset.windows[i]['trial_idx'] for i in indices))
            labels = [dataset.trials[t]['label'] for t in trial_ids]
            return len(trial_ids), dict(enumerate(np.bincount(labels, minlength=2)))
        
        train_n_trials, train_trial_labels = get_trial_label_dist(train_window_indices)
        val_n_trials, val_trial_labels = get_trial_label_dist(val_window_indices)
        test_n_trials, test_trial_labels = get_trial_label_dist(test_window_indices)
        
        # Window-level label distribution
        train_labels = [dataset.windows[idx]['label'] for idx in train_window_indices]
        val_labels = [dataset.windows[idx]['label'] for idx in val_window_indices]
        test_labels = [dataset.windows[idx]['label'] for idx in test_window_indices]
        
        print(f"\nSplitting by window (WARNING: may cause data leakage):")
        print(f"  Train: {train_n_trials} trials, {len(train_window_indices)} windows")
        print(f"    Trial label dist: {train_trial_labels}, Window label dist: {dict(enumerate(np.bincount(train_labels, minlength=2)))}")
        print(f"  Val: {val_n_trials} trials, {len(val_window_indices)} windows")
        print(f"    Trial label dist: {val_trial_labels}, Window label dist: {dict(enumerate(np.bincount(val_labels, minlength=2)))}")
        print(f"  Test: {test_n_trials} trials, {len(test_window_indices)} windows")
        print(f"    Trial label dist: {test_trial_labels}, Window label dist: {dict(enumerate(np.bincount(test_labels, minlength=2)))}")
    else:
        raise ValueError(f"split_by must be 'subject', 'trial', or 'window', got '{split_by}'")
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_window_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_window_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_window_indices)
    
    # Print label distribution in each split
    train_labels = [dataset.windows[idx]['label'] for idx in train_window_indices]
    val_labels = [dataset.windows[idx]['label'] for idx in val_window_indices]
    test_labels = [dataset.windows[idx]['label'] for idx in test_window_indices]
    
    print(f"\nLabel distribution:")
    print(f"  Train: {dict(enumerate(np.bincount(train_labels, minlength=2)))}")
    print(f"  Val: {dict(enumerate(np.bincount(val_labels, minlength=2)))}")
    print(f"  Test: {dict(enumerate(np.bincount(test_labels, minlength=2)))}")
    
    return train_dataset, val_dataset, test_dataset


class CNNLOCTrainer:
    """Trainer for CNN-LOC model."""
    
    def __init__(self, model: CNNLOCModel, device: torch.device, output_dir: str = "fulcnnloc_results"):
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.best_val_acc = 0.0
        self.best_model_path = self.output_dir / "best_model.pth"
    
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module, scheduler: Optional[Any] = None, 
                   track_logits: bool = False, epoch: int = 0,
                   use_trial_level_loss: bool = False) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            scheduler: Optional LR scheduler to step per batch (e.g., OneCycleLR)
            track_logits: If True, collect and return logit statistics for diagnostics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0  # Count actual processed batches
        
        epoch_logits = [] if track_logits else None
        epoch_probs = [] if track_logits else None
        
        for batch_idx, batch_data in enumerate(tqdm(train_loader, desc="Training")):
            # Handle both 2-value (backward compat) and 3-value (with trial_idx) returns
            if len(batch_data) == 3:
                data, target, trial_indices = batch_data
            else:
                data, target = batch_data
                trial_indices = None
            
            data, target = data.to(self.device), target.to(self.device)
            # Target is already scalar (no view needed)
            
            # Skip empty batches
            if target.size(0) == 0:
                continue
            
            n_batches += 1  # Count this batch
            
            # Forward
            output = self.model(data)
            
            # TRIAL-LEVEL LOSS AGGREGATION (Option A from user's fix)
            # If enabled, group windows by trial, average logits per trial, compute loss once per trial
            if use_trial_level_loss and trial_indices is not None:
                # Group outputs by trial_idx
                trial_indices = trial_indices.to(self.device)
                unique_trials = torch.unique(trial_indices)
                trial_logits = []
                trial_labels = []
                
                for trial_idx in unique_trials:
                    trial_mask = (trial_indices == trial_idx)
                    trial_output = output[trial_mask]  # All windows from this trial
                    # Label is same for all windows in trial; take first window's label
                    trial_target = target[trial_mask][0].unsqueeze(0)
                    
                    # Average logits across windows in this trial
                    avg_logits = trial_output.mean(dim=0, keepdim=True)
                    trial_logits.append(avg_logits)
                    trial_labels.append(trial_target)
                
                # Stack and compute loss
                trial_logits = torch.cat(trial_logits, dim=0)
                trial_labels = torch.cat(trial_labels, dim=0)
                loss = criterion(trial_logits, trial_labels)
            else:
                # Standard window-level loss
                loss = criterion(output, target)
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠ WARNING: Invalid loss detected at batch {batch_idx}: {loss.item()}")
                continue
            
            # Early detection: Check if model is collapsing in first epoch
            if track_logits and batch_idx == 0 and epoch == 0:
                with torch.no_grad():
                    probs = F.softmax(output, dim=1)
                    preds = output.argmax(dim=1)
                    pred_dist = torch.bincount(preds, minlength=2).float() / len(preds)
                    print(f"  First batch predictions: {pred_dist[0]*100:.1f}% class 0, {pred_dist[1]*100:.1f}% class 1")
                    if pred_dist[0] > 0.9 or pred_dist[1] > 0.9:
                        print(f"  ⚠ WARNING: Model already biased toward one class in first batch!")
                        print(f"  This suggests initialization or early gradient issues.")
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Monitor gradients (on first epoch only)
            if track_logits and batch_idx == 0:
                total_grad_norm = 0.0
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        param_grad_norm = param.grad.data.norm(2)
                        total_grad_norm += param_grad_norm.item() ** 2
                        if 'classifier' in name and 'weight' in name:
                            print(f"  Grad norm for {name}: {param_grad_norm.item():.6f}")
                total_grad_norm = total_grad_norm ** (1. / 2)
                print(f"  Total gradient norm: {total_grad_norm:.6f}")
            
            # Relaxed clipping (5.0) so useful gradients aren't clipped when model is learning
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            optimizer.step()
            
            # Step scheduler per batch (required for OneCycleLR)
            if scheduler is not None:
                scheduler.step()
            
            # Accumulate loss and accuracy
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            
            # Track logits for diagnostics
            if track_logits:
                probs = F.softmax(output, dim=1)
                epoch_logits.extend(output[:, 1].detach().cpu().numpy())
                epoch_probs.extend(probs[:, 1].detach().cpu().numpy())
        
        if total == 0:
            return float('inf'), 0.0
        
        # Divide by number of processed batches, not len(train_loader)
        avg_loss = total_loss / max(1, n_batches)
        accuracy = 100. * correct / total
        
        # Print logit statistics if tracking
        if track_logits and epoch_logits:
            logits_arr = np.array(epoch_logits)
            probs_arr = np.array(epoch_probs)
            logit_mean = np.mean(logits_arr)
            logit_std = np.std(logits_arr)
            prob_mean = np.mean(probs_arr)
            prob_std = np.std(probs_arr)
            print(f"  Train logits: mean={logit_mean:.3f}, std={logit_std:.3f}")
            print(f"  Train probs: mean={prob_mean:.3f}, std={prob_std:.3f}")
            
            # Warn if logits are too small (indicates model not learning)
            if abs(logit_mean) < 0.1 and logit_std < 0.1:
                print(f"  ⚠ CRITICAL: Logits are near zero (mean={logit_mean:.3f}, std={logit_std:.3f})!")
                print(f"    This indicates the model is not learning. Check:")
                print(f"    1) Model initialization (may be too small)")
                print(f"    2) Learning rate (may be too low)")
                print(f"    3) Feature normalization (may be removing signal)")
                print(f"    4) Gradient flow (check gradient norms)")
            elif prob_std < 0.05:
                print(f"  ⚠ WARNING: Probability std is very small ({prob_std:.3f}) - model output is near-constant")
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0  # Count actual processed batches
        
        # Collect predictions for bias/collapse diagnostics
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc="Validation"):
                # Handle both 2-value and 3-value returns
                if len(batch_data) == 3:
                    data, target, _ = batch_data  # Ignore trial_indices in validation
                else:
                    data, target = batch_data
                data, target = data.to(self.device), target.to(self.device)
                # Target is already scalar (no view needed)
                
                # Skip empty batches
                if target.size(0) == 0:
                    continue
                
                n_batches += 1  # Count this batch
                
                output = self.model(data)
                loss = criterion(output, target)
                probabilities = F.softmax(output, dim=1)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
                
                # Collect for diagnostics
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probabilities[:, 1].cpu().numpy())
        
        if total == 0:
            return float('inf'), 0.0
        
        # Divide by number of processed batches, not len(val_loader)
        avg_loss = total_loss / max(1, n_batches)
        accuracy = 100. * correct / total
        
        # Bias/collapse diagnostics
        preds_array = np.array(all_preds)
        targets_array = np.array(all_targets)
        probs_array = np.array(all_probs)
        
        pred_class_0_pct = 100.0 * np.sum(preds_array == 0) / len(preds_array) if len(preds_array) > 0 else 0.0
        pred_class_1_pct = 100.0 * np.sum(preds_array == 1) / len(preds_array) if len(preds_array) > 0 else 0.0
        mean_prob = np.mean(probs_array) if len(probs_array) > 0 else 0.0
        std_prob = np.std(probs_array) if len(probs_array) > 0 else 0.0
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(targets_array, preds_array, labels=[0, 1])
        
        print(f"  Val pred dist: {pred_class_0_pct:.1f}% class 0, {pred_class_1_pct:.1f}% class 1")
        print(f"  Val prob stats: mean={mean_prob:.3f}, std={std_prob:.3f}")
        print(f"  Val confusion: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
        
        # Warn if collapsing to one class
        if pred_class_1_pct > 85.0:
            print(f"  ⚠ WARNING: Model collapsing to class 1 ({pred_class_1_pct:.1f}% predictions)")
        elif pred_class_0_pct > 85.0:
            print(f"  ⚠ WARNING: Model collapsing to class 0 ({pred_class_0_pct:.1f}% predictions)")
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 50, learning_rate: float = 1e-4,
              weight_decay: float = 1e-5, patience: int = 10,
              class_weights: Optional[torch.Tensor] = None,
              diagnostic_baseline: bool = False,
              use_trial_level_loss: bool = False):
        """Train the model.
        
        Args:
            diagnostic_baseline: If True, disable all regularization (label_smoothing=0, dropout=0, 
                               no scheduler, no class weights) to test if optimization stack is suppressing learning.
            use_trial_level_loss: If True, compute loss per trial (average logits per trial) instead of per window.
        """
        
        # DIAGNOSTIC BASELINE MODE: Remove all regularization to test if learning is possible
        if diagnostic_baseline:
            print("\n" + "="*70)
            print("🔬 DIAGNOSTIC BASELINE MODE: All regularization disabled")
            print("="*70)
            print("  - label_smoothing = 0.0")
            print("  - dropout = 0.0 (model dropout will be disabled)")
            print("  - class_weights = None")
            print("  - scheduler = None")
            print("  - weight_decay = 0.0")
            print("="*70 + "\n")
            
            # Disable dropout in model
            for module in self.model.modules():
                if isinstance(module, nn.Dropout):
                    module.p = 0.0
            
            criterion = nn.CrossEntropyLoss(weight=None, label_smoothing=0.0)
            optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.0)
            scheduler = None
        else:
            # Use class weights if provided, otherwise use uniform weights
            # Cap weights to prevent extreme values that cause collapse
            if class_weights is not None:
                class_weights = torch.clamp(class_weights, 0.5, 2.0)
                class_weights = class_weights.to(self.device)
                print(f"Using class weights (capped 0.5-2.0): {class_weights.cpu().numpy()}")
            
            # Use moderate label smoothing; 0.15 was encouraging near-constant ~0.5 outputs and collapse
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
            # Increased weight decay for better regularization
            effective_weight_decay = max(weight_decay, 1e-4)  # At least 1e-4
            optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=effective_weight_decay)
            # OneCycleLR: longer warmup and gentler peak to reduce collapse
            scheduler = OneCycleLR(
                optimizer,
                max_lr=learning_rate * 2,  # Gentler peak (2x) for stability
                epochs=num_epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.4,  # Longer warmup (40% of steps)
                div_factor=10.0,
                final_div_factor=100.0
            )
        
        patience_counter = 0
        initial_val_acc = None  # Track initial validation accuracy
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Track logits on first epoch and every 10th epoch to catch collapse early
            track_logits = (epoch == 0) or (epoch % 10 == 0)
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion, scheduler, 
                                                     track_logits=track_logits, epoch=epoch,
                                                     use_trial_level_loss=use_trial_level_loss)
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # Track initial validation accuracy to detect if model starts collapsed
            if initial_val_acc is None:
                initial_val_acc = val_acc
                # Check if within 1% of 50% (chance level for balanced binary classification)
                if abs(initial_val_acc - 50.0) < 1.0:
                    print(f"⚠ WARNING: Initial validation accuracy is near chance level ({initial_val_acc:.2f}%)!")
                    print(f"  This suggests the model started collapsed to one class.")
                    print(f"  Check: 1) Model initialization, 2) First batch predictions, 3) Label distribution")
            
            # Note: scheduler is stepped per batch in train_epoch() for OneCycleLR
            # Do NOT step here again or it will break the schedule
            
            # Calculate train-val gap to detect overfitting
            train_val_gap = train_acc - val_acc
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}%")
            print(f"Train-Val Gap: {train_val_gap:.2f}%")
            
            # Warn if overfitting is detected
            if train_val_gap > 25.0:
                print(f"⚠ WARNING: Severe overfitting detected! Train-Val gap: {train_val_gap:.2f}%")
                print("   Consider: 1) Increasing dropout, 2) Reducing model capacity, 3) More data augmentation")
            elif train_val_gap > 15.0:
                print(f"⚠ WARNING: Moderate overfitting detected! Train-Val gap: {train_val_gap:.2f}%")
            
            # Only update best_val_acc if we actually improved (not just equal)
            # This prevents saving a model that's stuck at 50% from the start
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'train_acc': train_acc,
                    'train_val_gap': train_val_gap,
                }, self.best_model_path)
                print(f"New best model saved! Val Acc: {val_acc:.4f}% (gap: {train_val_gap:.2f}%)")
            else:
                patience_counter += 1
                # Warn if stuck at exactly 50% for multiple epochs
                if val_acc == 50.0 and patience_counter >= 3:
                    print(f"⚠ WARNING: Validation accuracy stuck at 50% for {patience_counter} epochs")
                    print(f"  Model may be collapsed to one class. Check prediction distribution.")
            
            if patience_counter >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                # Check if model is stuck at chance level (50% ± 1%)
                if initial_val_acc is not None and abs(initial_val_acc - 50.0) < 1.0:
                    if abs(self.best_val_acc - 50.0) < 1.0:
                        print(f"⚠ CRITICAL: Model stuck at chance level (~50%)!")
                        print(f"  Initial val acc: {initial_val_acc:.4f}%, Best val acc: {self.best_val_acc:.4f}%")
                        print(f"  This indicates the model collapsed immediately and never recovered.")
                elif self.best_val_acc == 0.0:
                    print(f"⚠ CRITICAL: Model never improved from initial state!")
                    print(f"  Best val acc: {self.best_val_acc:.4f}% (initial: {initial_val_acc:.4f}%)")
                    print(f"  This indicates the model collapsed immediately and never recovered.")
                break
        
        print(f"Training completed. Best validation accuracy: {self.best_val_acc:.4f}%")
        return self.best_val_acc
    
    def _aggregate_by_trial(self, dataset: 'FULCNNLOCDataset', subset: torch.utils.data.Subset,
                           window_preds: np.ndarray, window_probs: np.ndarray, 
                           window_targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Aggregate window predictions into trial predictions (majority vote) and trial probabilities (mean).
        subset is a torch.utils.data.Subset, so subset.indices maps to dataset.windows indices.
        """
        trial_dict = {}
        for i, window_idx in enumerate(subset.indices):
            w = dataset.windows[window_idx]
            tid = w['trial_idx']
            if tid not in trial_dict:
                trial_dict[tid] = {'preds': [], 'probs': [], 'targets': []}
            trial_dict[tid]['preds'].append(int(window_preds[i]))
            trial_dict[tid]['probs'].append(float(window_probs[i]))
            trial_dict[tid]['targets'].append(int(window_targets[i]))

        trial_preds, trial_probs, trial_targets = [], [], []
        # Sort by trial id for stable outputs across runs
        for tid in sorted(trial_dict.keys()):
            v = trial_dict[tid]
            # target should be constant within a trial; use majority to be safe
            t = int(np.round(np.mean(v['targets'])))
            counts = np.bincount(np.array(v['preds']), minlength=2)
            p = int(np.argmax(counts))
            prob = float(np.mean(v['probs']))
            trial_targets.append(t)
            trial_preds.append(p)
            trial_probs.append(prob)

        return np.array(trial_preds), np.array(trial_probs), np.array(trial_targets)
    
    def _find_optimal_threshold(self, val_probs: np.ndarray, val_targets: np.ndarray) -> float:
        """Find optimal threshold using Youden's J (maximize TPR + TNR - 1)."""
        from sklearn.metrics import roc_curve
        try:
            fpr, tpr, thresholds = roc_curve(val_targets, val_probs)
            youden_j = tpr - fpr
            optimal_idx = np.argmax(youden_j)
            optimal_threshold = thresholds[optimal_idx]
            return optimal_threshold
        except:
            return 0.5  # Default if ROC curve calculation fails
    
    def test(self, dataset: 'FULCNNLOCDataset', test_subset: torch.utils.data.Subset, 
             test_loader: DataLoader, val_subset: Optional[torch.utils.data.Subset] = None,
             val_loader: Optional[DataLoader] = None) -> Dict:
        """
        Test model and compute metrics (trial-level). Threshold is tuned on val only;
        all reported metrics (accuracy, AUC, etc.) are computed on test only (no leakage).
        
        Args:
            val_subset: Optional validation subset for threshold tuning
            val_loader: Optional validation loader for threshold tuning
        """
        checkpoint = torch.load(self.best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc="Testing"):
                # Handle both 2-value and 3-value returns
                if len(batch_data) == 3:
                    data, target, _ = batch_data  # Ignore trial_indices in testing
                else:
                    data, target = batch_data
                data, target = data.to(self.device), target.to(self.device)
                # Target is already scalar (no view needed)
                
                # Skip empty batches
                if target.size(0) == 0:
                    continue
                
                output = self.model(data)
                probabilities = F.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
        
        # Window-level arrays
        preds = np.array(all_predictions)
        targets = np.array(all_targets)
        probs = np.array(all_probabilities)
        
        # Trial-level aggregation
        trial_preds, trial_probs, trial_targets = self._aggregate_by_trial(
            dataset, test_subset, preds, probs, targets
        )
        
        # Optimize threshold on validation set (if available) using Youden's J
        optimal_threshold = 0.5  # Default
        if val_subset is not None and val_loader is not None:
            # Compute validation predictions for threshold tuning
            val_preds, val_probs, val_targets = [], [], []
            with torch.no_grad():
                for batch_data in val_loader:
                    # Handle both 2-value and 3-value returns
                    if len(batch_data) == 3:
                        data, target, _ = batch_data  # Ignore trial_indices
                    else:
                        data, target = batch_data
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    probabilities = F.softmax(output, dim=1)
                    val_probs.extend(probabilities[:, 1].cpu().numpy())
                    val_targets.extend(target.cpu().numpy())
            
            # Aggregate validation predictions to trial level
            val_window_probs = np.array(val_probs)
            val_window_targets = np.array(val_targets)
            val_trial_probs, val_trial_targets, _ = self._aggregate_by_trial(
                dataset, val_subset, np.zeros_like(val_window_targets), val_window_probs, val_window_targets
            )
            
            # Find optimal threshold
            optimal_threshold = self._find_optimal_threshold(val_trial_probs, val_trial_targets)
            print(f"  Optimal threshold (from validation): {optimal_threshold:.4f}")
        else:
            print(f"  Using default threshold: 0.5 (no validation set provided for tuning)")
        
        trial_preds_thresholded = (trial_probs >= optimal_threshold).astype(int)
        
        # Compute window-level metrics
        window_accuracy = accuracy_score(targets, preds)
        
        # Compute trial-level metrics (with default threshold)
        accuracy = accuracy_score(trial_targets, trial_preds)
        
        # Also compute with thresholded predictions
        accuracy_thresholded = accuracy_score(trial_targets, trial_preds_thresholded)
        
        # Calculate trial-level ROC-AUC with error handling
        try:
            unique_trial_targets = np.unique(trial_targets)
            if len(unique_trial_targets) < 2:
                trial_roc_auc = float('nan')
                print(f"Warning: Only one class in trial-level test set ({unique_trial_targets}), ROC-AUC is undefined")
            else:
                trial_roc_auc = roc_auc_score(trial_targets, trial_probs)
        except Exception as e:
            print(f"Warning: Could not calculate trial-level ROC-AUC: {e}")
            trial_roc_auc = float('nan')
        
        # Calculate trial-level additional metrics
        try:
            # FIXED: precision_recall_fscore_support returns 4 values: precision, recall, f1, support
            trial_precision, trial_recall, trial_f1, _ = precision_recall_fscore_support(
                trial_targets, trial_preds, average='binary', zero_division=0)
            # FIXED: For macro, also returns 4 values
            trial_f1_macro_result = precision_recall_fscore_support(
                trial_targets, trial_preds, average='macro', zero_division=0)
            trial_f1_macro = trial_f1_macro_result[2]  # f1 is the 3rd element (index 2)
            trial_balanced_acc = balanced_accuracy_score(trial_targets, trial_preds)
        except Exception as e:
            print(f"Warning: Could not calculate trial-level additional metrics: {e}")
            import traceback
            traceback.print_exc()
            trial_precision = trial_recall = trial_f1 = trial_f1_macro = trial_balanced_acc = float('nan')
        
        # Confusion matrix for trial-level
        cm = confusion_matrix(trial_targets, trial_preds, labels=[0, 1])
        
        # Window-level metrics (for reference)
        try:
            unique_window_targets = np.unique(targets)
            if len(unique_window_targets) < 2:
                window_roc_auc = float('nan')
            else:
                window_roc_auc = roc_auc_score(targets, probs)
        except:
            window_roc_auc = float('nan')
        
        try:
            window_precision, window_recall, window_f1, _ = precision_recall_fscore_support(
                targets, preds, average='binary', zero_division=0)
            window_balanced_acc = balanced_accuracy_score(targets, preds)
        except:
            window_precision = window_recall = window_f1 = window_balanced_acc = float('nan')
        
        results = {
            # Trial-level metrics (primary)
            'trial_level': {
                'accuracy': float(accuracy),
                'roc_auc': None if np.isnan(trial_roc_auc) else float(trial_roc_auc),
                'precision': None if np.isnan(trial_precision) else float(trial_precision),
                'recall': None if np.isnan(trial_recall) else float(trial_recall),
                'f1_score': None if np.isnan(trial_f1) else float(trial_f1),
                'f1_macro': None if np.isnan(trial_f1_macro) else float(trial_f1_macro),
                'balanced_accuracy': None if np.isnan(trial_balanced_acc) else float(trial_balanced_acc),
                'confusion_matrix': cm.tolist(),
                'n_trials': int(len(trial_targets)),
                'predictions': trial_preds.tolist(),
                'targets': trial_targets.tolist(),
                'probabilities': trial_probs.tolist()
            },
            # Window-level metrics (for reference)
            'window_level': {
                'accuracy': float(window_accuracy),
                'roc_auc': None if np.isnan(window_roc_auc) else float(window_roc_auc),
                'precision': None if np.isnan(window_precision) else float(window_precision),
                'recall': None if np.isnan(window_recall) else float(window_recall),
                'f1_score': None if np.isnan(window_f1) else float(window_f1),
                'balanced_accuracy': None if np.isnan(window_balanced_acc) else float(window_balanced_acc),
                'n_windows': int(len(targets))
            },
            # Backward compatibility (trial-level)
            'accuracy': float(accuracy),
            'roc_auc': None if np.isnan(trial_roc_auc) else float(trial_roc_auc),
            'precision': None if np.isnan(trial_precision) else float(trial_precision),
            'recall': None if np.isnan(trial_recall) else float(trial_recall),
            'f1_score': None if np.isnan(trial_f1) else float(trial_f1),
            'f1_macro': None if np.isnan(trial_f1_macro) else float(trial_f1_macro),
            'balanced_accuracy': None if np.isnan(trial_balanced_acc) else float(trial_balanced_acc),
            'best_val_acc': self.best_val_acc
        }
        
        # Print confusion matrix and key metrics
        print("\nTrial-level Confusion Matrix:")
        print(f"  True Neg (0): {cm[0,0]}, False Pos (0→1): {cm[0,1]}")
        print(f"  False Neg (1→0): {cm[1,0]}, True Pos (1): {cm[1,1]}")
        print(f"\nTrial-level Metrics Summary:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Balanced Accuracy: {trial_balanced_acc:.4f}" if not np.isnan(trial_balanced_acc) else "  Balanced Accuracy: N/A")
        print(f"  F1 (binary): {trial_f1:.4f}" if not np.isnan(trial_f1) else "  F1 (binary): N/A")
        print(f"  F1 (macro): {trial_f1_macro:.4f}" if not np.isnan(trial_f1_macro) else "  F1 (macro): N/A")
        print(f"  ROC-AUC: {trial_roc_auc:.4f}" if not np.isnan(trial_roc_auc) else "  ROC-AUC: N/A")
        
        # Check for class bias
        pred_class_0_pct = 100.0 * np.sum(trial_preds == 0) / len(trial_preds) if len(trial_preds) > 0 else 0.0
        pred_class_1_pct = 100.0 * np.sum(trial_preds == 1) / len(trial_preds) if len(trial_preds) > 0 else 0.0
        true_class_0_pct = 100.0 * np.sum(trial_targets == 0) / len(trial_targets) if len(trial_targets) > 0 else 0.0
        true_class_1_pct = 100.0 * np.sum(trial_targets == 1) / len(trial_targets) if len(trial_targets) > 0 else 0.0
        
        print(f"\nClass Distribution:")
        print(f"  True: {true_class_0_pct:.1f}% class 0, {true_class_1_pct:.1f}% class 1")
        print(f"  Pred: {pred_class_0_pct:.1f}% class 0, {pred_class_1_pct:.1f}% class 1")
        
        if pred_class_1_pct > 85.0 or pred_class_0_pct > 85.0:
            print(f"  ⚠ WARNING: Strong class bias detected! Model collapsing to one class.")
        
        return results


def run_overfit_test(tfrecord_dir: str, window_size: int = 512, overlap: float = 0.5,
                     num_epochs: int = 100, learning_rate: float = 1e-3,
                     sampling_rate: int = 64, seed: int = 42) -> Dict:
    """
    Sanity check: Overfit test on tiny subset (1 subject, 2-4 trials).
    
    Goal: Prove the pipeline can learn at all. If it can't reach ~100% train accuracy
    on a tiny subset with minimal regularization, there's likely a data/label or model bug.
    
    Returns:
        Dict with train accuracy, train AUC, logit statistics, and success/failure flag
    """
    print("\n" + "="*80)
    print("OVERFIT TEST: Tiny Subset (1 subject, 2-4 trials)")
    print("="*80)
    print("This test verifies the pipeline can learn at all.")
    print("Expected: Train accuracy → ~100%, Train AUC → ~1.0, logits separate strongly")
    print("If this fails, there's likely a data/label or model/loss bug.")
    print("="*80)
    
    seed_everything(seed)
    
    # Overfit test: NO normalization and NO augmentation so the supervision signal is preserved.
    # Subject-wise normalization and augmentation would make the target moving and prevent overfitting.
    dataset = FULCNNLOCDataset(
        tfrecord_dir=tfrecord_dir,
        window_size=window_size,
        overlap=overlap,
        transform_eeg=True,
        augment=False,
        subject_wise_normalization=False,
        global_normalization=False
    )
    
    # Find one subject with both classes
    subject_trials = {}
    for trial in dataset.trials:
        subj = trial['subject_id']
        if subj not in subject_trials:
            subject_trials[subj] = {'trials': [], 'labels': []}
        subject_trials[subj]['trials'].append(trial)
        subject_trials[subj]['labels'].append(trial['label'])
    
    # Find subject with both classes
    test_subject = None
    for subj, data in subject_trials.items():
        unique_labels = set(data['labels'])
        if len(unique_labels) == 2 and len(data['trials']) >= 4:
            test_subject = subj
            break
    
    if test_subject is None:
        raise ValueError("No subject found with both classes and >= 4 trials")
    
    # Select 2 trials per class (4 trials total)
    trials_by_label = {0: [], 1: []}
    for trial in subject_trials[test_subject]['trials']:
        trials_by_label[trial['label']].append(trial)
    
    selected_trials = []
    for label in [0, 1]:
        if len(trials_by_label[label]) >= 2:
            selected_trials.extend(trials_by_label[label][:2])
        else:
            selected_trials.extend(trials_by_label[label])
    
    selected_trial_indices = [t['trial_idx'] for t in selected_trials]
    print(f"\nSelected subject: {test_subject}")
    print(f"Selected trials: {selected_trial_indices}")
    print(f"Trial labels: {[t['label'] for t in selected_trials]}")
    
    # Get all windows from selected trials
    train_window_indices = [i for i, w in enumerate(dataset.windows) 
                           if w['trial_idx'] in selected_trial_indices]
    
    print(f"Total windows: {len(train_window_indices)}")
    
    # Create tiny dataset (all windows go to train, no val/test)
    train_dataset = torch.utils.data.Subset(dataset, train_window_indices)
    val_dataset = torch.utils.data.Subset(dataset, train_window_indices)  # Same for monitoring
    
    # No class weights, minimal regularization
    train_loader = DataLoader(train_dataset, batch_size=min(8, len(train_dataset)), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=min(8, len(val_dataset)), shuffle=False)
    
    # Create model
    # Determine input dimensions from time-frequency transform
    window_seconds = dataset.window_size / dataset.sampling_rate
    input_time = 128 if window_seconds >= 8.0 else 64
    input_freq = 8  # Updated: 8 frequency bands
    
    model = CNNLOCModel(
        input_channels=dataset.n_channels,
        input_time=input_time,
        input_freq=input_freq,
        num_classes=2,
        dropout_rate=0.0  # No dropout for overfit test
    )
    
    # Create trainer
    trainer = CNNLOCTrainer(model=model, device=device, output_dir="fulcnnloc_results/overfit_test")
    
    # Train with minimal regularization
    criterion = nn.CrossEntropyLoss()  # No weights, no label smoothing
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)  # No weight decay
    
    model.train()
    all_logits = []
    all_probs = []
    all_targets = []
    
    print(f"\nTraining for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        epoch_logits = []
        epoch_probs = []
        epoch_targets = []
        correct = 0
        total = 0
        
        for batch_data in train_loader:
            # Handle both 2-value and 3-value returns
            if len(batch_data) == 3:
                data, target, _ = batch_data  # Ignore trial_indices
            else:
                data, target = batch_data
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            probs = F.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            
            # Collect logits and probs
            epoch_logits.extend(output[:, 1].detach().cpu().numpy())
            epoch_probs.extend(probs[:, 1].detach().cpu().numpy())
            epoch_targets.extend(target.cpu().numpy())
        
        train_acc = 100.0 * correct / total
        all_logits.extend(epoch_logits)
        all_probs.extend(epoch_probs)
        all_targets.extend(epoch_targets)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logits_arr = np.array(epoch_logits)
            probs_arr = np.array(epoch_probs)
            targets_arr = np.array(epoch_targets)
            
            logit_mean = np.mean(logits_arr)
            logit_std = np.std(logits_arr)
            prob_mean = np.mean(probs_arr)
            prob_std = np.std(probs_arr)
            
            # Compute AUC
            try:
                train_auc = roc_auc_score(targets_arr, probs_arr) if len(np.unique(targets_arr)) == 2 else float('nan')
            except:
                train_auc = float('nan')
            
            print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Train AUC: {train_auc:.4f}")
            print(f"  Logits: mean={logit_mean:.3f}, std={logit_std:.3f}")
            print(f"  Probs: mean={prob_mean:.3f}, std={prob_std:.3f}")
        
        if train_acc >= 99.0:
            print(f"\n✓ SUCCESS: Reached {train_acc:.2f}% train accuracy at epoch {epoch+1}")
            break
    
    # Final evaluation
    model.eval()
    final_logits = []
    final_probs = []
    final_targets = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_data in val_loader:
            # Handle both 2-value and 3-value returns
            if len(batch_data) == 3:
                data, target, _ = batch_data  # Ignore trial_indices
            else:
                data, target = batch_data
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            
            final_logits.extend(output[:, 1].cpu().numpy())
            final_probs.extend(probs[:, 1].cpu().numpy())
            final_targets.extend(target.cpu().numpy())
    
    final_acc = 100.0 * correct / total
    final_logits_arr = np.array(final_logits)
    final_probs_arr = np.array(final_probs)
    final_targets_arr = np.array(final_targets)
    
    try:
        final_auc = roc_auc_score(final_targets_arr, final_probs_arr) if len(np.unique(final_targets_arr)) == 2 else float('nan')
    except:
        final_auc = float('nan')
    
    logit_mean = np.mean(final_logits_arr)
    logit_std = np.std(final_logits_arr)
    prob_mean = np.mean(final_probs_arr)
    prob_std = np.std(final_probs_arr)
    
    # Separate logits by class
    class_0_logits = final_logits_arr[final_targets_arr == 0]
    class_1_logits = final_logits_arr[final_targets_arr == 1]
    
    print(f"\n{'='*80}")
    print("OVERFIT TEST RESULTS")
    print("="*80)
    print(f"Final Train Accuracy: {final_acc:.2f}%")
    print(f"Final Train AUC: {final_auc:.4f}")
    print(f"Logit stats: mean={logit_mean:.3f}, std={logit_std:.3f}")
    print(f"  Class 0 logits: mean={np.mean(class_0_logits):.3f}, std={np.std(class_0_logits):.3f}")
    print(f"  Class 1 logits: mean={np.mean(class_1_logits):.3f}, std={np.std(class_1_logits):.3f}")
    print(f"Prob stats: mean={prob_mean:.3f}, std={prob_std:.3f}")
    
    success = final_acc >= 95.0 and not np.isnan(final_auc) and final_auc >= 0.95
    
    if success:
        print(f"\n✓ PASS: Pipeline can learn! (Acc >= 95%, AUC >= 0.95)")
    else:
        print(f"\n✗ FAIL: Pipeline cannot learn properly!")
        print(f"  This suggests a data/label alignment issue, model bug, or loss problem.")
        if final_acc < 95.0:
            print(f"  - Train accuracy too low: {final_acc:.2f}% (expected >= 95%)")
        if np.isnan(final_auc) or final_auc < 0.95:
            print(f"  - Train AUC too low: {final_auc:.4f} (expected >= 0.95)")
        if logit_std < 0.5:
            print(f"  - Logits have low variance (std={logit_std:.3f}), model may be stuck")
    
    return {
        'success': success,
        'final_accuracy': float(final_acc),
        'final_auc': float(final_auc) if not np.isnan(final_auc) else None,
        'logit_mean': float(logit_mean),
        'logit_std': float(logit_std),
        'prob_mean': float(prob_mean),
        'prob_std': float(prob_std),
        'class_0_logit_mean': float(np.mean(class_0_logits)),
        'class_1_logit_mean': float(np.mean(class_1_logits)),
        'n_trials': len(selected_trials),
        'n_windows': len(train_window_indices),
        'subject': test_subject
    }


def run_single_experiment(tfrecord_dir: str, window_size: int, overlap: float, 
                          batch_size: int, num_epochs: int, learning_rate: float,
                          dropout_rate: float, output_dir: str, 
                          sampling_rate: int = 64, split_by: str = 'subject',
                          seed: int = 42, shuffle_labels: bool = False,
                          use_leaked_split: bool = False,
                          diagnostic_baseline: bool = False,
                          use_trial_level_loss: bool = False,
                          subject_wise_normalization: bool = True,
                          global_normalization: bool = False,  # NEW: Use global stats instead of per-subject
                          augment: bool = False) -> Dict:
    """Run a single experiment with given hyperparameters."""
    
    # Set random seed for reproducibility
    seed_everything(seed)
    
    # Create dataset
    # CRITICAL FIX: Augmentation disabled by default (was destroying AAD signal)
    # Augmentation is too aggressive for low-SNR, phase-sensitive AAD signals
    dataset = FULCNNLOCDataset(
        tfrecord_dir=tfrecord_dir,
        window_size=window_size,
        overlap=overlap,
        transform_eeg=True,
        augment=augment,  # Default False - only enable if needed
        subject_wise_normalization=subject_wise_normalization,
        global_normalization=global_normalization  # NEW: Use global normalization
    )
    
    # Handle special diagnostic modes
    if use_leaked_split:
        # Leaked split: allow within-subject split (for diagnostic purposes)
        print("\n⚠ DIAGNOSTIC MODE: Using leaked split (train/val from same subjects)")
        print("  This tests if model can learn subject signatures (not task generalization)")
        train_dataset, val_dataset, test_dataset = split_dataset(dataset, split_by='trial')
    else:
        # Normal split (default: subject-independent evaluation)
        train_dataset, val_dataset, test_dataset = split_dataset(dataset, split_by=split_by)
    
    # Shuffle labels if requested (diagnostic baseline)
    if shuffle_labels:
        print("\n⚠ DIAGNOSTIC MODE: Shuffling labels (baseline test)")
        print("  Expected: Performance should drop to chance (~0.5 AUC)")
        # Shuffle labels in the dataset windows
        original_labels = [dataset.windows[idx]['label'] for idx in train_dataset.indices]
        shuffled_labels = original_labels.copy()
        np.random.seed(seed)
        np.random.shuffle(shuffled_labels)
        # Update dataset windows (this is a bit hacky but works for diagnostic)
        for i, idx in enumerate(train_dataset.indices):
            dataset.windows[idx]['label'] = shuffled_labels[i]
        print(f"  Shuffled {len(shuffled_labels)} training labels")
    
    # Class weights per trial (not per window) to match trial-level evaluation
    class_weights, _, _ = compute_class_weights_per_trial(
        dataset, train_dataset.indices, num_classes=2, cap=(0.5, 2.0))
    
    # Create data loaders with optimized num_workers
    # Use more workers for faster data loading (especially for spectrogram computation)
    import os
    max_workers = min(8, os.cpu_count() or 4)  # Use up to 8 workers, but not more than CPU cores
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=max_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=max_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=max_workers, pin_memory=True, persistent_workers=True)
    
    # Create model
    # Determine input dimensions from time-frequency transform
    # Updated: 8 frequency bands (was 5), 64/128 time frames (was 32/64)
    window_seconds = window_size / dataset.sampling_rate
    input_time = 128 if window_seconds >= 8.0 else 64
    input_freq = 8  # Updated: 8 frequency bands (Delta, Theta, Alpha-low, Alpha-high, Beta-low, Beta-mid, Beta-high, Gamma)
    
    model = CNNLOCModel(
        input_channels=dataset.n_channels,
        input_time=input_time,
        input_freq=input_freq,
        num_classes=2,
        dropout_rate=dropout_rate
    )
    
    # Create trainer with unique output directory
    exp_output_dir = Path(output_dir) / f"window_{window_size}samples"
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = CNNLOCTrainer(
        model=model,
        device=device,
        output_dir=str(exp_output_dir)
    )
    
    # Train model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        diagnostic_baseline=diagnostic_baseline,
        use_trial_level_loss=use_trial_level_loss,
        class_weights=class_weights
    )
    
    # Test model (with validation for threshold tuning)
    test_metrics = trainer.test(dataset, test_dataset, test_loader, 
                                val_subset=val_dataset, val_loader=val_loader)
    
    return {
        'window_size_samples': window_size,
        'window_size_seconds': window_size / sampling_rate,
        'n_windows': len(dataset),
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        'accuracy': float(test_metrics['accuracy']),
        'roc_auc': test_metrics['roc_auc'] if test_metrics['roc_auc'] is not None else None,
        'precision': test_metrics.get('precision'),
        'recall': test_metrics.get('recall'),
        'f1_score': test_metrics.get('f1_score'),
        'f1_macro': test_metrics.get('f1_macro'),
        'balanced_accuracy': test_metrics.get('balanced_accuracy'),
        'best_val_acc': float(test_metrics['best_val_acc']),
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'dropout_rate': dropout_rate
    }


def sweep_window_sizes(tfrecord_dir: str, window_sizes_seconds: List[float],
                       overlap: float, batch_size: int, num_epochs: int,
                       learning_rate: float, dropout_rate: float,
                       output_dir: str, sampling_rate: int = 64) -> Dict:
    """
    Sweep window sizes and return results.
    
    Note: The time-frequency transform always outputs (channels=66, time_frames=32, freq_bins=4)
    regardless of window size. Different window sizes change the temporal averaging per frame
    (more samples per frame for larger windows), not the output resolution.
    """
    
    all_results = []
    window_sizes_samples = [int(ws * sampling_rate) for ws in window_sizes_seconds]
    
    print("\n" + "="*80)
    print("WINDOW SIZE SWEEP: Representative sizes")
    print("="*80)
    print(f"Testing {len(window_sizes_seconds)} window sizes...")
    print(f"Window sizes (seconds): {window_sizes_seconds}")
    print(f"Window sizes (samples): {window_sizes_samples}")
    
    for i, (ws_sec, ws_samples) in enumerate(zip(window_sizes_seconds, window_sizes_samples), 1):
        print(f"\n{'='*80}")
        print(f"Experiment {i}/{len(window_sizes_seconds)}: Window Size = {ws_sec:.1f}s ({ws_samples} samples)")
        print(f"{'='*80}")
        
        try:
            result = run_single_experiment(
                tfrecord_dir=tfrecord_dir,
                window_size=ws_samples,
                overlap=overlap,
                batch_size=batch_size,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                dropout_rate=dropout_rate,
                output_dir=output_dir,
                sampling_rate=sampling_rate,
                split_by='subject',
                seed=42
            )
            all_results.append(result)
            
            print(f"\n✓ Completed: Window {ws_sec:.1f}s")
            print(f"  Accuracy: {result['accuracy']:.4f}")
            print(f"  ROC-AUC: {result['roc_auc']:.4f}" if result['roc_auc'] is not None else "  ROC-AUC: N/A")
            print(f"  F1-Score: {result['f1_score']:.4f}" if result['f1_score'] is not None else "  F1-Score: N/A")
            
        except Exception as e:
            print(f"\n✗ Failed: Window {ws_sec:.1f}s - {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'window_size_samples': ws_samples,
                'window_size_seconds': ws_sec,
                'error': str(e)
            })
    
    # Create summary table
    print("\n" + "="*80)
    print("WINDOW SIZE SWEEP RESULTS SUMMARY")
    print("="*80)
    print(f"{'Window (s)':<12} {'Samples':<10} {'Accuracy':<12} {'ROC-AUC':<12} {'F1-Score':<12} {'Val Acc':<12}")
    print("-" * 80)
    
    for result in all_results:
        if 'error' not in result:
            ws_sec = result['window_size_seconds']
            ws_samples = result['window_size_samples']
            acc = result['accuracy']
            roc = result['roc_auc'] if result['roc_auc'] is not None else float('nan')
            f1 = result['f1_score'] if result['f1_score'] is not None else float('nan')
            val_acc = result['best_val_acc']
            
            roc_str = f"{roc:.4f}" if not np.isnan(roc) else "N/A"
            f1_str = f"{f1:.4f}" if not np.isnan(f1) else "N/A"
            
            print(f"{ws_sec:<12.1f} {ws_samples:<10} {acc:<12.4f} {roc_str:<12} {f1_str:<12} {val_acc:<12.4f}")
        else:
            print(f"{result['window_size_seconds']:<12.1f} {result['window_size_samples']:<10} ERROR: {result['error']}")
    
    # Save results
    results_file = Path(output_dir) / 'window_size_sweep_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    # Find best window size (use validation accuracy, not test)
    valid_results = [r for r in all_results if 'error' not in r and 'best_val_acc' in r]
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['best_val_acc'])
        print(f"\n✓ Best Window Size: {best_result['window_size_seconds']:.1f}s ({best_result['window_size_samples']} samples)")
        print(f"  Accuracy: {best_result['accuracy']:.4f}")
        print(f"  ROC-AUC: {best_result['roc_auc']:.4f}" if best_result['roc_auc'] is not None else "  ROC-AUC: N/A")
    
    return {'results': all_results, 'summary_file': str(results_file)}


def test_64sample_temporal_integration(tfrecord_dir: str, integration_periods_seconds: List[float],
                                       overlap: float, batch_size: int, num_epochs: int,
                                       learning_rate: float, dropout_rate: float,
                                       output_dir: str, sampling_rate: int = 64) -> Dict:
    """
    Test 64-sample (1 second) windows with temporal integration over different time periods.
    
    This function:
    1. Trains a model on 64-sample windows (1 second)
    2. Tests by aggregating predictions over different time periods (1s, 2s, 3s, etc.)
    3. Uses majority voting to combine predictions from multiple consecutive 1-second windows
    """
    base_window_size = 64  # Always use 64 samples (1 second)
    
    print("\n" + "="*80)
    print("64-SAMPLE TEMPORAL INTEGRATION TEST")
    print("="*80)
    print(f"Base window size: {base_window_size} samples (1.0s)")
    print(f"Testing temporal integration periods: {integration_periods_seconds} seconds")
    print(f"Integration periods (samples): {[int(t * sampling_rate) for t in integration_periods_seconds]}")
    
    # Step 1: Train model on 64-sample windows
    print(f"\n{'='*80}")
    print("Step 1: Training model on 64-sample windows")
    print(f"{'='*80}")
    
    train_result = run_single_experiment(
        tfrecord_dir=tfrecord_dir,
        window_size=base_window_size,
        overlap=overlap,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        output_dir=output_dir,
        sampling_rate=sampling_rate,
        split_by='subject',
        seed=42
    )
    
    # Load the trained model
    model_output_dir = Path(output_dir) / f"window_{base_window_size}samples"
    best_model_path = model_output_dir / "best_model.pth"
    
    if not best_model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {best_model_path}")
    
    # Step 2: Test with different temporal integration periods
    print(f"\n{'='*80}")
    print("Step 2: Testing with different temporal integration periods")
    print(f"{'='*80}")
    
    # Create dataset for testing
    dataset = FULCNNLOCDataset(
        tfrecord_dir=tfrecord_dir,
        window_size=base_window_size,
        overlap=overlap,
        transform_eeg=True
    )
    
    # Split dataset by subject (subject-independent evaluation - recommended)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, split_by='subject')
    
    # Create test loader with batch_size=1 to preserve temporal order
    import os
    max_workers = min(8, os.cpu_count() or 4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                            num_workers=max_workers, pin_memory=True, persistent_workers=True)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Determine input dimensions from time-frequency transform
    window_seconds = base_window_size / dataset.sampling_rate
    input_time = 128 if window_seconds >= 8.0 else 64
    input_freq = 8  # Updated: 8 frequency bands
    model = CNNLOCModel(
        input_channels=dataset.n_channels,
        input_time=input_time,
        input_freq=input_freq,
        num_classes=2,
        dropout_rate=dropout_rate
    )
    
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Precompute mappings for efficient lookup (O(N) instead of O(N²))
    test_indices_set = set(test_dataset.indices)
    
    # Build trial -> sorted window indices map for test set
    trial_to_windows = {}
    for window_idx in test_dataset.indices:
        trial_idx = dataset.windows[window_idx]['trial_idx']
        if trial_idx not in trial_to_windows:
            trial_to_windows[trial_idx] = []
        trial_to_windows[trial_idx].append(window_idx)
    
    # Sort windows within each trial by start_sample (true temporal order)
    # This is O(N log N) per trial, much better than O(N²) with index lookups
    for trial_idx in trial_to_windows:
        trial_to_windows[trial_idx].sort(key=lambda w_idx: dataset.windows[w_idx]['start_sample'])
    
    # Build window_idx -> position_in_trial dict
    window_to_position = {}
    for trial_idx, window_indices in trial_to_windows.items():
        for pos, window_idx in enumerate(window_indices):
            window_to_position[window_idx] = pos
    
    # Get all predictions and targets from test set, preserving trial information
    window_data = []
    
    with torch.no_grad():
        for window_idx, batch_data in enumerate(tqdm(test_loader, desc="Getting base predictions")):
            # Handle both 2-value and 3-value returns
            if len(batch_data) == 3:
                data, target, _ = batch_data  # Ignore trial_indices
            else:
                data, target = batch_data
            data, target = data.to(device), target.to(device)
            target = target.view(-1)
            
            if target.size(0) == 0:
                continue
            
            # Get the actual window index in the test dataset
            actual_window_idx = test_dataset.indices[window_idx]
            window_info = dataset.windows[actual_window_idx]
            trial_idx = window_info['trial_idx']
            
            # Get position using precomputed mapping (O(1) lookup)
            window_idx_in_trial = window_to_position[actual_window_idx]
            
            output = model(data)
            probabilities = F.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            
            window_data.append({
                'trial_idx': trial_idx,
                'window_idx_in_trial': window_idx_in_trial,
                'prediction': pred.cpu().item(),
                'target': target.cpu().item(),
                'probability': probabilities[0, 1].cpu().item()
            })
    
    # Step 3: Test different temporal integration periods (within trials only)
    all_results = []
    
    for integration_seconds in integration_periods_seconds:
        integration_samples = int(integration_seconds * sampling_rate)
        num_windows = integration_samples // base_window_size
        
        if num_windows < 1:
            num_windows = 1
        
        print(f"\nTesting {integration_seconds}s integration ({num_windows} windows of 1s each)...")
        print(f"  Note: Sliding integration (stride=1) for AAD: cumulative tracking, not block-wise.")
        
        # Group windows by trial
        trials_dict = {}
        for wd in window_data:
            tidx = wd['trial_idx']
            if tidx not in trials_dict:
                trials_dict[tidx] = []
            trials_dict[tidx].append(wd)
        
        # Sort windows within each trial by window_idx_in_trial
        for tidx in trials_dict:
            trials_dict[tidx].sort(key=lambda x: x['window_idx_in_trial'])
        
        # Aggregate predictions within each trial using SLIDING integration (stride=1).
        # AAD is cumulative; block-wise (stride=num_windows) breaks temporal continuity.
        aggregated_predictions = []
        aggregated_targets = []
        aggregated_probabilities = []
        
        stride = 1  # Sliding: compare 1s, 2s, 4s, 8s, 16s with monotonic improvement if model works
        
        for trial_idx, trial_windows in trials_dict.items():
            if len(trial_windows) < num_windows:
                # Skip trials with too few windows
                continue
            
            # Sliding windows within this trial
            for i in range(0, len(trial_windows) - num_windows + 1, stride):
                window_group = trial_windows[i:i+num_windows]
                
                window_preds = [w['prediction'] for w in window_group]
                window_probs = [w['probability'] for w in window_group]
                window_target = window_group[num_windows // 2]['target']  # Use middle target
                
                # Majority voting for prediction
                pred_counts = np.bincount(np.array(window_preds).astype(int), minlength=2)
                aggregated_pred = np.argmax(pred_counts)
                
                # Average probability
                aggregated_prob = np.mean(window_probs)
                
                aggregated_predictions.append(aggregated_pred)
                aggregated_targets.append(window_target)
                aggregated_probabilities.append(aggregated_prob)
        
        if len(aggregated_predictions) > 0:
            agg_preds = np.array(aggregated_predictions)
            agg_targets = np.array(aggregated_targets)
            agg_probs = np.array(aggregated_probabilities)
            
            # Calculate metrics
            accuracy = accuracy_score(agg_targets, agg_preds)
            
            try:
                unique_targets = np.unique(agg_targets)
                if len(unique_targets) < 2:
                    roc_auc = float('nan')
                else:
                    roc_auc = roc_auc_score(agg_targets, agg_probs)
            except:
                roc_auc = float('nan')
            
            try:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    agg_targets, agg_preds, average='binary', zero_division=0)
                balanced_acc = balanced_accuracy_score(agg_targets, agg_preds)
            except:
                precision = recall = f1 = balanced_acc = float('nan')
            
            result = {
                'integration_period_seconds': integration_seconds,
                'integration_period_samples': integration_samples,
                'num_windows_integrated': num_windows,
                'base_window_size': base_window_size,
                'accuracy': float(accuracy),
                'roc_auc': roc_auc if not np.isnan(roc_auc) else None,
                'precision': float(precision) if not np.isnan(precision) else None,
                'recall': float(recall) if not np.isnan(recall) else None,
                'f1_score': float(f1) if not np.isnan(f1) else None,
                'balanced_accuracy': float(balanced_acc) if not np.isnan(balanced_acc) else None,
                'n_predictions': len(aggregated_predictions)
            }
            
            all_results.append(result)
            
            roc_str = f"{roc_auc:.4f}" if not np.isnan(roc_auc) else "N/A"
            f1_str = f"{f1:.4f}" if not np.isnan(f1) else "N/A"
            print(f"  Accuracy: {accuracy:.4f}, ROC-AUC: {roc_str}, F1: {f1_str}")
        else:
            print(f"  No valid predictions for {integration_seconds}s integration")
            all_results.append({
                'integration_period_seconds': integration_seconds,
                'error': 'No valid predictions'
            })
    
    # Create summary table
    print("\n" + "="*80)
    print("TEMPORAL INTEGRATION RESULTS SUMMARY")
    print("="*80)
    print(f"{'Integration (s)':<18} {'Windows':<10} {'Accuracy':<12} {'ROC-AUC':<12} {'F1-Score':<12}")
    print("-" * 80)
    
    for result in all_results:
        if 'error' not in result:
            int_sec = result['integration_period_seconds']
            n_windows = result['num_windows_integrated']
            acc = result['accuracy']
            roc = result['roc_auc'] if result['roc_auc'] is not None else float('nan')
            f1 = result['f1_score'] if result['f1_score'] is not None else float('nan')
            
            roc_str = f"{roc:.4f}" if not np.isnan(roc) else "N/A"
            f1_str = f"{f1:.4f}" if not np.isnan(f1) else "N/A"
            
            print(f"{int_sec:<18.1f} {n_windows:<10} {acc:<12.4f} {roc_str:<12} {f1_str:<12}")
        else:
            print(f"{result['integration_period_seconds']:<18.1f} ERROR: {result['error']}")
    
    # Save results
    results_file = Path(output_dir) / 'temporal_integration_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    # Find best integration period (use validation accuracy, not test)
    valid_results = [r for r in all_results if 'error' not in r and 'best_val_acc' in r]
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['best_val_acc'])
        print(f"\n✓ Best Integration Period: {best_result['integration_period_seconds']:.1f}s ({best_result['num_windows_integrated']} windows)")
        print(f"  Accuracy: {best_result['accuracy']:.4f}")
        print(f"  ROC-AUC: {best_result['roc_auc']:.4f}" if best_result['roc_auc'] is not None else "  ROC-AUC: N/A")
    
    return {'results': all_results, 'summary_file': str(results_file)}


def tune_hyperparameters(tfrecord_dir: str, window_size: int, overlap: float,
                         output_dir: str, sampling_rate: int = 64) -> Dict:
    """Tune hyperparameters for best performance."""
    
    # Reduced to 5 strategic experiments for faster tuning
    # Selected combinations cover key hyperparameter ranges
    experiments = [
        {'learning_rate': 1e-3, 'batch_size': 32, 'dropout_rate': 0.2},
        {'learning_rate': 1e-3, 'batch_size': 32, 'dropout_rate': 0.3},
        {'learning_rate': 2e-3, 'batch_size': 32, 'dropout_rate': 0.2},
        {'learning_rate': 5e-3, 'batch_size': 32, 'dropout_rate': 0.3},
        {'learning_rate': 1e-3, 'batch_size': 16, 'dropout_rate': 0.3},
    ]
    num_epochs = 30  # Reduced for faster tuning
    
    all_results = []
    total_experiments = len(experiments)
    exp_num = 0
    
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING")
    print("="*80)
    print(f"Window size: {window_size} samples ({window_size/sampling_rate:.1f}s)")
    print(f"Total experiments: {total_experiments}")
    print(f"Experiments: {experiments}")
    
    for exp_config in experiments:
        exp_num += 1
        lr = exp_config['learning_rate']
        bs = exp_config['batch_size']
        dr = exp_config['dropout_rate']
        
        print(f"\n{'='*80}")
        print(f"Experiment {exp_num}/{total_experiments}")
        print(f"  Learning Rate: {lr}")
        print(f"  Batch Size: {bs}")
        print(f"  Dropout Rate: {dr}")
        print(f"{'='*80}")
        
        try:
            result = run_single_experiment(
                tfrecord_dir=tfrecord_dir,
                window_size=window_size,
                overlap=overlap,
                batch_size=bs,
                num_epochs=num_epochs,
                learning_rate=lr,
                dropout_rate=dr,
                output_dir=output_dir,
                sampling_rate=sampling_rate,
                split_by='subject',
                seed=42
            )
            all_results.append(result)
            
            print(f"\n✓ Completed")
            print(f"  Accuracy: {result['accuracy']:.4f}")
            print(f"  ROC-AUC: {result['roc_auc']:.4f}" if result['roc_auc'] is not None else "  ROC-AUC: N/A")
            print(f"  F1-Score: {result['f1_score']:.4f}" if result['f1_score'] is not None else "  F1-Score: N/A")
            
        except Exception as e:
            print(f"\n✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'learning_rate': lr,
                'batch_size': bs,
                'dropout_rate': dr,
                'error': str(e)
            })
    
    # Create summary table
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING RESULTS SUMMARY")
    print("="*80)
    print(f"{'LR':<10} {'BS':<8} {'DR':<8} {'Accuracy':<12} {'ROC-AUC':<12} {'F1-Score':<12} {'Val Acc':<12}")
    print("-" * 80)
    
    for result in all_results:
        if 'error' not in result:
            lr = result['learning_rate']
            bs = result['batch_size']
            dr = result['dropout_rate']
            acc = result['accuracy']
            roc = result['roc_auc'] if result['roc_auc'] is not None else float('nan')
            f1 = result['f1_score'] if result['f1_score'] is not None else float('nan')
            val_acc = result['best_val_acc']
            
            roc_str = f"{roc:.4f}" if not np.isnan(roc) else "N/A"
            f1_str = f"{f1:.4f}" if not np.isnan(f1) else "N/A"
            
            print(f"{lr:<10.0e} {bs:<8} {dr:<8.2f} {acc:<12.4f} {roc_str:<12} {f1_str:<12} {val_acc:<12.4f}")
        else:
            print(f"{result['learning_rate']:<10.0e} {result['batch_size']:<8} {result['dropout_rate']:<8.2f} ERROR: {result['error']}")
    
    # Save results
    results_file = Path(output_dir) / 'hyperparameter_tuning_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    # Find best hyperparameters: prefer runs that did not collapse (F1 > 0 or ROC-AUC > 0.52)
    valid_results = [r for r in all_results if 'error' not in r and 'best_val_acc' in r]
    if valid_results:
        non_collapsed = [r for r in valid_results if (r.get('f1_score') or 0) > 0 or (r.get('roc_auc') or 0) > 0.52]
        candidates = non_collapsed if non_collapsed else valid_results
        best_result = max(candidates, key=lambda x: (x['best_val_acc'], x.get('roc_auc') or 0))
        if not non_collapsed and valid_results:
            print(f"\n⚠ All runs collapsed (F1=0 or ROC-AUC ≤ 0.52). Best by val acc (may generalize poorly):")
        print(f"\n✓ Best Hyperparameters (selected by validation accuracy, excluding collapsed when possible):")
        print(f"  Learning Rate: {best_result['learning_rate']:.0e}")
        print(f"  Batch Size: {best_result['batch_size']}")
        print(f"  Dropout Rate: {best_result['dropout_rate']:.2f}")
        print(f"  Val Accuracy: {best_result['best_val_acc']:.4f}")
        print(f"  Test Accuracy: {best_result['accuracy']:.4f}")
        print(f"  Test ROC-AUC: {best_result['roc_auc']:.4f}" if best_result['roc_auc'] is not None else "  Test ROC-AUC: N/A")
    
    return {'results': all_results, 'summary_file': str(results_file)}


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fulsang CNN-LOC using FULPRE.py preprocessing')
    parser.add_argument('--tfrecord_dir', type=str, default='fulsang_preprocessed/tfrecords',
                       help='Directory containing TFRecord files from FULPRE.py')
    parser.add_argument('--window_size', type=int, default=512,
                       help='Window size in samples (default: 512 = 8s at 64Hz)')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Window overlap fraction (default: 0.5)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (default: 16 - best from hyperparameter tuning)')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs (default: 50). Reduce to 20-30 for faster iteration.')
    parser.add_argument('--learning_rate', type=float, default=5e-3,
                       help='Learning rate (default: 5e-3 - best from hyperparameter tuning)')
    parser.add_argument('--dropout_rate', type=float, default=0.20,
                       help='Dropout rate (default: 0.20 - best from hyperparameter tuning)')
    parser.add_argument('--output_dir', type=str, default='fulcnnloc_results',
                       help='Output directory for results')
    parser.add_argument('--sweep_window_sizes', action='store_true',
                       help='Sweep window sizes from 1s to 30s (overrides --window_size)')
    parser.add_argument('--tune_hyperparameters', action='store_true',
                       help='Tune hyperparameters (learning rate, batch size, dropout)')
    parser.add_argument('--run_all', action='store_true',
                       help='Run window size sweep AND hyperparameter tuning (comprehensive)')
    parser.add_argument('--test_64sample_temporal', action='store_true',
                       help='Test 64-sample windows with temporal integration over different time periods (1s, 2s, 3s, etc.)')
    parser.add_argument('--integration_periods', type=str, default='1,2,4,8,16',
                       help='Comma-separated integration periods in seconds for sliding integration (default: 1,2,4,8,16)')
    parser.add_argument('--sampling_rate', type=int, default=64,
                       help='Sampling rate in Hz (default: 64)')
    parser.add_argument('--split_by', type=str, default='subject',
                       choices=['subject', 'trial', 'window'],
                       help='Split strategy: subject (recommended, subject-independent), trial, or window (leaky)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--overfit_test', action='store_true',
                       help='Run overfit test on tiny subset (1 subject, 2-4 trials) to verify pipeline can learn')
    parser.add_argument('--shuffle_labels', action='store_true',
                       help='Shuffle labels as baseline test (expected: chance performance)')
    parser.add_argument('--use_leaked_split', action='store_true',
                       help='Use leaked split (within-subject) for diagnostic purposes')
    parser.add_argument('--global_normalization', action='store_true',
                       help='Use global normalization (across all training subjects) instead of per-subject normalization. May preserve task-relevant signal better.')
    parser.add_argument('--use_trial_level_loss', action='store_true',
                       help='Use trial-level loss (average logits per trial). Default: True when --split_by subject.')
    
    args = parser.parse_args()
    # Default use_trial_level_loss=True when split_by is subject (avoids label overcounting)
    use_trial_level_loss = args.use_trial_level_loss or (args.split_by == 'subject')
    args.use_trial_level_loss = use_trial_level_loss
    
    # Set random seed for reproducibility
    seed_everything(args.seed)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Overfit test mode (diagnostic)
    if args.overfit_test:
        result = run_overfit_test(
            tfrecord_dir=args.tfrecord_dir,
            window_size=args.window_size,
            overlap=args.overlap,
            num_epochs=100,
            learning_rate=args.learning_rate,
            sampling_rate=args.sampling_rate,
            seed=args.seed
        )
        print(f"\n✓ Overfit test completed!")
        print(f"  Success: {result['success']}")
        print(f"  Final Accuracy: {result['final_accuracy']:.2f}%")
        print(f"  Final AUC: {result['final_auc']:.4f}" if result['final_auc'] is not None else "  Final AUC: N/A")
        return
    
    # Comprehensive mode: run window size sweep, then tune best window
    if args.run_all:
        print("="*80)
        print("COMPREHENSIVE MODE: Window Size Sweep + Hyperparameter Tuning")
        print("="*80)
        
        # Step 1: Window size sweep (reduced to 5 for faster execution)
        window_sizes_seconds = [1, 8, 16, 24, 30]  # Representative window sizes
        sweep_results = sweep_window_sizes(
            tfrecord_dir=args.tfrecord_dir,
            window_sizes_seconds=window_sizes_seconds,
            overlap=args.overlap,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            dropout_rate=args.dropout_rate,
            output_dir=args.output_dir,
            sampling_rate=args.sampling_rate
        )
        
        # Step 2: Find best window size and tune hyperparameters (use validation, not test)
        valid_results = [r for r in sweep_results['results'] if 'error' not in r and 'best_val_acc' in r]
        if valid_results:
            best_window_result = max(valid_results, key=lambda x: x['best_val_acc'])
            best_window_samples = best_window_result['window_size_samples']
            best_window_seconds = best_window_result['window_size_seconds']
            
            print(f"\n{'='*80}")
            print(f"HYPERPARAMETER TUNING FOR BEST WINDOW SIZE")
            print(f"{'='*80}")
            print(f"Best window size from sweep: {best_window_seconds:.1f}s ({best_window_samples} samples)")
            print(f"Accuracy: {best_window_result['accuracy']:.4f}")
            
            # Tune hyperparameters for the best window size
            tune_results = tune_hyperparameters(
                tfrecord_dir=args.tfrecord_dir,
                window_size=best_window_samples,
                overlap=args.overlap,
                output_dir=args.output_dir,
                sampling_rate=args.sampling_rate
            )
            
            print(f"\n✓ Comprehensive analysis completed!")
            print(f"  Window size sweep results: {sweep_results['summary_file']}")
            print(f"  Hyperparameter tuning results: {tune_results['summary_file']}")
        else:
            print("\n✗ No valid window size results found. Skipping hyperparameter tuning.")
        
        return
    
    # Window size sweep mode
    if args.sweep_window_sizes:
        # Generate window sizes (reduced to 5 for faster execution)
        window_sizes_seconds = [1, 8, 16, 24, 30]  # Representative window sizes
        sweep_results = sweep_window_sizes(
            tfrecord_dir=args.tfrecord_dir,
            window_sizes_seconds=window_sizes_seconds,
            overlap=args.overlap,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            dropout_rate=args.dropout_rate,
            output_dir=args.output_dir,
            sampling_rate=args.sampling_rate
        )
        print(f"\n✓ Window size sweep completed!")
        print(f"  Results saved to {sweep_results['summary_file']}")
        return
    
    # 64-sample temporal integration test mode
    if args.test_64sample_temporal:
        # Parse integration periods
        integration_periods = [float(x.strip()) for x in args.integration_periods.split(',')]
        integration_periods = sorted(integration_periods)  # Sort from smallest to largest
        
        print("="*80)
        print("64-SAMPLE TEMPORAL INTEGRATION TEST")
        print("="*80)
        print(f"Integration periods: {integration_periods} seconds")
        print(f"This will train on 64-sample (1s) windows and test with temporal integration")
        
        temporal_results = test_64sample_temporal_integration(
            tfrecord_dir=args.tfrecord_dir,
            integration_periods_seconds=integration_periods,
            overlap=args.overlap,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            dropout_rate=args.dropout_rate,
            output_dir=args.output_dir,
            sampling_rate=args.sampling_rate
        )
        
        print(f"\n✓ Temporal integration test completed!")
        print(f"  Results saved to {temporal_results['summary_file']}")
        return
    
    # Hyperparameter tuning mode
    if args.tune_hyperparameters:
        tune_results = tune_hyperparameters(
            tfrecord_dir=args.tfrecord_dir,
            window_size=args.window_size,
            overlap=args.overlap,
            output_dir=args.output_dir,
            sampling_rate=args.sampling_rate
        )
        print(f"\n✓ Hyperparameter tuning completed!")
        print(f"  Results saved to {tune_results['summary_file']}")
        return
    
    # Single experiment mode
    print("="*80)
    print("FULCNNLOC - CNN-LOC Algorithm for Fulsang Dataset")
    print("="*80)
    print("\n⚠ PERFORMANCE NOTE:")
    print("  Training can be slow due to:")
    print("  - Spectrogram computation (66 channels × many windows)")
    print("  - Large number of windows created from trials (overlap=0.5 creates ~11 windows/trial)")
    print("  - Multiple experiments if --sweep_window_sizes or --tune_hyperparameters enabled")
    print("  Optimizations applied: increased num_workers, pin_memory, persistent_workers")
    print("  To speed up: reduce --num_epochs (e.g., 20-30), avoid --sweep_window_sizes/--tune_hyperparameters")
    print("="*80)
    print(f"Using CNN-LOC architecture from CombinedCNNLOC.py")
    print(f"  TFRecord directory: {args.tfrecord_dir}")
    print(f"  Window size: {args.window_size} samples ({args.window_size/args.sampling_rate:.1f}s)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.num_epochs}")
    
    # Create dataset
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)
    dataset = FULCNNLOCDataset(
        tfrecord_dir=args.tfrecord_dir,
        window_size=args.window_size,
        overlap=args.overlap,
        transform_eeg=True,
        global_normalization=args.global_normalization,
        subject_wise_normalization=not args.global_normalization  # Use global OR per-subject, not both
    )
    
    # Split dataset
    print("\n" + "="*80)
    print("SPLITTING DATASET")
    print("="*80)
    
    # Handle diagnostic modes
    use_leaked = args.use_leaked_split
    if use_leaked:
        print("⚠ DIAGNOSTIC MODE: Using leaked split (train/val from same subjects)")
        train_dataset, val_dataset, test_dataset = split_dataset(dataset, split_by='trial')
    else:
        train_dataset, val_dataset, test_dataset = split_dataset(dataset, split_by=args.split_by)
    
    # Shuffle labels if requested (diagnostic baseline)
    if args.shuffle_labels:
        print("\n⚠ DIAGNOSTIC MODE: Shuffling labels (baseline test)")
        print("  Expected: Performance should drop to chance (~0.5 AUC)")
        original_labels = [dataset.windows[idx]['label'] for idx in train_dataset.indices]
        shuffled_labels = original_labels.copy()
        np.random.seed(args.seed)
        np.random.shuffle(shuffled_labels)
        for i, idx in enumerate(train_dataset.indices):
            dataset.windows[idx]['label'] = shuffled_labels[i]
        print(f"  Shuffled {len(shuffled_labels)} training labels")
    
    # Class weights per trial (not per window) to match trial-level evaluation
    class_weights, trial_label_counts, total_trials = compute_class_weights_per_trial(
        dataset, train_dataset.indices, num_classes=2, cap=(0.5, 2.0))
    
    print(f"\nClass weights (per-trial inverse frequency, capped 0.5-2.0): {class_weights.numpy()}")
    print(f"  Class 0: {trial_label_counts[0]} trials -> weight {class_weights[0]:.4f}")
    print(f"  Class 1: {trial_label_counts[1]} trials -> weight {class_weights[1]:.4f} (total trials: {total_trials})")
    
    # Create data loaders with optimized num_workers
    import os
    max_workers = min(8, os.cpu_count() or 4)  # Use up to 8 workers, but not more than CPU cores
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=max_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=max_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=max_workers, pin_memory=True, persistent_workers=True)
    
    print(f"\n  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # Create CNN-LOC model
    print("\n" + "="*80)
    print("INITIALIZING CNN-LOC MODEL")
    print("="*80)
    # Determine input dimensions from time-frequency transform
    window_seconds = window_size / dataset.sampling_rate
    input_time = 128 if window_seconds >= 8.0 else 64
    input_freq = 8  # Updated: 8 frequency bands (was 5)
    model = CNNLOCModel(
        input_channels=dataset.n_channels,
        input_time=input_time,  # CNN-LOC time frames (64 or 128)
        input_freq=input_freq,    # CNN-LOC freq bins (8 bands)
        num_classes=2,
        dropout_rate=args.dropout_rate
    )
    
    # Create trainer
    trainer = CNNLOCTrainer(
        model=model,
        device=device,
        output_dir=args.output_dir
    )
    
    # Train model
    print("\n" + "="*80)
    print("TRAINING CNN-LOC MODEL")
    print("="*80)
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        class_weights=class_weights,
        use_trial_level_loss=args.use_trial_level_loss
    )
    
    # Test model (tune threshold on val, report metrics on test only)
    print("\n" + "="*80)
    print("TESTING MODEL")
    print("="*80)
    test_metrics = trainer.test(dataset, test_dataset, test_loader, 
                                val_subset=val_dataset, val_loader=val_loader)
    
    # Save results
    results_json = {
        'accuracy': float(test_metrics['accuracy']),
        'roc_auc': test_metrics['roc_auc'] if test_metrics['roc_auc'] is not None else None,
        'precision': test_metrics.get('precision'),
        'recall': test_metrics.get('recall'),
        'f1_score': test_metrics.get('f1_score'),
        'f1_macro': test_metrics.get('f1_macro'),
        'balanced_accuracy': test_metrics.get('balanced_accuracy'),
        'best_val_acc': float(test_metrics['best_val_acc']),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(Path(args.output_dir) / 'results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✓ Training Complete")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    if test_metrics['roc_auc'] is not None and not np.isnan(test_metrics['roc_auc']):
        print(f"  Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    else:
        print(f"  Test ROC-AUC: N/A (perfect predictions or single class)")
    if 'f1_score' in test_metrics and test_metrics['f1_score'] is not None:
        print(f"  Test F1-Score (binary): {test_metrics['f1_score']:.4f}")
    if 'f1_macro' in test_metrics and test_metrics['f1_macro'] is not None:
        print(f"  Test F1-Score (macro): {test_metrics['f1_macro']:.4f}")
    if 'balanced_accuracy' in test_metrics and test_metrics['balanced_accuracy'] is not None:
        print(f"  Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"\n✓ Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()

