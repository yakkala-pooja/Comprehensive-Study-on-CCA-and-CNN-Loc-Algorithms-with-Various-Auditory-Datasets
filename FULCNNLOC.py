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
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
import tensorflow as tf
from pathlib import Path
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
        channel_att = self.channel_attention(x)
        return x * channel_att


class ResidualBlock(nn.Module):
    """Residual block with attention."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut for residual connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
        self.attention = SpatialTemporalAttention(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attention(out)
        
        out += residual
        out = self.relu(out)
        
        return out


class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale features using different kernel sizes."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(MultiScaleFeatureExtractor, self).__init__()
        
        # Two scales: 1x1 and 3x1
        self.conv1x1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        self.conv3x1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=(3, 1), padding=(1, 0))
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        feat1 = self.conv1x1(x)
        feat3 = self.conv3x1(x)
        
        # Concatenate
        out = torch.cat([feat1, feat3], dim=1)
        out = self.relu(self.bn(out))
        
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
    """Backbone network: attention, residual blocks, multi-scale features."""
    
    def __init__(self, input_channels: int = 66, input_time: int = 32, input_freq: int = 4):
        super(CNNLOCBackbone, self).__init__()
        
        self.input_channels = input_channels
        self.input_time = input_time
        self.input_freq = input_freq
        
        print(f"Building CNN-LOC backbone: channels={input_channels}, time={input_time}, freq={input_freq}")
        
        # Initial multi-scale features
        self.initial_features = MultiScaleFeatureExtractor(input_channels, 32)
        
        # Temporal blocks
        self.temporal_block1 = ResidualBlock(32, 32, stride=1)
        self.temporal_pool1 = nn.MaxPool2d((2, 1), (2, 1))
        
        self.temporal_block2 = ResidualBlock(32, 64, stride=1)
        self.temporal_pool2 = nn.MaxPool2d((2, 1), (2, 1))
        
        # Spatial blocks
        self.spatial_block1 = ResidualBlock(64, 64, stride=1)
        self.spatial_pool1 = nn.MaxPool2d((1, 2), (1, 2))
        
        self.spatial_block2 = ResidualBlock(64, 128, stride=1)
        self.spatial_pool2 = nn.MaxPool2d((1, 2), (1, 2))
        
        # Global attention
        self.global_attention = SpatialTemporalAttention(128)
        
        # Adaptive pooling
        self.adaptive_pooling = AdaptivePooling(output_size=1)
        
        # Calculate output size
        self._calculate_output_size()
    
    def _calculate_output_size(self):
        """Figure out output size by running a dummy input."""
        dummy_input = torch.randn(1, self.input_channels, self.input_time, self.input_freq)
        
        with torch.no_grad():
            x = self.forward(dummy_input)
            self.output_size = x.numel()
    
    def forward(self, x):
        """Forward pass."""
        # Multi-scale features
        x = self.initial_features(x)
        
        # Temporal processing
        x = self.temporal_block1(x)
        x = self.temporal_pool1(x)
        
        x = self.temporal_block2(x)
        x = self.temporal_pool2(x)
        
        # Spatial processing
        x = self.spatial_block1(x)
        x = self.spatial_pool1(x)
        
        x = self.spatial_block2(x)
        x = self.spatial_pool2(x)
        
        # Attention
        x = self.global_attention(x)
        
        # Pool and flatten
        x = self.adaptive_pooling(x)
        x = x.view(x.size(0), -1)
        
        return x


class CNNLOCModel(nn.Module):
    """Full CNN-LOC model: backbone + classifier for EEG attention decoding."""
    
    def __init__(self, input_channels: int = 66, input_time: int = 32, input_freq: int = 4,
                 num_classes: int = 2, dropout_rate: float = 0.3):
        super(CNNLOCModel, self).__init__()
        
        # Create backbone
        self.backbone = CNNLOCBackbone(input_channels, input_time, input_freq)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.backbone.output_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes)
        )
        
        self._initialize_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model created with {n_params:,} parameters")
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
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
                 transform_eeg: bool = True, allow_cross_trial: bool = False):
        self.tfrecord_dir = Path(tfrecord_dir)
        self.mode = mode
        self.window_size = window_size
        self.overlap = overlap
        self.transform_eeg = transform_eeg
        self.allow_cross_trial = allow_cross_trial  # If False, windows stay within trials
        
        # Fulsang dataset parameters
        self.sampling_rate = 64  # Hz
        self.n_channels = 66  # EEG channels
        self.trial_length = 3200  # samples per trial (50 seconds at 64 Hz)
        
        # Load trials (preserving boundaries)
        self.trials = self._load_trials()
        
        # Create windows from trials
        self.windows = self._create_windows_from_trials()
        
        print(f"\nFULCNNLOCDataset initialized:")
        print(f"  Mode: {mode}")
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
        tfrecord_files = list(self.tfrecord_dir.glob("*.tfrecords"))
        
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
        subject_extraction_count = {}  # Track how many times we extract from filename per file
        
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
                        
                        # Fallback: try to extract subject_id from filename if not in TFRecord
                        if subject_id == "unknown":
                            # Use regex to find subject pattern (S followed by digits) in filename
                            # Handles: fulsang_S1.tfrecords, S1_train.tfrecords, S01.tfrecords, etc.
                            filename = tfrecord_file.stem  # Without .tfrecords extension
                            match = re.search(r"(S\d+)", filename, re.IGNORECASE)
                            if match:
                                subject_id = match.group(1).upper()  # Normalize to uppercase
                                # Track extraction (print once per file, not per record)
                                if tfrecord_file.name not in subject_extraction_count:
                                    subject_extraction_count[tfrecord_file.name] = 0
                                subject_extraction_count[tfrecord_file.name] += 1
                        
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
        
        # Report subject extraction from filenames (if any)
        if subject_extraction_count:
            total_extracted = sum(subject_extraction_count.values())
            print(f"  Note: Extracted subject_id from filename for {len(subject_extraction_count)} files ({total_extracted} trials)")
            if 'unknown' in subject_counts:
                print(f"  WARNING: {subject_counts['unknown']} trials still have subject_id='unknown' - subject-level splitting may not work correctly")
        
        return trials
    
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
        step_size = int(self.window_size * (1 - self.overlap))
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
            raise ValueError(
                f"No windows created! Check window_size ({self.window_size}) vs trial_length ({self.trial_length}). "
                f"Total trials: {len(self.trials)}"
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
        Transform EEG window to time-frequency representation (CNN-LOC format).
        
        Note: This is a coarse frequency representation (4 bins) using FFT on chunked frames,
        not a full spectrogram. The output shape is fixed to (channels=66, time_frames=32, freq_bins=4)
        regardless of window_size. Different window sizes change temporal averaging per frame,
        not the output resolution.
        """
        n_samples, n_channels = eeg_window.shape
        
        # Compute spectrogram for each channel
        freq_bins = 4  # Number of frequency bands
        time_frames = 32  # Number of time frames
        
        # Reshape to (channels, time_frames, freq_bins)
        if n_samples >= time_frames:
            samples_per_frame = n_samples // time_frames
            eeg_reshaped = eeg_window[:time_frames * samples_per_frame].reshape(
                time_frames, samples_per_frame, n_channels
            )
            
            # Apply FFT to each frame
            eeg_fft = np.fft.rfft(eeg_reshaped, axis=1)
            eeg_fft = np.abs(eeg_fft)
            
            # Ensure we have at least freq_bins frequency bins
            n_freq_bins_available = eeg_fft.shape[1]
            if n_freq_bins_available < freq_bins:
                # Pad with zeros
                padding = np.zeros((eeg_fft.shape[0], freq_bins - n_freq_bins_available, eeg_fft.shape[2]))
                eeg_fft = np.concatenate([eeg_fft, padding], axis=1)
            
            eeg_fft = eeg_fft[:, :freq_bins, :]
            
            # Reshape to (channels, time_frames, freq_bins)
            eeg_tf = np.transpose(eeg_fft, (2, 0, 1))
        else:
            # Pad if needed
            eeg_tf = np.zeros((n_channels, time_frames, freq_bins), dtype=np.float32)
            eeg_transposed = eeg_window.T  # (n_channels, n_samples)
            eeg_tf[:, :min(n_samples, time_frames), 0] = eeg_transposed[:, :min(n_samples, time_frames)]
        
        return eeg_tf.astype(np.float32)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        trial = self.trials[window['trial_idx']]
        
        # Extract window from trial
        eeg_window = trial['eeg'][window['start_sample']:window['end_sample']]
        
        # Validate shape
        assert eeg_window.shape == (self.window_size, self.n_channels), \
            f"Window shape mismatch: {eeg_window.shape} != ({self.window_size}, {self.n_channels})"
        
        # Preprocess (baseline correction and normalization)
        eeg_window = eeg_window - np.mean(eeg_window, axis=0, keepdims=True)
        std_vals = np.std(eeg_window, axis=0, keepdims=True)
        std_vals = np.where(std_vals == 0, 1.0, std_vals)
        eeg_window = eeg_window / std_vals
        
        # Transform to time-frequency representation
        if self.transform_eeg:
            eeg_tf = self._transform_eeg(eeg_window)
        else:
            # Simple reshape
            eeg_tf = eeg_window.T[:, :, np.newaxis]
        
        # Convert to tensors
        eeg_tensor = torch.FloatTensor(eeg_tf)
        label_tensor = torch.LongTensor([window['label']])
        
        return eeg_tensor, label_tensor


def split_dataset(dataset: FULCNNLOCDataset, train_ratio: float = 0.7, 
                  val_ratio: float = 0.15, split_by: str = 'trial') -> Tuple[Dataset, Dataset, Dataset]:
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
        
        # Random split of subjects
        np.random.seed(42)
        np.random.shuffle(unique_subjects)
        
        n_subjects = len(unique_subjects)
        train_subjects = int(train_ratio * n_subjects)
        val_subjects = int(val_ratio * n_subjects)
        
        train_subject_set = set(unique_subjects[:train_subjects])
        val_subject_set = set(unique_subjects[train_subjects:train_subjects + val_subjects])
        test_subject_set = set(unique_subjects[train_subjects + val_subjects:])
        
        # Create window indices for each split (all windows from subjects in each set)
        train_window_indices = [i for i, w in enumerate(dataset.windows) 
                               if dataset.trials[w['trial_idx']]['subject_id'] in train_subject_set]
        val_window_indices = [i for i, w in enumerate(dataset.windows) 
                             if dataset.trials[w['trial_idx']]['subject_id'] in val_subject_set]
        test_window_indices = [i for i, w in enumerate(dataset.windows) 
                              if dataset.trials[w['trial_idx']]['subject_id'] in test_subject_set]
        
        print(f"\nSplitting by subject (subject-independent evaluation):")
        print(f"  Train: {len(train_subject_set)} subjects, {len(train_window_indices)} windows")
        print(f"  Val: {len(val_subject_set)} subjects, {len(val_window_indices)} windows")
        print(f"  Test: {len(test_subject_set)} subjects, {len(test_window_indices)} windows")
        
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
        
        print(f"\nSplitting by trial:")
        print(f"  Train: {len(train_trial_set)} trials, {len(train_window_indices)} windows")
        print(f"  Val: {len(val_trial_set)} trials, {len(val_window_indices)} windows")
        print(f"  Test: {len(test_trial_set)} trials, {len(test_window_indices)} windows")
        
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
        
        print(f"\nSplitting by window (WARNING: may cause data leakage):")
        print(f"  Train: {len(train_window_indices)} windows")
        print(f"  Val: {len(val_window_indices)} windows")
        print(f"  Test: {len(test_window_indices)} windows")
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
                   criterion: nn.Module, scheduler: Optional[Any] = None) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            scheduler: Optional LR scheduler to step per batch (e.g., OneCycleLR)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0  # Count actual processed batches
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            # Flatten target to 1D: [batch_size, 1] -> [batch_size]
            target = target.view(-1)
            
            # Skip empty batches
            if target.size(0) == 0:
                continue
            
            n_batches += 1  # Count this batch
            
            # Forward
            output = self.model(data)
            loss = criterion(output, target)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Step scheduler per batch (required for OneCycleLR)
            if scheduler is not None:
                scheduler.step()
            
            # Accumulate loss and accuracy
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        if total == 0:
            return float('inf'), 0.0
        
        # Divide by number of processed batches, not len(train_loader)
        avg_loss = total_loss / max(1, n_batches)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0  # Count actual processed batches
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                # Flatten target to 1D: [batch_size, 1] -> [batch_size]
                target = target.view(-1)
                
                # Skip empty batches
                if target.size(0) == 0:
                    continue
                
                n_batches += 1  # Count this batch
                
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        if total == 0:
            return float('inf'), 0.0
        
        # Divide by number of processed batches, not len(val_loader)
        avg_loss = total_loss / max(1, n_batches)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 50, learning_rate: float = 1e-4,
              weight_decay: float = 1e-5, patience: int = 10,
              class_weights: Optional[torch.Tensor] = None):
        """Train the model."""
        
        # Use class weights if provided, otherwise use uniform weights
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            print(f"Using class weights: {class_weights.cpu().numpy()}")
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = OneCycleLR(optimizer, max_lr=learning_rate * 5, 
                              total_steps=num_epochs * len(train_loader), pct_start=0.3)
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion, scheduler)
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
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
            
            if patience_counter >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break
        
        print(f"Training completed. Best validation accuracy: {self.best_val_acc:.4f}%")
        return self.best_val_acc
    
    def test(self, test_loader: DataLoader) -> Dict:
        """Test model and compute metrics."""
        checkpoint = torch.load(self.best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Testing"):
                data, target = data.to(self.device), target.to(self.device)
                # Flatten target to 1D: [batch_size, 1] -> [batch_size]
                target = target.view(-1)
                
                # Skip empty batches
                if target.size(0) == 0:
                    continue
                
                output = self.model(data)
                probabilities = F.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
        
        # Calculate metrics
        preds = np.array(all_predictions)
        targets = np.array(all_targets)
        probs = np.array(all_probabilities)
        
        accuracy = accuracy_score(targets, preds)
        
        # Calculate ROC-AUC with error handling
        try:
            # Check if we have both classes in targets
            unique_targets = np.unique(targets)
            if len(unique_targets) < 2:
                # Only one class present - ROC-AUC is undefined
                roc_auc = float('nan')
                print(f"Warning: Only one class in test set ({unique_targets}), ROC-AUC is undefined")
            else:
                roc_auc = roc_auc_score(targets, probs)
        except Exception as e:
            print(f"Warning: Could not calculate ROC-AUC: {e}")
            roc_auc = float('nan')
        
        # Calculate additional metrics
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='binary', zero_division=0)
            balanced_acc = balanced_accuracy_score(targets, preds)
        except Exception as e:
            print(f"Warning: Could not calculate additional metrics: {e}")
            precision = recall = f1 = balanced_acc = float('nan')
        
        results = {
            'accuracy': accuracy,
            'roc_auc': roc_auc if not np.isnan(roc_auc) else None,
            'precision': float(precision) if not np.isnan(precision) else None,
            'recall': float(recall) if not np.isnan(recall) else None,
            'f1_score': float(f1) if not np.isnan(f1) else None,
            'balanced_accuracy': float(balanced_acc) if not np.isnan(balanced_acc) else None,
            'predictions': preds.tolist(),
            'targets': targets.tolist(),
            'probabilities': probs.tolist(),
            'best_val_acc': self.best_val_acc
        }
        
        return results


def run_single_experiment(tfrecord_dir: str, window_size: int, overlap: float, 
                          batch_size: int, num_epochs: int, learning_rate: float,
                          dropout_rate: float, output_dir: str, 
                          sampling_rate: int = 64) -> Dict:
    """Run a single experiment with given hyperparameters."""
    
    # Create dataset
    dataset = FULCNNLOCDataset(
        tfrecord_dir=tfrecord_dir,
        window_size=window_size,
        overlap=overlap,
        transform_eeg=True
    )
    
    # Split dataset by trial (prevents data leakage)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, split_by='trial')
    
    # Calculate class weights
    train_labels = [dataset.windows[idx]['label'] for idx in train_dataset.indices]
    train_label_counts = np.bincount(train_labels, minlength=2)
    total_samples = len(train_labels)
    
    class_weights = torch.FloatTensor([
        total_samples / (2 * train_label_counts[0]) if train_label_counts[0] > 0 else 1.0,
        total_samples / (2 * train_label_counts[1]) if train_label_counts[1] > 0 else 1.0
    ])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Create model
    model = CNNLOCModel(
        input_channels=dataset.n_channels,
        input_time=32,
        input_freq=4,
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
        class_weights=class_weights
    )
    
    # Test model
    test_metrics = trainer.test(test_loader)
    
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
    print("WINDOW SIZE SWEEP: 1s to 30s")
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
                sampling_rate=sampling_rate
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
    
    # Find best window size
    valid_results = [r for r in all_results if 'error' not in r and 'accuracy' in r]
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['accuracy'])
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
        sampling_rate=sampling_rate
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
    
    # Split dataset by trial (prevents data leakage)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, split_by='trial')
    
    # Create test loader with batch_size=1 to preserve temporal order
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNLOCModel(
        input_channels=dataset.n_channels,
        input_time=32,
        input_freq=4,
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
        for window_idx, (data, target) in enumerate(tqdm(test_loader, desc="Getting base predictions")):
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
        print(f"  Note: Aggregating within trials only to preserve temporal order")
        print(f"  Using stride={num_windows} (non-overlapping) for fair comparison across integration lengths")
        
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
        
        # Aggregate predictions within each trial
        # Use stride=num_windows for non-overlapping windows (fairer comparison)
        aggregated_predictions = []
        aggregated_targets = []
        aggregated_probabilities = []
        
        stride = num_windows  # Non-overlapping for fair comparison
        
        for trial_idx, trial_windows in trials_dict.items():
            if len(trial_windows) < num_windows:
                # Skip trials with too few windows
                continue
            
            # Aggregate within this trial using non-overlapping stride
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
    
    # Find best integration period
    valid_results = [r for r in all_results if 'error' not in r and 'accuracy' in r]
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['accuracy'])
        print(f"\n✓ Best Integration Period: {best_result['integration_period_seconds']:.1f}s ({best_result['num_windows_integrated']} windows)")
        print(f"  Accuracy: {best_result['accuracy']:.4f}")
        print(f"  ROC-AUC: {best_result['roc_auc']:.4f}" if best_result['roc_auc'] is not None else "  ROC-AUC: N/A")
    
    return {'results': all_results, 'summary_file': str(results_file)}


def tune_hyperparameters(tfrecord_dir: str, window_size: int, overlap: float,
                         output_dir: str, sampling_rate: int = 64) -> Dict:
    """Tune hyperparameters for best performance."""
    
    # Define hyperparameter search space
    learning_rates = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]
    batch_sizes = [16, 32, 64]
    dropout_rates = [0.2, 0.3, 0.4, 0.5]
    num_epochs = 30  # Reduced for faster tuning
    
    all_results = []
    total_experiments = len(learning_rates) * len(batch_sizes) * len(dropout_rates)
    exp_num = 0
    
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING")
    print("="*80)
    print(f"Window size: {window_size} samples ({window_size/sampling_rate:.1f}s)")
    print(f"Total experiments: {total_experiments}")
    print(f"Learning rates: {learning_rates}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Dropout rates: {dropout_rates}")
    
    for lr in learning_rates:
        for bs in batch_sizes:
            for dr in dropout_rates:
                exp_num += 1
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
                        sampling_rate=sampling_rate
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
    
    # Find best hyperparameters
    valid_results = [r for r in all_results if 'error' not in r and 'accuracy' in r]
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['accuracy'])
        print(f"\n✓ Best Hyperparameters:")
        print(f"  Learning Rate: {best_result['learning_rate']:.0e}")
        print(f"  Batch Size: {best_result['batch_size']}")
        print(f"  Dropout Rate: {best_result['dropout_rate']:.2f}")
        print(f"  Accuracy: {best_result['accuracy']:.4f}")
        print(f"  ROC-AUC: {best_result['roc_auc']:.4f}" if best_result['roc_auc'] is not None else "  ROC-AUC: N/A")
    
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
                       help='Number of training epochs (default: 50)')
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
    parser.add_argument('--integration_periods', type=str, default='1,2,3,4,5,6,7,8',
                       help='Comma-separated list of integration periods in seconds (default: 1,2,3,4,5,6,7,8)')
    parser.add_argument('--sampling_rate', type=int, default=64,
                       help='Sampling rate in Hz (default: 64)')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Comprehensive mode: run window size sweep, then tune best window
    if args.run_all:
        print("="*80)
        print("COMPREHENSIVE MODE: Window Size Sweep + Hyperparameter Tuning")
        print("="*80)
        
        # Step 1: Window size sweep
        window_sizes_seconds = list(range(1, 31))  # 1s to 30s
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
        
        # Step 2: Find best window size and tune hyperparameters
        valid_results = [r for r in sweep_results['results'] if 'error' not in r and 'accuracy' in r]
        if valid_results:
            best_window_result = max(valid_results, key=lambda x: x['accuracy'])
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
        # Generate window sizes from 1s to 30s
        window_sizes_seconds = list(range(1, 31))  # 1s to 30s
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
        transform_eeg=True
    )
    
    # Split dataset
    print("\n" + "="*80)
    print("SPLITTING DATASET")
    print("="*80)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)
    
    # Calculate class weights from training data (inverse frequency weighting)
    train_labels = [dataset.windows[idx]['label'] for idx in train_dataset.indices]
    train_label_counts = np.bincount(train_labels, minlength=2)
    total_samples = len(train_labels)
    
    # Compute class weights: weight = total_samples / (num_classes * class_count)
    # This gives more weight to underrepresented classes
    class_weights = torch.FloatTensor([
        total_samples / (2 * train_label_counts[0]) if train_label_counts[0] > 0 else 1.0,
        total_samples / (2 * train_label_counts[1]) if train_label_counts[1] > 0 else 1.0
    ])
    
    print(f"\nClass weights (inverse frequency): {class_weights.numpy()}")
    print(f"  Class 0: {train_label_counts[0]} samples -> weight {class_weights[0]:.4f}")
    print(f"  Class 1: {train_label_counts[1]} samples -> weight {class_weights[1]:.4f}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f"\n  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # Create CNN-LOC model
    print("\n" + "="*80)
    print("INITIALIZING CNN-LOC MODEL")
    print("="*80)
    model = CNNLOCModel(
        input_channels=dataset.n_channels,
        input_time=32,  # CNN-LOC time frames
        input_freq=4,    # CNN-LOC freq bins
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
        class_weights=class_weights
    )
    
    # Test model
    print("\n" + "="*80)
    print("TESTING MODEL")
    print("="*80)
    test_metrics = trainer.test(test_loader)
    
    # Save results
    results_json = {
        'accuracy': float(test_metrics['accuracy']),
        'roc_auc': test_metrics['roc_auc'] if test_metrics['roc_auc'] is not None else None,
        'precision': test_metrics.get('precision'),
        'recall': test_metrics.get('recall'),
        'f1_score': test_metrics.get('f1_score'),
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
        print(f"  Test F1-Score: {test_metrics['f1_score']:.4f}")
    if 'balanced_accuracy' in test_metrics and test_metrics['balanced_accuracy'] is not None:
        print(f"  Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"\n✓ Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()

