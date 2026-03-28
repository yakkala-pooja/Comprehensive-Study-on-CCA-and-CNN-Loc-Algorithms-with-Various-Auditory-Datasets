#!/usr/bin/env python3
"""
CombinedCNNLOCSub - CNN-LOC Algorithm for Combined Das and Fulsang Dataset
with Subject-Level Splitting

This script implements CNN-LOC (Convolutional Neural Network - Localization) for the 
combined dataset using the FULCNN architecture, adapted for CombinedDataset.
Uses subject-level splitting to prevent data leakage.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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
from scipy.signal import butter, filtfilt

# Import combined dataset
from CombinedDataset import CombinedDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================================
# FULCNN Architecture Components (extracted from FULCNN.py)
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
    
    def __init__(self, input_channels: int = 64, input_time: int = 32, input_freq: int = 4):
        super(CNNLOCBackbone, self).__init__()
        
        self.input_channels = input_channels
        self.input_time = input_time
        self.input_freq = input_freq
        
        print(f"Building CNN-LOC backbone: channels={input_channels}, time={input_time}, freq={input_freq} (reduced filters for better subject generalization)")
        
        # Initial multi-scale features (reduced: 32->16)
        self.initial_features = MultiScaleFeatureExtractor(input_channels, 16)
        
        # Temporal blocks (16->16, 16->32)
        self.temporal_block1 = ResidualBlock(16, 16, stride=1)
        self.temporal_pool1 = nn.MaxPool2d((2, 1), (2, 1))
        
        self.temporal_block2 = ResidualBlock(16, 32, stride=1)
        self.temporal_pool2 = nn.MaxPool2d((2, 1), (2, 1))
        
        # Spatial blocks (32->32, 32->64)
        self.spatial_block1 = ResidualBlock(32, 32, stride=1)
        self.spatial_pool1 = nn.MaxPool2d((1, 2), (1, 2))
        
        self.spatial_block2 = ResidualBlock(32, 64, stride=1)
        self.spatial_pool2 = nn.MaxPool2d((1, 2), (1, 2))
        
        # Global attention (64 channels)
        self.global_attention = SpatialTemporalAttention(64)
        
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
    
    def __init__(self, input_channels: int = 64, input_time: int = 32, input_freq: int = 4,
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

class CombinedCNNLOCDataset(Dataset):
    """
    PyTorch Dataset wrapper for CombinedDataset for CNN-LOC training.
    Converts numpy arrays to PyTorch tensors compatible with CNN-LOC.
    """
    
    def __init__(self, combined_dataset: CombinedDataset, mode: str = 'train', transform_eeg: bool = True,
                 bandpass_low: float = 0, bandpass_high: float = 0):
        self.combined_dataset = combined_dataset
        self.mode = mode
        self.transform_eeg = transform_eeg
        self.window_size = combined_dataset.window_size
        self.sampling_rate = combined_dataset.sampling_rate
        self.n_channels = combined_dataset.n_channels
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        
        # Get window indices (now includes grouping info: start, end, label, subject_id, trial_idx, dataset)
        self.window_indices = combined_dataset.get_window_indices()
        
        # Extract grouping info for subject-based splitting
        # Format: (start_idx, end_idx, label, subject_id, trial_idx, dataset)
        self.window_groups = [(meta[3], meta[4], meta[5]) for meta in self.window_indices] if len(self.window_indices) > 0 and len(self.window_indices[0]) > 3 else None
        
        # Per-subject normalization (avoids leaking subject identity; critical for generalization)
        self.subject_stats = self._compute_per_subject_stats()
        
        print(f"\nCombinedCNNLOCDataset initialized:")
        print(f"  Mode: {mode}")
        print(f"  Total windows: {len(self.window_indices)}")
        print(f"  Window size: {self.window_size} samples")
        print(f"  Sampling rate: {self.sampling_rate} Hz")
        print(f"  Channels: {self.n_channels}")
        print(f"  Transform EEG: {transform_eeg}")
        if self.window_groups:
            unique_groups = len(set(self.window_groups))
            unique_subjects = len(set(meta[3] for meta in self.window_indices))
            print(f"  Unique subjects: {unique_subjects}")
            print(f"  Unique groups (subject+trial+dataset): {unique_groups}")
            print(f"  ✓ Windows include subject_id for proper subject-level splitting")
        print(f"  Per-subject normalization: enabled (avoids subject leakage)")
    
    def _compute_per_subject_stats(self) -> Dict:
        """Compute mean and std per channel per subject over all that subject's windows (Welford online)."""
        from collections import defaultdict
        subject_ranges = defaultdict(list)
        for win in self.window_indices:
            if len(win) >= 4:
                subject_ranges[win[3]].append((win[0], win[1]))
        subject_stats = {}
        eeg = self.combined_dataset.eeg_data
        n_ch = self.n_channels
        for subject_id, ranges in subject_ranges.items():
            n_samp = 0
            mean = np.zeros(n_ch, dtype=np.float64)
            m2 = np.zeros(n_ch, dtype=np.float64)
            for start, end in ranges:
                chunk = eeg[start:end].astype(np.float64)
                for i in range(chunk.shape[0]):
                    n_samp += 1
                    delta = chunk[i] - mean
                    mean += delta / n_samp
                    delta2 = chunk[i] - mean
                    m2 += delta * delta2
            if n_samp < 2:
                std = np.ones(n_ch, dtype=np.float32)
            else:
                std = np.sqrt(m2 / (n_samp - 1))
                std = np.where(std == 0, 1.0, std)
            subject_stats[subject_id] = (mean.astype(np.float32), std.astype(np.float32))
        return subject_stats
    
    def _transform_eeg(self, eeg_window: np.ndarray) -> np.ndarray:
        """Transform EEG window to time-frequency representation (FULCNN format)."""
        n_samples, n_channels = eeg_window.shape
        
        # Compute spectrogram for each channel (same as FULCNN)
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
                # Pad with zeros to reach freq_bins
                padding = np.zeros((eeg_fft.shape[0], freq_bins - n_freq_bins_available, eeg_fft.shape[2]))
                eeg_fft = np.concatenate([eeg_fft, padding], axis=1)
            
            eeg_fft = eeg_fft[:, :freq_bins, :]
            
            # Reshape to (channels, time_frames, freq_bins)
            eeg_tf = np.transpose(eeg_fft, (2, 0, 1))
        else:
            # Pad if needed: repeat the signal to reach time_frames
            # First, pad the input signal to have at least time_frames samples
            if n_samples == 0:
                # Edge case: empty window
                eeg_tf = np.zeros((n_channels, time_frames, freq_bins), dtype=np.float32)
            else:
                # Repeat the signal to reach time_frames
                repeat_factor = (time_frames // n_samples) + 1
                eeg_padded = np.tile(eeg_window, (repeat_factor, 1))[:time_frames]
                
                # Now apply the same transformation as above
                samples_per_frame = len(eeg_padded) // time_frames
                if samples_per_frame > 0:
                    eeg_reshaped = eeg_padded[:time_frames * samples_per_frame].reshape(
                        time_frames, samples_per_frame, n_channels
                    )
                    eeg_fft = np.fft.rfft(eeg_reshaped, axis=1)
                    eeg_fft = np.abs(eeg_fft)
                    
                    # Ensure we have at least freq_bins frequency bins
                    n_freq_bins_available = eeg_fft.shape[1]
                    if n_freq_bins_available < freq_bins:
                        # Pad with zeros to reach freq_bins
                        padding = np.zeros((eeg_fft.shape[0], freq_bins - n_freq_bins_available, eeg_fft.shape[2]))
                        eeg_fft = np.concatenate([eeg_fft, padding], axis=1)
                    
                    eeg_fft = eeg_fft[:, :freq_bins, :]
                    eeg_tf = np.transpose(eeg_fft, (2, 0, 1))
                else:
                    # Fallback: create a simple representation
                    eeg_tf = np.zeros((n_channels, time_frames, freq_bins), dtype=np.float32)
                    # Transpose eeg_window to (channels, samples) and pad
                    eeg_transposed = eeg_window.T  # (n_channels, n_samples)
                    eeg_tf[:, :min(n_samples, time_frames), 0] = eeg_transposed[:, :min(n_samples, time_frames)]
        
        return eeg_tf.astype(np.float32)
    
    def __len__(self):
        return len(self.window_indices)
    
    def __getitem__(self, idx):
        # Window indices format: (start_idx, end_idx, label, subject_id, trial_idx, dataset)
        window_info = self.window_indices[idx]
        start_idx, end_idx, label = window_info[0], window_info[1], window_info[2]
        
        # Extract window
        eeg_window = self.combined_dataset.eeg_data[start_idx:end_idx].astype(np.float64)
        
        # Optional bandpass filter (e.g. 1–40 Hz for speech AAD)
        if self.bandpass_high > self.bandpass_low and self.bandpass_high > 0:
            nyq = 0.5 * self.sampling_rate
            low = max(0.5, self.bandpass_low) / nyq
            high = min(nyq - 0.5, self.bandpass_high) / nyq
            if low < high:
                b, a = butter(4, [low, high], btype='band')
                eeg_window = filtfilt(b, a, eeg_window, axis=0).astype(np.float32)
            else:
                eeg_window = eeg_window.astype(np.float32)
        else:
            eeg_window = eeg_window.astype(np.float32)
        
        # Per-subject normalization (not per-window or global — avoids subject identity leakage)
        subject_id = window_info[3] if len(window_info) >= 4 else None
        if subject_id is not None and subject_id in self.subject_stats:
            subj_mean, subj_std = self.subject_stats[subject_id]
            eeg_window = eeg_window - subj_mean[np.newaxis, :]
            eeg_window = eeg_window / subj_std[np.newaxis, :]
        else:
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
        label_tensor = torch.LongTensor([label])
        
        return eeg_tensor, label_tensor


def split_dataset_by_window(dataset: CombinedCNNLOCDataset, 
                            train_ratio: float = 0.7, val_ratio: float = 0.15,
                            random_seed: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train/val/test sets using window-level splitting.
    This allows windows from the same subject to appear in different splits.
    """
    print("\n" + "="*80)
    print("WINDOW-LEVEL SPLITTING")
    print("="*80)
    
    n_windows = len(dataset.window_indices)
    indices = list(range(n_windows))
    
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
    n_train = int(train_ratio * n_windows)
    n_val = int(val_ratio * n_windows)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    print(f"Total windows: {n_windows}")
    print(f"  Train windows: {len(train_indices)} ({len(train_indices)/n_windows*100:.1f}%)")
    print(f"  Val windows: {len(val_indices)} ({len(val_indices)/n_windows*100:.1f}%)")
    print(f"  Test windows: {len(test_indices)} ({len(test_indices)/n_windows*100:.1f}%)")
    
    # Verify no overlap
    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)
    
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("CRITICAL: Data leakage detected in window split!")
    
    print("✓ No data leakage detected (window-level split)")
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset


def split_dataset_by_subject(dataset: CombinedCNNLOCDataset, combined_dataset: CombinedDataset,
                             train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train/val/test sets using subject-level splitting.
    No subject appears in more than one split. The held-out portion (1 - train_ratio)
    is split 50-50 into val and test so validation and test difficulty are comparable.
    (val_ratio is kept for API compatibility but not used for the val/test split.)
    """
    print("\n" + "="*80)
    print("SUBJECT-LEVEL SPLITTING")
    print("="*80)
    
    # Map windows to subjects - now using subject_id directly from window indices
    # Window indices format: (start_idx, end_idx, label, subject_id, trial_idx, dataset)
    subject_windows = {}  # {subject_id: [window_indices]}
    window_to_subject = {}  # {window_idx: subject_id}
    
    # Map each window to its subject (subject_id is now directly in window indices)
    # Window indices format: (start_idx, end_idx, label, subject_id, trial_idx, dataset)
    for window_idx, window_info in enumerate(dataset.window_indices):
        # Ensure window_info is a tuple/list with at least 4 elements
        if not isinstance(window_info, (tuple, list)):
            raise ValueError(f"Window info at index {window_idx} is not a tuple/list: {type(window_info)}")
        
        # Extract subject_id from window info (index 3)
        # Format: (start_idx, end_idx, label, subject_id, trial_idx, dataset)
        if len(window_info) >= 4:
            subject_id = window_info[3]
        elif len(window_info) == 3:
            # Old format - this shouldn't happen with new CombinedDataset
            print(f"Warning: Window {window_idx} has old 3-tuple format. Expected 6-tuple.")
            subject_id = 'unknown'
        else:
            raise ValueError(f"Window info at index {window_idx} has unexpected length {len(window_info)}. "
                           f"Expected 6 elements (start_idx, end_idx, label, subject_id, trial_idx, dataset), "
                           f"got: {window_info}")
        
        # Group windows by subject
        if subject_id not in subject_windows:
            subject_windows[subject_id] = []
        subject_windows[subject_id].append(window_idx)
        window_to_subject[window_idx] = subject_id
    
    print(f"Found {len(subject_windows)} unique subjects:")
    for subject_id, windows in sorted(subject_windows.items()):
        print(f"  {subject_id}: {len(windows)} windows")
    
    # Split subjects (not windows) to prevent data leakage.
    # Val and test get equal number of held-out subjects so difficulty is comparable.
    subjects = list(subject_windows.keys())
    np.random.seed(42)  # Fixed seed for reproducibility
    np.random.shuffle(subjects)
    
    n_subjects = len(subjects)
    n_train_subjects = int(train_ratio * n_subjects)
    n_heldout = n_subjects - n_train_subjects
    n_val_subjects = n_heldout // 2
    n_test_subjects = n_heldout - n_val_subjects  # same or +1 so val and test are balanced
    
    train_subjects = subjects[:n_train_subjects]
    val_subjects = subjects[n_train_subjects:n_train_subjects + n_val_subjects]
    test_subjects = subjects[n_train_subjects + n_val_subjects:]
    
    print(f"\nSubject-wise split (val and test same # subjects for comparable difficulty):")
    print(f"  Train subjects: {len(train_subjects)} ({train_subjects})")
    print(f"  Val subjects: {len(val_subjects)} ({val_subjects})")
    print(f"  Test subjects: {len(test_subjects)} ({test_subjects})")
    
    # Create subject-based window indices
    train_indices = []
    val_indices = []
    test_indices = []
    
    for subject_id in train_subjects:
        train_indices.extend(subject_windows[subject_id])
    for subject_id in val_subjects:
        val_indices.extend(subject_windows[subject_id])
    for subject_id in test_subjects:
        test_indices.extend(subject_windows[subject_id])
    
    print(f"\nWindow-based split:")
    print(f"  Train windows: {len(train_indices)}")
    print(f"  Val windows: {len(val_indices)}")
    print(f"  Test windows: {len(test_indices)}")
    
    # Verify no overlap between splits (data leakage check)
    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)
    
    if train_set & val_set:
        raise ValueError("CRITICAL: Data leakage detected - train/val overlap!")
    if train_set & test_set:
        raise ValueError("CRITICAL: Data leakage detected - train/test overlap!")
    if val_set & test_set:
        raise ValueError("CRITICAL: Data leakage detected - val/test overlap!")
    
    # Verify subject separation
    train_subject_set = set(train_subjects)
    val_subject_set = set(val_subjects)
    test_subject_set = set(test_subjects)
    
    if train_subject_set & val_subject_set:
        raise ValueError("CRITICAL: Subject overlap between train/val!")
    if train_subject_set & test_subject_set:
        raise ValueError("CRITICAL: Subject overlap between train/test!")
    if val_subject_set & test_subject_set:
        raise ValueError("CRITICAL: Subject overlap between val/test!")
    
    print("✓ No data leakage detected - subjects properly separated")
    print("  Val and test each have the same number of held-out subjects (balanced difficulty).")
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset


def split_dataset_by_subject_cv(dataset: CombinedCNNLOCDataset, combined_dataset: CombinedDataset,
                                k_folds: int, fold_index: int, random_seed: int = 42
                                ) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Subject-level k-fold: for fold_index in [0, k_folds-1], assign test=fold_index, val=(fold_index+1)%k, train=rest.
    Val and test each get one fold (same number of subjects) for comparable difficulty.
    """
    subject_windows = {}
    for window_idx, window_info in enumerate(dataset.window_indices):
        if len(window_info) >= 4:
            subject_id = window_info[3]
        else:
            subject_id = 'unknown'
        if subject_id not in subject_windows:
            subject_windows[subject_id] = []
        subject_windows[subject_id].append(window_idx)
    
    subjects = list(subject_windows.keys())
    np.random.seed(random_seed)
    np.random.shuffle(subjects)
    n = len(subjects)
    fold_size = (n + k_folds - 1) // k_folds
    folds = [subjects[i * fold_size:(i + 1) * fold_size] for i in range(k_folds)]
    
    test_subjects = folds[fold_index]
    val_subjects = folds[(fold_index + 1) % k_folds]
    train_subjects = [s for i in range(k_folds) if i != fold_index and i != (fold_index + 1) % k_folds for s in folds[i]]
    
    train_indices = []
    for sid in train_subjects:
        train_indices.extend(subject_windows[sid])
    val_indices = []
    for sid in val_subjects:
        val_indices.extend(subject_windows[sid])
    test_indices = []
    for sid in test_subjects:
        test_indices.extend(subject_windows[sid])
    
    return (torch.utils.data.Subset(dataset, train_indices),
            torch.utils.data.Subset(dataset, val_indices),
            torch.utils.data.Subset(dataset, test_indices))


def get_subject_balanced_sampler(train_dataset: torch.utils.data.Subset) -> Optional[WeightedRandomSampler]:
    """Return a sampler that weights each window by 1/(windows per subject) so each subject contributes equally."""
    dataset = train_dataset.dataset
    indices = train_dataset.indices
    if not hasattr(dataset, 'window_indices') or not dataset.window_indices:
        return None
    subject_counts = {}
    for idx in indices:
        win = dataset.window_indices[idx]
        sid = win[3] if len(win) >= 4 else 'unknown'
        subject_counts[sid] = subject_counts.get(sid, 0) + 1
    weights = []
    for idx in indices:
        sid = dataset.window_indices[idx][3] if len(dataset.window_indices[idx]) >= 4 else 'unknown'
        w = 1.0 / max(1, subject_counts[sid])
        weights.append(w)
    w_tensor = torch.tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights=w_tensor, num_samples=len(indices), replacement=True)


def compute_class_weights(train_dataset: torch.utils.data.Subset, num_classes: int = 2) -> Optional[torch.Tensor]:
    """
    Compute balanced class weights from training set for imbalanced data.
    weight_k = n_samples / (num_classes * n_class_k). Returns None if a class is missing.
    Uses window_indices (label at index 2) to avoid loading full samples.
    """
    dataset = train_dataset.dataset
    indices = train_dataset.indices
    if not hasattr(dataset, 'window_indices') or not dataset.window_indices:
        return None
    labels = np.array([dataset.window_indices[idx][2] for idx in indices])
    counts = np.bincount(labels.astype(int), minlength=num_classes)
    if np.any(counts == 0):
        return None
    n = len(labels)
    weights = n / (num_classes * counts.astype(np.float64))
    weights = weights / weights.sum() * num_classes  # normalize so mean weight = 1
    return torch.tensor(weights.astype(np.float32))


class CNNLOCTrainer:
    """Trainer for CNN-LOC model."""
    
    def __init__(self, model: CNNLOCModel, device: torch.device, output_dir: str = "combined_cnnloc_sub_results"):
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.best_val_acc = 0.0
        self.best_model_path = self.output_dir / "best_model.pth"
    
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            target = target.squeeze()
            # Optional input augmentation to reduce overfitting
            if getattr(self, '_augment_std', 0.0) > 0:
                data = data + torch.randn_like(data, device=data.device) * self._augment_std
            if getattr(self, '_augment_channel_dropout', 0.0) > 0:
                # Randomly zero out some channels (axis 1 is channels)
                n_ch = data.size(1)
                n_drop = max(1, int(n_ch * self._augment_channel_dropout))
                perm = torch.randperm(n_ch, device=data.device)[:n_drop]
                data = data.clone()
                data[:, perm, :, :] = 0
            # Forward
            output = self.model(data)
            loss = criterion(output, target)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            total_loss += loss.item()
        
        if total == 0:
            return float('inf'), 0.0
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Validate for one epoch (window-level metrics)."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                target = target.squeeze()
                
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        if total == 0:
            return float('inf'), 0.0
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch_trial(self, val_loader: DataLoader, val_dataset: torch.utils.data.Subset,
                             dataset: CombinedCNNLOCDataset, criterion: nn.Module
                             ) -> Tuple[float, float, Optional[float]]:
        """Validate and compute trial-level accuracy (majority vote per trial). Returns (loss, window_acc, trial_acc)."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                target = target.squeeze()
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        if not all_preds:
            return float('inf'), 0.0, None
        
        preds = np.array(all_preds)
        targets = np.array(all_targets)
        window_acc = 100.0 * (preds == targets).mean()
        avg_loss = total_loss / len(val_loader)
        
        # Trial-level: group by (subject_id, trial_idx), majority vote
        from collections import defaultdict
        indices = val_dataset.indices
        agg_pred = defaultdict(list)
        agg_tar = defaultdict(lambda: None)
        for i, idx in enumerate(indices):
            if i >= len(preds):
                break
            win = dataset.window_indices[idx]
            if len(win) >= 5:
                subj, trial_idx = win[3], win[4]
                agg_pred[(subj, trial_idx)].append(int(preds[i]))
                agg_tar[(subj, trial_idx)] = int(targets[i])
        if not agg_pred:
            return avg_loss, window_acc, None
        trial_pred_list = [int(np.round(np.mean(ps))) for ps in agg_pred.values()]
        trial_tar_list = [agg_tar[k] for k in agg_pred]
        trial_acc = 100.0 * accuracy_score(trial_tar_list, trial_pred_list)
        return avg_loss, window_acc, trial_acc
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 50, learning_rate: float = 1e-4,
              weight_decay: float = 1e-5, patience: int = 10,
              label_smoothing: float = 0.0, augment_std: float = 0.0,
              augment_channel_dropout: float = 0.0,
              class_weights: Optional[torch.Tensor] = None,
              val_dataset: Optional[torch.utils.data.Subset] = None,
              dataset: Optional[CombinedCNNLOCDataset] = None):
        """Train the model. If val_dataset and dataset are provided, early-stop on trial-level val accuracy."""
        criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            weight=class_weights.to(self.device) if class_weights is not None else None
        )
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = OneCycleLR(optimizer, max_lr=learning_rate * 5,
                              total_steps=num_epochs * len(train_loader), pct_start=0.3)
        self._augment_std = augment_std
        self._augment_channel_dropout = augment_channel_dropout
        patience_counter = 0
        use_trial_val = (val_dataset is not None and dataset is not None and
                         hasattr(dataset, 'window_indices') and len(val_dataset.indices) > 0)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            if use_trial_val:
                val_loss, val_window_acc, val_trial_acc = self.validate_epoch_trial(
                    val_loader, val_dataset, dataset, criterion
                )
                val_acc = val_trial_acc if val_trial_acc is not None else val_window_acc
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Window Acc: {val_window_acc:.4f}, Val Trial Acc: {val_trial_acc:.4f}" if val_trial_acc is not None else f"Val Loss: {val_loss:.4f}, Val Acc: {val_window_acc:.4f}")
            else:
                val_loss, val_acc = self.validate_epoch(val_loader, criterion)
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            scheduler.step()
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, self.best_model_path)
                print(f"New best model saved! Val {'Trial ' if use_trial_val and val_trial_acc is not None else ''}Acc: {val_acc:.4f}")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break
        
        print(f"Training completed. Best validation {'trial ' if use_trial_val else ''}accuracy: {self.best_val_acc:.4f}")
        return self.best_val_acc
    
    def test(self, test_loader: DataLoader, test_dataset: Optional[torch.utils.data.Subset] = None,
             dataset: Optional[CombinedCNNLOCDataset] = None, trial_aggregation: bool = True) -> Dict:
        """Test model and compute metrics. If test_dataset and dataset are provided and trial_aggregation=True,
        also compute trial-level accuracy (majority vote over windows per trial)."""
        checkpoint = torch.load(self.best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Testing"):
                data, target = data.to(self.device), target.to(self.device)
                target = target.squeeze()
                
                output = self.model(data)
                probabilities = F.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
        
        preds = np.array(all_predictions)
        targets = np.array(all_targets)
        probs = np.array(all_probabilities)
        
        accuracy = accuracy_score(targets, preds)
        try:
            roc_auc = roc_auc_score(targets, probs)
        except ValueError:
            roc_auc = 0.5
        balanced_acc = balanced_accuracy_score(targets, preds)
        
        results = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'roc_auc': roc_auc,
            'trial_accuracy': None,
            'trial_balanced_accuracy': None,
            'predictions': preds,
            'targets': targets,
            'probabilities': probs,
            'best_val_acc': self.best_val_acc
        }
        
        # Trial-level aggregation: majority vote per (subject_id, trial_idx)
        if trial_aggregation and test_dataset is not None and dataset is not None and hasattr(dataset, 'window_indices'):
            trial_preds, trial_targets = [], []
            indices = test_dataset.indices
            for i, idx in enumerate(indices):
                if i >= len(preds):
                    break
                win = dataset.window_indices[idx]
                if len(win) >= 5:
                    subj, trial_idx = win[3], win[4]
                    trial_preds.append((subj, trial_idx, preds[i]))
                    trial_targets.append((subj, trial_idx, targets[i]))
            if trial_preds:
                from collections import defaultdict
                agg_pred = defaultdict(list)
                agg_tar = defaultdict(lambda: None)
                for (s, t, p), (_, _, y) in zip(trial_preds, trial_targets):
                    agg_pred[(s, t)].append(p)
                    agg_tar[(s, t)] = y
                trial_pred_list = [int(np.round(np.mean(ps))) for ps in agg_pred.values()]
                trial_tar_list = [agg_tar[k] for k in agg_pred]
                results['trial_accuracy'] = accuracy_score(trial_tar_list, trial_pred_list)
                results['trial_balanced_accuracy'] = balanced_accuracy_score(trial_tar_list, trial_pred_list)
        
        return results


SAMPLING_RATE_HZ = 128.0


def run_single_window_experiment(args, window_size_samples: int) -> Dict:
    """Run train/val/test for one window size. Returns dict with trial_accuracy, trial_balanced_accuracy, best_val_acc."""
    combined_dataset = CombinedDataset(
        das_data_dir=args.das_data_dir,
        das_preprocessing_type=args.das_preprocessing_type,
        das_original_dir=getattr(args, 'das_original_dir', 'Data/Das/4004271'),
        das_audio_dir=getattr(args, 'das_audio_dir', 'Data/Das/4004271/stimuli/stimuli'),
        fulsang_raw_dir=args.fulsang_raw_dir,
        fulsang_audio_dir=args.fulsang_audio_dir,
        fulsang_mwf_output_dir=args.fulsang_mwf_dir,
        combined_dataset_dir=getattr(args, 'combined_dataset_dir', 'combined_dataset'),
        window_size=window_size_samples,
        overlap=args.overlap
    )
    bandpass_low = getattr(args, 'bandpass_low', 0)
    bandpass_high = getattr(args, 'bandpass_high', 0)
    if bandpass_high > bandpass_low and bandpass_high > 0:
        pytorch_dataset = CombinedCNNLOCDataset(combined_dataset, transform_eeg=True,
                                                bandpass_low=bandpass_low, bandpass_high=bandpass_high)
    else:
        pytorch_dataset = CombinedCNNLOCDataset(combined_dataset, transform_eeg=True)
    
    train_dataset, val_dataset, test_dataset = split_dataset_by_subject(
        pytorch_dataset, combined_dataset, train_ratio=0.7, val_ratio=0.15
    )
    train_sampler = get_subject_balanced_sampler(train_dataset)
    if train_sampler is not None:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=2)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    class_weights = None
    if not getattr(args, 'no_class_weights', False):
        class_weights = compute_class_weights(train_dataset, num_classes=2)
    
    model = CNNLOCModel(
        input_channels=combined_dataset.n_channels,
        input_time=32,
        input_freq=4,
        num_classes=2,
        dropout_rate=args.dropout_rate
    )
    out_dir = Path(args.output_dir) / f"sweep_{window_size_samples}"
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer = CNNLOCTrainer(model=model, device=device, output_dir=str(out_dir))
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        label_smoothing=args.label_smoothing,
        augment_std=args.augment_std,
        augment_channel_dropout=getattr(args, 'augment_channel_dropout', 0.0),
        class_weights=class_weights,
        val_dataset=val_dataset,
        dataset=pytorch_dataset
    )
    test_metrics = trainer.test(
        test_loader,
        test_dataset=test_dataset,
        dataset=pytorch_dataset,
        trial_aggregation=True
    )
    return {
        'window_sec': window_size_samples / SAMPLING_RATE_HZ,
        'window_samples': window_size_samples,
        'trial_accuracy': test_metrics.get('trial_accuracy'),
        'trial_balanced_accuracy': test_metrics.get('trial_balanced_accuracy'),
        'best_val_acc': test_metrics.get('best_val_acc'),
    }


def run_window_sweep(args, window_seconds_min: int = 1, window_seconds_max: int = 30) -> List[Dict]:
    """Run 1s to 30s window sweep; each result includes trial_accuracy and trial_balanced_accuracy."""
    results = []
    for sec in range(window_seconds_min, window_seconds_max + 1):
        samples = sec * int(SAMPLING_RATE_HZ)
        print(f"\n{'='*80}\nWindow: {sec}s ({samples} samples)\n{'='*80}")
        try:
            r = run_single_window_experiment(args, samples)
            r['window_sec'] = sec
            results.append(r)
            ta = r.get('trial_accuracy')
            tba = r.get('trial_balanced_accuracy')
            ta_str = f"{ta:.4f}" if ta is not None else "N/A"
            tba_str = f"{tba:.4f}" if tba is not None else "N/A"
            print(f"  → Trial Accuracy: {ta_str}, Trial Balanced Accuracy: {tba_str}")
        except Exception as e:
            print(f"  → Failed: {e}")
            results.append({
                'window_sec': sec,
                'window_samples': samples,
                'trial_accuracy': None,
                'trial_balanced_accuracy': None,
                'best_val_acc': None,
                'error': str(e)
            })
    return results


def save_window_sweep_results(results: List[Dict], output_dir: Path):
    """Save 1s–30s sweep to CSV and plot (trial-level metrics only)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # CSV: window_sec, window_samples, trial_accuracy, trial_balanced_accuracy, best_val_acc
    rows = []
    for r in results:
        rows.append({
            'window_sec': r['window_sec'],
            'window_samples': r.get('window_samples'),
            'trial_accuracy': r.get('trial_accuracy'),
            'trial_balanced_accuracy': r.get('trial_balanced_accuracy'),
            'best_val_acc': r.get('best_val_acc'),
        })
    csv_path = output_dir / "window_sweep_1s_to_30s.csv"
    with open(csv_path, 'w') as f:
        f.write("window_sec,window_samples,trial_accuracy,trial_balanced_accuracy,best_val_acc\n")
        for row in rows:
            ta = row['trial_accuracy'] if row['trial_accuracy'] is not None else ''
            tba = row['trial_balanced_accuracy'] if row['trial_balanced_accuracy'] is not None else ''
            va = row['best_val_acc'] if row['best_val_acc'] is not None else ''
            f.write(f"{row['window_sec']},{row['window_samples']},{ta},{tba},{va}\n")
    print(f"\nSweep results saved to {csv_path}")
    
    # Plot: Trial Accuracy and Trial Balanced Accuracy vs window (s)
    secs = [r['window_sec'] for r in results if r.get('trial_accuracy') is not None]
    ta_list = [r['trial_accuracy'] for r in results if r.get('trial_accuracy') is not None]
    tba_list = [r['trial_balanced_accuracy'] for r in results if r.get('trial_balanced_accuracy') is not None]
    if secs and ta_list:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(secs, ta_list, 'b-o', label='Trial Accuracy', markersize=4)
            if tba_list:
                ax.plot(secs, tba_list, 'g-s', label='Trial Balanced Accuracy', markersize=4)
            ax.set_xlabel('Window size (s)')
            ax.set_ylabel('Accuracy')
            ax.set_ylim(0, 1.05)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.title('Combined CNN-LOC: Trial-level accuracy vs window size (1s–30s)')
            plt.tight_layout()
            plot_path = output_dir / "window_sweep_1s_to_30s.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Sweep plot saved to {plot_path}")
        except Exception as e:
            print(f"Could not save plot: {e}")


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Combined Das+Fulsang CNN-LOC using FULCNN architecture with Subject-Level Splitting')
    parser.add_argument('--das_data_dir', type=str, default='das_combined_preprocessed',
                       help='Directory containing Das preprocessed data')
    parser.add_argument('--das_preprocessing_type', type=str, default='COMBINED_DAS',
                       choices=['COMBINED_DAS', '16SUBJECTS', 'MWF', 'DASPREPROCESS'],
                       help='Type of Das preprocessing (16SUBJECTS is alias for COMBINED_DAS)')
    parser.add_argument('--fulsang_raw_dir', type=str, default=None,
                       help='Directory containing Fulsang raw EEG data (optional if MWF files exist)')
    parser.add_argument('--fulsang_audio_dir', type=str, default=None,
                       help='Directory containing Fulsang audio data (optional)')
    parser.add_argument('--fulsang_mwf_dir', type=str, default='/home/py9363/telluride_decoding/MWF_cleaned_Fuglsang',
                       help='Directory containing Fulsang MWF-processed data (legacy)')
    parser.add_argument('--combined_dataset_dir', type=str, default='combined_dataset',
                       help='Centralized directory for all processed files (default: combined_dataset)')
    parser.add_argument('--split_method', type=str, default='both', choices=['subject', 'window', 'both'],
                       help='Split method: subject (no leakage), window (random), or both (comparison)')
    parser.add_argument('--window_size', type=int, default=1024,
                       help='Window size in samples (default: 1024 = 8s at 128Hz; 8s recommended for ~77%% test accuracy)')
    parser.add_argument('--overlap', type=float, default=0.25,
                       help='Window overlap fraction (default: 0.25 for less correlated windows)')
    parser.add_argument('--bandpass_low', type=float, default=1.0,
                       help='Bandpass low cutoff Hz; 0 to disable (default: 1)')
    parser.add_argument('--bandpass_high', type=float, default=40.0,
                       help='Bandpass high cutoff Hz (default: 40 for speech AAD)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--num_epochs', type=int, default=80,
                       help='Number of training epochs (default: 80)')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                       help='Learning rate (default: 5e-4)')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                       help='Dropout rate (default: 0.5 for stronger regularization)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='AdamW weight decay (default: 1e-4 for better subject generalization)')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                       help='CrossEntropy label smoothing (default: 0)')
    parser.add_argument('--augment_std', type=float, default=0.0,
                       help='Gaussian noise std on input (default: 0; try 0.02 for light augmentation)')
    parser.add_argument('--augment_channel_dropout', type=float, default=0.1,
                       help='Random channel dropout fraction during training (default: 0.1)')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience in epochs (default: 20)')
    parser.add_argument('--no_class_weights', action='store_true',
                       help='Disable class weights for imbalanced labels (default: use weights)')
    parser.add_argument('--no_trial_aggregation', action='store_true',
                       help='Disable trial-level majority-vote metrics (default: report trial-level acc)')
    parser.add_argument('--cv_folds', type=int, default=1,
                       help='Subject-level k-fold CV; 1 = single split (default: 1)')
    parser.add_argument('--single_run', action='store_true',
                       help='Run single window size only. Default: 1s–30s window sweep (trial metrics only).')
    parser.add_argument('--window_seconds_min', type=int, default=1,
                       help='Min window size in seconds for sweep (default: 1)')
    parser.add_argument('--window_seconds_max', type=int, default=30,
                       help='Max window size in seconds for sweep (default: 30)')
    parser.add_argument('--output_dir', type=str, default='combined_cnnloc_sub_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    # Normalize alias: 16SUBJECTS -> COMBINED_DAS (same pipeline)
    if args.das_preprocessing_type == '16SUBJECTS':
        args.das_preprocessing_type = 'COMBINED_DAS'
        print("Using combined Das (16SUBJECTS -> COMBINED_DAS)")

    print("="*80)
    print("COMBINED CNN-LOC (SUBJECT-LEVEL SPLITTING) - Das (MWF) + Fulsang (MWF) CNN-LOC Training")
    print("="*80)
    print(f"Using CNN-LOC architecture from FULCNN")
    window_sec = args.window_size / 128.0
    print(f"  Window size: {args.window_size} samples ({window_sec:.1f}s at 128 Hz)" + (" (8s — target test acc 77%%)" if args.window_size == 1024 else ""))
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  Augment std: {args.augment_std}")
    print(f"  Channel dropout: {getattr(args, 'augment_channel_dropout', 0)}")
    print(f"  Splitting: Subject-level (prevents data leakage)")
    if getattr(args, 'cv_folds', 1) > 1:
        print(f"  CV: {args.cv_folds}-fold")
    
    # Handle Fulsang raw directory (same logic as STANETCNN)
    if args.fulsang_raw_dir is None:
        # Check if MWF-processed data already exists
        mwf_dir = Path(args.fulsang_mwf_dir)
        existing_mwf_files = list(mwf_dir.glob("sub*_MWF.mat")) if mwf_dir.exists() else []
        
        if existing_mwf_files:
            print(f"Found {len(existing_mwf_files)} existing MWF-processed Fulsang files in {args.fulsang_mwf_dir}")
            print("  Using existing MWF files directly (skipping raw data loading)")
            
            # Create a minimal dummy raw directory with dummy files to satisfy CombinedDataset's check
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="fulsang_raw_dummy_")
            # Create dummy .mat files matching the MWF files
            for mwf_file in existing_mwf_files:
                subject_id_str = mwf_file.stem.replace('_MWF', '').replace('sub', '')
                try:
                    subject_id = int(subject_id_str)
                    dummy_file = Path(temp_dir) / f"S{subject_id}.mat"
                    dummy_file.touch()
                except ValueError:
                    dummy_file = Path(temp_dir) / "S1.mat"
                    dummy_file.touch()
                    break
            args.fulsang_raw_dir = temp_dir
            print(f"  Created temporary dummy raw directory with {len(list(Path(temp_dir).glob('S*.mat')))} dummy files")
    
    # Default: 1s–30s window sweep (trial-level metrics only)
    if not getattr(args, 'single_run', False):
        print("\n" + "="*80)
        print("WINDOW SWEEP: 1s to 30s (Trial-level accuracy)")
        print("="*80)
        sweep_results = run_window_sweep(args, args.window_seconds_min, args.window_seconds_max)
        save_window_sweep_results(sweep_results, Path(args.output_dir))
        print("\n" + "="*80)
        print("WINDOW SWEEP SUMMARY (Trial accuracy)")
        print("="*80)
        for r in sweep_results:
            ta = r.get('trial_accuracy')
            tba = r.get('trial_balanced_accuracy')
            if ta is not None:
                tba_str = f", Trial Bal Acc = {tba:.4f}" if tba is not None else ""
                print(f"  {int(r['window_sec']):2d}s: Trial Acc = {ta:.4f}{tba_str}")
            else:
                print(f"  {int(r['window_sec']):2d}s: Failed ({r.get('error', 'unknown')})")
        print("="*80)
        print(f"\n✓ Sweep complete. Results in {args.output_dir}/window_sweep_1s_to_30s.csv")
        return
    
    # Single run (--single_run)
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
    
    # Fulsang label reminder
    fulsang_raw = getattr(args, 'fulsang_raw_dir', None)
    if fulsang_raw and ('dummy' in str(fulsang_raw).lower() or 'temp' in str(fulsang_raw).lower()):
        print("\n  ⚠ For best accuracy, provide Fulsang raw EEG so true attention labels are used.")
        print("    Expected: Data/Fulsang/EEG/EEG.zip or Data/Fulsang/EEG/EEG (extracted).")
    
    # Create PyTorch dataset
    print("\n" + "="*80)
    print("CREATING PYTORCH DATASET")
    print("="*80)
    bandpass_low = getattr(args, 'bandpass_low', 0)
    bandpass_high = getattr(args, 'bandpass_high', 0)
    if bandpass_high > bandpass_low and bandpass_high > 0:
        pytorch_dataset = CombinedCNNLOCDataset(combined_dataset, transform_eeg=True,
                                                bandpass_low=bandpass_low, bandpass_high=bandpass_high)
        print(f"  Bandpass: {bandpass_low}-{bandpass_high} Hz")
    else:
        pytorch_dataset = CombinedCNNLOCDataset(combined_dataset, transform_eeg=True)
    
    # Split dataset based on method (or CV)
    if args.split_method == 'subject':
        print("\nUsing SUBJECT-LEVEL splitting (prevents data leakage)")
        train_dataset, val_dataset, test_dataset = split_dataset_by_subject(
            pytorch_dataset, combined_dataset, train_ratio=0.7, val_ratio=0.15
        )
    elif args.split_method == 'window':
        print("\nUsing WINDOW-LEVEL splitting (may have data leakage)")
        train_dataset, val_dataset, test_dataset = split_dataset_by_window(
            pytorch_dataset, train_ratio=0.7, val_ratio=0.15
        )
    else:  # both
        print("\n" + "="*80)
        print("COMPARING BOTH SPLIT METHODS")
        print("="*80)
        
        # Subject-wise split
        print("\n[1/2] SUBJECT-LEVEL SPLIT:")
        train_subj, val_subj, test_subj = split_dataset_by_subject(
            pytorch_dataset, combined_dataset, train_ratio=0.7, val_ratio=0.15
        )
        
        # Window-wise split
        print("\n[2/2] WINDOW-LEVEL SPLIT:")
        train_win, val_win, test_win = split_dataset_by_window(
            pytorch_dataset, train_ratio=0.7, val_ratio=0.15
        )
        
        # Use subject-wise by default (safer)
        print("\n" + "="*80)
        print("Using SUBJECT-LEVEL split for training (no data leakage)")
        print("Window-level split statistics shown for comparison")
        print("="*80)
        train_dataset, val_dataset, test_dataset = train_subj, val_subj, test_subj
    
    # Optional: k-fold subject-level CV
    if getattr(args, 'cv_folds', 1) > 1:
        k = args.cv_folds
        print("\n" + "="*80)
        print(f"SUBJECT-LEVEL {k}-FOLD CROSS-VALIDATION")
        print("="*80)
        splits = [split_dataset_by_subject_cv(pytorch_dataset, combined_dataset, k, f) for f in range(k)]
    else:
        splits = [(train_dataset, val_dataset, test_dataset)]
    
    cv_results = []
    for split_idx, (train_dataset, val_dataset, test_dataset) in enumerate(splits):
        if len(splits) > 1:
            print(f"\n--- Fold {split_idx + 1}/{len(splits)} ---")
        
        # Save comparison statistics (only for single split, both method)
        if len(splits) == 1 and args.split_method == 'both':
            comparison_stats = {
                'subject_split': {
                    'train': len(train_subj),
                    'val': len(val_subj),
                    'test': len(test_subj),
                    'total': len(train_subj) + len(val_subj) + len(test_subj)
                },
                'window_split': {
                    'train': len(train_win),
                    'val': len(val_win),
                    'test': len(test_win),
                    'total': len(train_win) + len(val_win) + len(test_win)
                }
            }
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            with open(output_path / "split_comparison.json", 'w') as f:
                json.dump(comparison_stats, f, indent=2)
            print(f"\n✓ Split comparison saved to {output_path / 'split_comparison.json'}")
        
        out_dir = Path(args.output_dir) if len(splits) == 1 else Path(args.output_dir) / f"fold_{split_idx}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        train_sampler = get_subject_balanced_sampler(train_dataset) if len(splits) == 1 else None
        if train_sampler is not None:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=2)
        else:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
        
        class_weights = None
        if not getattr(args, 'no_class_weights', False):
            class_weights = compute_class_weights(train_dataset, num_classes=2)
            if class_weights is not None and len(splits) == 1:
                print(f"  Class weights (from train): {class_weights.tolist()}")
            elif class_weights is None and len(splits) == 1:
                print("  Class weights: skipped (missing class or balanced)")
        
        model = CNNLOCModel(
            input_channels=combined_dataset.n_channels,
            input_time=32,
            input_freq=4,
            num_classes=2,
            dropout_rate=args.dropout_rate
        )
        
        trainer = CNNLOCTrainer(
            model=model,
            device=device,
            output_dir=str(out_dir)
        )
        
        print("\n" + "="*80)
        print("TRAINING CNN-LOC MODEL")
        print("="*80)
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            label_smoothing=args.label_smoothing,
            augment_std=args.augment_std,
            augment_channel_dropout=getattr(args, 'augment_channel_dropout', 0.0),
            class_weights=class_weights,
            val_dataset=val_dataset,
            dataset=pytorch_dataset
        )
        
        print("\n" + "="*80)
        print("TESTING MODEL")
        print("="*80)
        trial_agg = not getattr(args, 'no_trial_aggregation', False)
        test_metrics = trainer.test(
            test_loader,
            test_dataset=test_dataset,
            dataset=pytorch_dataset,
            trial_aggregation=trial_agg
        )
        
        if len(splits) > 1:
            cv_results.append(test_metrics)
        else:
            results_json = {
                'accuracy': float(test_metrics['accuracy']),
                'balanced_accuracy': float(test_metrics.get('balanced_accuracy', test_metrics['accuracy'])),
                'roc_auc': float(test_metrics['roc_auc']),
                'best_val_acc': float(test_metrics['best_val_acc']),
                'splitting': 'subject-level',
                'timestamp': datetime.now().isoformat()
            }
            if test_metrics.get('trial_accuracy') is not None:
                results_json['trial_accuracy'] = float(test_metrics['trial_accuracy'])
                results_json['trial_balanced_accuracy'] = float(test_metrics['trial_balanced_accuracy'])
            with open(Path(args.output_dir) / 'results.json', 'w') as f:
                json.dump(results_json, f, indent=2)
            print(f"\n✓ Training Complete (trial-level metrics)")
            if test_metrics.get('trial_accuracy') is not None:
                print(f"  Trial Accuracy: {test_metrics['trial_accuracy']:.4f}")
                print(f"  Trial Balanced Accuracy: {test_metrics['trial_balanced_accuracy']:.4f}")
            else:
                print(f"  Trial Accuracy: N/A")
            print(f"\n✓ Results saved to {args.output_dir}")
    
    if len(splits) > 1:
        accs = [r['accuracy'] for r in cv_results]
        bal_accs = [r.get('balanced_accuracy', r['accuracy']) for r in cv_results]
        rocs = [r['roc_auc'] for r in cv_results]
        print("\n" + "="*80)
        print("CROSS-VALIDATION SUMMARY")
        print("="*80)
        print(f"  Test Accuracy:          {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  Test Balanced Accuracy: {np.mean(bal_accs):.4f} ± {np.std(bal_accs):.4f}")
        print(f"  Test ROC-AUC:           {np.mean(rocs):.4f} ± {np.std(rocs):.4f}")
        trial_accs = [r.get('trial_accuracy') for r in cv_results if r.get('trial_accuracy') is not None]
        if trial_accs:
            print(f"  Trial Accuracy:         {np.mean(trial_accs):.4f} ± {np.std(trial_accs):.4f}")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(args.output_dir) / 'cv_results.json', 'w') as f:
            json.dump({
                'n_folds': len(splits),
                'accuracy_mean': float(np.mean(accs)),
                'accuracy_std': float(np.std(accs)),
                'balanced_accuracy_mean': float(np.mean(bal_accs)),
                'balanced_accuracy_std': float(np.std(bal_accs)),
                'roc_auc_mean': float(np.mean(rocs)),
                'roc_auc_std': float(np.std(rocs)),
                'fold_results': [{'accuracy': r['accuracy'], 'balanced_accuracy': r.get('balanced_accuracy'), 'roc_auc': r['roc_auc']} for r in cv_results]
            }, f, indent=2)
        print(f"\n✓ CV results saved to {args.output_dir}/cv_results.json")


if __name__ == '__main__':
    main()

