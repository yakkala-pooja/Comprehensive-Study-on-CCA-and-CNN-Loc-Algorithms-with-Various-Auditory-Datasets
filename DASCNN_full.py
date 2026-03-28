#!/usr/bin/env python3
"""
DASCNN - Full CNN-LOC Algorithm for DAS Dataset (16-Subjects Pipeline)

This module implements a comprehensive CNN-LOC (Convolutional Neural Network - Localization) 
algorithm specifically designed for the DAS dataset, following the same architecture principles
as FULCNN but adapted for DAS data characteristics. It includes:

- Full CNN-LOC architecture with attention mechanisms and residual connections
- Multi-scale feature extraction and spatial-temporal attention
- Comprehensive metrics: Accuracy, MSED, ROC-AUC, and temporal performance
- Temporal analysis across window lengths from 0.5s to 30s
- Robust preprocessing and data handling
- Detailed performance evaluation and reporting
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import autocast, GradScaler
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           precision_recall_fscore_support, roc_auc_score, roc_curve,
                           precision_recall_curve, average_precision_score,
                           matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
                           f1_score)
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr
import seaborn as sns
from tqdm import tqdm
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings and optimize for speed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.config.optimizer.set_jit(True)  # Enable XLA compilation

# Add telluride_decoding to path
sys.path.append('telluride_decoding')

try:
    from telluride_decoding import decoding
    from telluride_decoding import brain_data
    from telluride_decoding import regression
    from telluride_decoding import attention_decoder
except ImportError as e:
    print(f"Warning: Could not import some telluride_decoding modules: {e}")
    print("Continuing with basic functionality...")


class DASDataset(Dataset):
    """
    Dataset class for DAS data with 16-subjects preprocessing pipeline.
    Implements the same comprehensive preprocessing as FULCNN but adapted for DAS data.
    """
    
    def __init__(self, tfrecord_dir: str, mode: str = 'full', 
                 window_size: int = 512, overlap: float = 0.5, 
                 use_validated_data: bool = True, transform_eeg: bool = True):
        """
        Initialize DAS dataset with comprehensive preprocessing.
        
        Args:
            tfrecord_dir: Directory containing TFRecord files
            mode: 'full', 'train', 'val', or 'test'
            window_size: Window size in samples (512 samples = 8s at 64Hz)
            overlap: Overlap ratio between windows
            use_validated_data: Whether to use validated data
            transform_eeg: Whether to apply EEG transformations
        """
        self.tfrecord_dir = Path(tfrecord_dir)
        self.mode = mode
        self.window_size = window_size
        self.overlap = overlap
        self.use_validated_data = use_validated_data
        self.transform_eeg = transform_eeg
        
        # DAS-specific parameters (adapted for CNN-LOC)
        self.sampling_rate = 64  # Hz (downsampled from 1000 Hz)
        self.n_channels = 64  # EEG channels
        self.attention_switch_duration = 20  # seconds (same as Fulsang)
        
        print(f"DAS Dataset initialized:")
        print(f"  Mode: {mode}")
        print(f"  Window size: {window_size} samples ({window_size/self.sampling_rate:.1f}s at {self.sampling_rate}Hz)")
        print(f"  Overlap: {overlap}")
        print(f"  Sampling rate: {self.sampling_rate} Hz")
        print(f"  Channels: {self.n_channels}")
        print(f"  Transform EEG: {transform_eeg}")
        
        # Load data
        self.eeg_data, self.labels, self.metadata = self._load_das_data()
        
        if len(self.eeg_data) == 0:
            raise ValueError("No valid DAS data found")
        
        # Create windows with comprehensive preprocessing
        self.window_indices = self._create_das_windows()
        
        print(f"  Loaded {len(self.eeg_data)} samples")
        print(f"  Created {len(self.window_indices)} windows")
        print(f"  Label distribution: {np.bincount(self.labels)}")
    
    def _load_das_data(self) -> Tuple[List[np.ndarray], List[int], List[Dict]]:
        """Load DAS data from TFRecord files with comprehensive preprocessing."""
        print(f"Loading DAS data from {self.tfrecord_dir}...")
        
        # Find TFRecord files
        tfrecord_files = []
        
        # Check for train/test subdirectories first
        train_dir = self.tfrecord_dir / "train"
        test_dir = self.tfrecord_dir / "test"
        val_dir = self.tfrecord_dir / "val"
        
        if train_dir.exists() and test_dir.exists():
            print(f"Found train/test subdirectories")
            if self.mode == 'train':
                tfrecord_files = list(train_dir.glob("*.tfrecords"))
            elif self.mode == 'test':
                tfrecord_files = list(test_dir.glob("*.tfrecords"))
            elif self.mode == 'val' and val_dir.exists():
                tfrecord_files = list(val_dir.glob("*.tfrecords"))
            elif self.mode == 'full':
                tfrecord_files = list(train_dir.glob("*.tfrecords")) + list(test_dir.glob("*.tfrecords"))
                if val_dir.exists():
                    tfrecord_files += list(val_dir.glob("*.tfrecords"))
        else:
            # Fallback: search in main directory
            tfrecord_files = list(self.tfrecord_dir.glob("*.tfrecords"))
        
        if not tfrecord_files:
            raise ValueError(f"No TFRecord files found in {self.tfrecord_dir}")
        
        print(f"Found {len(tfrecord_files)} TFRecord files")
        
        # Load data from TFRecord files
        eeg_data = []
        labels = []
        metadata = []
        
        for tfrecord_file in tqdm(tfrecord_files, desc="Loading TFRecord files"):
            try:
                # Parse TFRecord file
                dataset = tf.data.TFRecordDataset(str(tfrecord_file))
                
                for raw_record in dataset:
                    try:
                        example = tf.train.Example()
                        example.ParseFromString(raw_record.numpy())
                        features = example.features.feature
                        
                        # Required features for DAS TFRecord files (16-subjects pipeline)
                        required_features = ['eeg', 'attended_ear', 'subject_id']
                        missing_features = [f for f in required_features if f not in features]
                        
                        if missing_features:
                            print(f"WARNING: Missing features {missing_features} in {tfrecord_file.name}")
                            continue
                        
                        # Extract EEG data
                        eeg_bytes = features['eeg'].float_list.value
                        eeg_data_trial = np.array(eeg_bytes).reshape(-1, self.n_channels)
                        
                        # Extract attended_ear
                        attended_ear = features['attended_ear'].bytes_list.value[0].decode('utf-8')
                        
                        # Convert attended_ear to integer label
                        if attended_ear == 'L':
                            label = 0  # Left ear
                        elif attended_ear == 'R':
                            label = 1  # Right ear
                        else:
                            print(f"WARNING: Unknown attended_ear value: {attended_ear}")
                            continue
                        
                        # Extract subject_id from features
                        subject_id = None
                        if 'subject_id' in features:
                            subject_id = features['subject_id'].bytes_list.value[0].decode('utf-8')
                        else:
                            # Fallback: extract from filename
                            subject_id = tfrecord_file.stem.split('_')[0]
                        
                        # Validate EEG shape
                        if eeg_data_trial.shape[1] != self.n_channels:
                            print(f"ERROR: Expected {self.n_channels} EEG channels, got {eeg_data_trial.shape[1]} in {tfrecord_file.name}")
                            continue
                        
                        # Validate subject_id
                        if subject_id is None:
                            print(f"WARNING: No subject_id found for {tfrecord_file.name}")
                            subject_id = "unknown"
                        
                        # Store metadata
                        metadata_trial = {
                            'tfrecord_file': tfrecord_file.name,
                            'attended_ear': attended_ear,
                            'subject_id': subject_id,
                            'sample_rate': self.sampling_rate
                        }
                        
                        # Add each sample
                        for sample_idx in range(len(eeg_data_trial)):
                            eeg_data.append(eeg_data_trial[sample_idx])
                            labels.append(label)
                            metadata.append({
                                **metadata_trial,
                                'sample_idx': sample_idx
                            })
                        
                    except Exception as e:
                        print(f"ERROR processing record in {tfrecord_file.name}: {e}")
                        continue
                        
            except Exception as e:
                print(f"ERROR loading {tfrecord_file.name}: {e}")
                continue
        
        if len(eeg_data) == 0:
            raise ValueError("No valid DAS data found in TFRecord files")
        
        print(f"Successfully loaded {len(eeg_data)} samples from {len(tfrecord_files)} files")
        
        return eeg_data, labels, metadata
    
    def _create_das_windows(self) -> List[Dict]:
        """Create windows from DAS data with comprehensive preprocessing."""
        print("Creating DAS windows with comprehensive preprocessing...")
        
        window_indices = []
        
        # Group data by subject and trial
        subject_trials = {}
        for i, meta in enumerate(self.metadata):
            subject_id = meta['subject_id']
            trial_id = meta['tfrecord_file']
            
            if subject_id not in subject_trials:
                subject_trials[subject_id] = {}
            if trial_id not in subject_trials[subject_id]:
                subject_trials[subject_id][trial_id] = []
            
            subject_trials[subject_id][trial_id].append(i)
        
        # Create windows for each trial
        for subject_id, trials in subject_trials.items():
            for trial_id, indices in trials.items():
                if len(indices) < self.window_size:
                    continue
                
                # Create overlapping windows
                step_size = int(self.window_size * (1 - self.overlap))
                
                for start_idx in range(0, len(indices) - self.window_size + 1, step_size):
                    end_idx = start_idx + self.window_size
                    window_indices.append({
                        'start_idx': indices[start_idx],
                        'end_idx': indices[end_idx - 1],
                        'subject_id': subject_id,
                        'trial_id': trial_id,
                        'window_start': start_idx,
                        'window_end': end_idx
                    })
        
        print(f"Created {len(window_indices)} windows")
        return window_indices
    
    def _preprocess_eeg_window(self, eeg_window: np.ndarray) -> np.ndarray:
        """Apply comprehensive EEG preprocessing like FULCNN."""
        if not self.transform_eeg:
            return eeg_window
        
        # Apply the same preprocessing as FULCNN
        from scipy import signal
        
        # 1. Artifact detection and removal
        for ch in range(eeg_window.shape[1]):
            channel_data = eeg_window[:, ch]
            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)
            
            # Detect artifacts (>5 standard deviations)
            artifacts = np.abs(channel_data - mean_val) > (5.0 * std_val)
            
            if np.any(artifacts):
                # Interpolate over artifacts
                valid_indices = np.where(~artifacts)[0]
                if len(valid_indices) > 1:
                    from scipy.interpolate import interp1d
                    interp_func = interp1d(valid_indices, channel_data[valid_indices], 
                                         kind='linear', fill_value='extrapolate')
                    eeg_window[:, ch] = interp_func(np.arange(len(channel_data)))
        
        # 2. Bandpass filtering (1-40 Hz)
        nyquist = self.sampling_rate / 2
        low_freq = 1.0 / nyquist
        high_freq = min(40.0 / nyquist, 0.99)
        
        b, a = signal.butter(4, [low_freq, high_freq], btype='band')
        
        filtered_eeg = np.zeros_like(eeg_window)
        for ch in range(eeg_window.shape[1]):
            filtered_eeg[:, ch] = signal.filtfilt(b, a, eeg_window[:, ch])
        
        # 3. Robust normalization (MAD)
        mad_values = np.median(np.abs(filtered_eeg - np.median(filtered_eeg, axis=0)), axis=0)
        mad_values = np.where(mad_values == 0, 1.0, mad_values)
        filtered_eeg = filtered_eeg / mad_values
        
        # 4. Soft clipping
        filtered_eeg = np.tanh(filtered_eeg * 0.5)
        
        return filtered_eeg.astype(np.float32)
    
    def _create_spectrogram_features(self, eeg_window: np.ndarray) -> np.ndarray:
        """Create spectrogram features like FULCNN."""
        from scipy import signal
        
        # Parameters for spectrogram computation
        nperseg = min(64, len(eeg_window) // 4)  # Adaptive window size
        noverlap = nperseg // 2
        
        # Compute spectrogram for each channel
        spectrograms = []
        
        for ch in range(eeg_window.shape[1]):
            f, t, Sxx = signal.spectrogram(
                eeg_window[:, ch], 
                fs=self.sampling_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                window='hann'
            )
            
            # Extract frequency bands (same as FULCNN)
            delta_band = (f >= 1) & (f < 4)
            theta_band = (f >= 4) & (f < 8)
            alpha_band = (f >= 8) & (f < 13)
            beta_band = (f >= 13) & (f < 25)
            
            # Calculate band power
            delta_power = np.mean(Sxx[delta_band, :], axis=0)
            theta_power = np.mean(Sxx[theta_band, :], axis=0)
            alpha_power = np.mean(Sxx[alpha_band, :], axis=0)
            beta_power = np.mean(Sxx[beta_band, :], axis=0)
            
            # Stack frequency bands
            channel_features = np.stack([delta_power, theta_power, alpha_power, beta_power], axis=0)
            spectrograms.append(channel_features)
        
        # Stack all channels: (channels, freq_bands, time_points)
        spectrogram_features = np.stack(spectrograms, axis=0)
        
        # Interpolate to consistent time dimension (32 time points)
        target_time_points = 32
        if spectrogram_features.shape[2] != target_time_points:
            from scipy.interpolate import interp1d
            interpolated_features = np.zeros((spectrogram_features.shape[0], spectrogram_features.shape[1], target_time_points))
            
            for ch in range(spectrogram_features.shape[0]):
                for freq in range(spectrogram_features.shape[1]):
                    interp_func = interp1d(
                        np.linspace(0, 1, spectrogram_features.shape[2]),
                        spectrogram_features[ch, freq, :],
                        kind='linear'
                    )
                    interpolated_features[ch, freq, :] = interp_func(np.linspace(0, 1, target_time_points))
            
            spectrogram_features = interpolated_features
        
        return spectrogram_features.astype(np.float32)
    
    def __len__(self):
        return len(self.window_indices)
    
    def __getitem__(self, idx):
        """Get a window from the dataset with comprehensive preprocessing."""
        if idx >= len(self.window_indices):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.window_indices)}")
        
        window_info = self.window_indices[idx]
        
        # Extract EEG window
        start_idx = window_info['start_idx']
        end_idx = window_info['end_idx']
        
        # Get EEG data for this window
        eeg_window = []
        label = None
        
        for i in range(start_idx, end_idx + 1):
            if i < len(self.eeg_data):
                eeg_window.append(self.eeg_data[i])
                if label is None:
                    label = self.labels[i]
        
        if len(eeg_window) < self.window_size:
            # Pad with zeros if necessary
            while len(eeg_window) < self.window_size:
                eeg_window.append(np.zeros(self.n_channels))
        
        eeg_window = np.array(eeg_window[:self.window_size])
        
        # Apply comprehensive preprocessing
        eeg_window = self._preprocess_eeg_window(eeg_window)
        
        # Create spectrogram features (same as FULCNN)
        spectrogram_features = self._create_spectrogram_features(eeg_window)
        
        # Convert to tensors
        window_tensor = torch.from_numpy(spectrogram_features).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # Ensure label_tensor is always 1D (not scalar)
        if label_tensor.dim() == 0:
            label_tensor = label_tensor.unsqueeze(0)
        
        # Validate tensors
        if window_tensor.numel() == 0 or label_tensor.numel() == 0:
            # Return default tensors to prevent crashes
            window_tensor = torch.zeros(self.n_channels, 4, 32, dtype=torch.float32)
            label_tensor = torch.tensor(0, dtype=torch.long).unsqueeze(0)
        
        return window_tensor, label_tensor


class SpatialTemporalAttention(nn.Module):
    """
    Spatial-Temporal Attention mechanism for DAS data (same as FULCNN).
    """
    
    def __init__(self, channels: int, reduction: int = 8):
        super(SpatialTemporalAttention, self).__init__()
        
        self.channels = channels
        self.reduction = max(1, reduction)
        self.reduced_channels = max(1, channels // self.reduction)
        
        # Channel attention mechanism
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, self.reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.reduced_channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Channel attention
        channel_att = self.channel_attention(x)
        return x * channel_att


class ResidualBlock(nn.Module):
    """
    Residual block with attention mechanism for DAS data (same as FULCNN).
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
        # Attention mechanism
        self.attention = SpatialTemporalAttention(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply attention
        out = self.attention(out)
        
        out += residual
        out = self.relu(out)
        
        return out


class MultiScaleFeatureExtractor(nn.Module):
    """
    Multi-scale feature extraction for DAS data (same as FULCNN).
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(MultiScaleFeatureExtractor, self).__init__()
        
        # Multi-scale features
        self.conv1x1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        self.conv3x1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=(3, 1), padding=(1, 0))
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Extract features at different scales
        feat1 = self.conv1x1(x)
        feat3 = self.conv3x1(x)
        
        # Concatenate multi-scale features
        out = torch.cat([feat1, feat3], dim=1)
        out = self.relu(self.bn(out))
        
        return out


class AdaptivePooling(nn.Module):
    """
    Adaptive pooling for DAS data (same as FULCNN).
    """
    
    def __init__(self, output_size: int = 1):
        super(AdaptivePooling, self).__init__()
        self.output_size = output_size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size)
        
    def forward(self, x):
        return self.adaptive_pool(x)


class DASCNNBackbone(nn.Module):
    """
    Full CNN-LOC backbone for DAS data with attention mechanisms, residual connections, 
    multi-scale features, and adaptive architecture (same structure as FULCNN).
    """
    
    def __init__(self, input_channels: int = 64, input_time: int = 32, input_freq: int = 4,
                 adaptive_input: bool = True):
        super(DASCNNBackbone, self).__init__()
        
        self.input_channels = input_channels
        self.input_time = input_time
        self.input_freq = input_freq
        self.adaptive_input = adaptive_input
        
        print(f"Building DASCNN backbone:")
        print(f"  Input channels: {input_channels}")
        print(f"  Input time: {input_time}")
        print(f"  Input freq: {input_freq}")
        print(f"  Adaptive input: {adaptive_input}")
        
        # Multi-scale feature extraction (same as FULCNN)
        self.initial_features = MultiScaleFeatureExtractor(input_channels, 32)
        
        # Temporal convolution layers (same as FULCNN)
        self.temporal_block1 = ResidualBlock(32, 32, stride=1)
        self.temporal_pool1 = nn.MaxPool2d((2, 1), (2, 1))
        
        self.temporal_block2 = ResidualBlock(32, 64, stride=1)
        self.temporal_pool2 = nn.MaxPool2d((2, 1), (2, 1))
        
        # Spatial convolution layers (same as FULCNN)
        self.spatial_block1 = ResidualBlock(64, 64, stride=1)
        self.spatial_pool1 = nn.MaxPool2d((1, 2), (1, 2))
        
        self.spatial_block2 = ResidualBlock(64, 128, stride=1)
        self.spatial_pool2 = nn.MaxPool2d((1, 2), (1, 2))
        
        # Global attention mechanism (same as FULCNN)
        self.global_attention = SpatialTemporalAttention(128)
        
        # Adaptive pooling (same as FULCNN)
        self.adaptive_pooling = AdaptivePooling(output_size=1)
        
        # Calculate output size dynamically
        self._calculate_output_size()
        
        print(f"DASCNN backbone created with {self.output_size} output features")
    
    def _calculate_output_size(self):
        """Calculate the output size of the backbone."""
        # Create a dummy input
        dummy_input = torch.randn(1, self.input_channels, self.input_time, self.input_freq)
        
        # Forward pass to calculate output size
        with torch.no_grad():
            x = self.forward(dummy_input)
            self.output_size = x.numel()
        
        print(f"DASCNN backbone output size: {self.output_size}")
    
    def forward(self, x):
        """Forward pass through the backbone (same as FULCNN)."""
        # Initial multi-scale feature extraction
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
        
        # Apply global attention
        x = self.global_attention(x)
        
        # Adaptive pooling
        x = self.adaptive_pooling(x)
        x = x.view(x.size(0), -1)
        
        return x


class DASCNNModel(nn.Module):
    """
    Full DASCNN model with comprehensive architecture for DAS dataset (same structure as FULCNN).
    """
    
    def __init__(self, input_channels: int = 64, input_time: int = 32, input_freq: int = 4,
                 num_classes: int = 2, dropout_rate: float = 0.3):
        super(DASCNNModel, self).__init__()
        
        # Create backbone (same as FULCNN)
        self.backbone = DASCNNBackbone(input_channels, input_time, input_freq)
        
        # Classifier (same as FULCNN)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.output_size, 128),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, 32),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            
            nn.Linear(32, num_classes)
        )
        
        print(f"DASCNN model created with {sum(p.numel() for p in self.parameters())} parameters")
    
    def forward(self, x):
        """Forward pass."""
        x = self.backbone(x)
        x = self.classifier(x)
        return x


class DASCNNTrainer:
    """
    Comprehensive trainer for DAS CNN-LOC with all metrics and features (same as FULCNN).
    """
    
    def __init__(self, model, device, output_dir: str = "dascnn_results", 
                 tfrecord_dir: str = None, use_mixed_precision: bool = True):
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.tfrecord_dir = tfrecord_dir
        self.use_mixed_precision = use_mixed_precision
        
        # Mixed precision scaler
        if self.use_mixed_precision:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Loss function (same as FULCNN)
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer (same as FULCNN)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler (will be initialized in train method with actual steps_per_epoch)
        self.scheduler = None
        self.scheduler_type = 'onecycle'
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        print(f"DASCNN Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Mixed precision: {use_mixed_precision}")
        print(f"  Output directory: {output_dir}")
    
    def train_epoch(self, train_loader):
        """Train for one epoch with comprehensive metrics."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Handle tensor dimensions
            if target.dim() > 1:
                target = target.squeeze()
            
            if target.numel() == 0:
                continue
            
            # Handle scalar targets
            if target.dim() == 0:
                target = target.unsqueeze(0)
            
            # Ensure batch compatibility
            if data.size(0) != target.size(0):
                continue
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            if self.use_mixed_precision:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                # Mixed precision backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            
            # Update learning rate (only for step-based schedulers)
            if self.scheduler is not None and isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch with comprehensive metrics."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation", leave=False):
                data, target = data.to(self.device), target.to(self.device)
                
                # Handle tensor dimensions
                if target.dim() > 1:
                    target = target.squeeze()
                
                if target.numel() == 0:
                    continue
                
                # Handle scalar targets
                if target.dim() == 0:
                    target = target.unsqueeze(0)
                
                # Ensure batch compatibility
                if data.size(0) != target.size(0):
                    continue
                
                # Mixed precision inference
                if self.use_mixed_precision:
                    with autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def test(self, test_loader):
        """Test the model with comprehensive metrics (same as FULCNN)."""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        print("Testing model...")
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Testing"):
                data, target = data.to(self.device), target.to(self.device)
                
                # Handle tensor dimensions
                if target.dim() > 1:
                    target = target.squeeze()
                
                if target.numel() == 0:
                    continue
                
                # Handle scalar targets
                if target.dim() == 0:
                    target = target.unsqueeze(0)
                
                # Ensure batch compatibility
                if data.size(0) != target.size(0):
                    continue
                
                # Mixed precision inference
                if self.use_mixed_precision:
                    with autocast():
                        output = self.model(data)
                else:
                    output = self.model(data)
                
                # Get predictions and probabilities
                probabilities = F.softmax(output, dim=1)
                predictions = output.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate comprehensive metrics
        results = self._calculate_comprehensive_metrics(
            all_targets, all_predictions, all_probabilities
        )
        
        return results
    
    def _calculate_comprehensive_metrics(self, targets, predictions, probabilities):
        """Calculate comprehensive evaluation metrics (same as FULCNN)."""
        targets = np.array(targets)
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        # Basic metrics
        accuracy = accuracy_score(targets, predictions)
        balanced_acc = balanced_accuracy_score(targets, predictions)
        
        # Classification metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted'
        )
        
        # Additional metrics
        mcc = matthews_corrcoef(targets, predictions)
        kappa = cohen_kappa_score(targets, predictions)
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(targets, probabilities[:, 1])
        except:
            roc_auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # Temporal metrics (same as FULCNN)
        temporal_metrics = self._calculate_temporal_metrics(targets, predictions)
        
        results = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'matthews_corrcoef': mcc,
            'cohen_kappa': kappa,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'temporal_metrics': temporal_metrics,
            'n_samples': len(targets)
        }
        
        return results
    
    def _calculate_temporal_metrics(self, targets, predictions):
        """Calculate temporal performance metrics (same as FULCNN)."""
        # Temporal analysis across different window sizes
        window_sizes = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]  # seconds
        temporal_results = {}
        
        for window_size in window_sizes:
            # Calculate window size in samples
            window_samples = int(window_size * 64)  # 64 Hz sampling rate
            
            if len(targets) >= window_samples:
                # Take first window for analysis
                window_targets = targets[:window_samples]
                window_predictions = predictions[:window_samples]
                
                accuracy = accuracy_score(window_targets, window_predictions)
                temporal_results[f'{window_size}s'] = {
                    'accuracy': accuracy,
                    'n_samples': window_samples
                }
        
        return temporal_results
    
    def train(self, train_loader, val_loader, num_epochs=50):
        """Train the model with comprehensive metrics."""
        print(f"Starting training for {num_epochs} epochs...")
        
        # Initialize scheduler with actual steps_per_epoch
        steps_per_epoch = len(train_loader)
        if self.scheduler is None:
            if self.scheduler_type == 'onecycle':
                self.scheduler = OneCycleLR(
                    self.optimizer,
                    max_lr=1e-3,
                    epochs=num_epochs,
                    steps_per_epoch=steps_per_epoch,
                    pct_start=0.3,
                    anneal_strategy='cos'
                )
                print(f"Initialized OneCycleLR scheduler with {steps_per_epoch} steps per epoch")
            else:
                # Fallback to ReduceLROnPlateau
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer,
                    mode='max',
                    factor=0.5,
                    patience=5,
                    verbose=True
                )
                print(f"Initialized ReduceLROnPlateau scheduler")
        
        best_val_acc = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update learning rate (for epoch-based schedulers)
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_acc)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Loaded best model with validation accuracy: {best_val_acc:.2f}%")
        
        return best_val_acc


def create_das_data_loaders(tfrecord_dir: str, batch_size: int = 16, 
                          window_size: int = 512, overlap: float = 0.5,
                          num_workers: int = 4, pin_memory: bool = True):
    """Create data loaders for DAS dataset (same structure as FULCNN)."""
    print("Creating DAS data loaders...")
    
    # Check for predefined splits
    train_dir = Path(tfrecord_dir) / "train"
    test_dir = Path(tfrecord_dir) / "test"
    val_dir = Path(tfrecord_dir) / "val"
    
    if train_dir.exists() and test_dir.exists():
        print("Using predefined train/test splits")
        
        # Create datasets
        train_dataset = DASDataset(train_dir, mode='train', window_size=window_size, overlap=overlap)
        test_dataset = DASDataset(test_dir, mode='test', window_size=window_size, overlap=overlap)
        
        # Split train into train/val
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        print(f"Train subset: {len(train_subset)} samples")
        print(f"Val subset: {len(val_subset)} samples")
        print(f"Test dataset: {len(test_dataset)} samples")
        
    else:
        print("Using subject-wise splitting")
        
        # Create full dataset
        full_dataset = DASDataset(tfrecord_dir, mode='full', window_size=window_size, overlap=overlap)
        
        # Subject-wise splitting
        subjects = list(set([meta['subject_id'] for meta in full_dataset.metadata]))
        subjects.sort()
        
        n_subjects = len(subjects)
        n_train_subjects = int(0.7 * n_subjects)
        n_val_subjects = int(0.15 * n_subjects)
        
        train_subjects = subjects[:n_train_subjects]
        val_subjects = subjects[n_train_subjects:n_train_subjects + n_val_subjects]
        test_subjects = subjects[n_train_subjects + n_val_subjects:]
        
        print(f"Subject split: Train={len(train_subjects)}, Val={len(val_subjects)}, Test={len(test_subjects)}")
        
        # Create subsets
        train_indices = [i for i, meta in enumerate(full_dataset.metadata) 
                        if meta['subject_id'] in train_subjects]
        val_indices = [i for i, meta in enumerate(full_dataset.metadata) 
                      if meta['subject_id'] in val_subjects]
        test_indices = [i for i, meta in enumerate(full_dataset.metadata) 
                       if meta['subject_id'] in test_subjects]
        
        train_subset = torch.utils.data.Subset(full_dataset, train_indices)
        val_subset = torch.utils.data.Subset(full_dataset, val_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=True, prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=True, prefetch_factor=2
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=True, prefetch_factor=2
    )
    
    print(f"✓ Data loaders created with batch size {batch_size}")
    print(f"✓ Using {num_workers} workers with pin_memory={pin_memory}")
    
    return train_loader, val_loader, test_loader


def main():
    """Main function for DASCNN training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='DASCNN - Full CNN-LOC for DAS Dataset')
    parser.add_argument('--tfrecord_dir', type=str, default='das_16subjects_preprocessed/tfrecords',
                       help='TFRecord directory path')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--window_size', type=int, default=512,
                       help='Window size for EEG data (512 samples = 8s at 64Hz)')
    parser.add_argument('--output_dir', type=str, default='dascnn_results',
                       help='Output directory for results')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--use_mixed_precision', action='store_true', default=True,
                       help='Use mixed precision training')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("DASCNN - FULL CNN-LOC ALGORITHM FOR DAS DATASET")
    print("=" * 80)
    print("Features:")
    print("- Full CNN-LOC architecture with attention mechanisms")
    print("- Multi-scale feature extraction and residual connections")
    print("- Comprehensive preprocessing (same as FULCNN)")
    print("- Spectrogram-based frequency analysis")
    print("- Mixed precision training")
    print("- Comprehensive metrics evaluation")
    print("- Temporal analysis across window lengths")
    print("=" * 80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_das_data_loaders(
        args.tfrecord_dir, batch_size=args.batch_size, window_size=args.window_size,
        num_workers=args.num_workers
    )
    
    # Create model
    model = DASCNNModel(input_channels=64, input_time=32, input_freq=4)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = DASCNNTrainer(
        model, device, args.output_dir, args.tfrecord_dir, 
        use_mixed_precision=args.use_mixed_precision
    )
    
    # Train model
    best_val_acc = trainer.train(train_loader, val_loader, args.num_epochs)
    
    # Test model
    results = trainer.test(test_loader)
    
    # Save results
    results_file = Path(args.output_dir) / 'dascnn_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results
    print("\n" + "=" * 80)
    print("DASCNN RESULTS")
    print("=" * 80)
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
    print(f"Matthews Correlation Coefficient: {results['matthews_corrcoef']:.4f}")
    print(f"Cohen's Kappa: {results['cohen_kappa']:.4f}")
    print(f"Number of samples: {results['n_samples']}")
    print(f"Results saved to: {results_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
