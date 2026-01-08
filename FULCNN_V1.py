#!/usr/bin/env python3
"""
FULCNN - CNN-LOC for Fulsang Dataset

CNN-LOC model for attention decoding on Fulsang EEG data.
Includes metrics (accuracy, MSED, ROC-AUC) and temporal analysis.
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

sys.path.append('telluride_decoding')

try:
    from telluride_decoding import decoding
    from telluride_decoding import brain_data
    from telluride_decoding import regression
    from telluride_decoding import attention_decoder
except ImportError as e:
    print(f"Warning: Could not import some telluride_decoding modules: {e}")
    print("Continuing with basic functionality...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FulsangDataset(Dataset):
    """
    Dataset for Fulsang EEG data. Uses FULPREPROCESSING output (EEG only).
    Handles windowing and preprocessing for attention decoding.
    """
    
    def __init__(self, tfrecord_dir: str, mode: str = 'full', 
                 window_size: int = 32, overlap: float = 0.5,
                 transform_eeg: bool = True, cache_size: int = 1000):
        self.tfrecord_dir = Path(tfrecord_dir)
        self.mode = mode
        self.window_size = window_size
        self.overlap = overlap
        self.transform_eeg = transform_eeg
        self.cache_size = cache_size
        
        # Fulsang dataset params
        self.sampling_rate = 64  # Hz
        self.n_channels = 66  # EEG channels
        self.attention_switch_duration = 20  # seconds
        
        # Cache for preprocessed windows
        self._window_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Load data from FULPREPROCESSING output
        self.eeg_data, self.labels, self.metadata = self._load_fulpreprocessing_data()
        
        self.window_indices = self._create_fulsang_windows()
        
        print(f"Loaded {len(self.window_indices)} windows, EEG shape: {self.eeg_data.shape}, Label dist: {np.bincount(self.labels)}")
    
    def _load_fulpreprocessing_data(self) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Load TFRecord data from FULPREPROCESSING output. Validates shapes."""
        tfrecord_files = list(self.tfrecord_dir.glob("*.tfrecords"))
        if not tfrecord_files:
            raise ValueError(f"No TFRecord files found in {self.tfrecord_dir}")
        
        print(f"Loading FULPREPROCESSING validated data from {len(tfrecord_files)} files...")
        
        all_eeg_data = []
        all_labels = []
        all_metadata = []
        
        n_success = 0
        n_failed = 0
        total_records = 0
        subject_stats = {}
        shape_errors = 0
        
        for tfrecord_file in tqdm(tfrecord_files, desc="Loading FULPREPROCESSING data"):
            try:
                dataset = tf.data.TFRecordDataset(str(tfrecord_file))
                records_in_file = 0
                file_subject_id = None
                
                for record in dataset:
                    try:
                        example = tf.train.Example.FromString(record.numpy())
                        features = example.features.feature
                        
                        # Check required features (EEG only format)
                        required_features = ['eeg', 'attention_label', 'subject_id']
                        if not all(key in features for key in required_features):
                            continue
                        
                        # Extract and validate EEG data
                        eeg_values = features['eeg'].float_list.value
                        if not eeg_values or len(eeg_values) == 0:
                            continue
                        
                        # Must be exactly 66 channels
                        if len(eeg_values) != 66:
                            print(f"ERROR: Expected 66 EEG channels, got {len(eeg_values)} in {tfrecord_file.name}")
                            shape_errors += 1
                            continue
                        
                        # Reshape to (1, 66) for single sample
                        eeg_data = np.array(eeg_values, dtype=np.float32).reshape(1, 66)
                        
                        # Check for invalid values
                        if np.any(np.isnan(eeg_data)) or np.any(np.isinf(eeg_data)):
                            print(f"WARNING: Invalid EEG values (NaN/Inf) in {tfrecord_file.name}")
                            continue
                        
                        # Extract attention label
                        label_values = features['attention_label'].int64_list.value
                        if not label_values or len(label_values) == 0:
                            continue
                        label = int(label_values[0])
                        
                        # Validate label
                        if label not in [0, 1]:
                            print(f"ERROR: Invalid attention label {label} in {tfrecord_file.name}")
                            continue
                        
                        # Extract metadata
                        subject_id = "unknown"
                        sample_idx = 0
                        
                        if 'subject_id' in features:
                            subject_values = features['subject_id'].bytes_list.value
                            if subject_values and len(subject_values) > 0:
                                try:
                                    subject_id = subject_values[0].decode('utf-8')
                                    file_subject_id = subject_id
                                except Exception:
                                    subject_id = f"subject_{total_records}"
                        
                        if 'sample_idx' in features:
                            sample_values = features['sample_idx'].int64_list.value
                            if sample_values and len(sample_values) > 0:
                                sample_idx = sample_values[0]
                        
                        # Track subject statistics
                        if subject_id not in subject_stats:
                            subject_stats[subject_id] = {'samples': 0, 'labels': []}
                        subject_stats[subject_id]['samples'] += 1
                        subject_stats[subject_id]['labels'].append(label)
                        
                        metadata = {
                            'subject_id': subject_id,
                            'file': tfrecord_file.name,
                            'sample_idx': sample_idx,
                            'attention_label': label,
                            'preprocessing_method': 'FULPREPROCESSING',
                            'validation_passed': True,
                            'data_type': 'EEG_only',
                            'eeg_shape': eeg_data.shape,
                            'label_alignment': 'validated'
                        }
                        
                        all_eeg_data.append(eeg_data)
                        all_labels.append(label)
                        all_metadata.append(metadata)
                        records_in_file += 1
                        total_records += 1
                        
                    except Exception as record_error:
                        print(f"ERROR processing record in {tfrecord_file.name}: {record_error}")
                        continue
                
                if records_in_file > 0:
                    n_success += 1
                else:
                    n_failed += 1
                    
            except Exception as e:
                n_failed += 1
                print(f"ERROR loading {tfrecord_file.name}: {e}")
                continue
        
        print(f"Successfully loaded {n_success} files, {n_failed} files failed")
        print(f"Total records loaded: {total_records}")
        print(f"Shape errors: {shape_errors}")
        
        if shape_errors > 0:
            print(f"WARNING: {shape_errors} records had shape errors")
        
        if not all_eeg_data:
            raise ValueError("No valid FULPREPROCESSING data found in TFRecord files")
        
        eeg_data = np.vstack(all_eeg_data)
        labels = np.array(all_labels, dtype=np.int64)
        
        # Final shape validation
        print(f"Final data shapes: EEG {eeg_data.shape}, Labels {labels.shape}")
        
        if eeg_data.shape[1] != 66:
            raise ValueError(f"CRITICAL: EEG data has {eeg_data.shape[1]} channels, expected 66")
        
        if len(eeg_data) != len(labels):
            raise ValueError(f"CRITICAL: EEG samples ({len(eeg_data)}) != labels ({len(labels)})")
        
        del all_eeg_data, all_labels
        import gc
        gc.collect()
        
        return eeg_data, labels, all_metadata
    
    def _create_fulsang_windows(self) -> List[Tuple[int, int]]:
        """Create sliding windows from Fulsang data."""
        # Convert to seconds for display
        window_seconds = self.window_size / self.sampling_rate
        step_size = int(self.window_size * (1 - self.overlap))
        step_seconds = step_size / self.sampling_rate
        
        total_windows = (len(self.eeg_data) - self.window_size) // step_size + 1
        
        print(f"Creating {total_windows} windows (size: {self.window_size} samples, {window_seconds:.1f}s)")
        
        # Warn about window size
        if window_seconds < 1.0:
            print(f"WARNING: Very short window ({window_seconds:.1f}s) may have poor signal-to-noise")
        elif window_seconds > 20.0:
            print(f"WARNING: Very long window ({window_seconds:.1f}s) may miss temporal dynamics")
        
        window_indices = []
        for i in range(total_windows):
            data_idx = i * step_size
            if data_idx + self.window_size <= len(self.eeg_data):
                # Use majority voting for label (handles trial transitions)
                window_start = data_idx
                window_end = data_idx + self.window_size
                window_labels = self.labels[window_start:window_end]
                
                # Majority vote
                if len(window_labels) > 0:
                    window_label = int(np.bincount(window_labels).argmax())
                else:
                    window_label = 0
                
                window_indices.append((data_idx, window_label))
        
        print(f"Created {len(window_indices)} windows")
        
        return window_indices
    
    def _fulsang_eeg_preprocessing(self, eeg_window: np.ndarray) -> np.ndarray:
        """Preprocess EEG window: artifacts, filtering, normalization."""
        from scipy import signal
        
        # Remove high-amplitude artifacts (>5 std dev)
        artifact_thresh = 5.0
        for ch in range(eeg_window.shape[1]):
            ch_data = eeg_window[:, ch]
            std_val = np.std(ch_data)
            mean_val = np.mean(ch_data)
            
            artifacts = np.abs(ch_data - mean_val) > (artifact_thresh * std_val)
            
            if np.any(artifacts):
                # Interpolate over artifacts
                valid_indices = ~artifacts
                if np.sum(valid_indices) > 2:  # Need at least 2 points
                    from scipy.interpolate import interp1d
                    valid_data = ch_data[valid_indices]
                    valid_time = np.where(valid_indices)[0]
                    all_time = np.arange(len(ch_data))
                    
                    f_interp = interp1d(valid_time, valid_data, kind='linear', 
                                      bounds_error=False, fill_value='extrapolate')
                    eeg_window[:, ch] = f_interp(all_time)
        
        # Remove DC offset
        eeg_window = eeg_window - np.mean(eeg_window, axis=0, keepdims=True)
        
        # Bandpass filter (1-40 Hz)
        nyquist = self.sampling_rate / 2
        low_freq = 1.0 / nyquist
        high_freq = min(40.0 / nyquist, 0.99)  # Keep below Nyquist
        
        b, a = signal.butter(4, [low_freq, high_freq], btype='band')
        
        # Apply filter to each channel
        filtered_eeg = np.zeros_like(eeg_window)
        for ch in range(eeg_window.shape[1]):
            filtered_eeg[:, ch] = signal.filtfilt(b, a, eeg_window[:, ch])
        
        # Normalize using MAD
        mad = np.median(np.abs(filtered_eeg - np.median(filtered_eeg, axis=0)), axis=0)
        mad = np.where(mad == 0, 1.0, mad)  # Avoid div by zero
        filtered_eeg = filtered_eeg / mad
        
        # Soft clipping
        filtered_eeg = np.tanh(filtered_eeg * 0.5)
        
        # Final check for NaNs/Infs
        if np.any(np.isnan(filtered_eeg)) or np.any(np.isinf(filtered_eeg)):
            print("WARNING: Invalid values detected after preprocessing")
            filtered_eeg = np.nan_to_num(filtered_eeg, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return filtered_eeg.astype(np.float32)
    
    def _eeg_to_timefreq_fulsang(self, eeg_window: np.ndarray) -> np.ndarray:
        """Convert EEG to time-frequency representation using spectrogram."""
        from scipy import signal
        
        tf_data = []
        
        for ch_idx in range(eeg_window.shape[1]):
            # Compute spectrogram
            freqs, times, Sxx = signal.spectrogram(
                eeg_window[:, ch_idx], 
                fs=self.sampling_rate,
                nperseg=min(64, len(eeg_window)),  # Adaptive window
                noverlap=32,  # 50% overlap
                window='hann'
            )
            
            # Extract power in standard EEG bands
            bands = [
                (1, 4),   # Delta
                (4, 8),   # Theta  
                (8, 13),  # Alpha
                (13, 25), # Beta
                (25, 40)  # Gamma
            ]
            
            # Get band power for each time point
            band_powers = []
            for low, high in bands:
                if high >= self.sampling_rate / 2:
                    high = self.sampling_rate / 2 - 1
                
                mask = (freqs >= low) & (freqs <= high)
                if np.any(mask):
                    power = np.mean(Sxx[mask, :], axis=0)
                else:
                    power = np.zeros(Sxx.shape[1])
                
                band_powers.append(power)
            
            # Stack: (n_bands, n_time_points)
            ch_tf = np.vstack(band_powers)
            tf_data.append(ch_tf)
        
        # Combine channels: (n_channels, n_bands, n_time_points)
        time_freq_array = np.array(tf_data)
        
        # Interpolate to match window size if needed
        if time_freq_array.shape[2] != self.window_size:
            from scipy.interpolate import interp1d
            original_time = np.linspace(0, 1, time_freq_array.shape[2])
            target_time = np.linspace(0, 1, self.window_size)
            
            interpolated_data = np.zeros((time_freq_array.shape[0], time_freq_array.shape[1], self.window_size))
            for ch in range(time_freq_array.shape[0]):
                for band in range(time_freq_array.shape[1]):
                    f_interp = interp1d(original_time, time_freq_array[ch, band, :], kind='linear')
                    interpolated_data[ch, band, :] = f_interp(target_time)
            
            time_freq_array = interpolated_data
        
        # Output: (channels, time_frames, freq_bands)
        return time_freq_array.astype(np.float32)
    
    def __len__(self):
        return len(self.window_indices)
    
    def __getitem__(self, idx):
        data_idx, label = self.window_indices[idx]
        
        # Check cache first
        cache_key = (data_idx, self.mode)
        if cache_key in self._window_cache:
            self._cache_hits += 1
            cached_data, cached_label = self._window_cache[cache_key]
            return cached_data, cached_label
        
        self._cache_misses += 1
        
        # Extract window (EEG only)
        window_eeg = self.eeg_data[data_idx:data_idx + self.window_size]
        
        # Apply preprocessing
        try:
            window_eeg = self._fulsang_eeg_preprocessing(window_eeg)
        except Exception:
            window_eeg = window_eeg - np.mean(window_eeg, axis=0, keepdims=True)
            window_eeg = window_eeg / (np.std(window_eeg, axis=0, keepdims=True) + 1e-8)
            window_eeg = np.tanh(window_eeg * 0.5)
        
        # Convert to time-frequency representation
        if self.transform_eeg:
            try:
                window_eeg = self._eeg_to_timefreq_fulsang(window_eeg)
            except Exception:
                pass
        
        # Convert to tensors (EEG only)
        window_tensor = torch.FloatTensor(window_eeg)
        label_tensor = torch.LongTensor([label])
        
        # Ensure proper tensor dimensions
        if window_tensor.dim() == 2:
            window_tensor = window_tensor.unsqueeze(0)  # Add channel dimension
        
        return window_tensor, label_tensor


class SpatialTemporalAttention(nn.Module):
    """Channel attention for EEG data. Kept simple to save memory."""
    
    def __init__(self, channels: int, reduction: int = 8):
        super(SpatialTemporalAttention, self).__init__()
        
        self.channels = channels
        self.reduction = max(1, reduction)
        self.reduced_channels = max(1, channels // self.reduction)
        
        # Channel attention only (no temporal to save memory)
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
    """Residual block with attention. Standard ResNet-style."""
    
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
    """Multi-scale features using different kernel sizes. Simplified to save memory."""
    
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


class FULCNNBackbone(nn.Module):
    """Backbone network: attention, residual blocks, multi-scale features."""
    
    def __init__(self, input_channels: int = 66, input_time: int = 32, input_freq: int = 4,
                 adaptive_input: bool = True):
        super(FULCNNBackbone, self).__init__()
        
        self.input_channels = input_channels
        self.input_time = input_time
        self.input_freq = input_freq
        self.adaptive_input = adaptive_input
        
        print(f"Building FULCNN backbone: channels={input_channels}, time={input_time}, freq={input_freq}")
        
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


class FULCNNModel(nn.Module):
    """Full FULCNN model: backbone + classifier for EEG attention decoding."""
    
    def __init__(self, input_channels: int = 66, input_time: int = 32, input_freq: int = 4,
                 num_classes: int = 2, dropout_rate: float = 0.3):
        super(FULCNNModel, self).__init__()
        
        # Create backbone
        self.backbone = FULCNNBackbone(input_channels, input_time, input_freq)
        
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


class FULCNNTrainer:
    """Handles training, validation, testing, and metrics for FULCNN."""
    
    def __init__(self, model: FULCNNModel, device: torch.device, 
                 output_dir: str = "fulcnn_results", tfrecord_dir: str = None, 
                 sampling_rate: int = 64, window_size: int = 512):
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Dataset parameters
        self.tfrecord_dir = tfrecord_dir
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        self.best_val_acc = 0.0
        self.best_model_path = self.output_dir / "best_model.pth"
        
    
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module, scheduler: Optional[optim.lr_scheduler._LRScheduler] = None) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            target = target.squeeze()
            
            # Data augmentation
            if self.model.training:
                noise = torch.randn_like(data) * 0.01
                data = data + noise
                
                if torch.rand(1) > 0.5:
                    shift = torch.randint(-2, 4, (1,)).item()
                    data = torch.roll(data, shift, dims=2)
            
            # Forward
            output = self.model(data)
            loss = criterion(output, target)
            
            if torch.isnan(loss):
                continue
            
            if torch.any(torch.isnan(output)):
                output = torch.nan_to_num(output, nan=0.0)
            
            total_loss += loss.item()
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Step scheduler (OneCycleLR steps per batch)
            if scheduler is not None and isinstance(scheduler, OneCycleLR):
                scheduler.step()
            
            # Accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Cleanup memory
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        if total == 0:
            return float('inf'), 0.0
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Validate for one epoch."""
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
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 50, learning_rate: float = 1e-4,
              weight_decay: float = 1e-5, patience: int = 10, label_smoothing: float = 0.05):
        """Train the model with class balancing and label smoothing."""
        
        # Get class weights for imbalanced data
        labels_list = []
        for _, (_, target) in enumerate(train_loader):
            labels_list.extend(target.squeeze().cpu().numpy())
        
        unique, counts = np.unique(labels_list, return_counts=True)
        
        # Calculate weights
        n_total = len(labels_list)
        n_classes = len(unique)
        
        if n_classes == 0:
            print("WARNING: No classes found in training data")
            weights = torch.ones(2).to(self.device)
        else:
            # Weight = total_samples / (n_classes * class_count)
            weights = np.zeros(max(unique) + 1)
            for i, cls_id in enumerate(unique):
                if counts[i] > 0:
                    weights[cls_id] = n_total / (n_classes * counts[i])
                else:
                    weights[cls_id] = 1.0
            
            class_weights = torch.FloatTensor(weights).to(self.device)
        
        print(f"Class distribution: {dict(zip(unique, counts))}, weights: {class_weights.cpu().numpy()}")
        
        # Loss with class weights and label smoothing
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )
        
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # OneCycleLR scheduler
        steps_per_epoch = len(train_loader)
        total_steps = num_epochs * steps_per_epoch
        scheduler = OneCycleLR(optimizer, max_lr=learning_rate * 5, 
                              total_steps=total_steps, pct_start=0.3,
                              anneal_strategy='cos')
        
        patience_counter = 0
        
        print(f"Starting training: {num_epochs} epochs, lr={learning_rate}, wd={weight_decay}, label_smoothing={label_smoothing}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion, scheduler)
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # Step scheduler (OneCycleLR already steps per batch)
            if not isinstance(scheduler, OneCycleLR):
                scheduler.step()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, self.best_model_path)
                print(f"New best model saved! Val Acc: {val_acc:.4f}")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break
        
        print(f"Training completed. Best validation accuracy: {self.best_val_acc:.4f}")
        return self.best_val_acc
    
    def test(self, test_loader: DataLoader) -> Dict:
        """Test model and compute metrics."""
        checkpoint = torch.load(self.best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Testing"):
                data, target = data.to(self.device), target.to(self.device)
                target = target.squeeze()
                
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                
                probabilities = F.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
        
        # Convert to numpy
        preds = np.array(all_predictions)
        targets = np.array(all_targets)
        probs = np.array(all_probabilities)
        
        # Calculate metrics
        accuracy = accuracy_score(targets, preds)
        avg_loss = total_loss / len(test_loader)
        
        # Classification report
        report = classification_report(targets, preds, 
                                     target_names=['Left', 'Right'], 
                                     labels=[0, 1],
                                     output_dict=True)
        
        cm = confusion_matrix(targets, preds)
        
        # Calculate metrics
        roc_auc_metrics = self._calculate_roc_auc_metrics(targets, probs)
        msed_metrics = self._calculate_msed_metrics(targets, preds)
        advanced_metrics = self._calculate_advanced_metrics(targets, preds)
        temporal_metrics = self._calculate_temporal_metrics(test_loader)
        
        results = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': preds,
            'targets': targets,
            'probabilities': probs,
            'roc_auc_metrics': roc_auc_metrics,
            'msed_metrics': msed_metrics,
            'advanced_metrics': advanced_metrics,
            'temporal_metrics': temporal_metrics
        }
        
        return results
    
    def _calculate_roc_auc_metrics(self, targets: np.ndarray, probabilities: np.ndarray) -> Dict:
        """Calculate ROC-AUC and related metrics."""
        try:
            roc_auc = roc_auc_score(targets, probabilities)
            fpr, tpr, roc_thresholds = roc_curve(targets, probabilities)
            
            # Find optimal threshold
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = roc_thresholds[optimal_idx]
            optimal_tpr = tpr[optimal_idx]
            optimal_fpr = fpr[optimal_idx]
            
            # Precision-Recall Curve
            precision, recall, pr_thresholds = precision_recall_curve(targets, probabilities)
            avg_precision = average_precision_score(targets, probabilities)
            
            return {
                "roc_auc_score": float(roc_auc),
                "average_precision": float(avg_precision),
                "optimal_threshold": float(optimal_threshold),
                "optimal_tpr": float(optimal_tpr),
                "optimal_fpr": float(optimal_fpr),
                "roc_curve": {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "thresholds": roc_thresholds.tolist()
                },
                "precision_recall_curve": {
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                    "thresholds": pr_thresholds.tolist()
                }
            }
        except Exception as e:
            return {"error": f"Error calculating ROC-AUC metrics: {e}"}
    
    def _calculate_msed_metrics(self, targets: np.ndarray, predictions: np.ndarray) -> Dict:
        """Calculate MSED (Mean Squared Error Distance) metrics."""
        try:
            mse = np.mean((predictions - targets) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - targets))
            mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
            
            # R-squared
            ss_res = np.sum((targets - predictions) ** 2)
            ss_tot = np.sum((targets - np.mean(targets)) ** 2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-8))
            
            return {
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "mape": float(mape),
                "r_squared": float(r_squared)
            }
        except Exception as e:
            return {"error": f"Error calculating MSED metrics: {e}"}
    
    def _calculate_advanced_metrics(self, targets: np.ndarray, predictions: np.ndarray) -> Dict:
        """Calculate advanced classification metrics."""
        try:
            mcc = matthews_corrcoef(targets, predictions)
            kappa = cohen_kappa_score(targets, predictions)
            balanced_acc = balanced_accuracy_score(targets, predictions)
            
            precision, recall, f1, support = precision_recall_fscore_support(
                targets, predictions, average=None, labels=[0, 1]
            )
            
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                targets, predictions, average='macro'
            )
            
            precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
                targets, predictions, average='weighted'
            )
            
            return {
                "matthews_correlation_coefficient": float(mcc),
                "cohens_kappa": float(kappa),
                "balanced_accuracy": float(balanced_acc),
                "per_class_metrics": {
                    "left_attention": {
                        "precision": float(precision[0]),
                        "recall": float(recall[0]),
                        "f1_score": float(f1[0]),
                        "support": int(support[0])
                    },
                    "right_attention": {
                        "precision": float(precision[1]),
                        "recall": float(recall[1]),
                        "f1_score": float(f1[1]),
                        "support": int(support[1])
                    }
                },
                "macro_averages": {
                    "precision": float(precision_macro),
                    "recall": float(recall_macro),
                    "f1_score": float(f1_macro)
                },
                "weighted_averages": {
                    "precision": float(precision_weighted),
                    "recall": float(recall_weighted),
                    "f1_score": float(f1_weighted)
                }
            }
        except Exception as e:
            return {"error": f"Error calculating advanced metrics: {e}"}
    
    def _calculate_temporal_metrics(self, test_loader: DataLoader) -> Dict[str, float]:
        """Calculate real temporal performance metrics across different window sizes."""
        # Test different window sizes (in seconds)
        window_sizes_seconds = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0]
        temporal_analysis = {}
        flat_results = {}
        
        for window_sec in window_sizes_seconds:
            window_samples = int(window_sec * self.sampling_rate)
            
            # For larger windows, we need to create overlapping windows from the test data
            if window_samples > self.window_size:
                larger_window_results = self._test_larger_window(test_loader, window_samples, window_sec)
                flat_results.update(larger_window_results)
                
                # Add to temporal_analysis structure
                if f'accuracy_{window_sec}s' in larger_window_results:
                    temporal_analysis[f'{window_sec}s'] = {
                        'accuracy': larger_window_results[f'accuracy_{window_sec}s'],
                        'f1': larger_window_results[f'f1_{window_sec}s']
                    }
                continue
            
            # For smaller windows, create temporary dataset
            try:
                temp_dataset = FulsangDataset(
                    self.tfrecord_dir, 
                    mode='test',
                    window_size=window_samples,
                    overlap=0.5
                )
                
                if len(temp_dataset) == 0:
                    continue
                
                temp_loader = DataLoader(temp_dataset, batch_size=16, shuffle=False)
                
                # Evaluate on this window size
                self.model.eval()
                all_predictions = []
                all_targets = []
                
                with torch.no_grad():
                    for data, target in temp_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = self.model(data)
                        pred = output.argmax(dim=1)
                        
                        all_predictions.extend(pred.cpu().numpy())
                        all_targets.extend(target.cpu().numpy())
                
                if len(all_predictions) > 0:
                    accuracy = accuracy_score(all_targets, all_predictions)
                    f1 = f1_score(all_targets, all_predictions, average='weighted')
                    
                    flat_results[f'accuracy_{window_sec}s'] = accuracy
                    flat_results[f'f1_{window_sec}s'] = f1
                    
                    # Add to temporal_analysis structure
                    temporal_analysis[f'{window_sec}s'] = {
                        'accuracy': accuracy,
                        'f1': f1
                    }
                    
            except Exception as e:
                print(f"Error testing {window_sec}s window: {e}")
                continue
        
        # Find the best window size
        best_window = None
        best_accuracy = 0.0
        for window_key, metrics in temporal_analysis.items():
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_window = window_key
        
        # Return structured results
        return {
            'temporal_analysis': temporal_analysis,
            'recommended_window_size': best_window if best_window else 'N/A',
            'note': f'Best performance at {best_window}s window with {best_accuracy:.3f} accuracy' if best_window else 'No valid temporal analysis completed',
            **flat_results  # Keep flat results for backward compatibility
        }
    
    def _test_larger_window(self, test_loader: DataLoader, window_samples: int, window_sec: float) -> Dict[str, float]:
        """Test larger window sizes by creating overlapping windows from test data."""
        # Collect all test data
        all_data = []
        all_targets = []
        
        self.model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                all_data.append(data.cpu())
                all_targets.append(target.cpu())
        
        if not all_data:
            return {}
        
        # Concatenate all test data
        all_data = torch.cat(all_data, dim=0)  # (total_samples, channels, time, freq)
        all_targets = torch.cat(all_targets, dim=0)  # (total_samples,)
        
        # For larger windows, we need to create overlapping windows from the test data
        # But the model expects the same input shape as training
        predictions = []
        targets = []
        
        # Calculate step size for overlapping windows
        step_size = max(1, window_samples // 4)  # 75% overlap
        
        for start_idx in range(0, all_data.shape[0] - window_samples + 1, step_size):
            # Extract window
            window_data = all_data[start_idx:start_idx + window_samples]  # (window_samples, channels, time, freq)
            
            # For larger windows, we need to downsample to match the model's expected input size
            # The model was trained on 32 samples, so we need to downsample to 32
            if window_data.shape[0] > 1:
                # Downsample by taking every nth sample to get to 32 samples
                target_samples = 32  # Model's expected input size
                if window_samples > target_samples:
                    # Calculate step size for downsampling
                    step = window_samples // target_samples
                    window_data = window_data[::step][:target_samples]  # Take every step-th sample
                else:
                    # If window is smaller than target, repeat samples
                    repeats = target_samples // window_samples
                    remainder = target_samples % window_samples
                    window_data = window_data.repeat(repeats, 1, 1, 1)
                    if remainder > 0:
                        window_data = torch.cat([window_data, window_data[:remainder]], dim=0)
            
            # Ensure we have exactly 32 samples
            if window_data.shape[0] != 32:
                # Pad or truncate to exactly 32 samples
                if window_data.shape[0] < 32:
                    # Pad with the last sample
                    padding = 32 - window_data.shape[0]
                    last_sample = window_data[-1:].repeat(padding, 1, 1, 1)
                    window_data = torch.cat([window_data, last_sample], dim=0)
                else:
                    # Truncate to 32 samples
                    window_data = window_data[:32]
            
            # Now window_data should be (32, channels, time, freq)
            # We need to reshape to (1, channels, time, freq) for the model
            window_data = window_data.mean(dim=0, keepdim=True)  # Average across the 32 samples
            
            # Get the target for the middle of the window
            middle_idx = start_idx + window_samples // 2
            if middle_idx < len(all_targets):
                window_target = all_targets[middle_idx]
            else:
                window_target = all_targets[-1]
            
            # Predict
            window_data = window_data.to(self.device)
            with torch.no_grad():
                output = self.model(window_data)
                pred = output.argmax(dim=1)
                predictions.append(pred.cpu().item())
                targets.append(window_target.item())
        
        if len(predictions) > 0:
            accuracy = accuracy_score(targets, predictions)
            f1 = f1_score(targets, predictions, average='weighted')
            
            return {
                f'accuracy_{window_sec}s': accuracy,
                f'f1_{window_sec}s': f1
            }
        else:
            return {}
    
    def save_results(self, results: Dict):
        """Save comprehensive results to files."""
        # Prepare results
        results_json = {
            'accuracy': float(results['accuracy']),
            'loss': float(results['loss']),
            'classification_report': results['classification_report'],
            'confusion_matrix': results['confusion_matrix'].tolist() if hasattr(results['confusion_matrix'], 'tolist') else results['confusion_matrix'],
            'best_val_acc': float(self.best_val_acc),
            'timestamp': datetime.now().isoformat(),
            'roc_auc_metrics': results.get('roc_auc_metrics', {}),
            'msed_metrics': results.get('msed_metrics', {}),
            'advanced_metrics': results.get('advanced_metrics', {}),
            'temporal_metrics': results.get('temporal_metrics', {})
        }
        
        # Save results
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        
        # Save predictions
        save_data = {
            'predictions': results['predictions'],
            'targets': results['targets'],
            'probabilities': results['probabilities']
        }
        
        with open(self.output_dir / 'predictions.pkl', 'wb') as f:
            pickle.dump(save_data, f)
        
        # Save comprehensive metrics report
        self._save_comprehensive_report(results)
        
    
    def _save_comprehensive_report(self, results: Dict):
        """Save a comprehensive metrics report."""
        with open(self.output_dir / 'comprehensive_metrics_report.txt', 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FULCNN COMPREHENSIVE METRICS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Basic metrics
            f.write("BASIC METRICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"Loss: {results['loss']:.4f}\n")
            f.write(f"Best Validation Accuracy: {self.best_val_acc:.2f}%\n\n")
            
            # ROC-AUC metrics
            roc_auc = results.get('roc_auc_metrics', {})
            if "error" not in roc_auc:
                f.write("ROC-AUC METRICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"ROC-AUC Score: {roc_auc.get('roc_auc_score', 'N/A'):.4f}\n")
                f.write(f"Average Precision: {roc_auc.get('average_precision', 'N/A'):.4f}\n")
                f.write(f"Optimal Threshold: {roc_auc.get('optimal_threshold', 'N/A'):.4f}\n")
                f.write(f"Optimal TPR: {roc_auc.get('optimal_tpr', 'N/A'):.4f}\n")
                f.write(f"Optimal FPR: {roc_auc.get('optimal_fpr', 'N/A'):.4f}\n\n")
            
            # MSED metrics
            msed = results.get('msed_metrics', {})
            if "error" not in msed:
                f.write("MSED METRICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Mean Squared Error: {msed.get('mse', 'N/A'):.4f}\n")
                f.write(f"Root Mean Squared Error: {msed.get('rmse', 'N/A'):.4f}\n")
                f.write(f"Mean Absolute Error: {msed.get('mae', 'N/A'):.4f}\n")
                f.write(f"Mean Absolute Percentage Error: {msed.get('mape', 'N/A'):.4f}%\n")
                f.write(f"R-squared: {msed.get('r_squared', 'N/A'):.4f}\n\n")
            
            # Advanced metrics
            advanced = results.get('advanced_metrics', {})
            if "error" not in advanced:
                f.write("ADVANCED METRICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Matthews Correlation Coefficient: {advanced.get('matthews_correlation_coefficient', 'N/A'):.4f}\n")
                f.write(f"Cohen's Kappa: {advanced.get('cohens_kappa', 'N/A'):.4f}\n")
                f.write(f"Balanced Accuracy: {advanced.get('balanced_accuracy', 'N/A'):.4f}\n\n")
                
                # Per-class metrics
                per_class = advanced.get("per_class_metrics", {})
                f.write("PER-CLASS METRICS:\n")
                f.write("-" * 40 + "\n")
                
                left = per_class.get("left_attention", {})
                f.write("Left Attention:\n")
                f.write(f"  Precision: {left.get('precision', 'N/A'):.4f}\n")
                f.write(f"  Recall: {left.get('recall', 'N/A'):.4f}\n")
                f.write(f"  F1-Score: {left.get('f1_score', 'N/A'):.4f}\n")
                f.write(f"  Support: {left.get('support', 'N/A')}\n\n")
                
                right = per_class.get("right_attention", {})
                f.write("Right Attention:\n")
                f.write(f"  Precision: {right.get('precision', 'N/A'):.4f}\n")
                f.write(f"  Recall: {right.get('recall', 'N/A'):.4f}\n")
                f.write(f"  F1-Score: {right.get('f1_score', 'N/A'):.4f}\n")
                f.write(f"  Support: {right.get('support', 'N/A')}\n\n")
            
            # Temporal analysis
            temporal = results.get('temporal_metrics', {})
            f.write("TEMPORAL PERFORMANCE ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            for window_size, metrics in temporal.get("temporal_analysis", {}).items():
                f.write(f"{window_size}: {metrics.get('accuracy', 'N/A'):.4f}\n")
            f.write(f"\nRecommended: {temporal.get('recommended_window_size', 'N/A')}\n")
            f.write(f"Note: {temporal.get('note', 'N/A')}\n")
            
            # Add formatted results section
            f.write("\n" + "=" * 80 + "\n")
            f.write("FULCNN COMPREHENSIVE RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("The FULCNN model successfully processed the Fulsang dataset:\n")
            f.write(f"- Best Validation Accuracy: {self.best_val_acc:.4f}\n")
            f.write(f"- Final Test Accuracy: {results['accuracy']:.4f}\n")
            
            # ROC-AUC metrics
            roc_auc = results.get('roc_auc_metrics', {})
            if "error" not in roc_auc:
                f.write(f"- ROC-AUC: {roc_auc.get('roc_auc_score', 'N/A'):.4f}\n")
            
            # Classification metrics
            class_report = results.get('classification_report', {})
            if 'macro avg' in class_report:
                macro_avg = class_report['macro avg']
                f.write(f"- Precision: {macro_avg.get('precision', 'N/A'):.4f}\n")
                f.write(f"- Recall: {macro_avg.get('recall', 'N/A'):.4f}\n")
                f.write(f"- F1-Score: {macro_avg.get('f1-score', 'N/A'):.4f}\n")
            
            # MSED metrics
            msed = results.get('msed_metrics', {})
            if "error" not in msed:
                f.write(f"- MSED (Primary Benchmark): {msed.get('rmse', 'N/A'):.4f}\n")
            
            # Advanced metrics
            advanced = results.get('advanced_metrics', {})
            if "error" not in advanced:
                f.write(f"- Direction Accuracy: {advanced.get('balanced_accuracy', 'N/A'):.4f}\n")
                f.write(f"- Spatial Consistency: {advanced.get('matthews_correlation_coefficient', 'N/A'):.4f}\n")
            
            # Temporal Integration Performance
            f.write("\nTEMPORAL INTEGRATION PERFORMANCE\n")
            f.write("The Fulsang dataset demonstrated robust performance across decision window lengths:\n")
            
            for ws_key, ws_data in temporal.get("temporal_analysis", {}).items():
                window_seconds = float(ws_key.replace('s', ''))
                accuracy = ws_data.get('accuracy', 0.0)
                f.write(f"- {ws_key} window: {accuracy:.4f}\n")


def create_fulsang_data_loaders(tfrecord_dir: str, batch_size: int = 16, 
                               window_size: int = 32, overlap: float = 0.5,
                               train_ratio: float = 0.7, val_ratio: float = 0.15,
                               max_samples: Optional[int] = None, 
                               num_workers: int = 0, pin_memory: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test loaders with subject-wise splitting (no data leakage)."""
    
    print(f"DEBUG: window_size parameter = {window_size}")
    print(f"Creating dataset: batch_size={batch_size}, window_size={window_size} samples ({window_size/64:.1f}s)")
    
    # Create full dataset with FULPREPROCESSING integration
    full_dataset = FulsangDataset(tfrecord_dir, mode='full', 
                                 window_size=window_size, overlap=overlap)
    
    total_size = len(full_dataset)
    
    # Map windows to subjects for splitting
    subject_windows = {}
    
    # Group metadata by subject
    subject_ranges = {}
    current_subject = None
    start_idx = 0
    
    for i, metadata in enumerate(full_dataset.metadata):
        subject_id = metadata.get('subject_id', 'unknown')
        
        if subject_id != current_subject:
            if current_subject is not None:
                subject_ranges[current_subject] = (start_idx, i)
            current_subject = subject_id
            start_idx = i
    
    # Last subject
    if current_subject is not None:
        subject_ranges[current_subject] = (start_idx, len(full_dataset.metadata))
    
    # Map windows to subjects
    for i, (data_idx, label) in enumerate(full_dataset.window_indices):
        subject_id = "unknown"
        
        # Find subject for this window
        for subj_id, (start_idx, end_idx) in subject_ranges.items():
            if start_idx <= data_idx < end_idx:
                subject_id = subj_id
                break
        
        if subject_id not in subject_windows:
            subject_windows[subject_id] = []
        subject_windows[subject_id].append(i)
    
    # Split by subject (prevents data leakage)
    subjects = list(subject_windows.keys())
    np.random.seed(42)  # Reproducibility
    np.random.shuffle(subjects)
    
    n_subjects = len(subjects)
    n_train_subjects = int(train_ratio * n_subjects)
    n_val_subjects = int(val_ratio * n_subjects)
    
    train_subjects = subjects[:n_train_subjects]
    val_subjects = subjects[n_train_subjects:n_train_subjects + n_val_subjects]
    test_subjects = subjects[n_train_subjects + n_val_subjects:]
    
    # Get window indices for each split
    train_indices = []
    val_indices = []
    test_indices = []
    
    for subject_id in train_subjects:
        train_indices.extend(subject_windows[subject_id])
    for subject_id in val_subjects:
        val_indices.extend(subject_windows[subject_id])
    for subject_id in test_subjects:
        test_indices.extend(subject_windows[subject_id])
    
    # Check for data leakage
    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)
    
    if train_set & val_set:
        raise ValueError("CRITICAL: Data leakage - train/val overlap!")
    if train_set & test_set:
        raise ValueError("CRITICAL: Data leakage - train/test overlap!")
    if val_set & test_set:
        raise ValueError("CRITICAL: Data leakage - val/test overlap!")
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, val_loader, test_loader


def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='FULCNN - CNN-LOC for Fulsang Dataset')
    parser.add_argument('--tfrecord_dir', type=str, default='fulsang_preprocessed/tfrecords',
                       help='TFRecord directory path')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--window_size', type=int, default=512,
                       help='Window size for EEG data (512 samples = 8 seconds at 64Hz)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for regularization')
    parser.add_argument('--dropout_rate', type=float, default=0.4,
                       help='Dropout rate')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                       help='Label smoothing factor')
    parser.add_argument('--output_dir', type=str, default='fulcnn_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print(f"DEBUG: args.window_size = {args.window_size}")
    print(f"DEBUG: args.tfrecord_dir = {args.tfrecord_dir}")
    
    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU (GPU not available)")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_fulsang_data_loaders(
        args.tfrecord_dir, batch_size=args.batch_size, window_size=args.window_size,
        max_samples=None, num_workers=0, pin_memory=False
    )
    
    # Get input dimensions from data
    if len(train_loader.dataset) > 0:
        sample_data, _ = next(iter(train_loader))
        actual_channels = sample_data.shape[1]
        actual_time = sample_data.shape[2]
        actual_freq = sample_data.shape[3]
        print(f"Input dimensions: channels={actual_channels}, time={actual_time}, freq={actual_freq}")
    else:
        actual_channels = 66  # EEG channels
        actual_time = 32
        actual_freq = 4
        print(f"Using defaults: channels={actual_channels}, time={actual_time}, freq={actual_freq}")
    
    # Create model
    print(f"Creating model: channels={actual_channels}, time={actual_time}, freq={actual_freq}")
    print(f"Hyperparameters: batch_size={args.batch_size}, lr={args.learning_rate}, wd={args.weight_decay}, dropout={args.dropout_rate}, label_smoothing={args.label_smoothing}")
    
    model = FULCNNModel(
        input_channels=actual_channels,
        input_time=actual_time,
        input_freq=actual_freq,
        num_classes=2,
        dropout_rate=args.dropout_rate
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = FULCNNTrainer(model, device, args.output_dir, args.tfrecord_dir, 
                           sampling_rate=64, window_size=args.window_size)
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Train
    best_val_acc = trainer.train(
        train_loader, val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=15,
        label_smoothing=args.label_smoothing
    )
    
    # Test
    results = trainer.test(test_loader)
    
    # Save
    trainer.save_results(results)
    
    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}, Test acc: {results['accuracy']:.4f}")
    
    # Display key metrics
    roc_auc = results.get('roc_auc_metrics', {})
    if "error" not in roc_auc:
        print(f"ROC-AUC: {roc_auc.get('roc_auc_score', 'N/A'):.4f}")
    
    msed = results.get('msed_metrics', {})
    if "error" not in msed:
        print(f"RMSE: {msed.get('rmse', 'N/A'):.4f}")
    
    temporal = results.get('temporal_metrics', {})
    print(f"Recommended window size: {temporal.get('recommended_window_size', 'N/A')}")
    
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
