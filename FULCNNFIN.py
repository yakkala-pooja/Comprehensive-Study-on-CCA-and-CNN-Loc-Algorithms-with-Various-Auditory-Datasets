#!/usr/bin/env python3
"""
FULCNNFIN - CNN-LOC for Fulsang Dataset (Final Version)

CNN-LOC model for attention decoding on Fulsang EEG data using TFRecord files.
Uses the same CNN architecture as CombinedCNNLOC.py.
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

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, Gaussian blur augmentation will be skipped")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Import CNN architecture from CombinedCNNLOC (same as DASCNNFIN)
class SpatialTemporalAttention(nn.Module):
    """Channel attention for EEG data."""
    
    def __init__(self, channels: int, reduction: int = 8):
        super(SpatialTemporalAttention, self).__init__()
        
        self.channels = channels
        self.reduction = max(1, reduction)
        self.reduced_channels = max(1, channels // self.reduction)
        
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
        
        self.conv1x1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        self.conv3x1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=(3, 1), padding=(1, 0))
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        feat1 = self.conv1x1(x)
        feat3 = self.conv3x1(x)
        
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


class FULCNNFINDataset(Dataset):
    """PyTorch Dataset for Fulsang TFRecord files."""
    
    def __init__(self, tfrecord_dir: str, mode: str = 'full', 
                 window_size: int = 512, overlap: float = 0.5,
                 transform_eeg: bool = True, use_64_channels: bool = True,
                 augment: bool = False):
        self.tfrecord_dir = Path(tfrecord_dir)
        self.mode = mode
        self.window_size = window_size
        self.overlap = overlap
        self.transform_eeg = transform_eeg
        self.use_64_channels = use_64_channels
        self.augment = augment
        
        # Fulsang dataset params
        self.sampling_rate = 64  # Hz
        self.n_channels = 64 if use_64_channels else 66  # Use 64 channels for better compatibility
        
        # Load data from TFRecord files
        self.eeg_data, self.labels, self.metadata = self._load_tfrecord_data()
        
        # Create windows
        self.window_indices = self._create_windows()
        
        print(f"\nFULCNNFINDataset initialized:")
        print(f"  Mode: {mode}")
        print(f"  Total windows: {len(self.window_indices)}")
        print(f"  Window size: {self.window_size} samples")
        print(f"  Sampling rate: {self.sampling_rate} Hz")
        print(f"  Channels: {self.n_channels}")
    
    def _load_tfrecord_data(self) -> Tuple[List[np.ndarray], List[int], List[Dict]]:
        """Load Fulsang data from TFRecord files."""
        print(f"Loading Fulsang data from {self.tfrecord_dir}...")
        
        # Find TFRecord files
        tfrecord_files = list(self.tfrecord_dir.glob("*.tfrecords"))
        
        if not tfrecord_files:
            raise ValueError(f"No TFRecord files found in {self.tfrecord_dir}")
        
        print(f"Found {len(tfrecord_files)} TFRecord files")
        
        # Load data from TFRecord files
        eeg_data = []
        labels = []
        metadata = []
        
        current_trial_eeg = []
        current_trial_label = None
        current_subject_id = None
        
        for tfrecord_file in tqdm(tfrecord_files, desc="Loading TFRecord files"):
            try:
                dataset = tf.data.TFRecordDataset(str(tfrecord_file))
                
                for record in dataset:
                    try:
                        example = tf.train.Example.FromString(record.numpy())
                        features = example.features.feature
                        
                        # Required features for Fulsang
                        if 'eeg' not in features or 'attention_label' not in features:
                            continue
                        
                        # Extract EEG data
                        eeg_values = features['eeg'].float_list.value
                        if len(eeg_values) < self.n_channels:
                            continue
                        
                        # Use first 64 channels if using 64-channel mode (better compatibility with Das)
                        if self.use_64_channels and len(eeg_values) >= 64:
                            eeg_values = eeg_values[:64]
                        elif not self.use_64_channels and len(eeg_values) != 66:
                            continue
                        
                        eeg_sample = np.array(eeg_values, dtype=np.float32).reshape(1, self.n_channels)
                        
                        # Extract attention_label
                        label_values = features['attention_label'].int64_list.value
                        if not label_values or len(label_values) == 0:
                            continue
                        label = int(label_values[0])
                        
                        # Validate label
                        if label not in [0, 1]:
                            continue
                        
                        # Extract subject_id
                        subject_id = "unknown"
                        if 'subject_id' in features:
                            subject_id = features['subject_id'].bytes_list.value[0].decode('utf-8')
                        
                        # Group samples by trial (same subject and label)
                        if current_subject_id is None:
                            current_subject_id = subject_id
                            current_trial_label = label
                            current_trial_eeg = [eeg_sample]
                        elif current_subject_id == subject_id and current_trial_label == label:
                            current_trial_eeg.append(eeg_sample)
                        else:
                            # Save previous trial
                            if len(current_trial_eeg) > 0:
                                trial_eeg = np.vstack(current_trial_eeg)
                                eeg_data.append(trial_eeg)
                                labels.append(current_trial_label)
                                metadata.append({
                                    'subject_id': current_subject_id,
                                    'file': tfrecord_file.name
                                })
                            
                            # Start new trial
                            current_subject_id = subject_id
                            current_trial_label = label
                            current_trial_eeg = [eeg_sample]
                    
                    except Exception as e:
                        continue
                
                # Save last trial
                if len(current_trial_eeg) > 0:
                    trial_eeg = np.vstack(current_trial_eeg)
                    eeg_data.append(trial_eeg)
                    labels.append(current_trial_label)
                    metadata.append({
                        'subject_id': current_subject_id,
                        'file': tfrecord_file.name
                    })
                    current_trial_eeg = []
            
            except Exception as e:
                print(f"Error loading {tfrecord_file}: {e}")
                continue
        
        if not eeg_data:
            raise ValueError("No valid Fulsang data loaded")
        
        return eeg_data, np.array(labels), metadata
    
    def _create_windows(self) -> List[Tuple[int, int, int]]:
        """Create sliding windows from loaded data."""
        window_indices = []
        step_size = int(self.window_size * (1 - self.overlap))
        
        current_idx = 0
        for trial_eeg, label in zip(self.eeg_data, self.labels):
            trial_length = trial_eeg.shape[0]
            
            for start_idx in range(0, trial_length - self.window_size + 1, step_size):
                end_idx = start_idx + self.window_size
                window_indices.append((current_idx + start_idx, current_idx + end_idx, label))
            
            current_idx += trial_length
        
        return window_indices
    
    def _transform_eeg(self, eeg_window: np.ndarray) -> np.ndarray:
        """Transform EEG window to time-frequency representation."""
        n_samples, n_channels = eeg_window.shape
        
        freq_bins = 4
        time_frames = 32
        
        if n_samples >= time_frames:
            samples_per_frame = n_samples // time_frames
            eeg_reshaped = eeg_window[:time_frames * samples_per_frame].reshape(
                time_frames, samples_per_frame, n_channels
            )
            
            eeg_fft = np.fft.rfft(eeg_reshaped, axis=1)
            eeg_fft = np.abs(eeg_fft[:, :freq_bins, :])
            
            eeg_tf = np.transpose(eeg_fft, (2, 0, 1))
        else:
            eeg_tf = np.zeros((n_channels, time_frames, freq_bins))
            eeg_tf[:, :n_samples, :] = eeg_window[:n_samples, :, np.newaxis]
        
        return eeg_tf.astype(np.float32)
    
    def __len__(self):
        return len(self.window_indices)
    
    def __getitem__(self, idx):
        start_idx, end_idx, label = self.window_indices[idx]
        
        # Find which trial this window belongs to
        current_idx = 0
        trial_idx = 0
        for i, trial_eeg in enumerate(self.eeg_data):
            trial_length = trial_eeg.shape[0]
            if current_idx + trial_length > start_idx:
                trial_idx = i
                break
            current_idx += trial_length
        
        # Extract window from the correct trial
        trial_start = current_idx
        window_start_in_trial = start_idx - trial_start
        window_end_in_trial = end_idx - trial_start
        
        eeg_window = self.eeg_data[trial_idx][window_start_in_trial:window_end_in_trial].copy()
        
        # VERY aggressive data augmentation (only during training) to prevent overfitting
        if self.augment and self.mode == 'train':
            # Apply multiple augmentations per sample to increase diversity
            
            # 1. Random noise (ALWAYS apply, higher variance)
            noise_std = np.random.uniform(0.01, 0.03)
            noise = np.random.normal(0, noise_std, eeg_window.shape).astype(np.float32)
            eeg_window = eeg_window + noise
            
            # 2. Random channel dropout (30% chance, more aggressive)
            if np.random.rand() < 0.3:
                n_drop = max(1, int(self.n_channels * 0.2))
                drop_channels = np.random.choice(self.n_channels, n_drop, replace=False)
                eeg_window[:, drop_channels] = 0
            
            # 3. Random time shift (ALWAYS apply)
            shift = np.random.randint(-self.window_size // 8, self.window_size // 8 + 1)
            if shift != 0:
                eeg_window = np.roll(eeg_window, shift, axis=0)
            
            # 4. Random scaling (ALWAYS apply, wider range)
            scale = np.random.uniform(0.7, 1.3)
            eeg_window = eeg_window * scale
            
            # 5. Random channel scaling (50% chance)
            if np.random.rand() < 0.5:
                channel_scales = np.random.uniform(0.85, 1.15, (1, self.n_channels))
                eeg_window = eeg_window * channel_scales
            
            # 6. Gaussian blur (20% chance)
            if np.random.rand() < 0.2 and SCIPY_AVAILABLE:
                for ch in range(self.n_channels):
                    eeg_window[:, ch] = ndimage.gaussian_filter1d(eeg_window[:, ch], 
                                                                  sigma=np.random.uniform(0.3, 0.8))
            
            # 7. Random time masking (15% chance)
            if np.random.rand() < 0.15:
                mask_length = int(self.window_size * np.random.uniform(0.05, 0.15))
                mask_start = np.random.randint(0, max(1, self.window_size - mask_length))
                eeg_window[mask_start:mask_start + mask_length, :] = 0
            
            # 8. Mixup augmentation (15% chance) - very effective for small datasets
            # Note: Mixup is complex to implement correctly here, so we'll rely on label smoothing
            # which provides similar regularization benefits
        
        # Enhanced preprocessing
        # 1. Baseline correction
        eeg_window = eeg_window - np.mean(eeg_window, axis=0, keepdims=True)
        
        # 2. Robust normalization (using median for outliers)
        median_vals = np.median(np.abs(eeg_window), axis=0, keepdims=True)
        std_vals = np.std(eeg_window, axis=0, keepdims=True)
        std_vals = np.where(std_vals == 0, 1.0, std_vals)
        
        # Use robust scaling (median-based)
        robust_scale = np.maximum(median_vals * 1.4826, std_vals)  # MAD estimator
        robust_scale = np.where(robust_scale == 0, 1.0, robust_scale)
        eeg_window = eeg_window / robust_scale
        
        # 3. Soft clipping to handle outliers
        eeg_window = np.tanh(eeg_window * 0.5)
        
        # 4. Ensure no invalid values
        eeg_window = np.nan_to_num(eeg_window, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Transform to time-frequency representation
        if self.transform_eeg:
            eeg_tf = self._transform_eeg(eeg_window)
        else:
            eeg_tf = eeg_window.T[:, :, np.newaxis]
        
        # Convert to tensors
        eeg_tensor = torch.FloatTensor(eeg_tf)
        label_tensor = torch.LongTensor([label])
        
        return eeg_tensor, label_tensor


class CNNLOCTrainer:
    """Trainer for CNN-LOC model."""
    
    def __init__(self, model: CNNLOCModel, device: torch.device, output_dir: str = "fulcnnfin_results"):
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
            
            output = self.model(data)
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            # Moderate gradient clipping to prevent overfitting (increased from 0.3 to 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
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
              weight_decay: float = 1e-5, patience: int = 20,
              class_weights: Optional[torch.Tensor] = None,
              label_smoothing: float = 0.2):
        """Train the model."""
        
        # Use label smoothing to prevent overconfidence
        criterion = nn.CrossEntropyLoss(
            weight=class_weights, 
            label_smoothing=label_smoothing
        ) if class_weights is not None else nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        # Increased weight decay for stronger regularization (5x for very small datasets, reduced from 20x)
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay * 5)
        
        # Use ReduceLROnPlateau with minimum learning rate protection
        min_lr = learning_rate * 1e-4  # Prevent LR from going below 0.01% of initial
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=min_lr)
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # Learning rate warmup for first few epochs
            if epoch < 5:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate * (epoch + 1) / 5
            else:
                # Use ReduceLROnPlateau after warmup (with min_lr protection)
                scheduler.step(val_acc)
                # Ensure LR doesn't go below minimum
                min_lr = learning_rate * 1e-4
                for param_group in optimizer.param_groups:
                    if param_group['lr'] < min_lr:
                        param_group['lr'] = min_lr
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}%")
            
            # Check for overfitting
            train_val_gap = train_acc - val_acc
            if train_val_gap > 20:
                print(f"⚠ WARNING: Large train-val gap ({train_val_gap:.2f}%), possible overfitting!")
                # Reduce learning rate more aggressively if overfitting, but protect minimum LR
                if train_val_gap > 30:
                    min_lr = learning_rate * 1e-4  # Minimum LR threshold
                    for param_group in optimizer.param_groups:
                        new_lr = param_group['lr'] * 0.5
                        # Only reduce if above minimum threshold
                        if new_lr >= min_lr:
                            param_group['lr'] = new_lr
                            print(f"  Reduced learning rate to {param_group['lr']:.2e} due to severe overfitting")
                        else:
                            print(f"  Learning rate at minimum threshold ({min_lr:.2e}), not reducing further")
            
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
                print(f"Final train-val gap: {train_val_gap:.2f}%")
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
        roc_auc = roc_auc_score(targets, probs)
        
        results = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'predictions': preds,
            'targets': targets,
            'probabilities': probs,
            'best_val_acc': self.best_val_acc
        }
        
        return results


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fulsang CNN-LOC using CombinedCNNLOC architecture')
    parser.add_argument('--tfrecord_dir', type=str, default='fulsang_preprocessed/tfrecords',
                       help='Directory containing Fulsang TFRecord files')
    parser.add_argument('--window_size', type=int, default=512,
                       help='Window size in samples (default: 512)')
    parser.add_argument('--overlap', type=float, default=0.9,
                       help='Window overlap fraction (default: 0.9 for small datasets)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (default: 16 for small datasets)')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs (default: 100 for small datasets)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4, increased from 1e-5 to allow better learning)')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                       help='Dropout rate (default: 0.5, reduced from 0.7 to allow model to learn)')
    parser.add_argument('--label_smoothing', type=float, default=0.2,
                       help='Label smoothing factor (default: 0.2 for very small datasets)')
    parser.add_argument('--use_64_channels', action='store_true', default=True,
                       help='Use 64 channels instead of 66 (better compatibility)')
    parser.add_argument('--output_dir', type=str, default='fulcnnfin_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("FULSANG CNN-LOC Training")
    print("="*80)
    print(f"TFRecord directory: {args.tfrecord_dir}")
    print(f"Window size: {args.window_size} samples")
    print(f"Overlap: {args.overlap}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Dropout rate: {args.dropout_rate}")
    print(f"Label smoothing: {args.label_smoothing}")
    print(f"Epochs: {args.num_epochs}")
    
    # Warn if window size is large (will create few samples)
    if args.window_size >= 512:
        print("\n⚠ WARNING: Large window size will create few training samples!")
        print(f"  Consider using --window_size 256 or --window_size 128 for more training data")
        print(f"  Current: {args.window_size} samples → ~108 windows")
        print(f"  With 256: ~400+ windows (4x more data)")
        print(f"  With 128: ~800+ windows (8x more data)")
    
    # Create dataset
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)
    dataset = FULCNNFINDataset(
        tfrecord_dir=args.tfrecord_dir,
        mode='full',
        window_size=args.window_size,
        overlap=args.overlap,
        transform_eeg=True,
        use_64_channels=args.use_64_channels,
        augment=False  # Will be enabled after split
    )
    
    # Subject-level splitting to avoid data leakage
    print("\n" + "="*80)
    print("SPLITTING DATASET BY SUBJECT (NO DATA LEAKAGE)")
    print("="*80)
    
    # Group windows by subject
    subject_windows = {}
    for idx, (start_idx, end_idx, label) in enumerate(dataset.window_indices):
        # Find which trial this window belongs to
        current_idx = 0
        trial_idx = 0
        for i, trial_eeg in enumerate(dataset.eeg_data):
            trial_length = trial_eeg.shape[0]
            if current_idx + trial_length > start_idx:
                trial_idx = i
                break
            current_idx += trial_length
        
        subject_id = dataset.metadata[trial_idx]['subject_id']
        if subject_id not in subject_windows:
            subject_windows[subject_id] = []
        subject_windows[subject_id].append(idx)
    
    # Split subjects (not windows)
    subjects = list(subject_windows.keys())
    np.random.seed(42)
    np.random.shuffle(subjects)
    
    n_subjects = len(subjects)
    train_subjects = subjects[:int(0.7 * n_subjects)]
    val_subjects = subjects[int(0.7 * n_subjects):int(0.85 * n_subjects)]
    test_subjects = subjects[int(0.85 * n_subjects):]
    
    # Get window indices for each split
    train_indices = []
    val_indices = []
    test_indices = []
    
    for subject_id, window_indices in subject_windows.items():
        if subject_id in train_subjects:
            train_indices.extend(window_indices)
        elif subject_id in val_subjects:
            val_indices.extend(window_indices)
        elif subject_id in test_subjects:
            test_indices.extend(window_indices)
    
    print(f"  Subjects: {n_subjects} total")
    print(f"    Train: {len(train_subjects)} subjects, {len(train_indices)} windows")
    print(f"    Val: {len(val_subjects)} subjects, {len(val_indices)} windows")
    print(f"    Test: {len(test_subjects)} subjects, {len(test_indices)} windows")
    
    # Create wrapper datasets with proper mode settings
    class ModeDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, indices, mode):
            self.base_dataset = base_dataset
            self.indices = indices
            self.mode = mode
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            # Temporarily set mode for this sample
            old_mode = self.base_dataset.mode
            old_augment = self.base_dataset.augment
            self.base_dataset.mode = self.mode
            self.base_dataset.augment = (self.mode == 'train')
            
            result = self.base_dataset[self.indices[idx]]
            
            # Restore original settings
            self.base_dataset.mode = old_mode
            self.base_dataset.augment = old_augment
            
            return result
    
    train_dataset = ModeDataset(dataset, train_indices, 'train')
    val_dataset = ModeDataset(dataset, val_indices, 'val')
    test_dataset = ModeDataset(dataset, test_indices, 'test')
    
    # Calculate class weights for imbalanced data
    all_train_labels = []
    for idx in train_indices:
        _, _, label = dataset.window_indices[idx]
        all_train_labels.append(label)
    class_counts = np.bincount(all_train_labels)
    total = len(all_train_labels)
    class_weights = torch.FloatTensor([total / (2.0 * class_counts[0]), total / (2.0 * class_counts[1])]).to(device)
    print(f"  Class weights: {class_weights.cpu().numpy()}")
    
    # Create data loaders with more aggressive settings for small datasets
    # Use drop_last=False to use all available data
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=2, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=2, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=2, pin_memory=True, drop_last=False)
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # Create model
    print("\n" + "="*80)
    print("INITIALIZING CNN-LOC MODEL")
    print("="*80)
    
    model = CNNLOCModel(
        input_channels=64 if args.use_64_channels else 66,
        input_time=32,
        input_freq=4,
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
        learning_rate=args.learning_rate,  # Already set to lower value
        class_weights=class_weights,
        label_smoothing=args.label_smoothing
    )
    
    # Test model
    print("\n" + "="*80)
    print("TESTING MODEL")
    print("="*80)
    test_metrics = trainer.test(test_loader)
    
    # Save results
    results_json = {
        'accuracy': float(test_metrics['accuracy']),
        'roc_auc': float(test_metrics['roc_auc']),
        'best_val_acc': float(test_metrics['best_val_acc']),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(Path(args.output_dir) / 'results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✓ Training Complete")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"\n✓ Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()

