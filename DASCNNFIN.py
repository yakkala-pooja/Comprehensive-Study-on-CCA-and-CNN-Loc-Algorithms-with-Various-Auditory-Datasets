#!/usr/bin/env python3
"""
DASCNNFIN - CNN-LOC for Das Dataset (Final Version)

CNN-LOC model for attention decoding on Das EEG data using TFRecord files.
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# CNN-LOC Architecture from CombinedCNNLOC.py (exact copy)
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
        
        # Classifier (exact from CombinedCNNLOC.py)
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


class DASCNNFINDataset(Dataset):
    """PyTorch Dataset for Das TFRecord files."""
    
    def __init__(self, tfrecord_dir: str, mode: str = 'train', 
                 window_size: int = 512, overlap: float = 0.5,
                 transform_eeg: bool = True):
        self.tfrecord_dir = Path(tfrecord_dir)
        self.mode = mode
        self.window_size = window_size
        self.overlap = overlap
        self.transform_eeg = transform_eeg
        
        # Das dataset params
        self.sampling_rate = 64  # Hz
        self.n_channels = 64
        
        # Load data from TFRecord files
        self.eeg_data, self.labels, self.metadata = self._load_tfrecord_data()
        
        # Create windows
        self.window_indices = self._create_windows()
        
        print(f"\nDASCNNFINDataset initialized:")
        print(f"  Mode: {mode}")
        print(f"  Total trials: {len(self.eeg_data)}")
        print(f"  Total windows: {len(self.window_indices)}")
        print(f"  Window size: {self.window_size} samples")
        print(f"  Sampling rate: {self.sampling_rate} Hz")
        print(f"  Channels: {self.n_channels}")
        if len(self.labels) > 0:
            label_counts = np.bincount(self.labels)
            print(f"  Label distribution (trials): {dict(zip(range(len(label_counts)), label_counts))}")
            # Count windows per label
            window_labels = [w[2] for w in self.window_indices]
            window_label_counts = np.bincount(window_labels)
            print(f"  Label distribution (windows): {dict(zip(range(len(window_label_counts)), window_label_counts))}")
    
    def _load_tfrecord_data(self) -> Tuple[List[np.ndarray], List[int], List[Dict]]:
        """Load Das data from TFRecord files."""
        print(f"Loading Das data from {self.tfrecord_dir}...")
        
        # Find TFRecord files based on mode
        tfrecord_files = []
        
        train_dir = self.tfrecord_dir / "train"
        test_dir = self.tfrecord_dir / "test"
        
        if train_dir.exists() and test_dir.exists():
            print(f"Found train/test subdirectories")
            if self.mode == 'train':
                tfrecord_files = list(train_dir.glob("*.tfrecords"))
            elif self.mode == 'test':
                tfrecord_files = list(test_dir.glob("*.tfrecords"))
            else:
                # Use both for 'full' mode
                tfrecord_files = list(train_dir.glob("*.tfrecords")) + list(test_dir.glob("*.tfrecords"))
        else:
            # Fallback: search in main directory
            tfrecord_files = list(self.tfrecord_dir.glob("*.tfrecords"))
        
        if not tfrecord_files:
            raise ValueError(f"No TFRecord files found in {self.tfrecord_dir}")
        
        print(f"Found {len(tfrecord_files)} TFRecord files")
        
        # Load data from TFRecord files
        # Group samples by trial_id to properly reconstruct trials
        trials_dict = {}  # key: (subject_id, trial_id), value: {'eeg': list, 'label': int, 'file': str}
        
        for tfrecord_file in tqdm(tfrecord_files, desc="Loading TFRecord files"):
            try:
                dataset = tf.data.TFRecordDataset(str(tfrecord_file))
                
                for raw_record in dataset:
                    try:
                        example = tf.train.Example()
                        example.ParseFromString(raw_record.numpy())
                        features = example.features.feature
                        
                        # Required features
                        if 'eeg' not in features or 'attended_ear' not in features:
                            continue
                        
                        # Extract EEG data
                        eeg_bytes = features['eeg'].float_list.value
                        if len(eeg_bytes) != self.n_channels:
                            continue
                        
                        eeg_sample = np.array(eeg_bytes, dtype=np.float32).reshape(1, self.n_channels)
                        
                        # Extract attended_ear
                        attended_ear = features['attended_ear'].bytes_list.value[0].decode('utf-8')
                        if attended_ear.upper() not in ['L', 'R']:
                            continue
                        label = 0 if attended_ear.upper() == 'L' else 1
                        
                        # Extract subject_id
                        subject_id = "unknown"
                        if 'subject_id' in features:
                            subject_id = features['subject_id'].bytes_list.value[0].decode('utf-8')
                        
                        # Extract trial_id (use 0 as default if not present)
                        trial_id = 0
                        if 'trial_id' in features:
                            trial_id_values = features['trial_id'].int64_list.value
                            if trial_id_values:
                                trial_id = int(trial_id_values[0])
                        
                        # Group samples by (subject_id, trial_id)
                        trial_key = (subject_id, trial_id)
                        if trial_key not in trials_dict:
                            trials_dict[trial_key] = {
                                'eeg': [],
                                'label': label,
                                'file': tfrecord_file.name,
                                'subject_id': subject_id
                            }
                        
                        # Validate label consistency within trial
                        if trials_dict[trial_key]['label'] != label:
                            print(f"WARNING: Label mismatch in trial {trial_key} - expected {trials_dict[trial_key]['label']}, got {label}")
                            # Use the first label encountered
                        
                        trials_dict[trial_key]['eeg'].append(eeg_sample)
                    
                    except Exception as e:
                        continue
            
            except Exception as e:
                print(f"Error loading {tfrecord_file}: {e}")
                continue
        
        # Convert trials_dict to lists
        eeg_data = []
        labels = []
        metadata = []
        
        for (subject_id, trial_id), trial_data in trials_dict.items():
            if len(trial_data['eeg']) == 0:
                continue
            
            trial_eeg = np.vstack(trial_data['eeg'])
            eeg_data.append(trial_eeg)
            labels.append(trial_data['label'])
            metadata.append({
                'subject_id': subject_id,
                'trial_id': trial_id,
                'file': trial_data['file']
            })
        
        if not eeg_data:
            raise ValueError("No valid Das data loaded")
        
        print(f"Loaded {len(eeg_data)} trials")
        print(f"Label distribution: {np.bincount(labels)}")
        
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
        
        eeg_window = self.eeg_data[trial_idx][window_start_in_trial:window_end_in_trial]
        
        # Preprocess
        eeg_window = eeg_window - np.mean(eeg_window, axis=0, keepdims=True)
        std_vals = np.std(eeg_window, axis=0, keepdims=True)
        std_vals = np.where(std_vals == 0, 1.0, std_vals)
        eeg_window = eeg_window / std_vals
        
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
    
    def __init__(self, model: CNNLOCModel, device: torch.device, output_dir: str = "dascnnfin_results"):
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.best_val_acc = 0.0
        self.best_model_path = self.output_dir / "best_model.pth"
    
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module, use_augmentation: bool = True) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            target = target.squeeze()
            
            # Data augmentation during training
            if use_augmentation and self.model.training:
                data = self._apply_augmentation(data)
            
            output = self.model(data)
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            # Increased gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
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
    
    def _apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation to reduce overfitting - balanced approach."""
        # Random noise injection (moderate)
        if np.random.rand() < 0.6:  # Reduced from 0.7
            noise_scale = 0.015  # Reduced from 0.02
            noise = torch.randn_like(x) * noise_scale
            x = x + noise
        
        # Random channel dropout (spatial dropout) - moderate
        if np.random.rand() < 0.4:  # Reduced from 0.5
            dropout_prob = 0.12  # Reduced from 0.15
            mask = torch.bernoulli(torch.ones(x.shape[0], x.shape[1], 1, 1, device=x.device) * (1 - dropout_prob))
            x = x * mask / (1 - dropout_prob)
        
        # Random time masking - moderate
        if np.random.rand() < 0.4:  # Reduced from 0.5
            time_mask_size = int(x.shape[2] * 0.12)  # Reduced from 0.15
            if time_mask_size > 0:
                t0 = np.random.randint(0, max(1, x.shape[2] - time_mask_size))
                x[:, :, t0:t0+time_mask_size, :] = 0
        
        # Random frequency masking - moderate
        if np.random.rand() < 0.3:  # Reduced from 0.4
            freq_mask_size = int(x.shape[3] * 0.15)  # Reduced from 0.2
            if freq_mask_size > 0:
                f0 = np.random.randint(0, max(1, x.shape[3] - freq_mask_size))
                x[:, :, :, f0:f0+freq_mask_size] = 0
        
        return x
    
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
              weight_decay: float = 1e-5, patience: int = 10, label_smoothing: float = 0.1):
        """Train the model."""
        
        # Calculate class weights for imbalanced data
        print("Calculating class weights for training data...")
        train_labels = []
        for _, (_, target) in enumerate(train_loader):
            train_labels.extend(target.squeeze().cpu().numpy())
        
        unique_classes, class_counts = np.unique(train_labels, return_counts=True)
        total_samples = len(train_labels)
        n_classes = len(unique_classes)
        
        if n_classes == 0:
            print("WARNING: No classes found in training data")
            class_weights = torch.ones(2).to(self.device)
        else:
            # Calculate weights: total_samples / (n_classes * class_count)
            class_weights = np.zeros(max(unique_classes) + 1)
            for i, class_id in enumerate(unique_classes):
                if class_counts[i] > 0:
                    class_weights[class_id] = total_samples / (n_classes * class_counts[i])
                else:
                    class_weights[class_id] = 1.0
            
            class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        print(f"  Unique classes: {unique_classes}")
        print(f"  Class counts: {class_counts}")
        print(f"  Class weights: {class_weights.cpu().numpy()}")
        
        # Use weighted loss with label smoothing
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # Use OneCycleLR scheduler (same as CombinedCNNLOC)
        scheduler = OneCycleLR(optimizer, max_lr=learning_rate * 5, 
                              total_steps=num_epochs * len(train_loader), pct_start=0.3)
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}%")
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, self.best_model_path)
                print(f"New best model saved! Val Acc: {val_acc:.4f}%")
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
    
    parser = argparse.ArgumentParser(description='Das CNN-LOC using CombinedCNNLOC architecture')
    parser.add_argument('--tfrecord_dir', type=str, default='das_16subjects_preprocessed/tfrecords',
                       help='Directory containing Das TFRecord files')
    parser.add_argument('--window_size', type=int, default=512,
                       help='Window size in samples (default: 512)')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Window overlap fraction (default: 0.5)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                       help='Learning rate (default: 5e-4, reduced to prevent overfitting)')
    parser.add_argument('--dropout_rate', type=float, default=0.45,
                       help='Dropout rate (default: 0.45, balanced to prevent overfitting)')
    parser.add_argument('--weight_decay', type=float, default=5e-5,
                       help='Weight decay (default: 5e-5, balanced for regularization)')
    parser.add_argument('--label_smoothing', type=float, default=0.08,
                       help='Label smoothing factor (default: 0.08, reduced for better balance)')
    parser.add_argument('--output_dir', type=str, default='dascnnfin_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("DAS CNN-LOC Training")
    print("="*80)
    print(f"TFRecord directory: {args.tfrecord_dir}")
    print(f"Window size: {args.window_size} samples")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.num_epochs}")
    
    # Create datasets
    print("\n" + "="*80)
    print("LOADING DATASETS")
    print("="*80)
    train_dataset = DASCNNFINDataset(
        tfrecord_dir=args.tfrecord_dir,
        mode='train',
        window_size=args.window_size,
        overlap=args.overlap,
        transform_eeg=True
    )
    
    test_dataset = DASCNNFINDataset(
        tfrecord_dir=args.tfrecord_dir,
        mode='test',
        window_size=args.window_size,
        overlap=args.overlap,
        transform_eeg=True
    )
    
    # Subject-wise splitting to prevent data leakage
    # Group windows by subject_id from metadata
    subject_windows = {}
    for idx in range(len(train_dataset)):
        # Get subject_id from metadata
        window_info = train_dataset.window_indices[idx]
        # Find corresponding metadata
        current_idx = 0
        trial_idx = 0
        for i, trial_eeg in enumerate(train_dataset.eeg_data):
            trial_length = trial_eeg.shape[0]
            if current_idx + trial_length > window_info[0]:
                trial_idx = i
                break
            current_idx += trial_length
        
        if trial_idx < len(train_dataset.metadata):
            subject_id = train_dataset.metadata[trial_idx].get('subject_id', 'unknown')
        else:
            subject_id = 'unknown'
        
        if subject_id not in subject_windows:
            subject_windows[subject_id] = []
        subject_windows[subject_id].append(idx)
    
    # Also check test set subjects to ensure no overlap
    test_subject_windows = {}
    for idx in range(len(test_dataset)):
        window_info = test_dataset.window_indices[idx]
        current_idx = 0
        trial_idx = 0
        for i, trial_eeg in enumerate(test_dataset.eeg_data):
            trial_length = trial_eeg.shape[0]
            if current_idx + trial_length > window_info[0]:
                trial_idx = i
                break
            current_idx += trial_length
        
        if trial_idx < len(test_dataset.metadata):
            subject_id = test_dataset.metadata[trial_idx].get('subject_id', 'unknown')
        else:
            subject_id = 'unknown'
        
        if subject_id not in test_subject_windows:
            test_subject_windows[subject_id] = []
        test_subject_windows[subject_id].append(idx)
    
    # Verify no subject overlap between train and test
    train_subject_set = set(subject_windows.keys())
    test_subject_set = set(test_subject_windows.keys())
    overlap = train_subject_set & test_subject_set
    if overlap:
        print(f"⚠ WARNING: {len(overlap)} subjects overlap between train and test sets: {overlap}")
        print("  This may cause data leakage. Consider using separate subject sets.")
    else:
        print(f"✓ No subject overlap between train and test sets")
    
    # Split subjects (not windows) to prevent data leakage
    subjects = list(subject_windows.keys())
    np.random.seed(42)
    np.random.shuffle(subjects)
    
    train_subjects = subjects[:int(0.85 * len(subjects))]
    val_subjects = subjects[int(0.85 * len(subjects)):]
    
    train_indices = []
    val_indices = []
    for subject_id in train_subjects:
        train_indices.extend(subject_windows[subject_id])
    for subject_id in val_subjects:
        val_indices.extend(subject_windows[subject_id])
    
    # Create subset datasets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(train_dataset, val_indices)
    
    print(f"  Subject-wise split:")
    print(f"    Train subjects: {len(train_subjects)} ({train_subjects})")
    print(f"    Val subjects: {len(val_subjects)} ({val_subjects})")
    print(f"    Test subjects: {len(test_subject_set)} ({list(test_subject_set)})")
    print(f"    Train windows: {len(train_indices)}")
    print(f"    Val windows: {len(val_indices)}")
    print(f"    Test windows: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f"  Train samples: {len(train_subset)}")
    print(f"  Val samples: {len(val_subset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # Create model
    print("\n" + "="*80)
    print("INITIALIZING CNN-LOC MODEL")
    print("="*80)
    model = CNNLOCModel(
        input_channels=64,
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
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
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

