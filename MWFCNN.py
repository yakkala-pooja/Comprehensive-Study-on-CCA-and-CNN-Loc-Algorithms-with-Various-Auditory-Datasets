#!/usr/bin/env python3
"""
MWFCNN - CNN-LOC Algorithm for Combined Das (DASPREPROCESS) and Fuglsang (MWF) Datasets

This module implements a CNN-LOC (Convolutional Neural Network - Localization) 
algorithm for combined Das and Fuglsang datasets.

Features:
- Loads preprocessed Das data (using DASPREPROCESS) and MWF-cleaned Fuglsang data
- Combines datasets for training
- CNN-LOC architecture optimized for preprocessed EEG data
- Comprehensive metrics: Accuracy, MSED, ROC-AUC, temporal performance
"""

import os
import sys
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve,
                           average_precision_score, matthews_corrcoef, 
                           cohen_kappa_score, balanced_accuracy_score, f1_score)
import seaborn as sns
from tqdm import tqdm
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class CombinedMWFDataset(Dataset):
    """
    Combined dataset class for preprocessed Das and MWF-cleaned Fuglsang data.
    
    Loads preprocessed Das data (using DASPREPROCESS) and MWF-cleaned Fuglsang data.
    """
    
    def __init__(self, das_preprocessed_dir: str = "preprocessed_Das",
                 fuglsang_mwf_dir: str = "MWF_cleaned_Fuglsang",
                 mode: str = 'train',
                 window_size: int = 512,  # samples at 128 Hz = 4 seconds
                 overlap: float = 0.5,
                 transform_eeg: bool = True):
        self.das_preprocessed_dir = Path(das_preprocessed_dir)
        self.fuglsang_mwf_dir = Path(fuglsang_mwf_dir)
        self.mode = mode
        self.window_size = window_size
        self.overlap = overlap
        self.transform_eeg = transform_eeg
        
        # Parameters
        self.sampling_rate = 128  # Hz (both datasets)
        self.n_channels = 64  # Common number of channels (may need adjustment)
        
        # Load preprocessed data from both datasets
        print("Loading preprocessed data from Das dataset...")
        das_eeg, das_labels, das_metadata, das_trial_lengths = self._load_das_preprocessed_data()
        
        print("Loading MWF-cleaned data from Fuglsang dataset...")
        fuglsang_eeg, fuglsang_labels, fuglsang_metadata, fuglsang_trial_lengths = self._load_fuglsang_mwf_data()
        
        # Normalize channel count BEFORE combining (use max, pad smaller dataset)
        max_channels = max(das_eeg.shape[1], fuglsang_eeg.shape[1])
        if das_eeg.shape[1] != fuglsang_eeg.shape[1]:
            print(f"Warning: Channel mismatch - Das: {das_eeg.shape[1]}, Fuglsang: {fuglsang_eeg.shape[1]}")
            print(f"Padding to {max_channels} channels (keeping all Das channels)")
            
            # Pad Fuglsang data if it has fewer channels
            if fuglsang_eeg.shape[1] < max_channels:
                padding = max_channels - fuglsang_eeg.shape[1]
                # Pad with zeros (or replicate last channel)
                pad_data = np.zeros((fuglsang_eeg.shape[0], padding), dtype=fuglsang_eeg.dtype)
                fuglsang_eeg = np.hstack([fuglsang_eeg, pad_data])
            
            # Trim Das if it has more channels (shouldn't happen, but just in case)
            if das_eeg.shape[1] > max_channels:
                das_eeg = das_eeg[:, :max_channels]
        
        # Combine datasets
        print("Combining datasets...")
        self.eeg_data = np.vstack([das_eeg, fuglsang_eeg])
        self.labels = np.hstack([das_labels, fuglsang_labels])
        self.metadata = das_metadata + fuglsang_metadata
        
        self.n_channels = max_channels
        
        # Track trial boundaries for label mapping
        # Each trial has a label, but we need to map sample indices to trial labels
        self.trial_boundaries = []
        self.trial_labels = []
        current_idx = 0
        
        # Track Das trial boundaries using stored trial lengths
        for i, (label, trial_length) in enumerate(zip(das_labels, das_trial_lengths)):
            self.trial_boundaries.append((current_idx, current_idx + trial_length))
            self.trial_labels.append(label)
            current_idx += trial_length
        
        # Track Fuglsang trial boundaries using stored trial lengths
        for i, (label, trial_length) in enumerate(zip(fuglsang_labels, fuglsang_trial_lengths)):
            self.trial_boundaries.append((current_idx, current_idx + trial_length))
            self.trial_labels.append(label)
            current_idx += trial_length
        
        # Create windows with labels
        self.window_indices = self._create_windows()
        
        print(f"Combined dataset loaded:")
        print(f"  Total samples: {len(self.eeg_data)}")
        print(f"  Total windows: {len(self.window_indices)}")
        print(f"  EEG shape: {self.eeg_data.shape}")
        print(f"  Label distribution: {np.bincount(self.labels)}")
        print(f"  Channels: {self.n_channels}")
    
    def _load_das_preprocessed_data(self) -> Tuple[np.ndarray, np.ndarray, List[Dict], List[int]]:
        """Load preprocessed Das dataset (using DASPREPROCESS)."""
        if not self.das_preprocessed_dir.exists():
            raise ValueError(f"Das preprocessed directory does not exist: {self.das_preprocessed_dir}\n"
                           f"Please run DASPREPROCESS first: python3 unified_preprocessing.py")
        
        preprocessed_files = list(self.das_preprocessed_dir.glob("S*_preprocessed.mat"))
        if not preprocessed_files:
            raise ValueError(f"No preprocessed Das files found in {self.das_preprocessed_dir}\n"
                           f"Expected files: S1_preprocessed.mat, S2_preprocessed.mat, etc.\n"
                           f"Please run DASPREPROCESS first")
        
        all_eeg = []
        all_labels = []
        all_metadata = []
        trial_lengths = []
        
        for preprocessed_file in tqdm(preprocessed_files, desc="Loading Das preprocessed data"):
            try:
                data = sio.loadmat(str(preprocessed_file), squeeze_me=True, struct_as_record=False)
                subject_id = preprocessed_file.stem.replace('_preprocessed', '')
                
                if 'trials' in data:
                    trials = data['trials']
                    if not isinstance(trials, np.ndarray):
                        trials = [trials]
                    else:
                        trials = trials.flatten()
                    
                    for trial_idx, trial in enumerate(trials):
                        if hasattr(trial, 'eeg_data'):
                            eeg_data = trial.eeg_data
                        elif isinstance(trial, dict):
                            eeg_data = trial['eeg_data']
                        else:
                            continue
                        
                        # Ensure eeg_data is 2D (samples x channels)
                        if len(eeg_data.shape) == 1:
                            eeg_data = eeg_data.reshape(-1, 1)
                        elif len(eeg_data.shape) > 2:
                            # Flatten extra dimensions
                            eeg_data = eeg_data.reshape(eeg_data.shape[0], -1)
                        
                        # Get attended ear label
                        if hasattr(trial, 'attended_ear'):
                            attended_ear = trial.attended_ear
                        elif isinstance(trial, dict):
                            attended_ear = trial.get('attended_ear', 'L')
                        else:
                            attended_ear = 'L'
                        
                        # Convert to label (L=0, R=1)
                        label = 0 if str(attended_ear).upper() == 'L' else 1
                        
                        all_eeg.append(eeg_data)
                        all_labels.append(label)
                        trial_lengths.append(eeg_data.shape[0])  # Store trial length
                        all_metadata.append({
                            'subject_id': subject_id,
                            'trial_idx': trial_idx,
                            'dataset': 'Das',
                            'attended_ear': attended_ear,
                            'preprocessing': 'DASPREPROCESS'
                        })
            except Exception as e:
                print(f"Error loading {preprocessed_file}: {e}")
                continue
        
        if not all_eeg:
            raise ValueError("No valid Das preprocessed data loaded")
        
        # Normalize channel count - find minimum and trim all to that
        channel_counts = [eeg.shape[1] for eeg in all_eeg]
        min_channels = min(channel_counts)
        
        if len(set(channel_counts)) > 1:
            print(f"Warning: Das data has inconsistent channels: {set(channel_counts)}")
            print(f"Using first {min_channels} channels from all trials")
            # Normalize all arrays to have the same number of channels
            all_eeg = [eeg[:, :min_channels] for eeg in all_eeg]
        
        eeg_data = np.vstack(all_eeg)
        labels = np.array(all_labels)
        
        return eeg_data, labels, all_metadata, trial_lengths
    
    def _load_fuglsang_mwf_data(self) -> Tuple[np.ndarray, np.ndarray, List[Dict], List[int]]:
        """Load MWF-cleaned Fuglsang dataset."""
        if not self.fuglsang_mwf_dir.exists():
            raise ValueError(f"Fuglsang MWF directory does not exist: {self.fuglsang_mwf_dir}\n"
                           f"Please run MWF processing first: bash FULMWF.sh")
        
        mwf_files = list(self.fuglsang_mwf_dir.glob("sub*_MWF.mat"))
        if not mwf_files:
            raise ValueError(f"No MWF-cleaned Fuglsang files found in {self.fuglsang_mwf_dir}\n"
                           f"Expected files: sub01_MWF.mat, sub02_MWF.mat, etc.\n"
                           f"Please run MWF processing first: bash FULMWF.sh")
        
        all_eeg = []
        all_labels = []
        all_metadata = []
        trial_lengths = []
        
        for mwf_file in tqdm(mwf_files, desc="Loading Fuglsang MWF data"):
            try:
                data = sio.loadmat(str(mwf_file), squeeze_me=True, struct_as_record=False)
                subject_id = mwf_file.stem.replace('_MWF', '')
                
                if 'trials' in data:
                    trials = data['trials']
                    if not isinstance(trials, np.ndarray):
                        trials = [trials]
                    else:
                        trials = trials.flatten()
                    
                    for trial_idx, trial in enumerate(trials):
                        if hasattr(trial, 'eeg_data'):
                            eeg_data = trial.eeg_data
                        elif isinstance(trial, dict):
                            eeg_data = trial['eeg_data']
                        else:
                            continue
                        
                        # Ensure eeg_data is 2D (samples x channels)
                        if len(eeg_data.shape) == 1:
                            eeg_data = eeg_data.reshape(-1, 1)
                        elif len(eeg_data.shape) > 2:
                            # Flatten extra dimensions
                            eeg_data = eeg_data.reshape(eeg_data.shape[0], -1)
                        
                        # Get attention label
                        if hasattr(trial, 'attention_label'):
                            label = int(trial.attention_label)
                        elif isinstance(trial, dict):
                            label = int(trial.get('attention_label', 0))
                        else:
                            label = 0
                        
                        all_eeg.append(eeg_data)
                        all_labels.append(label)
                        trial_lengths.append(eeg_data.shape[0])  # Store trial length
                        all_metadata.append({
                            'subject_id': subject_id,
                            'trial_idx': trial_idx,
                            'dataset': 'Fuglsang',
                            'attention_label': label
                        })
            except Exception as e:
                print(f"Error loading {mwf_file}: {e}")
                continue
        
        if not all_eeg:
            raise ValueError("No valid Fuglsang MWF data loaded")
        
        # Normalize channel count - find maximum and pad all to that
        channel_counts = [eeg.shape[1] for eeg in all_eeg]
        max_channels = max(channel_counts)
        
        if len(set(channel_counts)) > 1:
            print(f"Warning: Fuglsang data has inconsistent channels: {set(channel_counts)}")
            print(f"Padding to {max_channels} channels (keeping maximum)")
            # Normalize all arrays to have the same number of channels by padding
            normalized_eeg = []
            for eeg in all_eeg:
                if eeg.shape[1] < max_channels:
                    # Pad with zeros
                    padding = max_channels - eeg.shape[1]
                    pad_data = np.zeros((eeg.shape[0], padding), dtype=eeg.dtype)
                    eeg = np.hstack([eeg, pad_data])
                normalized_eeg.append(eeg)
            all_eeg = normalized_eeg
        
        eeg_data = np.vstack(all_eeg)
        labels = np.array(all_labels)
        
        return eeg_data, labels, all_metadata, trial_lengths
    
    def _create_windows(self) -> List[Tuple[int, int, int]]:
        """Create sliding windows from continuous EEG data with labels."""
        window_indices = []
        step_size = int(self.window_size * (1 - self.overlap))
        
        for start_idx in range(0, len(self.eeg_data) - self.window_size + 1, step_size):
            end_idx = start_idx + self.window_size
            
            # Find which trial this window belongs to (use middle of window)
            mid_idx = start_idx + self.window_size // 2
            window_label = 0  # Default
            
            for trial_idx, (trial_start, trial_end) in enumerate(self.trial_boundaries):
                if trial_start <= mid_idx < trial_end:
                    window_label = self.trial_labels[trial_idx]
                    break
            
            window_indices.append((start_idx, end_idx, window_label))
        
        return window_indices
    
    def _transform_eeg(self, eeg_window: np.ndarray) -> np.ndarray:
        """Transform EEG window to time-frequency representation."""
        # Simple time-frequency transform: STFT-like
        n_samples, n_channels = eeg_window.shape
        
        # Compute spectrogram for each channel
        freq_bins = 4  # Number of frequency bands
        time_frames = 32  # Number of time frames
        
        # Reshape to (channels, time_frames, freq_bins)
        # Simple approach: reshape and apply FFT
        if n_samples >= time_frames:
            samples_per_frame = n_samples // time_frames
            eeg_reshaped = eeg_window[:time_frames * samples_per_frame].reshape(
                time_frames, samples_per_frame, n_channels
            )
            
            # Apply FFT to each frame
            eeg_fft = np.fft.rfft(eeg_reshaped, axis=1)
            eeg_fft = np.abs(eeg_fft[:, :freq_bins, :])
            
            # Reshape to (channels, time_frames, freq_bins)
            eeg_tf = np.transpose(eeg_fft, (2, 0, 1))
        else:
            # Pad if needed
            eeg_tf = np.zeros((n_channels, time_frames, freq_bins))
            eeg_tf[:, :n_samples, :] = eeg_window[:n_samples, :, np.newaxis]
        
        return eeg_tf
    
    def __len__(self):
        return len(self.window_indices)
    
    def __getitem__(self, idx):
        start_idx, end_idx, label = self.window_indices[idx]
        eeg_window = self.eeg_data[start_idx:end_idx]
        
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


# CNN Architecture (similar to FULCNN but adapted for combined data)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MWFCNNBackbone(nn.Module):
    """CNN backbone for MWF-cleaned combined data."""
    
    def __init__(self, input_channels: int = 64, input_time: int = 32, input_freq: int = 4):
        super(MWFCNNBackbone, self).__init__()
        
        self.input_channels = input_channels
        self.input_time = input_time
        self.input_freq = input_freq
        
        # Initial feature extraction
        self.initial_conv = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(32)
        
        # Temporal processing
        self.temporal_block1 = ResidualBlock(32, 32, stride=1)
        self.temporal_pool1 = nn.MaxPool2d((2, 1), (2, 1))
        
        self.temporal_block2 = ResidualBlock(32, 64, stride=1)
        self.temporal_pool2 = nn.MaxPool2d((2, 1), (2, 1))
        
        # Spatial processing
        self.spatial_block1 = ResidualBlock(64, 64, stride=1)
        self.spatial_pool1 = nn.MaxPool2d((1, 2), (1, 2))
        
        self.spatial_block2 = ResidualBlock(64, 128, stride=1)
        self.spatial_pool2 = nn.MaxPool2d((1, 2), (1, 2))
        
        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        
        # Calculate output size
        self.output_size = 128
    
    def forward(self, x):
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        
        x = self.temporal_block1(x)
        x = self.temporal_pool1(x)
        
        x = self.temporal_block2(x)
        x = self.temporal_pool2(x)
        
        x = self.spatial_block1(x)
        x = self.spatial_pool1(x)
        
        x = self.spatial_block2(x)
        x = self.spatial_pool2(x)
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        return x


class MWFCNNModel(nn.Module):
    """MWFCNN model for combined Das and Fuglsang MWF-cleaned data."""
    
    def __init__(self, input_channels: int = 64, input_time: int = 32, input_freq: int = 4,
                 num_classes: int = 2, dropout_rate: float = 0.3):
        super(MWFCNNModel, self).__init__()
        
        self.backbone = MWFCNNBackbone(input_channels, input_time, input_freq)
        
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
        print(f"MWFCNN model created with {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def _initialize_weights(self):
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
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class MWFCNNTrainer:
    """Trainer for MWFCNN model."""
    
    def __init__(self, model, train_loader, val_loader, device, 
                 learning_rate=1e-3, weight_decay=1e-4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for eeg, labels in tqdm(self.train_loader, desc="Training"):
            eeg = eeg.to(self.device)
            labels = labels.squeeze().to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(eeg)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for eeg, labels in tqdm(self.val_loader, desc="Validating"):
                eeg = eeg.to(self.device)
                labels = labels.squeeze().to(self.device)
                
                outputs = self.model(eeg)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        return total_loss / len(self.val_loader), accuracy
    
    def train(self, num_epochs=50):
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            self.scheduler.step(val_loss)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'mwfcnn_best_model.pth')
                print(f"Saved best model with val accuracy: {best_val_acc:.4f}")


def main():
    """Main function to train MWFCNN."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MWFCNN on combined preprocessed Das and MWF-cleaned Fuglsang datasets')
    parser.add_argument('--das_preprocessed_dir', type=str, default='preprocessed_Das',
                       help='Directory containing Das preprocessed data (from DASPREPROCESS)')
    parser.add_argument('--fuglsang_mwf_dir', type=str, default='MWF_cleaned_Fuglsang',
                       help='Directory containing Fuglsang MWF-cleaned data')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--window_size', type=int, default=512,
                       help='Window size in samples')
    
    args = parser.parse_args()
    
    print("="*60)
    print("MWFCNN - Combined Das (DASPREPROCESS) and Fuglsang (MWF) Data")
    print("="*60)
    
    # Load dataset
    dataset = CombinedMWFDataset(
        das_preprocessed_dir=args.das_preprocessed_dir,
        fuglsang_mwf_dir=args.fuglsang_mwf_dir,
        window_size=args.window_size
    )
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = MWFCNNModel(input_channels=dataset.n_channels)
    
    # Train
    trainer = MWFCNNTrainer(model, train_loader, val_loader, device,
                           learning_rate=args.learning_rate)
    trainer.train(num_epochs=args.num_epochs)
    
    print("\nTraining complete!")
    print(f"Best validation accuracy: {max(trainer.val_accuracies):.4f}")


if __name__ == '__main__':
    main()

