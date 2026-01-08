#!/usr/bin/env python3
"""
CombinedCNNLOC - CNN-LOC Algorithm for Combined Das and Fulsang Dataset

This script implements CNN-LOC (Convolutional Neural Network - Localization) for the 
combined dataset using the FULCNN architecture, adapted for CombinedDataset.
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


# ============================================================================
# Dataset and Training
# ============================================================================

class CombinedCNNLOCDataset(Dataset):
    """
    PyTorch Dataset wrapper for CombinedDataset for CNN-LOC training.
    Converts numpy arrays to PyTorch tensors compatible with CNN-LOC.
    """
    
    def __init__(self, combined_dataset: CombinedDataset, mode: str = 'train', transform_eeg: bool = True):
        self.combined_dataset = combined_dataset
        self.mode = mode
        self.transform_eeg = transform_eeg
        self.window_size = combined_dataset.window_size
        self.sampling_rate = combined_dataset.sampling_rate
        self.n_channels = combined_dataset.n_channels
        
        # Get window indices
        self.window_indices = combined_dataset.get_window_indices()
        
        print(f"\nCombinedCNNLOCDataset initialized:")
        print(f"  Mode: {mode}")
        print(f"  Total windows: {len(self.window_indices)}")
        print(f"  Window size: {self.window_size} samples")
        print(f"  Sampling rate: {self.sampling_rate} Hz")
        print(f"  Channels: {self.n_channels}")
        print(f"  Transform EEG: {transform_eeg}")
    
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
        start_idx, end_idx, label = self.window_indices[idx]
        
        # Extract window
        eeg_window = self.combined_dataset.eeg_data[start_idx:end_idx]
        
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
        label_tensor = torch.LongTensor([label])
        
        return eeg_tensor, label_tensor


def split_dataset(dataset: CombinedCNNLOCDataset, train_ratio: float = 0.7, 
                  val_ratio: float = 0.15) -> Tuple[Dataset, Dataset, Dataset]:
    """Split dataset into train/val/test sets."""
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    return train_dataset, val_dataset, test_dataset


class CNNLOCTrainer:
    """Trainer for CNN-LOC model."""
    
    def __init__(self, model: CNNLOCModel, device: torch.device, output_dir: str = "combined_cnnloc_results"):
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
            
            # Forward
            output = self.model(data)
            loss = criterion(output, target)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate loss and accuracy
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
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
              weight_decay: float = 1e-5, patience: int = 10):
        """Train the model."""
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = OneCycleLR(optimizer, max_lr=learning_rate * 5, 
                              total_steps=num_epochs * len(train_loader), pct_start=0.3)
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
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
        
        # Calculate metrics
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
    
    parser = argparse.ArgumentParser(description='Combined Das+Fulsang CNN-LOC using FULCNN architecture')
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
                       help='Output directory for Fulsang MWF processing (legacy)')
    parser.add_argument('--combined_dataset_dir', type=str, default='combined_dataset',
                       help='Centralized directory for all processed files (default: combined_dataset)')
    parser.add_argument('--split_method', type=str, default='window', choices=['subject', 'window', 'both'],
                       help='Split method: subject (no leakage), window (random), or both (comparison)')
    parser.add_argument('--window_size', type=int, default=512,
                       help='Window size in samples (default: 512 = 4s at 128Hz)')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Window overlap fraction (default: 0.5)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout rate (default: 0.3)')
    parser.add_argument('--output_dir', type=str, default='combined_cnnloc_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("COMBINED CNN-LOC - Das (MWF) + Fulsang (MWF) CNN-LOC Training")
    print("="*80)
    print(f"Using CNN-LOC architecture from FULCNN")
    print(f"  Window size: {args.window_size} samples")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.num_epochs}")
    
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
    
    # Create PyTorch dataset
    print("\n" + "="*80)
    print("CREATING PYTORCH DATASET")
    print("="*80)
    pytorch_dataset = CombinedCNNLOCDataset(combined_dataset, transform_eeg=True)
    
    # Split dataset based on method
    if args.split_method == 'subject':
        print("\nUsing SUBJECT-LEVEL splitting (prevents data leakage)")
        # Use window split for now (subject split requires metadata mapping)
        print("Note: Subject-level split requires metadata - using window split")
        train_dataset, val_dataset, test_dataset = split_dataset(pytorch_dataset)
    elif args.split_method == 'window':
        print("\nUsing WINDOW-LEVEL splitting (may have data leakage)")
        train_dataset, val_dataset, test_dataset = split_dataset(pytorch_dataset)
    else:  # both
        print("\n" + "="*80)
        print("COMPARING BOTH SPLIT METHODS")
        print("="*80)
        print("Note: Subject-level split requires CombinedCNNLOCSub.py")
        print("Using window-level split for now")
        train_dataset, val_dataset, test_dataset = split_dataset(pytorch_dataset)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # Create CNN-LOC model
    print("\n" + "="*80)
    print("INITIALIZING CNN-LOC MODEL")
    print("="*80)
    model = CNNLOCModel(
        input_channels=combined_dataset.n_channels,
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
        learning_rate=args.learning_rate
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

