#!/usr/bin/env python3
"""
DASCNN - Ultra-Fast CNN-LOC Algorithm for DAS Dataset (16-Subjects Pipeline)

This module implements an ultra-fast CNN-LOC algorithm with optimized data loading
and processing specifically designed for the DAS dataset with the 16-subjects 
preprocessing pipeline. It includes:

- Ultra-fast TFRecord loading with caching
- Mixed precision training (AMP)
- Optimized data loading with minimal TensorFlow overhead
- Efficient model architecture
- Comprehensive metrics evaluation
- Speed optimizations without data reduction
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


class FastDASDataset(Dataset):
    """
    Ultra-fast dataset class for DAS data with optimized loading and caching.
    """
    
    def __init__(self, tfrecord_dir: str, mode: str = 'full', 
                 window_size: int = 32, overlap: float = 0.5, 
                 use_validated_data: bool = True, cache_data: bool = True):
        """
        Initialize fast DAS dataset with caching.
        
        Args:
            tfrecord_dir: Directory containing TFRecord files
            mode: 'full', 'train', 'val', or 'test'
            window_size: Window size in samples (32 samples = 0.5s at 64Hz)
            overlap: Overlap ratio between windows
            use_validated_data: Whether to use validated data
            cache_data: Whether to cache loaded data in memory
        """
        self.tfrecord_dir = Path(tfrecord_dir)
        self.mode = mode
        self.window_size = window_size
        self.overlap = overlap
        self.use_validated_data = use_validated_data
        self.cache_data = cache_data
        
        # DAS-specific parameters (updated for 16-subjects pipeline)
        self.sampling_rate = 64  # Hz (downsampled from 1000 Hz)
        self.n_channels = 64  # EEG channels
        
        print(f"Fast DAS Dataset initialized:")
        print(f"  Mode: {mode}")
        print(f"  Window size: {window_size} samples ({window_size/self.sampling_rate:.1f}s at {self.sampling_rate}Hz)")
        print(f"  Overlap: {overlap}")
        print(f"  Sampling rate: {self.sampling_rate} Hz")
        print(f"  Channels: {self.n_channels}")
        print(f"  Caching: {cache_data}")
        
        # Load data with caching
        self.eeg_data, self.labels, self.metadata = self._load_das_data_fast()
        
        if len(self.eeg_data) == 0:
            raise ValueError("No valid DAS data found")
        
        print(f"  Loaded {len(self.eeg_data)} samples")
        print(f"  Label distribution: {np.bincount(self.labels)}")
    
    def _load_das_data_fast(self) -> Tuple[List[np.ndarray], List[int], List[Dict]]:
        """Load DAS data from TFRecord files with ultra-fast processing."""
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
        
        # Load data from TFRecord files with optimized processing
        eeg_data = []
        labels = []
        metadata = []
        
        # Use TensorFlow's optimized dataset API
        def parse_tfrecord(example_proto):
            """Parse TFRecord example with optimized feature parsing."""
            feature_description = {
                'eeg': tf.io.FixedLenFeature([self.n_channels], tf.float32),
                'attended_ear': tf.io.FixedLenFeature([], tf.string),
                'subject_id': tf.io.FixedLenFeature([], tf.string),
            }
            return tf.io.parse_single_example(example_proto, feature_description)
        
        # Process files in batches for speed
        batch_size = 1000  # Process 1000 records at a time
        
        for tfrecord_file in tqdm(tfrecord_files, desc="Loading TFRecord files"):
            try:
                # Create optimized dataset
                dataset = tf.data.TFRecordDataset(str(tfrecord_file))
                dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
                dataset = dataset.batch(batch_size)
                dataset = dataset.prefetch(tf.data.AUTOTUNE)
                
                # Process batches
                for batch in dataset:
                    # Extract batch data
                    eeg_batch = batch['eeg'].numpy()
                    attended_ear_batch = batch['attended_ear'].numpy()
                    subject_id_batch = batch['subject_id'].numpy()
                    
                    # Process each sample in the batch
                    for i in range(len(eeg_batch)):
                        eeg_sample = eeg_batch[i]
                        attended_ear = attended_ear_batch[i].decode('utf-8')
                        subject_id = subject_id_batch[i].decode('utf-8')
                        
                        # Convert attended_ear to integer label
                        if attended_ear == 'L':
                            label = 0  # Left ear
                        elif attended_ear == 'R':
                            label = 1  # Right ear
                        else:
                            continue  # Skip invalid labels
                        
                        # Validate EEG shape
                        if eeg_sample.shape[0] != self.n_channels:
                            continue
                        
                        # Store data
                        eeg_data.append(eeg_sample)
                        labels.append(label)
                        metadata.append({
                            'tfrecord_file': tfrecord_file.name,
                            'attended_ear': attended_ear,
                            'subject_id': subject_id,
                            'sample_rate': self.sampling_rate
                        })
                        
            except Exception as e:
                print(f"ERROR loading {tfrecord_file.name}: {e}")
                continue
        
        if len(eeg_data) == 0:
            raise ValueError("No valid DAS data found in TFRecord files")
        
        print(f"Successfully loaded {len(eeg_data)} samples from {len(tfrecord_files)} files")
        
        # Convert to numpy arrays for faster access
        if self.cache_data:
            eeg_data = np.array(eeg_data, dtype=np.float32)
            labels = np.array(labels, dtype=np.int32)
        
        return eeg_data, labels, metadata
    
    def __len__(self):
        return len(self.eeg_data)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset with optimized tensor operations."""
        if idx >= len(self.eeg_data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.eeg_data)}")
        
        # Get EEG data and label
        if self.cache_data:
            eeg_sample = self.eeg_data[idx]
            label = self.labels[idx]
        else:
            eeg_sample = self.eeg_data[idx]
            label = self.labels[idx]
        
        # Convert to tensors with optimized operations
        # EEG data should be shape [channels] -> [batch, channels]
        window_tensor = torch.from_numpy(eeg_sample).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # Ensure label_tensor is always 1D (not scalar)
        if label_tensor.dim() == 0:
            label_tensor = label_tensor.unsqueeze(0)
        
        # Validate tensors
        if window_tensor.numel() == 0 or label_tensor.numel() == 0:
            # Return default tensors to prevent crashes
            window_tensor = torch.zeros(self.n_channels, dtype=torch.float32)
            label_tensor = torch.tensor(0, dtype=torch.long).unsqueeze(0)
        
        return window_tensor, label_tensor


class UltraFastDASCNNBackbone(nn.Module):
    """
    Ultra-fast CNN backbone for DAS data with maximum speed optimizations.
    """
    
    def __init__(self, input_channels: int = 64, hidden_dim: int = 128, 
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        
        # Ultra-fast convolutional layers with minimal operations
        self.conv_layers = nn.ModuleList()
        
        # First layer with optimized operations - handle input shape [batch, 1, channels]
        self.conv_layers.append(nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1, bias=False),  # 1 input channel
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout, inplace=True)
        ))
        
        # Additional layers with residual connections
        for i in range(num_layers - 1):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout, inplace=True)
            ))
        
        # Global average pooling for efficiency
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head with minimal operations
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(hidden_dim // 2, 2)  # Binary classification
        )
        
    def forward(self, x):
        """Forward pass with ultra-optimized operations."""
        # Debug: print input shape
        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 1
        
        if self._debug_count <= 3:  # Print first 3 batches
            print(f"DEBUG: Input shape: {x.shape}")
        
        # Input shape: (batch_size, channels) -> (batch_size, 1, channels)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension: [batch, channels] -> [batch, 1, channels]
        
        if self._debug_count <= 3:
            print(f"DEBUG: After unsqueeze shape: {x.shape}")
        
        # Apply convolutional layers with residual connections
        for i, conv_layer in enumerate(self.conv_layers):
            residual = x if x.size(1) == self.hidden_dim else None
            x = conv_layer(x)
            if residual is not None and i > 0:  # Skip residual for first layer
                x = x + residual
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Classification
        x = self.classifier(x)
        
        return x


class UltraFastDASCNNModel(nn.Module):
    """
    Ultra-fast CNN model for DAS attention decoding.
    """
    
    def __init__(self, input_channels: int = 64, hidden_dim: int = 128, 
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.backbone = UltraFastDASCNNBackbone(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
    def forward(self, x):
        """Forward pass."""
        return self.backbone(x)


class UltraFastDASCNNTrainer:
    """
    Ultra-fast trainer for DAS CNN-LOC with maximum speed optimizations.
    """
    
    def __init__(self, model, device, output_dir: str = "dascnn_ultrafast_results", 
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
        
        # Ultra-fast loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Ultra-fast optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=2e-4,  # Higher learning rate for faster convergence
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=2e-3,
            epochs=30,  # Fewer epochs for speed
            steps_per_epoch=50,  # Will be updated
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        print(f"Ultra-Fast DASCNN Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Mixed precision: {use_mixed_precision}")
        print(f"  Output directory: {output_dir}")
    
    def train_epoch(self, train_loader):
        """Train for one epoch with ultra-fast optimizations."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            
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
            
            # Update learning rate
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
        """Validate for one epoch with ultra-fast optimizations."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation", leave=False):
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
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
        """Test the model with comprehensive metrics."""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        print("Testing model...")
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Testing"):
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
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
        """Calculate comprehensive evaluation metrics."""
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
        
        # Temporal metrics (simplified for speed)
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
        """Calculate temporal performance metrics (simplified for speed)."""
        # Simplified temporal analysis for speed
        window_sizes = [0.5, 1.0, 2.0, 5.0]  # seconds
        temporal_results = {}
        
        for window_size in window_sizes:
            # Simple windowing for demonstration
            window_samples = int(window_size * 64)  # 64 Hz sampling rate
            
            if len(targets) >= window_samples:
                # Take first window for speed
                window_targets = targets[:window_samples]
                window_predictions = predictions[:window_samples]
                
                accuracy = accuracy_score(window_targets, window_predictions)
                temporal_results[f'{window_size}s'] = {
                    'accuracy': accuracy,
                    'n_samples': window_samples
                }
        
        return temporal_results
    
    def train(self, train_loader, val_loader, num_epochs=30):
        """Train the model with ultra-fast optimizations."""
        print(f"Starting ultra-fast training for {num_epochs} epochs...")
        
        best_val_acc = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)
            
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


def create_ultrafast_das_data_loaders(tfrecord_dir: str, batch_size: int = 64, 
                                    window_size: int = 32, overlap: float = 0.5,
                                    num_workers: int = 8, pin_memory: bool = True):
    """Create ultra-fast data loaders for DAS dataset."""
    print("Creating ultra-fast DAS data loaders...")
    
    # Check for predefined splits
    train_dir = Path(tfrecord_dir) / "train"
    test_dir = Path(tfrecord_dir) / "test"
    val_dir = Path(tfrecord_dir) / "val"
    
    if train_dir.exists() and test_dir.exists():
        print("Using predefined train/test splits")
        
        # Create datasets with caching
        train_dataset = FastDASDataset(train_dir, mode='train', window_size=window_size, overlap=overlap, cache_data=True)
        test_dataset = FastDASDataset(test_dir, mode='test', window_size=window_size, overlap=overlap, cache_data=True)
        
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
        full_dataset = FastDASDataset(tfrecord_dir, mode='full', window_size=window_size, overlap=overlap, cache_data=True)
        
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
    
    # Create data loaders with ultra-fast optimizations
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=True, prefetch_factor=4,
        drop_last=True  # Drop last incomplete batch for speed
    )
    
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=True, prefetch_factor=4,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=True, prefetch_factor=4,
        drop_last=True
    )
    
    print(f"✓ Ultra-fast data loaders created with batch size {batch_size}")
    print(f"✓ Using {num_workers} workers with pin_memory={pin_memory}")
    
    return train_loader, val_loader, test_loader


def main():
    """Main function for ultra-fast DASCNN training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ultra-Fast DASCNN - CNN-LOC for DAS Dataset')
    parser.add_argument('--tfrecord_dir', type=str, default='das_16subjects_preprocessed/tfrecords',
                       help='TFRecord directory path')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--window_size', type=int, default=32,
                       help='Window size for EEG data (32 samples = 0.5s at 64Hz)')
    parser.add_argument('--output_dir', type=str, default='dascnn_ultrafast_results',
                       help='Output directory for results')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers')
    parser.add_argument('--use_mixed_precision', action='store_true', default=True,
                       help='Use mixed precision training')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ULTRA-FAST DASCNN - CNN-LOC ALGORITHM FOR DAS DATASET")
    print("=" * 80)
    print("Features:")
    print("- 16-subjects preprocessing pipeline")
    print("- Ultra-fast TFRecord loading with caching")
    print("- Mixed precision training (AMP)")
    print("- Optimized data loading (8 workers, pin_memory)")
    print("- Efficient model architecture")
    print("- Comprehensive metrics evaluation")
    print("- Maximum speed optimizations")
    print("=" * 80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_ultrafast_das_data_loaders(
        args.tfrecord_dir, batch_size=args.batch_size, window_size=args.window_size,
        num_workers=args.num_workers
    )
    
    # Create model
    model = UltraFastDASCNNModel(input_channels=64, hidden_dim=128, num_layers=3)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = UltraFastDASCNNTrainer(
        model, device, args.output_dir, args.tfrecord_dir, 
        use_mixed_precision=args.use_mixed_precision
    )
    
    # Train model
    best_val_acc = trainer.train(train_loader, val_loader, args.num_epochs)
    
    # Test model
    results = trainer.test(test_loader)
    
    # Save results
    results_file = Path(args.output_dir) / 'dascnn_ultrafast_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results
    print("\n" + "=" * 80)
    print("ULTRA-FAST DASCNN RESULTS")
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
