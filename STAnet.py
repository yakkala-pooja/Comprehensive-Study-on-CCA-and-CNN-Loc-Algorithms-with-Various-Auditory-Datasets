#!/usr/bin/env python3
"""
STAnet - SpatioTemporal Attention Network for Auditory Attention Detection with EEG

This module implements STAnet as described in:
"Leveraging Graphic and Convolutional Neural Networks for Auditory Attention Detection 
with EEG on Das Dataset" by Pahuja et al., Interspeech 2024.

STAnet architecture:
- Spatial Attention Mechanism: Assigns dynamic weights to EEG channels
- Temporal Attention Mechanism: Assigns weights to temporal patterns
- Graph Convolutional Networks: Models relationships between EEG channels
- Convolutional Layers: Extract hierarchical features
- Fully Connected Layers: Classification output
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                           precision_recall_fscore_support, roc_auc_score, roc_curve,
                           precision_recall_curve, average_precision_score,
                           matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
                           f1_score)
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# Add telluride_decoding to path
sys.path.append('telluride_decoding')

try:
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except ImportError:
    print("Warning: TensorFlow not available, TFRecord loading may not work")
    tf = None

from pathlib import Path
from scipy import signal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GraphConvolution(nn.Module):
    """
    Graph Convolutional Layer for modeling relationships between EEG channels.
    Implements graph convolution using adjacency matrix.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of graph convolution.
        
        Args:
            x: Input features [batch, num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes]
        
        Returns:
            Output features [batch, num_nodes, out_features]
        """
        # x shape: [batch, num_nodes, in_features]
        # adj shape: [num_nodes, num_nodes]
        # weight shape: [in_features, out_features]
        
        batch_size, num_nodes, in_features = x.size()
        
        # Linear transformation: [batch, num_nodes, in_features] @ [in_features, out_features]
        # -> [batch, num_nodes, out_features]
        support = torch.matmul(x, self.weight)
        
        # Graph convolution: adj @ support for each batch
        # adj: [num_nodes, num_nodes]
        # support: [batch, num_nodes, out_features]
        # We want: [batch, num_nodes, out_features] where output[b] = adj @ support[b]
        
        # Use batch matrix multiplication: bmm expects [batch, n, m] @ [batch, m, p]
        # Expand adj to [batch, num_nodes, num_nodes]
        adj_batch = adj.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_nodes, num_nodes]
        
        # Batch matrix multiplication: [batch, num_nodes, num_nodes] @ [batch, num_nodes, out_features]
        # -> [batch, num_nodes, out_features]
        output = torch.bmm(adj_batch, support)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class SpatialAttention(nn.Module):
    """
    Spatial Attention Mechanism for assigning weights to EEG channels.
    This module learns to focus on the most relevant EEG channels for auditory attention detection.
    """
    
    def __init__(self, num_channels: int, reduction: int = 16):
        super(SpatialAttention, self).__init__()
        self.num_channels = num_channels
        self.reduction = max(1, reduction)
        
        # Spatial attention network
        self.spatial_attention = nn.Sequential(
            nn.Linear(num_channels, num_channels // self.reduction),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // self.reduction, num_channels),
            nn.Sigmoid()
        )
        
        # Channel-wise feature extraction
        self.channel_conv = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(num_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of spatial attention.
        
        Args:
            x: Input EEG data [batch, channels, time, features]
        
        Returns:
            Spatially attended features [batch, channels, time, features]
        """
        batch_size, channels, time, features = x.size()
        
        # Global average pooling over time and frequency dimensions
        spatial_avg = torch.mean(x, dim=[2, 3], keepdim=False)  # [batch, channels]
        
        # Compute spatial attention weights
        spatial_weights = self.spatial_attention(spatial_avg)  # [batch, channels]
        spatial_weights = spatial_weights.unsqueeze(2).unsqueeze(3)  # [batch, channels, 1, 1]
        
        # Apply spatial attention
        attended_features = x * spatial_weights
        
        # Channel-wise convolution
        attended_features = self.channel_conv(attended_features)
        attended_features = self.bn(attended_features)
        
        return attended_features


class TemporalAttention(nn.Module):
    """
    Temporal Attention Mechanism for assigning weights to temporal patterns.
    This module learns to focus on the most relevant time points for auditory attention detection.
    """
    
    def __init__(self, time_dim: int, reduction: int = 16):
        super(TemporalAttention, self).__init__()
        self.time_dim = time_dim
        self.reduction = max(1, reduction)
        
        # Temporal attention network
        self.temporal_attention = nn.Sequential(
            nn.Linear(time_dim, time_dim // self.reduction),
            nn.ReLU(inplace=True),
            nn.Linear(time_dim // self.reduction, time_dim),
            nn.Sigmoid()
        )
        
        # Temporal feature extraction
        self.temporal_conv = nn.Conv2d(time_dim, time_dim, kernel_size=1)
        self.bn = nn.BatchNorm2d(time_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of temporal attention.
        
        Args:
            x: Input features [batch, channels, time, features]
        
        Returns:
            Temporally attended features [batch, channels, time, features]
        """
        batch_size, channels, time, features = x.size()
        
        # Global average pooling over channel and frequency dimensions
        temporal_avg = torch.mean(x, dim=[1, 3], keepdim=False)  # [batch, time]
        
        # Compute temporal attention weights
        temporal_weights = self.temporal_attention(temporal_avg)  # [batch, time]
        temporal_weights = temporal_weights.unsqueeze(1).unsqueeze(3)  # [batch, 1, time, 1]
        
        # Apply temporal attention
        attended_features = x * temporal_weights
        
        # Temporal convolution (transpose to apply temporal conv)
        # Transpose to [batch, time, channels, features] for temporal conv
        attended_features = attended_features.permute(0, 2, 1, 3)  # [batch, time, channels, features]
        attended_features = self.temporal_conv(attended_features)
        attended_features = self.bn(attended_features)
        attended_features = attended_features.permute(0, 2, 1, 3)  # Back to [batch, channels, time, features]
        
        return attended_features


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer combining GCN with attention mechanism.
    """
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1, alpha: float = 0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of graph attention layer.
        
        Args:
            x: Input features [batch, num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes]
        
        Returns:
            Output features [batch, num_nodes, out_features]
        """
        batch_size, num_nodes, _ = x.size()
        
        # Linear transformation
        h = torch.matmul(x, self.W)  # [batch, num_nodes, out_features]
        
        # Compute attention coefficients
        a_input = torch.cat([h.repeat(1, 1, num_nodes).view(batch_size, num_nodes * num_nodes, -1),
                            h.repeat(1, num_nodes, 1)], dim=2)  # [batch, num_nodes*num_nodes, 2*out_features]
        a_input = a_input.view(batch_size, num_nodes, num_nodes, 2 * self.out_features)
        
        e = self.leaky_relu(torch.matmul(a_input, self.a).squeeze(-1))  # [batch, num_nodes, num_nodes]
        
        # Mask attention with adjacency matrix
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.unsqueeze(0) > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = self.dropout_layer(attention)
        
        # Apply attention
        h_prime = torch.matmul(attention, h)  # [batch, num_nodes, out_features]
        
        return h_prime


class STAnetBackbone(nn.Module):
    """
    STAnet Backbone: Combines Graph Convolutional Networks with Spatial and Temporal Attention.
    """
    
    def __init__(self, num_channels: int = 64, time_steps: int = 32, 
                 num_features: int = 5, gcn_hidden: int = 64, dropout: float = 0.1):
        super(STAnetBackbone, self).__init__()
        
        self.num_channels = num_channels
        self.time_steps = time_steps
        self.num_features = num_features
        self.gcn_hidden = gcn_hidden
        
        # Build adjacency matrix for EEG channels (distance-based)
        self.adjacency_matrix = self._build_adjacency_matrix(num_channels)
        
        # Graph Convolutional Layers
        self.gcn1 = GraphConvolution(num_features, gcn_hidden)
        self.gcn2 = GraphConvolution(gcn_hidden, gcn_hidden)
        self.gcn_bn1 = nn.BatchNorm1d(num_channels)
        self.gcn_bn2 = nn.BatchNorm1d(num_channels)
        
        # Reshape for CNN processing: [batch, channels, time, features]
        # After GCN: [batch, channels, time, gcn_hidden]
        
        # Spatial Attention Module
        self.spatial_attention = SpatialAttention(num_channels, reduction=16)
        
        # Temporal Attention Module
        self.temporal_attention = TemporalAttention(time_steps, reduction=16)
        
        # Convolutional feature extraction layers
        self.conv1 = nn.Conv2d(gcn_hidden, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(256)
        
        # Pooling layers
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Calculate output size
        self._calculate_output_size()
        
        print(f"STAnet Backbone initialized:")
        print(f"  Input channels: {num_channels}")
        print(f"  Time steps: {time_steps}")
        print(f"  Features: {num_features}")
        print(f"  GCN hidden: {gcn_hidden}")
        print(f"  Output size: {self.output_size}")
    
    def _build_adjacency_matrix(self, num_channels: int) -> torch.Tensor:
        """
        Build adjacency matrix for EEG channels.
        Uses distance-based connectivity (all channels connected).
        """
        # Full connectivity (all channels connected)
        adj = torch.ones(num_channels, num_channels)
        
        # Self-connections
        adj = adj + torch.eye(num_channels)
        
        # Normalize adjacency matrix (symmetric normalization)
        rowsum = adj.sum(dim=1)
        d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        
        adj_normalized = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        
        return adj_normalized
    
    def _calculate_output_size(self):
        """Calculate the output feature size."""
        dummy_input = torch.randn(1, self.num_channels, self.time_steps, self.num_features)
        
        with torch.no_grad():
            x = self.forward(dummy_input)
            self.output_size = x.numel() // x.size(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through STAnet backbone.
        
        Args:
            x: Input EEG data [batch, channels, time, features]
        
        Returns:
            Flattened feature vector [batch, output_size]
        """
        batch_size = x.size(0)
        
        # Step 1: Graph Convolutional Processing
        # Input x shape: [batch, channels, time, features] = [batch, 64, 32, 5]
        batch_size, channels, time, features = x.size()
        
        # Reshape for GCN: [batch, channels, time, features] -> [batch * time, channels, features]
        # We want to process each time step independently with GCN
        x_gcn = x.permute(0, 2, 1, 3).contiguous()  # [batch, time, channels, features]
        x_gcn = x_gcn.view(batch_size * time, channels, features)  # [batch*time, channels, features]
        
        # Verify dimensions
        assert x_gcn.size(1) == channels, f"Expected {channels} channels, got {x_gcn.size(1)}"
        assert x_gcn.size(2) == features, f"Expected {features} features, got {x_gcn.size(2)}"
        
        # Apply GCN layers
        adj = self.adjacency_matrix.to(x.device)
        assert adj.size(0) == channels and adj.size(1) == channels, \
            f"Adjacency matrix shape {adj.shape} doesn't match channels {channels}"
        
        x_gcn = F.relu(self.gcn1(x_gcn, adj))  # [batch*time, channels, gcn_hidden]
        x_gcn = self.gcn_bn1(x_gcn.permute(0, 2, 1)).permute(0, 2, 1)  # BatchNorm on channel dimension
        x_gcn = F.relu(self.gcn2(x_gcn, adj))  # [batch*time, channels, gcn_hidden]
        x_gcn = self.gcn_bn2(x_gcn.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Reshape back: [batch*time, channels, gcn_hidden] -> [batch, channels, time, gcn_hidden]
        x_gcn = x_gcn.view(batch_size, time, channels, self.gcn_hidden)
        x_gcn = x_gcn.permute(0, 2, 1, 3)  # [batch, channels, time, gcn_hidden]
        
        # Step 2: Spatial Attention
        x_spatial = self.spatial_attention(x_gcn)
        
        # Step 3: Temporal Attention
        x_temporal = self.temporal_attention(x_spatial)
        
        # Step 4: Convolutional Feature Extraction
        x = F.relu(self.bn1(self.conv1(x_temporal)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Step 5: Global Average Pooling
        x = self.global_pool(x)
        
        # Flatten
        x = x.view(batch_size, -1)
        
        return x


class STAnetModel(nn.Module):
    """
    Complete STAnet Model for Auditory Attention Detection.
    Combines STAnet backbone with classifier head.
    """
    
    def __init__(self, num_channels: int = 64, time_steps: int = 32, 
                 num_features: int = 5, gcn_hidden: int = 64,
                 num_classes: int = 2, dropout_rate: float = 0.3):
        super(STAnetModel, self).__init__()
        
        # STAnet backbone
        self.backbone = STAnetBackbone(
            num_channels=num_channels,
            time_steps=time_steps,
            num_features=num_features,
            gcn_hidden=gcn_hidden,
            dropout=dropout_rate
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.backbone.output_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
        
        print(f"STAnet Model created:")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"  Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Use kaiming initialization for better gradient flow
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, GraphConvolution):
                # Initialize GCN weights properly
                if hasattr(m, 'weight'):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through STAnet.
        
        Args:
            x: Input EEG data [batch, channels, time, features]
        
        Returns:
            Classification logits [batch, num_classes]
        """
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class STAnetDataset(Dataset):
    """
    Dataset class for STAnet compatible with DAS dataset format.
    Loads EEG data from TFRecord files and converts to time-frequency representation.
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
        
        # DAS-specific parameters
        self.sampling_rate = 1000  # Hz (DAS uses 1000 Hz)
        self.n_channels = 64  # EEG channels
        self.attention_switch_duration = 20  # seconds
        
        # Cache for preprocessed windows
        self._window_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Load DAS data
        self.eeg_data, self.labels, self.metadata = self._load_das_data()
        
        self.window_indices = self._create_das_windows()
        
        print(f"Loaded {len(self.window_indices)} DAS windows for {mode} mode")
        print(f"DAS EEG shape: {self.eeg_data.shape}")
        print(f"DAS Label distribution: {np.bincount(self.labels)}")
    
    def _load_das_data(self) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Load DAS TFRecord data."""
        if tf is None:
            raise ImportError("TensorFlow is required to load TFRecord files")
        
        # Find TFRecord files based on mode
        tfrecord_files = []
        
        # Check if mode-specific subdirectory exists (train, val, test)
        mode_dir = self.tfrecord_dir / self.mode
        if mode_dir.exists() and mode_dir.is_dir():
            # Load from mode-specific directory
            mode_files = list(mode_dir.glob("*.tfrecords"))
            if mode_files:
                tfrecord_files.extend(mode_files)
                print(f"Loading DAS data from {self.mode} directory: {mode_dir}")
        
        # If no mode-specific files found, try parent directory
        if not tfrecord_files:
            # Try direct files in tfrecord_dir
            direct_files = list(self.tfrecord_dir.glob("*.tfrecords"))
            if direct_files:
                tfrecord_files.extend(direct_files)
            
            # Try subdirectories (one level down)
            subdir_files = list(self.tfrecord_dir.glob("*/*.tfrecords"))
            if subdir_files:
                tfrecord_files.extend(subdir_files)
            
            # Try nested subdirectories (two levels down)
            nested_files = list(self.tfrecord_dir.glob("*/*/*.tfrecords"))
            if nested_files:
                tfrecord_files.extend(nested_files)
        
        if not tfrecord_files:
            raise ValueError(f"No TFRecord files found in {self.tfrecord_dir} for mode '{self.mode}'")
        
        print(f"Loading DAS data from {len(tfrecord_files)} files (mode: {self.mode})...")
        
        all_eeg_data = []
        all_labels = []
        all_metadata = []
        
        for tfrecord_file in tqdm(tfrecord_files, desc="Loading DAS data"):
            try:
                dataset = tf.data.TFRecordDataset(str(tfrecord_file))
                
                for record in dataset:
                    try:
                        example = tf.train.Example.FromString(record.numpy())
                        features = example.features.feature
                        
                        # Check required features
                        if 'eeg' not in features or 'attended_ear' not in features:
                            continue
                        
                        # Extract EEG data
                        eeg_values = features['eeg'].float_list.value
                        if not eeg_values or len(eeg_values) != self.n_channels:
                            continue
                        
                        eeg_data = np.array(eeg_values, dtype=np.float32).reshape(1, self.n_channels)
                        
                        # Validate EEG data
                        if np.any(np.isnan(eeg_data)) or np.any(np.isinf(eeg_data)):
                            continue
                        
                        # Extract label
                        attended_ear_values = features['attended_ear'].bytes_list.value
                        if not attended_ear_values:
                            continue
                        
                        attended_ear = attended_ear_values[0].decode('utf-8')
                        label = 0 if attended_ear == 'L' else 1
                        
                        # Extract metadata
                        subject_id = "unknown"
                        if 'subject_id' in features:
                            subject_values = features['subject_id'].bytes_list.value
                            if subject_values:
                                subject_id = subject_values[0].decode('utf-8')
                        
                        all_eeg_data.append(eeg_data)
                        all_labels.append(label)
                        all_metadata.append({
                            'subject_id': subject_id,
                            'file': tfrecord_file.name,
                            'attention_label': label,
                            'attended_ear': attended_ear
                        })
                    except Exception as e:
                        continue
            except Exception as e:
                print(f"Error loading {tfrecord_file}: {e}")
                continue
        
        if not all_eeg_data:
            raise ValueError("No valid EEG data loaded from TFRecord files")
        
        eeg_array = np.vstack(all_eeg_data)
        labels_array = np.array(all_labels, dtype=np.int64)
        
        return eeg_array, labels_array, all_metadata
    
    def _create_das_windows(self) -> List[Tuple[int, int]]:
        """Create sliding windows from EEG data."""
        step_size = max(1, int(self.window_size * (1 - self.overlap)))
        window_indices = []
        
        for i in range(0, len(self.eeg_data) - self.window_size + 1, step_size):
            label = self.labels[i + self.window_size // 2]  # Use middle label
            window_indices.append((i, label))
        
        return window_indices
    
    def _das_eeg_preprocessing(self, eeg_window: np.ndarray) -> np.ndarray:
        """DAS-specific EEG preprocessing."""
        # Z-score normalization
        eeg_window = eeg_window - np.mean(eeg_window, axis=0, keepdims=True)
        eeg_window = eeg_window / (np.std(eeg_window, axis=0, keepdims=True) + 1e-8)
        
        # Bandpass filtering (0.5-40 Hz)
        try:
            sos = signal.butter(4, [0.5, 40], btype='band', fs=self.sampling_rate, output='sos')
            filtered_eeg = signal.sosfiltfilt(sos, eeg_window, axis=0)
        except:
            filtered_eeg = eeg_window
        
        return filtered_eeg.astype(np.float32)
    
    def _eeg_to_timefreq_das(self, eeg_window: np.ndarray) -> np.ndarray:
        """Convert EEG to time-frequency representation for DAS."""
        time_freq_data = []
        
        for ch in range(eeg_window.shape[1]):
            # Compute spectrogram
            f, t, Sxx = signal.spectrogram(
                eeg_window[:, ch],
                fs=self.sampling_rate,
                nperseg=min(128, len(eeg_window)),
                noverlap=64,
                window='hann'
            )
            
            # Extract frequency bands
            freq_bands = [
                (0.5, 4),   # Delta
                (4, 8),     # Theta
                (8, 13),    # Alpha
                (13, 25),   # Beta
                (25, 40)    # Gamma
            ]
            
            # Extract band power for each time point
            band_powers = []
            for low_freq, high_freq in freq_bands:
                if high_freq >= self.sampling_rate / 2:
                    high_freq = self.sampling_rate / 2 - 1
                
                freq_mask = (f >= low_freq) & (f <= high_freq)
                if np.any(freq_mask):
                    band_power = np.mean(Sxx[freq_mask, :], axis=0)
                else:
                    band_power = np.zeros(Sxx.shape[1])
                
                band_powers.append(band_power)
            
            # Stack band powers: (n_bands, n_time_points)
            channel_tf = np.vstack(band_powers)
            time_freq_data.append(channel_tf)
        
        # Combine all channels: (n_channels, n_bands, n_time_points)
        time_freq_array = np.array(time_freq_data)
        
        # Ensure consistent time dimension (pad or crop to window_size)
        if time_freq_array.shape[2] != self.window_size:
            # Interpolate to desired time dimension
            from scipy.interpolate import interp1d
            time_original = np.linspace(0, 1, time_freq_array.shape[2])
            time_target = np.linspace(0, 1, self.window_size)
            
            resampled = []
            for ch in range(time_freq_array.shape[0]):
                for band in range(time_freq_array.shape[1]):
                    f_interp = interp1d(time_original, time_freq_array[ch, band, :], 
                                       kind='linear', fill_value='extrapolate')
                    resampled.append(f_interp(time_target))
            
            time_freq_array = np.array(resampled).reshape(
                time_freq_array.shape[0], time_freq_array.shape[1], self.window_size
            )
        
        # Transpose to (channels, time, features)
        time_freq_array = time_freq_array.transpose(0, 2, 1)
        
        return time_freq_array.astype(np.float32)
    
    def __len__(self):
        return len(self.window_indices)
    
    def __getitem__(self, idx):
        data_idx, label = self.window_indices[idx]
        
        # Check cache
        cache_key = (data_idx, self.mode)
        if cache_key in self._window_cache:
            self._cache_hits += 1
            return self._window_cache[cache_key]
        
        self._cache_misses += 1
        
        # Extract window
        window_eeg = self.eeg_data[data_idx:data_idx + self.window_size]
        
        # Apply preprocessing
        try:
            window_eeg = self._das_eeg_preprocessing(window_eeg)
        except:
            window_eeg = window_eeg - np.mean(window_eeg, axis=0, keepdims=True)
            window_eeg = window_eeg / (np.std(window_eeg, axis=0, keepdims=True) + 1e-8)
        
        # Convert to time-frequency representation
        if self.transform_eeg:
            try:
                window_eeg = self._eeg_to_timefreq_das(window_eeg)
            except:
                # Fallback: use raw data with dummy frequency dimension
                window_eeg = np.expand_dims(window_eeg, axis=2)
                window_eeg = np.repeat(window_eeg, 5, axis=2)
        
        # Ensure proper shape: (channels, time, features)
        if window_eeg.ndim == 2:
            window_eeg = np.expand_dims(window_eeg, axis=2)
            window_eeg = np.repeat(window_eeg, 5, axis=2)
        
        # Convert to tensor
        window_tensor = torch.FloatTensor(window_eeg)
        label_tensor = torch.LongTensor([label])
        
        # Verify tensor shape: should be (channels, time, features) = (64, 32, 5)
        if window_tensor.ndim != 3:
            raise ValueError(f"Expected 3D tensor (channels, time, features), got {window_tensor.ndim}D with shape {window_tensor.shape}")
        
        expected_channels, expected_time, expected_features = self.n_channels, self.window_size, 5
        actual_channels, actual_time, actual_features = window_tensor.shape
        
        if actual_channels != expected_channels or actual_time != expected_time:
            # Try to fix: maybe channels and time are swapped
            if actual_time == expected_channels and actual_channels == expected_time:
                print(f"WARNING: Swapping channels and time dimensions. Shape: {window_tensor.shape} -> ", end="")
                window_tensor = window_tensor.permute(1, 0, 2)  # Swap first two dimensions
                print(f"{window_tensor.shape}")
            else:
                raise ValueError(f"Tensor shape mismatch: Expected ({expected_channels}, {expected_time}, {expected_features}), "
                               f"got ({actual_channels}, {actual_time}, {actual_features})")
        
        # Cache if cache not full
        if len(self._window_cache) < self.cache_size:
            self._window_cache[cache_key] = (window_tensor, label_tensor)
        
        return window_tensor, label_tensor


class STAnetTrainer:
    """
    Trainer class for STAnet model.
    """
    
    def __init__(self, model: STAnetModel, train_loader: DataLoader,
                 val_loader: DataLoader = None, test_loader: DataLoader = None,
                 learning_rate: float = 0.001, num_epochs: int = 100,
                 device: torch.device = device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device
        
        # Loss and optimizer (matching DASCNN settings)
        self.criterion = nn.CrossEntropyLoss()
        # Use AdamW with weight decay for better generalization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        # Use OneCycleLR for better learning rate scheduling
        # Will be initialized in train() method with actual steps_per_epoch
        self.scheduler = None
        self.scheduler_type = 'onecycle'
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, desc="Training")):
            data = data.to(self.device)
            target = target.squeeze().to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Step scheduler if using OneCycleLR (step per batch)
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, loader: DataLoader):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(loader, desc="Validating"):
                data = data.to(self.device)
                target = target.squeeze().to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100. * correct / total
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(all_targets, all_preds)
        balanced_acc = balanced_accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        
        # Precision, recall, F1 per class
        precision, recall, f1_per_class, support = precision_recall_fscore_support(
            all_targets, all_preds, average=None, zero_division=0
        )
        precision_weighted = precision_recall_fscore_support(
            all_targets, all_preds, average='weighted', zero_division=0
        )[0]
        recall_weighted = precision_recall_fscore_support(
            all_targets, all_preds, average='weighted', zero_division=0
        )[1]
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(all_targets, all_preds) if len(np.unique(all_targets)) > 1 else 0.0
        except:
            roc_auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        
        # Additional metrics
        mcc = matthews_corrcoef(all_targets, all_preds)
        kappa = cohen_kappa_score(all_targets, all_preds)
        
        metrics = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'f1_score': f1,
            'f1_score_weighted': f1,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'precision_per_class': precision.tolist() if hasattr(precision, 'tolist') else list(precision),
            'recall_per_class': recall.tolist() if hasattr(recall, 'tolist') else list(recall),
            'f1_per_class': f1_per_class.tolist() if hasattr(f1_per_class, 'tolist') else list(f1_per_class),
            'support': support.tolist() if hasattr(support, 'tolist') else list(support),
            'roc_auc': roc_auc,
            'matthews_corrcoef': mcc,
            'cohens_kappa': kappa,
            'confusion_matrix': cm.tolist() if hasattr(cm, 'tolist') else cm
        }
        
        return epoch_loss, epoch_acc, metrics
    
    def train(self):
        """Train the model."""
        best_val_acc = 0.0
        best_model_state = None
        
        # Initialize OneCycleLR scheduler
        if self.scheduler is None and self.scheduler_type == 'onecycle':
            steps_per_epoch = len(self.train_loader)
            total_steps = steps_per_epoch * self.num_epochs
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.learning_rate,
                total_steps=total_steps,
                pct_start=0.3,
                anneal_strategy='cos',
                cycle_momentum=True,
                base_momentum=0.85,
                max_momentum=0.95
            )
            print(f"Initialized OneCycleLR scheduler:")
            print(f"  Steps per epoch: {steps_per_epoch}")
            print(f"  Total steps: {total_steps}")
            print(f"  Max LR: {self.learning_rate}")
        
        print(f"Starting training for {self.num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Learning rate: {self.learning_rate}")
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
            print(f"  - Train Accuracy: {train_acc:.4f}%")
            
            # Validate
            if self.val_loader is not None:
                val_loss, val_acc, val_metrics = self.validate(self.val_loader)
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)
                
                print(f"\nValidation Results:")
                print(f"  - Val Loss: {val_loss:.4f}")
                print(f"  - Val Accuracy: {val_acc:.2f}% ({val_metrics['accuracy']:.4f})")
                print(f"  - Balanced Accuracy: {val_metrics['balanced_accuracy']:.4f}")
                print(f"  - F1 Score (weighted): {val_metrics['f1_score']:.4f}")
                print(f"  - Precision (weighted): {val_metrics['precision_weighted']:.4f}")
                print(f"  - Recall (weighted): {val_metrics['recall_weighted']:.4f}")
                print(f"  - ROC-AUC: {val_metrics['roc_auc']:.4f}")
                print(f"  - Matthews Correlation Coefficient: {val_metrics['matthews_corrcoef']:.4f}")
                print(f"  - Cohen's Kappa: {val_metrics['cohens_kappa']:.4f}")
                
                # Per-class metrics
                if len(val_metrics['precision_per_class']) >= 2:
                    print(f"\n  Per-Class Metrics:")
                    for i, (p, r, f, s) in enumerate(zip(
                        val_metrics['precision_per_class'],
                        val_metrics['recall_per_class'],
                        val_metrics['f1_per_class'],
                        val_metrics['support']
                    )):
                        print(f"    Class {i}: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}, Support={s}")
                
                # Confusion matrix
                print(f"\n  Confusion Matrix:")
                cm = np.array(val_metrics['confusion_matrix'])
                print(f"    {cm}")
                
                # Learning rate scheduling (for epoch-based schedulers)
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                # OneCycleLR is stepped per batch, not per epoch
                
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  - Current Learning Rate: {current_lr:.6f}")
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = self.model.state_dict().copy()
                    print(f"\n  *** New best validation accuracy: {best_val_acc:.2f}% ***")
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\n{'='*80}")
            print(f"Training Complete - Best Validation Accuracy: {best_val_acc:.2f}%")
            print(f"{'='*80}")
        
        # Test
        if self.test_loader is not None:
            print(f"\n{'='*80}")
            print("TEST SET EVALUATION")
            print(f"{'='*80}")
            test_loss, test_acc, test_metrics = self.validate(self.test_loader)
            
            print(f"\nTest Set Results:")
            print(f"  - Test Loss: {test_loss:.4f}")
            print(f"  - Test Accuracy: {test_acc:.2f}% ({test_metrics['accuracy']:.4f})")
            print(f"  - Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
            print(f"  - F1 Score (weighted): {test_metrics['f1_score']:.4f}")
            print(f"  - Precision (weighted): {test_metrics['precision_weighted']:.4f}")
            print(f"  - Recall (weighted): {test_metrics['recall_weighted']:.4f}")
            print(f"  - ROC-AUC: {test_metrics['roc_auc']:.4f}")
            print(f"  - Matthews Correlation Coefficient: {test_metrics['matthews_corrcoef']:.4f}")
            print(f"  - Cohen's Kappa: {test_metrics['cohens_kappa']:.4f}")
            
            # Per-class metrics
            if len(test_metrics['precision_per_class']) >= 2:
                print(f"\n  Per-Class Metrics:")
                for i, (p, r, f, s) in enumerate(zip(
                    test_metrics['precision_per_class'],
                    test_metrics['recall_per_class'],
                    test_metrics['f1_per_class'],
                    test_metrics['support']
                )):
                    print(f"    Class {i}: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}, Support={s}")
            
            # Confusion matrix
            print(f"\n  Confusion Matrix:")
            cm = np.array(test_metrics['confusion_matrix'])
            print(f"    {cm}")
            print(f"{'='*80}\n")
            
            return test_metrics
        
        return None


if __name__ == "__main__":
    # Test STAnet model
    print("Testing STAnet Model...")
    
    # Create model
    model = STAnetModel(
        num_channels=64,
        time_steps=32,
        num_features=5,
        gcn_hidden=64,
        num_classes=2,
        dropout_rate=0.3
    ).to(device)
    
    # Test forward pass
    dummy_input = torch.randn(2, 64, 32, 5).to(device)
    output = model(dummy_input)
    
    print(f"\nModel test successful!")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")

