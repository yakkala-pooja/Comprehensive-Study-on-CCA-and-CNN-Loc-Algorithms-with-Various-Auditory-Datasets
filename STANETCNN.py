#!/usr/bin/env python3
"""
STANETCNN - Spatial-Temporal Attention CNN with Graph Convolution Network

Dual-branch architecture for auditory attention decoding:
1. STAtNet Branch: Spatial-Temporal Attention CNN
   - Spatial Attention Module
   - Temporal Feature Extraction (1D Conv + Max-pooling)
   - Temporal Attention
   - Classification Head

2. ST-GCN Branch: Spatio-Temporal Graph Convolution Network
   - Graph Construction (channels as nodes, correlations as edges)
   - Graph Convolution Module
   - Temporal Attention
   - Classification Head

3. Soft-Voting Fusion Layer: Combines outputs from both branches
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import json
import pickle
from datetime import datetime
from pathlib import Path
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                           precision_recall_fscore_support, roc_auc_score, roc_curve,
                           precision_recall_curve, average_precision_score,
                           matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
                           f1_score)
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.append('.')

# Import dataset classes
try:
    from CombinedDataset import CombinedDataset
except ImportError:
    print("Warning: CombinedDataset not found")
    CombinedDataset = None

try:
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except ImportError:
    print("Warning: TensorFlow not available")
    tf = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================================
# Graph Convolution Module
# ============================================================================

class GraphConvolution(nn.Module):
    """Graph Convolutional Layer for modeling relationships between EEG channels."""
    
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
        batch_size, num_nodes, in_features = x.size()
        
        # Linear transformation
        support = torch.matmul(x, self.weight)  # [batch, num_nodes, out_features]
        
        # Graph convolution: adj @ support for each batch
        adj_batch = adj.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_nodes, num_nodes]
        output = torch.bmm(adj_batch, support)  # [batch, num_nodes, out_features]
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output


# ============================================================================
# STAtNet Branch: Spatial-Temporal Attention CNN
# ============================================================================

class SpatialAttentionModule(nn.Module):
    """Spatial Attention Module - learns which EEG channels are most important."""
    
    def __init__(self, num_channels: int, reduction: int = 8):
        super(SpatialAttentionModule, self).__init__()
        self.num_channels = num_channels
        self.reduction = max(1, reduction)
        
        # Channel attention network
        # Input is already [batch, channels] from mean pooling, so no need for AdaptiveAvgPool1d
        self.channel_attention = nn.Sequential(
            nn.Linear(num_channels, num_channels // self.reduction),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // self.reduction, num_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of spatial attention.
        
        Args:
            x: Input EEG [batch, channels, time]
        
        Returns:
            Attention-weighted features [batch, channels, time]
        """
        # Compute channel attention weights
        # x: [batch, channels, time]
        channel_avg = torch.mean(x, dim=2, keepdim=False)  # [batch, channels]
        channel_weights = self.channel_attention(channel_avg)  # [batch, channels]
        channel_weights = channel_weights.unsqueeze(2)  # [batch, channels, 1]
        
        # Apply attention
        attended = x * channel_weights
        return attended


class TemporalFeatureExtractor(nn.Module):
    """Temporal Feature Extraction using 1D Convolutions and Max-pooling."""
    
    def __init__(self, in_channels: int, out_channels: int = 64):
        super(TemporalFeatureExtractor, self).__init__()
        
        # 1D Convolution layers for temporal pattern extraction
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels * 2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(out_channels * 2, out_channels * 2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(out_channels * 2)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of temporal feature extraction.
        
        Args:
            x: Input [batch, channels, time]
        
        Returns:
            Temporal features [batch, out_channels*2, reduced_time]
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        return x


class TemporalAttentionModule(nn.Module):
    """Temporal Attention Module - learns which time segments contribute most."""
    
    def __init__(self, time_dim: int, reduction: int = 8):
        super(TemporalAttentionModule, self).__init__()
        self.time_dim = max(2, time_dim)  # Ensure at least 2 for proper attention
        self.reduction = max(1, reduction)
        self.reduced_dim = max(1, self.time_dim // self.reduction)
        
        # Temporal attention network - NO AdaptiveAvgPool1d here, we work directly with time dimension
        self.temporal_attention = nn.Sequential(
            nn.Linear(self.time_dim, self.reduced_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.reduced_dim, self.time_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of temporal attention.
        
        Args:
            x: Input features [batch, channels, time]
        
        Returns:
            Temporally attended features [batch, channels, time]
        """
        batch_size, channels, time = x.size()
        
        # Handle case where time dimension is 1 or very small
        if time == 1:
            # No temporal attention needed, just return input
            return x
        
        # Compute temporal attention weights
        # x: [batch, channels, time]
        temporal_avg = torch.mean(x, dim=1, keepdim=False)  # [batch, time]
        
        # If actual time dimension differs from expected, adapt
        if time != self.time_dim:
            # Use adaptive approach - pool to expected dimension if needed
            if time > self.time_dim:
                # Pool down to expected dimension using adaptive pooling
                pool = nn.AdaptiveAvgPool1d(self.time_dim).to(x.device)
                temporal_avg_pooled = pool(temporal_avg.unsqueeze(1)).squeeze(1)  # [batch, time_dim]
            elif time < self.time_dim:
                # Interpolate up to expected dimension
                temporal_avg_pooled = F.interpolate(
                    temporal_avg.unsqueeze(1), 
                    size=self.time_dim, 
                    mode='linear', 
                    align_corners=False
                ).squeeze(1)  # [batch, time_dim]
            else:
                temporal_avg_pooled = temporal_avg
        else:
            temporal_avg_pooled = temporal_avg
        
        # Apply temporal attention - input is [batch, time_dim], output is [batch, time_dim]
        temporal_weights_pooled = self.temporal_attention(temporal_avg_pooled)  # [batch, time_dim]
        
        # Resize back to actual time dimension if needed
        if time != self.time_dim:
            temporal_weights = F.interpolate(
                temporal_weights_pooled.unsqueeze(1),
                size=time,
                mode='linear',
                align_corners=False
            ).squeeze(1)  # [batch, time]
        else:
            temporal_weights = temporal_weights_pooled
        
        temporal_weights = temporal_weights.unsqueeze(1)  # [batch, 1, time]
        
        # Apply attention
        attended = x * temporal_weights
        return attended


class STAtNetBranch(nn.Module):
    """STAtNet Branch: Spatial-Temporal Attention CNN."""
    
    def __init__(self, num_channels: int = 64, time_steps: int = 512, 
                 num_classes: int = 2, dropout_rate: float = 0.3):
        super(STAtNetBranch, self).__init__()
        
        self.num_channels = num_channels
        self.time_steps = time_steps
        
        # Spatial Attention Module
        self.spatial_attention = SpatialAttentionModule(num_channels, reduction=8)
        
        # Temporal Feature Extraction
        self.temporal_extractor = TemporalFeatureExtractor(num_channels, out_channels=64)
        
        # Temporal Attention Module
        # Calculate temporal dimension after feature extraction
        # After 3 pooling layers with stride 2: time_steps / 8
        reduced_time = max(1, time_steps // 8)
        # Ensure reduced_time is at least 2 for proper attention (or handle size 1 specially)
        self.temporal_attention = TemporalAttentionModule(max(2, reduced_time), reduction=8)
        
        # Calculate output size
        self._calculate_output_size()
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.output_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
    
    def _calculate_output_size(self):
        """Calculate output feature size."""
        dummy_input = torch.randn(1, self.num_channels, self.time_steps)
        
        with torch.no_grad():
            x = self.spatial_attention(dummy_input)
            x = self.temporal_extractor(x)
            x = self.temporal_attention(x)
            self.output_size = x.numel() // x.size(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through STAtNet branch.
        
        Args:
            x: Input EEG [batch, channels, time]
        
        Returns:
            Class logits [batch, num_classes]
        """
        # Spatial Attention
        x = self.spatial_attention(x)  # [batch, channels, time]
        
        # Temporal Feature Extraction
        x = self.temporal_extractor(x)  # [batch, channels*2, reduced_time]
        
        # Temporal Attention
        x = self.temporal_attention(x)  # [batch, channels*2, reduced_time]
        
        # Flatten
        x = x.view(x.size(0), -1)  # [batch, features]
        
        # Classification
        output = self.classifier(x)  # [batch, num_classes]
        
        return output


# ============================================================================
# ST-GCN Branch: Spatio-Temporal Graph Convolution Network
# ============================================================================

class GraphConstruction(nn.Module):
    """Graph Construction - builds adjacency matrix from functional correlations."""
    
    def __init__(self, num_channels: int, correlation_type: str = 'plv'):
        super(GraphConstruction, self).__init__()
        self.num_channels = num_channels
        self.correlation_type = correlation_type
        
        # Learnable adjacency matrix
        self.adjacency = nn.Parameter(torch.ones(num_channels, num_channels))
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize adjacency matrix."""
        # Initialize with identity + small random values
        self.adjacency.data = torch.eye(self.num_channels) + 0.1 * torch.randn(
            self.num_channels, self.num_channels
        )
        # Make symmetric
        self.adjacency.data = (self.adjacency.data + self.adjacency.data.t()) / 2
    
    def compute_functional_correlation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute functional correlation (PLV-like) between channels.
        
        Args:
            x: Input EEG [batch, channels, time]
        
        Returns:
            Correlation matrix [channels, channels]
        """
        batch_size, channels, time = x.size()
        
        # Normalize
        x_norm = F.normalize(x, p=2, dim=2)  # [batch, channels, time]
        
        # Compute correlation matrix
        # Correlation = mean of element-wise products across time
        x_t = x_norm.transpose(1, 2)  # [batch, time, channels]
        correlation = torch.bmm(x_norm, x_t) / time  # [batch, channels, channels]
        
        # Average over batch
        correlation = torch.mean(correlation, dim=0)  # [channels, channels]
        
        # Make symmetric and non-negative
        correlation = (correlation + correlation.t()) / 2
        correlation = torch.abs(correlation)
        
        return correlation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Build adjacency matrix from functional correlations.
        
        Args:
            x: Input EEG [batch, channels, time]
        
        Returns:
            Adjacency matrix [channels, channels]
        """
        # Compute functional correlation
        functional_adj = self.compute_functional_correlation(x)
        
        # Combine with learnable adjacency
        adj = self.adjacency * functional_adj
        
        # Normalize adjacency matrix (symmetric normalization)
        rowsum = adj.sum(dim=1)
        d_inv_sqrt = torch.pow(rowsum + 1e-6, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        adj_normalized = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        
        return adj_normalized


class STGCNBranch(nn.Module):
    """ST-GCN Branch: Spatio-Temporal Graph Convolution Network."""
    
    def __init__(self, num_channels: int = 64, time_steps: int = 512,
                 num_classes: int = 2, dropout_rate: float = 0.3,
                 gcn_hidden: int = 64):
        super(STGCNBranch, self).__init__()
        
        self.num_channels = num_channels
        self.time_steps = time_steps
        self.gcn_hidden = gcn_hidden
        
        # Graph Construction
        self.graph_construction = GraphConstruction(num_channels, correlation_type='plv')
        
        # Graph Convolution Layers
        self.gcn1 = GraphConvolution(1, gcn_hidden)  # Input: single time point per channel
        self.gcn2 = GraphConvolution(gcn_hidden, gcn_hidden)
        # BatchNorm normalizes over the feature dimension (gcn_hidden), not channels
        # After permute: [batch*time, gcn_hidden, channels], so BatchNorm1d(gcn_hidden)
        self.gcn_bn1 = nn.BatchNorm1d(gcn_hidden)
        self.gcn_bn2 = nn.BatchNorm1d(gcn_hidden)
        
        # Temporal pooling to reduce time dimension
        self.temporal_pool = nn.AdaptiveAvgPool1d(32)  # Reduce to 32 time steps
        
        # Temporal Attention (after pooling)
        reduced_time = 32
        self.temporal_attention = TemporalAttentionModule(reduced_time, reduction=8)
        
        # Calculate output size
        self._calculate_output_size()
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.output_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
    
    def _calculate_output_size(self):
        """Calculate output feature size."""
        dummy_input = torch.randn(1, self.num_channels, self.time_steps)
        
        with torch.no_grad():
            # Build graph
            adj = self.graph_construction(dummy_input)
            
            # Process through GCN
            x_gcn = dummy_input.permute(0, 2, 1).unsqueeze(-1)  # [batch, time, channels, 1]
            batch_size, time, channels, features = x_gcn.size()
            x_gcn = x_gcn.reshape(batch_size * time, channels, features)
            
            x_gcn = F.relu(self.gcn1(x_gcn, adj))
            x_gcn = self.gcn_bn1(x_gcn.permute(0, 2, 1)).permute(0, 2, 1)
            x_gcn = F.relu(self.gcn2(x_gcn, adj))
            x_gcn = self.gcn_bn2(x_gcn.permute(0, 2, 1)).permute(0, 2, 1)
            
            # Reshape back and pool
            x_gcn = x_gcn.reshape(batch_size, time, channels, self.gcn_hidden)
            x_gcn = x_gcn.permute(0, 2, 3, 1)  # [batch, channels, gcn_hidden, time]
            x_gcn = x_gcn.reshape(batch_size * channels, self.gcn_hidden, time)
            x_gcn = self.temporal_pool(x_gcn)  # [batch*channels, gcn_hidden, 32]
            x_gcn = x_gcn.reshape(batch_size, channels, self.gcn_hidden, 32)
            x_gcn = x_gcn.permute(0, 2, 1, 3)  # [batch, gcn_hidden, channels, 32]
            x_gcn = x_gcn.reshape(batch_size, self.gcn_hidden * channels, 32)
            
            # Temporal attention
            x_gcn = self.temporal_attention(x_gcn)
            
            self.output_size = x_gcn.numel() // x_gcn.size(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ST-GCN branch (memory-optimized).
        
        Args:
            x: Input EEG [batch, channels, time]
        
        Returns:
            Class logits [batch, num_classes]
        """
        batch_size, channels, time = x.size()
        
        # Build graph from functional correlations (no gradients needed for adj)
        with torch.cuda.amp.autocast(enabled=False):  # Disable mixed precision for graph construction
            adj = self.graph_construction(x)  # [channels, channels]
        
        # Process time steps in smaller chunks to reduce memory usage
        # Reduced chunk size from 128 to 32 for better memory efficiency
        chunk_size = min(32, time)  # Process 32 time steps at a time
        num_chunks = (time + chunk_size - 1) // chunk_size
        
        chunk_outputs = []
        for chunk_idx in range(num_chunks):
            start_t = chunk_idx * chunk_size
            end_t = min((chunk_idx + 1) * chunk_size, time)
            chunk_time = end_t - start_t
            
            # Extract chunk: [batch, channels, chunk_time]
            x_chunk = x[:, :, start_t:end_t]
            
            # Graph Convolution: process each time step in chunk
            # Reshape: [batch, channels, chunk_time] -> [batch, chunk_time, channels, 1]
            x_gcn = x_chunk.permute(0, 2, 1).unsqueeze(-1)  # [batch, chunk_time, channels, 1]
            x_gcn = x_gcn.reshape(batch_size * chunk_time, channels, 1)  # [batch*chunk_time, channels, 1]
            
            # Apply GCN layers
            x_gcn = F.relu(self.gcn1(x_gcn, adj))  # [batch*chunk_time, channels, gcn_hidden]
            x_gcn = self.gcn_bn1(x_gcn.permute(0, 2, 1)).permute(0, 2, 1)
            x_gcn = F.relu(self.gcn2(x_gcn, adj))  # [batch*chunk_time, channels, gcn_hidden]
            x_gcn = self.gcn_bn2(x_gcn.permute(0, 2, 1)).permute(0, 2, 1)
            
            # Reshape back: [batch*chunk_time, channels, gcn_hidden] -> [batch, channels, gcn_hidden, chunk_time]
            x_gcn = x_gcn.reshape(batch_size, chunk_time, channels, self.gcn_hidden)
            x_gcn = x_gcn.permute(0, 2, 3, 1)  # [batch, channels, gcn_hidden, chunk_time]
            
            chunk_outputs.append(x_gcn)
            
            # Clear intermediate variables to free memory
            del x_chunk
            if torch.cuda.is_available() and chunk_idx % 4 == 0:
                torch.cuda.empty_cache()
        
        # Concatenate chunks along time dimension
        x_gcn = torch.cat(chunk_outputs, dim=3)  # [batch, channels, gcn_hidden, time]
        del chunk_outputs  # Free memory
        
        # Temporal pooling: reduce time dimension
        x_gcn = x_gcn.reshape(batch_size * channels, self.gcn_hidden, time)
        x_gcn = self.temporal_pool(x_gcn)  # [batch*channels, gcn_hidden, 32]
        x_gcn = x_gcn.reshape(batch_size, channels, self.gcn_hidden, 32)
        
        # Reshape for temporal attention: [batch, gcn_hidden*channels, 32]
        x_gcn = x_gcn.permute(0, 2, 1, 3)  # [batch, gcn_hidden, channels, 32]
        x_gcn = x_gcn.reshape(batch_size, self.gcn_hidden * channels, 32)
        
        # Temporal Attention
        x_gcn = self.temporal_attention(x_gcn)  # [batch, gcn_hidden*channels, 32]
        
        # Flatten
        x_gcn = x_gcn.view(x_gcn.size(0), -1)  # [batch, features]
        
        # Classification
        output = self.classifier(x_gcn)  # [batch, num_classes]
        
        return output


# ============================================================================
# Soft-Voting Fusion Layer
# ============================================================================

class SoftVotingFusion(nn.Module):
    """Soft-Voting Fusion Layer - combines outputs from both branches."""
    
    def __init__(self, num_classes: int = 2):
        super(SoftVotingFusion, self).__init__()
        self.num_classes = num_classes
    
    def forward(self, statnet_logits: torch.Tensor, stgcn_logits: torch.Tensor) -> torch.Tensor:
        """
        Combine outputs from both branches using soft voting.
        
        Args:
            statnet_logits: Logits from STAtNet branch [batch, num_classes]
            stgcn_logits: Logits from ST-GCN branch [batch, num_classes]
        
        Returns:
            Averaged class probabilities [batch, num_classes]
        """
        # Convert logits to probabilities
        statnet_probs = F.softmax(statnet_logits, dim=1)
        stgcn_probs = F.softmax(stgcn_logits, dim=1)
        
        # Average probabilities (soft voting)
        fused_probs = (statnet_probs + stgcn_probs) / 2.0
        
        # Convert back to logits for loss computation
        fused_logits = torch.log(fused_probs + 1e-8)
        
        return fused_logits


# ============================================================================
# Complete STANETCNN Model
# ============================================================================

class STANETCNNModel(nn.Module):
    """Complete STANETCNN Model with dual-branch architecture and soft-voting fusion."""
    
    def __init__(self, num_channels: int = 64, time_steps: int = 512,
                 num_classes: int = 2, dropout_rate: float = 0.3,
                 gcn_hidden: int = 64):
        super(STANETCNNModel, self).__init__()
        
        # STAtNet Branch
        self.statnet_branch = STAtNetBranch(
            num_channels=num_channels,
            time_steps=time_steps,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
        
        # ST-GCN Branch
        self.stgcn_branch = STGCNBranch(
            num_channels=num_channels,
            time_steps=time_steps,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            gcn_hidden=gcn_hidden
        )
        
        # Soft-Voting Fusion
        self.fusion = SoftVotingFusion(num_classes=num_classes)
        
        self._initialize_weights()
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"STANETCNN Model created with {n_params:,} parameters")
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, GraphConvolution):
                if hasattr(m, 'weight'):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through STANETCNN.
        
        Args:
            x: Input EEG [batch, channels, time]
        
        Returns:
            Fused class logits [batch, num_classes]
        """
        # STAtNet Branch
        statnet_logits = self.statnet_branch(x)  # [batch, num_classes]
        
        # ST-GCN Branch
        stgcn_logits = self.stgcn_branch(x)  # [batch, num_classes]
        
        # Soft-Voting Fusion
        fused_logits = self.fusion(statnet_logits, stgcn_logits)  # [batch, num_classes]
        
        return fused_logits


# ============================================================================
# Dataset Classes
# ============================================================================

class STANETCNNDataset(Dataset):
    """PyTorch Dataset for STANETCNN - handles raw EEG (channels x time)."""
    
    def __init__(self, eeg_data: np.ndarray, labels: np.ndarray,
                 window_size: int = 512, overlap: float = 0.5):
        """
        Initialize dataset.
        
        Args:
            eeg_data: EEG data [samples, channels]
            labels: Labels [samples] or [trials]
            window_size: Window size in samples
            overlap: Window overlap fraction
        """
        self.eeg_data = eeg_data
        self.labels = labels
        self.window_size = window_size
        self.overlap = overlap
        
        # Create windows
        self.window_indices = self._create_windows()
        
        print(f"\nSTANETCNNDataset initialized:")
        print(f"  Total windows: {len(self.window_indices)}")
        print(f"  Window size: {self.window_size} samples")
        print(f"  EEG shape: {self.eeg_data.shape}")
    
    def _create_windows(self) -> List[Tuple[int, int, int]]:
        """Create sliding windows from EEG data."""
        window_indices = []
        step_size = int(self.window_size * (1 - self.overlap))
        
        for start_idx in range(0, len(self.eeg_data) - self.window_size + 1, step_size):
            end_idx = start_idx + self.window_size
            # Use label at middle of window
            mid_idx = start_idx + self.window_size // 2
            if mid_idx < len(self.labels):
                label = self.labels[mid_idx]
            else:
                label = self.labels[-1]
            window_indices.append((start_idx, end_idx, label))
        
        return window_indices
    
    def __len__(self):
        return len(self.window_indices)
    
    def __getitem__(self, idx):
        start_idx, end_idx, label = self.window_indices[idx]
        
        # Extract window
        eeg_window = self.eeg_data[start_idx:end_idx]  # [window_size, channels]
        
        # Preprocess: baseline correction and normalization
        eeg_window = eeg_window - np.mean(eeg_window, axis=0, keepdims=True)
        std_vals = np.std(eeg_window, axis=0, keepdims=True)
        std_vals = np.where(std_vals == 0, 1.0, std_vals)
        eeg_window = eeg_window / std_vals
        
        # Transpose to [channels, time] for model input
        eeg_window = eeg_window.T  # [channels, window_size]
        
        # Convert to tensors
        eeg_tensor = torch.FloatTensor(eeg_window)
        label_tensor = torch.LongTensor([label])
        
        return eeg_tensor, label_tensor


# ============================================================================
# Trainer Class
# ============================================================================

class STANETCNNTrainer:
    """Trainer for STANETCNN model."""
    
    def __init__(self, model: STANETCNNModel, device: torch.device, 
                 output_dir: str = "stanetcnn_results"):
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.best_val_acc = 0.0
        self.best_model_path = self.output_dir / "best_model.pth"
    
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer,
                    criterion: nn.Module, use_augmentation: bool = True,
                    gradient_accumulation_steps: int = 1) -> Tuple[float, float]:
        """Train for one epoch with gradient accumulation to reduce memory usage."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Clear cache at start
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            target = target.squeeze()
            
            # Data augmentation
            if use_augmentation and self.model.training:
                data = self._apply_augmentation(data)
            
            # Forward pass
            output = self.model(data)
            loss = criterion(output, target)
            
            # Scale loss by accumulation steps
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # Store predictions before clearing
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            total_loss += loss.item() * gradient_accumulation_steps  # Unscale for logging
            
            # Clear output to free memory
            del output, loss, pred
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)  # More memory efficient
                
                # Clear cache more frequently
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Clear cache every few batches
            if batch_idx % 5 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Handle remaining gradients
        if len(train_loader) % gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if total == 0:
            return float('inf'), 0.0
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def _apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation."""
        # Random noise injection
        if np.random.rand() < 0.5:
            noise_scale = 0.01
            noise = torch.randn_like(x) * noise_scale
            x = x + noise
        
        # Random channel dropout
        if np.random.rand() < 0.3:
            dropout_prob = 0.1
            mask = torch.bernoulli(torch.ones(x.shape[0], x.shape[1], 1, device=x.device) * (1 - dropout_prob))
            x = x * mask / (1 - dropout_prob)
        
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
               num_epochs: int = 50, learning_rate: float = 1e-3,
               weight_decay: float = 1e-5, patience: int = 10,
               gradient_accumulation_steps: int = 1):
        """Train the model with class balancing for imbalanced datasets."""
        # Calculate class weights for imbalanced data
        print("Calculating class weights for training data...")
        train_labels = []
        for _, (_, target) in enumerate(train_loader):
            target_np = target.squeeze().cpu().numpy()
            # Handle both scalar (0-d array) and array cases
            if target_np.ndim == 0:
                train_labels.append(int(target_np))
            else:
                train_labels.extend(target_np.tolist())
        
        # Use np.unique to get unique classes and their counts
        unique_classes, class_counts = np.unique(train_labels, return_counts=True)
        
        # Calculate class weights with safety checks
        total_samples = len(train_labels)
        n_classes = len(unique_classes)
        
        if n_classes == 0:
            print("WARNING: No classes found in training data")
            class_weights = torch.ones(2).to(self.device)  # Default to equal weights
        else:
            # Calculate weights: total_samples / (n_classes * class_count)
            class_weights = np.zeros(max(unique_classes) + 1)  # Handle sparse class indices
            for i, class_id in enumerate(unique_classes):
                if class_counts[i] > 0:  # Avoid division by zero
                    class_weights[class_id] = total_samples / (n_classes * class_counts[i])
                else:
                    class_weights[class_id] = 1.0  # Default weight for empty classes
            
            class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        print(f"  Unique classes: {unique_classes}")
        print(f"  Class counts: {class_counts}")
        print(f"  Class weights: {class_weights.cpu().numpy()}")
        
        # Check for severe imbalance
        if len(class_counts) == 2:
            ratio = max(class_counts) / min(class_counts)
            if ratio > 10:
                print(f"  WARNING: Severe class imbalance detected (ratio: {ratio:.1f}:1)")
                print(f"  Using weighted loss to handle imbalance")
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = OneCycleLR(optimizer, max_lr=learning_rate * 5,
                              total_steps=num_epochs * len(train_loader), pct_start=0.3)
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion, 
                                                     gradient_accumulation_steps=gradient_accumulation_steps)
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


# ============================================================================
# Dataset Loading Functions
# ============================================================================

def load_combined_dataset(das_data_dir: str = "das_combined_preprocessed",
                          das_preprocessing_type: str = "COMBINED_DAS",
                          fulsang_raw_dir: str = None,
                          fulsang_audio_dir: str = None,
                          fulsang_mwf_dir: str = "/home/py9363/telluride_decoding/MWF_cleaned_Fuglsang",
                          window_size: int = 512,
                          overlap: float = 0.5,
                          filter_invalid_labels: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load combined Das and Fulsang dataset.
    
    Note: CombinedDataset loads data in memory and combines it. The combined data
    is stored in memory as numpy arrays (self.eeg_data and self.labels) and is not
    saved to disk. MWF-processed Fulsang data is stored in fulsang_mwf_dir if MWF
    processing is applied.
    
    Args:
        das_data_dir: Directory containing DAS data (default: das_combined_preprocessed from das_preprocessing_combined.py)
        das_preprocessing_type: COMBINED_DAS (default), MWF, or DASPREPROCESS
        fulsang_raw_dir: Directory containing Fulsang raw EEG data (optional)
        fulsang_audio_dir: Directory containing Fulsang audio data (optional)
        fulsang_mwf_dir: Directory containing Fulsang MWF-processed data (default: /home/py9363/telluride_decoding/MWF_cleaned_Fuglsang)
        window_size: Window size in samples
        overlap: Window overlap fraction
        filter_invalid_labels: If True, filter out windows with invalid labels (default: True)
    
    Returns:
        Tuple of (eeg_data, labels) as numpy arrays
    """
    if CombinedDataset is None:
        raise ImportError("CombinedDataset not available")
    
    # Check if MWF-processed data already exists - if so, use it directly
    mwf_dir = Path(fulsang_mwf_dir)
    existing_mwf_files = list(mwf_dir.glob("sub*_MWF.mat")) if mwf_dir.exists() else []
    
    if existing_mwf_files:
        print(f"Found {len(existing_mwf_files)} existing MWF-processed Fulsang files in {fulsang_mwf_dir}")
        print("  Using existing MWF files directly (skipping raw data loading)")
        print("  Note: Some Fulsang trials may have missing attention labels (warnings are expected)")
        # When MWF files exist, CombinedDataset will use them directly
        # We still need to provide a valid directory path, but CombinedDataset will prioritize MWF files
        # Use the MWF directory itself as the raw_dir (it exists and has MWF files)
        if fulsang_raw_dir is None:
            fulsang_raw_dir = fulsang_mwf_dir
    elif fulsang_raw_dir is None:
        # Try common default paths
        default_paths = [
            "/home/py9363/telluride_decoding/Data/Fulsang/EEG",
            "Data/Fulsang/EEG",
            "fulsang_raw",
        ]
        fulsang_raw_dir = None
        for path in default_paths:
            if Path(path).exists():
                fulsang_raw_dir = path
                print(f"Using default Fulsang raw directory: {fulsang_raw_dir}")
                break
        
        if fulsang_raw_dir is None:
            raise ValueError(
                "fulsang_raw_dir is required for CombinedDataset. "
                "Please provide --fulsang_raw_dir argument, or ensure MWF-processed data exists in "
                f"{fulsang_mwf_dir}, or place Fulsang raw data in one of these locations: "
                f"{', '.join(default_paths)}"
            )
    
    print("Loading Combined Dataset...")
    print("  Note: Trials without extractable attention labels will be skipped (not inferred)")
    
    combined_dataset = CombinedDataset(
        das_data_dir=das_data_dir,
        das_preprocessing_type=das_preprocessing_type,
        fulsang_raw_dir=fulsang_raw_dir,
        fulsang_audio_dir=fulsang_audio_dir,
        fulsang_mwf_output_dir=fulsang_mwf_dir,
        window_size=window_size,
        overlap=overlap
    )
    
    # Get window indices and metadata
    window_indices = combined_dataset.get_window_indices()
    metadata = combined_dataset.metadata if hasattr(combined_dataset, 'metadata') else []
    
    # Extract windows and labels, filtering out inferred labels
    eeg_windows = []
    labels = []
    valid_indices = []
    filtered_count = 0
    
    # Build a map of trial indices to label sources
    # For Fulsang trials, check if label was inferred
    trial_to_label_source = {}
    if metadata:
        for meta in metadata:
            if meta.get('dataset') == 'Fulsang':
                trial_key = (meta.get('subject_id'), meta.get('trial_idx'))
                label_source = meta.get('label_source', 'extracted')
                trial_to_label_source[trial_key] = label_source
    
    # Map window indices to trials to check label sources
    # We need to determine which trial each window belongs to
    trial_boundaries = combined_dataset.trial_boundaries if hasattr(combined_dataset, 'trial_boundaries') else []
    
    for idx, (start_idx, end_idx, label) in enumerate(window_indices):
        # Validate label (should be 0 or 1)
        if label not in [0, 1]:
            if filter_invalid_labels:
                filtered_count += 1
                continue  # Skip invalid labels
            else:
                # Use default label if filtering is disabled
                label = 0
        
        # Check if this window belongs to a trial with inferred label
        if filter_invalid_labels and trial_boundaries and trial_to_label_source:
            # Find which trial this window belongs to
            mid_idx = start_idx + (end_idx - start_idx) // 2
            trial_idx = None
            for i, (trial_start, trial_end) in enumerate(trial_boundaries):
                if trial_start <= mid_idx < trial_end:
                    trial_idx = i
                    break
            
            if trial_idx is not None and trial_idx < len(metadata):
                meta = metadata[trial_idx]
                if meta.get('dataset') == 'Fulsang':
                    trial_key = (meta.get('subject_id'), meta.get('trial_idx'))
                    label_source = trial_to_label_source.get(trial_key, 'extracted')
                    if label_source == 'fallback':
                        # Skip trials with inferred labels
                        filtered_count += 1
                        continue
        
        eeg_window = combined_dataset.eeg_data[start_idx:end_idx]
        
        # Validate window size
        if eeg_window.shape[0] < window_size:
            if filter_invalid_labels:
                filtered_count += 1
                continue  # Skip windows that are too small
            else:
                # Pad if needed
                padding = window_size - eeg_window.shape[0]
                eeg_window = np.vstack([eeg_window, np.zeros((padding, eeg_window.shape[1]))])
        
        eeg_windows.append(eeg_window)
        labels.append(label)
        valid_indices.append(idx)
    
    if not eeg_windows:
        raise ValueError("No valid windows found after filtering")
    
    # Stack windows
    eeg_data = np.vstack(eeg_windows)
    labels = np.array(labels)
    
    # Report statistics
    print(f"\n✓ Combined dataset loaded:")
    print(f"  Total valid windows: {len(labels)}")
    print(f"  EEG shape: {eeg_data.shape}")
    print(f"  Label distribution: {np.bincount(labels)}")
    if filter_invalid_labels:
        filtered_total = len(window_indices) - len(valid_indices)
        if filtered_total > 0:
            print(f"  Filtered out {filtered_total} windows (invalid labels or inferred labels)")
            if filtered_count > 0:
                print(f"    - {filtered_count} windows from trials with inferred/missing labels")
    
    return eeg_data, labels


def load_das_dataset(tfrecord_dir: str, window_size: int = 512,
                     overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Load Das dataset from TFRecord files."""
    if tf is None:
        raise ImportError("TensorFlow not available")
    
    print("Loading Das Dataset...")
    tfrecord_dir = Path(tfrecord_dir)
    
    # Find TFRecord files
    train_dir = tfrecord_dir / "train"
    test_dir = tfrecord_dir / "test"
    
    if train_dir.exists() and test_dir.exists():
        tfrecord_files = list(train_dir.glob("*.tfrecords")) + list(test_dir.glob("*.tfrecords"))
    else:
        tfrecord_files = list(tfrecord_dir.glob("*.tfrecords"))
    
    if not tfrecord_files:
        raise ValueError(f"No TFRecord files found in {tfrecord_dir}")
    
    all_eeg = []
    all_labels = []
    n_channels = 64
    
    for tfrecord_file in tqdm(tfrecord_files, desc="Loading TFRecord files"):
        try:
            dataset = tf.data.TFRecordDataset(str(tfrecord_file))
            
            current_trial_eeg = []
            current_trial_label = None
            
            for raw_record in dataset:
                try:
                    example = tf.train.Example()
                    example.ParseFromString(raw_record.numpy())
                    features = example.features.feature
                    
                    if 'eeg' not in features or 'attended_ear' not in features:
                        continue
                    
                    eeg_bytes = features['eeg'].float_list.value
                    if len(eeg_bytes) != n_channels:
                        continue
                    
                    eeg_sample = np.array(eeg_bytes, dtype=np.float32).reshape(1, n_channels)
                    
                    attended_ear = features['attended_ear'].bytes_list.value[0].decode('utf-8')
                    label = 0 if attended_ear.upper() == 'L' else 1
                    
                    if current_trial_label is None:
                        current_trial_label = label
                        current_trial_eeg = [eeg_sample]
                    elif current_trial_label == label:
                        current_trial_eeg.append(eeg_sample)
                    else:
                        if len(current_trial_eeg) > 0:
                            trial_eeg = np.vstack(current_trial_eeg)
                            all_eeg.append(trial_eeg)
                            all_labels.append(current_trial_label)
                        current_trial_label = label
                        current_trial_eeg = [eeg_sample]
                
                except Exception as e:
                    continue
            
            if len(current_trial_eeg) > 0:
                trial_eeg = np.vstack(current_trial_eeg)
                all_eeg.append(trial_eeg)
                all_labels.append(current_trial_label)
        
        except Exception as e:
            print(f"Error loading {tfrecord_file}: {e}")
            continue
    
    if not all_eeg:
        raise ValueError("No valid Das data loaded")
    
    eeg_data = np.vstack(all_eeg)
    labels = np.array(all_labels)
    
    # Expand labels to match samples
    expanded_labels = []
    for trial_eeg, label in zip(all_eeg, all_labels):
        expanded_labels.extend([label] * trial_eeg.shape[0])
    labels = np.array(expanded_labels)
    
    return eeg_data, labels


def load_fulsang_dataset(tfrecord_dir: str, window_size: int = 512,
                        overlap: float = 0.5,
                        filter_invalid_labels: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Fulsang dataset from TFRecord files with improved label validation.
    
    Args:
        tfrecord_dir: Directory containing TFRecord files
        window_size: Window size in samples
        overlap: Window overlap fraction
        filter_invalid_labels: If True, filter out trials with invalid/missing labels (default: True)
    
    Returns:
        Tuple of (eeg_data, labels) as numpy arrays
    """
    if tf is None:
        raise ImportError("TensorFlow not available")
    
    print("Loading Fulsang Dataset...")
    print(f"  TFRecord directory: {tfrecord_dir}")
    tfrecord_dir = Path(tfrecord_dir)
    
    # Find TFRecord files
    train_dir = tfrecord_dir / "train"
    test_dir = tfrecord_dir / "test"
    
    if train_dir.exists() and test_dir.exists():
        # Has train/test subdirectories
        tfrecord_files = list(train_dir.glob("*.tfrecords")) + list(test_dir.glob("*.tfrecords"))
        print(f"  Found train/test subdirectories")
    else:
        # Files directly in the directory
        tfrecord_files = list(tfrecord_dir.glob("*.tfrecords"))
        print(f"  Looking for files directly in directory")
    
    if not tfrecord_files:
        raise ValueError(f"No TFRecord files found in {tfrecord_dir}")
    
    all_eeg = []
    all_labels = []
    n_channels = 66
    trials_with_missing_labels = 0
    trials_with_invalid_labels = 0
    total_trials = 0
    
    for tfrecord_file in tqdm(tfrecord_files, desc="Loading TFRecord files"):
        try:
            dataset = tf.data.TFRecordDataset(str(tfrecord_file))
            
            current_trial_eeg = []
            current_trial_label = None
            current_trial_has_valid_label = False
            
            for raw_record in dataset:
                try:
                    example = tf.train.Example()
                    example.ParseFromString(raw_record.numpy())
                    features = example.features.feature
                    
                    if 'eeg' not in features:
                        continue
                    
                    eeg_bytes = features['eeg'].float_list.value
                    if len(eeg_bytes) != n_channels:
                        continue
                    
                    eeg_sample = np.array(eeg_bytes, dtype=np.float32).reshape(1, n_channels)
                    
                    # Try to get attention label
                    label = None
                    if 'attention_label' in features:
                        label_values = features['attention_label'].int64_list.value
                        if label_values:
                            label = int(label_values[0])
                            # Validate label (should be 0 or 1)
                            if label not in [0, 1]:
                                if filter_invalid_labels:
                                    label = None  # Mark as invalid
                                else:
                                    label = 0  # Use default
                    elif 'attended_ear' in features:
                        # Fallback to attended_ear format
                        try:
                            attended_ear = features['attended_ear'].bytes_list.value[0].decode('utf-8')
                            label = 0 if attended_ear.upper() == 'L' else 1
                        except:
                            label = None
                    
                    if label is None:
                        # Missing label
                        if current_trial_label is not None:
                            # End current trial and start new one
                            if len(current_trial_eeg) > 0 and current_trial_has_valid_label:
                                trial_eeg = np.vstack(current_trial_eeg)
                                all_eeg.append(trial_eeg)
                                all_labels.append(current_trial_label)
                            elif len(current_trial_eeg) > 0:
                                trials_with_missing_labels += 1
                            current_trial_eeg = []
                            current_trial_label = None
                            current_trial_has_valid_label = False
                        continue
                    
                    # Valid label found
                    if current_trial_label is None:
                        current_trial_label = label
                        current_trial_eeg = [eeg_sample]
                        current_trial_has_valid_label = True
                        total_trials += 1
                    elif current_trial_label == label:
                        current_trial_eeg.append(eeg_sample)
                    else:
                        # Label changed - end current trial
                        if len(current_trial_eeg) > 0 and current_trial_has_valid_label:
                            trial_eeg = np.vstack(current_trial_eeg)
                            all_eeg.append(trial_eeg)
                            all_labels.append(current_trial_label)
                        elif len(current_trial_eeg) > 0:
                            trials_with_missing_labels += 1
                        
                        # Start new trial
                        current_trial_label = label
                        current_trial_eeg = [eeg_sample]
                        current_trial_has_valid_label = True
                        total_trials += 1
                
                except Exception as e:
                    continue
            
            # Handle last trial
            if len(current_trial_eeg) > 0:
                if current_trial_has_valid_label:
                    trial_eeg = np.vstack(current_trial_eeg)
                    all_eeg.append(trial_eeg)
                    all_labels.append(current_trial_label)
                else:
                    trials_with_missing_labels += 1
        
        except Exception as e:
            print(f"Error loading {tfrecord_file}: {e}")
            continue
    
    if not all_eeg:
        raise ValueError("No valid Fulsang data loaded")
    
    # Validate labels
    labels_array = np.array(all_labels)
    invalid_mask = ~np.isin(labels_array, [0, 1])
    if np.any(invalid_mask):
        trials_with_invalid_labels = np.sum(invalid_mask)
        if filter_invalid_labels:
            # Filter out invalid labels
            valid_mask = ~invalid_mask
            all_eeg = [trial for i, trial in enumerate(all_eeg) if valid_mask[i]]
            all_labels = labels_array[valid_mask].tolist()
            print(f"  Filtered out {trials_with_invalid_labels} trials with invalid labels")
        else:
            # Replace invalid labels with default
            labels_array[invalid_mask] = 0
            all_labels = labels_array.tolist()
            print(f"  Replaced {trials_with_invalid_labels} invalid labels with default (0)")
    
    # Use first 64 channels (EEG only, exclude EOG)
    eeg_data = np.vstack([trial[:, :64] for trial in all_eeg])
    labels = np.array(all_labels)
    
    # Expand labels to match samples
    expanded_labels = []
    for trial_eeg, label in zip(all_eeg, all_labels):
        expanded_labels.extend([label] * trial_eeg.shape[0])
    labels = np.array(expanded_labels)
    
    # Report statistics
    print(f"\n✓ Fulsang dataset loaded:")
    print(f"  Total trials: {total_trials}")
    print(f"  Valid trials: {len(all_eeg)}")
    print(f"  Trials with missing labels: {trials_with_missing_labels}")
    print(f"  Trials with invalid labels: {trials_with_invalid_labels}")
    print(f"  EEG shape: {eeg_data.shape}")
    print(f"  Label distribution: {np.bincount(labels)}")
    
    return eeg_data, labels


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='STANETCNN - Dual-Branch Architecture for Auditory Attention Decoding')
    parser.add_argument('--dataset', type=str, default='combined',
                       choices=['combined', 'das', 'fulsang'],
                       help='Dataset to use (combined, das, or fulsang)')
    parser.add_argument('--das_data_dir', type=str, default='das_combined_preprocessed',
                       help='Directory containing DAS data (from das_preprocessing_combined.py)')
    parser.add_argument('--das_preprocessing_type', type=str, default='COMBINED_DAS',
                       choices=['COMBINED_DAS', 'MWF', 'DASPREPROCESS'],
                       help='DAS preprocessing: COMBINED_DAS (run das_preprocessing_combined.py)')
    parser.add_argument('--das_tfrecord_dir', type=str, default='das_combined_preprocessed/tfrecords',
                       help='Directory containing Das TFRecord files')
    parser.add_argument('--fulsang_tfrecord_dir', type=str, default='fulsang_preprocessed/tfrecords', nargs='?', const=None,
                       help='Directory containing Fulsang TFRecord files (default: fulsang_preprocessed/tfrecords)')
    parser.add_argument('--fulsang_raw_dir', type=str, default=None, nargs='?', const=None,
                       help='Directory containing Fulsang raw EEG data')
    parser.add_argument('--fulsang_audio_dir', type=str, default=None, nargs='?', const=None,
                       help='Directory containing Fulsang audio data')
    parser.add_argument('--fulsang_mwf_dir', type=str, default='/home/py9363/telluride_decoding/MWF_cleaned_Fuglsang',
                       help='Directory containing Fulsang MWF-processed data (default: /home/py9363/telluride_decoding/MWF_cleaned_Fuglsang)')
    parser.add_argument('--window_size', type=int, default=512,
                       help='Window size in samples (default: 512)')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Window overlap fraction (default: 0.5)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default: 8, reduced for memory efficiency with dual-branch architecture)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help='Gradient accumulation steps (default: 2, effective batch size = batch_size * steps)')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout rate (default: 0.3)')
    parser.add_argument('--gcn_hidden', type=int, default=32,
                       help='GCN hidden dimension (default: 32, reduced from 64 for memory efficiency)')
    parser.add_argument('--output_dir', type=str, default='stanetcnn_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Convert empty strings to None for optional arguments
    if args.fulsang_raw_dir == '':
        args.fulsang_raw_dir = None
    if args.fulsang_audio_dir == '':
        args.fulsang_audio_dir = None
    if args.fulsang_tfrecord_dir == '':
        args.fulsang_tfrecord_dir = None
    
    print("="*80)
    print("STANETCNN - Dual-Branch Architecture")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Window size: {args.window_size} samples")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.num_epochs}")
    
    # Load dataset
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)
    
    if args.dataset == 'combined':
        print("Note: Warnings about missing attention labels for Fulsang trials are expected")
        print("      and will be handled automatically (invalid labels will be filtered).")
        eeg_data, labels = load_combined_dataset(
            das_data_dir=args.das_data_dir,
            das_preprocessing_type=args.das_preprocessing_type,
            fulsang_raw_dir=args.fulsang_raw_dir,
            fulsang_audio_dir=args.fulsang_audio_dir,
            fulsang_mwf_dir=args.fulsang_mwf_dir,
            window_size=args.window_size,
            overlap=args.overlap,
            filter_invalid_labels=True
        )
        num_channels = 64
    elif args.dataset == 'das':
        eeg_data, labels = load_das_dataset(
            tfrecord_dir=args.das_tfrecord_dir,
            window_size=args.window_size,
            overlap=args.overlap
        )
        num_channels = 64
    elif args.dataset == 'fulsang':
        # Use default path if not provided
        fulsang_tfrecord_dir = args.fulsang_tfrecord_dir if args.fulsang_tfrecord_dir else 'fulsang_preprocessed/tfrecords'
        eeg_data, labels = load_fulsang_dataset(
            tfrecord_dir=fulsang_tfrecord_dir,
            window_size=args.window_size,
            overlap=args.overlap,
            filter_invalid_labels=True
        )
        num_channels = 64
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    print(f"  EEG shape: {eeg_data.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Label distribution: {np.bincount(labels)}")
    
    # Create dataset
    dataset = STANETCNNDataset(
        eeg_data=eeg_data,
        labels=labels,
        window_size=args.window_size,
        overlap=args.overlap
    )
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # Create data loaders (reduce num_workers to save memory)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps} (batch_size={args.batch_size} × accumulation={args.gradient_accumulation_steps})")
    
    # Create model
    print("\n" + "="*80)
    print("INITIALIZING STANETCNN MODEL")
    print("="*80)
    model = STANETCNNModel(
        num_channels=num_channels,
        time_steps=args.window_size,
        num_classes=2,
        dropout_rate=args.dropout_rate,
        gcn_hidden=args.gcn_hidden
    )
    
    # Create trainer
    trainer = STANETCNNTrainer(
        model=model,
        device=device,
        output_dir=args.output_dir
    )
    
    # Clear GPU memory before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Train model
    print("\n" + "="*80)
    print("TRAINING STANETCNN MODEL")
    print("="*80)
    print(f"Note: Using batch_size={args.batch_size}, gradient_accumulation={args.gradient_accumulation_steps}")
    print(f"      Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # Test model
    print("\n" + "="*80)
    print("TESTING MODEL")
    print("="*80)
    test_metrics = trainer.test(test_loader)
    
    # Save results
    results_json = {
        'dataset': args.dataset,
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

