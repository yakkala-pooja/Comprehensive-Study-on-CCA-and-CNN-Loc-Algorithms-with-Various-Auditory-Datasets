#!/usr/bin/env python3

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR, CosineAnnealingWarmRestarts
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

from CombinedDataset import CombinedDataset
import scipy.io as sio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CombinedThreeDataset(CombinedDataset):
    def __init__(self,
                 das_data_dir: str = "das_combined_preprocessed",
                 das_preprocessing_type: str = "COMBINED_DAS",
                 das_original_dir: str = "Data/Das/4004271",
                 das_audio_dir: str = "Data/Das/4004271/stimuli/stimuli",
                 fulsang_raw_dir: str = "/home/py9363/telluride_decoding/Data/Fulsang/EEG",
                 fulsang_audio_dir: str = "/home/py9363/telluride_decoding/Data/Fulsang/AUDIO",
                 fulsang_mwf_output_dir: str = "MWF_cleaned_Fuglsang",
                 kuleuven_preprocessed_dir: str = "kuleuven_255_preprocessed",
                 combined_dataset_dir: str = "combined_dataset",
                 window_size: int = 512,
                 overlap: float = 0.5,
                 target_channels: int = 64,
                 target_sampling_rate: int = 128):
        
        self.kuleuven_preprocessed_dir = Path(kuleuven_preprocessed_dir)
        
        super().__init__(
            das_data_dir=das_data_dir,
            das_preprocessing_type=das_preprocessing_type,
            das_original_dir=das_original_dir,
            das_audio_dir=das_audio_dir,
            fulsang_raw_dir=fulsang_raw_dir,
            fulsang_audio_dir=fulsang_audio_dir,
            fulsang_mwf_output_dir=fulsang_mwf_output_dir,
            combined_dataset_dir=combined_dataset_dir,
            window_size=window_size,
            overlap=overlap,
            target_channels=target_channels,
            target_sampling_rate=target_sampling_rate
        )
        
        print("\nLoading KU Leuven 255 data...")
        kuleuven_eeg, kuleuven_labels, kuleuven_metadata, kuleuven_trial_lengths, kuleuven_left_envs, kuleuven_right_envs = self._load_kuleuven_data()
        
        if kuleuven_eeg.shape[1] != self.target_channels:
            if kuleuven_eeg.shape[1] > self.target_channels:
                kuleuven_eeg = kuleuven_eeg[:, :self.target_channels]
            else:
                padding = np.zeros((kuleuven_eeg.shape[0], self.target_channels - kuleuven_eeg.shape[1]), dtype=kuleuven_eeg.dtype)
                kuleuven_eeg = np.hstack([kuleuven_eeg, padding])
        
        print("\nCombining all three datasets...")
        current_idx = len(self.eeg_data)
        self.eeg_data = np.vstack([self.eeg_data, kuleuven_eeg])
        self.labels = np.hstack([self.labels, kuleuven_labels])
        self.metadata = self.metadata + kuleuven_metadata
        
        def ensure_2d(env_list):
            result = []
            for env in env_list:
                if env is None:
                    continue
                env = np.asarray(env)
                if len(env.shape) == 1:
                    env = env.reshape(-1, 1)
                elif len(env.shape) > 2:
                    env = env.reshape(-1, 1)
                result.append(env)
            return result
        
        kuleuven_left_envs_2d = ensure_2d(kuleuven_left_envs) if kuleuven_left_envs else []
        kuleuven_right_envs_2d = ensure_2d(kuleuven_right_envs) if kuleuven_right_envs else []
        
        kuleuven_eeg_length = len(kuleuven_eeg)
        
        if self.left_envelope_stream is not None and self.right_envelope_stream is not None:
            existing_left = self.left_envelope_stream
            existing_right = self.right_envelope_stream
            
            if existing_left.ndim == 1:
                existing_left = existing_left.reshape(-1, 1)
            if existing_right.ndim == 1:
                existing_right = existing_right.reshape(-1, 1)
            
            if kuleuven_left_envs_2d and kuleuven_right_envs_2d:
                all_left_envs = [existing_left] + kuleuven_left_envs_2d
                all_right_envs = [existing_right] + kuleuven_right_envs_2d
                
                self.left_envelope_stream = np.vstack(all_left_envs).astype(np.float32)
                self.right_envelope_stream = np.vstack(all_right_envs).astype(np.float32)
            else:
                kuleuven_left_fallback = np.zeros((kuleuven_eeg_length, 1), dtype=np.float32)
                kuleuven_right_fallback = np.zeros((kuleuven_eeg_length, 1), dtype=np.float32)
                
                self.left_envelope_stream = np.vstack([existing_left, kuleuven_left_fallback]).astype(np.float32)
                self.right_envelope_stream = np.vstack([existing_right, kuleuven_right_fallback]).astype(np.float32)
            
            self._total_frames = self.left_envelope_stream.shape[0]
        else:
            if kuleuven_left_envs_2d and kuleuven_right_envs_2d:
                self.left_envelope_stream = np.vstack(kuleuven_left_envs_2d).astype(np.float32)
                self.right_envelope_stream = np.vstack(kuleuven_right_envs_2d).astype(np.float32)
            else:
                self.left_envelope_stream = np.zeros((len(self.eeg_data), 1), dtype=np.float32)
                self.right_envelope_stream = np.zeros((len(self.eeg_data), 1), dtype=np.float32)
            self._total_frames = len(self.eeg_data)
        
        if len(self.left_envelope_stream) != len(self.eeg_data):
            print(f"⚠️  WARNING: Envelope stream length ({len(self.left_envelope_stream)}) != EEG data length ({len(self.eeg_data)})")
            print(f"   Trimming/padding envelopes to match EEG data length")
            if len(self.left_envelope_stream) > len(self.eeg_data):
                self.left_envelope_stream = self.left_envelope_stream[:len(self.eeg_data)]
                self.right_envelope_stream = self.right_envelope_stream[:len(self.eeg_data)]
            else:
                padding_left = np.zeros((len(self.eeg_data) - len(self.left_envelope_stream), 1), dtype=np.float32)
                padding_right = np.zeros((len(self.eeg_data) - len(self.right_envelope_stream), 1), dtype=np.float32)
                self.left_envelope_stream = np.vstack([self.left_envelope_stream, padding_left])
                self.right_envelope_stream = np.vstack([self.right_envelope_stream, padding_right])
            self._total_frames = len(self.eeg_data)
        
        for label, trial_length in zip(kuleuven_labels, kuleuven_trial_lengths):
            self.trial_boundaries.append((current_idx, current_idx + trial_length))
            self.trial_labels.append(label)
            current_idx += trial_length
        
        print(f"\n✓ Combined three datasets loaded:")
        print(f"  Total samples: {len(self.eeg_data)}")
        print(f"  EEG shape: {self.eeg_data.shape}")
        print(f"  Channels: {self.n_channels}")
        print(f"  Sampling rate: {self.sampling_rate} Hz")
        print(f"  Label distribution: {np.bincount(self.labels)}")
        print(f"  Das trials: {len([m for m in self.metadata if m.get('dataset') == 'Das'])}")
        print(f"  Fulsang trials: {len([m for m in self.metadata if m.get('dataset') == 'Fulsang'])}")
        print(f"  KU Leuven trials: {len([m for m in self.metadata if m.get('dataset') == 'KULeuven'])}")
    
    def _load_kuleuven_data(self) -> Tuple[np.ndarray, np.ndarray, List[Dict], List[int], List, List]:
        if not self.kuleuven_preprocessed_dir.exists():
            raise ValueError(f"KU Leuven preprocessed directory does not exist: {self.kuleuven_preprocessed_dir}\n"
                           f"Please run PREPROCESS255.py first")
        
        preprocessed_files = sorted(list(self.kuleuven_preprocessed_dir.glob("S*_preprocessed.mat")))
        if not preprocessed_files:
            raise ValueError(f"No KU Leuven preprocessed files found in {self.kuleuven_preprocessed_dir}\n"
                           f"Expected files: S0_preprocessed.mat, S1_preprocessed.mat, etc.\n"
                           f"Please run PREPROCESS255.py first")
        
        all_eeg = []
        all_labels = []
        all_metadata = []
        trial_lengths = []
        all_left_env = []
        all_right_env = []
        
        for preprocessed_file in tqdm(preprocessed_files, desc="Loading KU Leuven data"):
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
                            eeg_data = trial.get('eeg_data', None)
                        else:
                            continue
                        
                        if eeg_data is None:
                            continue
                        
                        if len(eeg_data.shape) == 1:
                            eeg_data = eeg_data.reshape(-1, 1)
                        elif len(eeg_data.shape) > 2:
                            eeg_data = eeg_data.reshape(eeg_data.shape[0], -1)
                        
                        if hasattr(trial, 'attention_label'):
                            label = trial.attention_label
                        elif isinstance(trial, dict):
                            label = trial.get('attention_label', 0)
                        elif hasattr(trial, 'attended_ear'):
                            attended_ear = trial.attended_ear
                            label = 0 if str(attended_ear).upper() == 'L' else 1
                        else:
                            label = 0
                        
                        if isinstance(label, np.ndarray):
                            if label.size > 0:
                                label = int(label.item() if label.size == 1 else label.flatten()[0])
                            else:
                                label = 0
                        else:
                            label = int(label)
                        
                        label = 0 if label == 0 else 1
                        
                        if hasattr(trial, 'left_envelope'):
                            left_env = trial.left_envelope
                        elif isinstance(trial, dict):
                            left_env = trial.get('left_envelope', None)
                        else:
                            left_env = None
                        
                        if hasattr(trial, 'right_envelope'):
                            right_env = trial.right_envelope
                        elif isinstance(trial, dict):
                            right_env = trial.get('right_envelope', None)
                        else:
                            right_env = None
                        
                        if left_env is not None:
                            left_env = np.asarray(left_env)
                            if len(left_env.shape) == 1:
                                left_env = left_env.reshape(-1, 1)
                        if right_env is not None:
                            right_env = np.asarray(right_env)
                            if len(right_env.shape) == 1:
                                right_env = right_env.reshape(-1, 1)
                        
                        if left_env is None or right_env is None:
                            length = eeg_data.shape[0]
                            left_env, right_env = self._fallback_envelopes(length, label)
                        
                        if np.any(left_env != 0) and np.any(right_env != 0):
                            self._real_envelope_frames += eeg_data.shape[0]
                        
                        all_eeg.append(eeg_data)
                        all_labels.append(label)
                        trial_lengths.append(eeg_data.shape[0])
                        all_metadata.append({
                            'subject_id': subject_id,
                            'trial_idx': trial_idx,
                            'dataset': 'KULeuven',
                            'attention_label': label,
                            'preprocessing': 'PREPROCESS255'
                        })
                        all_left_env.append(left_env)
                        all_right_env.append(right_env)
            except Exception as e:
                print(f"Error loading {preprocessed_file}: {e}")
                continue
        
        if not all_eeg:
            raise ValueError("No valid KU Leuven data loaded")
        
        channel_counts = [eeg.shape[1] for eeg in all_eeg]
        max_channels = max(channel_counts)
        
        if len(set(channel_counts)) > 1:
            print(f"Warning: KU Leuven data has inconsistent channels: {set(channel_counts)}")
            print(f"Normalizing to {max_channels} channels")
            normalized_eeg = []
            for eeg in all_eeg:
                if eeg.shape[1] < max_channels:
                    padding = max_channels - eeg.shape[1]
                    pad_data = np.zeros((eeg.shape[0], padding), dtype=eeg.dtype)
                    eeg = np.hstack([eeg, pad_data])
                normalized_eeg.append(eeg)
            all_eeg = normalized_eeg
        
        eeg_data = np.vstack(all_eeg)
        labels = np.array(all_labels)
        
        return eeg_data, labels, all_metadata, trial_lengths, all_left_env, all_right_env

class SpatialTemporalAttention(nn.Module):
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
    def __init__(self, output_size: int = 1):
        super(AdaptivePooling, self).__init__()
        self.output_size = output_size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size)
    
    def forward(self, x):
        return self.adaptive_pool(x)

class CNNLOCBackbone(nn.Module):
    def __init__(self, input_channels: int = 64, input_time: int = 32, input_freq: int = 4):
        super(CNNLOCBackbone, self).__init__()
        self.input_channels = input_channels
        self.input_time = input_time
        self.input_freq = input_freq
        
        print(f"Building CNN-LOC backbone: channels={input_channels}, time={input_time}, freq={input_freq}")
        
        self.initial_features = MultiScaleFeatureExtractor(input_channels, 32)
        self.temporal_block1 = ResidualBlock(32, 32, stride=1)
        self.temporal_pool1 = nn.MaxPool2d((2, 1), (2, 1))
        self.temporal_block2 = ResidualBlock(32, 64, stride=1)
        self.temporal_pool2 = nn.MaxPool2d((2, 1), (2, 1))
        self.spatial_block1 = ResidualBlock(64, 64, stride=1)
        self.spatial_pool1 = nn.MaxPool2d((1, 2), (1, 2))
        self.spatial_block2 = ResidualBlock(64, 128, stride=1)
        self.spatial_pool2 = nn.MaxPool2d((1, 2), (1, 2))
        self.global_attention = SpatialTemporalAttention(128)
        self.adaptive_pooling = AdaptivePooling(output_size=1)
        self._calculate_output_size()
    
    def _calculate_output_size(self):
        dummy_input = torch.randn(1, self.input_channels, self.input_time, self.input_freq)
        with torch.no_grad():
            x = self.forward(dummy_input)
            self.output_size = x.numel()
    
    def forward(self, x):
        x = self.initial_features(x)
        x = self.temporal_block1(x)
        x = self.temporal_pool1(x)
        x = self.temporal_block2(x)
        x = self.temporal_pool2(x)
        x = self.spatial_block1(x)
        x = self.spatial_pool1(x)
        x = self.spatial_block2(x)
        x = self.spatial_pool2(x)
        x = self.global_attention(x)
        x = self.adaptive_pooling(x)
        x = x.view(x.size(0), -1)
        return x

class CNNLOCModel(nn.Module):
    def __init__(self, input_channels: int = 64, input_time: int = 32, input_freq: int = 4,
                 num_classes: int = 2, dropout_rate: float = 0.5):
        super(CNNLOCModel, self).__init__()
        self.backbone = CNNLOCBackbone(input_channels, input_time, input_freq)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.backbone.output_size, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(96, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(32, num_classes)
        )
        self._initialize_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model created with {n_params:,} parameters")
    
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

class CombinedThreeCNNDataset(Dataset):
    def __init__(self, combined_dataset: CombinedThreeDataset, mode: str = 'train', transform_eeg: bool = True, 
                 use_augmentation: bool = False):
        self.combined_dataset = combined_dataset
        self.mode = mode
        self.transform_eeg = transform_eeg
        self.use_augmentation = use_augmentation and (mode == 'train')
        self.window_size = combined_dataset.window_size
        self.sampling_rate = combined_dataset.sampling_rate
        self.n_channels = combined_dataset.n_channels
        self.window_indices = combined_dataset.get_window_indices()
        
        print(f"\nCombinedThreeCNNDataset initialized:")
        print(f"  Mode: {mode}")
        print(f"  Total windows: {len(self.window_indices)}")
        print(f"  Window size: {self.window_size} samples")
        print(f"  Sampling rate: {self.sampling_rate} Hz")
        print(f"  Channels: {self.n_channels}")
        print(f"  Transform EEG: {transform_eeg}")
        print(f"  Data Augmentation: {self.use_augmentation}")
    
    def _augment_eeg(self, eeg_window: np.ndarray) -> np.ndarray:
        if not self.use_augmentation:
            return eeg_window
        
        augmented = eeg_window.copy()
        
        # Gaussian noise (more aggressive for subject-level generalization)
        if np.random.rand() < 0.6:
            noise_scale = np.random.uniform(0.03, 0.08)
            noise = np.random.normal(0, noise_scale, eeg_window.shape)
            augmented = augmented + noise
        
        # Scaling (helps with amplitude variations across subjects)
        if np.random.rand() < 0.5:
            scale_factor = np.random.uniform(0.85, 1.15)
            augmented = augmented * scale_factor
        
        # Shifting (helps with baseline variations)
        if np.random.rand() < 0.4:
            shift_amount = np.random.uniform(-0.15, 0.15)
            augmented = augmented + shift_amount
        
        # Channel dropout (simulates missing channels)
        if np.random.rand() < 0.3:
            dropout_prob = np.random.uniform(0.05, 0.2)
            channel_dropout = np.random.rand(self.n_channels) > dropout_prob
            augmented = augmented * channel_dropout[np.newaxis, :]
        
        # Temporal smoothing (simulates slower/faster responses)
        if np.random.rand() < 0.3 and augmented.shape[0] > 10:
            # Apply a simple moving average with random window size
            window_size = np.random.randint(3, 7)
            if window_size % 2 == 0:
                window_size += 1
            kernel = np.ones(window_size) / window_size
            for ch in range(augmented.shape[1]):
                smoothed = np.convolve(augmented[:, ch], kernel, mode='same')
                # Blend original and smoothed
                blend_factor = np.random.uniform(0.3, 0.7)
                augmented[:, ch] = (1 - blend_factor) * augmented[:, ch] + blend_factor * smoothed
        
        return augmented
    
    def _transform_eeg(self, eeg_window: np.ndarray) -> np.ndarray:
        n_samples, n_channels = eeg_window.shape
        freq_bins = 4
        time_frames = 32
        
        if n_samples >= time_frames:
            samples_per_frame = n_samples // time_frames
            eeg_reshaped = eeg_window[:time_frames * samples_per_frame].reshape(
                time_frames, samples_per_frame, n_channels
            )
            eeg_fft = np.fft.rfft(eeg_reshaped, axis=1)
            eeg_fft = np.abs(eeg_fft)
            n_freq_bins_available = eeg_fft.shape[1]
            if n_freq_bins_available < freq_bins:
                padding = np.zeros((eeg_fft.shape[0], freq_bins - n_freq_bins_available, eeg_fft.shape[2]))
                eeg_fft = np.concatenate([eeg_fft, padding], axis=1)
            eeg_fft = eeg_fft[:, :freq_bins, :]
            eeg_tf = np.transpose(eeg_fft, (2, 0, 1))
        else:
            if n_samples == 0:
                eeg_tf = np.zeros((n_channels, time_frames, freq_bins), dtype=np.float32)
            else:
                repeat_factor = (time_frames // n_samples) + 1
                eeg_padded = np.tile(eeg_window, (repeat_factor, 1))[:time_frames]
                samples_per_frame = len(eeg_padded) // time_frames
                if samples_per_frame > 0:
                    eeg_reshaped = eeg_padded[:time_frames * samples_per_frame].reshape(
                        time_frames, samples_per_frame, n_channels
                    )
                    eeg_fft = np.fft.rfft(eeg_reshaped, axis=1)
                    eeg_fft = np.abs(eeg_fft)
                    n_freq_bins_available = eeg_fft.shape[1]
                    if n_freq_bins_available < freq_bins:
                        padding = np.zeros((eeg_fft.shape[0], freq_bins - n_freq_bins_available, eeg_fft.shape[2]))
                        eeg_fft = np.concatenate([eeg_fft, padding], axis=1)
                    eeg_fft = eeg_fft[:, :freq_bins, :]
                    eeg_tf = np.transpose(eeg_fft, (2, 0, 1))
                else:
                    eeg_tf = np.zeros((n_channels, time_frames, freq_bins), dtype=np.float32)
                    eeg_transposed = eeg_window.T
                    eeg_tf[:, :min(n_samples, time_frames), 0] = eeg_transposed[:, :min(n_samples, time_frames)]
        
        return eeg_tf.astype(np.float32)
    
    def __len__(self):
        return len(self.window_indices)
    
    def __getitem__(self, idx):
        start_idx, end_idx, label = self.window_indices[idx]
        eeg_window = self.combined_dataset.eeg_data[start_idx:end_idx]
        
        # Apply consistent preprocessing FIRST to normalize all datasets to the same scale
        # This ensures augmentation parameters work consistently across all datasets
        # Note: KU Leuven already has some preprocessing in PREPROCESS255.py,
        # but we re-apply normalization here to ensure consistency
        # 1. Baseline correction
        eeg_window = eeg_window - np.mean(eeg_window, axis=0, keepdims=True)
        
        # 2. MAD normalization (more robust than std, matches KU Leuven preprocessing)
        # This ensures all datasets use the same normalization method
        mad_vals = np.median(np.abs(eeg_window - np.median(eeg_window, axis=0)), axis=0, keepdims=True)
        mad_vals = np.where(mad_vals == 0, 1.0, mad_vals)
        eeg_window = eeg_window / mad_vals
        
        # 3. Soft clipping (matches KU Leuven preprocessing)
        # Applied to all datasets for consistency
        eeg_window = np.tanh(eeg_window * 0.5)
        
        # 4. Apply augmentation AFTER normalization (data is now in consistent [-1, 1] range)
        # This ensures augmentation parameters work the same for all datasets
        eeg_window = self._augment_eeg(eeg_window)
        
        if self.transform_eeg:
            eeg_tf = self._transform_eeg(eeg_window)
        else:
            eeg_tf = eeg_window.T[:, :, np.newaxis]
        
        eeg_tensor = torch.FloatTensor(eeg_tf)
        label_tensor = torch.LongTensor([label])
        
        return eeg_tensor, label_tensor

def split_dataset_by_subject(dataset: CombinedThreeCNNDataset, combined_dataset: CombinedThreeDataset,
                             train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[Dataset, Dataset, Dataset]:
    print("\n" + "="*80)
    print("SUBJECT-LEVEL SPLITTING")
    print("="*80)
    
    subject_windows = {}
    window_to_subject = {}
    
    trial_boundaries = combined_dataset.trial_boundaries
    metadata = combined_dataset.metadata
    
    for window_idx, (start_idx, end_idx, label) in enumerate(dataset.window_indices):
        mid_idx = start_idx + (end_idx - start_idx) // 2
        trial_idx = None
        for i, (trial_start, trial_end) in enumerate(trial_boundaries):
            if trial_start <= mid_idx < trial_end:
                trial_idx = i
                break
        
        if trial_idx is not None and trial_idx < len(metadata):
            subject_id = metadata[trial_idx].get('subject_id', 'unknown')
        else:
            subject_id = 'unknown'
        
        if subject_id not in subject_windows:
            subject_windows[subject_id] = []
        subject_windows[subject_id].append(window_idx)
        window_to_subject[window_idx] = subject_id
    
    print(f"Found {len(subject_windows)} unique subjects:")
    for subject_id, windows in sorted(subject_windows.items()):
        print(f"  {subject_id}: {len(windows)} windows")
    
    subjects = list(subject_windows.keys())
    np.random.seed(42)
    np.random.shuffle(subjects)
    
    n_subjects = len(subjects)
    n_train_subjects = int(train_ratio * n_subjects)
    n_val_subjects = int(val_ratio * n_subjects)
    
    train_subjects = subjects[:n_train_subjects]
    val_subjects = subjects[n_train_subjects:n_train_subjects + n_val_subjects]
    test_subjects = subjects[n_train_subjects + n_val_subjects:]
    
    print(f"\nSubject-wise split:")
    print(f"  Train subjects: {len(train_subjects)}")
    print(f"  Val subjects: {len(val_subjects)}")
    print(f"  Test subjects: {len(test_subjects)}")
    
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
    
    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)
    
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("CRITICAL: Data leakage detected!")
    
    print("✓ No data leakage detected - subjects properly separated")
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset

class CombinedThreeCNNTrainer:
    def __init__(self, model: CNNLOCModel, device: torch.device, output_dir: str = "combined_three_cnn_results"):
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.best_val_acc = 0.0
        self.best_model_path = self.output_dir / "best_model.pth"
    
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module) -> Tuple[float, float]:
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
              weight_decay: float = 1e-4, patience: int = 15, 
              class_weights: Optional[torch.Tensor] = None, use_focal_loss: bool = True,
              focal_gamma: float = 2.0):
        if use_focal_loss:
            if class_weights is not None:
                class_weights = class_weights.to(self.device)
                criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
                print(f"Using Focal Loss with class weights: {class_weights.cpu().numpy()}, gamma={focal_gamma}")
            else:
                criterion = FocalLoss(gamma=focal_gamma)
                print(f"Using Focal Loss (no class weights), gamma={focal_gamma}")
        else:
            if class_weights is not None:
                class_weights = class_weights.to(self.device)
                criterion = nn.CrossEntropyLoss(weight=class_weights)
                print(f"Using CrossEntropyLoss with class weights: {class_weights.cpu().numpy()}")
            else:
                criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # Use CosineAnnealingWarmRestarts for better subject-level generalization
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=learning_rate * 0.01)
        
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Combined Three Datasets (Das+Fulsang+KU Leuven 255) CNN-LOC')
    parser.add_argument('--das_data_dir', type=str, default='das_combined_preprocessed',
                       help='Directory containing Das preprocessed data')
    parser.add_argument('--das_preprocessing_type', type=str, default='COMBINED_DAS',
                       choices=['COMBINED_DAS', 'MWF', 'DASPREPROCESS'],
                       help='Type of Das preprocessing')
    parser.add_argument('--fulsang_raw_dir', type=str, default=None,
                       help='Directory containing Fulsang raw EEG data')
    parser.add_argument('--fulsang_audio_dir', type=str, default=None,
                       help='Directory containing Fulsang audio data')
    parser.add_argument('--fulsang_mwf_dir', type=str, default='/home/py9363/telluride_decoding/MWF_cleaned_Fuglsang',
                       help='Directory containing Fulsang MWF-processed data')
    parser.add_argument('--kuleuven_preprocessed_dir', type=str, default='kuleuven_255_preprocessed',
                       help='Directory containing KU Leuven 255 preprocessed data')
    parser.add_argument('--combined_dataset_dir', type=str, default='combined_dataset',
                       help='Centralized directory for all processed files')
    parser.add_argument('--window_size', type=int, default=512,
                       help='Window size in samples (default: 512 = 4s at 128Hz)')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Window overlap fraction (default: 0.5)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                       help='Learning rate (default: 5e-4)')
    parser.add_argument('--dropout_rate', type=float, default=0.6,
                       help='Dropout rate (default: 0.6)')
    parser.add_argument('--weight_decay', type=float, default=1.5e-4,
                       help='Weight decay for regularization (default: 1.5e-4)')
    parser.add_argument('--use_focal_loss', action='store_true', default=False,
                       help='Use Focal Loss instead of CrossEntropyLoss')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Gamma parameter for Focal Loss (default: 2.0)')
    parser.add_argument('--use_augmentation', action='store_true', default=False,
                       help='Use data augmentation during training')
    parser.add_argument('--output_dir', type=str, default='combined_three_cnn_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    args.use_focal_loss = True
    args.use_augmentation = True
    
    print("="*80)
    print("COMBINED THREE DATASETS CNN-LOC - Das + Fulsang + KU Leuven 255")
    print("="*80)
    print(f"Window size: {args.window_size} samples")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Dropout rate: {args.dropout_rate}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Focal Loss: {args.use_focal_loss} (gamma={args.focal_gamma})")
    print(f"Data Augmentation: {args.use_augmentation}")
    
    if args.fulsang_raw_dir is None:
        mwf_dir = Path(args.fulsang_mwf_dir)
        existing_mwf_files = list(mwf_dir.glob("sub*_MWF.mat")) if mwf_dir.exists() else []
        if existing_mwf_files:
            print(f"Found {len(existing_mwf_files)} existing MWF-processed Fulsang files")
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="fulsang_raw_dummy_")
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
    
    print("\n" + "="*80)
    print("LOADING COMBINED THREE DATASETS")
    print("="*80)
    combined_dataset = CombinedThreeDataset(
        das_data_dir=args.das_data_dir,
        das_preprocessing_type=args.das_preprocessing_type,
        das_original_dir=getattr(args, 'das_original_dir', 'Data/Das/4004271'),
        das_audio_dir=getattr(args, 'das_audio_dir', 'Data/Das/4004271/stimuli/stimuli'),
        fulsang_raw_dir=args.fulsang_raw_dir,
        fulsang_audio_dir=args.fulsang_audio_dir,
        fulsang_mwf_output_dir=args.fulsang_mwf_dir,
        kuleuven_preprocessed_dir=args.kuleuven_preprocessed_dir,
        combined_dataset_dir=getattr(args, 'combined_dataset_dir', 'combined_dataset'),
        window_size=args.window_size,
        overlap=args.overlap
    )
    
    print("\n" + "="*80)
    print("CREATING PYTORCH DATASET")
    print("="*80)
    pytorch_dataset = CombinedThreeCNNDataset(combined_dataset, transform_eeg=True, use_augmentation=args.use_augmentation)
    
    print("\nUsing SUBJECT-LEVEL splitting")
    train_dataset, val_dataset, test_dataset = split_dataset_by_subject(
        pytorch_dataset, combined_dataset, train_ratio=0.7, val_ratio=0.15
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    print("\nComputing class weights for label imbalance handling...")
    all_train_labels = []
    for idx in range(len(train_dataset)):
        _, label = train_dataset[idx]
        all_train_labels.append(label.item())
    train_labels_array = np.array(all_train_labels)
    unique_labels, label_counts = np.unique(train_labels_array, return_counts=True)
    print(f"  Train label distribution: {dict(zip(unique_labels, label_counts))}")
    
    total_samples = len(train_labels_array)
    n_classes = 2
    class_weights = torch.ones(n_classes, dtype=torch.float32)
    for label_val in unique_labels:
        label_idx = int(label_val)
        if label_idx < n_classes:
            count = label_counts[unique_labels == label_val][0]
            class_weights[label_idx] = total_samples / (n_classes * count)
    print(f"  Computed class weights: {class_weights.numpy()}")
    
    print("\n" + "="*80)
    print("INITIALIZING CNN-LOC MODEL")
    print("="*80)
    model = CNNLOCModel(
        input_channels=combined_dataset.n_channels,
        input_time=32,
        input_freq=4,
        num_classes=2,
        dropout_rate=args.dropout_rate
    )
    
    trainer = CombinedThreeCNNTrainer(
        model=model,
        device=device,
        output_dir=args.output_dir
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
        class_weights=class_weights,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma
    )
    
    print("\n" + "="*80)
    print("TESTING MODEL")
    print("="*80)
    test_metrics = trainer.test(test_loader)
    
    results_json = {
        'accuracy': float(test_metrics['accuracy']),
        'roc_auc': float(test_metrics['roc_auc']),
        'best_val_acc': float(test_metrics['best_val_acc']),
        'splitting': 'subject-level',
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

