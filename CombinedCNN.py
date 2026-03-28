#!/usr/bin/env python3
"""
CombinedCNN - Combined Das and Fulsang CNN-LOC using FULCNN Architecture

This script trains a CNN-LOC model on combined Das (MWF) and Fulsang (MWF) data
using the FULCNN architecture from FULCNN.py.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.append('.')

# Import FULCNN model and trainer
try:
    from FULCNN import FULCNNModel, FULCNNTrainer
except ImportError as e:
    print(f"Error: Could not import FULCNN: {e}")
    sys.exit(1)

# Import combined dataset
from CombinedDataset import CombinedDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class CombinedCNNDataset(Dataset):
    """
    PyTorch Dataset wrapper for CombinedDataset for CNN-LOC training.
    Converts numpy arrays to PyTorch tensors compatible with FULCNN.
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
        
        print(f"\nCombinedCNNDataset initialized:")
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
            eeg_fft = np.abs(eeg_fft[:, :freq_bins, :])
            
            # Reshape to (channels, time_frames, freq_bins)
            eeg_tf = np.transpose(eeg_fft, (2, 0, 1))
        else:
            # Pad if needed
            eeg_tf = np.zeros((n_channels, time_frames, freq_bins))
            eeg_tf[:, :n_samples, :] = eeg_window[:n_samples, :, np.newaxis]
        
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


def split_dataset(dataset: CombinedCNNDataset, train_ratio: float = 0.7, 
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


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Combined Das+Fulsang CNN-LOC using FULCNN architecture')
    parser.add_argument('--das_data_dir', type=str, default='das_16subjects_preprocessed',
                       help='Directory containing DAS data from das_preprocessing_combined.py')
    parser.add_argument('--das_preprocessing_type', type=str, default='COMBINED_DAS',
                       choices=['COMBINED_DAS', 'MWF', 'DASPREPROCESS'],
                       help='DAS preprocessing: COMBINED_DAS (run das_preprocessing_combined.py)')
    parser.add_argument('--fulsang_raw_dir', type=str, 
                       default='/home/py9363/telluride_decoding/Data/Fulsang/EEG',
                       help='Directory containing Fulsang raw EEG data')
    parser.add_argument('--fulsang_audio_dir', type=str,
                       default='/home/py9363/telluride_decoding/Data/Fulsang/AUDIO',
                       help='Directory containing Fulsang audio data')
    parser.add_argument('--fulsang_mwf_dir', type=str, default='MWF_cleaned_Fuglsang',
                       help='Output directory for Fulsang MWF processing')
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
    parser.add_argument('--output_dir', type=str, default='combined_cnn_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("COMBINED CNN - Das (MWF) + Fulsang (MWF) CNN-LOC Training")
    print("="*80)
    print(f"Using FULCNN architecture")
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
        fulsang_raw_dir=args.fulsang_raw_dir,
        fulsang_audio_dir=args.fulsang_audio_dir,
        fulsang_mwf_output_dir=args.fulsang_mwf_dir,
        window_size=args.window_size,
        overlap=args.overlap
    )
    
    # Create PyTorch dataset
    print("\n" + "="*80)
    print("CREATING PYTORCH DATASET")
    print("="*80)
    pytorch_dataset = CombinedCNNDataset(combined_dataset, transform_eeg=True)
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(pytorch_dataset)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # Create FULCNN model
    print("\n" + "="*80)
    print("INITIALIZING FULCNN MODEL")
    print("="*80)
    model = FULCNNModel(
        input_channels=combined_dataset.n_channels,
        input_time=32,  # FULCNN time frames
        input_freq=4,    # FULCNN freq bins
        num_classes=2,
        dropout_rate=args.dropout_rate
    )
    
    # Create trainer
    trainer = FULCNNTrainer(
        model=model,
        device=device,
        output_dir=args.output_dir,
        sampling_rate=combined_dataset.sampling_rate,
        window_size=args.window_size
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
    
    print(f"\n✓ Training Complete")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test ROC-AUC: {test_metrics.get('roc_auc', 0.0):.4f}")
    
    print(f"\n✓ Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
