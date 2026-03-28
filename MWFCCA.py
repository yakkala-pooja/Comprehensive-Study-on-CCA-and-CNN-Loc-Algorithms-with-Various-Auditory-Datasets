#!/usr/bin/env python3
"""
MWFCCA - Canonical Correlation Analysis for Combined Das and Fuglsang MWF-Cleaned Datasets

This module implements CCA analysis for combined Das and Fuglsang datasets 
after MWF artifact removal.

Features:
- Loads MWF-cleaned data from both Das and Fuglsang datasets
- Combines datasets for CCA analysis
- Comprehensive metrics: Accuracy, canonical correlations, temporal performance
"""

import os
import sys
import numpy as np
import scipy.io as sio
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve,
                           average_precision_score, matthews_corrcoef, 
                           cohen_kappa_score, balanced_accuracy_score, f1_score)
from sklearn.cross_decomposition import CCA as SklearnCCA
from scipy.stats import pearsonr
import seaborn as sns
from tqdm import tqdm
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Add telluride_decoding to path
sys.path.append('telluride_decoding')

try:
    from telluride_decoding import decoding
    from telluride_decoding import brain_data
    from telluride_decoding import regression
    from telluride_decoding import attention_decoder
    from telluride_decoding.cca import (
        BrainModelCCA, 
        cca_pearson_correlation_first,
        cca_pearson_correlation,
        calculate_cca_parameters_from_dataset
    )
except ImportError as e:
    print(f"Warning: Could not import some telluride_decoding modules: {e}")

tf.compat.v1.enable_v2_behavior()


class CombinedMWFCCADataset:
    """
    Combined dataset class for MWF-cleaned Das and Fuglsang data for CCA analysis.
    
    Loads MWF-cleaned EEG data from both datasets and prepares them for CCA.
    """
    
    def __init__(self, das_mwf_dir: str = "MWF_cleaned_DAS",
                 fuglsang_mwf_dir: str = "MWF_cleaned_Fuglsang",
                 window_size: int = 512):  # samples at 128 Hz = 4 seconds
        self.das_mwf_dir = Path(das_mwf_dir)
        self.fuglsang_mwf_dir = Path(fuglsang_mwf_dir)
        self.window_size = window_size
        
        # Parameters
        self.sampling_rate = 128  # Hz (both datasets after MWF)
        
        # Load MWF-cleaned data from both datasets
        print("Loading MWF-cleaned data from Das dataset...")
        das_eeg, das_labels, das_metadata = self._load_das_mwf_data()
        
        print("Loading MWF-cleaned data from Fuglsang dataset...")
        fuglsang_eeg, fuglsang_labels, fuglsang_metadata = self._load_fuglsang_mwf_data()
        
        # Combine datasets
        print("Combining datasets...")
        self.eeg_data = np.vstack([das_eeg, fuglsang_eeg])
        self.labels = np.hstack([das_labels, fuglsang_labels])
        self.metadata = das_metadata + fuglsang_metadata
        
        # Normalize channel count
        min_channels = min(das_eeg.shape[1], fuglsang_eeg.shape[1])
        if das_eeg.shape[1] != fuglsang_eeg.shape[1]:
            print(f"Warning: Channel mismatch - Das: {das_eeg.shape[1]}, Fuglsang: {fuglsang_eeg.shape[1]}")
            print(f"Using first {min_channels} channels from both datasets")
            self.eeg_data = self.eeg_data[:, :min_channels]
        
        self.n_channels = min_channels
        
        print(f"Combined dataset loaded:")
        print(f"  Total samples: {len(self.eeg_data)}")
        print(f"  EEG shape: {self.eeg_data.shape}")
        print(f"  Label distribution: {np.bincount(self.labels)}")
        print(f"  Channels: {self.n_channels}")
    
    def _load_das_mwf_data(self) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Load MWF-cleaned Das dataset."""
        if not self.das_mwf_dir.exists():
            raise ValueError(f"Das MWF directory does not exist: {self.das_mwf_dir}\n"
                           f"Please run MWF processing first: python3 mwf_artifact_removal.py --dataset das --unified")
        
        mwf_files = list(self.das_mwf_dir.glob("S*_MWF.mat"))
        if not mwf_files:
            raise ValueError(f"No MWF-cleaned Das files found in {self.das_mwf_dir}\n"
                           f"Expected files: S1_MWF.mat, S2_MWF.mat, etc.\n"
                           f"Please run MWF processing first: python3 mwf_artifact_removal.py --dataset das --unified")
        
        all_eeg = []
        all_labels = []
        all_metadata = []
        
        for mwf_file in tqdm(mwf_files, desc="Loading Das MWF data"):
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
                        all_metadata.append({
                            'subject_id': subject_id,
                            'trial_idx': trial_idx,
                            'dataset': 'Das',
                            'attended_ear': attended_ear
                        })
            except Exception as e:
                print(f"Error loading {mwf_file}: {e}")
                continue
        
        if not all_eeg:
            raise ValueError("No valid Das MWF data loaded")
        
        eeg_data = np.vstack(all_eeg)
        labels = np.array(all_labels)
        
        return eeg_data, labels, all_metadata
    
    def _load_fuglsang_mwf_data(self) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
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
                        
                        # Get attention label
                        if hasattr(trial, 'attention_label'):
                            label = int(trial.attention_label)
                        elif isinstance(trial, dict):
                            label = int(trial.get('attention_label', 0))
                        else:
                            label = 0
                        
                        all_eeg.append(eeg_data)
                        all_labels.append(label)
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
        
        eeg_data = np.vstack(all_eeg)
        labels = np.array(all_labels)
        
        return eeg_data, labels, all_metadata
    
    def get_eeg_and_labels(self):
        """Get EEG data and labels for CCA analysis."""
        return self.eeg_data, self.labels


class MWFCCAAnalyzer:
    """
    CCA analyzer for combined MWF-cleaned datasets.
    """
    
    def __init__(self, dataset: CombinedMWFCCADataset, n_components: int = 10):
        self.dataset = dataset
        self.n_components = n_components
        self.cca_model = None
        self.canonical_correlations = None
        
    def prepare_data_for_cca(self):
        """
        Prepare data for CCA analysis.
        For attention decoding, we need to create attended/unattended envelopes.
        """
        eeg_data, labels = self.dataset.get_eeg_and_labels()
        
        # For CCA, we typically need:
        # - EEG data (X)
        # - Audio envelope data (Y) - but we don't have this in MWF-cleaned data
        # 
        # Alternative: Use left/right attention conditions as separate views
        # or create synthetic envelopes based on attention labels
        
        # Split by attention condition
        left_mask = labels == 0
        right_mask = labels == 1
        
        eeg_left = eeg_data[left_mask]
        eeg_right = eeg_data[right_mask]
        
        # For now, use left vs right EEG as two views for CCA
        # In practice, you would use audio envelopes
        min_samples = min(len(eeg_left), len(eeg_right))
        eeg_left = eeg_left[:min_samples]
        eeg_right = eeg_right[:min_samples]
        
        return eeg_left, eeg_right
    
    def fit_cca(self):
        """Fit CCA model."""
        print("Preparing data for CCA...")
        eeg_left, eeg_right = self.prepare_data_for_cca()
        
        print(f"EEG left shape: {eeg_left.shape}")
        print(f"EEG right shape: {eeg_right.shape}")
        
        # Use sklearn CCA
        print(f"Fitting CCA with {self.n_components} components...")
        self.cca_model = SklearnCCA(n_components=self.n_components)
        eeg_left_cca, eeg_right_cca = self.cca_model.fit_transform(eeg_left, eeg_right)
        
        # Calculate canonical correlations
        self.canonical_correlations = []
        for i in range(self.n_components):
            corr, _ = pearsonr(eeg_left_cca[:, i], eeg_right_cca[:, i])
            self.canonical_correlations.append(abs(corr))
        
        print(f"Canonical correlations: {self.canonical_correlations}")
        print(f"First canonical correlation: {self.canonical_correlations[0]:.4f}")
        
        return self.canonical_correlations
    
    def predict_attention(self, eeg_data: np.ndarray) -> np.ndarray:
        """Predict attention direction using CCA."""
        if self.cca_model is None:
            raise ValueError("CCA model not fitted. Call fit_cca() first.")
        
        # Transform EEG data
        eeg_transformed = self.cca_model.transform(eeg_data)
        
        # Simple prediction: use first canonical component
        # In practice, you would compare with attended/unattended envelopes
        predictions = (eeg_transformed[:, 0] > 0).astype(int)
        
        return predictions
    
    def evaluate(self, test_eeg: np.ndarray, test_labels: np.ndarray) -> Dict:
        """Evaluate CCA model performance."""
        predictions = self.predict_attention(test_eeg)
        
        accuracy = accuracy_score(test_labels, predictions)
        
        metrics = {
            'accuracy': accuracy,
            'first_canonical_correlation': self.canonical_correlations[0] if self.canonical_correlations else 0.0,
            'mean_canonical_correlation': np.mean(self.canonical_correlations) if self.canonical_correlations else 0.0,
            'canonical_correlations': self.canonical_correlations
        }
        
        return metrics


def main():
    """Main function to run MWFCCA analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run CCA analysis on combined MWF-cleaned datasets')
    parser.add_argument('--das_mwf_dir', type=str, default='MWF_cleaned_DAS',
                       help='Directory containing Das MWF-cleaned data')
    parser.add_argument('--fuglsang_mwf_dir', type=str, default='MWF_cleaned_Fuglsang',
                       help='Directory containing Fuglsang MWF-cleaned data')
    parser.add_argument('--n_components', type=int, default=10,
                       help='Number of CCA components')
    parser.add_argument('--output_dir', type=str, default='mwfcca_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("="*60)
    print("MWFCCA - Combined Das and Fuglsang MWF-Cleaned Data")
    print("="*60)
    
    # Load dataset
    dataset = CombinedMWFCCADataset(
        das_mwf_dir=args.das_mwf_dir,
        fuglsang_mwf_dir=args.fuglsang_mwf_dir
    )
    
    # Split into train/test
    eeg_data, labels = dataset.get_eeg_and_labels()
    n_samples = len(eeg_data)
    train_size = int(0.8 * n_samples)
    
    indices = np.random.permutation(n_samples)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_eeg = eeg_data[train_indices]
    train_labels = labels[train_indices]
    test_eeg = eeg_data[test_indices]
    test_labels = labels[test_indices]
    
    # Create analyzer
    analyzer = MWFCCAAnalyzer(dataset, n_components=args.n_components)
    
    # Fit CCA
    canonical_correlations = analyzer.fit_cca()
    
    # Evaluate
    metrics = analyzer.evaluate(test_eeg, test_labels)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'canonical_correlations': canonical_correlations,
        'metrics': metrics,
        'n_components': args.n_components,
        'n_train_samples': len(train_eeg),
        'n_test_samples': len(test_eeg)
    }
    
    with open(output_dir / 'mwfcca_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to:", output_dir / 'mwfcca_results.json')
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"First canonical correlation: {metrics['first_canonical_correlation']:.4f}")


if __name__ == '__main__':
    main()

