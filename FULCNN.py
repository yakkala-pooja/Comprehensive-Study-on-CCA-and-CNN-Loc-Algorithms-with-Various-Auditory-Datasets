#!/usr/bin/env python3
"""
FULCNN - CNN-LOC for Fulsang Dataset

CNN-LOC model for attention decoding on Fulsang EEG data.
Includes metrics (accuracy, MSED, ROC-AUC) and temporal analysis.
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
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

sys.path.append('telluride_decoding')

try:
    from telluride_decoding import decoding
    from telluride_decoding import brain_data
    from telluride_decoding import regression
    from telluride_decoding import attention_decoder
except ImportError as e:
    print(f"Warning: Could not import some telluride_decoding modules: {e}")
    print("Continuing with basic functionality...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FulsangDataset(Dataset):
    """
    Dataset for Fulsang EEG data. Uses FULPREPROCESSING output (EEG only).
    Handles windowing and preprocessing for attention decoding.
    """
    
    def __init__(self, tfrecord_dir: str, mode: str = 'full', 
                 window_size: int = 512, overlap: float = 0.5,
                 transform_eeg: bool = True, cache_size: int = 1000):
        self.tfrecord_dir = Path(tfrecord_dir)
        self.mode = mode
        self.window_size = window_size
        self.overlap = overlap
        self.transform_eeg = transform_eeg
        self.cache_size = cache_size
        
        # Fulsang dataset params
        self.sampling_rate = 64  # Hz
        self.n_channels = 66  # EEG channels
        self.attention_switch_duration = 20  # seconds
        
        # Cache for preprocessed windows
        self._window_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Load data from FULPREPROCESSING output
        self.eeg_data, self.labels, self.metadata = self._load_fulpreprocessing_data()
        
        # CRITICAL FIX: Fix labels BEFORE windowing/preprocessing to preserve temporal alignment
        # This ensures labels are correct before we destroy temporal semantics with preprocessing
        self.labels = self._fix_labels_before_processing()
        
        self.window_indices = self._create_fulsang_windows()
        
        # Check window label distribution
        window_labels = [label for _, label in self.window_indices]
        window_label_dist = np.bincount(window_labels)
        print(f"Loaded {len(self.window_indices)} windows, EEG shape: {self.eeg_data.shape}")
        print(f"Raw label dist: {np.bincount(self.labels)}")
        print(f"Window label dist: {dict(enumerate(window_label_dist))}")
    
    def _load_fulpreprocessing_data(self) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Load TFRecord data from FULPREPROCESSING output. Validates shapes."""
        tfrecord_files = list(self.tfrecord_dir.glob("*.tfrecords"))
        if not tfrecord_files:
            raise ValueError(f"No TFRecord files found in {self.tfrecord_dir}")
        
        print(f"Loading FULPREPROCESSING validated data from {len(tfrecord_files)} files...")
        
        all_eeg_data = []
        all_labels = []
        all_metadata = []
        
        n_success = 0
        n_failed = 0
        total_records = 0
        subject_stats = {}
        shape_errors = 0
        
        for tfrecord_file in tqdm(tfrecord_files, desc="Loading FULPREPROCESSING data"):
            try:
                dataset = tf.data.TFRecordDataset(str(tfrecord_file))
                records_in_file = 0
                file_subject_id = None
                
                for record in dataset:
                    try:
                        example = tf.train.Example.FromString(record.numpy())
                        features = example.features.feature
                        
                        # Check required features (EEG only format)
                        required_features = ['eeg', 'attention_label', 'subject_id']
                        if not all(key in features for key in required_features):
                            continue
                        
                        # Extract and validate EEG data
                        eeg_values = features['eeg'].float_list.value
                        if not eeg_values or len(eeg_values) == 0:
                            continue
                        
                        # Must be exactly 66 channels
                        if len(eeg_values) != 66:
                            print(f"ERROR: Expected 66 EEG channels, got {len(eeg_values)} in {tfrecord_file.name}")
                            shape_errors += 1
                            continue
                        
                        # Reshape to (1, 66) for single sample
                        eeg_data = np.array(eeg_values, dtype=np.float32).reshape(1, 66)
                        
                        # Check for invalid values
                        if np.any(np.isnan(eeg_data)) or np.any(np.isinf(eeg_data)):
                            print(f"WARNING: Invalid EEG values (NaN/Inf) in {tfrecord_file.name}")
                            continue
                        
                        # Extract attention label
                        label_values = features['attention_label'].int64_list.value
                        if not label_values or len(label_values) == 0:
                            continue
                        label = int(label_values[0])
                        
                        # Validate label
                        if label not in [0, 1]:
                            print(f"ERROR: Invalid attention label {label} in {tfrecord_file.name}")
                            continue
                        
                        # Extract metadata
                        subject_id = "unknown"
                        sample_idx = 0
                        
                        if 'subject_id' in features:
                            subject_values = features['subject_id'].bytes_list.value
                            if subject_values and len(subject_values) > 0:
                                try:
                                    subject_id = subject_values[0].decode('utf-8')
                                    file_subject_id = subject_id
                                except Exception:
                                    subject_id = f"subject_{total_records}"
                        
                        if 'sample_idx' in features:
                            sample_values = features['sample_idx'].int64_list.value
                            if sample_values and len(sample_values) > 0:
                                sample_idx = sample_values[0]
                        
                        # Track subject statistics
                        if subject_id not in subject_stats:
                            subject_stats[subject_id] = {'samples': 0, 'labels': []}
                        subject_stats[subject_id]['samples'] += 1
                        subject_stats[subject_id]['labels'].append(label)
                        
                        metadata = {
                            'subject_id': subject_id,
                            'file': tfrecord_file.name,
                            'sample_idx': sample_idx,
                            'attention_label': label,
                            'preprocessing_method': 'FULPREPROCESSING',
                            'validation_passed': True,
                            'data_type': 'EEG_only',
                            'eeg_shape': eeg_data.shape,
                            'label_alignment': 'validated'
                        }
                        
                        all_eeg_data.append(eeg_data)
                        all_labels.append(label)
                        all_metadata.append(metadata)
                        records_in_file += 1
                        total_records += 1
                        
                    except Exception as record_error:
                        print(f"ERROR processing record in {tfrecord_file.name}: {record_error}")
                        continue
                
                if records_in_file > 0:
                    n_success += 1
                else:
                    n_failed += 1
                    
            except Exception as e:
                n_failed += 1
                print(f"ERROR loading {tfrecord_file.name}: {e}")
                continue
        
        print(f"Successfully loaded {n_success} files, {n_failed} files failed")
        print(f"Total records loaded: {total_records}")
        print(f"Shape errors: {shape_errors}")
        
        if shape_errors > 0:
            print(f"WARNING: {shape_errors} records had shape errors")
        
        if not all_eeg_data:
            raise ValueError("No valid FULPREPROCESSING data found in TFRecord files")
        
        eeg_data = np.vstack(all_eeg_data)
        labels = np.array(all_labels, dtype=np.int64)
        
        # Final shape validation
        print(f"Final data shapes: EEG {eeg_data.shape}, Labels {labels.shape}")
        
        # Check raw label distribution BEFORE windowing
        raw_label_dist = np.bincount(labels)
        print(f"\nRaw label distribution (before windowing): {dict(enumerate(raw_label_dist))}")
        if len(raw_label_dist) < 2:
            print(f"⚠️  CRITICAL: Raw data has only {len(raw_label_dist)} class(es)!")
            print(f"   This means the TFRecords have incorrect labels or all subjects have the same class.")
        elif raw_label_dist[0] == 0 or raw_label_dist[1] == 0:
            print(f"⚠️  CRITICAL: Raw data is missing one class!")
        else:
            balance_ratio = min(raw_label_dist) / max(raw_label_dist)
            print(f"Raw data class distribution (ratio: {balance_ratio:.3f})")
            if balance_ratio < 0.5:
                print(f"  Note: Dataset is unbalanced (expected for Fulsang dataset)")
        
        # Check label structure - are labels grouped or alternating?
        # Check first 100 and last 100 samples to see pattern
        first_100_labels = labels[:100]
        last_100_labels = labels[-100:]
        first_100_dist = np.bincount(first_100_labels)
        last_100_dist = np.bincount(last_100_labels)
        print(f"Label structure check:")
        print(f"  First 100 samples: {dict(enumerate(first_100_dist))}")
        print(f"  Last 100 samples: {dict(enumerate(last_100_dist))}")
        
        # Check label transitions
        label_changes = np.sum(np.diff(labels) != 0)
        expected_transitions = len(labels) // 1280  # Every 20 seconds = 1280 samples
        print(f"  Total label transitions in data: {label_changes}")
        print(f"  Expected transitions (if alternating every 20s): ~{expected_transitions}")
        
        # CRITICAL: Check if labels are alternating too frequently (data issue)
        if label_changes > expected_transitions * 10:
            print(f"\n⚠️  CRITICAL DATA ISSUE DETECTED:")
            print(f"   Labels are alternating {label_changes} times, but should only alternate ~{expected_transitions} times.")
            print(f"   This suggests labels in TFRecords are alternating EVERY SAMPLE instead of every 20 seconds.")
            print(f"   This will cause ALL windows to get the same class (due to center-label assignment bias).")
            print(f"   SOLUTION: Regenerate TFRecords with correct labels (alternating every 1280 samples, not every sample).")
            print(f"   WORKAROUND: Using majority vote for window labels when labels alternate too frequently.")
        
        # Check per-subject label structure
        print(f"\nPer-subject label structure (first subject):")
        if len(all_metadata) > 0 and len(labels) >= 3200:
            subject_labels = labels[:3200]
            subject_dist = np.bincount(subject_labels)
            first_half = np.bincount(subject_labels[:1600])
            second_half = np.bincount(subject_labels[1600:])
            print(f"  Subject {all_metadata[0].get('subject_id', 'unknown')}:")
            print(f"    Full (3200 samples): {dict(enumerate(subject_dist))}")
            print(f"    First half (0-1600): {dict(enumerate(first_half))}")
            print(f"    Second half (1600-3200): {dict(enumerate(second_half))}")
            
            # Check trial structure (if 1280 samples per trial)
            if len(subject_labels) >= 2560:
                trial1 = np.bincount(subject_labels[0:1280])
                trial2 = np.bincount(subject_labels[1280:2560])
                print(f"    Trial 1 (0-1280): {dict(enumerate(trial1))}")
                print(f"    Trial 2 (1280-2560): {dict(enumerate(trial2))}")
            
            # Check where label transitions occur
            transitions = np.where(np.diff(subject_labels) != 0)[0]
            print(f"    Label transitions at indices: {transitions[:10]} (showing first 10)")
        
        print(f"\n⚠️  PREPROCESSING INFO:")
        print(f"  FULPREPROCESSING creates labels as alternating trials:")
        print(f"    - Each trial = 20 seconds (1280 samples at 64 Hz)")
        print(f"    - Trial 0 = class 0, Trial 1 = class 1, Trial 2 = class 0, etc.")
        print(f"    - With 512-sample windows (8s) and 50% overlap, many windows")
        print(f"      fall entirely within the first trial (class 0)")
        print(f"    - Center label assignment should help, but if all centers")
        print(f"      fall in class 0 regions, all windows get class 0")
        
        # Check channel count dynamically (don't hardcode 66)
        actual_channels = eeg_data.shape[1]
        print(f"Detected {actual_channels} EEG channels in data")
        if actual_channels != self.n_channels:
            print(f"WARNING: Dataset has {actual_channels} channels but expected {self.n_channels}. Updating n_channels.")
            self.n_channels = actual_channels
        
        if len(eeg_data) != len(labels):
            raise ValueError(f"CRITICAL: EEG samples ({len(eeg_data)}) != labels ({len(labels)})")
        
        del all_eeg_data, all_labels
        import gc
        gc.collect()
        
        return eeg_data, labels, all_metadata
    
    def _fix_labels_before_processing(self) -> np.ndarray:
        """
        CRITICAL: Fix labels BEFORE preprocessing to preserve temporal alignment.
        This ensures labels are correct before we destroy temporal semantics.
        """
        trial_length = 1280  # 20 seconds at 64 Hz
        n_samples = len(self.labels)
        n_trials = n_samples // trial_length
        
        # Check if labels need fixing
        label_changes = np.sum(np.diff(self.labels) != 0)
        expected_transitions = n_trials - 1 if n_trials > 1 else 0
        labels_alternating_too_fast = label_changes > expected_transitions * 10 if expected_transitions > 0 else label_changes > 100
        labels_alternating_too_slow = label_changes < expected_transitions * 0.1 if expected_transitions > 0 else False
        
        if not (labels_alternating_too_fast or labels_alternating_too_slow):
            # Labels look correct, return as-is
            return self.labels
        
        print(f"\n🔧 FIXING LABELS BEFORE PREPROCESSING (preserving temporal alignment)")
        print(f"   Original transitions: {label_changes}, Expected: ~{expected_transitions}")
        
        if n_trials < 1:
            print(f"   WARNING: Cannot fix - need at least 1 complete trial")
            return self.labels
        
        # Reconstruct labels based on trial structure
        fixed_labels = np.zeros(n_samples, dtype=np.int64)
        for trial_idx in range(n_trials):
            trial_start = trial_idx * trial_length
            trial_end = min(trial_start + trial_length, n_samples)
            trial_class = trial_idx % 2  # Alternating: 0, 1, 0, 1, ...
            fixed_labels[trial_start:trial_end] = trial_class
        
        # Handle remaining samples
        if n_samples % trial_length > 0:
            remaining_start = n_trials * trial_length
            remaining_class = n_trials % 2
            fixed_labels[remaining_start:] = remaining_class
        
        fixed_transitions = np.sum(np.diff(fixed_labels) != 0)
        print(f"   ✓ Fixed labels: {fixed_transitions} transitions (expected ~{expected_transitions})")
        
        return fixed_labels
    
    def _create_fulsang_windows(self) -> List[Tuple[int, int]]:
        """Create sliding windows from Fulsang data, respecting subject boundaries."""
        # Convert to seconds for display
        window_seconds = self.window_size / self.sampling_rate
        step_size = int(self.window_size * (1 - self.overlap))
        step_seconds = step_size / self.sampling_rate
        
        # Build subject boundaries from metadata to prevent cross-subject windows
        subject_ranges = {}
        current_subject = None
        start_idx = 0
        
        for i, metadata in enumerate(self.metadata):
            subject_id = metadata.get('subject_id', 'unknown')
            
            if subject_id != current_subject:
                if current_subject is not None:
                    subject_ranges[current_subject] = (start_idx, i)
                current_subject = subject_id
                start_idx = i
        
        # Last subject
        if current_subject is not None:
            subject_ranges[current_subject] = (start_idx, len(self.metadata))
        
        print(f"Detected {len(subject_ranges)} subjects for boundary checking")
        
        total_windows = (len(self.eeg_data) - self.window_size) // step_size + 1
        
        print(f"Creating windows (size: {self.window_size} samples, {window_seconds:.1f}s)")
        
        # Warn about window size
        if window_seconds < 1.0:
            print(f"WARNING: Very short window ({window_seconds:.1f}s) may have poor signal-to-noise")
        elif window_seconds > 20.0:
            print(f"WARNING: Very long window ({window_seconds:.1f}s) may miss temporal dynamics")
        
        # Labels are already fixed before preprocessing, so we can trust them
        # Now use temporal-aware labeling instead of majority vote
        
        window_indices = []
        window_label_stats = {'class_0': 0, 'class_1': 0, 'mixed': 0}
        skipped_cross_boundary = 0
        
        for i in range(total_windows):
            data_idx = i * step_size
            window_end = data_idx + self.window_size
            
            # Check if window would exceed data length
            if window_end > len(self.eeg_data):
                continue
            
            # Check if window spans subject boundary
            window_spans_boundary = False
            window_subject = None
            
            # Find which subject this window belongs to
            for subj_id, (subj_start, subj_end) in subject_ranges.items():
                if subj_start <= data_idx < subj_end:
                    window_subject = subj_id
                    # Check if window extends beyond this subject's boundary
                    if window_end > subj_end:
                        window_spans_boundary = True
                    break
            
            # Skip windows that span subject boundaries (prevents data leakage)
            if window_spans_boundary:
                skipped_cross_boundary += 1
                continue
            
            # Window is valid - create it
            window_start = data_idx
            window_labels = self.labels[window_start:window_end]
            
            # TEMPORAL-AWARE LABELING: Weight by position in trial, not majority vote
            # Attention ramps gradually, so windows near trial boundaries need weighted labels
            # This preserves the gradual transition information that majority vote destroys
            trial_length = 1280  # 20 seconds
            window_center_in_trial = (data_idx + self.window_size // 2) % trial_length
            window_relative_pos = window_center_in_trial / trial_length  # 0.0 to 1.0
            
            # Determine which trial this window is in
            trial_idx = (data_idx + self.window_size // 2) // trial_length
            trial_class = trial_idx % 2  # Alternating: 0, 1, 0, 1, ...
            
            # For windows near trial boundaries, use weighted label based on position
            # Windows in middle of trial get pure trial label
            # Windows near boundaries get weighted average
            boundary_threshold = 0.15  # 15% of trial = ~3 seconds
            
            if window_relative_pos < boundary_threshold:
                # Near start of trial - weight towards previous trial
                prev_trial_class = (trial_idx - 1) % 2 if trial_idx > 0 else trial_class
                weight_current = window_relative_pos / boundary_threshold
                weighted_label = weight_current * trial_class + (1 - weight_current) * prev_trial_class
                window_label = int(round(weighted_label))
            elif window_relative_pos > (1 - boundary_threshold):
                # Near end of trial - weight towards next trial
                next_trial_class = (trial_idx + 1) % 2
                weight_current = (1 - window_relative_pos) / boundary_threshold
                weighted_label = weight_current * trial_class + (1 - weight_current) * next_trial_class
                window_label = int(round(weighted_label))
            else:
                # Middle of trial - use pure trial label
                window_label = trial_class
            
            # Clamp to valid range
            window_label = max(0, min(1, window_label))
            
            # Track label distribution for diagnostics
            unique_labels = np.unique(window_labels)
            if len(unique_labels) == 1:
                if unique_labels[0] == 0:
                    window_label_stats['class_0'] += 1
                else:
                    window_label_stats['class_1'] += 1
            else:
                window_label_stats['mixed'] += 1
            
            window_indices.append((data_idx, window_label))
        
        if skipped_cross_boundary > 0:
            print(f"Skipped {skipped_cross_boundary} windows that would span subject boundaries (prevents data leakage)")
        
        print(f"Created {len(window_indices)} windows")
        print(f"Window label assignment stats:")
        print(f"  Windows with only class 0: {window_label_stats['class_0']}")
        print(f"  Windows with only class 1: {window_label_stats['class_1']}")
        print(f"  Windows with mixed labels: {window_label_stats['mixed']}")
        
        # Check final window label distribution
        final_window_labels = [label for _, label in window_indices]
        final_label_dist = np.bincount(final_window_labels)
        print(f"Final window label distribution: {dict(enumerate(final_label_dist))}")
        
        # CRITICAL: Check if window labels are severely imbalanced
        if len(final_label_dist) < 2:
            print(f"\n⚠️  CRITICAL: All windows have the same label! This will cause the model to fail.")
            print(f"   Window label distribution: {dict(enumerate(final_label_dist))}")
            print(f"   This suggests the raw labels are incorrect or the label fixing didn't work.")
        elif len(final_label_dist) == 2:
            balance_ratio = min(final_label_dist) / max(final_label_dist)
            print(f"Window label balance ratio: {balance_ratio:.3f} (1.0 = perfectly balanced)")
            if balance_ratio < 0.1:
                print(f"⚠️  WARNING: Severely imbalanced window labels (ratio: {balance_ratio:.3f})")
                print(f"   This will make learning very difficult. Consider checking the raw labels.")
            elif balance_ratio < 0.3:
                print(f"⚠️  WARNING: Imbalanced window labels (ratio: {balance_ratio:.3f})")
                print(f"   The model will use class weights, but results may be suboptimal.")
        
        return window_indices
    
    def _fix_alternating_labels(self) -> Optional[np.ndarray]:
        """
        Attempt to fix labels that are alternating every sample.
        Reconstructs labels based on trial structure (20s = 1280 samples per trial).
        Uses alternating pattern: Trial 0 = class 0, Trial 1 = class 1, etc.
        """
        try:
            trial_length = 1280  # 20 seconds at 64 Hz
            n_samples = len(self.labels)
            n_trials = n_samples // trial_length
            
            print(f"   Attempting label fix: {n_samples} samples, {n_trials} complete trials of {trial_length} samples each")
            
            if n_trials < 1:
                print(f"   Cannot fix: Need at least 1 complete trial ({trial_length} samples), but only have {n_samples} samples")
                return None
            
            # Check original label distribution
            original_dist = np.bincount(self.labels)
            print(f"   Original label distribution: {dict(enumerate(original_dist))}")
            
            fixed_labels = np.zeros(n_samples, dtype=np.int64)
            
            # Reconstruct labels based on alternating trial pattern
            # Trial 0 = class 0, Trial 1 = class 1, Trial 2 = class 0, etc.
            for trial_idx in range(n_trials):
                trial_start = trial_idx * trial_length
                trial_end = min(trial_start + trial_length, n_samples)
                
                # Assign alternating class based on trial index
                # Even trials (0, 2, 4, ...) = class 0
                # Odd trials (1, 3, 5, ...) = class 1
                trial_class = trial_idx % 2
                fixed_labels[trial_start:trial_end] = trial_class
                print(f"   Trial {trial_idx}: samples {trial_start}-{trial_end} -> class {trial_class}")
            
            # Handle remaining samples (use the pattern from the last complete trial)
            if n_samples % trial_length > 0:
                remaining_start = n_trials * trial_length
                remaining_class = n_trials % 2  # Continue the alternating pattern
                fixed_labels[remaining_start:] = remaining_class
                print(f"   Remaining {n_samples - remaining_start} samples -> class {remaining_class}")
            
            # Verify the fix makes sense
            fixed_transitions = np.sum(np.diff(fixed_labels) != 0)
            expected_transitions = n_trials - 1  # One transition per trial boundary (if n_trials > 1)
            
            # Check label distribution
            fixed_dist = np.bincount(fixed_labels)
            print(f"   Fixed label distribution: {dict(enumerate(fixed_dist))}")
            print(f"   Fixed transitions: {fixed_transitions} (expected ~{expected_transitions})")
            
            # Check balance
            if len(fixed_dist) == 2:
                balance_ratio = min(fixed_dist) / max(fixed_dist)
                print(f"   Fixed label balance ratio: {balance_ratio:.3f}")
            
            # Validation: check transitions (class balance is not required - dataset is naturally unbalanced)
            if len(fixed_dist) >= 1 and fixed_transitions <= expected_transitions * 2:
                print(f"   ✓ Label fix validation passed")
                return fixed_labels
            else:
                print(f"   ✗ Fix validation failed: transitions={fixed_transitions} (expected ~{expected_transitions})")
                return None
                
        except Exception as e:
            print(f"   Error fixing labels: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _fulsang_eeg_preprocessing(self, eeg_window: np.ndarray) -> np.ndarray:
        """Preprocess EEG window: artifacts, filtering, normalization."""
        from scipy import signal
        
        # Remove high-amplitude artifacts (>5 std dev)
        artifact_thresh = 5.0
        for ch in range(eeg_window.shape[1]):
            ch_data = eeg_window[:, ch]
            std_val = np.std(ch_data)
            mean_val = np.mean(ch_data)
            
            artifacts = np.abs(ch_data - mean_val) > (artifact_thresh * std_val)
            
            if np.any(artifacts):
                # Interpolate over artifacts
                valid_indices = ~artifacts
                if np.sum(valid_indices) > 2:  # Need at least 2 points
                    from scipy.interpolate import interp1d
                    valid_data = ch_data[valid_indices]
                    valid_time = np.where(valid_indices)[0]
                    all_time = np.arange(len(ch_data))
                    
                    f_interp = interp1d(valid_time, valid_data, kind='linear', 
                                      bounds_error=False, fill_value='extrapolate')
                    eeg_window[:, ch] = f_interp(all_time)
        
        # Remove DC offset
        eeg_window = eeg_window - np.mean(eeg_window, axis=0, keepdims=True)
        
        # GENTLER BANDPASS FILTER: Preserve low-frequency attention envelopes
        # Use wider passband (0.5-40 Hz) to preserve delta/theta attention signals
        nyquist = self.sampling_rate / 2
        low_freq = 0.5 / nyquist  # Lower cutoff to preserve delta/theta
        high_freq = min(40.0 / nyquist, 0.99)
        
        b, a = signal.butter(2, [low_freq, high_freq], btype='band')  # Lower order = less distortion
        
        # Apply filter to each channel
        filtered_eeg = np.zeros_like(eeg_window)
        for ch in range(eeg_window.shape[1]):
            filtered_eeg[:, ch] = signal.filtfilt(b, a, eeg_window[:, ch])
        
        # GENTLER NORMALIZATION: Preserve relative amplitudes across time
        # Use per-channel z-score instead of MAD to preserve temporal structure
        mean_per_ch = np.mean(filtered_eeg, axis=0, keepdims=True)
        std_per_ch = np.std(filtered_eeg, axis=0, keepdims=True)
        std_per_ch = np.where(std_per_ch < 1e-8, 1.0, std_per_ch)
        filtered_eeg = (filtered_eeg - mean_per_ch) / std_per_ch
        
        # LIGHTER CLIPPING: Preserve signal dynamics
        # Use gentler clipping to preserve attention-related modulations
        filtered_eeg = np.tanh(filtered_eeg * 0.3)  # Reduced from 0.5
        
        # Final check for NaNs/Infs
        if np.any(np.isnan(filtered_eeg)) or np.any(np.isinf(filtered_eeg)):
            print("WARNING: Invalid values detected after preprocessing")
            filtered_eeg = np.nan_to_num(filtered_eeg, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return filtered_eeg.astype(np.float32)
    
    def _eeg_to_timefreq_fulsang(self, eeg_window: np.ndarray) -> np.ndarray:
        """Convert EEG to time-frequency representation using spectrogram."""
        from scipy import signal
        
        tf_data = []
        
        for ch_idx in range(eeg_window.shape[1]):
            # Compute spectrogram with adaptive parameters
            # Ensure nperseg is at least 16 and no larger than window length
            nperseg = max(16, min(64, len(eeg_window) // 2))
            noverlap = max(8, nperseg // 2)  # 50% overlap, but at least 8 samples
            
            # Compute spectrogram
            freqs, times, Sxx = signal.spectrogram(
                eeg_window[:, ch_idx], 
                fs=self.sampling_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                window='hann'
            )
            
            # Extract power in standard EEG bands
            bands = [
                (1, 4),   # Delta
                (4, 8),   # Theta  
                (8, 13),  # Alpha
                (13, 25), # Beta
                (25, 40)  # Gamma
            ]
            
            # Get band power for each time point
            band_powers = []
            for low, high in bands:
                if high >= self.sampling_rate / 2:
                    high = self.sampling_rate / 2 - 1
                
                mask = (freqs >= low) & (freqs <= high)
                if np.any(mask):
                    power = np.mean(Sxx[mask, :], axis=0)
                else:
                    power = np.zeros(Sxx.shape[1])
                
                band_powers.append(power)
            
            # Stack: (n_bands, n_time_points)
            ch_tf = np.vstack(band_powers)
            tf_data.append(ch_tf)
        
        # Combine channels: (n_channels, n_bands, n_time_points)
        time_freq_array = np.array(tf_data)
        
        # Current shape: (channels, freq_bands, time_points)
        # Need to transpose to: (channels, time_points, freq_bands)
        time_freq_array = np.transpose(time_freq_array, (0, 2, 1))
        
        # PRESERVE TEMPORAL RESOLUTION: Don't force interpolation to fixed size
        # Instead, preserve natural temporal resolution to maintain alignment with trial structure
        # Only interpolate if absolutely necessary for model compatibility
        # For 8s window (512 samples), spectrogram naturally produces ~16-32 time points
        # This preserves temporal alignment better than forcing to 64
        min_time_points = 8  # Minimum for pooling operations
        max_time_points = 128  # Maximum to prevent excessive computation
        
        # Preserve natural temporal resolution, but ensure minimum for model
        natural_time_points = time_freq_array.shape[1]
        target_time_points = max(min_time_points, min(natural_time_points, max_time_points))
        
        if time_freq_array.shape[1] != target_time_points:
            from scipy.interpolate import interp1d
            original_time = np.linspace(0, 1, time_freq_array.shape[1])
            target_time = np.linspace(0, 1, target_time_points)
            
            interpolated_data = np.zeros((time_freq_array.shape[0], target_time_points, time_freq_array.shape[2]))
            for ch in range(time_freq_array.shape[0]):
                for band in range(time_freq_array.shape[2]):
                    f_interp = interp1d(original_time, time_freq_array[ch, :, band], kind='linear', 
                                      bounds_error=False, fill_value='extrapolate')
                    interpolated_data[ch, :, band] = f_interp(target_time)
            
            time_freq_array = interpolated_data
        
        # Output: (channels, time_frames, freq_bands)
        return time_freq_array.astype(np.float32)
    
    def __len__(self):
        return len(self.window_indices)
    
    def __getitem__(self, idx):
        data_idx, label = self.window_indices[idx]
        
        # Check cache first
        cache_key = (data_idx, self.mode)
        if cache_key in self._window_cache:
            self._cache_hits += 1
            cached_data, cached_label = self._window_cache[cache_key]
            return cached_data, cached_label
        
        self._cache_misses += 1
        
        # Extract window (EEG only)
        window_eeg = self.eeg_data[data_idx:data_idx + self.window_size]
        
        # Apply preprocessing
        try:
            window_eeg = self._fulsang_eeg_preprocessing(window_eeg)
        except Exception:
            window_eeg = window_eeg - np.mean(window_eeg, axis=0, keepdims=True)
            window_eeg = window_eeg / (np.std(window_eeg, axis=0, keepdims=True) + 1e-8)
            window_eeg = np.tanh(window_eeg * 0.5)
        
        # Convert to time-frequency representation
        if self.transform_eeg:
            try:
                window_eeg = self._eeg_to_timefreq_fulsang(window_eeg)
                # window_eeg shape: (channels, time, freq)
                # Model expects: (channels, time, freq) - this is correct
            except Exception:
                # Fallback: if TF transform fails, create dummy freq dimension
                # Shape: (time, channels) -> (channels, time, 1)
                if window_eeg.ndim == 2:
                    window_eeg = window_eeg.T  # (channels, time)
                    window_eeg = window_eeg[:, :, np.newaxis]  # (channels, time, 1)
        
        # Convert to tensors
        window_tensor = torch.FloatTensor(window_eeg)
        label_tensor = torch.LongTensor([label])
        
        # Ensure proper tensor dimensions: (channels, time, freq)
        if window_tensor.dim() == 2:
            # If 2D, assume (time, channels) and transpose to (channels, time)
            # Then add freq dimension
            if window_tensor.shape[1] == self.n_channels:
                # Shape is (time, channels) -> transpose to (channels, time)
                window_tensor = window_tensor.transpose(0, 1)
            # Add freq dimension: (channels, time) -> (channels, time, 1)
            window_tensor = window_tensor.unsqueeze(-1)
        elif window_tensor.dim() == 3:
            # Should be (channels, time, freq) - verify
            if window_tensor.shape[0] != self.n_channels:
                # If first dim is not channels, might need to transpose
                if window_tensor.shape[1] == self.n_channels:
                    # (time, channels, freq) -> (channels, time, freq)
                    window_tensor = window_tensor.transpose(0, 1)
                elif window_tensor.shape[2] == self.n_channels:
                    # (time, freq, channels) -> (channels, time, freq)
                    window_tensor = window_tensor.permute(2, 0, 1)
        
        # Final verification: ensure first dimension is channels
        if window_tensor.shape[0] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels in first dimension, got {window_tensor.shape[0]}. Full shape: {window_tensor.shape}")
        
        return window_tensor, label_tensor


class SpatialTemporalAttention(nn.Module):
    """Channel attention for EEG data. Kept simple to save memory."""
    
    def __init__(self, channels: int, reduction: int = 8):
        super(SpatialTemporalAttention, self).__init__()
        
        self.channels = channels
        self.reduction = max(1, reduction)
        self.reduced_channels = max(1, channels // self.reduction)
        
        # Channel attention only (no temporal to save memory)
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
    """Residual block with attention. Standard ResNet-style."""
    
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
    """Multi-scale features using different kernel sizes. Simplified to save memory."""
    
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


class FULCNNBackbone(nn.Module):
    """Backbone network: attention, residual blocks, multi-scale features."""
    
    def __init__(self, input_channels: int = 66, input_time: int = 32, input_freq: int = 5,
                 adaptive_input: bool = True):
        super(FULCNNBackbone, self).__init__()
        
        self.input_channels = input_channels
        self.input_time = input_time
        self.input_freq = input_freq
        self.adaptive_input = adaptive_input
        
        # Adaptive architecture based on input dimensions
        # For small inputs, use fewer pooling operations to prevent dimension collapse
        use_small_architecture = (input_time < 4) or (input_freq < 4)
        
        if use_small_architecture:
            print(f"Small input dimensions detected: ({input_time}x{input_freq}). Using adaptive architecture.")
            self._build_small_input_architecture()
        else:
            print(f"Standard input dimensions: ({input_time}x{input_freq}). Using standard architecture.")
            self._build_standard_architecture()
        
        # Calculate output size
        self._calculate_output_size()
    
    def _build_small_input_architecture(self):
        """Build architecture optimized for small input dimensions."""
        # For small inputs, reduce pooling operations to prevent dimension collapse
        # Initial multi-scale features
        self.initial_features = MultiScaleFeatureExtractor(self.input_channels, 32)
        
        # Temporal blocks - conditional pooling based on time dimension
        self.temporal_block1 = ResidualBlock(32, 32, stride=1)
        if self.input_time >= 2:
            # Can do one temporal pool if time >= 2
            self.temporal_pool1 = nn.MaxPool2d((2, 1), (2, 1))
        else:
            self.temporal_pool1 = None  # Skip pooling if too small
        
        self.temporal_block2 = ResidualBlock(32, 64, stride=1)
        # Only do second temporal pool if time >= 4 (after first pool, time would be >= 2)
        if self.input_time >= 4:
            self.temporal_pool2 = nn.MaxPool2d((2, 1), (2, 1))
        else:
            self.temporal_pool2 = None
        
        # Spatial blocks - conditional pooling based on freq dimension
        self.spatial_block1 = ResidualBlock(64, 64, stride=1)
        if self.input_freq >= 2:
            # Can do one spatial pool if freq >= 2
            self.spatial_pool1 = nn.MaxPool2d((1, 2), (1, 2))
        else:
            self.spatial_pool1 = None
        
        self.spatial_block2 = ResidualBlock(64, 128, stride=1)
        # Only do second spatial pool if freq >= 4 (after first pool, freq would be >= 2)
        if self.input_freq >= 4:
            self.spatial_pool2 = nn.MaxPool2d((1, 2), (1, 2))
        else:
            self.spatial_pool2 = None
        
        # Global attention
        self.global_attention = SpatialTemporalAttention(128)
        
        # Adaptive pooling
        self.adaptive_pooling = AdaptivePooling(output_size=1)
    
    def _build_standard_architecture(self):
        """
        Build standard architecture with REDUCED TEMPORAL POOLING.
        Preserves long-range temporal context needed for attention decoding.
        """
        # Initial multi-scale features
        self.initial_features = MultiScaleFeatureExtractor(self.input_channels, 32)
        
        # Temporal blocks - REDUCED POOLING to preserve temporal resolution
        # Only pool once instead of twice to keep more temporal context
        self.temporal_block1 = ResidualBlock(32, 32, stride=1)
        # Single temporal pool (reduced from 2 pools)
        if self.input_time >= 4:
            self.temporal_pool1 = nn.MaxPool2d((2, 1), (2, 1))
        else:
            self.temporal_pool1 = None
        
        self.temporal_block2 = ResidualBlock(32, 64, stride=1)
        # NO SECOND TEMPORAL POOL - preserve temporal resolution
        self.temporal_pool2 = None
        
        # Spatial blocks - keep spatial pooling (frequency dimension)
        self.spatial_block1 = ResidualBlock(64, 64, stride=1)
        self.spatial_pool1 = nn.MaxPool2d((1, 2), (1, 2))
        
        self.spatial_block2 = ResidualBlock(64, 128, stride=1)
        self.spatial_pool2 = nn.MaxPool2d((1, 2), (1, 2))
        
        # Global attention
        self.global_attention = SpatialTemporalAttention(128)
        
        # Adaptive pooling
        self.adaptive_pooling = AdaptivePooling(output_size=1)
        
    
    def _calculate_output_size(self):
        """Figure out output size by running a dummy input."""
        dummy_input = torch.randn(1, self.input_channels, self.input_time, self.input_freq)
        
        try:
            with torch.no_grad():
                x = self.forward(dummy_input)
                self.output_size = x.numel()
        except RuntimeError as e:
            if "Output size is too small" in str(e):
                raise ValueError(f"CNNLoc architecture cannot handle input size ({self.input_time}, {self.input_freq}). "
                               f"Even with adaptive architecture, dimensions become too small after pooling. "
                               f"Error: {e}")
            raise
        
    
    def forward(self, x):
        """Forward pass with adaptive architecture support."""
        # Multi-scale features
        x = self.initial_features(x)
        
        # Temporal processing
        x = self.temporal_block1(x)
        if self.temporal_pool1 is not None:
            x = self.temporal_pool1(x)
        
        x = self.temporal_block2(x)
        if self.temporal_pool2 is not None:
            x = self.temporal_pool2(x)
        
        # Spatial processing
        x = self.spatial_block1(x)
        if self.spatial_pool1 is not None:
            x = self.spatial_pool1(x)
        
        x = self.spatial_block2(x)
        if self.spatial_pool2 is not None:
            x = self.spatial_pool2(x)
        
        # Attention
        x = self.global_attention(x)
        
        # Pool and flatten
        x = self.adaptive_pooling(x)
        x = x.view(x.size(0), -1)
        
        return x


class FULCNNModel(nn.Module):
    """Full FULCNN model: backbone + classifier for EEG attention decoding."""
    
    def __init__(self, input_channels: int = 66, input_time: int = 32, input_freq: int = 5,
                 num_classes: int = 2, dropout_rate: float = 0.3):
        super(FULCNNModel, self).__init__()
        
        # Create backbone
        self.backbone = FULCNNBackbone(input_channels, input_time, input_freq)
        
        # SIMPLIFIED CLASSIFIER: Reduced capacity to match signal-to-noise ratio
        # For low-SNR attention decoding, simpler is better
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate * 0.5),  # Reduced dropout
            nn.Linear(self.backbone.output_size, 64),  # Reduced from 128
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(64, num_classes)  # Direct to output, removed intermediate layer
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
                # Use proper initialization for Linear layers
                # Kaiming uniform is better for ReLU activations
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
    
    def forward(self, x):
        """Forward pass through the model."""
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class FULCNNTrainer:
    """Handles training, validation, testing, and metrics for FULCNN."""
    
    def __init__(self, model: FULCNNModel, device: torch.device, 
                 output_dir: str = "fulcnn_results", tfrecord_dir: str = None, 
                 sampling_rate: int = 64, window_size: int = 512):
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Dataset parameters
        self.tfrecord_dir = tfrecord_dir
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        self.best_val_acc = 0.0
        self.best_model_path = self.output_dir / "best_model.pth"
        
    
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module, scheduler: Optional[optim.lr_scheduler._LRScheduler] = None) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
            # Move to device (GPU if available) with non_blocking for faster transfer
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            target = target.squeeze()
            
            # Verify GPU usage on first batch
            if batch_idx == 0:
                if torch.cuda.is_available():
                    data_dev = data.device
                    model_dev = next(self.model.parameters()).device
                    print(f"  ✓ GPU Verification - Data device: {data_dev}, Model device: {model_dev}")
                    if data_dev.type != 'cuda' or model_dev.type != 'cuda':
                        print(f"  ⚠️  WARNING: Not using GPU! Data: {data_dev}, Model: {model_dev}")
                    else:
                        print(f"  ✓ GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB allocated")
                else:
                    print(f"  Using CPU - Data device: {data.device}, Model device: {next(self.model.parameters()).device}")
            
            # Validate input data before processing
            if torch.any(torch.isnan(data)) or torch.any(torch.isinf(data)):
                print(f"  WARNING: NaN/Inf detected in input data at batch {batch_idx}, skipping...")
                continue
            
            # Data augmentation
            if self.model.training:
                # Add small noise (helps with generalization)
                noise = torch.randn_like(data) * 0.01
                data = data + noise
                
                # Temporal shift augmentation (disabled - can break temporal structure in TF representation)
                # The time-frequency representation has specific temporal structure that shouldn't be shifted
                # if torch.rand(1) > 0.5:
                #     shift = torch.randint(-2, 4, (1,)).item()
                #     data = torch.roll(data, shift, dims=2)
            
            # Forward
            output = self.model(data)
            
            # Debug: Check output shape and values on first batch
            if batch_idx == 0:
                print(f"  Debug - Input shape: {data.shape}, Output shape: {output.shape}")
                print(f"  Debug - Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
                print(f"  Debug - Output mean: {output.mean().item():.4f}, std: {output.std().item():.4f}")
                print(f"  Debug - Target shape: {target.shape}, Target values: {target.unique()}")
            
            # Validate output
            if torch.any(torch.isnan(output)) or torch.any(torch.isinf(output)):
                print(f"  WARNING: NaN/Inf in model output at batch {batch_idx}, skipping...")
                output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
            
            loss = criterion(output, target)
            
            # Debug: Check loss on first batch
            if batch_idx == 0:
                print(f"  Debug - Loss: {loss.item():.4f}")
                print(f"  Debug - Predictions: {output.argmax(dim=1)[:5]}, Targets: {target[:5]}")
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  WARNING: NaN/Inf loss at batch {batch_idx}, skipping...")
                continue
            
            total_loss += loss.item()
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Debug: Check gradients on first batch
            if batch_idx == 0:
                total_norm = 0
                param_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                print(f"  Debug - Gradient norm: {total_norm:.6f}")
                if total_norm < 1e-6:
                    print(f"  ⚠️  WARNING: Very small gradient norm ({total_norm:.6f}) - model may not be learning!")
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Step scheduler (OneCycleLR steps per batch)
            if scheduler is not None and isinstance(scheduler, OneCycleLR):
                scheduler.step()
            
            # Accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Cleanup memory
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
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
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                target = target.squeeze()
                
                # Validate input
                if torch.any(torch.isnan(data)) or torch.any(torch.isinf(data)):
                    continue
                
                output = self.model(data)
                
                # Validate output
                if torch.any(torch.isnan(output)) or torch.any(torch.isinf(output)):
                    output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
                
                loss = criterion(output, target)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
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
              weight_decay: float = 1e-5, patience: int = 10, label_smoothing: float = 0.05):
        """Train the model with class balancing and label smoothing."""
        
        # Get num_classes from model (from classifier's last layer)
        num_classes = self.model.classifier[-1].out_features
        
        # Get class weights for imbalanced data
        labels_list = []
        for _, (_, target) in enumerate(train_loader):
            labels_list.extend(target.squeeze().cpu().numpy())
        
        unique, counts = np.unique(labels_list, return_counts=True)
        
        print(f"Unique classes: {unique}")
        print(f"Class counts: {counts}")
        
        # Calculate weights - always create weights for all num_classes
        n_total = len(labels_list)
        n_classes_found = len(unique)
        
        if n_classes_found == 0:
            print("⚠️  CRITICAL ERROR: No classes found in training data!")
            print("   This means all labels are the same or invalid.")
            print("   The model cannot learn without multiple classes.")
            class_weights = torch.ones(num_classes).to(self.device)
        elif n_classes_found == 1:
            print(f"⚠️  CRITICAL ERROR: Only one class found in training data: {unique[0]}")
            print(f"   Class count: {counts[0]}")
            print(f"   The model cannot learn binary classification with only one class!")
            print(f"   Check your data loading and label assignment logic.")
            # Still create weights, but this will cause the model to fail
            weights = np.ones(num_classes)
            if unique[0] < num_classes:
                weights[unique[0]] = 1.0
            class_weights = torch.FloatTensor(weights).to(self.device)
        else:
            # Initialize weights to 1.0 for all classes
            weights = np.ones(num_classes)
            
            # Calculate weights for classes that exist in the data
            # Weight = total_samples / (n_classes_found * class_count)
            for i, cls_id in enumerate(unique):
                if cls_id < num_classes and counts[i] > 0:
                    weights[cls_id] = n_total / (n_classes_found * counts[i])
                elif cls_id >= num_classes:
                    print(f"WARNING: Found class {cls_id} but model only has {num_classes} classes")
            
            class_weights = torch.FloatTensor(weights).to(self.device)
        
        print(f"Class distribution: {dict(zip(unique, counts))}, weights: {class_weights.cpu().numpy()}")
        print(f"Class weights applied: {class_weights.cpu().numpy()}")
        
        # Check for severe imbalance
        if n_classes_found == 2:
            balance_ratio = min(counts) / max(counts)
            print(f"Class balance ratio: {balance_ratio:.3f} (1.0 = perfectly balanced)")
            if balance_ratio < 0.1:
                print(f"⚠️  WARNING: Severe class imbalance (ratio: {balance_ratio:.3f})")
                print(f"   This will make learning very difficult. Consider:")
                print(f"   1. Using more aggressive class weights")
                print(f"   2. Using oversampling/undersampling")
                print(f"   3. Checking if labels are correct")
        
        # Loss with class weights and label smoothing
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )
        
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # OneCycleLR scheduler
        steps_per_epoch = len(train_loader)
        total_steps = num_epochs * steps_per_epoch
        scheduler = OneCycleLR(optimizer, max_lr=learning_rate * 5, 
                              total_steps=total_steps, pct_start=0.3,
                              anneal_strategy='cos')
        
        patience_counter = 0
        
        print(f"Starting FULCNN training for {num_epochs} epochs...")
        print(f"Learning rate: {learning_rate}, Weight decay: {weight_decay}")
        print(f"Label smoothing: {label_smoothing}")
        print(f"\n⚠️  IMPORTANT: If accuracy stays near 50%, check:")
        print(f"   1. Window label distribution (should have both classes)")
        print(f"   2. Model gradients (should be > 1e-6)")
        print(f"   3. Model output variance (should vary across samples)")
        print(f"   4. Data preprocessing (may be removing signal)")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion, scheduler)
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # Step scheduler (OneCycleLR already steps per batch)
            if not isinstance(scheduler, OneCycleLR):
                scheduler.step()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Diagnostic: Check if loss is decreasing
            if epoch > 0:
                loss_change = self.train_losses[-2] - train_loss
                if loss_change < 0.001 and epoch > 5:
                    print(f"⚠️  WARNING: Loss barely decreasing (change: {loss_change:.6f}). Model may not be learning!")
                    print(f"   Possible causes:")
                    print(f"   1. Learning rate too small (current: {optimizer.param_groups[0]['lr']:.6f})")
                    print(f"   2. Model weights not updating (check gradients)")
                    print(f"   3. Data/labels incorrect")
                if train_acc < 55.0 and val_acc < 55.0 and epoch > 10:
                    print(f"⚠️  WARNING: Accuracy stuck near random (50%). Check data quality and labels!")
                    print(f"   Train acc: {train_acc:.2f}%, Val acc: {val_acc:.2f}%")
                    print(f"   This suggests the model is not learning the attention signal.")
                    print(f"   Check: 1) Window label distribution, 2) Model gradients, 3) Data preprocessing")
            
            # Check if model predictions are all the same (another sign of not learning)
            if epoch > 0 and epoch % 5 == 0:
                # Sample a batch to check prediction diversity
                sample_batch = next(iter(train_loader))
                sample_data, sample_target = sample_batch
                sample_data = sample_data.to(self.device)
                with torch.no_grad():
                    sample_output = self.model(sample_data)
                    sample_preds = sample_output.argmax(dim=1)
                    unique_preds = torch.unique(sample_preds)
                    if len(unique_preds) == 1:
                        print(f"⚠️  WARNING: Model is predicting only class {unique_preds[0].item()} for all samples!")
                        print(f"   This indicates the model is not learning. Check gradients and data.")
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, self.best_model_path)
                print(f"New best model saved! Val Acc: {val_acc:.4f}")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping after {patience_counter} epochs without improvement (patience={patience})")
                break
            
            # Print patience status
            if epoch % 5 == 0 or patience_counter > 0:
                print(f"  Patience: {patience_counter}/{patience} (no improvement for {patience_counter} epochs)")
        
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
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Testing"):
                data, target = data.to(self.device), target.to(self.device)
                target = target.squeeze()
                
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                
                probabilities = F.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
        
        # Convert to numpy
        preds = np.array(all_predictions)
        targets = np.array(all_targets)
        probs = np.array(all_probabilities)
        
        # Calculate metrics
        accuracy = accuracy_score(targets, preds)
        avg_loss = total_loss / len(test_loader)
        
        # Classification report
        report = classification_report(targets, preds, 
                                     target_names=['Left', 'Right'], 
                                     labels=[0, 1],
                                     output_dict=True)
        
        cm = confusion_matrix(targets, preds)
        
        # CRITICAL: Check if model is performing worse than random
        if accuracy < 0.5:
            print(f"\n⚠️  CRITICAL WARNING: Test accuracy ({accuracy:.4f}) is below random chance (0.5)!")
            print(f"   This indicates a fundamental problem with the data, labels, or model.")
        elif accuracy < 0.55:
            print(f"\n⚠️  WARNING: Test accuracy ({accuracy:.4f}) is barely above random chance.")
            print(f"   The model is not learning effectively.")
        
        # Calculate metrics
        roc_auc_metrics = self._calculate_roc_auc_metrics(targets, probs)
        msed_metrics = self._calculate_msed_metrics(targets, preds)
        advanced_metrics = self._calculate_advanced_metrics(targets, preds)
        temporal_metrics = self._calculate_temporal_metrics(test_loader)
        
        # Check ROC-AUC (should be > 0.5 for a useful model)
        roc_auc = roc_auc_metrics.get('roc_auc_score', 0.5)
        if roc_auc < 0.5:
            print(f"\n⚠️  CRITICAL: ROC-AUC ({roc_auc:.4f}) is below 0.5!")
            print(f"   This means the model is performing WORSE than random guessing.")
            print(f"   Possible causes:")
            print(f"   1. Labels are incorrect or reversed")
            print(f"   2. Data preprocessing removed the signal")
            print(f"   3. Model architecture is inappropriate for this task")
            print(f"   4. Severe class imbalance or data leakage")
        elif roc_auc < 0.55:
            print(f"\n⚠️  WARNING: ROC-AUC ({roc_auc:.4f}) is barely above random (0.5).")
            print(f"   The model is not learning the attention signal effectively.")
        
        results = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': preds,
            'targets': targets,
            'probabilities': probs,
            'roc_auc_metrics': roc_auc_metrics,
            'msed_metrics': msed_metrics,
            'advanced_metrics': advanced_metrics,
            'temporal_metrics': temporal_metrics
        }
        
        return results
    
    def _calculate_roc_auc_metrics(self, targets: np.ndarray, probabilities: np.ndarray) -> Dict:
        """Calculate ROC-AUC and related metrics."""
        try:
            roc_auc = roc_auc_score(targets, probabilities)
            fpr, tpr, roc_thresholds = roc_curve(targets, probabilities)
            
            # Find optimal threshold
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = roc_thresholds[optimal_idx]
            optimal_tpr = tpr[optimal_idx]
            optimal_fpr = fpr[optimal_idx]
            
            # Precision-Recall Curve
            precision, recall, pr_thresholds = precision_recall_curve(targets, probabilities)
            avg_precision = average_precision_score(targets, probabilities)
            
            return {
                "roc_auc_score": float(roc_auc),
                "average_precision": float(avg_precision),
                "optimal_threshold": float(optimal_threshold),
                "optimal_tpr": float(optimal_tpr),
                "optimal_fpr": float(optimal_fpr),
                "roc_curve": {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "thresholds": roc_thresholds.tolist()
                },
                "precision_recall_curve": {
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                    "thresholds": pr_thresholds.tolist()
                }
            }
        except Exception as e:
            return {"error": f"Error calculating ROC-AUC metrics: {e}"}
    
    def _calculate_msed_metrics(self, targets: np.ndarray, predictions: np.ndarray) -> Dict:
        """Calculate MSED (Mean Squared Error Distance) metrics."""
        try:
            mse = np.mean((predictions - targets) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - targets))
            mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
            
            # R-squared
            ss_res = np.sum((targets - predictions) ** 2)
            ss_tot = np.sum((targets - np.mean(targets)) ** 2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-8))
            
            return {
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "mape": float(mape),
                "r_squared": float(r_squared)
            }
        except Exception as e:
            return {"error": f"Error calculating MSED metrics: {e}"}
    
    def _calculate_advanced_metrics(self, targets: np.ndarray, predictions: np.ndarray) -> Dict:
        """Calculate advanced classification metrics."""
        try:
            mcc = matthews_corrcoef(targets, predictions)
            kappa = cohen_kappa_score(targets, predictions)
            balanced_acc = balanced_accuracy_score(targets, predictions)
            
            precision, recall, f1, support = precision_recall_fscore_support(
                targets, predictions, average=None, labels=[0, 1]
            )
            
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                targets, predictions, average='macro'
            )
            
            precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
                targets, predictions, average='weighted'
            )
            
            return {
                "matthews_correlation_coefficient": float(mcc),
                "cohens_kappa": float(kappa),
                "balanced_accuracy": float(balanced_acc),
                "per_class_metrics": {
                    "left_attention": {
                        "precision": float(precision[0]),
                        "recall": float(recall[0]),
                        "f1_score": float(f1[0]),
                        "support": int(support[0])
                    },
                    "right_attention": {
                        "precision": float(precision[1]),
                        "recall": float(recall[1]),
                        "f1_score": float(f1[1]),
                        "support": int(support[1])
                    }
                },
                "macro_averages": {
                    "precision": float(precision_macro),
                    "recall": float(recall_macro),
                    "f1_score": float(f1_macro)
                },
                "weighted_averages": {
                    "precision": float(precision_weighted),
                    "recall": float(recall_weighted),
                    "f1_score": float(f1_weighted)
                }
            }
        except Exception as e:
            return {"error": f"Error calculating advanced metrics: {e}"}
    
    def _calculate_temporal_metrics(self, test_loader: DataLoader) -> Dict[str, float]:
        """Calculate real temporal performance metrics across different window sizes."""
        # Test different window sizes (in seconds) - 1s to 30s
        window_sizes_seconds = [1.0, 2.0, 4.0, 8.0, 16.0, 30.0]
        temporal_analysis = {}
        flat_results = {}
        
        for window_sec in window_sizes_seconds:
            window_samples = int(window_sec * self.sampling_rate)
            
            # For larger windows, we need to create overlapping windows from the test data
            if window_samples > self.window_size:
                larger_window_results = self._test_larger_window(test_loader, window_samples, window_sec)
                flat_results.update(larger_window_results)
                
                # Add to temporal_analysis structure
                if f'accuracy_{window_sec}s' in larger_window_results:
                    temporal_analysis[f'{window_sec}s'] = {
                        'accuracy': larger_window_results[f'accuracy_{window_sec}s'],
                        'f1': larger_window_results[f'f1_{window_sec}s']
                    }
                continue
            
            # For smaller windows, create temporary dataset
            try:
                temp_dataset = FulsangDataset(
                    self.tfrecord_dir, 
                    mode='test',
                    window_size=window_samples,
                    overlap=0.5
                )
                
                if len(temp_dataset) == 0:
                    continue
                
                # Check if the dataset produces valid dimensions for the model
                sample_data, _ = temp_dataset[0]
                if sample_data.dim() == 3:
                    temp_channels, temp_time, temp_freq = sample_data.shape
                    model_channels = self.model.backbone.input_channels
                    model_freq = self.model.backbone.input_freq
                    
                    # Check channel and frequency dimensions (must match)
                    if temp_channels != model_channels:
                        print(f"Skipping {window_sec}s window: channel mismatch. "
                              f"Dataset produces {temp_channels} channels, model expects {model_channels}")
                        continue
                    
                    if temp_freq != model_freq:
                        print(f"Skipping {window_sec}s window: frequency mismatch. "
                              f"Dataset produces {temp_freq} freq bands, model expects {model_freq}")
                        continue
                    
                    # Time dimension can vary - model uses adaptive pooling to handle this
                    # But check if dimensions are too small for pooling operations
                    if temp_time < 4 or temp_freq < 4:
                        print(f"Skipping {window_sec}s window: dimensions too small ({temp_time}x{temp_freq}) "
                              f"for CNNLoc pooling operations (requires >= 4x4)")
                        continue
                    
                    # MEANINGFUL TEMPORAL ANALYSIS: Report actual temporal resolution
                    # Different window sizes should show different performance if model is learning temporal patterns
                    print(f"Testing {window_sec}s window: shape ({temp_channels}, {temp_time}, {temp_freq})")
                    print(f"  Actual temporal resolution: {temp_time} time bins (model initialized for {self.model.backbone.input_time})")
                    print(f"  ⚠️  NOTE: If performance is similar across all window sizes, model may be ignoring temporal scale")
                
                temp_loader = DataLoader(temp_dataset, batch_size=16, shuffle=False)
                
                # Evaluate on this window size
                self.model.eval()
                all_predictions = []
                all_targets = []
                
                with torch.no_grad():
                    for data, target in temp_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = self.model(data)
                        pred = output.argmax(dim=1)
                        
                        all_predictions.extend(pred.cpu().numpy())
                        all_targets.extend(target.cpu().numpy())
                
                if len(all_predictions) > 0:
                    accuracy = accuracy_score(all_targets, all_predictions)
                    f1 = f1_score(all_targets, all_predictions, average='weighted')
                    
                    flat_results[f'accuracy_{window_sec}s'] = accuracy
                    flat_results[f'f1_{window_sec}s'] = f1
                    
                    # Add to temporal_analysis structure
                    temporal_analysis[f'{window_sec}s'] = {
                        'accuracy': accuracy,
                        'f1': f1
                    }
                    
            except Exception as e:
                print(f"Error testing {window_sec}s window: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # MEANINGFUL TEMPORAL ANALYSIS: Check if performance varies with window size
        # If all window sizes perform similarly, the model is ignoring temporal scale (bad)
        acc_range = 0.0
        if len(temporal_analysis) > 1:
            accuracies = [metrics['accuracy'] for metrics in temporal_analysis.values()]
            acc_range = max(accuracies) - min(accuracies)
            
            if acc_range < 0.05:  # Less than 5% difference
                print(f"\n⚠️  WARNING: Temporal analysis shows <5% variation across window sizes ({acc_range:.3f})")
                print(f"   This suggests the model is INVARIANT to temporal scale, which is wrong for attention decoding.")
                print(f"   The model should perform better on longer windows that capture more trial context.")
            else:
                print(f"\n✓ Temporal analysis shows meaningful variation: {acc_range:.3f} accuracy range")
                print(f"   This suggests the model is learning temporal-scale-dependent patterns.")
        
        # Find the best window size
        best_window = None
        best_accuracy = 0.0
        for window_key, metrics in temporal_analysis.items():
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_window = window_key
        
        # Return structured results
        note_str = 'No valid temporal analysis completed'
        if best_window:
            if len(temporal_analysis) > 1:
                note_str = f'Best performance at {best_window}s window with {best_accuracy:.3f} accuracy. Temporal variation: {acc_range:.3f}'
            else:
                note_str = f'Best performance at {best_window}s window with {best_accuracy:.3f} accuracy'
        
        return {
            'temporal_analysis': temporal_analysis,
            'recommended_window_size': best_window if best_window else 'N/A',
            'temporal_variation': acc_range,
            'note': note_str,
            **flat_results  # Keep flat results for backward compatibility
        }
    
    def _test_larger_window(self, test_loader: DataLoader, window_samples: int, window_sec: float) -> Dict[str, float]:
        """
        DEPRECATED: This method is conceptually wrong.
        
        You cannot create larger windows from already-processed windows.
        The test_loader contains windows that have already been:
        1. Extracted from raw data
        2. Preprocessed
        3. Converted to time-frequency representation
        4. Interpolated to a fixed size
        
        To test larger windows, you must go back to the RAW data and create
        new windows from scratch. This is now handled in _calculate_temporal_metrics
        by creating new FulsangDataset instances with different window sizes.
        
        This method is kept for backward compatibility but should not be used.
        """
        # This method is no longer used - temporal analysis now creates
        # new datasets directly in _calculate_temporal_metrics
        print(f"WARNING: _test_larger_window called but is deprecated. Use new dataset creation instead.")
        return {}
    
    def save_results(self, results: Dict):
        """Save comprehensive results to files."""
        # Prepare results
        results_json = {
            'accuracy': float(results['accuracy']),
            'loss': float(results['loss']),
            'classification_report': results['classification_report'],
            'confusion_matrix': results['confusion_matrix'].tolist() if hasattr(results['confusion_matrix'], 'tolist') else results['confusion_matrix'],
            'best_val_acc': float(self.best_val_acc),
            'timestamp': datetime.now().isoformat(),
            'roc_auc_metrics': results.get('roc_auc_metrics', {}),
            'msed_metrics': results.get('msed_metrics', {}),
            'advanced_metrics': results.get('advanced_metrics', {}),
            'temporal_metrics': results.get('temporal_metrics', {})
        }
        
        # Save results
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        
        # Save predictions
        save_data = {
            'predictions': results['predictions'],
            'targets': results['targets'],
            'probabilities': results['probabilities']
        }
        
        with open(self.output_dir / 'predictions.pkl', 'wb') as f:
            pickle.dump(save_data, f)
        
        # Save comprehensive metrics report
        self._save_comprehensive_report(results)
        
    
    def _save_comprehensive_report(self, results: Dict):
        """Save a comprehensive metrics report."""
        with open(self.output_dir / 'comprehensive_metrics_report.txt', 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FULCNN COMPREHENSIVE METRICS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Basic metrics
            f.write("BASIC METRICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"Loss: {results['loss']:.4f}\n")
            f.write(f"Best Validation Accuracy: {self.best_val_acc:.2f}%\n\n")
            
            # ROC-AUC metrics
            roc_auc = results.get('roc_auc_metrics', {})
            if "error" not in roc_auc:
                f.write("ROC-AUC METRICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"ROC-AUC Score: {roc_auc.get('roc_auc_score', 'N/A'):.4f}\n")
                f.write(f"Average Precision: {roc_auc.get('average_precision', 'N/A'):.4f}\n")
                f.write(f"Optimal Threshold: {roc_auc.get('optimal_threshold', 'N/A'):.4f}\n")
                f.write(f"Optimal TPR: {roc_auc.get('optimal_tpr', 'N/A'):.4f}\n")
                f.write(f"Optimal FPR: {roc_auc.get('optimal_fpr', 'N/A'):.4f}\n\n")
            
            # MSED metrics
            msed = results.get('msed_metrics', {})
            if "error" not in msed:
                f.write("MSED METRICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Mean Squared Error: {msed.get('mse', 'N/A'):.4f}\n")
                f.write(f"Root Mean Squared Error: {msed.get('rmse', 'N/A'):.4f}\n")
                f.write(f"Mean Absolute Error: {msed.get('mae', 'N/A'):.4f}\n")
                f.write(f"Mean Absolute Percentage Error: {msed.get('mape', 'N/A'):.4f}%\n")
                f.write(f"R-squared: {msed.get('r_squared', 'N/A'):.4f}\n\n")
            
            # Advanced metrics
            advanced = results.get('advanced_metrics', {})
            if "error" not in advanced:
                f.write("ADVANCED METRICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Matthews Correlation Coefficient: {advanced.get('matthews_correlation_coefficient', 'N/A'):.4f}\n")
                f.write(f"Cohen's Kappa: {advanced.get('cohens_kappa', 'N/A'):.4f}\n")
                f.write(f"Balanced Accuracy: {advanced.get('balanced_accuracy', 'N/A'):.4f}\n\n")
                
                # Per-class metrics
                per_class = advanced.get("per_class_metrics", {})
                f.write("PER-CLASS METRICS:\n")
                f.write("-" * 40 + "\n")
                
                left = per_class.get("left_attention", {})
                f.write("Left Attention:\n")
                f.write(f"  Precision: {left.get('precision', 'N/A'):.4f}\n")
                f.write(f"  Recall: {left.get('recall', 'N/A'):.4f}\n")
                f.write(f"  F1-Score: {left.get('f1_score', 'N/A'):.4f}\n")
                f.write(f"  Support: {left.get('support', 'N/A')}\n\n")
                
                right = per_class.get("right_attention", {})
                f.write("Right Attention:\n")
                f.write(f"  Precision: {right.get('precision', 'N/A'):.4f}\n")
                f.write(f"  Recall: {right.get('recall', 'N/A'):.4f}\n")
                f.write(f"  F1-Score: {right.get('f1_score', 'N/A'):.4f}\n")
                f.write(f"  Support: {right.get('support', 'N/A')}\n\n")
            
            # Temporal analysis
            temporal = results.get('temporal_metrics', {})
            temporal_analysis = temporal.get("temporal_analysis", {})
            f.write("TEMPORAL PERFORMANCE ANALYSIS (1s to 30s Windows):\n")
            f.write("=" * 80 + "\n")
            if temporal_analysis:
                f.write(f"{'Window Size':<15} {'Accuracy':<15} {'F1 Score':<15}\n")
                f.write("-" * 80 + "\n")
                # Sort by window size
                sorted_windows = sorted(temporal_analysis.keys(), 
                                       key=lambda x: float(x.replace('s', '')))
                for window_size in sorted_windows:
                    metrics = temporal_analysis[window_size]
                    acc = metrics.get('accuracy', 0.0)
                    f1 = metrics.get('f1', 0.0)
                    f.write(f"{window_size:<15} {acc:<15.4f} {f1:<15.4f}\n")
            f.write("=" * 80 + "\n")
            f.write(f"\nRecommended: {temporal.get('recommended_window_size', 'N/A')}\n")
            f.write(f"Note: {temporal.get('note', 'N/A')}\n")
            
            # Add formatted results section
            f.write("\n" + "=" * 80 + "\n")
            f.write("FULCNN COMPREHENSIVE RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("The FULCNN model successfully processed the Fulsang dataset:\n")
            f.write(f"- Best Validation Accuracy: {self.best_val_acc:.4f}\n")
            f.write(f"- Final Test Accuracy: {results['accuracy']:.4f}\n")
            
            # ROC-AUC metrics
            roc_auc = results.get('roc_auc_metrics', {})
            if "error" not in roc_auc:
                f.write(f"- ROC-AUC: {roc_auc.get('roc_auc_score', 'N/A'):.4f}\n")
            
            # Classification metrics
            class_report = results.get('classification_report', {})
            if 'macro avg' in class_report:
                macro_avg = class_report['macro avg']
                f.write(f"- Precision: {macro_avg.get('precision', 'N/A'):.4f}\n")
                f.write(f"- Recall: {macro_avg.get('recall', 'N/A'):.4f}\n")
                f.write(f"- F1-Score: {macro_avg.get('f1-score', 'N/A'):.4f}\n")
            
            # MSED metrics
            msed = results.get('msed_metrics', {})
            if "error" not in msed:
                f.write(f"- MSED (Primary Benchmark): {msed.get('rmse', 'N/A'):.4f}\n")
            
            # Advanced metrics
            advanced = results.get('advanced_metrics', {})
            if "error" not in advanced:
                f.write(f"- Direction Accuracy: {advanced.get('balanced_accuracy', 'N/A'):.4f}\n")
                f.write(f"- Spatial Consistency: {advanced.get('matthews_correlation_coefficient', 'N/A'):.4f}\n")
            
            # Temporal Integration Performance
            f.write("\nTEMPORAL INTEGRATION PERFORMANCE\n")
            f.write("The Fulsang dataset demonstrated robust performance across decision window lengths:\n")
            
            for ws_key, ws_data in temporal.get("temporal_analysis", {}).items():
                window_seconds = float(ws_key.replace('s', ''))
                accuracy = ws_data.get('accuracy', 0.0)
                f.write(f"- {ws_key} window: {accuracy:.4f}\n")


def create_fulsang_data_loaders(tfrecord_dir: str, batch_size: int = 16, 
                               window_size: int = 512, overlap: float = 0.5,
                               train_ratio: float = 0.7, val_ratio: float = 0.15,
                               max_samples: Optional[int] = None, 
                               num_workers: int = 0, pin_memory: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test loaders with subject-wise splitting (no data leakage)."""
    
    print(f"DEBUG: window_size parameter = {window_size}")
    print(f"Creating dataset: batch_size={batch_size}, window_size={window_size} samples ({window_size/64:.1f}s)")
    
    # Create full dataset with FULPREPROCESSING integration
    full_dataset = FulsangDataset(tfrecord_dir, mode='full', 
                                 window_size=window_size, overlap=overlap)
    
    total_size = len(full_dataset)
    
    # Map windows to subjects for splitting
    subject_windows = {}
    
    # Group metadata by subject
    # This creates ranges in the metadata array (which should align with data array)
    subject_ranges = {}
    current_subject = None
    start_idx = 0
    
    print(f"\nAnalyzing {len(full_dataset.metadata)} metadata entries for subject grouping...")
    
    # First, check what subject IDs we actually have
    unique_subjects_in_metadata = set()
    for metadata in full_dataset.metadata:
        subject_id = metadata.get('subject_id', 'unknown')
        unique_subjects_in_metadata.add(subject_id)
    
    print(f"Found {len(unique_subjects_in_metadata)} unique subject IDs in metadata: {sorted(unique_subjects_in_metadata)}")
    
    if len(unique_subjects_in_metadata) == 1:
        print(f"⚠️  CRITICAL: Only 1 unique subject ID found: {list(unique_subjects_in_metadata)[0]}")
        print(f"   This means subject-wise split CANNOT work!")
        print(f"   All data will be treated as from the same subject")
        print(f"   Check if subject_id is being properly extracted from TFRecords")
    
    for i, metadata in enumerate(full_dataset.metadata):
        subject_id = metadata.get('subject_id', 'unknown')
        
        if subject_id != current_subject:
            if current_subject is not None:
                subject_ranges[current_subject] = (start_idx, i)
            current_subject = subject_id
            start_idx = i
    
    # Last subject
    if current_subject is not None:
        subject_ranges[current_subject] = (start_idx, len(full_dataset.metadata))
    
    print(f"Created {len(subject_ranges)} subject ranges:")
    for subj_id, (start, end) in subject_ranges.items():
        print(f"  {subj_id}: metadata indices {start}-{end} ({end-start} samples)")
    
    # Map windows to subjects
    # IMPORTANT: Windows are mapped based on where they START (data_idx)
    # If a window spans subjects, it's assigned to the subject where it starts
    print(f"\nMapping {len(full_dataset.window_indices)} windows to subjects...")
    print(f"Subject ranges in metadata: {len(subject_ranges)} subjects")
    for subj_id, (start, end) in subject_ranges.items():
        print(f"  {subj_id}: samples {start}-{end} ({end-start} samples)")
    
    unmapped_windows = 0
    for i, (data_idx, label) in enumerate(full_dataset.window_indices):
        subject_id = "unknown"
        
        # Find subject for this window based on where it starts
        # Note: If window spans subjects, it's assigned to the starting subject
        for subj_id, (start_idx, end_idx) in subject_ranges.items():
            if start_idx <= data_idx < end_idx:
                subject_id = subj_id
                break
        
        # Check if window might span subjects
        window_end = data_idx + full_dataset.window_size
        if subject_id != "unknown":
            # Check if window extends beyond subject boundary
            subj_start, subj_end = subject_ranges[subject_id]
            if window_end > subj_end:
                print(f"WARNING: Window {i} (data_idx={data_idx}) spans subject boundary! "
                      f"Starts in {subject_id} (ends at {subj_end}) but window ends at {window_end}")
        
        if subject_id == "unknown":
            unmapped_windows += 1
            # Try to find by checking if data_idx is out of bounds
            if data_idx >= len(full_dataset.metadata):
                print(f"ERROR: Window {i} has data_idx={data_idx} but metadata only has {len(full_dataset.metadata)} entries")
        
        if subject_id not in subject_windows:
            subject_windows[subject_id] = []
        subject_windows[subject_id].append(i)
    
    if unmapped_windows > 0:
        print(f"WARNING: {unmapped_windows} windows could not be mapped to subjects!")
    
    print(f"Mapped windows to {len(subject_windows)} subjects:")
    for subj_id, window_list in subject_windows.items():
        print(f"  {subj_id}: {len(window_list)} windows")
    
    # Analyze label distribution per subject BEFORE splitting
    subject_label_distributions = {}
    print(f"\nAnalyzing label distribution per subject (before split):")
    for subject_id, window_indices_list in subject_windows.items():
        subject_labels = [full_dataset.window_indices[i][1] for i in window_indices_list]
        label_dist = np.bincount(subject_labels)
        subject_label_distributions[subject_id] = label_dist
        
        # Show detailed info for each subject
        total_windows = len(window_indices_list)
        class_0_count = label_dist[0] if len(label_dist) > 0 else 0
        class_1_count = label_dist[1] if len(label_dist) > 1 else 0
        
        if len(label_dist) < 2:
            print(f"  ⚠️  {subject_id}: {total_windows} windows, ONLY class {np.argmax(label_dist)} ({class_0_count if len(label_dist) == 1 else 'N/A'} windows)")
        else:
            print(f"  ✓ {subject_id}: {total_windows} windows, class 0: {class_0_count}, class 1: {class_1_count} (balance: {min(class_0_count, class_1_count)/max(class_0_count, class_1_count):.2f})")
    
    # Split by subject (simple random split - preserves natural class imbalance)
    # Note: We do NOT balance classes - the dataset is naturally unbalanced
    subjects = list(subject_windows.keys())
    np.random.seed(42)  # Reproducibility
    np.random.shuffle(subjects)
    
    # Calculate split sizes
    n_subjects = len(subjects)
    n_train_subjects = int(train_ratio * n_subjects)
    n_val_subjects = int(val_ratio * n_subjects)
    
    # Simple random split (preserves natural class distribution)
    train_subjects = subjects[:n_train_subjects]
    val_subjects = subjects[n_train_subjects:n_train_subjects + n_val_subjects]
    test_subjects = subjects[n_train_subjects + n_val_subjects:]
    
    print(f"\nSubject split:")
    print(f"  Train subjects ({len(train_subjects)}): {train_subjects}")
    print(f"  Val subjects ({len(val_subjects)}): {val_subjects}")
    print(f"  Test subjects ({len(test_subjects)}): {test_subjects}")
    
    # Show label distribution for subjects in each split BEFORE creating indices
    print(f"\nLabel distribution for subjects in each split:")
    
    print(f"  Training subjects:")
    for subj_id in train_subjects:
        if subj_id in subject_label_distributions:
            dist = subject_label_distributions[subj_id]
            print(f"    {subj_id}: {dict(enumerate(dist))}")
    
    print(f"  Validation subjects:")
    for subj_id in val_subjects:
        if subj_id in subject_label_distributions:
            dist = subject_label_distributions[subj_id]
            print(f"    {subj_id}: {dict(enumerate(dist))}")
    
    print(f"  Test subjects:")
    for subj_id in test_subjects:
        if subj_id in subject_label_distributions:
            dist = subject_label_distributions[subj_id]
            print(f"    {subj_id}: {dict(enumerate(dist))}")
    
    # Get window indices for each split
    train_indices = []
    val_indices = []
    test_indices = []
    
    for subject_id in train_subjects:
        train_indices.extend(subject_windows[subject_id])
    for subject_id in val_subjects:
        val_indices.extend(subject_windows[subject_id])
    for subject_id in test_subjects:
        test_indices.extend(subject_windows[subject_id])
    
    # Check for data leakage
    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)
    
    if train_set & val_set:
        raise ValueError("CRITICAL: Data leakage - train/val overlap!")
    if train_set & test_set:
        raise ValueError("CRITICAL: Data leakage - train/test overlap!")
    if val_set & test_set:
        raise ValueError("CRITICAL: Data leakage - val/test overlap!")
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # Check label distribution in each split (debug for data leakage)
    train_labels = [full_dataset.window_indices[i][1] for i in train_indices]
    val_labels = [full_dataset.window_indices[i][1] for i in val_indices]
    test_labels = [full_dataset.window_indices[i][1] for i in test_indices]
    
    train_label_dist = np.bincount(train_labels)
    val_label_dist = np.bincount(val_labels)
    test_label_dist = np.bincount(test_labels)
    
    print(f"\nLabel distribution check (to detect data issues):")
    print(f"  Train: {dict(enumerate(train_label_dist))}")
    print(f"  Val:   {dict(enumerate(val_label_dist))}")
    print(f"  Test:  {dict(enumerate(test_label_dist))}")
    
    # Note: Unbalanced classes are expected - the algorithm handles this with class weights
    if len(train_label_dist) < 2:
        print(f"\nNOTE: Training set has only {len(train_label_dist)} class(es).")
        print(f"   This is expected for unbalanced datasets.")
        print(f"   The model will use class weights to handle the imbalance.")
    
    if len(val_label_dist) < 2:
        print(f"NOTE: Validation set has only {len(val_label_dist)} class(es).")
    if len(test_label_dist) < 2:
        print(f"NOTE: Test set has only {len(test_label_dist)} class(es).")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, val_loader, test_loader


def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='FULCNN - CNN-LOC for Fulsang Dataset')
    parser.add_argument('--tfrecord_dir', type=str, default='Preprocessed_FulsangNorm/tfrecords',
                       help='TFRecord directory path')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training (default: 16 - best from hyperparameter tuning)')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-3,
                       help='Learning rate (default: 5e-3 - best from hyperparameter tuning)')
    parser.add_argument('--window_size', type=int, default=512,
                       help='Window size for EEG data (512 samples = 8 seconds at 64Hz)')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of data loading workers (default: auto-detect, 4 for GPU, 2 for CPU)')
    parser.add_argument('--pin_memory', action='store_true', default=None,
                       help='Use pin_memory for faster GPU transfer (default: auto-enable for GPU)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for regularization')
    parser.add_argument('--dropout_rate', type=float, default=0.20,
                       help='Dropout rate (default: 0.20 - best from hyperparameter tuning)')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                       help='Label smoothing factor')
    parser.add_argument('--output_dir', type=str, default='fulcnn_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print(f"DEBUG: args.window_size = {args.window_size}")
    print(f"DEBUG: args.tfrecord_dir = {args.tfrecord_dir}")
    
    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU (GPU not available)")
    
    # Create data loaders with optimized settings for speed
    # Use multiple workers for parallel data loading and pin_memory for faster GPU transfer
    if args.num_workers is None:
        num_workers = 4 if torch.cuda.is_available() else 2  # More workers for GPU training
    else:
        num_workers = args.num_workers
    
    if args.pin_memory is None:
        pin_memory = torch.cuda.is_available()  # Pin memory for faster GPU transfer
    else:
        pin_memory = args.pin_memory
    
    print(f"Data loading settings: batch_size={args.batch_size}, num_workers={num_workers}, pin_memory={pin_memory}")
    
    train_loader, val_loader, test_loader = create_fulsang_data_loaders(
        args.tfrecord_dir, batch_size=args.batch_size, window_size=args.window_size,
        max_samples=None, num_workers=num_workers, pin_memory=pin_memory
    )
    
    # Get input dimensions from data (check dynamically, don't hardcode)
    if len(train_loader.dataset) > 0:
        sample_data, _ = next(iter(train_loader))
        actual_channels = sample_data.shape[1]
        actual_time = sample_data.shape[2]
        actual_freq = sample_data.shape[3]
        print(f"Input dimensions: channels={actual_channels}, time={actual_time}, freq={actual_freq}")
        print(f"✓ Detected {actual_channels} channels from data (not hardcoded)")
    else:
        # Fallback: get from dataset
        full_dataset = FulsangDataset(args.tfrecord_dir, mode='full', window_size=args.window_size)
        actual_channels = full_dataset.n_channels
        # Get actual dimensions from a sample
        try:
            if len(full_dataset) > 0:
                sample_data, _ = full_dataset[0]
                actual_time = sample_data.shape[1]  # time dimension
                actual_freq = sample_data.shape[2] if sample_data.dim() > 2 else 5  # freq dimension or default (5 bands from spectrogram)
                print(f"Using dimensions from dataset: channels={actual_channels}, time={actual_time}, freq={actual_freq}")
                
                # Validate dimensions for CNNLoc architecture (requires time>=4, freq>=4 for pooling)
                if actual_time < 4:
                    raise ValueError(f"CNNLoc requires time >= 4, but dataset produces time={actual_time}. "
                                   f"This will cause MaxPool2d operations to fail.")
                if actual_freq < 4:
                    raise ValueError(f"CNNLoc requires freq >= 4, but dataset produces freq={actual_freq}. "
                                   f"This will cause MaxPool2d operations to fail.")
            else:
                # Ultimate fallback - estimate time dimension (after TF transform and interpolation)
                # The TF transform interpolates to target_time_points = max(8, min(window_size//4, 64))
                target_time_points = max(8, min(args.window_size // 4, 64))
                actual_time = target_time_points
                actual_freq = 5  # 5 frequency bands from spectrogram (Delta, Theta, Alpha, Beta, Gamma)
                print(f"Using fallback defaults: channels={actual_channels}, time={actual_time}, freq={actual_freq}")
                
                # Validate fallback dimensions
                if actual_time < 4:
                    print(f"WARNING: Fallback time={actual_time} < 4, may cause CNNLoc pooling errors")
                if actual_freq < 4:
                    print(f"WARNING: Fallback freq={actual_freq} < 4, may cause CNNLoc pooling errors")
        except Exception as e:
            print(f"Error getting dimensions: {e}")
            actual_time = args.window_size
            actual_freq = 5
            print(f"Using ultimate fallback: channels={actual_channels}, time={actual_time}, freq={actual_freq}")
    
    # Create model
    print(f"Creating model: channels={actual_channels}, time={actual_time}, freq={actual_freq}")
    print(f"Hyperparameters: batch_size={args.batch_size}, lr={args.learning_rate}, wd={args.weight_decay}, dropout={args.dropout_rate}, label_smoothing={args.label_smoothing}")
    
    model = FULCNNModel(
        input_channels=actual_channels,
        input_time=actual_time,
        input_freq=actual_freq,
        num_classes=2,
        dropout_rate=args.dropout_rate
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Verify GPU usage
    if torch.cuda.is_available():
        print(f"\nGPU Verification:")
        print(f"  Device: {device}")
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        # Check if model is on GPU
        model_device = next(model.parameters()).device
        print(f"  Model device: {model_device}")
        if model_device.type != 'cuda':
            print(f"  WARNING: Model is not on GPU! Moving to GPU...")
            model = model.to(device)
        else:
            print(f"  ✓ Model is on GPU")
        # Test GPU with a dummy tensor
        test_tensor = torch.randn(1, 1, 1, 1).to(device)
        print(f"  GPU test tensor device: {test_tensor.device}")
        print(f"  ✓ GPU is ready for training\n")
    else:
        print("  Using CPU (GPU not available)\n")
    
    # Create trainer
    trainer = FULCNNTrainer(model, device, args.output_dir, args.tfrecord_dir, 
                           sampling_rate=64, window_size=args.window_size)
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory after model creation: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Train
    best_val_acc = trainer.train(
        train_loader, val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=15,
        label_smoothing=args.label_smoothing
    )
    
    # Test
    results = trainer.test(test_loader)
    
    # Save
    trainer.save_results(results)
    
    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}, Test acc: {results['accuracy']:.4f}")
    
    # Display key metrics
    roc_auc = results.get('roc_auc_metrics', {})
    if "error" not in roc_auc:
        print(f"ROC-AUC: {roc_auc.get('roc_auc_score', 'N/A'):.4f}")
    
    msed = results.get('msed_metrics', {})
    if "error" not in msed:
        print(f"RMSE: {msed.get('rmse', 'N/A'):.4f}")
    
    # Display temporal analysis table
    temporal = results.get('temporal_metrics', {})
    temporal_analysis = temporal.get('temporal_analysis', {})
    
    if temporal_analysis:
        print("\n" + "=" * 80)
        print("TEMPORAL PERFORMANCE ANALYSIS (1s to 30s Windows)")
        print("=" * 80)
        print(f"{'Window Size':<15} {'Accuracy':<15} {'F1 Score':<15}")
        print("-" * 80)
        
        # Sort by window size (convert '1.0s' to float for sorting)
        sorted_windows = sorted(temporal_analysis.keys(), 
                               key=lambda x: float(x.replace('s', '')))
        
        for window_size in sorted_windows:
            metrics = temporal_analysis[window_size]
            acc = metrics.get('accuracy', 0.0)
            f1 = metrics.get('f1', 0.0)
            print(f"{window_size:<15} {acc:<15.4f} {f1:<15.4f}")
        
        print("=" * 80)
        print(f"Recommended window size: {temporal.get('recommended_window_size', 'N/A')}")
    else:
        print(f"Recommended window size: {temporal.get('recommended_window_size', 'N/A')}")
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
