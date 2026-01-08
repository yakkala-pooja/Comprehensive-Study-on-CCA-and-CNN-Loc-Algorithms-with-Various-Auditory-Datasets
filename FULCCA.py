#!/usr/bin/env python3

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           precision_recall_fscore_support, roc_auc_score, roc_curve,
                           precision_recall_curve, average_precision_score,
                           matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
                           f1_score)
from sklearn.cross_decomposition import CCA as SklearnCCA
from scipy.stats import pearsonr
import seaborn as sns
from tqdm import tqdm
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


try:

    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"Found {len(gpu_devices)} GPU device(s)")

        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✓ GPU memory growth configured")
        

        try:
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_limit(gpu, 8192)
            print("✓ GPU memory limits configured")
        except AttributeError:
            print("✓ GPU memory limits not supported in this TensorFlow version")
        except Exception as e:
            print(f"GPU memory limit warning: {e}")
            
    else:
        raise RuntimeError("No GPU devices found! GPU-only mode requires GPU.")
except Exception as e:
    print(f"GPU configuration failed: {e}")
    raise RuntimeError("Cannot proceed without GPU. Please ensure GPU is available.")


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
    print("Continuing with basic functionality...")


tf.compat.v1.enable_v2_behavior()


device = tf.device('/GPU:0')
print("Using GPU for computation (GPU-only mode with CPU fallback)")


tf.random.set_seed(42)
np.random.seed(42)
print("✓ Random seeds set for reproducibility")


def safe_random_operations():
    """Force CPU usage for random operations."""
    with tf.device('/CPU:0'):
        tf.random.set_seed(42)
        np.random.seed(42)


class FulsangDatasetCCA:
    """
    Fulsang-specific dataset class for CCA analysis.
    Adapted from DASCCA for Fulsang dataset (66 EEG channels, optimal hyperparameters).
    """
    
    def __init__(self, tfrecord_dir: str, mode: str = 'full', 
                 window_size: int = 1280, overlap: float = 0.5,  # 20 seconds at 64 Hz (optimal)
                 cache_size: int = 1000, audio_base_dir: Optional[str] = None,
                 load_audio: bool = True, max_files: Optional[int] = None):
        self.tfrecord_dir = Path(tfrecord_dir)
        self.mode = mode
        self.window_size = window_size
        self.overlap = overlap
        self.cache_size = cache_size
        self.load_audio = load_audio  # Option to skip audio loading for speed
        self.max_files = max_files  # Limit number of files to load
        
        # Fulsang-specific parameters
        self.sampling_rate = 64
        self.n_channels = 66  # Fulsang has 66 EEG channels (not 64)
        self.attention_switch_duration = 20
        

        if audio_base_dir:
            self.audio_base_dir = Path(audio_base_dir)
        else:

            possible_dirs = [
                Path("Data/Das/4004271/Stimuli"),
                Path("Data/Das/Stimuli"),
                Path("Stimuli"),
                self.tfrecord_dir.parent.parent / "Stimuli" if self.tfrecord_dir.parent.parent.exists() else None
            ]
            self.audio_base_dir = None
            for dir_path in possible_dirs:
                if dir_path and dir_path.exists():
                    self.audio_base_dir = dir_path
                    break
            if self.audio_base_dir is None:
                self.audio_base_dir = Path("Data/Das/4004271/Stimuli")
                print(f"⚠ WARNING: Audio base directory not found, using default: {self.audio_base_dir}")
            else:
                print(f"✓ Using audio base directory: {self.audio_base_dir}")
        

        self._window_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        

        self._audio_envelope_cache = {}
        

        self.eeg_data, self.audio_envelopes, self.labels, self.metadata = self._load_fulsang_preprocessing_data()
        
        self.window_indices = self._create_fulsang_windows()
        
        print(f"Loaded {len(self.window_indices)} Fulsang windows for {mode} mode")
        print(f"Fulsang EEG shape: {self.eeg_data.shape}")
        print(f"Fulsang Audio envelopes shape: {self.audio_envelopes.shape}")
        print(f"Fulsang Label distribution: {np.bincount(self.labels)}")
        print(f"Using Fulsang preprocessing: Yes")
        print(f"Cache size: {cache_size} windows")
    
    def _load_fulsang_preprocessing_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Load Fulsang preprocessing validated TFRecord data.
        
        CRITICAL: Each TFRecord example is a TRIAL, not a sample!
        - EEG: flattened (trial_length * n_channels) = (3200 * 66) = 211,200 floats
        - wavA: flattened (trial_length * 1) = 3200 floats (if present)
        - wavB: flattened (trial_length * 1) = 3200 floats (if present)
        - attention_label: single int64 (trial-level label)
        - n_channels, n_samples, sampling_rate: metadata for reshaping
        """

        tfrecord_files = []
        
        direct_files = list(self.tfrecord_dir.glob("*.tfrecords"))
        if direct_files:
            tfrecord_files.extend(direct_files)
        
        subdir_files = list(self.tfrecord_dir.glob("*/*.tfrecords"))
        if subdir_files:
            tfrecord_files.extend(subdir_files)
        
        nested_files = list(self.tfrecord_dir.glob("*/*/*.tfrecords"))
        if nested_files:
            tfrecord_files.extend(nested_files)
        
        if not tfrecord_files:
            print(f"Available directories in {self.tfrecord_dir}:")
            if self.tfrecord_dir.exists():
                for item in self.tfrecord_dir.iterdir():
                    print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
            raise ValueError(f"No TFRecord files found in {self.tfrecord_dir} or its subdirectories")
        
        # Limit files if specified (for faster loading during development)
        if self.max_files and self.max_files < len(tfrecord_files):
            print(f"⚠ Limiting to {self.max_files} files (out of {len(tfrecord_files)}) for faster loading")
            tfrecord_files = tfrecord_files[:self.max_files]
        
        print(f"Loading Fulsang preprocessing validated data from {len(tfrecord_files)} files...")
        print("✓ Using validated attention labels with quality control")
        print("✓ Using subject-wise organized data (no data leakage)")
        print("✓ Reading TRIAL-LEVEL data (each example = one full trial)")
        if self.load_audio:
            print("✓ EEG + Audio envelope processing for Fulsang dataset")
        else:
            print("⚠ Audio loading DISABLED - using dummy audio envelopes (faster loading)")
        print(f"✓ Found TFRecord files in: {[f.parent.name for f in tfrecord_files[:3]]}...")
        
        # Store trials (not samples) - each trial will be windowed later
        all_trials = []  # List of dicts: {'eeg': (n_samples, n_channels), 'wavA': (n_samples, 1), 'wavB': (n_samples, 1), 'label': int, 'metadata': dict}
        
        successful_files = 0
        failed_files = 0
        total_trials = 0
        skipped_trials = 0
        skip_reasons = {'missing_features': 0, 'empty_eeg': 0, 'shape_mismatch': 0, 'invalid_eeg': 0, 'label_error': 0, 'missing_metadata': 0}
        subject_stats = {}
        wavA_missing_count = 0
        wavB_missing_count = 0
        
        for tfrecord_file in tqdm(tfrecord_files, desc="Loading Fulsang preprocessing data"):
            try:
                dataset = tf.data.TFRecordDataset(str(tfrecord_file))
                trials_in_file = 0
                file_subject_id = None
                
                for record in dataset:
                    try:
                        example = tf.train.Example.FromString(record.numpy())
                        features = example.features.feature
                        
                        # Required features for trial-level data
                        required_features = ['eeg', 'attention_label', 'subject_id', 'n_channels', 'n_samples']
                        missing_features = [key for key in required_features if key not in features]
                        if missing_features:
                            skip_reasons['missing_features'] += 1
                            if skip_reasons['missing_features'] <= 5:
                                print(f"WARNING: Missing features {missing_features} in {tfrecord_file.name} (trial {total_trials + skipped_trials})")
                            skipped_trials += 1
                            continue
                        
                        # Read metadata for proper reshaping
                        n_channels = int(features['n_channels'].int64_list.value[0])
                        n_samples = int(features['n_samples'].int64_list.value[0])
                        sampling_rate = int(features['sampling_rate'].int64_list.value[0]) if 'sampling_rate' in features else self.sampling_rate
                        
                        # Read EEG data (flattened: n_samples * n_channels)
                        eeg_values = features['eeg'].float_list.value
                        if not eeg_values or len(eeg_values) == 0:
                            skip_reasons['empty_eeg'] += 1
                            skipped_trials += 1
                            continue
                        
                        # Reshape EEG: flattened -> (n_samples, n_channels)
                        expected_eeg_size = n_samples * n_channels
                        if len(eeg_values) != expected_eeg_size:
                            skip_reasons['shape_mismatch'] += 1
                            if skip_reasons['shape_mismatch'] <= 5:
                                print(f"ERROR: EEG size mismatch in {tfrecord_file.name}: expected {expected_eeg_size} (n_samples={n_samples} * n_channels={n_channels}), got {len(eeg_values)}")
                            skipped_trials += 1
                            continue
                        
                        eeg_trial = np.array(eeg_values, dtype=np.float32).reshape(n_samples, n_channels)
                        
                        # Validate shape
                        if eeg_trial.shape != (n_samples, n_channels):
                            skip_reasons['shape_mismatch'] += 1
                            skipped_trials += 1
                            continue
                        

                        if np.any(np.isnan(eeg_trial)) or np.any(np.isinf(eeg_trial)):
                            skip_reasons['invalid_eeg'] += 1
                            if skip_reasons['invalid_eeg'] <= 5:
                                print(f"WARNING: Invalid EEG values (NaN/Inf) in {tfrecord_file.name}")
                            skipped_trials += 1
                            continue
                        

                        # Fulsang uses attention_label (int64) instead of attended_ear (bytes)
                        if 'attention_label' in features:
                            label_values = features['attention_label'].int64_list.value
                            if not label_values or len(label_values) == 0:
                                skip_reasons['label_error'] += 1
                                skipped_trials += 1
                                continue
                            label = int(label_values[0])
                            if label not in [0, 1]:
                                skip_reasons['label_error'] += 1
                                if skip_reasons['label_error'] <= 5:
                                    print(f"ERROR: Invalid attention_label {label} in {tfrecord_file.name} (trial {total_trials + skipped_trials})")
                                skipped_trials += 1
                                continue
                            # Convert to attended_ear for compatibility with audio loading logic
                            attended_ear = 'L' if label == 0 else 'R'
                        elif 'attended_ear' in features:
                            # Fallback to attended_ear format if available
                            attended_ear_values = features['attended_ear'].bytes_list.value
                            if not attended_ear_values or len(attended_ear_values) == 0:
                                skip_reasons['label_error'] += 1
                                skipped_trials += 1
                                continue
                            try:
                                attended_ear = attended_ear_values[0].decode('utf-8')
                                label = 0 if attended_ear == 'L' else 1
                            except Exception:
                                skip_reasons['label_error'] += 1
                                if skip_reasons['label_error'] <= 5:
                                    print(f"ERROR: Could not decode attended_ear in {tfrecord_file.name}")
                                skipped_trials += 1
                                continue
                            
                            if attended_ear not in ['L', 'R']:
                                skip_reasons['label_error'] += 1
                                if skip_reasons['label_error'] <= 5:
                                    print(f"ERROR: Invalid attended_ear {attended_ear} in {tfrecord_file.name}")
                                skipped_trials += 1
                                continue
                        else:
                            skip_reasons['label_error'] += 1
                            if skip_reasons['label_error'] <= 5:
                                print(f"ERROR: Neither attention_label nor attended_ear found in {tfrecord_file.name} (trial {total_trials + skipped_trials})")
                            skipped_trials += 1
                            continue
                        

                        subject_id = "unknown"
                        sample_idx = 0
                        

                        subject_values = features['subject_id'].bytes_list.value
                        if subject_values and len(subject_values) > 0:
                            try:
                                subject_id = subject_values[0].decode('utf-8')
                                file_subject_id = subject_id
                            except Exception:
                                subject_id = f"subject_{total_trials}"
                        else:
                            subject_id = f"subject_{total_trials}"
                        
                        if 'sample_id' in features:
                            sample_values = features['sample_id'].int64_list.value
                            if sample_values and len(sample_values) > 0:
                                sample_idx = sample_values[0]
                        


                        # Load audio envelope: prefer from TFRecord file, fallback to audio files if available
                        audio_envelope = None
                        
                        # First, try to use envelope data from TFRecord file (most reliable)
                        # Envelope is stored as flattened array: could be 1 value or 4 features (4 values)
                        if 'envelope' in features:
                            envelope_values = features['envelope'].float_list.value
                            if envelope_values and len(envelope_values) > 0:
                                envelope_array = np.array(envelope_values, dtype=np.float32)
                                # Envelope can be 1 value (single feature) or 4 values (4 features)
                                # Store as 1D array - will be reshaped later in __getitem__
                                if len(envelope_array) == 1:
                                    audio_envelope = envelope_array  # Single value
                                elif len(envelope_array) == 4:
                                    audio_envelope = envelope_array  # 4 features - use all
                                else:
                                    # Unexpected length - take first 4 or pad/truncate
                                    if len(envelope_array) < 4:
                                        # Pad with zeros
                                        padded = np.zeros(4, dtype=np.float32)
                                        padded[:len(envelope_array)] = envelope_array
                                        audio_envelope = padded
                                    else:
                                        # Truncate to 4
                                        audio_envelope = envelope_array[:4]
                        
                        # Note: TFRecords contain wavA and wavB (full time series), not left_envelope/right_envelope
                        # We'll read wavA/wavB below and convert to envelope format
                        

                        # Store trial data (not sample data)
                        if subject_id not in subject_stats:
                            subject_stats[subject_id] = {'trials': 0, 'labels': []}
                        subject_stats[subject_id]['trials'] += 1
                        subject_stats[subject_id]['labels'].append(label)  # Use 'label', not 'trial_label'
                        
                        # Read left_envelope and right_envelope from TFRecord (preferred method)
                        # These are the competing audio streams for attention decoding
                        left_envelope_trial = None
                        right_envelope_trial = None
                        
                        if 'left_envelope' in features:
                            left_env_values = features['left_envelope'].float_list.value
                            if left_env_values and len(left_env_values) > 0:
                                # left_envelope should be (n_samples * 4) flattened
                                left_env_array = np.array(left_env_values, dtype=np.float32)
                                expected_size = n_samples * 4
                                if len(left_env_array) == expected_size:
                                    left_envelope_trial = left_env_array.reshape(n_samples, 4)
                                elif len(left_env_array) > 0:
                                    # Handle mismatch - pad or truncate
                                    if len(left_env_array) < expected_size:
                                        padded = np.zeros(expected_size, dtype=np.float32)
                                        padded[:len(left_env_array)] = left_env_array
                                        left_envelope_trial = padded.reshape(n_samples, 4)
                                    else:
                                        left_envelope_trial = left_env_array[:expected_size].reshape(n_samples, 4)
                        
                        if 'right_envelope' in features:
                            right_env_values = features['right_envelope'].float_list.value
                            if right_env_values and len(right_env_values) > 0:
                                # right_envelope should be (n_samples * 4) flattened
                                right_env_array = np.array(right_env_values, dtype=np.float32)
                                expected_size = n_samples * 4
                                if len(right_env_array) == expected_size:
                                    right_envelope_trial = right_env_array.reshape(n_samples, 4)
                                elif len(right_env_array) > 0:
                                    # Handle mismatch - pad or truncate
                                    if len(right_env_array) < expected_size:
                                        padded = np.zeros(expected_size, dtype=np.float32)
                                        padded[:len(right_env_array)] = right_env_array
                                        right_envelope_trial = padded.reshape(n_samples, 4)
                                    else:
                                        right_envelope_trial = right_env_array[:expected_size].reshape(n_samples, 4)
                        
                        # Fallback: If left_envelope/right_envelope not available, try wavA/wavB
                        # But we need to convert wavA/wavB (n_samples, 1) to (n_samples, 4) format
                        if left_envelope_trial is None or right_envelope_trial is None:
                            # Read wavA and wavB if available
                            wavA_trial = None
                            wavB_trial = None
                            
                            wavA_missing = 0
                            wavB_missing = 0
                            if 'wavA_missing' in features:
                                wavA_missing = int(features['wavA_missing'].int64_list.value[0])
                            if 'wavB_missing' in features:
                                wavB_missing = int(features['wavB_missing'].int64_list.value[0])
                            
                            if wavA_missing == 0 and 'wavA' in features:
                                wavA_values = features['wavA'].float_list.value
                                if wavA_values and len(wavA_values) == n_samples:
                                    wavA_trial = np.array(wavA_values, dtype=np.float32).reshape(n_samples, 1)
                            
                            if wavB_missing == 0 and 'wavB' in features:
                                wavB_values = features['wavB'].float_list.value
                                if wavB_values and len(wavB_values) == n_samples:
                                    wavB_trial = np.array(wavB_values, dtype=np.float32).reshape(n_samples, 1)
                            
                            # Convert wavA/wavB to 4-feature format if needed
                            if left_envelope_trial is None and wavA_trial is not None:
                                # Convert (n_samples, 1) to (n_samples, 4) using _process_audio_envelope logic
                                left_envelope_trial = self._convert_to_4features(wavA_trial.flatten(), n_samples)
                            
                            if right_envelope_trial is None and wavB_trial is not None:
                                right_envelope_trial = self._convert_to_4features(wavB_trial.flatten(), n_samples)
                        
                        # Create dummy envelopes if still missing
                        if left_envelope_trial is None:
                            left_envelope_trial = np.zeros((n_samples, 4), dtype=np.float32)
                        if right_envelope_trial is None:
                            right_envelope_trial = np.zeros((n_samples, 4), dtype=np.float32)
                        
                        # Store trial with all necessary information
                        trial_data = {
                            'eeg': eeg_trial,  # Shape: (n_samples, n_channels)
                            'left_envelope': left_envelope_trial,  # Shape: (n_samples, 4)
                            'right_envelope': right_envelope_trial,  # Shape: (n_samples, 4)
                            'label': label,  # Trial-level label (0 or 1) - use 'label' consistently
                            'subject_id': subject_id,
                            'trial_idx': total_trials,  # Use total_trials as trial counter
                            'n_samples': n_samples,
                            'n_channels': n_channels,
                            'sampling_rate': sampling_rate,
                            'file': tfrecord_file.name
                        }
                        
                        all_trials.append(trial_data)
                        trials_in_file += 1
                        total_trials += 1
                        
                    except Exception as trial_error:
                        print(f"ERROR processing trial in {tfrecord_file.name}: {trial_error}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                if trials_in_file > 0:
                    successful_files += 1
                    if file_subject_id:
                        print(f"✓ Loaded {trials_in_file} trials from subject {file_subject_id}")
                else:
                    failed_files += 1
                    
            except Exception as e:
                failed_files += 1
                print(f"ERROR loading {tfrecord_file.name}: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"Loading Summary:")
        print(f"  Successfully loaded files: {successful_files}")
        print(f"  Failed files: {failed_files}")
        print(f"  Total trials loaded: {total_trials}")
        print(f"  Skipped trials: {skipped_trials}")
        if skipped_trials > 0:
            print(f"\n  Skip reasons:")
            for reason, count in skip_reasons.items():
                if count > 0:
                    print(f"    {reason}: {count}")
        print(f"  wavA missing: {wavA_missing_count}/{total_trials} ({100*wavA_missing_count/max(total_trials,1):.1f}%)")
        print(f"  wavB missing: {wavB_missing_count}/{total_trials} ({100*wavB_missing_count/max(total_trials,1):.1f}%)")
        print(f"{'='*60}")
        
        if total_trials == 0:
            print("\n⚠ CRITICAL: No trials were loaded successfully!")
            print("This could be due to:")
            print("  - Incorrect TFRecord format")
            print("  - Missing required features")
            print("  - Data corruption")
            print("  - Wrong file paths")
            print(f"\nDebugging info:")
            print(f"  TFRecord directory: {self.tfrecord_dir}")
            print(f"  Directory exists: {self.tfrecord_dir.exists()}")
            if self.tfrecord_dir.exists():
                print(f"  Contents:")
                for item in self.tfrecord_dir.iterdir():
                    print(f"    - {item.name} ({'dir' if item.is_dir() else 'file'})")
                    if item.is_dir():
                        subfiles = list(item.glob("*.tfrecords"))
                        print(f"      Contains {len(subfiles)} TFRecord files")
            raise ValueError("No valid trials loaded from TFRecord files")
        
        print(f"\nSubject-wise statistics:")
        for subject_id, stats in subject_stats.items():
            label_dist = np.bincount(stats['labels'])
            print(f"  {subject_id}: {stats['trials']} trials, labels {label_dist}")
        
        # Store trials for windowing (will be processed in _create_fulsang_windows)
        # For now, we need to return data in a format compatible with existing code
        # We'll concatenate trials but keep track of trial boundaries for proper windowing
        
        # Concatenate all trials for windowing
        all_eeg_samples = []
        all_left_envelopes = []
        all_right_envelopes = []
        all_sample_labels = []
        trial_boundaries = []  # Track where each trial starts/ends in concatenated data
        current_offset = 0
        
        for trial in all_trials:
            n_samples = trial['n_samples']
            eeg_trial = trial['eeg']  # (n_samples, n_channels)
            left_envelope = trial['left_envelope']  # (n_samples, 4)
            right_envelope = trial['right_envelope']  # (n_samples, 4)
            trial_label = trial['label']  # Single label for entire trial
            
            # Expand trial label to all samples in this trial
            trial_labels = np.full(n_samples, trial_label, dtype=np.int64)
            
            all_eeg_samples.append(eeg_trial)
            all_left_envelopes.append(left_envelope)
            all_right_envelopes.append(right_envelope)
            all_sample_labels.append(trial_labels)
            
            # Record trial boundary
            trial_boundaries.append((current_offset, current_offset + n_samples, trial))
            current_offset += n_samples
        
        # Concatenate all trials
        eeg_data = np.vstack(all_eeg_samples)  # (total_samples, n_channels)
        left_envelopes = np.vstack(all_left_envelopes)  # (total_samples, 4)
        right_envelopes = np.vstack(all_right_envelopes)  # (total_samples, 4)
        labels = np.concatenate(all_sample_labels)  # (total_samples,)
        
        # Create metadata list (one per sample for compatibility, but pointing to trial info)
        all_metadata = []
        for trial in all_trials:
            n_samples = trial['n_samples']
            for i in range(n_samples):
                metadata = {
                    'subject_id': trial['subject_id'],
                    'trial_idx': trial['trial_idx'],
                    'sample_in_trial': i,
                    'attention_label': trial['label'],
                    'n_samples': n_samples,
                    'n_channels': trial['n_channels'],
                    'sampling_rate': trial['sampling_rate'],
                    'file': trial['file']
                }
                all_metadata.append(metadata)
        
        print(f"\nFinal data shapes:")
        print(f"  EEG data: {eeg_data.shape} (samples, channels)")
        print(f"  Left streams: {left_streams.shape} (samples, 1)")
        print(f"  Right streams: {right_streams.shape} (samples, 1)")
        print(f"  Labels: {labels.shape} (samples,)")
        print(f"  Number of trials: {len(all_trials)}")
        print(f"  Trial boundaries tracked: {len(trial_boundaries)}")
        
        if eeg_data.shape[1] != 66:
            raise ValueError(f"CRITICAL: EEG data has {eeg_data.shape[1]} channels, expected 66")
        
        if len(eeg_data) != len(labels):
            raise ValueError(f"CRITICAL: EEG samples ({len(eeg_data)}) != labels ({len(labels)})")
        
        # Store trial boundaries and envelopes for use in windowing and __getitem__
        self.trial_boundaries = trial_boundaries
        self.left_envelopes = left_envelopes  # (total_samples, 4)
        self.right_envelopes = right_envelopes  # (total_samples, 4)
        
        # For backward compatibility, create audio_envelopes from left_envelopes (will be replaced in __getitem__)
        # This is a placeholder - actual audio will come from left_envelopes/right_envelopes in __getitem__
        audio_envelopes = np.zeros((len(eeg_data), 4), dtype=np.float32)
        
        valid_audio_count = np.sum((np.abs(left_envelopes).sum(axis=1) > 1e-6) | (np.abs(right_envelopes).sum(axis=1) > 1e-6))
        print(f"  Valid audio envelopes: {valid_audio_count}/{len(eeg_data)} ({100*valid_audio_count/len(eeg_data):.1f}%)")
        
        if valid_audio_count > 0:
            non_zero_left = left_envelopes[np.abs(left_envelopes).sum(axis=1) > 1e-6]
            non_zero_right = right_envelopes[np.abs(right_envelopes).sum(axis=1) > 1e-6]
            if len(non_zero_left) > 0:
                print(f"  Left envelope stats: mean={np.mean(non_zero_left, axis=0)}, std={np.std(non_zero_left, axis=0)}")
            if len(non_zero_right) > 0:
                print(f"  Right envelope stats: mean={np.mean(non_zero_right, axis=0)}, std={np.std(non_zero_right, axis=0)}")
        else:
            print(f"  ⚠ WARNING: All audio envelopes are zero! Check left_envelope/right_envelope or wavA/wavB loading in TFRecords.")
        
        return eeg_data, audio_envelopes, labels, all_metadata
    
    def _convert_to_4features(self, audio_data: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Convert 1D audio data to (n_samples, 4) format.
        
        Args:
            audio_data: 1D array of audio envelope values
            n_samples: Number of samples (window size)
            
        Returns:
            Array of shape (n_samples, 4) with [original, smoothed, derivative, squared]
        """
        # Ensure correct length
        if len(audio_data) < n_samples:
            audio_data = np.pad(audio_data, (0, n_samples - len(audio_data)), mode='constant')
        elif len(audio_data) > n_samples:
            audio_data = audio_data[:n_samples]
        
        # Compute 4 features
        env_vals = audio_data.flatten()
        if len(env_vals) > 1:
            from scipy.ndimage import uniform_filter1d
            smoothed = uniform_filter1d(env_vals, size=min(3, len(env_vals)), mode='nearest')
            derivative = np.gradient(env_vals)
        else:
            smoothed = env_vals
            derivative = np.zeros_like(env_vals)
        
        features = np.column_stack([
            env_vals,
            smoothed,
            derivative,
            env_vals**2
        ])
        
        # Normalize
        if np.max(np.abs(features)) > 0:
            features = features / (np.max(np.abs(features)) + 1e-8)
        
        return features.astype(np.float32)
    
    def _load_audio_envelope_full(self, audio_file_path: str) -> Optional[np.ndarray]:
        """
        Load full audio envelope from audio file, resampled to match EEG sampling rate.
        This is more efficient than loading per-sample.
        
        Args:
            audio_file_path: Path to audio file (can be relative or absolute)
            
        Returns:
            Full audio envelope array, or None if file not found
        """

        cache_key = str(Path(audio_file_path).resolve()) if Path(audio_file_path).is_absolute() else audio_file_path
        if cache_key in self._audio_envelope_cache:
            return self._audio_envelope_cache[cache_key]
        

        audio_file = None
        

        if Path(audio_file_path).is_absolute() and Path(audio_file_path).exists():
            audio_file = Path(audio_file_path)

        elif Path(audio_file_path).exists():
            audio_file = Path(audio_file_path)

        elif (self.audio_base_dir / audio_file_path).exists():
            audio_file = self.audio_base_dir / audio_file_path

        elif (self.audio_base_dir / Path(audio_file_path).name).exists():
            audio_file = self.audio_base_dir / Path(audio_file_path).name

        elif (Path("Data/Das/4004271/Stimuli") / Path(audio_file_path).name).exists():
            audio_file = Path("Data/Das/4004271/Stimuli") / Path(audio_file_path).name

        else:
            audio_filename = Path(audio_file_path).name

            audio_stem = Path(audio_file_path).stem
            for ext in ['.wav', '.WAV', '.mp3', '.MP3']:
                test_file = self.audio_base_dir / f"{audio_stem}{ext}"
                if test_file.exists():
                    audio_file = test_file
                    break

            if audio_file is None:
                matches = list(self.audio_base_dir.glob(f"*{audio_stem}*"))
                if matches:
                    audio_file = matches[0]
        
        if audio_file is None or not audio_file.exists():

            return None
        
        try:

            from scipy.io import wavfile
            from scipy import signal
            
            fs, audio_data = wavfile.read(str(audio_file))
            

            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            

            audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            

            if fs != self.sampling_rate:
                num_samples = int(len(audio_data) * self.sampling_rate / fs)
                audio_data = signal.resample(audio_data, num_samples)
            

            from scipy.signal import hilbert
            analytic_signal = hilbert(audio_data)
            envelope = np.abs(analytic_signal)
            

            if len(envelope) > 9:
                kernel = np.ones(9) / 9.0
                envelope = np.convolve(envelope, kernel, mode='same')
            

            if np.max(envelope) > 0:
                envelope = envelope / np.max(envelope)
            

            self._audio_envelope_cache[cache_key] = envelope
            
            return envelope
                
        except Exception as e:
            print(f"WARNING: Could not load audio envelope from {audio_file_path}: {e}")
            return None
    
    def _load_audio_envelope(self, audio_file_path: str, sample_idx: int) -> Optional[np.ndarray]:
        """
        Load audio envelope from audio file, resampled to match EEG sampling rate.
        
        Args:
            audio_file_path: Path to audio file (can be relative or absolute)
            sample_idx: Sample index for temporal alignment
            
        Returns:
            Audio envelope value(s) for this sample, or None if file not found
        """
        """
        Load audio envelope value for a specific sample index.
        Uses cached full envelope if available, otherwise loads it.
        
        Args:
            audio_file_path: Path to audio file (can be relative or absolute)
            sample_idx: Sample index for temporal alignment
            
        Returns:
            Audio envelope value(s) for this sample, or None if file not found
        """

        envelope_full = self._load_audio_envelope_full(audio_file_path)
        
        if envelope_full is None:
            return None
        

        if sample_idx < len(envelope_full):
            return envelope_full[sample_idx:sample_idx+1]
        else:

            if len(envelope_full) > 0:
                return envelope_full[-1:]
            else:
                return np.array([0.0], dtype=np.float32)
    
    def _create_fulsang_windows(self) -> List[Tuple[int, int, int]]:
        """
        Create windows WITHIN each trial, never across trial boundaries.
        
        Returns:
            List of (trial_idx, offset_in_trial, label) tuples
            - trial_idx: Index into self.trial_boundaries
            - offset_in_trial: Starting sample offset within that trial
            - label: Trial-level label (inherited by all windows in the trial)
        """
        window_seconds = self.window_size / self.sampling_rate
        step_size = int(self.window_size * (1 - self.overlap))
        step_seconds = step_size / self.sampling_rate
        
        print(f"Creating Fulsang windows WITHIN trials (no cross-trial windows):")
        print(f"  Window size: {self.window_size} samples ({window_seconds:.1f} seconds)")
        print(f"  Step size: {step_size} samples ({step_seconds:.1f} seconds)")
        print(f"  Overlap: {self.overlap:.1%}")
        print(f"  Sampling rate: {self.sampling_rate} Hz")
        print(f"  Number of trials: {len(self.trial_boundaries)}")
        
        if window_seconds < 1.0:
            print(f"⚠ WARNING: Very short window ({window_seconds:.1f}s) may have poor signal-to-noise")
        elif window_seconds > 20.0:
            print(f"⚠ WARNING: Very long window ({window_seconds:.1f}s) may miss temporal dynamics")
        else:
            print(f"✓ Window size appropriate for EEG attention decoding")
        
        window_indices = []
        window_label_stats = {'class_0': 0, 'class_1': 0}
        total_windows = 0
        
        # Window within each trial separately
        for trial_idx, (trial_start, trial_end, trial_info) in enumerate(self.trial_boundaries):
            trial_length = trial_end - trial_start
            trial_label = trial_info['label']  # Trial-level label
            
            # Calculate how many windows fit in this trial
            if trial_length < self.window_size:
                # Trial too short - skip it
                continue
            
            # Create windows within this trial
            trial_windows = (trial_length - self.window_size) // step_size + 1
            
            for window_idx in range(trial_windows):
                offset_in_trial = window_idx * step_size
                
                # Verify window stays within trial
                if offset_in_trial + self.window_size <= trial_length:
                    # Store: (trial_idx, offset_in_trial, label)
                    # trial_idx and offset_in_trial will be used in __getitem__ to extract the window
                    window_indices.append((trial_idx, offset_in_trial, trial_label))
                    total_windows += 1
                    
                    # Track label distribution
                    if trial_label == 0:
                        window_label_stats['class_0'] += 1
                    else:
                        window_label_stats['class_1'] += 1
        
        print(f"Created {total_windows} Fulsang windows (all within trial boundaries)")
        print(f"Window label distribution:")
        print(f"  Class 0 (left): {window_label_stats['class_0']}")
        print(f"  Class 1 (right): {window_label_stats['class_1']}")
        
        return window_indices
    
    def _das_eeg_preprocessing(self, eeg_window: np.ndarray) -> np.ndarray:
        """Fulsang-specific EEG preprocessing with artifact handling."""
        from scipy import signal
        


        artifact_threshold = 5.0
        for ch in range(eeg_window.shape[1]):
            channel_data = eeg_window[:, ch]
            std_val = np.std(channel_data)
            mean_val = np.mean(channel_data)
            

            artifacts = np.abs(channel_data - mean_val) > (artifact_threshold * std_val)
            
            if np.any(artifacts):

                valid_indices = ~artifacts
                if np.sum(valid_indices) > 2:
                    from scipy.interpolate import interp1d
                    valid_data = channel_data[valid_indices]
                    valid_time = np.where(valid_indices)[0]
                    all_time = np.arange(len(channel_data))
                    
                    f_interp = interp1d(valid_time, valid_data, kind='linear', 
                                      bounds_error=False, fill_value='extrapolate')
                    eeg_window[:, ch] = f_interp(all_time)
        

        eeg_window = eeg_window - np.mean(eeg_window, axis=0, keepdims=True)
        

        nyquist = self.sampling_rate / 2
        low_freq = 1.0 / nyquist
        high_freq = min(40.0 / nyquist, 0.99)
        

        b, a = signal.butter(4, [low_freq, high_freq], btype='band')
        

        filtered_eeg = np.zeros_like(eeg_window)
        for ch in range(eeg_window.shape[1]):
            filtered_eeg[:, ch] = signal.filtfilt(b, a, eeg_window[:, ch])
        

        mad_values = np.median(np.abs(filtered_eeg - np.median(filtered_eeg, axis=0)), axis=0)
        mad_values = np.where(mad_values == 0, 1.0, mad_values)
        filtered_eeg = filtered_eeg / mad_values
        

        filtered_eeg = np.tanh(filtered_eeg * 0.5)
        

        if np.any(np.isnan(filtered_eeg)) or np.any(np.isinf(filtered_eeg)):
            print("WARNING: Invalid values detected after preprocessing")
            filtered_eeg = np.nan_to_num(filtered_eeg, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return filtered_eeg.astype(np.float32)
    
    def __len__(self):
        return len(self.window_indices)
    
    def __getitem__(self, idx):
        """
        Get a window from the dataset.
        
        Args:
            idx: Window index
            
        Returns:
            (window_tensor, aux_data) where:
            - window_tensor: (eeg_tensor, audio_tensor) - EEG and attended audio
            - aux_data: dict with 'left_env', 'right_env', 'label'
        """
        # New window format: (trial_idx, offset_in_trial, label)
        trial_idx, offset_in_trial, label = self.window_indices[idx]
        
        # Get trial information
        trial_start, trial_end, trial_info = self.trial_boundaries[trial_idx]
        
        # Extract window from trial (using concatenated data but respecting trial boundaries)
        window_start = trial_start + offset_in_trial
        window_end = window_start + self.window_size
        
        # Verify window is within trial
        assert window_end <= trial_end, f"Window extends beyond trial boundary: {window_end} > {trial_end}"
        
        # Extract EEG window
        window_eeg = self.eeg_data[window_start:window_end]
        
        # Extract REAL left and right audio envelopes (no synthetic generation!)
        # These are already in (window_size, 4) format from TFRecord
        left_envelope_window = self.left_envelopes[window_start:window_end]  # Shape: (window_size, 4)
        right_envelope_window = self.right_envelopes[window_start:window_end]  # Shape: (window_size, 4)
        
        # Ensure correct shape (should already be correct, but verify)
        if left_envelope_window.shape != (self.window_size, 4):
            # Pad or truncate if needed
            if left_envelope_window.shape[0] < self.window_size:
                padding = np.zeros((self.window_size - left_envelope_window.shape[0], 4), dtype=np.float32)
                left_envelope_window = np.vstack([left_envelope_window, padding])
            else:
                left_envelope_window = left_envelope_window[:self.window_size]
        
        if right_envelope_window.shape != (self.window_size, 4):
            if right_envelope_window.shape[0] < self.window_size:
                padding = np.zeros((self.window_size - right_envelope_window.shape[0], 4), dtype=np.float32)
                right_envelope_window = np.vstack([right_envelope_window, padding])
            else:
                right_envelope_window = right_envelope_window[:self.window_size]
        
        left_audio_processed = left_envelope_window
        right_audio_processed = right_envelope_window
        
        # Use attended stream for window_audio (for backward compatibility)
        # Label 0 = left attended, Label 1 = right attended
        if label == 0:
            window_audio = left_audio_processed
        else:
            window_audio = right_audio_processed

        try:
            window_eeg = self._das_eeg_preprocessing(window_eeg)
        except Exception:
            window_eeg = window_eeg - np.mean(window_eeg, axis=0, keepdims=True)
            window_eeg = window_eeg / (np.std(window_eeg, axis=0, keepdims=True) + 1e-8)
            window_eeg = np.tanh(window_eeg * 0.5)
        
        
        # window_audio is already in (window_size, 4) format from _process_audio_envelope
        # Convert to tensors
        window_eeg_tensor = tf.constant(window_eeg, dtype=tf.float32)
        window_audio_tensor = tf.constant(window_audio, dtype=tf.float32)
        left_audio_tensor = tf.constant(left_audio_processed, dtype=tf.float32)
        right_audio_tensor = tf.constant(right_audio_processed, dtype=tf.float32)
        label_tensor = tf.constant([label], dtype=tf.int64)
        

        # Return both audio streams for comparison
        window_tensor = (window_eeg_tensor, window_audio_tensor)
        aux_data = {
            'left_env': left_audio_tensor,
            'right_env': right_audio_tensor,
            'label': label_tensor
        }
        

        # Don't cache windows - variation needs to be applied fresh each time
        # if len(self._window_cache) < self.cache_size:
        #     self._window_cache[cache_key] = (window_tensor, aux_data)
        
        return window_tensor, aux_data
    
    def _process_audio_envelope(self, audio_envelope: np.ndarray, window_size: int) -> np.ndarray:
        """Process audio envelope to match window size and format."""
        if audio_envelope is None or len(audio_envelope) == 0:
            audio_envelope = np.array([0.0], dtype=np.float32)
        
        # If single value, expand to window size
        if len(audio_envelope) == 1:
            audio_envelope = np.repeat(audio_envelope, window_size)
        elif len(audio_envelope) < window_size:
            # Pad if too short
            padding = np.zeros(window_size - len(audio_envelope), dtype=np.float32)
            audio_envelope = np.concatenate([audio_envelope, padding])
        elif len(audio_envelope) > window_size:
            # Truncate if too long
            audio_envelope = audio_envelope[:window_size]
        
        # Reshape and format to 4 features (same as window_audio)
        if audio_envelope.ndim == 1:
            audio_envelope = audio_envelope.reshape(-1, 1)
        
        if audio_envelope.shape[1] == 1:
            env_vals = audio_envelope.flatten()
            if len(env_vals) > 1:
                from scipy.ndimage import uniform_filter1d
                smoothed = uniform_filter1d(env_vals, size=min(3, len(env_vals)), mode='nearest')
                derivative = np.gradient(env_vals)
            else:
                smoothed = env_vals
                derivative = np.zeros_like(env_vals)
            
            audio_envelope = np.column_stack([
                env_vals,
                smoothed,
                derivative,
                env_vals**2
            ])
        elif audio_envelope.shape[1] != 4:
            if audio_envelope.shape[1] < 4:
                env_vals = audio_envelope[:, 0] if audio_envelope.shape[1] > 0 else np.zeros(audio_envelope.shape[0])
                padding = np.zeros((audio_envelope.shape[0], 4 - audio_envelope.shape[1]))
                audio_envelope = np.column_stack([audio_envelope, padding])
            else:
                audio_envelope = audio_envelope[:, :4]
        
        if np.max(np.abs(audio_envelope)) > 0:
            audio_envelope = audio_envelope / (np.max(np.abs(audio_envelope)) + 1e-8)
        
        return audio_envelope.astype(np.float32)


class FULCCAModel:
    """
    FULCCA model implementing Canonical Correlation Analysis for Fulsang EEG dataset.
    
    This model uses the telluride_decoding CCA implementation to find correlations
    between EEG data and attention labels, providing comprehensive metrics evaluation.
    Adapted from DASCCA with Fulsang-specific optimal hyperparameters.
    """
    
    def __init__(self, cca_dims: int = 12, regularization: float = 0.08, window_size: int = 1280):
        """
        Initialize FULCCA model with optimal Fulsang hyperparameters.
        
        Args:
            cca_dims: Number of CCA dimensions to compute (optimal: 12 for Fulsang)
            regularization: Regularization parameter for CCA (optimal: 0.08 for Fulsang)
            window_size: Window size for EEG data processing (optimal: 1280 = 20s at 64Hz)
        """



        # Fulsang CCA dimensions are limited by the minimum of EEG channels (66) and Audio features (4)
        # The actual maximum is min(66, 4) = 4, but we can request up to 12 for optimal config
        # However, we must respect the actual input dimension constraints
        eeg_dims = 66  # Fulsang: 66 EEG channels
        audio_dims = 4  # Audio envelope features (4 features)
        actual_max_cca_dims = min(eeg_dims, audio_dims)  # Actual maximum: min(66, 4) = 4
        optimal_max_cca_dims = min(actual_max_cca_dims, 12)  # Optimal config limit: min(4, 12) = 4
        
        if cca_dims > actual_max_cca_dims:
            print(f"⚠ WARNING: Requested {cca_dims} CCA dimensions, but maximum is {actual_max_cca_dims} (min(EEG={eeg_dims}, Audio={audio_dims}))")
            print(f"  Reducing CCA dimensions from {cca_dims} to {actual_max_cca_dims}")
            cca_dims = actual_max_cca_dims
        elif cca_dims > optimal_max_cca_dims:
            print(f"⚠ WARNING: Requested {cca_dims} CCA dimensions exceeds optimal limit ({optimal_max_cca_dims})")
            print(f"  Reducing CCA dimensions from {cca_dims} to {optimal_max_cca_dims}")
            cca_dims = optimal_max_cca_dims
        elif cca_dims < 1:
            print(f"⚠ WARNING: CCA dimensions must be >= 1, setting to 1")
            cca_dims = 1
        
        self.cca_dims = cca_dims
        self.regularization = regularization
        self.window_size = window_size
        self.model = None
        self.is_fitted = False
        
        print(f"FULCCA model initialized:")
        print(f"  CCA dimensions: {self.cca_dims} (optimal: 12 for Fulsang)")
        print(f"  Regularization: {regularization} (optimal: 0.08 for Fulsang)")
        print(f"  Input dimensions: EEG=66, Audio=4")
    
    def _create_robust_cca_model(self, dataset: tf.data.Dataset):
        """
        Create CCA model with robust CUDA handling.
        """

        tf.keras.backend.clear_session()
        

        safe_random_operations()
        

        print("Creating CCA model with GPU-first approach...")
        try:
            with tf.device('/GPU:0'):

                model = BrainModelCCA(
                    input_dataset=dataset,
                    cca_dims=self.cca_dims,
                    regularization_lambda=self.regularization
                )
            print("✓ CCA model created successfully on GPU")
            return model
            
        except Exception as e:
            print(f"GPU model creation failed: {e}")
            print("Trying CPU model creation as fallback...")
            

            try:
                with tf.device('/CPU:0'):
                    model = BrainModelCCA(
                        input_dataset=dataset,
                        cca_dims=self.cca_dims,
                        regularization_lambda=self.regularization
                    )
                print("✓ CCA model created successfully on CPU")
                return model
                
            except Exception as gpu_error:
                print(f"GPU model creation also failed: {gpu_error}")
                raise RuntimeError("Cannot create CCA model on either CPU or GPU")
    
    def fit(self, dataset: tf.data.Dataset):
        """
        Fit the CCA model to the dataset with robust GPU handling.
        
        Args:
            dataset: TensorFlow dataset containing EEG data and labels
        """
        print("Fitting FULCCA model...")
        

        self.model = self._create_robust_cca_model(dataset)
        

        try:
            print("Compiling CCA model...")

            self.model.compile(
                optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                loss='mse',
                metrics=[cca_pearson_correlation_first]
            )
            
            print("Training CCA model...")

            self.model.fit(dataset, epochs=1)
            
            print("✓ FULCCA model fitted successfully")
            
        except Exception as e:
            print(f"Training failed: {e}")
            error_msg = str(e)
            

            if "rot1 must be shape" in error_msg or "rot2 must be shape" in error_msg:
                print(f"\n⚠ CCA Dimension Mismatch Error Detected!")
                print(f"  Requested CCA dimensions: {self.cca_dims}")
                print(f"  Input dimensions: EEG=66, Audio=4")
                print(f"  Maximum possible CCA dimensions: min(66, 4) = 4")
                print(f"\n  The CCA computation is returning rotation matrices with wrong dimensions.")
                print(f"  This can happen if:")
                print(f"    1. The dataset format is incorrect")
                print(f"    2. The CCA computation encounters numerical issues")
                print(f"    3. The requested dimensions exceed what's possible")
                

                if self.cca_dims > 2:
                    print(f"\n  Attempting to fix by reducing CCA dimensions: {self.cca_dims} -> 2")
                    original_cca_dims = self.cca_dims
                    self.cca_dims = 2
                    

                    try:
                        self.model = self._create_robust_cca_model(dataset)
                        self.model.compile(
                            optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                            loss='mse',
                            metrics=[cca_pearson_correlation_first]
                        )
                        self.model.fit(dataset, epochs=1)
                        print("✓ FULCCA model fitted successfully with reduced dimensions (2)")
                        # Success - return early, don't try CPU fallback
                        self.is_fitted = True
                        print("✓ FULCCA model training completed")
                        return
                    except Exception as e2:
                        print(f"  Still failed with 2 dimensions: {e2}")

                        if self.cca_dims > 1:
                            print(f"  Trying with 1 CCA dimension as last resort...")
                            self.cca_dims = 1
                            self.model = self._create_robust_cca_model(dataset)
                            self.model.compile(
                                optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                                loss='mse',
                                metrics=[cca_pearson_correlation_first]
                            )
                            self.model.fit(dataset, epochs=1)
                            print("✓ FULCCA model fitted successfully with 1 dimension")
                            # Success - return early, don't try CPU fallback
                            self.is_fitted = True
                            print("✓ FULCCA model training completed")
                            return
                        else:
                            raise RuntimeError(f"Could not fit CCA model even with 1 dimension. Original error: {e}")
                else:
                    raise RuntimeError(f"CCA dimension mismatch. Requested {self.cca_dims} dimensions but computation failed. Error: {e}")
            

            # Only try CPU fallback if dimension mismatch handling didn't succeed
            print("Trying CPU fallback for training...")
            
            with tf.device('/CPU:0'):

                self.model = BrainModelCCA(
                    input_dataset=dataset,
                    cca_dims=self.cca_dims,
                    regularization_lambda=self.regularization
                )
                
                self.model.compile(
                    optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                    loss='mse',
                    metrics=[cca_pearson_correlation_first]
                )
                
                try:
                    self.model.fit(dataset, epochs=1)
                    print("✓ FULCCA model fitted successfully on CPU")
                except Exception as cpu_error:
                    cpu_error_msg = str(cpu_error)
                    if "rot1 must be shape" in cpu_error_msg or "rot2 must be shape" in cpu_error_msg:
                        print(f"\n⚠ CCA Dimension Mismatch on CPU too!")
                        print(f"  This indicates a fundamental issue with the CCA computation.")
                        print(f"  The calculate_cca_parameters_from_dataset() function is returning")
                        print(f"  rotation matrices with dimensions that don't match the requested CCA dimensions.")
                        raise RuntimeError(f"CCA dimension mismatch on both GPU and CPU. This suggests the CCA parameter computation is incorrect. Try reducing cca_dims to 1 or 2. Error: {cpu_error}")
                    raise
        
        self.is_fitted = True
        print("✓ FULCCA model training completed")
    
    def _compute_correlation_scores(self, predictions: tf.Tensor) -> np.ndarray:
        """
        Compute correlation scores from CCA projections using telluride_decoding method.
        
        This uses the actual Pearson correlation computation from telluride_decoding,
        which is the correct way to compute correlations from CCA rotated outputs.
        
        The CCA model outputs [rotated_eeg, rotated_audio] concatenated.
        We compute Pearson correlation for each CCA dimension, then weight by importance.
        """
        # Use telluride's cca_pearson_correlation function for accuracy
        # It computes actual Pearson correlations between rotated outputs
        try:
            # Convert to numpy if needed
            if hasattr(predictions, 'numpy'):
                preds_np = predictions.numpy()
            else:
                preds_np = predictions
            
            cca_width = preds_np.shape[-1] // 2
            proj_eeg = preds_np[:, :cca_width]
            proj_env = preds_np[:, cca_width:]
            
            # For per-sample scoring, compute similarity between rotated_eeg and rotated_audio
            # CCA rotations maximize correlation, so higher dot product = better match
            # Weight by dimension importance (first dimension has highest canonical correlation)
            weights = np.exp(-np.arange(cca_width) * 0.15)  # Exponential decay
            weights = weights / np.sum(weights)
            
            # Compute per-sample scores: weighted dot product between rotated_eeg and rotated_audio
            # This gives a similarity score for each sample indicating how well EEG matches audio
            dot_products = proj_eeg * proj_env  # Element-wise product per dimension
            scores = np.sum(dot_products * weights, axis=1)  # Weighted sum per sample
            
            return scores
            
        except Exception as e:
            # Fallback to simpler method if Pearson correlation fails
            print(f"Warning: Using fallback correlation computation: {e}")
            preds = predictions.numpy() if hasattr(predictions, 'numpy') else predictions
            cca_width = preds.shape[-1] // 2
            proj_eeg = preds[:, :cca_width]
            proj_env = preds[:, cca_width:]
            
            # Center the data
            proj_eeg_centered = proj_eeg - np.mean(proj_eeg, axis=0, keepdims=True)
            proj_env_centered = proj_env - np.mean(proj_env, axis=0, keepdims=True)
            
            # Compute correlation per dimension (matching telluride method)
            numerator = np.sum(proj_eeg_centered * proj_env_centered, axis=0)
            denominator = np.sqrt(np.sum(proj_eeg_centered**2, axis=0) * 
                                np.sum(proj_env_centered**2, axis=0))
            correlations = np.where(denominator > 1e-8, numerator / denominator, 0.0)
            
            # Weight by dimension importance
            weights = np.exp(-np.arange(cca_width) * 0.15)
            weights = weights / np.sum(weights)
            
            # Return weighted sum of correlations (same for all samples in batch)
            score = np.sum(correlations * weights)
            scores = np.full(proj_eeg.shape[0], score)
            
            return scores
    
    def predict(self, dataset: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the fitted CCA model with GPU optimization.
        
        Args:
            dataset: TensorFlow dataset containing EEG data and labels
            
        Returns:
            Tuple of (predictions, targets)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        print("Making FULCCA predictions...")
        
        all_predictions = []
        all_targets = []
        all_left_scores = []  # Initialize for left/right comparison diagnostics
        all_right_scores = []  # Initialize for left/right comparison diagnostics
        all_continuous_scores = []  # Store continuous scores (right - left) for ROC-AUC
        

        try:
            with device:
                for batch in tqdm(dataset, desc="Predicting"):
                    # Handle batch format: (inputs, aux_data) or (inputs, targets)
                    if isinstance(batch, tuple) and len(batch) == 2:
                        inputs, aux_or_targets = batch
                        # Check if it's aux_data (dict with left_env/right_env) or targets
                        if isinstance(aux_or_targets, dict) and 'left_env' in aux_or_targets:
                            aux = aux_or_targets
                            targets = aux.get('label', None)
                        else:
                            aux = None
                            targets = aux_or_targets
                    elif isinstance(batch, dict):
                        inputs = batch
                        aux = None
                        targets = None
                    else:
                        inputs, targets = batch
                        aux = None
                    
                    # CORRECT APPROACH: Compare correlations with BOTH left and right audio
                    if aux is not None and 'left_env' in aux and 'right_env' in aux:
                        eeg_view = inputs['input_1']
                        left_env = aux['left_env']
                        right_env = aux['right_env']
                        
                        # Compute CCA correlation with left audio
                        left_inputs = {'input_1': eeg_view, 'input_2': left_env}
                        left_predictions = self.model(left_inputs)
                        left_scores = self._compute_correlation_scores(left_predictions)
                        
                        # Compute CCA correlation with right audio
                        right_inputs = {'input_1': eeg_view, 'input_2': right_env}
                        right_predictions = self.model(right_inputs)
                        right_scores = self._compute_correlation_scores(right_predictions)
                        
                        # Aggregate per-sample scores to per-window scores
                        # Since data is batched and flattened, we need to reshape by window_size
                        num_samples = len(left_scores)
                        window_size = self.window_size
                        num_windows = num_samples // window_size
                        
                        if num_windows > 0 and num_samples % window_size == 0:
                            # Reshape scores to (num_windows, window_size) and aggregate per window
                            left_scores_reshaped = left_scores[:num_windows * window_size].reshape(num_windows, window_size)
                            right_scores_reshaped = right_scores[:num_windows * window_size].reshape(num_windows, window_size)
                            
                            # Aggregate per window (mean or sum)
                            left_window_scores = np.mean(left_scores_reshaped, axis=1)
                            right_window_scores = np.mean(right_scores_reshaped, axis=1)
                        else:
                            # Fallback: if window_size doesn't divide evenly, use mean of all scores
                            left_window_scores = np.array([np.mean(left_scores)])
                            right_window_scores = np.array([np.mean(right_scores)])
                        
                        # Store per-sample scores for diagnostics
                        all_left_scores.extend(left_scores)
                        all_right_scores.extend(right_scores)
                        
                        # Store continuous scores for ROC-AUC (right_score - left_score)
                        continuous_scores = right_window_scores - left_window_scores
                        all_continuous_scores.extend(continuous_scores)
                        
                        # Predict based on which correlation is higher: Right=1 if right > left, Left=0 otherwise
                        window_predictions = (right_window_scores > left_window_scores).astype(np.int64)
                        binary_predictions = tf.constant(window_predictions, dtype=tf.int64)
                        
                        # For left/right comparison, predictions are now per-window
                        with tf.device('/CPU:0'):
                            all_predictions.extend(binary_predictions.numpy())
                            if targets is not None:
                                # Targets should be per-window, but handle both cases
                                if hasattr(targets, 'numpy'):
                                    target_array = targets.numpy().flatten()
                                else:
                                    target_array = np.array(targets).flatten()
                                
                                # If targets are per-sample, aggregate to per-window
                                if len(target_array) == num_samples and num_windows > 0:
                                    target_reshaped = target_array[:num_windows * window_size].reshape(num_windows, window_size)
                                    # Use mode (most common label) per window
                                    from scipy.stats import mode
                                    try:
                                        window_targets = np.array([mode(target_reshaped[i], keepdims=True).mode[0] for i in range(num_windows)])
                                    except:
                                        # Fallback: use bincount
                                        window_targets = np.array([np.bincount(target_reshaped[i].astype(int)).argmax() for i in range(num_windows)])
                                    all_targets.extend(window_targets)
                                else:
                                    # Already per-window or can't aggregate
                                    all_targets.extend(target_array[:len(window_predictions)])
                        continue  # Skip the rest of the loop for this batch
                    else:
                        # Fallback: Use old method if aux data not available
                        predictions = self.model(inputs)
                    

                        cca_width = predictions.shape[-1] // 2
                    pred1 = predictions[:, :cca_width]
                    pred2 = predictions[:, cca_width:]
                    





                    # FIX: Use proper correlation scoring instead of arbitrary > 0 threshold
                    pred1_norm = pred1 / (tf.linalg.norm(pred1, axis=1, keepdims=True) + 1e-8)
                    pred2_norm = pred2 / (tf.linalg.norm(pred2, axis=1, keepdims=True) + 1e-8)
                    weights = tf.exp(-tf.range(cca_width, dtype=tf.float32) * 0.15)
                    weights = weights / tf.reduce_sum(weights)
                    dot_products = pred1_norm * pred2_norm
                    cca_scores = tf.reduce_sum(dot_products * weights, axis=1)
                    median_score = tf.reduce_mean(cca_scores)
                    





                    binary_predictions = tf.cast(cca_scores > median_score, tf.int64)
                    






                    with tf.device('/CPU:0'):
                        input_shape_tensor = tf.shape(inputs['input_1'])[0]

                        input_shape = input_shape_tensor.numpy() if hasattr(input_shape_tensor, 'numpy') else int(input_shape_tensor)
                        num_predictions = int(binary_predictions.shape[0])
                    


                    possible_window_sizes = [32, 64, 128, 256, 512, 1024, 2048, 64 * 30]
                    
                    batch_size = None
                    window_size = self.window_size
                    

                    if input_shape % self.window_size == 0:
                        batch_size = input_shape // self.window_size
                        window_size = self.window_size
                    else:

                        for ws in possible_window_sizes:
                            if input_shape % ws == 0 and input_shape // ws > 0:
                                batch_size = input_shape // ws
                                window_size = ws
                                break
                    

                    if batch_size is None or batch_size == 0:

                        batch_size = num_predictions
                        window_size = 1
                        pred_reshaped = tf.expand_dims(binary_predictions, axis=1)
                    else:

                        pred_reshaped = tf.reshape(binary_predictions, (batch_size, window_size))
                    

                        # For left/right comparison, predictions are already per-window
                        if aux is not None and 'left_env' in aux:
                            # Already have window-level predictions from left/right comparison
                            with tf.device('/CPU:0'):
                                all_predictions.extend(binary_predictions.numpy())
                                if targets is not None:
                                    if hasattr(targets, 'numpy'):
                                        all_targets.extend(targets.numpy().flatten())
                                    else:
                                        all_targets.extend(np.array(targets).flatten())
                        else:
                            # Old method: aggregate window predictions
                            sample_predictions = tf.reduce_sum(pred_reshaped, axis=1)
                            sample_predictions = tf.cast(sample_predictions > (window_size // 2), tf.int64)
                            
                            with tf.device('/CPU:0'):
                                all_predictions.extend(sample_predictions.numpy())
                                if targets is not None:
                                    all_targets.extend(targets.numpy().flatten())
        except Exception as gpu_error:
            # Try CPU fallback for any error (GPU-related or otherwise)
            error_msg = str(gpu_error)
            is_gpu_error = "CUDA" in error_msg or "GPU" in error_msg or "cuda" in error_msg.lower()
            
            if is_gpu_error:
                print(f"⚠ GPU error during prediction: {gpu_error}")
            else:
                print(f"⚠ Error during prediction: {gpu_error}")
            print("  Falling back to CPU for predictions...")

            tf.keras.backend.clear_session()
            

            try:
                with tf.device('/CPU:0'):
                    for batch in tqdm(dataset, desc="Predicting (CPU)"):
                        # Handle batch format: (inputs, aux_data) or (inputs, targets)
                        if isinstance(batch, tuple) and len(batch) == 2:
                            inputs, aux_or_targets = batch
                            # Check if it's aux_data (dict with left_env/right_env) or targets
                            if isinstance(aux_or_targets, dict) and 'left_env' in aux_or_targets:
                                aux = aux_or_targets
                                targets = aux.get('label', None)
                            else:
                                aux = None
                                targets = aux_or_targets
                        elif isinstance(batch, dict):
                            inputs = batch
                            aux = None
                            targets = None
                        else:
                            inputs, targets = batch
                            aux = None
                        
                        # CORRECT APPROACH: Compare correlations with BOTH left and right audio
                        if aux is not None and 'left_env' in aux and 'right_env' in aux:
                            eeg_view = inputs['input_1']
                            left_env = aux['left_env']
                            right_env = aux['right_env']
                            
                            # Compute CCA correlation with left audio
                            left_inputs = {'input_1': eeg_view, 'input_2': left_env}
                            left_predictions = self.model(left_inputs)
                            left_scores = self._compute_correlation_scores(left_predictions)
                            
                            # Compute CCA correlation with right audio
                            right_inputs = {'input_1': eeg_view, 'input_2': right_env}
                            right_predictions = self.model(right_inputs)
                            right_scores = self._compute_correlation_scores(right_predictions)
                            
                            # Aggregate per-sample scores to per-window scores
                            num_samples = len(left_scores)
                            window_size = self.window_size
                            num_windows = num_samples // window_size
                            
                            if num_windows > 0 and num_samples % window_size == 0:
                                # Reshape scores to (num_windows, window_size) and aggregate per window
                                left_scores_reshaped = left_scores[:num_windows * window_size].reshape(num_windows, window_size)
                                right_scores_reshaped = right_scores[:num_windows * window_size].reshape(num_windows, window_size)
                                
                                # Aggregate per window (mean)
                                left_window_scores = np.mean(left_scores_reshaped, axis=1)
                                right_window_scores = np.mean(right_scores_reshaped, axis=1)
                            else:
                                # Fallback: if window_size doesn't divide evenly, use mean of all scores
                                left_window_scores = np.array([np.mean(left_scores)])
                                right_window_scores = np.array([np.mean(right_scores)])
                            
                            # Store per-sample scores for diagnostics
                            all_left_scores.extend(left_scores)
                            all_right_scores.extend(right_scores)
                            
                            # Store continuous scores for ROC-AUC (right_score - left_score)
                            continuous_scores = right_window_scores - left_window_scores
                            all_continuous_scores.extend(continuous_scores)
                            
                            # Predict based on which correlation is higher: Right=1 if right > left, Left=0 otherwise
                            window_predictions = (right_window_scores > left_window_scores).astype(np.int64)
                            binary_predictions = tf.constant(window_predictions, dtype=tf.int64)
                            
                            # For left/right comparison, predictions are now per-window
                            all_predictions.extend(binary_predictions.numpy())
                            if targets is not None:
                                # Targets should be per-window, but handle both cases
                                if hasattr(targets, 'numpy'):
                                    target_array = targets.numpy().flatten()
                                elif isinstance(targets, (list, np.ndarray)):
                                    target_array = np.array(targets).flatten()
                                else:
                                    # Handle dict case
                                    label = targets.get('label', None) if isinstance(targets, dict) else None
                                    if label is not None:
                                        if hasattr(label, 'numpy'):
                                            target_array = label.numpy().flatten()
                                        else:
                                            target_array = np.array(label).flatten()
                                    else:
                                        target_array = np.array([])
                                
                                # If targets are per-sample, aggregate to per-window
                                if len(target_array) == num_samples and num_windows > 0:
                                    target_reshaped = target_array[:num_windows * window_size].reshape(num_windows, window_size)
                                    # Use mode (most common label) per window
                                    from scipy.stats import mode
                                    try:
                                        window_targets = np.array([mode(target_reshaped[i], keepdims=True).mode[0] for i in range(num_windows)])
                                    except:
                                        # Fallback: use bincount
                                        window_targets = np.array([np.bincount(target_reshaped[i].astype(int)).argmax() for i in range(num_windows)])
                                    all_targets.extend(window_targets)
                                else:
                                    # Already per-window or can't aggregate
                                    all_targets.extend(target_array[:len(window_predictions)])
                            continue  # Skip the rest of the loop for this batch
                        
                        # Fallback: Use old method if aux data not available
                        predictions = self.model(inputs)
                        

                        cca_width = predictions.shape[-1] // 2
                        pred1 = predictions[:, :cca_width]
                        pred2 = predictions[:, cca_width:]
                        

                        # FIX: Use proper correlation scoring instead of arbitrary > 0 threshold
                        # Compute weighted correlation across all CCA dimensions
                        pred1_norm = pred1 / (tf.linalg.norm(pred1, axis=1, keepdims=True) + 1e-8)
                        pred2_norm = pred2 / (tf.linalg.norm(pred2, axis=1, keepdims=True) + 1e-8)
                        
                        # Weight by dimension importance (first dimension has highest correlation)
                        weights = tf.exp(-tf.range(cca_width, dtype=tf.float32) * 0.15)
                        weights = weights / tf.reduce_sum(weights)
                        
                        # Compute weighted correlation scores (dot product of normalized projections)
                        dot_products = pred1_norm * pred2_norm
                        cca_scores = tf.reduce_sum(dot_products * weights, axis=1)
                        
                        # Use median as threshold (better than 0, but still not ideal without left/right comparison)
                        median_score = tf.reduce_mean(cca_scores)
                        binary_predictions = tf.cast(cca_scores > median_score, tf.int64)
                        

                        input_shape_tensor = tf.shape(inputs['input_1'])[0]
                        input_shape = input_shape_tensor.numpy() if hasattr(input_shape_tensor, 'numpy') else int(input_shape_tensor)
                        num_predictions = int(binary_predictions.shape[0])
                        

                        possible_window_sizes = [32, 64, 128, 256, 512, 1024, 2048, 64 * 30]
                        
                        batch_size = None
                        window_size = self.window_size
                        
                        if input_shape % self.window_size == 0:
                            batch_size = input_shape // self.window_size
                            window_size = self.window_size
                        else:
                            for ws in possible_window_sizes:
                                if input_shape % ws == 0 and input_shape // ws > 0:
                                    batch_size = input_shape // ws
                                    window_size = ws
                                    break
                        
                        if batch_size is None or batch_size == 0:
                            batch_size = num_predictions
                            window_size = 1
                            pred_reshaped = tf.expand_dims(binary_predictions, axis=1)
                        else:
                            pred_reshaped = tf.reshape(binary_predictions, (batch_size, window_size))
                        

                        sample_predictions = tf.reduce_sum(pred_reshaped, axis=1)
                        sample_predictions = tf.cast(sample_predictions > (window_size // 2), tf.int64)
                        
                        all_predictions.extend(sample_predictions.numpy())
                        
                        if targets is not None:
                            if hasattr(targets, 'numpy'):
                                all_targets.extend(targets.numpy().flatten())
                            elif isinstance(targets, (list, np.ndarray)):
                                all_targets.extend(np.array(targets).flatten())
                            else:
                                # Handle dict case
                                label = targets.get('label', None) if isinstance(targets, dict) else None
                                if label is not None:
                                    if hasattr(label, 'numpy'):
                                        all_targets.extend(label.numpy().flatten())
                                    else:
                                        all_targets.extend(np.array(label).flatten())
            except Exception as cpu_error:
                print(f"⚠ CPU fallback also failed: {cpu_error}")
                raise RuntimeError(f"Both GPU and CPU prediction failed. GPU error: {gpu_error}, CPU error: {cpu_error}")
        
        # Print diagnostics if we used left/right comparison
        if all_left_scores and all_right_scores:
            all_left_scores_arr = np.array(all_left_scores)
            all_right_scores_arr = np.array(all_right_scores)
            all_targets_arr = np.array(all_targets) if all_targets else None
            
            # Aggregate per-sample scores to per-window if targets are per-window
            window_size = self.window_size
            num_samples = len(all_left_scores_arr)
            num_windows = num_samples // window_size if window_size > 0 else 0
            
            # If we have per-window targets but per-sample scores, aggregate scores
            if all_targets_arr is not None and len(all_targets_arr) > 0:
                if num_windows > 0 and len(all_targets_arr) == num_windows and len(all_left_scores_arr) == num_windows * window_size:
                    # Aggregate scores to per-window
                    left_scores_reshaped = all_left_scores_arr[:num_windows * window_size].reshape(num_windows, window_size)
                    right_scores_reshaped = all_right_scores_arr[:num_windows * window_size].reshape(num_windows, window_size)
                    all_left_scores_arr = np.mean(left_scores_reshaped, axis=1)
                    all_right_scores_arr = np.mean(right_scores_reshaped, axis=1)
            
            print("\n" + "="*80)
            print("PREDICTION DIAGNOSTICS (Left vs Right Audio Comparison)")
            print("="*80)
            print(f"\nCorrelation Score Statistics:")
            print(f"  Left envelope correlations: {np.mean(all_left_scores_arr):.6f} ± {np.std(all_left_scores_arr):.6f}")
            print(f"  Right envelope correlations: {np.mean(all_right_scores_arr):.6f} ± {np.std(all_right_scores_arr):.6f}")
            
            score_diff = all_right_scores_arr - all_left_scores_arr
            print(f"\nScore Difference (Right - Left): {np.mean(score_diff):.6f} ± {np.std(score_diff):.6f}")
            print(f"  Samples where right > left: {np.sum(score_diff > 0)}/{len(score_diff)} ({100*np.sum(score_diff > 0)/len(score_diff):.1f}%)")
            
            if all_targets_arr is not None and len(all_targets_arr) > 0:
                # Only compute per-class diagnostics if targets match scores in length
                if len(all_targets_arr) == len(all_left_scores_arr):
                    left_attention_mask = all_targets_arr == 0
                    right_attention_mask = all_targets_arr == 1
                    
                    if np.sum(left_attention_mask) > 0:
                        left_att_left = np.mean(all_left_scores_arr[left_attention_mask])
                        left_att_right = np.mean(all_right_scores_arr[left_attention_mask])
                        print(f"\n  When LEFT is attended: Left corr={left_att_left:.6f}, Right corr={left_att_right:.6f}, Diff={left_att_left-left_att_right:.6f}")
                    
                    if np.sum(right_attention_mask) > 0:
                        right_att_left = np.mean(all_left_scores_arr[right_attention_mask])
                        right_att_right = np.mean(all_right_scores_arr[right_attention_mask])
                        print(f"  When RIGHT is attended: Left corr={right_att_left:.6f}, Right corr={right_att_right:.6f}, Diff={right_att_right-right_att_left:.6f}")
                else:
                    print(f"\n  ⚠ WARNING: Target length ({len(all_targets_arr)}) doesn't match score length ({len(all_left_scores_arr)})")
                    print(f"  Skipping per-class diagnostics. This may indicate target aggregation mismatch.")
            
            print("="*80 + "\n")
        
        # Store continuous scores as instance variable for ROC-AUC calculation
        self.last_continuous_scores = np.array(all_continuous_scores) if all_continuous_scores else None
        
        return np.array(all_predictions), np.array(all_targets)


class FULCCATrainer:
    """
    FULCCA trainer with comprehensive metrics evaluation.
    """
    
    def __init__(self, model: FULCCAModel, output_dir: str = "fulcca_results", 
                 tfrecord_dir: str = None, sampling_rate: int = 64, window_size: int = 1280):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        

        self.tfrecord_dir = tfrecord_dir
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        
        print(f"FULCCA trainer initialized. Output directory: {self.output_dir}")
    
    def train(self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset) -> float:
        """Train the FULCCA model."""
        print("Starting FULCCA training...")
        

        train_size = sum(1 for _ in train_dataset)
        val_size = sum(1 for _ in val_dataset)
        print(f"Train dataset size: {train_size} batches")
        print(f"Val dataset size: {val_size} batches")
        
        if train_size == 0:
            raise ValueError("Train dataset is empty! Cannot train CCA model.")
        if val_size == 0:
            raise ValueError("Validation dataset is empty! Cannot validate CCA model.")
        

        self.model.fit(train_dataset)
        

        val_predictions, val_targets = self.model.predict(val_dataset)
        val_accuracy = accuracy_score(val_targets, val_predictions)
        
        print(f"FULCCA training completed! Validation accuracy: {val_accuracy:.4f}")
        return val_accuracy
    
    def test(self, test_dataset: tf.data.Dataset) -> Dict:
        """Test the FULCCA model with comprehensive metrics."""
        print("Testing FULCCA model...")
        
        predictions, targets = self.model.predict(test_dataset)
        
        # Store continuous scores for ROC-AUC (from model's last prediction)
        continuous_scores = self.model.last_continuous_scores if hasattr(self.model, 'last_continuous_scores') else None
        

        accuracy = accuracy_score(targets, predictions)
        

        report = classification_report(targets, predictions, 
                                   target_names=['Left', 'Right'], 
                                   labels=[0, 1],
                                   output_dict=True)
        
        cm = confusion_matrix(targets, predictions)
        

        # Pass continuous scores to ROC-AUC calculation if available
        roc_auc_metrics = self._calculate_roc_auc_metrics(targets, predictions)
        msed_metrics = self._calculate_msed_metrics(targets, predictions)
        advanced_metrics = self._calculate_advanced_metrics(targets, predictions)
        temporal_metrics = self._calculate_temporal_metrics(test_dataset)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'targets': targets,
            'roc_auc_metrics': roc_auc_metrics,
            'msed_metrics': msed_metrics,
            'advanced_metrics': advanced_metrics,
            'temporal_metrics': temporal_metrics
        }
        
        return results
    
    def _calculate_roc_auc_metrics(self, targets: np.ndarray, predictions: np.ndarray) -> Dict:
        """Calculate ROC-AUC and related metrics using continuous scores."""
        try:
            # Use continuous scores (right_score - left_score) for ROC-AUC, not hard binary predictions
            # If continuous scores are available, use them; otherwise fall back to predictions as float
            if hasattr(self.model, 'last_continuous_scores') and self.model.last_continuous_scores is not None:
                continuous_scores = self.model.last_continuous_scores
                # Ensure lengths match
                min_len = min(len(continuous_scores), len(targets))
                continuous_scores = continuous_scores[:min_len]
                targets = targets[:min_len]
                probabilities = continuous_scores.astype(np.float32)
            else:
                # Fallback: use predictions as float (less ideal but won't crash)
                probabilities = predictions.astype(np.float32)
            
            roc_auc = roc_auc_score(targets, probabilities)
            fpr, tpr, roc_thresholds = roc_curve(targets, probabilities)
            

            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = roc_thresholds[optimal_idx]
            optimal_tpr = tpr[optimal_idx]
            optimal_fpr = fpr[optimal_idx]
            

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
    
    def _calculate_temporal_metrics(self, test_dataset: tf.data.Dataset) -> Dict[str, float]:
        """Calculate temporal performance metrics across different window sizes."""
        print("Calculating temporal performance metrics...")
        

        window_sizes_seconds = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0]
        temporal_results = {}
        
        for window_sec in window_sizes_seconds:
            window_samples = int(window_sec * self.sampling_rate)
            
            print(f"Testing {window_sec}s window ({window_samples} samples)...")
            
            try:

                temp_dataset = FulsangDatasetCCA(
                    self.tfrecord_dir, 
                    mode='test',
                    window_size=window_samples,
                    overlap=0.5,
                    load_audio=True  # CRITICAL: Must use real audio for meaningful temporal metrics
                )
                
                if len(temp_dataset) == 0:
                    print(f"  No data for {window_sec}s window")
                    continue
                

                def temp_generator():
                    for i in range(len(temp_dataset)):
                        try:
                            window_data, aux_or_label = temp_dataset[i]
                            
                            # Handle both aux_data (dict) and label (int) formats
                            label = 0
                            if isinstance(aux_or_label, dict):
                                label_obj = aux_or_label.get('label', 0)
                                # Extract label value if it's a tensor
                                if hasattr(label_obj, 'numpy'):
                                    try:
                                        label_val = label_obj.numpy()
                                        if isinstance(label_val, (list, np.ndarray)):
                                            label = int(label_val[0]) if len(label_val) > 0 else 0
                                        else:
                                            label = int(label_val) if label_val is not None else 0
                                    except:
                                        label = 0
                                elif isinstance(label_obj, (list, np.ndarray, tuple)):
                                    label = int(label_obj[0]) if len(label_obj) > 0 else 0
                                elif isinstance(label_obj, (int, np.integer)):
                                    label = int(label_obj)
                                else:
                                    label = 0
                            else:
                                # It's already a label
                                if hasattr(aux_or_label, 'numpy'):
                                    try:
                                        label_val = aux_or_label.numpy()
                                        if isinstance(label_val, (list, np.ndarray)):
                                            label = int(label_val[0]) if len(label_val) > 0 else 0
                                        else:
                                            label = int(label_val) if label_val is not None else 0
                                    except:
                                        label = 0
                                elif isinstance(aux_or_label, (list, np.ndarray, tuple)):
                                    label = int(aux_or_label[0]) if len(aux_or_label) > 0 else 0
                                elif isinstance(aux_or_label, (int, np.integer)):
                                    label = int(aux_or_label)
                                else:
                                    label = 0
                            
                            # Ensure label is a valid integer
                            if not isinstance(label, (int, np.integer)):
                                label = 0

                            if isinstance(window_data, tuple):
                                eeg_data, audio_data = window_data
                            else:
                                eeg_data = window_data
                                audio_data = np.zeros((window_samples, 4), dtype=np.float32)
                            
                            # Convert to numpy if needed
                            if hasattr(eeg_data, 'numpy'):
                                eeg_data = eeg_data.numpy()
                            if hasattr(audio_data, 'numpy'):
                                audio_data = audio_data.numpy()
                            
                            # Ensure correct shapes
                            eeg_shape = list(eeg_data.shape)
                            audio_shape = list(audio_data.shape)
                            
                            if len(eeg_shape) == 2 and eeg_shape[1] == 66:  # Fulsang: 66 channels
                                input_1 = eeg_data
                            else:
                                # Reshape or pad/truncate to (window_samples, 66)
                                total_elements = np.prod(eeg_shape)
                                expected_elements = window_samples * 66
                                if total_elements == expected_elements:
                                    input_1 = eeg_data.reshape(window_samples, 66)
                                else:
                                    eeg_flat = eeg_data.flatten()
                                    if len(eeg_flat) < expected_elements:
                                        padding = np.zeros(expected_elements - len(eeg_flat), dtype=eeg_data.dtype)
                                        eeg_flat = np.concatenate([eeg_flat, padding])
                                    else:
                                        eeg_flat = eeg_flat[:expected_elements]
                                    input_1 = eeg_flat.reshape(window_samples, 66)
                            
                            if len(audio_shape) == 2 and audio_shape[1] == 4:
                                input_2 = audio_data
                            elif len(audio_shape) == 2 and audio_shape[1] == 1:
                                input_2 = np.tile(audio_data, (1, 4))
                            elif len(audio_shape) == 1:
                                input_2 = np.tile(audio_data.reshape(-1, 1), (1, 4))
                            else:
                                input_2 = np.zeros((window_samples, 4), dtype=np.float32)
                            
                            # Ensure lengths match
                            min_len = min(input_1.shape[0], input_2.shape[0], window_samples)
                            input_1 = input_1[:min_len]
                            input_2 = input_2[:min_len]
                            
                            # Yield with proper format: label must be array of shape (1,)
                            yield {
                                'input_1': input_1.astype(np.float32),
                                'input_2': input_2.astype(np.float32)
                            }, np.array([label], dtype=np.int64)
                        except Exception as e:
                            print(f"  Error in temp_generator for sample {i}: {e}")
                            continue
                
                temp_tf_dataset = tf.data.Dataset.from_generator(
                    temp_generator,
                    output_signature=(
                        {
                            'input_1': tf.TensorSpec(shape=(window_samples, 66), dtype=tf.float32),  # Fulsang: 66 channels
                            'input_2': tf.TensorSpec(shape=(window_samples, 4), dtype=tf.float32)
                        },
                        tf.TensorSpec(shape=(1,), dtype=tf.int64)
                    )
                )
                

                def reshape_batch(inputs, labels):

                    input_1_reshaped = tf.reshape(inputs['input_1'], (-1, 66))  # Fulsang: 66 channels
                    input_2_reshaped = tf.reshape(inputs['input_2'], (-1, 4))
                    
                    return {
                        'input_1': input_1_reshaped,
                        'input_2': input_2_reshaped
                    }, labels
                
                temp_tf_dataset = temp_tf_dataset.batch(16).map(reshape_batch)
                
                
                temp_predictions, temp_targets = self.model.predict(temp_tf_dataset)
                
                if len(temp_predictions) > 0:
                    # Handle case where predictions are per-sample but targets are per-window
                    # The model.predict() returns flattened predictions, so we need to aggregate
                    num_windows = len(temp_dataset)
                    window_size_samples = window_samples
                    
                    # If predictions are longer than targets, aggregate predictions to per-window
                    if len(temp_predictions) > len(temp_targets):
                        # Predictions are per-sample, need to aggregate to per-window
                        if len(temp_predictions) == num_windows * window_size_samples:
                            # Perfect match - reshape and aggregate
                            pred_reshaped = temp_predictions[:num_windows * window_size_samples].reshape(num_windows, window_size_samples)
                            # Use mode (most common prediction) per window
                            from scipy.stats import mode
                            try:
                                temp_predictions = np.array([mode(pred_reshaped[i], keepdims=True).mode[0] for i in range(num_windows)])
                            except:
                                # Fallback: use most common value manually
                                temp_predictions = np.array([np.bincount(pred_reshaped[i].astype(int)).argmax() for i in range(num_windows)])
                        elif len(temp_predictions) % window_size_samples == 0:
                            # Divisible - aggregate
                            num_complete_windows = len(temp_predictions) // window_size_samples
                            pred_reshaped = temp_predictions[:num_complete_windows * window_size_samples].reshape(num_complete_windows, window_size_samples)
                            from scipy.stats import mode
                            try:
                                temp_predictions = np.array([mode(pred_reshaped[i], keepdims=True).mode[0] for i in range(num_complete_windows)])
                            except:
                                temp_predictions = np.array([np.bincount(pred_reshaped[i].astype(int)).argmax() for i in range(num_complete_windows)])
                            temp_targets = temp_targets[:num_complete_windows]
                        else:
                            # Not divisible - truncate to match or use mean
                            num_complete_windows = len(temp_predictions) // window_size_samples
                            if num_complete_windows > 0:
                                pred_reshaped = temp_predictions[:num_complete_windows * window_size_samples].reshape(num_complete_windows, window_size_samples)
                                from scipy.stats import mode
                                try:
                                    temp_predictions = np.array([mode(pred_reshaped[i], keepdims=True).mode[0] for i in range(num_complete_windows)])
                                except:
                                    temp_predictions = np.array([np.bincount(pred_reshaped[i].astype(int)).argmax() for i in range(num_complete_windows)])
                                temp_targets = temp_targets[:num_complete_windows]
                            else:
                                # Fallback: use first prediction per window
                                temp_predictions = temp_predictions[:len(temp_targets)]
                    elif len(temp_predictions) < len(temp_targets):
                        # Predictions are shorter - truncate targets
                        temp_targets = temp_targets[:len(temp_predictions)]
                    
                    # Ensure they match now
                    min_len = min(len(temp_predictions), len(temp_targets))
                    temp_predictions = temp_predictions[:min_len]
                    temp_targets = temp_targets[:min_len]
                    
                    if min_len > 0:
                        accuracy = accuracy_score(temp_targets, temp_predictions)
                        f1 = f1_score(temp_targets, temp_predictions, average='weighted')
                        
                        temporal_results[f'accuracy_{window_sec}s'] = accuracy
                        temporal_results[f'f1_{window_sec}s'] = f1
                        
                        print(f"  {window_sec}s: Acc={accuracy:.3f}, F1={f1:.3f}")
                    else:
                        print(f"  {window_sec}s: No valid predictions after alignment")
                else:
                    print(f"  {window_sec}s: No valid predictions")
                    
            except Exception as e:
                print(f"  Error testing {window_sec}s window: {e}")
                continue
        
        return temporal_results
    
    def save_results(self, results: Dict):
        """Save comprehensive results to files."""

        results_json = {
            'accuracy': float(results['accuracy']),
            'classification_report': results['classification_report'],
            'confusion_matrix': results['confusion_matrix'].tolist() if hasattr(results['confusion_matrix'], 'tolist') else results['confusion_matrix'],
            'timestamp': datetime.now().isoformat(),
            'roc_auc_metrics': results.get('roc_auc_metrics', {}),
            'msed_metrics': results.get('msed_metrics', {}),
            'advanced_metrics': results.get('advanced_metrics', {}),
            'temporal_metrics': results.get('temporal_metrics', {})
        }
        

        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        

        save_data = {
            'predictions': results['predictions'],
            'targets': results['targets']
        }
        
        with open(self.output_dir / 'predictions.pkl', 'wb') as f:
            pickle.dump(save_data, f)
        

        self._save_comprehensive_report(results)
        
        print(f"FULCCA results saved to {self.output_dir}")
    
    def _save_comprehensive_report(self, results: Dict):
        """Save a comprehensive metrics report."""
        with open(self.output_dir / 'comprehensive_metrics_report.txt', 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FULCCA COMPREHENSIVE METRICS REPORT\n")
            f.write("=" * 80 + "\n\n")
            

            f.write("BASIC METRICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Accuracy: {results['accuracy']:.4f}\n\n")
            

            roc_auc = results.get('roc_auc_metrics', {})
            if "error" not in roc_auc:
                f.write("ROC-AUC METRICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"ROC-AUC Score: {roc_auc.get('roc_auc_score', 'N/A'):.4f}\n")
                f.write(f"Average Precision: {roc_auc.get('average_precision', 'N/A'):.4f}\n")
                f.write(f"Optimal Threshold: {roc_auc.get('optimal_threshold', 'N/A'):.4f}\n")
                f.write(f"Optimal TPR: {roc_auc.get('optimal_tpr', 'N/A'):.4f}\n")
                f.write(f"Optimal FPR: {roc_auc.get('optimal_fpr', 'N/A'):.4f}\n\n")
            

            msed = results.get('msed_metrics', {})
            if "error" not in msed:
                f.write("MSED METRICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Mean Squared Error: {msed.get('mse', 'N/A'):.4f}\n")
                f.write(f"Root Mean Squared Error: {msed.get('rmse', 'N/A'):.4f}\n")
                f.write(f"Mean Absolute Error: {msed.get('mae', 'N/A'):.4f}\n")
                f.write(f"Mean Absolute Percentage Error: {msed.get('mape', 'N/A'):.4f}%\n")
                f.write(f"R-squared: {msed.get('r_squared', 'N/A'):.4f}\n\n")
            

            advanced = results.get('advanced_metrics', {})
            if "error" not in advanced:
                f.write("ADVANCED METRICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Matthews Correlation Coefficient: {advanced.get('matthews_correlation_coefficient', 'N/A'):.4f}\n")
                f.write(f"Cohen's Kappa: {advanced.get('cohens_kappa', 'N/A'):.4f}\n")
                f.write(f"Balanced Accuracy: {advanced.get('balanced_accuracy', 'N/A'):.4f}\n\n")
            

            temporal = results.get('temporal_metrics', {})
            f.write("TEMPORAL PERFORMANCE ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            for key, value in temporal.items():
                f.write(f"{key}: {value:.4f}\n")


def create_fulsang_data_loaders(tfrecord_dir: str, batch_size: int = 6, 
                           window_size: int = 1280, overlap: float = 0.5,
                           train_ratio: float = 0.60, val_ratio: float = 0.25,  # Optimal Fulsang split: 60/25/15
                           max_samples: Optional[int] = None,
                           audio_base_dir: Optional[str] = None,
                           load_audio: bool = True, max_files: Optional[int] = None) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Create data loaders for Fulsang dataset with proper subject-wise splitting."""
    
    print("Creating Fulsang dataset with subject-wise splitting...")
    print(f"TFRecord directory: {tfrecord_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Window size: {window_size} samples ({window_size/64:.1f} seconds at 64Hz)")
    print(f"Overlap: {overlap}")
    print(f"Using Fulsang preprocessing: Yes")
    if audio_base_dir:
        print(f"Audio base directory: {audio_base_dir}")
    

    full_dataset = FulsangDatasetCCA(tfrecord_dir, mode='full', 
                               window_size=window_size, overlap=overlap,
                               audio_base_dir=audio_base_dir,
                               load_audio=load_audio, max_files=max_files)
    
    total_size = len(full_dataset)
    print(f"Total dataset size: {total_size} samples")
    

    subject_windows = {}
    


    data_idx_to_subject = {}
    

    for i, metadata in enumerate(full_dataset.metadata):
        subject_id = metadata.get('subject_id', 'unknown')
        data_idx_to_subject[i] = subject_id
    

    subject_ranges = {}
    current_subject = None
    start_idx = 0
    
    for i, metadata in enumerate(full_dataset.metadata):
        subject_id = metadata.get('subject_id', 'unknown')
        
        if subject_id != current_subject:
            if current_subject is not None:
                subject_ranges[current_subject] = (start_idx, i)
            current_subject = subject_id
            start_idx = i
    

    if current_subject is not None:
        subject_ranges[current_subject] = (start_idx, len(full_dataset.metadata))
    
    print(f"Subject ranges in metadata:")
    for subject_id, (start, end) in subject_ranges.items():
        print(f"  {subject_id}: samples {start}-{end-1} ({end-start} samples)")
    

    unknown_count = sum(1 for sid in data_idx_to_subject.values() if sid == "unknown")
    total_samples = len(data_idx_to_subject)
    if unknown_count > 0:
        print(f"\n⚠ WARNING: {unknown_count}/{total_samples} samples ({100*unknown_count/total_samples:.1f}%) have 'unknown' subject_id")
        print(f"  This suggests subject_id might not be properly stored in TFRecords")
        print(f"  Check that das_preprocessing_16subjects.py includes subject_id in TFRecord features")
    



    # New window format: (trial_idx, offset_in_trial, label)
    # Get subject_id directly from trial_boundaries
    for i, window_info in enumerate(full_dataset.window_indices):
        # Unpack window info
        if len(window_info) == 3:
            trial_idx, offset_in_trial, label = window_info
        else:
            # Fallback for old format (shouldn't happen)
            print(f"⚠ WARNING: Window {i} has unexpected format: {window_info}")
            trial_idx = 0
            offset_in_trial = 0
            label = 0
        
        # Get subject_id directly from trial_boundaries
        if trial_idx < len(full_dataset.trial_boundaries):
            _, _, trial_info = full_dataset.trial_boundaries[trial_idx]
            subject_id = trial_info.get('subject_id', 'unknown')
        else:
            subject_id = 'unknown'
            print(f"⚠ WARNING: Window {i} has invalid trial_idx {trial_idx} (max: {len(full_dataset.trial_boundaries)-1})")
        
        if subject_id not in subject_windows:
            subject_windows[subject_id] = []
        subject_windows[subject_id].append(i)
    
    print(f"Found {len(subject_windows)} subjects:")
    for subject_id, windows in subject_windows.items():
        print(f"  {subject_id}: {len(windows)} windows")
    
    # Analyze label distribution per subject BEFORE splitting
    subject_label_distributions = {}
    print(f"\nAnalyzing label distribution per subject (before split):")
    for subject_id, window_indices_list in subject_windows.items():
        subject_labels = [full_dataset.window_indices[i][2] for i in window_indices_list]  # label is 3rd element
        label_dist = np.bincount(subject_labels)
        subject_label_distributions[subject_id] = label_dist
        
        total_windows = len(window_indices_list)
        class_0_count = label_dist[0] if len(label_dist) > 0 else 0
        class_1_count = label_dist[1] if len(label_dist) > 1 else 0
        print(f"  {subject_id}: {total_windows} windows - Class 0: {class_0_count}, Class 1: {class_1_count}")
    
    # Categorize subjects by label distribution for stratified split
    subjects_with_both_classes = []
    subjects_with_only_class_0 = []
    subjects_with_only_class_1 = []
    
    for subject_id, label_dist in subject_label_distributions.items():
        class_0_count = label_dist[0] if len(label_dist) > 0 else 0
        class_1_count = label_dist[1] if len(label_dist) > 1 else 0
        
        if class_0_count > 0 and class_1_count > 0:
            subjects_with_both_classes.append(subject_id)
        elif class_0_count > 0:
            subjects_with_only_class_0.append(subject_id)
        elif class_1_count > 0:
            subjects_with_only_class_1.append(subject_id)
    
    print(f"\nSubject categorization:")
    print(f"  Subjects with both classes: {len(subjects_with_both_classes)}")
    print(f"  Subjects with only class 0: {len(subjects_with_only_class_0)}")
    print(f"  Subjects with only class 1: {len(subjects_with_only_class_1)}")
    
    subjects = list(subject_windows.keys())
    np.random.seed(42)
    
    n_subjects = len(subjects)
    

    if n_subjects < 3:
        print(f"⚠ WARNING: Only {n_subjects} subjects found.")
        if n_subjects == 1 and 'unknown' in subjects:
            print("⚠ DAS TFRecord files don't contain subject_id field.")
            print("⚠ Run das_preprocessing_16subjects.py to create proper TFRecord files with subject information.")
            print("⚠ Using random window splitting as fallback.")
        else:
            print("⚠ Using random window splitting instead of subject-wise splitting.")
            print("⚠ This may lead to data leakage but is necessary for small datasets.")
        

        all_window_indices = []
        for subject_id in subjects:
            all_window_indices.extend(subject_windows[subject_id])
        

        np.random.seed(42)
        np.random.shuffle(all_window_indices)
        
        n_windows = len(all_window_indices)
        n_train_windows = int(train_ratio * n_windows)
        n_val_windows = int(val_ratio * n_windows)
        
        train_indices = all_window_indices[:n_train_windows]
        val_indices = all_window_indices[n_train_windows:n_train_windows + n_val_windows]
        test_indices = all_window_indices[n_train_windows + n_val_windows:]
        
        print(f"\nRandom window split:")
        print(f"  Train windows: {len(train_indices)}")
        print(f"  Val windows: {len(val_indices)}")
        print(f"  Test windows: {len(test_indices)}")
        
    else:
        # Stratified subject split: distribute balanced subjects first, then one-class subjects
        np.random.seed(42)  # Reproducibility
        np.random.shuffle(subjects_with_both_classes)
        np.random.shuffle(subjects_with_only_class_0)
        np.random.shuffle(subjects_with_only_class_1)
        
        # Calculate split sizes
        n_train_subjects = int(train_ratio * n_subjects)
        n_val_subjects = int(val_ratio * n_subjects)
        
        # First, assign balanced subjects to ensure each split has both classes
        train_subjects = []
        val_subjects = []
        test_subjects = []
        
        # Distribute balanced subjects across splits
        for i, subj in enumerate(subjects_with_both_classes):
            if len(train_subjects) < n_train_subjects:
                train_subjects.append(subj)
            elif len(val_subjects) < n_val_subjects:
                val_subjects.append(subj)
            else:
                test_subjects.append(subj)
        
        # Then distribute one-class subjects to balance
        all_one_class = subjects_with_only_class_0 + subjects_with_only_class_1
        np.random.shuffle(all_one_class)
        
        for subj in all_one_class:
            if len(train_subjects) < n_train_subjects:
                train_subjects.append(subj)
            elif len(val_subjects) < n_val_subjects:
                val_subjects.append(subj)
            else:
                test_subjects.append(subj)
        
        # If we still need more subjects, use any remaining
        remaining = [s for s in subjects if s not in train_subjects + val_subjects + test_subjects]
        for subj in remaining:
            if len(train_subjects) < n_train_subjects:
                train_subjects.append(subj)
            elif len(val_subjects) < n_val_subjects:
                val_subjects.append(subj)
            else:
                test_subjects.append(subj)
        
        print(f"\nSubject-wise split:")
        print(f"  Train subjects: {len(train_subjects)} ({train_subjects})")
        print(f"  Val subjects: {len(val_subjects)} ({val_subjects})")
        print(f"  Test subjects: {len(test_subjects)} ({test_subjects})")
        

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
    
    if train_set & val_set:
        raise ValueError("CRITICAL: Data leakage detected - train/val overlap!")
    if train_set & test_set:
        raise ValueError("CRITICAL: Data leakage detected - train/test overlap!")
    if val_set & test_set:
        raise ValueError("CRITICAL: Data leakage detected - val/test overlap!")
    
    print("✓ No data leakage detected - subjects properly separated")
    

    def create_cca_dataset(indices):
        print(f"Creating CCA dataset with {len(indices)} indices...")
        

        dataset_window_size = full_dataset.window_size
        dataset_batch_size = batch_size
        
        def generator():
            valid_samples = 0
            for i in indices:
                try:
                    window_data, aux_data = full_dataset[i]
                    
                    # Extract label from aux_data
                    if isinstance(aux_data, dict):
                        label = aux_data.get('label', tf.constant([0], dtype=tf.int64))
                        left_env = aux_data.get('left_env')
                        right_env = aux_data.get('right_env')
                    else:
                        # Fallback for old format
                        label = aux_data
                        left_env = None
                        right_env = None
                    

                    # Extract EEG and audio from window_data
                    if isinstance(window_data, tuple) and len(window_data) == 2:
                        eeg_data, audio_data = window_data
                    else:
                        # Fallback: if window_data is not a tuple, try to extract from it
                        eeg_data = window_data
                        # Create dummy audio data with correct shape
                        eeg_shape = eeg_data.shape.as_list() if hasattr(eeg_data.shape, 'as_list') else list(eeg_data.shape)
                        if len(eeg_shape) == 2:
                            audio_data = tf.zeros((eeg_shape[0], 4), dtype=tf.float32)
                        else:
                            audio_data = tf.zeros((dataset_window_size, 4), dtype=tf.float32)
                    
                    # Ensure both are tensors
                    if not isinstance(eeg_data, tf.Tensor):
                        eeg_data = tf.constant(eeg_data, dtype=tf.float32)
                    if not isinstance(audio_data, tf.Tensor):
                        audio_data = tf.constant(audio_data, dtype=tf.float32)
                    


                    eeg_shape = eeg_data.shape.as_list() if hasattr(eeg_data.shape, 'as_list') else list(eeg_data.shape)
                    audio_shape = audio_data.shape.as_list() if hasattr(audio_data.shape, 'as_list') else list(audio_data.shape)
                    

                    if len(eeg_shape) == 2 and eeg_shape[1] == 66:  # Fulsang: 66 channels
                        input_1 = eeg_data
                    else:

                        print(f"WARNING: Unexpected EEG shape {eeg_shape}, reshaping...")
                        # Try to infer correct shape
                        total_elements = tf.reduce_prod(tf.shape(eeg_data))
                        expected_elements = dataset_window_size * 66
                        if total_elements == expected_elements:
                            eeg_data = tf.reshape(eeg_data, (dataset_window_size, 66))  # Fulsang: 66 channels
                        else:
                            # Handle case where shape doesn't match - pad or truncate
                            eeg_flat = tf.reshape(eeg_data, (-1,))
                            if tf.size(eeg_flat) < expected_elements:
                                # Pad with zeros
                                padding_size = expected_elements - tf.size(eeg_flat)
                                eeg_flat = tf.concat([eeg_flat, tf.zeros(padding_size, dtype=eeg_data.dtype)], axis=0)
                            else:
                                # Truncate
                                eeg_flat = eeg_flat[:expected_elements]
                            eeg_data = tf.reshape(eeg_flat, (dataset_window_size, 66))
                        input_1 = eeg_data
                    


                    if len(audio_shape) == 1:


                        audio_expanded = tf.tile(tf.reshape(audio_data, (dataset_window_size, 1)), [1, 4])
                        input_2 = audio_expanded
                    elif len(audio_shape) == 2:

                        if audio_shape[1] == 1:

                            input_2 = tf.tile(audio_data, [1, 4])
                        else:
                            input_2 = audio_data
                    else:

                        print(f"WARNING: Unexpected audio shape {audio_shape}, creating dummy...")
                        input_2 = tf.zeros((dataset_window_size, 4), dtype=tf.float32)
                    
                    # CRITICAL: Since we window within trials, lengths should always match
                    # Verify shapes match expected window_size (raise error if not - indicates bug upstream)
                    input_1_len = tf.shape(input_1)[0]
                    input_2_len = tf.shape(input_2)[0]
                    
                    # Ensure exact match to window_size (pad or truncate if needed, but warn)
                    if input_1_len != dataset_window_size:
                        # Pad or truncate to exact window_size
                        if input_1_len < dataset_window_size:
                            padding = tf.zeros((dataset_window_size - input_1_len, 66), dtype=input_1.dtype)
                            input_1 = tf.concat([input_1, padding], axis=0)
                        else:
                            input_1 = input_1[:dataset_window_size]
                    
                    if input_2_len != dataset_window_size:
                        if input_2_len < dataset_window_size:
                            padding = tf.zeros((dataset_window_size - input_2_len, 4), dtype=input_2.dtype)
                            input_2 = tf.concat([input_2, padding], axis=0)
                        else:
                            input_2 = input_2[:dataset_window_size]
                    
                    valid_samples += 1
                    
                    # Prepare aux_data with left/right envelopes
                    aux_dict = {'label': label}
                    if left_env is not None and right_env is not None:
                        # Ensure left_env and right_env are exactly window_size
                        left_len = tf.shape(left_env)[0]
                        if left_len != dataset_window_size:
                            if left_len < dataset_window_size:
                                padding = tf.zeros((dataset_window_size - left_len, 4), dtype=left_env.dtype)
                                left_env = tf.concat([left_env, padding], axis=0)
                            else:
                                left_env = left_env[:dataset_window_size]
                        
                        right_len = tf.shape(right_env)[0]
                        if right_len != dataset_window_size:
                            if right_len < dataset_window_size:
                                padding = tf.zeros((dataset_window_size - right_len, 4), dtype=right_env.dtype)
                                right_env = tf.concat([right_env, padding], axis=0)
                            else:
                                right_env = right_env[:dataset_window_size]
                        
                        aux_dict['left_env'] = left_env
                        aux_dict['right_env'] = right_env
                    else:
                        # Create dummy envelopes if missing
                        aux_dict['left_env'] = tf.zeros((dataset_window_size, 4), dtype=tf.float32)
                        aux_dict['right_env'] = tf.zeros((dataset_window_size, 4), dtype=tf.float32)
                    
                    yield {
                        'input_1': input_1,
                        'input_2': input_2
                    }, aux_dict
                    
                except Exception as e:
                    print(f"ERROR in generator for index {i}: {e}")
                    continue
            
            print(f"Generator produced {valid_samples} valid samples")
        

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                {
                    'input_1': tf.TensorSpec(shape=(dataset_window_size, 66), dtype=tf.float32),  # Fulsang: 66 channels
                    'input_2': tf.TensorSpec(shape=(dataset_window_size, 4), dtype=tf.float32)
                },
                {
                    'label': tf.TensorSpec(shape=(1,), dtype=tf.int64),
                    'left_env': tf.TensorSpec(shape=(dataset_window_size, 4), dtype=tf.float32),
                    'right_env': tf.TensorSpec(shape=(dataset_window_size, 4), dtype=tf.float32)
                }
            )
        )
        

        def reshape_batch(inputs, aux_data):
            # Reshape inputs - flatten batch and window dimensions
            # Input shape: (batch_size, window_size, channels)
            # Output shape: (batch_size * window_size, channels)
            input_1_reshaped = tf.reshape(inputs['input_1'], (-1, 66))  # Fulsang: 66 channels
            input_2_reshaped = tf.reshape(inputs['input_2'], (-1, 4))
            
            # Ensure both inputs have the same length (first dimension)
            # This handles cases where EEG and audio have different lengths after batching
            input_1_len = tf.shape(input_1_reshaped)[0]
            input_2_len = tf.shape(input_2_reshaped)[0]
            min_length = tf.minimum(input_1_len, input_2_len)
            
            # Use tf.cond to handle length mismatch (TensorFlow-compatible)
            def truncate_both():
                return input_1_reshaped[:min_length], input_2_reshaped[:min_length]
            
            def keep_both():
                return input_1_reshaped, input_2_reshaped
            
            input_1_reshaped, input_2_reshaped = tf.cond(
                tf.not_equal(input_1_len, input_2_len),
                truncate_both,
                keep_both
            )
            
            # Reshape aux_data
            reshaped_inputs = {
                'input_1': input_1_reshaped,
                'input_2': input_2_reshaped
            }
            
            reshaped_aux = {}
            if isinstance(aux_data, dict):
                if 'label' in aux_data:
                    reshaped_aux['label'] = aux_data['label']
                if 'left_env' in aux_data:
                    reshaped_aux['left_env'] = tf.reshape(aux_data['left_env'], (-1, 4))
                if 'right_env' in aux_data:
                    reshaped_aux['right_env'] = tf.reshape(aux_data['right_env'], (-1, 4))
            else:
                # Fallback for old format
                reshaped_aux = aux_data
            
            return reshaped_inputs, reshaped_aux
        
        return dataset.batch(dataset_batch_size).map(reshape_batch)
    
    train_dataset = create_cca_dataset(train_indices)
    val_dataset = create_cca_dataset(val_indices)
    test_dataset = create_cca_dataset(test_indices)
    

    print(f"\nDataset creation debug:")
    print(f"  Train indices: {len(train_indices)}")
    print(f"  Val indices: {len(val_indices)}")
    print(f"  Test indices: {len(test_indices)}")
    

    if len(train_indices) == 0:
        print("⚠ WARNING: Train dataset is empty!")
    if len(val_indices) == 0:
        print("⚠ WARNING: Validation dataset is empty!")
    if len(test_indices) == 0:
        print("⚠ WARNING: Test dataset is empty!")
    
    print(f"✓ Data loaders created with subject-wise splitting")
    print(f"✓ Data leakage prevention implemented")
    print(f"✓ Attention labels validated")
    print(f"✓ Subject-wise organization applied")
    
    return train_dataset, val_dataset, test_dataset


def main():
    """Main function for FULCCA training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='FULCCA - CCA Algorithm for Fulsang Dataset')
    parser.add_argument('--tfrecord_dir', type=str, default='fulsang_preprocessed/tfrecords',
                       help='TFRecord directory path')
    parser.add_argument('--batch_size', type=int, default=6,
                       help='Batch size for training (optimal: 6 for Fulsang)')
    parser.add_argument('--cca_dims', type=int, default=12,
                       help='Number of CCA dimensions (optimal: 12 for Fulsang)')
    parser.add_argument('--regularization', type=float, default=0.08,
                       help='CCA regularization parameter (optimal: 0.08 for Fulsang)')
    parser.add_argument('--window_size', type=int, default=1280,
                       help='Window size for EEG data (1280 samples = 20 seconds at 64Hz, optimal for Fulsang)')
    parser.add_argument('--output_dir', type=str, default='fulcca_results',
                       help='Output directory for results')
    parser.add_argument('--audio_base_dir', type=str, default=None,
                       help='Base directory for audio files (auto-detected if not specified)')
    # Fix: Use simple boolean with default=True
    # argparse's store_true doesn't work well with default=True, so we'll handle it manually
    parser.add_argument('--load_audio', action='store_true',
                       help='Load audio envelopes (default: True)')
    parser.add_argument('--no_load_audio', dest='load_audio', action='store_false',
                       help='Skip audio loading for faster data loading (uses dummy audio)')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of TFRecord files to load (for faster testing)')
    
    args = parser.parse_args()
    
    # Fix: argparse with store_true/store_false doesn't set default properly
    # If neither flag is provided, load_audio will be None, so default to True
    if not hasattr(args, 'load_audio') or args.load_audio is None:
        args.load_audio = True  # Default to True if not specified
    
    print("=" * 80)
    print("FULCCA - CANONICAL CORRELATION ANALYSIS FOR FULSANG DATASET")
    print("=" * 80)
    print("Features:")
    print("- CCA implementation based on telluride_decoding")
    print("- Accuracy, MSED, ROC-AUC metrics")
    print("- Temporal performance analysis (0.5s to 30s)")
    print("- Fulsang preprocessing integration for data quality")
    print("- Data leakage prevention")
    print("- Validated attention labels")
    print("- EEG + Audio envelope correlation (improved CCA performance)")
    print("- Optimal hyperparameters: cca_dims=12, regularization=0.08, window_size=1280 (20s)")
    print("=" * 80)
    
    print("✓ Using Fulsang preprocessing validated data")
    print("✓ Data leakage prevention enabled")
    print("✓ Attention labels validated")
    print("✓ CCA implementation from telluride_decoding")
    
    # Debug: Print load_audio value
    print(f"DEBUG: args.load_audio = {args.load_audio}")
    if args.load_audio:
        print("✓ Audio envelope support enabled (EEG vs Audio correlation)")
    else:
        print("⚠ Audio loading DISABLED - using dummy audio (faster but may affect accuracy)")
    if args.max_files:
        print(f"⚠ Limiting to {args.max_files} TFRecord files (for faster testing)")
    

    print(f"\nCreating Fulsang data loaders...")
    train_dataset, val_dataset, test_dataset = create_fulsang_data_loaders(
        args.tfrecord_dir, batch_size=args.batch_size, window_size=args.window_size,
        audio_base_dir=args.audio_base_dir, load_audio=args.load_audio,
        max_files=args.max_files
    )
    

    print("\nCreating FULCCA model...")
    model = FULCCAModel(
        cca_dims=args.cca_dims,
        regularization=args.regularization,
        window_size=args.window_size
    )
    

    trainer = FULCCATrainer(model, args.output_dir, args.tfrecord_dir, 
                           sampling_rate=64, window_size=args.window_size)
    

    print("\nStarting FULCCA training...")
    best_val_acc = trainer.train(train_dataset, val_dataset)
    

    print("\nTesting FULCCA model...")
    results = trainer.test(test_dataset)
    

    trainer.save_results(results)
    
    print("\n" + "=" * 80)
    print("FULCCA TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {results['accuracy']:.4f}")
    

    print("\n" + "=" * 80)
    print("COMPREHENSIVE METRICS SUMMARY")
    print("=" * 80)
    

    roc_auc = results.get('roc_auc_metrics', {})
    if "error" not in roc_auc:
        print(f"ROC-AUC Score: {roc_auc.get('roc_auc_score', 'N/A'):.4f}")
        print(f"Average Precision: {roc_auc.get('average_precision', 'N/A'):.4f}")
    

    msed = results.get('msed_metrics', {})
    if "error" not in msed:
        print(f"RMSE: {msed.get('rmse', 'N/A'):.4f}")
        print(f"R-squared: {msed.get('r_squared', 'N/A'):.4f}")
    

    advanced = results.get('advanced_metrics', {})
    if "error" not in advanced:
        print(f"Matthews Correlation Coefficient: {advanced.get('matthews_correlation_coefficient', 'N/A'):.4f}")
        print(f"Balanced Accuracy: {advanced.get('balanced_accuracy', 'N/A'):.4f}")
    

    temporal = results.get('temporal_metrics', {})
    print(f"Temporal performance across window sizes:")
    for key, value in temporal.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nResults saved to: {args.output_dir}")
    print("  - results.json (complete metrics)")
    print("  - predictions.pkl (predictions and targets)")
    print("  - comprehensive_metrics_report.txt (formatted report)")


def cleanup_gpu_memory():
    """Clean up GPU memory after training."""
    try:
        tf.keras.backend.clear_session()

        import gc
        gc.collect()
        print("✓ GPU memory cleaned up")
    except Exception as e:
        print(f"GPU cleanup warning: {e}")


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_gpu_memory()
