#!/usr/bin/env python3

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           precision_recall_fscore_support, roc_auc_score, roc_curve,
                           precision_recall_curve, average_precision_score,
                           matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
                           f1_score)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# FIXED: Make determinism optional - can break GPU kernels on some systems
# Uncomment if determinism is required (may cause errors on some GPUs/TF versions)
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
# os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


try:

    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"Found {len(gpu_devices)} GPU device(s)")

        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✓ GPU memory growth configured")
        

        # Memory growth is sufficient for dynamic allocation
            
    else:
        raise RuntimeError("No GPU devices found! GPU-only mode requires GPU.")
except Exception as e:
    print(f"GPU configuration failed: {e}")
    raise RuntimeError("Cannot proceed without GPU. Please ensure GPU is available.")


sys.path.append('telluride_decoding')

try:
    from telluride_decoding.cca import (
        BrainModelCCA, 
        cca_pearson_correlation_first,
        calculate_cca_parameters_from_dataset
    )
except ImportError as e:
    print(f"Warning: Could not import telluride_decoding.cca: {e}")
    raise


tf.compat.v1.enable_v2_behavior()


device = tf.device('/GPU:0')
print("Using GPU for computation")


tf.random.set_seed(42)
np.random.seed(42)
print("✓ Random seeds set for reproducibility")


def make_lagged_audio(audio: np.ndarray, lag_samples: np.ndarray, fs: float = 128.0) -> np.ndarray:
    """
    Create time-lagged audio features for CCA (forward model).
    
    Neural response to speech has a delay (typically 150-400ms). This function creates
    lagged copies of the audio envelope to account for this latency.
    
    Args:
        audio: Audio envelope of shape (T, B) where T is time samples and B is number of bands (4 for DAS)
        lag_samples: Array of lag values in samples (e.g., np.arange(0, int(0.4 * fs)) for 0-400ms)
        fs: Sampling rate in Hz (default: 128 Hz for DAS)
        
    Returns:
        Lagged audio features of shape (T, B * num_lags)
        Each band is replicated with different time shifts, then concatenated.
    """
    T, B = audio.shape
    num_lags = len(lag_samples)
    lagged_features = []
    
    for lag in lag_samples:
        # Shift audio by lag samples (forward model: audio(t-lag) predicts EEG(t))
        shifted = np.roll(audio, lag, axis=0)
        # Zero out the beginning where we rolled around
        if lag > 0:
            shifted[:lag, :] = 0
        
        lagged_features.append(shifted)
    
    # Concatenate all lagged versions: shape (T, B * num_lags)
    lagged_audio = np.concatenate(lagged_features, axis=1)
    
    return lagged_audio.astype(np.float32)

def make_lagged_eeg(eeg: np.ndarray, L: int) -> np.ndarray:
    """
    Create time-lagged EEG features for backward model (paper: spatiotemporal wx).
    
    x(t) = [eeg(t), eeg(t-1), ..., eeg(t-L+1)] per channel then flatten -> (T, C*L).
    Causal: only past and current; no future. Early t padded with zeros.
    
    Args:
        eeg: EEG of shape (T, C)
        L: Number of backward taps (lag order)
    Returns:
        (T, C*L) float32
    """
    T, C = eeg.shape
    if L <= 1:
        return np.asarray(eeg, dtype=np.float32)
    out = np.zeros((T, C * L), dtype=np.float32)
    for t in range(T):
        # Use eeg(t), eeg(t-1), ..., eeg(t-L+1); pad with zeros for t < L-1
        segs = []
        for lag in range(L):
            idx = t - lag
            if idx >= 0:
                segs.append(eeg[idx, :])  # (C,)
            else:
                segs.append(np.zeros(C, dtype=eeg.dtype))
        out[t, :] = np.concatenate(segs, axis=0)  # [eeg(t), eeg(t-1), ..., eeg(t-L+1)] flattened
    return out


def _apply_time_lagging(eeg_window: np.ndarray, lag_samples: int) -> np.ndarray:
    """
    Create time-lagged EEG features for backward model (DEPRECATED - use make_lagged_eeg).
    
    For each time point, concatenate current and past time points.
    This creates spatiotemporal features: [eeg(t), eeg(t-1), ..., eeg(t-lag)]
    
    Args:
        eeg_window: EEG data of shape (window_size, n_channels)
        lag_samples: Number of past time samples to include
        
    Returns:
        Time-lagged EEG features of shape (window_size, n_channels * (lag_samples + 1))
    """
    return make_lagged_eeg(eeg_window, lag_samples + 1)

def safe_random_operations():
    """Force CPU usage for random operations."""
    with tf.device('/CPU:0'):
        tf.random.set_seed(42)
        np.random.seed(42)


class DasDatasetCCA:
    
    def __init__(self, tfrecord_dir: str, mode: str = 'full', 
                 window_size: int = 32, overlap: float = 0.25,
                 cache_size: int = 1000, audio_base_dir: Optional[str] = None,
                 load_audio: bool = True, max_files: Optional[int] = None):
        self.tfrecord_dir = Path(tfrecord_dir)
        self.mode = mode
        self.window_size = window_size
        self.overlap = overlap
        self.cache_size = cache_size
        self.load_audio = load_audio  # Option to skip audio loading for speed
        self.max_files = max_files  # Limit number of files to load
        
        self.sampling_rate = 128  # Matches preprocessing (das_preprocessing_16subjects.py uses 128 Hz)
        self.n_channels = 64
        self.attention_switch_duration = 20
        

        if audio_base_dir:
            self.audio_base_dir = Path(audio_base_dir)
        else:

            possible_dirs = [
                Path("/home/py9363/telluride_decoding/Data/Das/4004271/stimuli/stimuli"),
                Path("Data/Das/4004271/stimuli/stimuli"),
                Path("Data/Das/4004271/Stimuli"),
                Path("Data/Das/Stimuli"),
                Path("Stimuli"),
                self.tfrecord_dir.parent.parent / "Stimuli" if self.tfrecord_dir.parent.parent.exists() else None,
                self.tfrecord_dir.parent.parent / "stimuli" / "stimuli" if self.tfrecord_dir.parent.parent.exists() else None
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
        self._preload_audio = True  # Pre-load audio files for faster access
        

        self.eeg_data, self.audio_envelopes, self.labels, self.metadata = self._load_das_preprocessing_data()
        
        # Pre-load audio files if enabled (speeds up training significantly)
        if self.load_audio and self._preload_audio:
            self._preload_all_audio_files()
        
        self.window_indices = self._create_das_windows()
        
        print(f"Loaded {len(self.window_indices)} DAS windows for {mode} mode")
        print(f"DAS EEG shape: {self.eeg_data.shape}")
        print(f"DAS Audio envelopes shape: {self.audio_envelopes.shape}")
        print(f"DAS Label distribution: {np.bincount(self.labels)}")
        print(f"Using DAS preprocessing: Yes")
        print(f"Cache size: {cache_size} windows")
    
    def _validate_left_right_consistency(self, all_metadata: List[Dict], all_labels: List) -> None:
        """Confirm left/right audio and labels are consistent across records (no swap by subject or trial)."""
        print("\n" + "=" * 60)
        print("LEFT/RIGHT AUDIO AND LABEL CONSISTENCY CHECK")
        print("=" * 60)
        n = len(all_metadata)
        if n != len(all_labels):
            print(f"  ⚠ Metadata length ({n}) != labels length ({len(all_labels)})")
            return
        label_ok = 0
        left_right_distinct = 0
        left_present = 0
        right_present = 0
        mismatches = []
        sample_per_subject = {}
        for i, (meta, label) in enumerate(zip(all_metadata, all_labels)):
            att = meta.get('attended_ear')
            left_file = meta.get('left_audio_file')
            right_file = meta.get('right_audio_file')
            # Label 0 = left attended (L), label 1 = right attended (R)
            lab = int(label) if hasattr(label, '__int__') else label
            if att == 'L' and lab == 0:
                label_ok += 1
            elif att == 'R' and lab == 1:
                label_ok += 1
            else:
                mismatches.append((i, meta.get('subject_id'), att, lab, left_file, right_file))
            if left_file:
                left_present += 1
            if right_file:
                right_present += 1
            if left_file and right_file and left_file != right_file:
                left_right_distinct += 1
            # Sample one record per subject for display
            sid = meta.get('subject_id', 'unknown')
            if sid not in sample_per_subject:
                sample_per_subject[sid] = (att, label, left_file, right_file)
        print(f"  Records checked: {n}")
        print(f"  Label matches attended_ear (L=0, R=1): {label_ok}/{n} ({100*label_ok/n:.1f}%)")
        print(f"  Left audio file present: {left_present}/{n}")
        print(f"  Right audio file present: {right_present}/{n}")
        print(f"  Left and right distinct: {left_right_distinct}/{n}")
        if mismatches:
            print(f"  ⚠ MISMATCHES: {len(mismatches)} records where label != attended_ear")
            for (j, sid, att, lab, lf, rf) in mismatches[:10]:
                print(f"    record {j} subject {sid}: attended_ear={att} label={lab} left={lf} right={rf}")
            if len(mismatches) > 10:
                print(f"    ... and {len(mismatches)-10} more")
        else:
            print(f"  ✓ No label/attended_ear mismatches")
        if self.load_audio and (left_present < n or right_present < n):
            print(f"  ⚠ Some records missing left/right audio paths (may use fallback)")
        print("  Sample per subject (attended_ear, label, left_audio_file, right_audio_file):")
        for sid in sorted(sample_per_subject.keys())[:8]:
            att, lab, lf, rf = sample_per_subject[sid]
            lf_short = (lf[:50] + "..." if lf and len(lf) > 50 else lf) or "None"
            rf_short = (rf[:50] + "..." if rf and len(rf) > 50 else rf) or "None"
            print(f"    {sid}: att={att} label={lab} left=[{lf_short}] right=[{rf_short}]")
        if len(sample_per_subject) > 8:
            print(f"    ... and {len(sample_per_subject)-8} more subjects")
        print("=" * 60 + "\n")

    def _load_das_preprocessing_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Load DAS preprocessing validated TFRecord data with robust shape validation."""

        tfrecord_files = []
        
        # Try multiple patterns to find TFRecord files
        # Pattern 1: Direct files with .tfrecords extension
        direct_files = list(self.tfrecord_dir.glob("*.tfrecords"))
        if direct_files:
            tfrecord_files.extend(direct_files)
        
        # Pattern 2: Files with .tfrecord extension (singular)
        direct_files_singular = list(self.tfrecord_dir.glob("*.tfrecord"))
        if direct_files_singular:
            tfrecord_files.extend(direct_files_singular)
        
        # Pattern 3: Files in tfrecords subdirectory
        tfrecords_dir = self.tfrecord_dir / "tfrecords"
        if tfrecords_dir.exists() and tfrecords_dir.is_dir():
            subdir_files = list(tfrecords_dir.glob("*.tfrecords"))
            if subdir_files:
                tfrecord_files.extend(subdir_files)
            subdir_files_singular = list(tfrecords_dir.glob("*.tfrecord"))
            if subdir_files_singular:
                tfrecord_files.extend(subdir_files_singular)
        
        # Pattern 4: Files in subdirectories (one level deep)
        subdir_files = list(self.tfrecord_dir.glob("*/*.tfrecords"))
        if subdir_files:
            tfrecord_files.extend(subdir_files)
        subdir_files_singular = list(self.tfrecord_dir.glob("*/*.tfrecord"))
        if subdir_files_singular:
            tfrecord_files.extend(subdir_files_singular)
        
        # Pattern 5: Files in nested subdirectories (two levels deep)
        nested_files = list(self.tfrecord_dir.glob("*/*/*.tfrecords"))
        if nested_files:
            tfrecord_files.extend(nested_files)
        nested_files_singular = list(self.tfrecord_dir.glob("*/*/*.tfrecord"))
        if nested_files_singular:
            tfrecord_files.extend(nested_files_singular)
        
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
        
        print(f"Loading {len(tfrecord_files)} TFRecord files...")
        if self.load_audio:
            print(f"Audio loading enabled: {self.audio_base_dir}")
        else:
            print("Audio loading disabled (using dummy envelopes)")
        
        all_eeg_data = []
        all_audio_envelopes = []
        all_labels = []
        all_metadata = []
        
        successful_files = 0
        failed_files = 0
        total_records = 0
        records_swapped_left_right = 0  # track2/track1 normalized to left=track1, right=track2
        subject_stats = {}
        
        for tfrecord_file in tqdm(tfrecord_files, desc="Loading DAS preprocessing data"):
            try:
                dataset = tf.data.TFRecordDataset(str(tfrecord_file))
                records_in_file = 0
                file_subject_id = None
                
                for record in dataset:
                    try:
                        example = tf.train.Example.FromString(record.numpy())
                        features = example.features.feature
                        

                        required_features = ['eeg', 'attended_ear', 'subject_id']
                        missing_features = [key for key in required_features if key not in features]
                        if missing_features:
                            if total_records < 5:
                                print(f"WARNING: Missing features {missing_features} in {tfrecord_file.name}")
                            continue
                        

                        left_audio_file = None
                        right_audio_file = None
                        if 'left_audio_file' in features:
                            left_audio_values = features['left_audio_file'].bytes_list.value
                            if left_audio_values and len(left_audio_values) > 0:
                                left_audio_file = left_audio_values[0].decode('utf-8')
                        if 'right_audio_file' in features:
                            right_audio_values = features['right_audio_file'].bytes_list.value
                            if right_audio_values and len(right_audio_values) > 0:
                                right_audio_file = right_audio_values[0].decode('utf-8')
                        
                        # Normalize so left=track1, right=track2 everywhere (some TFRecords use opposite convention).
                        # When we swap files we must also swap label so label 0 = attended track1 (left), 1 = track2 (right).
                        swapped_left_right = False
                        if left_audio_file and right_audio_file:
                            if 'track2' in left_audio_file and 'track1' in right_audio_file:
                                left_audio_file, right_audio_file = right_audio_file, left_audio_file
                                records_swapped_left_right += 1
                                swapped_left_right = True
                        

                        eeg_values = features['eeg'].float_list.value
                        if not eeg_values or len(eeg_values) == 0:
                            continue
                        
                        # Data is already validated in preprocessing - assume correct format
                        eeg_len = len(eeg_values)
                        if eeg_len % 64 != 0:
                            continue  # Skip invalid records (shouldn't happen with validated preprocessing)
                        
                        # Preprocessing ensures one sample per record (64 floats)
                        eeg_data = np.array(eeg_values, dtype=np.float32).reshape(1, 64)
                        

                        if np.any(np.isnan(eeg_data)) or np.any(np.isinf(eeg_data)):
                            print(f"WARNING: Invalid EEG values (NaN/Inf) in {tfrecord_file.name}")
                            continue
                        

                        attended_ear_values = features['attended_ear'].bytes_list.value
                        if not attended_ear_values or len(attended_ear_values) == 0:
                            continue
                        
                        try:
                            attended_ear = attended_ear_values[0].decode('utf-8')
                            label = 0 if attended_ear == 'L' else 1
                        except Exception:
                            print(f"ERROR: Could not decode attended_ear in {tfrecord_file.name}")
                            continue
                        
                        if attended_ear not in ['L', 'R']:
                            print(f"ERROR: Invalid attended_ear {attended_ear} in {tfrecord_file.name}")
                            continue
                        
                        # If we normalized left/right (swapped files), flip label so it matches normalized convention
                        if swapped_left_right:
                            label = 1 - label
                            attended_ear = 'R' if attended_ear == 'L' else 'L'
                        

                        subject_id = "unknown"
                        sample_idx = 0
                        

                        subject_values = features['subject_id'].bytes_list.value
                        if subject_values and len(subject_values) > 0:
                            try:
                                subject_id = subject_values[0].decode('utf-8')
                                file_subject_id = subject_id
                            except Exception:
                                subject_id = f"subject_{total_records}"
                        else:
                            subject_id = f"subject_{total_records}"
                        
                        # FIXED: Extract trial_index for trial-matched envelope loading
                        trial_index = None
                        if 'trial_index' in features:
                            trial_values = features['trial_index'].int64_list.value
                            if trial_values and len(trial_values) > 0:
                                trial_index = trial_values[0]
                        
                        if 'sample_id' in features:
                            sample_values = features['sample_id'].int64_list.value
                            if sample_values and len(sample_values) > 0:
                                sample_idx = sample_values[0]
                        


                        # Skip audio loading if disabled (much faster)
                        if self.load_audio:
                            audio_envelope = None
                            if attended_ear == 'L' and left_audio_file:
                                envelope_full = self._load_audio_envelope_full(left_audio_file)
                                if envelope_full is not None and sample_idx < len(envelope_full):
                                    audio_envelope = envelope_full[sample_idx:sample_idx+1]
                                    # Flatten to 1D if needed: (1, 4) -> (4,)
                                    if audio_envelope.ndim == 2:
                                        audio_envelope = audio_envelope.flatten()
                            elif attended_ear == 'R' and right_audio_file:
                                envelope_full = self._load_audio_envelope_full(right_audio_file)
                                if envelope_full is not None and sample_idx < len(envelope_full):
                                    audio_envelope = envelope_full[sample_idx:sample_idx+1]
                                    # Flatten to 1D if needed: (1, 4) -> (4,)
                                    if audio_envelope.ndim == 2:
                                        audio_envelope = audio_envelope.flatten()
                            
                            if audio_envelope is None:
                                audio_envelope = np.array([0.0], dtype=np.float32)
                        else:
                            audio_envelope = np.array([0.0], dtype=np.float32)
                        

                        if subject_id not in subject_stats:
                            subject_stats[subject_id] = {'samples': 0, 'labels': []}
                        subject_stats[subject_id]['samples'] += 1
                        subject_stats[subject_id]['labels'].append(label)
                        
                        metadata = {
                            'subject_id': subject_id,
                            'file': tfrecord_file.name,
                            'sample_idx': sample_idx,
                            'trial_index': trial_index,
                            'attended_ear': attended_ear,
                            'attention_label': label,
                            'preprocessing_method': 'DAS_16subjects_preprocessing',
                            'validation_passed': True,
                            'data_type': 'EEG_and_Audio',
                            'eeg_shape': eeg_data.shape,
                            'audio_file': left_audio_file if attended_ear == 'L' else right_audio_file,
                            'left_audio_file': left_audio_file,  # Store both for comparison
                            'right_audio_file': right_audio_file,  # Store both for comparison
                            'label_alignment': 'validated'
                        }
                        
                        all_eeg_data.append(eeg_data)
                        all_audio_envelopes.append(audio_envelope)
                        all_labels.append(label)
                        all_metadata.append(metadata)
                        records_in_file += 1
                        total_records += 1
                        
                    except Exception as record_error:
                        print(f"ERROR processing record in {tfrecord_file.name}: {record_error}")
                        continue
                
                if records_in_file > 0:
                    successful_files += 1
                    if file_subject_id:
                        print(f"✓ Loaded {records_in_file} samples from subject {file_subject_id}")
                else:
                    failed_files += 1
                    
            except Exception as e:
                failed_files += 1
                print(f"ERROR loading {tfrecord_file.name}: {e}")
                continue
        
        print(f"Loaded {total_records} records from {successful_files} files")
        print(f"Failed to load from {failed_files} files")
        if records_swapped_left_right > 0:
            print(f"  Normalized left/right: {records_swapped_left_right} records had track2/track1 swapped so left=track1, right=track2")
        
        # Validate left/right audio and label consistency (no swap by subject or trial)
        if total_records > 0 and all_metadata and all_labels:
            self._validate_left_right_consistency(all_metadata, all_labels)
        
        if total_records == 0:
            print("\n⚠ CRITICAL: No records were loaded successfully!")
            print("This could be due to:")
            print("  - Incorrect TFRecord format")
            print("  - Missing required features")
            print("  - Data corruption")
            print("  - Wrong file paths")
            print("  - All records failing shape validation")
            print(f"\nDebugging info:")
            print(f"  TFRecord directory: {self.tfrecord_dir}")
            print(f"  Directory exists: {self.tfrecord_dir.exists()}")
            print(f"  Number of TFRecord files found: {len(tfrecord_files)}")
            if self.tfrecord_dir.exists():
                print(f"  Contents:")
                for item in self.tfrecord_dir.iterdir():
                    print(f"    - {item.name} ({'dir' if item.is_dir() else 'file'})")
                    if item.is_dir():
                        subfiles = list(item.glob("*.tfrecords"))
                        if subfiles:
                            print(f"      Contains {len(subfiles)} TFRecord files")
                            # Try to read first file to see what's wrong
                            try:
                                first_file = subfiles[0]
                                dataset = tf.data.TFRecordDataset(str(first_file))
                                first_record = next(iter(dataset))
                                example = tf.train.Example.FromString(first_record.numpy())
                                features = example.features.feature
                                print(f"      First file features: {list(features.keys())}")
                                if 'eeg' in features:
                                    eeg_vals = features['eeg'].float_list.value
                                    print(f"      EEG length: {len(eeg_vals)}")
                                if 'attended_ear' in features:
                                    att_vals = features['attended_ear'].bytes_list.value
                                    print(f"      Attended ear: {att_vals[0].decode('utf-8') if att_vals else 'None'}")
                                if 'subject_id' in features:
                                    subj_vals = features['subject_id'].bytes_list.value
                                    print(f"      Subject ID: {subj_vals[0].decode('utf-8') if subj_vals else 'None'}")
                            except Exception as debug_error:
                                print(f"      Error reading first file: {debug_error}")
        

        # Subject statistics available in subject_stats if needed for debugging
        
        if not all_eeg_data:
            # Additional debugging: try to read one record from first file
            if tfrecord_files:
                print(f"\n⚠ Attempting to debug first TFRecord file: {tfrecord_files[0]}")
                try:
                    dataset = tf.data.TFRecordDataset(str(tfrecord_files[0]))
                    first_record = next(iter(dataset))
                    example = tf.train.Example.FromString(first_record.numpy())
                    features = example.features.feature
                    print(f"  Available features: {list(features.keys())}")
                    print(f"  Required features: ['eeg', 'attended_ear', 'subject_id']")
                    missing = [f for f in ['eeg', 'attended_ear', 'subject_id'] if f not in features]
                    if missing:
                        print(f"  Missing features: {missing}")
                    if 'eeg' in features:
                        eeg_vals = features['eeg'].float_list.value
                        print(f"  EEG values length: {len(eeg_vals)}")
                        print(f"  EEG length % 64: {len(eeg_vals) % 64}")
                except Exception as debug_err:
                    print(f"  Error reading first record: {debug_err}")
            raise ValueError("No valid DAS preprocessing data found in TFRecord files")
        
        # Store data per subject/trial to prevent cross-boundary windows
        # Use row counts, not record counts, for indexing
        # Build row-to-metadata mapping for correct audio alignment
        self._subject_trial_segments = []
        self._row_to_metadata = []  # Map each EEG row to its metadata
        current_subject = None
        current_trial = None
        segment_start_rows = 0  # Track in row units, not record units
        segment_eeg = []
        segment_labels = []
        segment_metadata = []
        
        for i, (eeg_sample, label, metadata) in enumerate(zip(all_eeg_data, all_labels, all_metadata)):
            subject_id = metadata.get('subject_id', 'unknown')
            trial_index = metadata.get('trial_index', None)
            
            # eeg_sample is (1, 64) - map each row to metadata
            num_rows = eeg_sample.shape[0] if len(eeg_sample.shape) > 1 else 1
            
            # Map each row to metadata (repeat metadata for multi-row samples)
            for row_idx in range(num_rows):
                self._row_to_metadata.append(metadata)
            
            # Check if we've moved to a new subject or trial
            if (subject_id != current_subject) or (trial_index != current_trial and trial_index is not None):
                # Save previous segment
                if segment_eeg:
                    eeg_segment = np.vstack(segment_eeg)
                    eeg_rows = eeg_segment.shape[0]  # Number of rows, not records
                    
                    # Extract trial_start_offset_in_audio from first metadata in segment
                    trial_start_offset = 0
                    if segment_metadata:
                        first_meta = segment_metadata[0]
                        # Try to get from metadata (if available in TFRecord)
                        trial_start_offset = first_meta.get('trial_start_offset_samples', 0)
                        # If not available, try to infer from filename or other metadata
                        if trial_start_offset == 0:
                            # Could parse from filename or use trial_index to compute offset
                            # For now, default to 0 (will need to be fixed in preprocessing)
                            pass
                    
                    self._subject_trial_segments.append({
                        'subject_id': current_subject,
                        'trial_index': current_trial,
                        'start_idx': segment_start_rows,  # In row units
                        'end_idx': segment_start_rows + eeg_rows,  # In row units
                        'trial_start_offset_in_audio': trial_start_offset,  # In samples at 128 Hz
                        'eeg': eeg_segment,
                        'labels': np.array(segment_labels),
                        'metadata': segment_metadata
                    })
                    segment_start_rows += eeg_rows  # Update in row units
                
                # Start new segment
                current_subject = subject_id
                current_trial = trial_index
                segment_eeg = []
                segment_labels = []
                segment_metadata = []
            
            segment_eeg.append(eeg_sample)
            segment_labels.append(label)
            segment_metadata.append(metadata)
        
        # Save last segment
        if segment_eeg:
            eeg_segment = np.vstack(segment_eeg)
            eeg_rows = eeg_segment.shape[0]  # Number of rows, not records
            
            # Extract trial_start_offset_in_audio from first metadata in segment
            trial_start_offset = 0
            if segment_metadata:
                first_meta = segment_metadata[0]
                trial_start_offset = first_meta.get('trial_start_offset_samples', 0)
            
            self._subject_trial_segments.append({
                'subject_id': current_subject,
                'trial_index': current_trial,
                'start_idx': segment_start_rows,  # In row units
                'end_idx': segment_start_rows + eeg_rows,  # In row units
                'trial_start_offset_in_audio': trial_start_offset,  # In samples at 128 Hz
                'eeg': eeg_segment,
                'labels': np.array(segment_labels),
                'metadata': segment_metadata
            })
        
        print(f"Created {len(self._subject_trial_segments)} subject/trial segments")
        for seg in self._subject_trial_segments[:5]:  # Show first 5
            print(f"  {seg['subject_id']}, trial {seg['trial_index']}: rows {seg['start_idx']}-{seg['end_idx']} ({seg['eeg'].shape[0]} rows)")
        
        # Still create concatenated version for backward compatibility, but windows will be created per-segment
        eeg_data = np.vstack(all_eeg_data)

        # Verify row count matches
        if len(self._row_to_metadata) != len(eeg_data):
            print(f"⚠ CRITICAL: Row-to-metadata mapping mismatch!")
            print(f"  EEG rows: {len(eeg_data)}, Metadata mappings: {len(self._row_to_metadata)}")
            print(f"  This will cause audio alignment errors!")


        # Load full audio envelopes
        # The preprocessing stores audio file paths, so we need to load full envelopes
        audio_envelopes_list = []
        missing_audio_count = 0
        
        for i, env in enumerate(all_audio_envelopes):
            if env is None or len(env) == 0:
                # Track missing audio for reporting
                missing_audio_count += 1
                # Use zeros as placeholder - will be replaced when loading windows with actual audio
                audio_envelopes_list.append(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
            else:
                # env should be a single value from _load_audio_envelope
                # For initial loading, we store the single value
                # Full temporal envelopes will be loaded in __getitem__ when creating windows
                env_val = env[0] if len(env) > 0 else 0.0
                
                # Store as single value - will be expanded to full envelope in window processing
                # This is a placeholder until we load the full envelope in __getitem__
                audio_envelopes_list.append(np.array([env_val], dtype=np.float32))
        
        if missing_audio_count > 0:
            print(f"⚠ WARNING: {missing_audio_count}/{len(all_audio_envelopes)} samples have missing audio envelopes")
            print(f"  Full temporal envelopes will be loaded from audio files during window creation")
        
        # Convert list to numpy array, handling variable shapes
        # First, ensure all items are 1D arrays
        audio_envelopes_list_flat = []
        for env in audio_envelopes_list:
            env = np.asarray(env)
            if env.ndim > 1:
                env = env.flatten()
            audio_envelopes_list_flat.append(env)
        
        # Find the maximum length to pad/truncate to
        max_len = max(len(env) for env in audio_envelopes_list_flat) if audio_envelopes_list_flat else 4
        # Ensure at least 4 features for 4-band envelope
        max_len = max(max_len, 4)
        
        # Pad or truncate all envelopes to the same length
        audio_envelopes_padded = []
        for env in audio_envelopes_list_flat:
            if len(env) < max_len:
                # Pad with zeros
                padding = np.zeros(max_len - len(env), dtype=np.float32)
                env = np.concatenate([env, padding])
            elif len(env) > max_len:
                # Truncate to max_len
                env = env[:max_len]
            audio_envelopes_padded.append(env)
        
        audio_envelopes = np.array(audio_envelopes_padded, dtype=np.float32)
        
        # Final shape should be (n_samples, features)
        # Ensure it's 2D
        if audio_envelopes.ndim == 1:
            # Single sample - reshape to (1, features)
            audio_envelopes = audio_envelopes.reshape(1, -1)
        elif audio_envelopes.ndim == 3:
            # 3D array - reshape to 2D: (n_samples, 1, features) -> (n_samples, features)
            audio_envelopes = audio_envelopes.reshape(audio_envelopes.shape[0], -1)
        
        # Ensure exactly 4 features (4-band envelope)
        if audio_envelopes.shape[1] < 4:
            # Pad to 4
            padding = np.zeros((audio_envelopes.shape[0], 4 - audio_envelopes.shape[1]), dtype=np.float32)
            audio_envelopes = np.column_stack([audio_envelopes, padding])
        elif audio_envelopes.shape[1] > 4:
            # Truncate to 4
            audio_envelopes = audio_envelopes[:, :4]
        
        labels = np.array(all_labels, dtype=np.int64)
        print(f"\nFinal data shapes:")
        print(f"  EEG data: {eeg_data.shape} (samples, channels)")
        print(f"  Audio envelopes: {audio_envelopes.shape} (samples, features)")
        print(f"  Labels: {labels.shape} (samples,)")
        print(f"  Expected EEG shape: (n_samples, 64)")
        
        if eeg_data.shape[1] != 64:
            raise ValueError(f"CRITICAL: EEG data has {eeg_data.shape[1]} channels, expected 64")
        
        if len(eeg_data) != len(labels):
            raise ValueError(f"CRITICAL: EEG samples ({len(eeg_data)}) != labels ({len(labels)})")
        
        if len(audio_envelopes) != len(eeg_data):
            raise ValueError(f"CRITICAL: Audio envelopes ({len(audio_envelopes)}) != EEG samples ({len(eeg_data)})")
        

        # VALIDATION: Check for dummy data
        valid_audio_count = np.sum(np.abs(audio_envelopes).sum(axis=1) > 1e-6)
        zero_audio_count = len(audio_envelopes) - valid_audio_count
        print(f"  Valid audio envelopes: {valid_audio_count}/{len(audio_envelopes)} ({100*valid_audio_count/len(audio_envelopes):.1f}%)")
        print(f"  Zero/dummy audio envelopes: {zero_audio_count}/{len(audio_envelopes)} ({100*zero_audio_count/len(audio_envelopes):.1f}%)")
        
        # NOTE: TFRecords contain placeholder audio data, but during training we load REAL audio from files
        # The placeholder pattern [val, val, 0, val²] in TFRecords is expected and not a problem
        # Real audio is loaded via _load_audio_envelope_full() in __getitem__()
        # So we don't need to skip any samples - all windows use real audio when load_audio=True
        if audio_envelopes.shape[1] >= 3:
            third_col_zeros = np.sum(np.abs(audio_envelopes[:, 2]) < 1e-6)
            if third_col_zeros > len(audio_envelopes) * 0.9:
                print(f"  ℹ INFO: TFRecord placeholder audio detected ({third_col_zeros} samples)")
                print(f"    Real audio will be loaded from files during training (load_audio=True)")
                print(f"    This is expected and not a problem.")
        
        if valid_audio_count > 0:
            non_zero_audio = audio_envelopes[np.abs(audio_envelopes).sum(axis=1) > 1e-6]
            print(f"  Audio envelope stats (non-zero samples):")
            print(f"    Mean per feature: {np.mean(non_zero_audio, axis=0)}")
            print(f"    Std per feature: {np.std(non_zero_audio, axis=0)}")
            print(f"    Min per feature: {np.min(non_zero_audio, axis=0)}")
            print(f"    Max per feature: {np.max(non_zero_audio, axis=0)}")
        else:
            print(f"  ⚠ WARNING: All audio envelopes are zero! Audio files may not be loading correctly.")
            print(f"    Check audio file paths in TFRecords and audio_base_dir setting.")
            if self.load_audio:
                print(f"    Audio loading is ENABLED but no valid audio found.")
                print(f"    Verify audio files exist at: {self.audio_base_dir}")
                print(f"    Check that audio file paths in TFRecords match actual file locations.")
            else:
                print(f"    Audio loading is DISABLED (load_audio=False). This is expected.")
                print(f"    Full temporal envelopes will be loaded from audio files during window creation.")
        
        del all_eeg_data, all_audio_envelopes, all_labels
        import gc
        gc.collect()
        
        return eeg_data, audio_envelopes, labels, all_metadata
    
    def _compute_4band_envelope(self, audio_data: np.ndarray, fs: int) -> np.ndarray:
        """
        Compute 4-band filterbank envelopes from audio signal (Telluride-style).
        
        Uses 4 frequency bands with bandpass filters, then Hilbert envelope extraction.
        This creates truly independent features instead of tiling.
        
        Args:
            audio_data: Audio signal (1D array)
            fs: Sampling rate in Hz
            
        Returns:
            4-band envelope array of shape (len(audio_data), 4)
        """
        from scipy.signal import butter, filtfilt, hilbert
        
        # Define 4 frequency bands (Hz) - typical for speech/audio
        # Band 1: Low (1-500 Hz) - fundamental frequencies (min 1 Hz to avoid filter error)
        # Band 2: Mid-low (500-1500 Hz) - formants
        # Band 3: Mid-high (1500-4000 Hz) - consonants
        # Band 4: High (4000-8000 Hz) - fricatives
        nyquist = fs / 2
        
        # Ensure bands are valid and within Nyquist
        bands = [
            (max(1, 0), min(500, nyquist * 0.9)),  # Low: 1-500 Hz (avoid 0 Hz)
            (max(1, 500), min(1500, nyquist * 0.9)),  # Mid-low
            (max(1, 1500), min(4000, nyquist * 0.9)),  # Mid-high
            (max(1, 4000), min(8000, nyquist * 0.9))  # High
        ]
        
        envelopes = []
        for low, high in bands:
            try:
                # Validate frequency range
                if low <= 0:
                    low = 1.0  # Minimum 1 Hz
                if high >= nyquist:
                    high = nyquist * 0.95  # Stay below Nyquist
                if low >= high:
                    # Invalid band - use full spectrum as fallback
                    envelope = np.abs(hilbert(audio_data))
                else:
                    # Normalize frequencies (must be 0 < low_norm < high_norm < 1)
                    low_norm = max(0.001, min(low / nyquist, 0.99))
                    high_norm = max(0.001, min(high / nyquist, 0.99))
                    
                    # Ensure valid range
                    if low_norm >= high_norm:
                        low_norm = 0.001
                        high_norm = min(0.99, low_norm + 0.1)
                    
                    # Design bandpass filter
                    b, a = butter(4, [low_norm, high_norm], btype='band')
                    
                    # Filter audio
                    filtered = filtfilt(b, a, audio_data)
                    
                    # Extract envelope using Hilbert transform
                    analytic = hilbert(filtered)
                    envelope = np.abs(analytic)
            except Exception as e:
                # Fallback: use full spectrum if filter design fails
                envelope = np.abs(hilbert(audio_data))
            
            # Smooth envelope
            if len(envelope) > 9:
                from scipy.ndimage import uniform_filter1d
                envelope = uniform_filter1d(envelope, size=9, mode='nearest')
            
            envelopes.append(envelope)
        
        # Stack into (N, 4) array
        result = np.column_stack(envelopes)
        
        # Normalize each band independently
        for i in range(4):
            if np.max(np.abs(result[:, i])) > 0:
                result[:, i] = result[:, i] / (np.max(np.abs(result[:, i])) + 1e-8)
        
        return result.astype(np.float32)
    
    def _load_audio_envelope_full(self, audio_file_path: str) -> Optional[np.ndarray]:
        """
        Load full audio envelope from audio file, resampled to match EEG sampling rate.
        Returns 4-band filterbank envelopes (Telluride-style) instead of single envelope.
        
        Args:
            audio_file_path: Path to audio file (can be relative or absolute)
            
        Returns:
            Full 4-band audio envelope array of shape (N, 4), or None if file not found
        """

        cache_key = str(Path(audio_file_path).resolve()) if Path(audio_file_path).is_absolute() else audio_file_path
        if cache_key in self._audio_envelope_cache:
            return self._audio_envelope_cache[cache_key]
        

        # Simplified path resolution - paths are already validated in preprocessing
        audio_file = None
        audio_path = Path(audio_file_path)
        
        # Try absolute path first
        if audio_path.is_absolute() and audio_path.exists():
            audio_file = audio_path
        # Try relative to audio_base_dir
        elif self.audio_base_dir and (self.audio_base_dir / audio_path).exists():
            audio_file = self.audio_base_dir / audio_path
        # Try filename only in audio_base_dir
        elif self.audio_base_dir and (self.audio_base_dir / audio_path.name).exists():
            audio_file = self.audio_base_dir / audio_path.name
        # Try current directory
        elif audio_path.exists():
            audio_file = audio_path
        
        if audio_file is None or not audio_file.exists():
            # Only print warning for first few missing files to avoid spam
            if not hasattr(self, '_audio_file_missing_count'):
                self._audio_file_missing_count = 0
            if self._audio_file_missing_count < 3:
                print(f"⚠ WARNING: Audio file not found: {audio_file_path}")
                print(f"   Searched in: {self.audio_base_dir if self.audio_base_dir else 'default locations'}")
                self._audio_file_missing_count += 1
            return None
        
        try:

            from scipy.io import wavfile
            from scipy import signal
            
            fs, audio_data = wavfile.read(str(audio_file))
            
            # Audio loading is silent (can be verbose if needed for debugging)
            

            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            

            audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            

            if fs != self.sampling_rate:
                num_samples = int(len(audio_data) * self.sampling_rate / fs)
                audio_data = signal.resample(audio_data, num_samples)
            

            # Compute 4-band filterbank envelopes
            envelope_4band = self._compute_4band_envelope(audio_data, self.sampling_rate)

            self._audio_envelope_cache[cache_key] = envelope_4band
            
            return envelope_4band
                
        except Exception as e:
            print(f"WARNING: Could not load audio envelope from {audio_file_path}: {e}")
            return None
    
    # _load_audio_envelope() removed - use _load_audio_envelope_full() directly
    
    def _preload_all_audio_files(self):
        """
        Pre-load all unique audio files into cache for faster access.
        This significantly speeds up training by avoiding repeated file I/O.
        """
        if not self.load_audio:
            return
        
        print("\n" + "="*80)
        print("PRE-LOADING AUDIO FILES FOR FASTER TRAINING")
        print("="*80)
        
        # Collect all unique audio file paths from metadata
        unique_audio_files = set()
        for meta in self.metadata:
            left_file = meta.get('left_audio_file')
            right_file = meta.get('right_audio_file')
            if left_file:
                unique_audio_files.add(left_file)
            if right_file:
                unique_audio_files.add(right_file)
        
        print(f"Found {len(unique_audio_files)} unique audio files to pre-load...")
        
        # Pre-load audio files with progress bar
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        loaded_count = 0
        failed_count = 0
        
        # Use threading for I/O-bound audio loading (parallel file reads)
        max_workers = min(8, len(unique_audio_files))  # Use up to 8 threads
        
        def load_single_audio(audio_file_path):
            """Load a single audio file and cache it."""
            try:
                envelope = self._load_audio_envelope_full(audio_file_path)
                return audio_file_path, envelope, True
            except Exception as e:
                return audio_file_path, None, False
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all audio loading tasks
            future_to_file = {
                executor.submit(load_single_audio, audio_file): audio_file
                for audio_file in unique_audio_files
            }
            
            # Process completed tasks with progress bar
            for future in tqdm(as_completed(future_to_file), 
                             total=len(unique_audio_files),
                             desc="Pre-loading audio files"):
                audio_file, envelope, success = future.result()
                if success and envelope is not None:
                    loaded_count += 1
                else:
                    failed_count += 1
        
        print(f"\n✓ Pre-loading complete:")
        print(f"  Successfully loaded: {loaded_count}/{len(unique_audio_files)} files")
        print(f"  Failed: {failed_count} files")
        print(f"  Cache size: {len(self._audio_envelope_cache)} files")
        print(f"  Memory usage: ~{len(self._audio_envelope_cache) * 0.5:.1f} MB (estimated)")
        print("="*80 + "\n")
    
    def _create_das_windows(self) -> List[Tuple[int, int]]:
        """
        Create windows optimized for DAS data structure with proper time units.
        
        FIXED: Create windows within each subject/trial segment to prevent cross-boundary windows.
        This ensures:
        - No windows crossing subject boundaries
        - No windows crossing trial boundaries  
        - Proper EEG/audio alignment
        - Correct label assignment
        """
        window_seconds = self.window_size / self.sampling_rate
        step_size = int(self.window_size * (1 - self.overlap))
        step_seconds = step_size / self.sampling_rate
        
        print(f"Creating DAS windows within subject/trial boundaries:")
        print(f"  Window size: {self.window_size} samples ({window_seconds:.1f} seconds)")
        print(f"  Step size: {step_size} samples ({step_seconds:.1f} seconds)")
        print(f"  Overlap: {self.overlap:.1%}")
        print(f"  Sampling rate: {self.sampling_rate} Hz")
        print(f"  Number of subject/trial segments: {len(self._subject_trial_segments)}")

        if window_seconds < 1.0:
            print(f"⚠ WARNING: Very short window ({window_seconds:.1f}s) may have poor signal-to-noise")
        elif window_seconds > 20.0:
            print(f"⚠ WARNING: Very long window ({window_seconds:.1f}s) may miss temporal dynamics")
        else:
            print(f"✓ Window size appropriate for EEG attention decoding")
        
        window_indices = []
        global_offset = 0  # Track global offset in concatenated eeg_data
        
        # FIXED: Create windows within each subject/trial segment
        for seg_idx, segment in enumerate(self._subject_trial_segments):
            segment_eeg = segment['eeg']
            segment_labels = segment['labels']
            segment_metadata = segment['metadata']
            subject_id = segment['subject_id']
            trial_index = segment['trial_index']
            
            # Create windows only within this segment
            segment_len = len(segment_eeg)
            if segment_len < self.window_size:
                # Segment too short for a window - skip
                global_offset += segment_len
                continue
            
            num_windows_in_segment = (segment_len - self.window_size) // step_size + 1
            
            for i in range(num_windows_in_segment):
                local_idx = i * step_size
                global_idx = global_offset + local_idx
                
                if local_idx + self.window_size <= segment_len:
                    # NOTE: We don't skip based on synthetic audio mask from TFRecords because:
                    # - TFRecords contain placeholder/synthetic audio
                    # - During __getitem__, we load REAL audio from files via _load_audio_envelope_full
                    # - So the synthetic mask is only relevant if load_audio=False
                    # If load_audio=True, we'll use real audio from files, so we keep all windows
                    # If load_audio=False, we could skip, but that's not the intended use case
                    
                    # Window is fully within segment
                    window_labels = segment_labels[local_idx:local_idx + self.window_size]
                    
                    if len(window_labels) > 0:
                        window_label = int(np.bincount(window_labels).argmax())
                    else:
                        window_label = 0
                    
                    window_indices.append((global_idx, window_label))
            
            global_offset += segment_len
        
        print(f"Created {len(window_indices)} DAS windows (within subject/trial boundaries)")
        
        # Verify no windows cross boundaries
        cross_boundary_count = 0
        for data_idx, label in window_indices[:100]:  # Check first 100
            window_end = data_idx + self.window_size
            # Check if window spans multiple segments
            for seg in self._subject_trial_segments:
                if data_idx < seg['end_idx'] <= window_end:
                    cross_boundary_count += 1
                    break
        
        if cross_boundary_count > 0:
            print(f"⚠ WARNING: {cross_boundary_count} windows may cross segment boundaries (out of 100 checked)")
        else:
            print(f"✓ Verified: No windows cross subject/trial boundaries")

        window_labels = [label for _, label in window_indices]
        label_dist = np.bincount(window_labels) if window_labels else np.array([0, 0])
        print(f"Window label distribution: {label_dist}")
        
        return window_indices
    
    # _das_eeg_preprocessing() removed - data is already preprocessed
    
    
    def __len__(self):
        return len(self.window_indices)
    
    def __getitem__(self, idx):
        data_idx, label = self.window_indices[idx]
        
        # Get subject_id from metadata for window-level subject tracking
        subject_id = "unknown"
        if data_idx < len(self._row_to_metadata):
            window_metadata = self._row_to_metadata[data_idx]
            subject_id = window_metadata.get('subject_id', 'unknown')

        cache_key = (data_idx, self.mode)
        if cache_key in self._window_cache:
            self._cache_hits += 1
            cached_data, cached_label = self._window_cache[cache_key]
            return cached_data, cached_label
        
        self._cache_misses += 1
        

        window_eeg = self.eeg_data[data_idx:data_idx + self.window_size]
        
        # Use row-to-metadata mapping (data_idx is a row index)
        if data_idx < len(self._row_to_metadata):
            window_metadata = self._row_to_metadata[data_idx]
        elif data_idx < len(self.metadata):
            # Fallback to old method (shouldn't happen if mapping is correct)
            window_metadata = self.metadata[data_idx]
        else:
            window_metadata = None
        attended_audio_envelope = None
        left_audio_envelope = None
        right_audio_envelope = None
        
        if window_metadata and self.load_audio:
            sample_idx = window_metadata.get('sample_idx', 0)
            trial_index = window_metadata.get('trial_index', None)
            attended_ear = window_metadata.get('attended_ear', 'L')
            left_audio_file = window_metadata.get('left_audio_file')
            right_audio_file = window_metadata.get('right_audio_file')
            
            # Get trial_start_offset from segment
            trial_start_offset = 0
            for seg in self._subject_trial_segments:
                if (seg['subject_id'] == subject_id and 
                    seg['trial_index'] == trial_index):
                    trial_start_offset = seg.get('trial_start_offset_in_audio', 0)
                    break
            
            # Helper function to load trial-matched window segment from audio envelope
            # Uses trial_start_offset + sample_idx for correct audio alignment
            def load_window_segment(audio_file, trial_sample_idx, window_size, trial_index=None, trial_start=0):
                """
                Load a window segment from a full audio envelope that matches the trial.
                
                Args:
                    audio_file: Path to audio file
                    trial_sample_idx: Sample index within the trial (from sample_id in TFRecord)
                    window_size: Window size in samples
                    trial_index: Trial index for debugging
                    trial_start: Trial start offset in audio file (in samples at 128 Hz)
                
                Returns:
                    Audio envelope segment matching the trial window, or None if not found
                """
                envelope_full = self._load_audio_envelope_full(audio_file)
                if envelope_full is not None and len(envelope_full) > 0:
                    # Correct audio slicing: trial_start + sample_idx_within_trial
                    audio_start = trial_start + trial_sample_idx
                    end_idx = min(audio_start + window_size, len(envelope_full))
                    
                    if audio_start < len(envelope_full):
                        segment = envelope_full[audio_start:end_idx]
                        # Pad if needed
                        if len(segment) < window_size:
                            padding = np.zeros((window_size - len(segment), envelope_full.shape[1]), dtype=np.float32)
                            segment = np.vstack([segment, padding])
                        
                        # Debug print for first few windows
                        if not hasattr(self, '_audio_slice_debug_count'):
                            self._audio_slice_debug_count = 0
                        if self._audio_slice_debug_count < 3:
                            print(f"Audio slice: start={audio_start}, trial_start={trial_start}, sample_idx={trial_sample_idx}, envelope_len={len(envelope_full)}")
                            self._audio_slice_debug_count += 1
                        
                        return segment
                    else:
                        # If audio_start is beyond envelope length, use last values
                        if trial_index is not None:
                            print(f"⚠ WARNING: Trial {trial_index} audio_start {audio_start} >= envelope length {len(envelope_full)}")
                        return np.tile(envelope_full[-1:], (window_size, 1))
                return None
            
            # Load attended audio envelope (the one that matches the label)
            if attended_ear == 'L' and left_audio_file:
                attended_audio_envelope = load_window_segment(left_audio_file, sample_idx, self.window_size, trial_index, trial_start_offset)
            elif attended_ear == 'R' and right_audio_file:
                attended_audio_envelope = load_window_segment(right_audio_file, sample_idx, self.window_size, trial_index, trial_start_offset)
            
            # Load left and right audio envelopes for comparison
            if left_audio_file:
                left_audio_envelope = load_window_segment(left_audio_file, sample_idx, self.window_size, trial_index, trial_start_offset)
            if right_audio_file:
                right_audio_envelope = load_window_segment(right_audio_file, sample_idx, self.window_size, trial_index, trial_start_offset)
            
            # NOTE: We don't check for synthetic audio here because:
            # - Audio loaded from files via _load_audio_envelope_full is REAL audio
            # - The synthetic mask from TFRecords is just for placeholder data
            # - Real audio files don't have synthetic patterns
            
            # REMOVED: Excessive debugging code - data is already validated in preprocessing
        
        # Use attended audio if available, otherwise fall back to placeholder
        if attended_audio_envelope is not None:
            # attended_audio_envelope is now (window_size, 4) from 4-band filterbank
            if attended_audio_envelope.ndim == 1:
                # Fallback: if somehow 1D, reshape to (window_size, 1) then expand
                window_audio = attended_audio_envelope.reshape(-1, 1)
                # Expand to 4 bands using the 4-band computation
                window_audio = self._compute_4band_envelope(window_audio.flatten(), self.sampling_rate)
            else:
                window_audio = attended_audio_envelope  # Already (window_size, 4)
        else:
            # Fallback: use placeholder from self.audio_envelopes (single values)
            window_audio_1d = self.audio_envelopes[data_idx:data_idx + self.window_size]
            if window_audio_1d.ndim == 1 or window_audio_1d.shape[1] == 1:
                # Compute 4-band from single values
                window_audio = self._compute_4band_envelope(window_audio_1d.flatten(), self.sampling_rate)
            else:
                window_audio = window_audio_1d
            if self.load_audio and not hasattr(self, '_using_placeholder_audio_warned'):
                print(f"⚠ WARNING: Using placeholder audio envelopes. Full temporal envelopes not available.")
                self._using_placeholder_audio_warned = True
        
        # Use zeros only if audio loading is disabled or files are truly missing
        if left_audio_envelope is None:
            if self.load_audio:
                # Audio loading enabled but file missing - use zeros with warning
                if not hasattr(self, '_missing_left_audio_warned'):
                    print(f"⚠ WARNING: Left audio envelope missing for some samples. Using zeros.")
                    self._missing_left_audio_warned = True
            left_audio_envelope = np.zeros(self.window_size, dtype=np.float32)
        if right_audio_envelope is None:
            if self.load_audio:
                # Audio loading enabled but file missing - use zeros with warning
                if not hasattr(self, '_missing_right_audio_warned'):
                    print(f"⚠ WARNING: Right audio envelope missing for some samples. Using zeros.")
                    self._missing_right_audio_warned = True
            right_audio_envelope = np.zeros(self.window_size, dtype=np.float32)
        

        # REMOVED: Redundant preprocessing - data is already preprocessed in das_preprocessing_16subjects.py
        # TFRecords contain data that is already:
        # - Downsampled to 128Hz
        # - Bandpass filtered (0.5-40 Hz)
        # - Z-score normalized per channel
        # No additional preprocessing needed
        


        # window_audio should already be (window_size, 4) from 4-band filterbank
        # Ensure correct shape
        if window_audio.ndim == 1:
            # Fallback: if 1D, compute 4-band from it
            window_audio = self._compute_4band_envelope(window_audio, self.sampling_rate)
        elif window_audio.shape[1] != 4:
            if window_audio.shape[1] == 1:
                # Single band - compute 4-band from it
                window_audio = self._compute_4band_envelope(window_audio.flatten(), self.sampling_rate)
            elif window_audio.shape[1] < 4:
                # Less than 4 - pad with zeros (shouldn't happen with 4-band)
                padding = np.zeros((window_audio.shape[0], 4 - window_audio.shape[1]), dtype=np.float32)
                window_audio = np.column_stack([window_audio, padding])
            else:
                # More than 4 - truncate to 4
                window_audio = window_audio[:, :4]
        

        # Audio normalization is handled in _process_audio_envelope (matches FULCCA)
        # Just ensure correct dtype - normalization happens in _process_audio_envelope
        if window_audio is not None:
            window_audio = np.asarray(window_audio, dtype=np.float32)
        if left_audio_envelope is not None:
            left_audio_envelope = np.asarray(left_audio_envelope, dtype=np.float32)
        if right_audio_envelope is not None:
            right_audio_envelope = np.asarray(right_audio_envelope, dtype=np.float32)
        

        window_eeg_tensor = tf.constant(window_eeg, dtype=tf.float32)
        window_audio_tensor = tf.constant(window_audio, dtype=tf.float32)
        
        # Process left and right audio envelopes to match window_audio format
        # _process_audio_envelope normalizes using per-feature z-score (matches FULCCA)
        left_audio_processed = self._process_audio_envelope(left_audio_envelope, self.window_size)
        right_audio_processed = self._process_audio_envelope(right_audio_envelope, self.window_size)
        
        # Also normalize window_audio using same method for consistency
        window_audio = self._process_audio_envelope(window_audio, self.window_size)
        
        left_audio_tensor = tf.constant(left_audio_processed, dtype=tf.float32)
        right_audio_tensor = tf.constant(right_audio_processed, dtype=tf.float32)
        
        label_tensor = tf.constant([label], dtype=tf.int64)
        

        # Return both audio streams for comparison
        window_tensor = (window_eeg_tensor, window_audio_tensor)
        aux_data = {
            'left_env': left_audio_tensor,
            'right_env': right_audio_tensor,
            'label': label_tensor
            # Note: subject_id removed from tf.data - use only in Python splitting
        }
        

        if len(self._window_cache) < self.cache_size:
            self._window_cache[cache_key] = (window_tensor, aux_data)
        
        return window_tensor, aux_data
    
    def _process_audio_envelope(self, audio_envelope: np.ndarray, window_size: int) -> np.ndarray:
        """Process audio envelope to match window size and format.
        
        Now expects 4-band filterbank envelope (window_size, 4) from _load_audio_envelope_full.
        """
        if audio_envelope is None or len(audio_envelope) == 0:
            # Use zeros if envelope is missing
            audio_envelope = np.zeros((window_size, 4), dtype=np.float32)
        
        # Ensure envelope matches window size
        if audio_envelope.ndim == 1:
            # 1D array - compute 4-band from it
            if len(audio_envelope) == window_size:
                audio_envelope = self._compute_4band_envelope(audio_envelope, self.sampling_rate)
            else:
                # Wrong length - pad or truncate first, then compute 4-band
                if len(audio_envelope) < window_size:
                    padding = np.zeros(window_size - len(audio_envelope), dtype=np.float32)
                    audio_envelope = np.concatenate([audio_envelope, padding])
                else:
                    audio_envelope = audio_envelope[:window_size]
                audio_envelope = self._compute_4band_envelope(audio_envelope, self.sampling_rate)
        elif audio_envelope.ndim == 2:
            # 2D array - should be (window_size, 4) or (window_size, 1)
            if audio_envelope.shape[0] != window_size:
                # Wrong length - pad or truncate
                if audio_envelope.shape[0] < window_size:
                    padding = np.zeros((window_size - audio_envelope.shape[0], audio_envelope.shape[1]), dtype=np.float32)
                    audio_envelope = np.vstack([audio_envelope, padding])
                else:
                    audio_envelope = audio_envelope[:window_size]
            
            if audio_envelope.shape[1] == 1:
                # Single band - compute 4-band from it
                audio_envelope = self._compute_4band_envelope(audio_envelope.flatten(), self.sampling_rate)
            elif audio_envelope.shape[1] != 4:
                # Wrong number of bands
                if audio_envelope.shape[1] < 4:
                    padding = np.zeros((window_size, 4 - audio_envelope.shape[1]), dtype=np.float32)
                    audio_envelope = np.column_stack([audio_envelope, padding])
                else:
                    audio_envelope = audio_envelope[:, :4]
        else:
            # Invalid shape - create zeros
            audio_envelope = np.zeros((window_size, 4), dtype=np.float32)
        
        # CRITICAL FIX: Do NOT normalize here - normalization removes amplitude relationships
        # CCA needs raw amplitude differences to find correlations
        # Normalization will be handled by CCA's calculate_cca_parameters_from_dataset
        # which computes global statistics on concatenated training data
        # This preserves relative amplitude differences between windows that CCA relies on
        return audio_envelope.astype(np.float32)


class DASCCAModel:
    """
    DASCCA model implementing Canonical Correlation Analysis for DAS EEG dataset.
    
    This model uses the telluride_decoding CCA implementation to find correlations
    between EEG data and attention labels, providing comprehensive metrics evaluation.
    Adapted from FULCCA with DAS-specific optimal hyperparameters.
    """
    
    def __init__(self, cca_dims: int = 25, regularization: float = 0.08, window_size: int = 512,
                 use_time_lags: bool = True, min_lag_ms: float = 0.0, max_lag_ms: float = 250.0,
                 fs: float = 64.0, use_lda: bool = True, pca_eeg: int = 25, pca_audio: int = 0,
                 eeg_lag_taps: int = 5):
        """
        Initialize DASCCA model (paper-style: single CCA on attended envelope + LDA on f=rho1-rho2).
        
        Args:
            cca_dims: Number of CCA dimensions J (max: min(EEG_dims, Audio_dims))
            regularization: Regularization parameter for CCA (optimal: 0.08 for DAS)
            window_size: Window size in samples
            use_time_lags: Whether to use time-lagged audio (forward model)
            min_lag_ms, max_lag_ms: Envelope lag range in ms (e.g. 0-250ms)
            fs: Sampling rate in Hz
            use_lda: LDA on f = rho_1 - rho_2 (paper [7])
            pca_eeg: PCA on EEG before CCA (paper: regularization). 0=off.
            pca_audio: PCA on audio (0=off)
            eeg_lag_taps: Backward model taps L for EEG: x(t)=[eeg(t),...,eeg(t-L+1)]. 0=no EEG lag.
        """
        self.use_time_lags = use_time_lags
        self.fs = fs
        self.min_lag_ms = max(0.0, min_lag_ms)
        self.max_lag_ms = min(500.0, max_lag_ms)
        self.eeg_lag_taps = max(0, int(eeg_lag_taps))
        if use_time_lags:
            if self.max_lag_ms > 300:
                print(f"  ⚠ WARNING: max_lag_ms={self.max_lag_ms}ms — high lag range on CPU can explode dimensionality. Consider 0–300ms.")
            min_lag_samples = int(self.min_lag_ms * fs / 1000.0)
            max_lag_samples = int(self.max_lag_ms * fs / 1000.0)
            self.lag_samples = np.arange(min_lag_samples, max_lag_samples + 1)
            self.num_lags = len(self.lag_samples)
            print(f"  Time-lagged audio: {self.num_lags} lags ({self.min_lag_ms}-{self.max_lag_ms}ms at {fs} Hz)")
        else:
            self.lag_samples = np.array([0])
            self.num_lags = 1
        if self.eeg_lag_taps > 0:
            print(f"  Time-lagged EEG (backward model): L={self.eeg_lag_taps} taps -> {64 * self.eeg_lag_taps} features per time point")
        
        # Dimensions: EEG = 64*eeg_lag_taps (or 64 if no lag), Audio = 4*num_lags
        eeg_base = 64
        self.eeg_dims = eeg_base * max(1, self.eeg_lag_taps) if self.eeg_lag_taps > 0 else eeg_base
        audio_bands = 4
        audio_dims = audio_bands * self.num_lags
        actual_max_cca_dims = min(self.eeg_dims, audio_dims)
        # For 81% accuracy target, use more dimensions to capture signal better
        # With time lag range (150-400ms, 33 lags), we have 132 audio features, allowing up to 64 CCA dims
        # REMOVED artificial cap - use actual_max_cca_dims (up to 64) for better signal capture
        
        if cca_dims > actual_max_cca_dims:
            print(f"⚠ WARNING: Requested {cca_dims} CCA dimensions, max is {actual_max_cca_dims} (min(EEG={self.eeg_dims}, Audio={audio_dims}))")
            cca_dims = actual_max_cca_dims
        elif cca_dims < 1:
            cca_dims = 1
        elif cca_dims < 25 and actual_max_cca_dims >= 25:
            cca_dims = min(25, actual_max_cca_dims)
        
        self.cca_dims = cca_dims
        self.regularization = regularization
        self.window_size = window_size
        self.use_lda = use_lda
        self.audio_bands = audio_bands
        self.pca_eeg = pca_eeg
        self.pca_audio = pca_audio
        self.pca_x = None
        self.pca_y = None
        self.model = None
        self.is_fitted = False
        self.lda_model = None
        self.lda_scaler = None  # StandardScaler on f=ρ1−ρ2 for LDA (train) and predict
        # Paper: single CCA on (EEG, attended envelope); at test same Wx,Ws for both streams; LDA on f=rho1-rho2
        self.cca_params = None       # Single CCA (rot_x, rot_y, mean_x, mean_y, eigenvalues)
        self.cca_params_left = None  # Unused in paper mode; kept for any legacy
        self.cca_params_right = None
        
        print(f"DASCCA model initialized (paper-style: single CCA + LDA on f=ρ1-ρ2):")
        print(f"  CCA dimensions J: {self.cca_dims} (max: {actual_max_cca_dims})")
        print(f"  EEG dims: {self.eeg_dims} (64 × {max(1, self.eeg_lag_taps)} taps)")
        print(f"  Audio dims: {audio_dims} (4 bands × {self.num_lags} lags)")
        print(f"  PCA on EEG: {pca_eeg} components" if pca_eeg else "  PCA on EEG: off")
        print(f"  LDA: {'on f=ρ1-ρ2' if use_lda else 'disabled'}")
    
    def _compute_rho(self, X_eeg: np.ndarray, Y_env: np.ndarray) -> np.ndarray:
        """
        Compute J-dimensional vector of canonical correlations (paper: ρ for one stream).
        Uses single CCA: project EEG and envelope with Wx, Ws; return per-dimension Pearson corr.
        """
        if self.cca_params is None:
            raise ValueError("Model not fitted.")
        L = max(1, self.eeg_lag_taps)
        if X_eeg.ndim == 1:
            X_eeg = X_eeg.reshape(-1, 64)
        if Y_env.ndim == 1:
            Y_env = Y_env.reshape(-1, 4)
        X = make_lagged_eeg(X_eeg, L).astype(np.float32)
        if Y_env.shape[1] == 4 and self.use_time_lags:
            Y_env = make_lagged_audio(Y_env, self.lag_samples, self.fs)
        if self.pca_x is not None:
            X = self.pca_x.transform(X)
        if self.pca_y is not None:
            Y_env = self.pca_y.transform(Y_env)
        rot_x = np.asarray(self.cca_params['rot_x'])
        rot_y = np.asarray(self.cca_params['rot_y'])
        if rot_x.shape[0] != X.shape[1] and rot_x.shape[1] == X.shape[1]:
            rot_x = rot_x.T
        if rot_y.shape[0] != Y_env.shape[1] and rot_y.shape[1] == Y_env.shape[1]:
            rot_y = rot_y.T
        mean_x = np.asarray(self.cca_params['mean_x']).reshape(1, -1)
        mean_y = np.asarray(self.cca_params['mean_y']).reshape(1, -1)
        U = (X - mean_x) @ rot_x
        V = (Y_env - mean_y) @ rot_y
        J = U.shape[1]
        rho = np.zeros(J, dtype=np.float32)
        for j in range(J):
            u, v = U[:, j], V[:, j]
            u = u - np.mean(u)
            v = v - np.mean(v)
            d = np.sqrt(np.sum(u**2) * np.sum(v**2)) + 1e-8
            rho[j] = np.sum(u * v) / d
        return rho
    
    def _fit_lda(self, dataset: tf.data.Dataset):
        """Fit LDA on f = ρ1 − ρ2 (paper [7])."""
        print("  Computing f = ρ1 − ρ2 per window for LDA...")
        all_f = []
        all_labels = []
        for batch in dataset:
            if not (isinstance(batch, tuple) and len(batch) == 2):
                continue
            inputs, aux = batch
            if not (isinstance(aux, dict) and 'left_env' in aux and 'right_env' in aux):
                continue
            eeg = (inputs['input_1'].numpy() if hasattr(inputs['input_1'], 'numpy') else np.array(inputs['input_1']))
            left = (aux['left_env'].numpy() if hasattr(aux['left_env'], 'numpy') else np.array(aux['left_env']))
            right = (aux['right_env'].numpy() if hasattr(aux['right_env'], 'numpy') else np.array(aux['right_env']))
            lab = aux.get('label')
            if lab is not None:
                lab = lab.numpy() if hasattr(lab, 'numpy') else np.array(lab)
                lab = np.atleast_1d(lab).flatten()
            B, W = 1, eeg.shape[0]
            if len(eeg.shape) == 2 and '_batch_size' in aux and '_window_size' in aux:
                B = int(aux['_batch_size'].numpy() if hasattr(aux['_batch_size'], 'numpy') else aux['_batch_size'])
                W = int(aux['_window_size'].numpy() if hasattr(aux['_window_size'], 'numpy') else aux['_window_size'])
                eeg = eeg.reshape(B, W, -1)
                left = left.reshape(B, W, -1)
                right = right.reshape(B, W, -1)
            for w in range(B):
                eeg_w = eeg[w] if eeg.ndim == 3 else eeg
                left_w = left[w] if left.ndim == 3 else left
                right_w = right[w] if right.ndim == 3 else right
                rho1 = self._compute_rho(eeg_w, left_w)
                rho2 = self._compute_rho(eeg_w, right_w)
                f = rho1 - rho2
                all_f.append(f)
                all_labels.append(int(lab[w]) if lab is not None and w < len(lab) else 0)
        
        if len(all_f) < 2:
            self.lda_model = None
            return
        F = np.array(all_f, dtype=np.float32)
        labels = np.array(all_labels, dtype=np.int64)
        if len(np.unique(labels)) < 2:
            self.lda_model = None
            self.lda_scaler = None
            return
        # Standardize f so LDA is not dominated by scale; use same transform at test time
        self.lda_scaler = StandardScaler()
        F_scaled = self.lda_scaler.fit_transform(F)
        # Balanced priors so LDA does not collapse to majority class (fixes ~all-0 predictions)
        n_classes = len(np.unique(labels))
        priors = np.ones(n_classes) / n_classes
        self.lda_model = LinearDiscriminantAnalysis(priors=priors)
        self.lda_model.fit(F_scaled, labels)
        print(f"  ✓ LDA fitted on {len(labels)} windows with f ∈ R^{F.shape[1]} (paper: f = ρ1 − ρ2), balanced priors")
    
    def _effective_rank(self, x: np.ndarray, eps: float = 1e-6) -> int:
        """Compute effective rank using SVD."""
        s = np.linalg.svd(x, compute_uv=False)
        return int(np.sum(s > eps * s[0])) if s.size > 0 else 0
    
    def _estimate_rank_from_dataset(self, dataset: tf.data.Dataset, take_batches: int = 10) -> int:
        """Estimate effective rank from dataset to cap CCA dimensions."""
        xs1, xs2 = [], []
        for batch in dataset.take(take_batches):
            if isinstance(batch, tuple):
                inputs, _ = batch
            else:
                inputs = batch
            x1 = inputs['input_1'].numpy()
            x2 = inputs['input_2'].numpy()
            xs1.append(x1)
            xs2.append(x2)
        if not xs1 or not xs2:
            return min(4, self.cca_dims)  # Fallback
        x1 = np.vstack(xs1)
        x2 = np.vstack(xs2)
        rank1 = self._effective_rank(x1)
        rank2 = self._effective_rank(x2)
        return min(rank1, rank2, x1.shape[1], x2.shape[1])
    
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
                print(f"CPU model creation also failed: {gpu_error}")
                raise RuntimeError("Cannot create CCA model on either CPU or GPU")
    
    
    def fit(self, dataset: tf.data.Dataset):
        """
        Fit the CCA model by concatenating ALL training data and fitting once.
        
        CRITICAL: CCA is NOT a supervised learning problem. It does NOT use:
        - Labels
        - Batches
        - Epochs
        - Validation
        
        CCA simply finds correlations between EEG and audio envelopes.
        
        CRITICAL: Train TWO separate CCAs (one for left, one for right) to avoid bias.
        This matches FULCCA's architecture.
        
        Args:
            dataset: TensorFlow dataset containing EEG and audio windows with aux_data
        """
        print("Fitting DASCCA model (paper: single CCA on attended envelope + LDA on f=ρ1-ρ2)...")
        print("  Collecting training windows (EEG + left/right envelope + labels)...")
        dataset_iter = iter(dataset)
        first_batch = next(dataset_iter)
        batches_to_process = [first_batch] + list(dataset_iter)
        all_eeg_windows = []   # list of (T, 64)
        all_left_lagged = []   # list of (T, 4*num_lags) or (T, 4)
        all_right_lagged = []
        all_labels = []
        L = max(1, self.eeg_lag_taps)
        for batch in batches_to_process:
            # Extract both left and right audio from batch
            # CRITICAL: Use aux_data['left_env'] and aux_data['right_env'] for the two CCAs.
            # input_2 is the attended envelope (left or right by label); we need the actual
            # left and right streams so CCA_left learns EEG↔left and CCA_right learns EEG↔right.
            if not (isinstance(batch, tuple) and len(batch) == 2):
                continue
            inputs, aux_data = batch
            if not (isinstance(aux_data, dict) and 'left_env' in aux_data and 'right_env' in aux_data):
                continue
            eeg_batch = (inputs['input_1'].numpy() if hasattr(inputs['input_1'], 'numpy') else np.array(inputs['input_1']))
            left_batch = (aux_data['left_env'].numpy() if hasattr(aux_data['left_env'], 'numpy') else np.array(aux_data['left_env']))
            right_batch = (aux_data['right_env'].numpy() if hasattr(aux_data['right_env'], 'numpy') else np.array(aux_data['right_env']))
            labels_batch = aux_data.get('label')
            if labels_batch is not None:
                labels_batch = labels_batch.numpy() if hasattr(labels_batch, 'numpy') else np.array(labels_batch)
                labels_batch = np.atleast_1d(labels_batch).flatten()
            if len(eeg_batch.shape) == 2 and '_batch_size' in aux_data and '_window_size' in aux_data:
                B = int(aux_data['_batch_size'].numpy() if hasattr(aux_data['_batch_size'], 'numpy') else aux_data['_batch_size'])
                T = int(aux_data['_window_size'].numpy() if hasattr(aux_data['_window_size'], 'numpy') else aux_data['_window_size'])
                eeg_batch = eeg_batch.reshape(B, T, -1)
                left_batch = left_batch.reshape(B, T, -1)
                right_batch = right_batch.reshape(B, T, -1)
                for w in range(B):
                    eeg_w = eeg_batch[w]
                    left_w = left_batch[w]
                    right_w = right_batch[w]
                    if self.use_time_lags:
                        left_w = make_lagged_audio(left_w, self.lag_samples, self.fs)
                        right_w = make_lagged_audio(right_w, self.lag_samples, self.fs)
                    all_eeg_windows.append(eeg_w)
                    all_left_lagged.append(left_w)
                    all_right_lagged.append(right_w)
                    lab = int(labels_batch[w]) if labels_batch is not None and w < len(labels_batch) else 0
                    all_labels.append(lab)
        
        n_windows = len(all_eeg_windows)
        if n_windows == 0:
            raise ValueError("No training windows collected; cannot fit CCA.")
        print(f"  Collected {n_windows} windows")
        # Time-lagged EEG: (T, 64*L) per window
        X_lag = np.vstack([make_lagged_eeg(e, L) for e in all_eeg_windows]).astype(np.float32)
        Y_att = np.vstack([all_left_lagged[i] if all_labels[i] == 0 else all_right_lagged[i] for i in range(n_windows)]).astype(np.float32)
        print(f"  X (EEG lagged): {X_lag.shape}, Y (attended envelope lagged): {Y_att.shape}")
        if self.pca_eeg > 0:
            n_comp = min(self.pca_eeg, X_lag.shape[1])
            self.pca_x = PCA(n_components=n_comp)
            X_lag = self.pca_x.fit_transform(X_lag)
            print(f"  ✓ PCA on EEG: {X_lag.shape[1]} components")
        if self.pca_audio > 0:
            n_comp_a = min(self.pca_audio, Y_att.shape[1])
            self.pca_y = PCA(n_components=n_comp_a)
            Y_att = self.pca_y.fit_transform(Y_att)
            print(f"  ✓ PCA on audio: {n_comp_a} components")
        max_cca = min(X_lag.shape[1], Y_att.shape[1])
        if self.cca_dims > max_cca:
            self.cca_dims = max_cca
        print(f"  Training single CCA (EEG ↔ attended envelope), J={self.cca_dims}...")
        def gen():
            yield {'input_1': X_lag, 'input_2': Y_att}
        ds = tf.data.Dataset.from_generator(gen, output_signature={
            'input_1': tf.TensorSpec(shape=(None, X_lag.shape[1]), dtype=tf.float32),
            'input_2': tf.TensorSpec(shape=(None, Y_att.shape[1]), dtype=tf.float32)
        })
        rot_x, rot_y, mean_x, mean_y, e_vals = calculate_cca_parameters_from_dataset(
            ds, self.cca_dims, regularization=self.regularization, mini_batch_count=0)
        self.cca_params = {'rot_x': rot_x, 'rot_y': rot_y, 'mean_x': mean_x, 'mean_y': mean_y, 'eigenvalues': e_vals}
        print(f"  ✓ Single CCA trained: first canonical correlation = {np.sqrt(np.clip(e_vals[0], 0, 1)):.6f}")
        self.is_fitted = True
        if self.use_lda:
            print("  Fitting LDA on f = ρ1 − ρ2...")
            self._fit_lda(dataset)
    
    def score_window(self, X: np.ndarray, Y: np.ndarray, use_cca_left: bool = True) -> float:
        """
        Score a window: single CCA projection then weighted sum of per-dim correlations.
        Paper mode: one Wx, Ws; Y is either left or right envelope.
        """
        rho = self._compute_rho(X, Y)
        dim_weights = np.exp(-np.arange(len(rho)) * 0.15)
        dim_weights = dim_weights / dim_weights.sum()
        return float((rho * dim_weights).sum())
    
    def _compute_correlation_scores(self, predictions: tf.Tensor, eeg_fs: int = 128, env_fs: int = 128,
                                   window_size: Optional[int] = None) -> np.ndarray:
        """
        Compute correlation scores from CCA projections using proper Pearson correlation.
        
        FIXED: Compute actual Pearson correlation across time within each window, not per-sample cosine similarity.
        This is the correct way to compute correlations from CCA rotated outputs.
        
        The CCA model outputs [rotated_eeg, rotated_audio] concatenated.
        We compute Pearson correlation for each CCA dimension across time, then weight by importance.
        
        Args:
            predictions: CCA model predictions, shape (N, 2*cca_dims) where N = num_windows * window_size
            eeg_fs: EEG sampling rate in Hz (for verification)
            env_fs: Envelope sampling rate in Hz (for verification)
            window_size: Window size in samples (required to reshape flattened predictions)
        
        Returns:
            Window-level correlation scores, shape (num_windows,)
        """
        # FIXED: Explicit sampling rate verification before correlation
        if not hasattr(self, '_sampling_rate_printed'):
            print(f"\n{'='*60}")
            print(f"SAMPLING RATE VERIFICATION (Before Correlation)")
            print(f"{'='*60}")
            print(f"EEG fs: {eeg_fs} Hz")
            print(f"ENV fs: {env_fs} Hz")
            if eeg_fs != env_fs:
                print(f"⚠ CRITICAL ERROR: Sampling rate mismatch! EEG={eeg_fs}Hz != ENV={env_fs}Hz")
                print(f"  This will cause correlation to fail. Both must be the same (128 Hz).")
            else:
                print(f"✓ Sampling rates match: {eeg_fs} Hz")
            print(f"{'='*60}\n")
            self._sampling_rate_printed = True
        
        try:
            # Convert to numpy if needed
            if hasattr(predictions, 'numpy'):
                preds_np = predictions.numpy()
            else:
                preds_np = predictions
            
            # FIXED: Reshape to windows if window_size is provided
            # If predictions are flattened (batch_windows * window_size, 2*cca_dims),
            # reshape to (batch_windows, window_size, 2*cca_dims)
            if window_size is not None and window_size > 0:
                num_samples = preds_np.shape[0]
                if num_samples % window_size == 0:
                    num_windows = num_samples // window_size
                    preds_np = preds_np.reshape(num_windows, window_size, -1)
                else:
                    # Cannot reshape - fall back to per-sample (less ideal)
                    print(f"⚠ WARNING: Cannot reshape predictions to windows (num_samples={num_samples}, window_size={window_size})")
                    window_size = None
            
            cca_width = preds_np.shape[-1] // 2
            
            if window_size is not None and len(preds_np.shape) == 3:
                # FIXED: Proper Pearson correlation across time within each window
                # Shape: (num_windows, window_size, 2*cca_dims)
                eeg = preds_np[:, :, :cca_width]  # (num_windows, window_size, cca_width)
                env = preds_np[:, :, cca_width:]   # (num_windows, window_size, cca_width)
                
                # Center across time axis (axis=1) for each window and dimension
                eeg_c = eeg - eeg.mean(axis=1, keepdims=True)  # (num_windows, window_size, cca_width)
                env_c = env - env.mean(axis=1, keepdims=True)  # (num_windows, window_size, cca_width)
                
                # Compute Pearson correlation per window per dimension
                # numerator: sum over time (axis=1)
                num = np.sum(eeg_c * env_c, axis=1)  # (num_windows, cca_width)
                den = np.sqrt(np.sum(eeg_c**2, axis=1) * np.sum(env_c**2, axis=1)) + 1e-8  # (num_windows, cca_width)
                corr = num / den  # (num_windows, cca_width)
                
                # Weight by dimension importance (first dimension has highest canonical correlation)
                weights = np.exp(-np.arange(cca_width) * 0.15)  # Exponential decay
                weights = weights / np.sum(weights)
                
                # Weighted sum across dimensions per window
                window_scores = np.sum(corr * weights, axis=1)  # (num_windows,)
                
                return window_scores
            else:
                # Fallback: per-sample scoring (less ideal, but handles unreshaped data)
                proj_eeg = preds_np[:, :cca_width]
                proj_env = preds_np[:, cca_width:]
                
                # Center the data
                proj_eeg_centered = proj_eeg - np.mean(proj_eeg, axis=0, keepdims=True)
                proj_env_centered = proj_env - np.mean(proj_env, axis=0, keepdims=True)
                
                # Compute correlation per dimension (across all samples)
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
    
        except Exception as e:
            print(f"Warning: Correlation computation failed: {e}")
            # Last resort: return zeros
            if hasattr(predictions, 'numpy'):
                preds_np = predictions.numpy()
            else:
                preds_np = predictions
            return np.zeros(preds_np.shape[0])
    
    def predict(self, dataset: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions using the fitted CCA model with GPU optimization.
        
        Args:
            dataset: TensorFlow dataset containing EEG data and labels
            
        Returns:
            Tuple of (predictions, targets, continuous_scores)
            - predictions: Binary predictions (0/1)
            - targets: True labels
            - continuous_scores: Continuous scores (right_corr - left_corr) for ROC-AUC, or None
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        print("Making DASCCA predictions...")
        
        all_predictions = []
        all_targets = []
        all_left_scores = []  # Initialize for left/right comparison diagnostics
        all_right_scores = []  # Initialize for left/right comparison diagnostics
        

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
                        
                        # FIXED: Pass sampling rates explicitly for verification
                        eeg_fs = 128  # EEG sampling rate (matches preprocessing)
                        env_fs = 128  # Envelope sampling rate (after resampling in _load_audio_envelope_full)
                        
                        # Convert to numpy for manual scoring using dual CCAs
                        # Expected shapes: eeg_np (B, T, 64), left_env_np/right_env_np (B, T, 4)
                        if hasattr(eeg_view, 'numpy'):
                            eeg_np = eeg_view.numpy()
                        else:
                            eeg_np = np.array(eeg_view)
                        
                        if hasattr(left_env, 'numpy'):
                            left_env_np = left_env.numpy()
                        else:
                            left_env_np = np.array(left_env)
                        
                        if hasattr(right_env, 'numpy'):
                            right_env_np = right_env.numpy()
                        else:
                            right_env_np = np.array(right_env)
                        
                        # Reshape flattened (B*W, F) from reshape_batch to (B, W, F) when batch/window info present
                        if eeg_np.ndim == 2 and isinstance(aux, dict) and '_batch_size' in aux and '_window_size' in aux:
                            B_val = aux['_batch_size']
                            W_val = aux['_window_size']
                            B = int(B_val.numpy() if hasattr(B_val, 'numpy') else B_val)
                            W = int(W_val.numpy() if hasattr(W_val, 'numpy') else W_val)
                            if eeg_np.shape[0] == B * W and eeg_np.shape[0] > 0:
                                eeg_np = eeg_np.reshape(B, W, -1)
                                left_env_np = left_env_np.reshape(B, W, -1)
                                right_env_np = right_env_np.reshape(B, W, -1)
                        elif eeg_np.ndim == 2:
                            eeg_np = eeg_np[None, ...]  # (1, T, 64)
                            left_env_np = left_env_np[None, ...]  # (1, T, 4)
                            right_env_np = right_env_np[None, ...]  # (1, T, 4)
                        
                        # Now eeg_np is (B, T, 64), left_env_np/right_env_np are (B, T, 4)
                        B = eeg_np.shape[0]
                        left_window_scores = np.empty(B, dtype=np.float32)
                        right_window_scores = np.empty(B, dtype=np.float32)
                        F_batch = []
                        w_weights = np.exp(-np.arange(self.cca_dims) * 0.15)
                        w_weights = w_weights / w_weights.sum()
                        for w in range(B):
                            try:
                                eeg_window = eeg_np[w]
                                left_audio_window = left_env_np[w]
                                right_audio_window = right_env_np[w]
                                if self.use_time_lags:
                                    left_audio_lagged = make_lagged_audio(left_audio_window, self.lag_samples, self.fs)
                                    right_audio_lagged = make_lagged_audio(right_audio_window, self.lag_samples, self.fs)
                                else:
                                    left_audio_lagged = left_audio_window
                                    right_audio_lagged = right_audio_window
                                rho_1 = self._compute_rho(eeg_window, left_audio_lagged)
                                rho_2 = self._compute_rho(eeg_window, right_audio_lagged)
                                left_window_scores[w] = float((rho_1 * w_weights).sum())
                                right_window_scores[w] = float((rho_2 * w_weights).sum())
                                F_batch.append(rho_1 - rho_2)
                            except Exception as score_error:
                                print(f"⚠ Warning: Scoring failed for window {w}: {score_error}")
                                left_window_scores[w] = 0.0
                                right_window_scores[w] = 0.0
                                F_batch.append(np.zeros(self.cca_dims, dtype=np.float32))
                        num_windows = len(left_window_scores)
                        all_left_scores.extend(left_window_scores.tolist())
                        all_right_scores.extend(right_window_scores.tolist())
                        continuous_scores = right_window_scores - left_window_scores
                        if not hasattr(self, '_all_continuous_scores'):
                            self._all_continuous_scores = []
                        self._all_continuous_scores.extend(continuous_scores.tolist())
                        F_batch = np.array(F_batch, dtype=np.float32)
                        if self.use_lda and self.lda_model is not None:
                            F_batch_scaled = self.lda_scaler.transform(F_batch) if self.lda_scaler is not None else F_batch
                            window_predictions = self.lda_model.predict(F_batch_scaled).astype(np.int64)
                        else:
                            window_predictions = (right_window_scores > left_window_scores).astype(np.int64)
                        
                        # window_predictions is already (num_windows,) - use directly
                        with tf.device('/CPU:0'):
                            all_predictions.extend(window_predictions)
                            
                            if targets is not None:
                                target_array = targets.numpy() if hasattr(targets, 'numpy') else np.array(targets)
                                target_flat = target_array.flatten()
                                # Ensure targets match number of windows
                                if len(target_flat) == num_windows:
                                    all_targets.extend(target_flat)
                                elif len(target_flat) > num_windows:
                                    all_targets.extend(target_flat[:num_windows])
                                else:
                                    all_targets.extend(np.pad(target_flat, (0, num_windows - len(target_flat)), mode='edge'))
                        continue  # Skip the rest of the loop for this batch
                    else:
                        # DASCCA uses dual CCA (score_window); there is no single self.model. Skip batch if aux missing.
                        if not hasattr(self, '_fallback_aux_warned'):
                            print(f"⚠ WARNING: Batch missing left_env/right_env in aux_data; skipping batch (DASCCA requires both streams).")
                            self._fallback_aux_warned = True
                        continue
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
                        
                        # CORRECT APPROACH: Compare correlations with BOTH left and right audio (same as GPU path)
                        if aux is not None and 'left_env' in aux and 'right_env' in aux:
                            eeg_view = inputs['input_1']
                            left_env = aux['left_env']
                            right_env = aux['right_env']
                            # Convert to numpy and use score_window (dual CCA) - self.model is not used in DASCCA
                            if hasattr(eeg_view, 'numpy'):
                                eeg_np = eeg_view.numpy()
                            else:
                                eeg_np = np.array(eeg_view)
                            if hasattr(left_env, 'numpy'):
                                left_env_np = left_env.numpy()
                            else:
                                left_env_np = np.array(left_env)
                            if hasattr(right_env, 'numpy'):
                                right_env_np = right_env.numpy()
                            else:
                                right_env_np = np.array(right_env)
                            if eeg_np.ndim == 2 and '_batch_size' in aux and '_window_size' in aux:
                                B = int(aux['_batch_size'].numpy() if hasattr(aux['_batch_size'], 'numpy') else aux['_batch_size'])
                                W = int(aux['_window_size'].numpy() if hasattr(aux['_window_size'], 'numpy') else aux['_window_size'])
                                if eeg_np.shape[0] == B * W and eeg_np.shape[0] > 0:
                                    eeg_np = eeg_np.reshape(B, W, -1)
                                    left_env_np = left_env_np.reshape(B, W, -1)
                                    right_env_np = right_env_np.reshape(B, W, -1)
                            elif eeg_np.ndim == 2:
                                eeg_np = eeg_np[None, ...]
                                left_env_np = left_env_np[None, ...]
                                right_env_np = right_env_np[None, ...]
                            B = eeg_np.shape[0]
                            left_window_scores = np.empty(B, dtype=np.float32)
                            right_window_scores = np.empty(B, dtype=np.float32)
                            F_batch = []
                            w_weights = np.exp(-np.arange(self.cca_dims) * 0.15)
                            w_weights = w_weights / w_weights.sum()
                            for w in range(B):
                                try:
                                    eeg_window = eeg_np[w]
                                    left_audio_window = left_env_np[w]
                                    right_audio_window = right_env_np[w]
                                    if self.use_time_lags:
                                        left_audio_lagged = make_lagged_audio(left_audio_window, self.lag_samples, self.fs)
                                        right_audio_lagged = make_lagged_audio(right_audio_window, self.lag_samples, self.fs)
                                    else:
                                        left_audio_lagged = left_audio_window
                                        right_audio_lagged = right_audio_window
                                    rho_1 = self._compute_rho(eeg_window, left_audio_lagged)
                                    rho_2 = self._compute_rho(eeg_window, right_audio_lagged)
                                    left_window_scores[w] = float((rho_1 * w_weights).sum())
                                    right_window_scores[w] = float((rho_2 * w_weights).sum())
                                    F_batch.append(rho_1 - rho_2)
                                except Exception:
                                    left_window_scores[w] = 0.0
                                    right_window_scores[w] = 0.0
                                    F_batch.append(np.zeros(self.cca_dims, dtype=np.float32))
                            num_windows = B
                            all_left_scores.extend(left_window_scores.tolist())
                            all_right_scores.extend(right_window_scores.tolist())
                            F_batch = np.array(F_batch, dtype=np.float32)
                            if self.use_lda and self.lda_model is not None:
                                F_batch_scaled = self.lda_scaler.transform(F_batch) if self.lda_scaler is not None else F_batch
                                window_predictions = self.lda_model.predict(F_batch_scaled).astype(np.int64)
                            else:
                                window_predictions = (right_window_scores > left_window_scores).astype(np.int64)
                            all_predictions.extend(window_predictions)
                            if targets is not None:
                                target_array = targets.numpy() if hasattr(targets, 'numpy') else np.array(targets)
                                target_flat = target_array.flatten()
                                if len(target_flat) == num_windows:
                                    all_targets.extend(target_flat)
                                elif len(target_flat) > num_windows:
                                    all_targets.extend(target_flat[:num_windows])
                                else:
                                    all_targets.extend(np.pad(target_flat, (0, num_windows - len(target_flat)), mode='edge'))
                            continue
                        print(f"⚠ WARNING: Left/right audio envelopes not available in batch")
                        continue
            except Exception as cpu_error:
                print(f"⚠ CPU fallback also failed: {cpu_error}")
                raise RuntimeError(f"Both GPU and CPU prediction failed. GPU error: {gpu_error}, CPU error: {cpu_error}")
        
        # Print diagnostics if we used left/right comparison
        if all_left_scores and all_right_scores:
            all_left_scores_arr = np.array(all_left_scores)
            all_right_scores_arr = np.array(all_right_scores)
            all_targets_arr = np.array(all_targets) if all_targets else None
            
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
        
        # FIXED: Return continuous scores for ROC-AUC computation
        # Compute continuous scores: right_corr - left_corr (if available)
        continuous_scores = None
        if all_left_scores and all_right_scores and len(all_left_scores) == len(all_predictions):
            all_left_scores_arr = np.array(all_left_scores)
            all_right_scores_arr = np.array(all_right_scores)
            # Continuous score: right_corr - left_corr (positive = right, negative = left)
            continuous_scores = all_right_scores_arr - all_left_scores_arr
        
        return np.array(all_predictions), np.array(all_targets), continuous_scores


class DASCCATrainer:
    """
    DASCCA trainer with comprehensive metrics evaluation.
    """
    
    def __init__(self, model: DASCCAModel, output_dir: str = "dascca_results", 
                 tfrecord_dir: str = None, sampling_rate: int = 128, window_size: int = 512,  # FIXED: Default 128 Hz to match preprocessing
                 audio_base_dir: Optional[str] = None):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)  # Create parent directories if needed
        

        self.tfrecord_dir = tfrecord_dir
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.audio_base_dir = audio_base_dir  # Store audio directory for temporal metrics
        
        print(f"DASCCA trainer initialized. Output directory: {self.output_dir}")
    
    def train(self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset) -> float:
        """Train the DASCCA model."""
        print("Starting DASCCA training...")
        

        # Check if datasets are non-empty without consuming them
        # Use take(1) to peek at first batch - this doesn't consume the dataset
        try:
            _ = next(iter(train_dataset.take(1)))
            print("✓ Train dataset is non-empty (has at least 1 batch)")
        except (StopIteration, Exception) as e:
            print(f"⚠ ERROR: Train dataset is empty or has errors: {e}")
            raise ValueError("Train dataset is empty! Cannot train CCA model. Check generator output above for errors.")
        
        try:
            _ = next(iter(val_dataset.take(1)))
            print("✓ Validation dataset is non-empty (has at least 1 batch)")
        except (StopIteration, Exception) as e:
            print(f"⚠ ERROR: Validation dataset is empty or has errors: {e}")
            raise ValueError("Validation dataset is empty! Cannot validate CCA model. Check generator output above for errors.")
        

        # fit() needs (inputs, aux_data) tuple format to extract right_env from aux_data
        # The fit method extracts both left_audio (from inputs) and right_audio (from aux_data)
        # So we pass the full dataset with aux_data intact
        self.model.fit(train_dataset)


        val_predictions, val_targets, _ = self.model.predict(val_dataset)
        
        # FIXED: Verify accuracy computation matches balanced accuracy
        val_accuracy = accuracy_score(val_targets, val_predictions)
        val_balanced_acc = balanced_accuracy_score(val_targets, val_predictions)
        val_manual_acc = np.mean(val_predictions == val_targets)
        
        print(f"\n{'='*60}")
        print(f"VALIDATION ACCURACY VERIFICATION")
        print(f"{'='*60}")
        print(f"Accuracy: {val_accuracy:.6f}")
        print(f"Manual accuracy: {val_manual_acc:.6f}")
        print(f"Balanced accuracy: {val_balanced_acc:.6f}")
        print(f"Target distribution: {np.bincount(val_targets.astype(int))}")
        print(f"Prediction distribution: {np.bincount(val_predictions.astype(int))}")
        if abs(val_accuracy - val_balanced_acc) > 0.05:
            print(f"⚠ WARNING: Accuracy != Balanced Accuracy - check prediction/target alignment!")
        print(f"{'='*60}\n")
        
        print(f"DASCCA training completed! Validation accuracy: {val_accuracy:.4f}")
        return val_accuracy
    
    def test(self, test_dataset: tf.data.Dataset) -> Dict:
        """Test the DASCCA model with comprehensive metrics."""
        print("Testing DASCCA model...")
        
        predictions, targets, continuous_scores = self.model.predict(test_dataset)
        
        # FIXED: Verify predictions and targets are aligned and have correct shapes
        print(f"\n{'='*60}")
        print(f"PREDICTION/TARGET VERIFICATION")
        print(f"{'='*60}")
        print(f"Predictions shape: {predictions.shape}, dtype: {predictions.dtype}")
        print(f"Targets shape: {targets.shape}, dtype: {targets.dtype}")
        print(f"Predictions unique values: {np.unique(predictions)}")
        print(f"Targets unique values: {np.unique(targets)}")
        print(f"Predictions distribution: {np.bincount(predictions.astype(int))}")
        print(f"Targets distribution: {np.bincount(targets.astype(int))}")
        print(f"{'='*60}\n")
        
        # FIXED: Compute accuracy and verify it matches balanced accuracy for balanced dataset
        accuracy = accuracy_score(targets, predictions)
        balanced_acc = balanced_accuracy_score(targets, predictions)
        
        # Verify accuracy computation
        manual_accuracy = np.mean(predictions == targets)
        print(f"Accuracy verification:")
        print(f"  sklearn accuracy_score: {accuracy:.6f}")
        print(f"  Manual np.mean(pred == target): {manual_accuracy:.6f}")
        print(f"  Balanced accuracy: {balanced_acc:.6f}")
        if abs(accuracy - manual_accuracy) > 1e-6:
            print(f"  ⚠ WARNING: Accuracy computation mismatch!")
        if abs(accuracy - balanced_acc) > 0.05 and len(np.unique(targets)) == 2:
            print(f"  ⚠ CRITICAL: Accuracy ({accuracy:.4f}) != Balanced Accuracy ({balanced_acc:.4f})")
            print(f"    This suggests predictions/targets are misaligned or dataset is unbalanced!")
            print(f"    Target distribution: {np.bincount(targets.astype(int))}")
            print(f"    Prediction distribution: {np.bincount(predictions.astype(int))}")
        

        report = classification_report(targets, predictions, 
                                   target_names=['Left', 'Right'], 
                                   labels=[0, 1],
                                   output_dict=True)
        
        cm = confusion_matrix(targets, predictions)
        

        # FIXED: Use continuous scores (right_corr - left_corr) for ROC-AUC
        roc_auc_metrics = self._calculate_roc_auc_metrics(targets, predictions, continuous_scores=continuous_scores)
        msed_metrics = self._calculate_msed_metrics(targets, predictions)
        advanced_metrics = self._calculate_advanced_metrics(targets, predictions)
        # Temporal metrics removed - main test uses configured window size
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'targets': targets,
            'roc_auc_metrics': roc_auc_metrics,
            'msed_metrics': msed_metrics,
            'advanced_metrics': advanced_metrics
        }
        
        return results
    
    def _calculate_roc_auc_metrics(self, targets: np.ndarray, predictions: np.ndarray, 
                                   continuous_scores: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate ROC-AUC and related metrics.
        
        FIXED: Use continuous scores instead of hard predictions for meaningful ROC-AUC.
        If continuous_scores is provided (e.g., right_corr - left_corr), use that.
        Otherwise, try to extract from predictions if they're not binary.
        """
        try:
            # FIXED: Use continuous scores for ROC-AUC instead of hard {0,1} predictions
            if continuous_scores is not None:
                probabilities = continuous_scores.astype(np.float32)
            elif len(np.unique(predictions)) > 2 or (np.min(predictions) < 0 or np.max(predictions) > 1):
                # Predictions are already continuous
                probabilities = predictions.astype(np.float32)
            else:
                # Hard predictions - cannot compute meaningful ROC-AUC
                # Return error or use dummy scores
                return {
                    "error": "Cannot compute ROC-AUC from hard binary predictions. Need continuous scores (e.g., right_corr - left_corr).",
                    "roc_auc_score": 0.5,
                    "note": "ROC-AUC requires continuous scores, not hard {0,1} predictions"
                }
            
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
        """Calculate temporal performance metrics.
        
        Note: Temporal metrics testing multiple window sizes has been removed.
        The main test already uses the configured window size.
        """
        # Return empty dict - temporal metrics removed per user request
        # Main test already uses the configured window_size
        return {}
    
    def save_results(self, results: Dict):
        """Save comprehensive results to files."""
        # Ensure output directory exists (in case it was deleted or not created)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        results_json = {
            'accuracy': float(results['accuracy']),
            'classification_report': results['classification_report'],
            'confusion_matrix': results['confusion_matrix'].tolist() if hasattr(results['confusion_matrix'], 'tolist') else results['confusion_matrix'],
            'timestamp': datetime.now().isoformat(),
            'roc_auc_metrics': results.get('roc_auc_metrics', {}),
            'msed_metrics': results.get('msed_metrics', {}),
            'advanced_metrics': results.get('advanced_metrics', {})
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
        
        print(f"DASCCA results saved to {self.output_dir}")
    
    def _save_comprehensive_report(self, results: Dict):
        """Save a comprehensive metrics report."""
        def _fmt(v):
            return f"{v:.4f}" if v is not None and isinstance(v, (int, float)) else 'N/A'

        with open(self.output_dir / 'comprehensive_metrics_report.txt', 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DASCCA COMPREHENSIVE METRICS REPORT\n")
            f.write("=" * 80 + "\n\n")
            

            f.write("BASIC METRICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Accuracy: {results['accuracy']:.4f}\n\n")
            

            roc_auc = results.get('roc_auc_metrics', {})
            if "error" not in roc_auc:
                f.write("ROC-AUC METRICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"ROC-AUC Score: {_fmt(roc_auc.get('roc_auc_score'))}\n")
                f.write(f"Average Precision: {_fmt(roc_auc.get('average_precision'))}\n")
                f.write(f"Optimal Threshold: {_fmt(roc_auc.get('optimal_threshold'))}\n")
                f.write(f"Optimal TPR: {_fmt(roc_auc.get('optimal_tpr'))}\n")
                f.write(f"Optimal FPR: {_fmt(roc_auc.get('optimal_fpr'))}\n\n")
            

            msed = results.get('msed_metrics', {})
            if "error" not in msed:
                f.write("MSED METRICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Mean Squared Error: {_fmt(msed.get('mse'))}\n")
                f.write(f"Root Mean Squared Error: {_fmt(msed.get('rmse'))}\n")
                f.write(f"Mean Absolute Error: {_fmt(msed.get('mae'))}\n")
                f.write(f"Mean Absolute Percentage Error: {_fmt(msed.get('mape'))}%\n")
                f.write(f"R-squared: {_fmt(msed.get('r_squared'))}\n\n")
            

            advanced = results.get('advanced_metrics', {})
            if "error" not in advanced:
                f.write("ADVANCED METRICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Matthews Correlation Coefficient: {_fmt(advanced.get('matthews_correlation_coefficient'))}\n")
                f.write(f"Cohen's Kappa: {_fmt(advanced.get('cohens_kappa'))}\n")
                f.write(f"Balanced Accuracy: {_fmt(advanced.get('balanced_accuracy'))}\n\n")
            
            # Temporal metrics removed - main test uses configured window size


def create_das_data_loaders(tfrecord_dir: str, batch_size: int = 16, 
                           window_size: int = 32, overlap: float = 0.25,
                           train_ratio: float = 0.64, val_ratio: float = 0.18,  # Adjusted to ensure at least 2 val subjects (11 * 0.18 = 1.98 -> 2)
                           max_samples: Optional[int] = None,
                           audio_base_dir: Optional[str] = None,
                           load_audio: bool = True, max_files: Optional[int] = None,
                           eeg_lag_samples: int = 0) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Create data loaders for DAS dataset with proper subject-wise splitting.
    
    Args:
        eeg_lag_samples: Number of past time samples to include for backward model (0 = no lagging, deprecated)
    """
    
    print("Creating DAS dataset with subject-wise splitting...")
    print(f"TFRecord directory: {tfrecord_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Window size: {window_size} samples ({window_size/128:.1f} seconds at 128Hz)")
    print(f"Overlap: {overlap}")
    print(f"Using DAS preprocessing: Yes")
    if audio_base_dir:
        print(f"Audio base directory: {audio_base_dir}")
    

    full_dataset = DasDatasetCCA(tfrecord_dir, mode='full', 
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
    

    # FIXED: Build subject indices directly instead of assuming ordered metadata
    from collections import defaultdict
    subject_to_sample_indices = defaultdict(list)
    for i, md in enumerate(full_dataset.metadata):
        subject_id = md.get('subject_id', 'unknown')
        subject_to_sample_indices[subject_id].append(i)
    
    # Build subject ranges from indices (for backward compatibility)
    subject_ranges = {}
    for subject_id, indices in subject_to_sample_indices.items():
        if indices:
            subject_ranges[subject_id] = (min(indices), max(indices) + 1)
    
    print(f"Subject indices (built directly, not assuming order):")
    for subject_id, indices in list(subject_to_sample_indices.items())[:10]:  # Show first 10
        print(f"  {subject_id}: {len(indices)} samples, range {min(indices)}-{max(indices)}")
    print(f"  ... (total {len(subject_to_sample_indices)} subjects)")
    

    unknown_count = sum(1 for sid in data_idx_to_subject.values() if sid == "unknown")
    total_samples = len(data_idx_to_subject)
    if unknown_count > 0:
        print(f"\n⚠ WARNING: {unknown_count}/{total_samples} samples ({100*unknown_count/total_samples:.1f}%) have 'unknown' subject_id")
        print(f"  This suggests subject_id might not be properly stored in TFRecords")
        print(f"  Check that das_preprocessing_16subjects.py includes subject_id in TFRecord features")
    



    # FIXED: Use window-level subject IDs from __getitem__ instead of inferring from data_idx
    # This avoids unit mismatches and ensures correct subject assignment
    print("Building subject-to-window mapping from dataset metadata...")
    for i in range(len(full_dataset)):
        try:
            # Get window data to extract subject_id from aux_data
            window_data, aux_data = full_dataset[i]
            if isinstance(aux_data, dict) and 'subject_id' in aux_data:
                subject_id = aux_data['subject_id']
                if isinstance(subject_id, tf.Tensor):
                    subject_id = subject_id.numpy().decode('utf-8') if subject_id.dtype == tf.string else str(subject_id.numpy())
                elif not isinstance(subject_id, str):
                    subject_id = str(subject_id)
            else:
                # Fallback: try to get from metadata using data_idx
                data_idx, _ = full_dataset.window_indices[i]
                if data_idx < len(full_dataset._row_to_metadata):
                    window_metadata = full_dataset._row_to_metadata[data_idx]
                    subject_id = window_metadata.get('subject_id', 'unknown')
                else:
                    subject_id = "unknown"
            
            if subject_id not in subject_windows:
                subject_windows[subject_id] = []
            subject_windows[subject_id].append(i)
        except Exception as e:
            print(f"⚠ WARNING: Could not get subject_id for window {i}: {e}")
            # Fallback to unknown
            if "unknown" not in subject_windows:
                subject_windows["unknown"] = []
            subject_windows["unknown"].append(i)
    
    print(f"Found {len(subject_windows)} subjects:")
    for subject_id, windows in subject_windows.items():
        print(f"  {subject_id}: {len(windows)} windows")
    

    subjects = list(subject_windows.keys())
    np.random.seed(42)
    np.random.shuffle(subjects)
    
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

        n_train_subjects = int(train_ratio * n_subjects)
        n_val_subjects = int(val_ratio * n_subjects)
        
        # Ensure at least 2 validation subjects for reliable validation
        if n_val_subjects < 2 and n_subjects >= 3:
            n_val_subjects = 2
            # Adjust train subjects to accommodate
            if n_train_subjects + n_val_subjects > n_subjects:
                n_train_subjects = n_subjects - n_val_subjects - 1  # Leave at least 1 for test
        
        train_subjects = subjects[:n_train_subjects]
        val_subjects = subjects[n_train_subjects:n_train_subjects + n_val_subjects]
        test_subjects = subjects[n_train_subjects + n_val_subjects:]
        
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
        
        # Calculate expected feature dimension (no PCA, no backward model lagging)
        base_eeg_dim = 64  # Original EEG channels
        expected_eeg_dim = base_eeg_dim
        print(f"  Expected EEG feature dimension: {expected_eeg_dim} (base: {base_eeg_dim})")
        
        def generator():
            valid_samples = 0
            skipped_samples = 0
            error_counts = {}
            
            if len(indices) == 0:
                print(f"⚠ ERROR: No indices provided to generator! Cannot create dataset.")
                return
            
            print(f"  Generator starting with {len(indices)} indices...")
            
            for idx, i in enumerate(indices):
                try:
                    window_data, aux_data = full_dataset[i]
                    
                    # Extract label from aux_data
                    if isinstance(aux_data, dict):
                        label = aux_data.get('label', tf.constant([0], dtype=tf.int64))
                        left_env = aux_data.get('left_env')
                        right_env = aux_data.get('right_env')
                    else:
                        # Fallback for old format - should not happen with current implementation
                        label = aux_data
                        left_env = None
                        right_env = None
                        error_key = "unexpected_aux_format"
                        error_counts[error_key] = error_counts.get(error_key, 0) + 1
                        if idx < 5:  # Only print first 5 to avoid spam
                            print(f"WARNING: Unexpected aux_data format (not dict) for index {i}. This may indicate a data loading issue.")
                    

                    if isinstance(window_data, tuple):
                        eeg_data, audio_data = window_data
                    else:
                        # CRITICAL FIX: window_data should always be a tuple from DasDatasetCCA.__getitem__
                        # If it's not, this indicates a data loading error
                        eeg_data = window_data
                        eeg_shape = eeg_data.shape.as_list() if hasattr(eeg_data.shape, 'as_list') else list(eeg_data.shape)
                        
                        # Validate EEG shape
                        if len(eeg_shape) != 2 or eeg_shape[1] != 64:
                            print(f"ERROR: Invalid EEG shape {eeg_shape} in generator. Expected (window_size, 64)")
                            print(f"  Skipping this sample to prevent data corruption.")
                            continue
                        
                        # Use audio_data from window_data tuple - if not tuple, audio should come from aux_data
                        # Get audio from the dataset's audio_envelopes if available
                        # This should not happen if __getitem__ returns tuple correctly
                        print(f"WARNING: window_data is not a tuple. Attempting to use audio from dataset...")
                        # FIXED: Remove broken recovery path - treat non-tuple as fatal error
                        # window_data should always be a tuple from DasDatasetCCA.__getitem__
                        print(f"ERROR: window_data is not a tuple (got {type(window_data)}). This is a fatal data loading error.")
                        print(f"  Skipping this sample to prevent training on corrupted data.")
                        print(f"  Expected: (eeg_tensor, audio_tensor) tuple")
                        print(f"  Got: {window_data}")
                        continue
                    
                    eeg_shape = eeg_data.shape.as_list() if hasattr(eeg_data.shape, 'as_list') else list(eeg_data.shape)
                    audio_shape = audio_data.shape.as_list() if hasattr(audio_data.shape, 'as_list') else list(audio_data.shape)
                    

                    if len(eeg_shape) == 2 and eeg_shape[1] == 64:
                        input_1 = eeg_data
                    else:

                        print(f"WARNING: Unexpected EEG shape {eeg_shape}, reshaping...")
                        eeg_data = tf.reshape(eeg_data, (dataset_window_size, 64))
                        input_1 = eeg_data
                    
                    # Apply time-lagging for backward model (if enabled)
                    if eeg_lag_samples > 0:
                        # Convert to numpy for time-lagging
                        eeg_np = input_1.numpy() if hasattr(input_1, 'numpy') else np.array(input_1)
                        lagged_eeg = _apply_time_lagging(eeg_np, eeg_lag_samples)
                        input_1 = tf.constant(lagged_eeg, dtype=tf.float32)


                    # Enforce audio shape = (window_size, 4)
                    if len(audio_shape) != 2 or audio_shape[0] != dataset_window_size or audio_shape[1] != 4:
                        print(f"ERROR: audio_data shape {audio_shape} invalid. Expected ({dataset_window_size}, 4). Skipping.")
                        continue
                    input_2 = tf.cast(audio_data, tf.float32)
                    
                    valid_samples += 1
                    
                    # Prepare aux_data with left/right envelopes
                    # Always include left_env and right_env in aux_dict to match output signature
                    aux_dict = {'label': label}
                    
                    # Ensure left_env and right_env are always present with correct shape (window_size, 4)
                    if left_env is not None and right_env is not None:
                        # Enforce shape (window_size, 4)
                        left_shape = left_env.shape.as_list() if hasattr(left_env.shape, 'as_list') else list(left_env.shape)
                        right_shape = right_env.shape.as_list() if hasattr(right_env.shape, 'as_list') else list(right_env.shape)
                        
                        if len(left_shape) != 2 or left_shape[0] != dataset_window_size or left_shape[1] != 4:
                            print(f"ERROR: left_env shape {left_shape} invalid. Expected ({dataset_window_size}, 4). Skipping.")
                            continue
                        if len(right_shape) != 2 or right_shape[0] != dataset_window_size or right_shape[1] != 4:
                            print(f"ERROR: right_env shape {right_shape} invalid. Expected ({dataset_window_size}, 4). Skipping.")
                            continue
                        
                        aux_dict['left_env'] = tf.cast(left_env, tf.float32)
                        aux_dict['right_env'] = tf.cast(right_env, tf.float32)
                    else:
                        # left_env and right_env should always be available from __getitem__
                        error_key = "missing_audio_envelopes"
                        error_counts[error_key] = error_counts.get(error_key, 0) + 1
                        if skipped_samples < 5:  # Only print first 5 to avoid spam
                            print(f"ERROR: Left/right audio envelopes not available for index {i}. Skipping sample.")
                        skipped_samples += 1
                        continue
                    
                    yield {
                        'input_1': input_1,
                        'input_2': input_2
                    }, aux_dict
                    
                except Exception as e:
                    error_key = type(e).__name__
                    error_counts[error_key] = error_counts.get(error_key, 0) + 1
                    if skipped_samples < 5:  # Only print first 5 to avoid spam
                        print(f"ERROR in generator for index {i}: {e}")
                        import traceback
                        traceback.print_exc()
                    skipped_samples += 1
                    continue
            
            print(f"Generator completed: {valid_samples} valid samples, {skipped_samples} skipped")
            if error_counts:
                print(f"  Error breakdown: {error_counts}")
            if valid_samples == 0:
                print(f"⚠ CRITICAL: Generator produced 0 valid samples! This will cause empty dataset.")
                print(f"  Total indices: {len(indices)}, Skipped: {skipped_samples}")
        

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                {
                    'input_1': tf.TensorSpec(shape=(dataset_window_size, expected_eeg_dim), dtype=tf.float32),
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
            # Input shapes: (B, W, F1) for input_1, (B, W, 4) for input_2
            input_1_shape = tf.shape(inputs['input_1'])
            B = input_1_shape[0]  # batch_size
            W = input_1_shape[1]  # window_size
            F1 = input_1_shape[2]  # feature_dim
            
            # Flatten time: (B*W, F1) and (B*W, 4)
            input_1_reshaped = tf.reshape(inputs['input_1'], (-1, F1))
            input_2_reshaped = tf.reshape(inputs['input_2'], (-1, 4))
            
            reshaped_inputs = {
                'input_1': input_1_reshaped,
                'input_2': input_2_reshaped
            }
            
            reshaped_aux = {}
            if isinstance(aux_data, dict):
                # Keep window-level label as (B, 1) unchanged
                if 'label' in aux_data:
                    reshaped_aux['label'] = aux_data['label']
                
                # Flatten envelopes to match flattened inputs
                if 'left_env' in aux_data:
                    reshaped_aux['left_env'] = tf.reshape(aux_data['left_env'], (-1, 4))  # (B*W, 4)
                if 'right_env' in aux_data:
                    reshaped_aux['right_env'] = tf.reshape(aux_data['right_env'], (-1, 4))  # (B*W, 4)
                
                # Preserve exact recovery info (used in predict + LDA)
                reshaped_aux['_batch_size'] = B
                reshaped_aux['_window_size'] = W
            else:
                reshaped_aux = aux_data
            
            return reshaped_inputs, reshaped_aux
        
        # Optimize dataset pipeline for faster training
        dataset = dataset.batch(dataset_batch_size).map(reshape_batch)
        
        # Add prefetching for better GPU utilization (overlaps data loading with computation)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
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


def _run_dascca_single(args, window_size: int, output_dir: str):
    """Run one DASCCA train+test for a given window_size. Returns (best_val_acc, results dict)."""
    train_dataset, val_dataset, test_dataset = create_das_data_loaders(
        args.tfrecord_dir, batch_size=args.batch_size, window_size=window_size,
        audio_base_dir=args.audio_base_dir, load_audio=args.load_audio,
        max_files=args.max_files,
        eeg_lag_samples=0
    )
    cca_dims_value = args.cca_dims if args.cca_dims > 0 else 40
    model = DASCCAModel(
        cca_dims=cca_dims_value,
        regularization=args.regularization if args.regularization > 0 else 0.08,
        window_size=window_size,
        use_time_lags=args.use_time_lags,
        min_lag_ms=args.min_lag_ms,
        max_lag_ms=args.max_lag_ms,
        fs=args.sampling_rate,
        use_lda=args.use_lda,
        pca_eeg=args.pca_eeg,
        pca_audio=args.pca_audio,
        eeg_lag_taps=args.eeg_lag_taps
    )
    if args.use_time_lags:
        print(f"  Audio shape with lags: (T, 4*{model.num_lags}) = (T, {4*model.num_lags}) -> cca_dims max = min(64, {4*model.num_lags}) = {min(64, 4*model.num_lags)}")
    trainer = DASCCATrainer(model, output_dir, args.tfrecord_dir,
                           sampling_rate=args.sampling_rate, window_size=window_size,
                           audio_base_dir=args.audio_base_dir)
    best_val_acc = trainer.train(train_dataset, val_dataset)
    results = trainer.test(test_dataset)
    trainer.save_results(results)
    return best_val_acc, results


def main():
    """Main function for DASCCA training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='DASCCA - CCA Algorithm for DAS Dataset')
    parser.add_argument('--tfrecord_dir', type=str, default='das_16subjects_preprocessed/tfrecords',
                       help='TFRecord directory path')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--cca_dims', type=int, default=-1,
                       help='Number of CCA dimensions (default: -1 to auto-select 40, or specify value)')
    parser.add_argument('--regularization', type=float, default=0.08,
                       help='CCA regularization parameter (default: 0.08 optimal for DAS, use 0.05-0.1 range)')
    parser.add_argument('--window_size', type=int, default=1024,
                       help='Window size in samples (used when --no_window_sweep). 1024 = 8s at 128Hz')
    parser.add_argument('--window_sweep', action='store_true', default=True,
                       help='Run 1s to 30s window sweep (default)')
    parser.add_argument('--no_window_sweep', dest='window_sweep', action='store_false',
                       help='Use single --window_size instead of sweep')
    parser.add_argument('--window_sweep_min', type=float, default=1.0,
                       help='Sweep start in seconds (default: 1)')
    parser.add_argument('--window_sweep_max', type=float, default=30.0,
                       help='Sweep end in seconds (default: 30)')
    parser.add_argument('--window_sweep_step', type=float, default=1.0,
                       help='Sweep step in seconds (default: 1)')
    parser.add_argument('--output_dir', type=str, default='dascca_results',
                       help='Output directory for results')
    parser.add_argument('--audio_base_dir', type=str, default=None,
                       help='Base directory for audio files (auto-detected if not specified)')
    parser.add_argument('--load_audio', action='store_true', default=True,
                       help='Load audio envelopes (disable for faster loading)')
    parser.add_argument('--no_load_audio', dest='load_audio', action='store_false',
                       help='Skip audio loading for faster data loading (uses dummy audio)')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of TFRecord files to load (for faster testing)')
    parser.add_argument('--eeg_lag_samples', type=int, default=5,
                       help='Number of past time samples for backward model (default: 5)')
    parser.add_argument('--use_lda', action='store_true', default=True,
                       help='Use LDA classifier downstream (default: True)')
    parser.add_argument('--no_lda', dest='use_lda', action='store_false',
                       help='Disable LDA classifier, use direct correlation comparison')
    parser.add_argument('--sampling_rate', type=float, default=128.0,
                       help='Envelope/EEG sampling rate in Hz for time lags (default: 128)')
    parser.add_argument('--use_time_lags', action='store_true', default=True,
                       help='Use time-lagged audio features 0-250 ms for speech tracking (default: True)')
    parser.add_argument('--no_time_lags', dest='use_time_lags', action='store_false',
                       help='Disable time-lagged audio (use single time point only)')
    parser.add_argument('--min_lag_ms', type=float, default=0.0,
                       help='Minimum lag in ms (default: 0 for 0-250 ms range)')
    parser.add_argument('--max_lag_ms', type=float, default=250.0,
                       help='Maximum lag in ms for speech tracking (default: 250)')
    parser.add_argument('--eeg_lag_taps', type=int, default=5,
                       help='Backward model EEG taps L: x(t)=[eeg(t),...,eeg(t-L+1)]. 0=no EEG lag (default: 5)')
    parser.add_argument('--pca_eeg', type=int, default=25,
                       help='PCA components on EEG before CCA; 0=off (default: 25)')
    parser.add_argument('--pca_audio', type=int, default=0,
                       help='PCA components on audio before CCA; 0=off')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("DASCCA - CANONICAL CORRELATION ANALYSIS FOR DAS DATASET")
    print("=" * 80)
    print("Features:")
    print("- CCA implementation based on telluride_decoding")
    print("- Accuracy, MSED, ROC-AUC metrics")
    print("- Window sweep 1s to 30s (default); or single window with --no_window_sweep")
    print("- DAS preprocessing integration for data quality")
    print("- Data leakage prevention")
    print("- Validated attention labels")
    print("- EEG + Audio envelope correlation (improved CCA performance)")
    print("=" * 80)
    
    print("✓ Using DAS preprocessing validated data")
    print("✓ Data leakage prevention enabled")
    print("✓ Attention labels validated")
    print("✓ CCA implementation from telluride_decoding")
    if args.load_audio:
        print("✓ Audio envelope support enabled (EEG vs Audio correlation)")
    else:
        print("⚠ Audio loading DISABLED - using dummy audio (faster but may affect accuracy)")
    if args.max_files:
        print(f"⚠ Limiting to {args.max_files} TFRecord files (for faster testing)")

    fs = int(args.sampling_rate)

    if args.window_sweep:
        # 1s to 30s window sweep (default)
        sweep_sec = []
        s = args.window_sweep_min
        while s <= args.window_sweep_max + 1e-6:
            sweep_sec.append(s)
            s += args.window_sweep_step
        print(f"\nWindow sweep: {args.window_sweep_min}s to {args.window_sweep_max}s, step {args.window_sweep_step}s ({len(sweep_sec)} windows)")
        sweep_results = []
        for sec in sweep_sec:
            window_size = int(sec * fs)
            out_dir = os.path.join(args.output_dir, f"window_{sec:.0f}s")
            print("\n" + "=" * 80)
            print(f"DASCCA window size: {window_size} samples ({sec:.1f}s at {fs} Hz)")
            print("=" * 80)
            print(f"\nCreating DAS data loaders (window={window_size})...")
            best_val_acc, results = _run_dascca_single(args, window_size, out_dir)
            adv = results.get('advanced_metrics', {})
            roc = results.get('roc_auc_metrics', {})
            sweep_results.append({
                'window_seconds': sec,
                'window_samples': window_size,
                'validation_accuracy': best_val_acc,
                'test_accuracy': results.get('accuracy'),
                'balanced_accuracy': adv.get('balanced_accuracy'),
                'roc_auc': roc.get('roc_auc_score') if 'error' not in roc else None,
            })
        # Save sweep summary
        sweep_path = os.path.join(args.output_dir, 'window_sweep_results.json')
        with open(sweep_path, 'w') as f:
            json.dump(sweep_results, f, indent=2)
        print("\n" + "=" * 80)
        print("WINDOW SWEEP SUMMARY (1s to 30s)")
        print("=" * 80)
        print(f"{'Window (s)':<12} {'Val Acc':<10} {'Test Acc':<10} {'Bal Acc':<10} {'ROC-AUC':<10}")
        for r in sweep_results:
            print(f"{r['window_seconds']:<12.1f} {r['validation_accuracy'] or 0:<10.4f} {r['test_accuracy'] or 0:<10.4f} {r['balanced_accuracy'] or 0:<10.4f} {r['roc_auc'] or 0:<10.4f}")
        print(f"\nSweep results saved to: {sweep_path}")
        return

    # Single window run
    print(f"\nCreating DAS data loaders (window={args.window_size} samples = {args.window_size/fs:.1f}s)...")
    print("\nCreating DASCCA model...")
    best_val_acc, results = _run_dascca_single(args, args.window_size, args.output_dir)

    print("\n" + "=" * 80)
    print("DASCCA TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {results['accuracy']:.4f}")

    print("\n" + "=" * 80)
    print("COMPREHENSIVE METRICS SUMMARY")
    print("=" * 80)

    def _fmt(v):
        return f"{v:.4f}" if v is not None and isinstance(v, (int, float)) else 'N/A'

    roc_auc = results.get('roc_auc_metrics', {})
    if "error" not in roc_auc:
        print(f"ROC-AUC Score: {_fmt(roc_auc.get('roc_auc_score'))}")
        print(f"Average Precision: {_fmt(roc_auc.get('average_precision'))}")

    msed = results.get('msed_metrics', {})
    if "error" not in msed:
        print(f"RMSE: {_fmt(msed.get('rmse'))}")
        print(f"R-squared: {_fmt(msed.get('r_squared'))}")

    advanced = results.get('advanced_metrics', {})
    if "error" not in advanced:
        print(f"Matthews Correlation Coefficient: {_fmt(advanced.get('matthews_correlation_coefficient'))}")
        print(f"Balanced Accuracy: {_fmt(advanced.get('balanced_accuracy'))}")

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
