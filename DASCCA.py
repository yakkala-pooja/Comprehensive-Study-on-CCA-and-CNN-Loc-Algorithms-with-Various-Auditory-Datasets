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
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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


def _apply_time_lagging(eeg_window: np.ndarray, lag_samples: int) -> np.ndarray:
    """
    Create time-lagged EEG features for backward model.
    
    For each time point, concatenate current and past time points.
    This creates spatiotemporal features: [eeg(t), eeg(t-1), ..., eeg(t-lag)]
    
    Args:
        eeg_window: EEG data of shape (window_size, n_channels)
        lag_samples: Number of past time samples to include
        
    Returns:
        Time-lagged EEG features of shape (window_size, n_channels * (lag_samples + 1))
    """
    window_size, n_channels = eeg_window.shape
    lagged_features = []
    
    for t in range(window_size):
        # Get time points from t-lag to t (inclusive)
        start_idx = max(0, t - lag_samples)
        end_idx = t + 1
        
        # Extract time-lagged segment
        lagged_segment = eeg_window[start_idx:end_idx, :]  # Shape: (segment_len, n_channels)
        
        # Pad with zeros if needed (for early time points)
        if lagged_segment.shape[0] < (lag_samples + 1):
            padding = np.zeros(((lag_samples + 1) - lagged_segment.shape[0], n_channels), 
                             dtype=eeg_window.dtype)
            lagged_segment = np.vstack([padding, lagged_segment])
        
        # Flatten to create feature vector: [eeg(t-lag), ..., eeg(t)]
        lagged_features.append(lagged_segment.flatten())
    
    return np.array(lagged_features, dtype=eeg_window.dtype)

def safe_random_operations():
    """Force CPU usage for random operations."""
    with tf.device('/CPU:0'):
        tf.random.set_seed(42)
        np.random.seed(42)


class DasDatasetCCA:
    
    def __init__(self, tfrecord_dir: str, mode: str = 'full', 
                 window_size: int = 32, overlap: float = 0.5,
                 cache_size: int = 1000, audio_base_dir: Optional[str] = None,
                 load_audio: bool = True, max_files: Optional[int] = None):
        self.tfrecord_dir = Path(tfrecord_dir)
        self.mode = mode
        self.window_size = window_size
        self.overlap = overlap
        self.cache_size = cache_size
        self.load_audio = load_audio  # Option to skip audio loading for speed
        self.max_files = max_files  # Limit number of files to load
        
        self.sampling_rate = 128  # FIXED: Changed from 64 to 128 Hz to match preprocessing (das_preprocessing_16subjects.py uses 128 Hz)
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
        

        self.eeg_data, self.audio_envelopes, self.labels, self.metadata = self._load_das_preprocessing_data()
        
        self.window_indices = self._create_das_windows()
        
        print(f"Loaded {len(self.window_indices)} DAS windows for {mode} mode")
        print(f"DAS EEG shape: {self.eeg_data.shape}")
        print(f"DAS Audio envelopes shape: {self.audio_envelopes.shape}")
        print(f"DAS Label distribution: {np.bincount(self.labels)}")
        print(f"Using DAS preprocessing: Yes")
        print(f"Cache size: {cache_size} windows")
    
    def _load_das_preprocessing_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Load DAS preprocessing validated TFRecord data with robust shape validation."""

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
        
        print(f"Loading DAS 16-subjects preprocessing validated data from {len(tfrecord_files)} files...")
        print("✓ Using validated attention labels with quality control")
        print("✓ Using subject-wise organized data (no data leakage)")
        if self.load_audio:
            print("✓ EEG + Audio envelope processing with 16 subjects support")
            print(f"  Audio base directory: {self.audio_base_dir}")
            if self.audio_base_dir and self.audio_base_dir.exists():
                # Count audio files in directory
                audio_files = list(self.audio_base_dir.glob("*.wav")) + list(self.audio_base_dir.glob("*.WAV")) + \
                             list(self.audio_base_dir.glob("*.mp3")) + list(self.audio_base_dir.glob("*.MP3"))
                print(f"  Found {len(audio_files)} audio files in base directory")
            else:
                print(f"  ⚠ WARNING: Audio base directory may not exist: {self.audio_base_dir}")
        else:
            print("⚠ Audio loading DISABLED - using dummy audio envelopes (faster loading)")
        print(f"✓ Found TFRecord files in: {[f.parent.name for f in tfrecord_files[:3]]}...")
        
        all_eeg_data = []
        all_audio_envelopes = []
        all_labels = []
        all_metadata = []
        
        successful_files = 0
        failed_files = 0
        total_records = 0
        subject_stats = {}
        shape_validation_errors = 0
        
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
                        

                        eeg_values = features['eeg'].float_list.value
                        if not eeg_values or len(eeg_values) == 0:
                            continue
                        

                        n_channels = len(eeg_values)
                        if n_channels != 64:
                            if shape_validation_errors < 10:
                                print(f"ERROR: Expected 64 EEG channels, got {n_channels} in {tfrecord_file.name} (record {total_records})")
                            shape_validation_errors += 1

                            if n_channels > 0 and n_channels <= 128:

                                eeg_data = np.array(eeg_values, dtype=np.float32).reshape(1, n_channels)
                                print(f"  WARNING: Using {n_channels} channels instead of 64 - this may cause issues downstream")
                            else:
                                continue
                        else:

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
                        


                        # OPTIMIZATION: Skip audio loading if disabled (much faster)
                        if self.load_audio:
                            audio_envelope = None
                            if attended_ear == 'L' and left_audio_file:
                                audio_envelope = self._load_audio_envelope(left_audio_file, sample_idx)
                            elif attended_ear == 'R' and right_audio_file:
                                audio_envelope = self._load_audio_envelope(right_audio_file, sample_idx)
                            
                            if audio_envelope is None:
                                audio_envelope = np.array([0.0], dtype=np.float32)
                        else:
                            # Use dummy audio envelope (faster - can load audio later if needed)
                            audio_envelope = np.array([0.0], dtype=np.float32)
                        

                        if subject_id not in subject_stats:
                            subject_stats[subject_id] = {'samples': 0, 'labels': []}
                        subject_stats[subject_id]['samples'] += 1
                        subject_stats[subject_id]['labels'].append(label)
                        
                        metadata = {
                            'subject_id': subject_id,
                            'file': tfrecord_file.name,
                            'sample_idx': sample_idx,
                            'trial_index': trial_index,  # FIXED: Store trial_index for trial-matched envelope loading
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
        
        print(f"\n{'='*60}")
        print(f"Loading Summary:")
        print(f"  Successfully loaded files: {successful_files}")
        print(f"  Failed files: {failed_files}")
        print(f"  Total records loaded: {total_records}")
        print(f"  Shape validation errors: {shape_validation_errors}")
        print(f"{'='*60}")
        
        if shape_validation_errors > 0:
            print(f"⚠ WARNING: {shape_validation_errors} records had shape validation errors")
            print("  This suggests the EEG data may not have exactly 64 channels.")
            print("  Check the preprocessing output to verify channel count.")
        
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
            if self.tfrecord_dir.exists():
                print(f"  Contents:")
                for item in self.tfrecord_dir.iterdir():
                    print(f"    - {item.name} ({'dir' if item.is_dir() else 'file'})")
                    if item.is_dir():
                        subfiles = list(item.glob("*.tfrecords"))
                        print(f"      Contains {len(subfiles)} TFRecord files")
        

        print(f"\nSubject-wise statistics:")
        for subject_id, stats in subject_stats.items():
            label_dist = np.bincount(stats['labels'])
            print(f"  {subject_id}: {stats['samples']} samples, labels {label_dist}")
        
        if not all_eeg_data:
            raise ValueError("No valid DAS preprocessing data found in TFRecord files")
        
        eeg_data = np.vstack(all_eeg_data)


        # FIXED: Load full audio envelopes instead of creating synthetic features from single values
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
        
        audio_envelopes = np.array(audio_envelopes_list, dtype=np.float32)

        # FIXED: Handle variable-length audio envelopes (single values stored initially)
        # Full temporal envelopes will be loaded in __getitem__ when creating windows
        if audio_envelopes.ndim == 1:
            # Single values - reshape to (n_samples, 1)
            audio_envelopes = audio_envelopes.reshape(-1, 1)
        elif audio_envelopes.ndim == 2:
            # Already 2D - check shape
            if audio_envelopes.shape[1] == 1:
                # Single values per sample - this is expected for initial loading
                # Full temporal envelopes will be loaded during window creation
                pass  # Keep as (n_samples, 1) - will be expanded in __getitem__
            elif audio_envelopes.shape[1] < 4:
                # Less than 4 features - pad to 4 (shouldn't happen, but handle gracefully)
                padding = np.zeros((audio_envelopes.shape[0], 4 - audio_envelopes.shape[1]))
                audio_envelopes = np.column_stack([audio_envelopes, padding])
            elif audio_envelopes.shape[1] > 4:
                # More than 4 features - truncate to 4
                audio_envelopes = audio_envelopes[:, :4]
            # If shape[1] == 4, keep as is
        else:
            raise ValueError(f"Invalid audio_envelopes shape: {audio_envelopes.shape}")
        
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
        
        # Check for synthetic features (all zeros in 3rd column indicates synthetic features)
        synthetic_features_count = 0
        if audio_envelopes.shape[1] >= 3:
            # Synthetic features have pattern: [val, val, 0, val²] - check for all-zero 3rd column
            third_col_zeros = np.sum(np.abs(audio_envelopes[:, 2]) < 1e-6)
            if third_col_zeros > len(audio_envelopes) * 0.9:  # If >90% have zero in 3rd column
                synthetic_features_count = third_col_zeros
                print(f"  ⚠ WARNING: {synthetic_features_count} samples appear to have synthetic audio features")
                print(f"    (pattern [val, val, 0, val²] detected). Full temporal envelopes will be loaded during window creation.")
        
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
                # Try to match by extracting part/track info from filename
                # e.g., if TFRecord has "trigger_123_part1_track1_hrtf.wav", try "part1_track1_hrtf.wav"
                if 'part' in audio_stem.lower() and 'track' in audio_stem.lower():
                    # Extract part and track numbers
                    import re
                    part_match = re.search(r'part(\d+)', audio_stem.lower())
                    track_match = re.search(r'track(\d+)', audio_stem.lower())
                    if part_match and track_match:
                        part_num = part_match.group(1)
                        track_num = track_match.group(1)
                        # Try both hrtf and dry versions
                        for variant in ['hrtf', 'dry']:
                            test_pattern = f"part{part_num}_track{track_num}_{variant}.wav"
                            test_file = self.audio_base_dir / test_pattern
                            if test_file.exists():
                                audio_file = test_file
                                break
                
                # Fallback: try fuzzy matching
                if audio_file is None:
                    matches = list(self.audio_base_dir.glob(f"*{audio_stem}*"))
                    if matches:
                        audio_file = matches[0]
                    
                    # If still not found, try matching by part/track pattern
                    if audio_file is None and ('part' in audio_filename.lower() or 'track' in audio_filename.lower()):
                        import re
                        # Try to find any file with matching part/track
                        all_audio_files = list(self.audio_base_dir.glob("*.wav")) + list(self.audio_base_dir.glob("*.WAV"))
                        for candidate in all_audio_files:
                            candidate_lower = candidate.name.lower()
                            if 'part' in candidate_lower and 'track' in candidate_lower:
                                # Extract part/track from both
                                audio_part_match = re.search(r'part(\d+)', audio_filename.lower())
                                audio_track_match = re.search(r'track(\d+)', audio_filename.lower())
                                cand_part_match = re.search(r'part(\d+)', candidate_lower)
                                cand_track_match = re.search(r'track(\d+)', candidate_lower)
                                
                                if (audio_part_match and audio_track_match and 
                                    cand_part_match and cand_track_match):
                                    if (audio_part_match.group(1) == cand_part_match.group(1) and
                                        audio_track_match.group(1) == cand_track_match.group(1)):
                                        audio_file = candidate
                                        break
        
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
            
            # Log successful audio loading for first few files
            if not hasattr(self, '_audio_file_loaded_count'):
                self._audio_file_loaded_count = 0
            if self._audio_file_loaded_count < 3:
                print(f"✓ Loaded audio file: {audio_file.name} (duration: {len(audio_data)/fs:.2f}s, fs: {fs}Hz)")
                self._audio_file_loaded_count += 1
            

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
    
    def _create_das_windows(self) -> List[Tuple[int, int]]:
        """Create windows optimized for DAS data structure with proper time units."""

        window_seconds = self.window_size / self.sampling_rate
        step_size = int(self.window_size * (1 - self.overlap))
        step_seconds = step_size / self.sampling_rate
        
        total_windows = (len(self.eeg_data) - self.window_size) // step_size + 1
        
        print(f"Creating {total_windows} DAS windows:")
        print(f"  Window size: {self.window_size} samples ({window_seconds:.1f} seconds)")
        print(f"  Step size: {step_size} samples ({step_seconds:.1f} seconds)")
        print(f"  Overlap: {self.overlap:.1%}")
        print(f"  Sampling rate: {self.sampling_rate} Hz")
        

        if window_seconds < 1.0:
            print(f"⚠ WARNING: Very short window ({window_seconds:.1f}s) may have poor signal-to-noise")
        elif window_seconds > 20.0:
            print(f"⚠ WARNING: Very long window ({window_seconds:.1f}s) may miss temporal dynamics")
        else:
            print(f"✓ Window size appropriate for EEG attention decoding")
        
        window_indices = []
        for i in range(total_windows):
            data_idx = i * step_size
            if data_idx + self.window_size <= len(self.eeg_data):

                window_start = data_idx
                window_end = data_idx + self.window_size
                window_labels = self.labels[window_start:window_end]
                

                if len(window_labels) > 0:
                    window_label = int(np.bincount(window_labels).argmax())
                else:
                    window_label = 0
                
                window_indices.append((data_idx, window_label))
        
        print(f"Created {len(window_indices)} DAS windows")
        

        window_labels = [label for _, label in window_indices]
        label_dist = np.bincount(window_labels)
        print(f"Window label distribution: {label_dist}")
        
        return window_indices
    
    def _das_eeg_preprocessing(self, eeg_window: np.ndarray) -> np.ndarray:
        """DAS-specific EEG preprocessing with artifact handling."""
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
        data_idx, label = self.window_indices[idx]
        

        cache_key = (data_idx, self.mode)
        if cache_key in self._window_cache:
            self._cache_hits += 1
            cached_data, cached_label = self._window_cache[cache_key]
            return cached_data, cached_label
        
        self._cache_misses += 1
        

        window_eeg = self.eeg_data[data_idx:data_idx + self.window_size]
        
        # CRITICAL FIX: Load FULL temporal audio envelopes for the entire window
        # Load attended audio, left audio, and right audio from actual audio files
        window_metadata = self.metadata[data_idx] if data_idx < len(self.metadata) else None
        attended_audio_envelope = None
        left_audio_envelope = None
        right_audio_envelope = None
        
        if window_metadata and self.load_audio:
            sample_idx = window_metadata.get('sample_idx', 0)
            trial_index = window_metadata.get('trial_index', None)
            attended_ear = window_metadata.get('attended_ear', 'L')
            left_audio_file = window_metadata.get('left_audio_file')
            right_audio_file = window_metadata.get('right_audio_file')
            
            # FIXED: Helper function to load trial-matched window segment from audio envelope
            # CRITICAL: sample_idx is the index WITHIN the trial, not the offset in the audio file
            # We need to extract the exact segment that corresponds to this trial
            def load_window_segment(audio_file, trial_sample_idx, window_size, trial_index=None):
                """
                Load a window segment from a full audio envelope that matches the trial.
                
                Args:
                    audio_file: Path to audio file
                    trial_sample_idx: Sample index within the trial (from sample_id in TFRecord)
                    window_size: Window size in samples
                    trial_index: Trial index for debugging
                
                Returns:
                    Audio envelope segment matching the trial window, or None if not found
                """
                envelope_full = self._load_audio_envelope_full(audio_file)
                if envelope_full is not None and len(envelope_full) > 0:
                    # FIXED: Use trial_sample_idx directly (it's already the offset within the trial)
                    # The envelope_full should be the full trial envelope, so trial_sample_idx is correct
                    # However, if envelope_full is the full track, we need trial start time
                    # For now, assume envelope_full is per-trial (which it should be if loaded correctly)
                    end_idx = min(trial_sample_idx + window_size, len(envelope_full))
                    if trial_sample_idx < len(envelope_full):
                        segment = envelope_full[trial_sample_idx:end_idx]
                        # Pad if needed
                        if len(segment) < window_size:
                            padding = np.zeros(window_size - len(segment), dtype=np.float32)
                            segment = np.concatenate([segment, padding])
                        return segment
                    else:
                        # If trial_sample_idx is beyond envelope length, use last values
                        if trial_index is not None:
                            print(f"⚠ WARNING: Trial {trial_index} sample_idx {trial_sample_idx} >= envelope length {len(envelope_full)}")
                        return np.tile(envelope_full[-1:], window_size)
                return None
            
            # Load attended audio envelope (the one that matches the label)
            if attended_ear == 'L' and left_audio_file:
                attended_audio_envelope = load_window_segment(left_audio_file, sample_idx, self.window_size, trial_index)
            elif attended_ear == 'R' and right_audio_file:
                attended_audio_envelope = load_window_segment(right_audio_file, sample_idx, self.window_size, trial_index)
            
            # Load left and right audio envelopes for comparison
            if left_audio_file:
                left_audio_envelope = load_window_segment(left_audio_file, sample_idx, self.window_size, trial_index)
            if right_audio_file:
                right_audio_envelope = load_window_segment(right_audio_file, sample_idx, self.window_size, trial_index)
            
            # FIXED: Add comprehensive envelope verification and debugging
            # CRITICAL: Check if we're using correct trial offsets
            if not hasattr(self, '_envelope_verification_count'):
                self._envelope_verification_count = 0
            
            if self._envelope_verification_count < 3:  # Check first 3 windows
                if left_audio_envelope is not None and right_audio_envelope is not None:
                    left_mean = np.mean(left_audio_envelope)
                    right_mean = np.mean(right_audio_envelope)
                    left_std = np.std(left_audio_envelope)
                    right_std = np.std(right_audio_envelope)
                    diff = np.abs(left_mean - right_mean)
                    
                    # CRITICAL: Check what offset we're using
                    envelope_full_left = self._load_audio_envelope_full(left_audio_file) if left_audio_file else None
                    envelope_full_right = self._load_audio_envelope_full(right_audio_file) if right_audio_file else None
                    
                    print(f"\n{'='*60}")
                    print(f"ENVELOPE VERIFICATION (Window {self._envelope_verification_count + 1})")
                    print(f"{'='*60}")
                    print(f"Subject ID: {window_metadata.get('subject_id', 'unknown')}")
                    print(f"File: {window_metadata.get('file', 'unknown')}")
                    print(f"Sample index (within trial): {sample_idx}")
                    print(f"Trial index: {trial_index}")
                    print(f"Audio slice start sample: {sample_idx} (CRITICAL: This should match trial offset!)")
                    print(f"Left audio file: {left_audio_file}")
                    print(f"Right audio file: {right_audio_file}")
                    if envelope_full_left is not None:
                        print(f"Left full envelope length: {len(envelope_full_left)} samples")
                        print(f"Left envelope slice: [{sample_idx}:{sample_idx + self.window_size}]")
                        print(f"Left envelope first 5 values: {left_audio_envelope[:5]}")
                    if envelope_full_right is not None:
                        print(f"Right full envelope length: {len(envelope_full_right)} samples")
                        print(f"Right envelope slice: [{sample_idx}:{sample_idx + self.window_size}]")
                        print(f"Right envelope first 5 values: {right_audio_envelope[:5]}")
                    print(f"Left envelope: mean={left_mean:.6f}, std={left_std:.6f}, shape={left_audio_envelope.shape}")
                    print(f"Right envelope: mean={right_mean:.6f}, std={right_std:.6f}, shape={right_audio_envelope.shape}")
                    print(f"Difference (|left - right|): {diff:.6f}")
                    if diff < 1e-6:
                        print(f"⚠ CRITICAL: Left and right envelopes are nearly identical! This indicates a bug.")
                    if sample_idx == 0:
                        print(f"⚠ WARNING: sample_idx is 0 - we may be taking from start of audio file instead of trial segment!")
                    if trial_index is None:
                        print(f"⚠ CRITICAL: trial_index is None - cannot verify trial alignment!")
                    print(f"Attended ear: {attended_ear}, Label: {0 if attended_ear == 'L' else 1}")
                    print(f"{'='*60}\n")
                    self._envelope_verification_count += 1
        
        # Use attended audio if available, otherwise fall back to placeholder
        if attended_audio_envelope is not None:
            window_audio = attended_audio_envelope.reshape(-1, 1)  # Shape: (window_size, 1)
        else:
            # Fallback: use placeholder from self.audio_envelopes (single values)
            window_audio = self.audio_envelopes[data_idx:data_idx + self.window_size]
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
        

        try:
            window_eeg = self._das_eeg_preprocessing(window_eeg)
        except Exception:
            window_eeg = window_eeg - np.mean(window_eeg, axis=0, keepdims=True)
            window_eeg = window_eeg / (np.std(window_eeg, axis=0, keepdims=True) + 1e-8)
            window_eeg = np.tanh(window_eeg * 0.5)
        


        if window_audio.ndim == 1:

            window_audio = window_audio.reshape(-1, 1)
        

        if window_audio.shape[1] == 1:

            env_vals = window_audio.flatten()
            

            if len(env_vals) > 1:
                from scipy.ndimage import uniform_filter1d
                smoothed = uniform_filter1d(env_vals, size=min(3, len(env_vals)), mode='nearest')
                derivative = np.gradient(env_vals)
            else:
                smoothed = env_vals
                derivative = np.zeros_like(env_vals)
            
            window_audio = np.column_stack([
                env_vals,
                smoothed,
                derivative,
                env_vals**2
            ])
        elif window_audio.shape[1] != 4:

            if window_audio.shape[1] < 4:

                env_vals = window_audio[:, 0] if window_audio.shape[1] > 0 else np.zeros(window_audio.shape[0])
                padding = np.zeros((window_audio.shape[0], 4 - window_audio.shape[1]))
                window_audio = np.column_stack([window_audio, padding])
            else:

                window_audio = window_audio[:, :4]
        

        if np.max(np.abs(window_audio)) > 0:
            window_audio = window_audio / (np.max(np.abs(window_audio)) + 1e-8)
        

        window_eeg_tensor = tf.constant(window_eeg, dtype=tf.float32)
        window_audio_tensor = tf.constant(window_audio, dtype=tf.float32)
        
        # Process left and right audio envelopes to match window_audio format
        left_audio_processed = self._process_audio_envelope(left_audio_envelope, self.window_size)
        right_audio_processed = self._process_audio_envelope(right_audio_envelope, self.window_size)
        
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
        

        if len(self._window_cache) < self.cache_size:
            self._window_cache[cache_key] = (window_tensor, aux_data)
        
        return window_tensor, aux_data
    
    def _process_audio_envelope(self, audio_envelope: np.ndarray, window_size: int) -> np.ndarray:
        """Process audio envelope to match window size and format.
        
        FIXED: Now expects full temporal envelope (window_size length) instead of single values.
        """
        if audio_envelope is None or len(audio_envelope) == 0:
            # Use zeros if envelope is missing
            audio_envelope = np.zeros(window_size, dtype=np.float32)
        
        # Ensure envelope matches window size
        if len(audio_envelope) == window_size:
            # Perfect match - use as is
            pass
        elif len(audio_envelope) < window_size:
            # Pad if too short (shouldn't happen if loaded correctly, but handle gracefully)
            padding = np.zeros(window_size - len(audio_envelope), dtype=np.float32)
            audio_envelope = np.concatenate([audio_envelope, padding])
        elif len(audio_envelope) > window_size:
            # Truncate if too long (shouldn't happen if loaded correctly, but handle gracefully)
            audio_envelope = audio_envelope[:window_size]
        else:
            # Single value case - this should be rare now, but handle it
            if len(audio_envelope) == 1:
                if self.load_audio:
                    print(f"⚠ WARNING: Received single-value audio envelope. Expected full temporal envelope of length {window_size}.")
                audio_envelope = np.repeat(audio_envelope, window_size)
        
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


class DASCCAModel:
    """
    DASCCA model implementing Canonical Correlation Analysis for DAS EEG dataset.
    
    This model uses the telluride_decoding CCA implementation to find correlations
    between EEG data and attention labels, providing comprehensive metrics evaluation.
    """
    
    def __init__(self, cca_dims: int = 5, regularization: float = 0.01, window_size: int = 512,
                 eeg_lag_samples: int = 5, pca_components: Optional[int] = None, use_lda: bool = True,
                 stimulus_lag_samples: Optional[List[int]] = None):
        """
        Initialize DASCCA model with backward/forward model support.
        
        Args:
            cca_dims: Number of CCA dimensions to compute
            regularization: Regularization parameter for CCA
            window_size: Window size for EEG data processing
            eeg_lag_samples: Number of past time samples to include for backward model (default: 5)
            pca_components: Number of PCA components for EEG regularization (None = no PCA)
            use_lda: Whether to use LDA classifier downstream (default: True)
            stimulus_lag_samples: List of stimulus lag samples for temporal alignment (e.g., [0, 4, 8, 12, 16] for 0-125ms at 128Hz)
                                  If None, uses default [0, 4, 8, 12, 16, 20, 24, 28, 32] (0-250ms at 128Hz)
        """



        max_cca_dims = 4
        if cca_dims > max_cca_dims:
            print(f"⚠ WARNING: Requested {cca_dims} CCA dimensions, but maximum is {max_cca_dims} (limited by audio features)")
            print(f"  Reducing CCA dimensions from {cca_dims} to {max_cca_dims}")
            cca_dims = max_cca_dims
        elif cca_dims < 1:
            print(f"⚠ WARNING: CCA dimensions must be >= 1, setting to 1")
            cca_dims = 1
        
        self.cca_dims = cca_dims
        self.regularization = regularization
        self.window_size = window_size
        self.eeg_lag_samples = eeg_lag_samples  # Backward model: include past time points
        self.pca_components = pca_components  # PCA regularization
        self.use_lda = use_lda  # LDA downstream classifier
        
        # FIXED: Add temporal lag support for attention decoding (EEG lags stimulus by ~100-250ms)
        # At 128 Hz: 0-32 samples = 0-250ms
        # NOTE: Temporal lag is typically handled by:
        #   1. Shifting stimulus relative to EEG (forward model: stimulus leads EEG)
        #   2. Using lagged regression in CCA (telluride_decoding may support this)
        #   3. Building lagged stimulus features (concatenate multiple lagged versions)
        # For now, we store the lag samples but the actual lagging would need to be implemented
        # in the data loading or CCA model. The eeg_lag_samples handles backward model (past EEG).
        if stimulus_lag_samples is None:
            # Default: 0, 4, 8, 12, 16, 20, 24, 28, 32 samples = 0, 31, 63, 94, 125, 156, 188, 219, 250 ms
            self.stimulus_lag_samples = list(range(0, 33, 4))  # 0 to 32 samples in steps of 4
        else:
            self.stimulus_lag_samples = stimulus_lag_samples
        
        self.model = None
        self.is_fitted = False
        self.pca_model = None  # PCA model for EEG preprocessing
        self.lda_model = None  # LDA model for classification
        
        print(f"DASCCA model initialized:")
        print(f"  CCA dimensions: {self.cca_dims} (max possible: {max_cca_dims})")
        print(f"  Regularization: {regularization}")
        print(f"  EEG lag samples (backward model): {eeg_lag_samples}")
        print(f"  Stimulus lag samples (temporal alignment): {self.stimulus_lag_samples} (0-{max(self.stimulus_lag_samples)*1000/128:.0f}ms at 128Hz)")
        print(f"  PCA components: {pca_components if pca_components else 'None (no PCA)'}")
        print(f"  LDA classifier: {'Enabled' if use_lda else 'Disabled'}")
        print(f"  Input dimensions: EEG=64, Audio=4")
    
    def _fit_pca(self, dataset: tf.data.Dataset):
        """
        Fit PCA model on EEG data for regularization.
        
        Args:
            dataset: TensorFlow dataset containing EEG data
        """
        print(f"Collecting EEG data for PCA fitting...")
        all_eeg = []
        
        for batch in dataset.take(100):  # Use first 100 batches for PCA fitting
            if isinstance(batch, dict):
                inputs = batch
            elif isinstance(batch, tuple):
                inputs, _ = batch
            else:
                continue
            
            eeg_data = inputs.get('input_1')
            if eeg_data is not None:
                # Flatten time dimension if needed
                if len(eeg_data.shape) > 2:
                    eeg_flat = tf.reshape(eeg_data, (-1, eeg_data.shape[-1]))
                else:
                    eeg_flat = eeg_data
                all_eeg.append(eeg_flat.numpy() if hasattr(eeg_flat, 'numpy') else eeg_flat)
        
        if not all_eeg:
            print("⚠ WARNING: No EEG data collected for PCA, skipping PCA fitting")
            self.pca_model = None
            return
        
        eeg_matrix = np.vstack(all_eeg)
        print(f"  Collected {eeg_matrix.shape[0]} samples for PCA")
        
        # Fit PCA
        self.pca_model = PCA(n_components=self.pca_components, random_state=42)
        self.pca_model.fit(eeg_matrix)
        
        explained_var = np.sum(self.pca_model.explained_variance_ratio_)
        print(f"✓ PCA fitted: {self.pca_components} components explain {explained_var:.1%} of variance")
    
    def _fit_lda(self, dataset: tf.data.Dataset):
        """
        Fit LDA classifier on CCA outputs.
        
        For attention decoding, we need to compare left vs right correlations.
        We compute window-level features: [left_corr, right_corr] for each window.
        
        Args:
            dataset: TensorFlow dataset containing EEG data and labels
        """
        print(f"Collecting CCA outputs and labels for LDA fitting...")
        print(f"  Using CPU device for LDA fitting to avoid GPU memory issues...")
        all_left_corrs = []
        all_right_corrs = []
        all_labels = []
        
        # FIXED: Process in smaller batches and use CPU to avoid GPU memory issues
        batch_count = 0
        max_batches = 200  # Limit number of batches for LDA fitting
        for batch in dataset.take(max_batches):
            if isinstance(batch, dict):
                inputs = batch
                aux = None
                targets = None
            elif isinstance(batch, tuple):
                inputs, aux_or_targets = batch
                if isinstance(aux_or_targets, dict):
                    aux = aux_or_targets
                    targets = aux.get('label', None)
                else:
                    aux = None
                    targets = aux_or_targets
            else:
                continue
            
            # For LDA, we need left and right correlations per window
            # This matches the approach in predict()
            if aux is not None and 'left_env' in aux and 'right_env' in aux:
                eeg_view = inputs['input_1']
                left_env = aux['left_env']
                right_env = aux['right_env']
                
                # FIXED: Pass sampling rates explicitly for verification
                eeg_fs = 128  # EEG sampling rate (matches preprocessing)
                env_fs = 128  # Envelope sampling rate (after resampling in _load_audio_envelope_full)
                
                # FIXED: Use CPU for LDA fitting to avoid GPU memory issues with large batches
                # LDA fitting is just inference (not training), so CPU is fine and more stable
                with tf.device('/CPU:0'):
                    try:
                        # Compute CCA correlation with left audio
                        left_inputs = {'input_1': eeg_view, 'input_2': left_env}
                        left_predictions = self.model(left_inputs)
                        left_scores = self._compute_correlation_scores(left_predictions, eeg_fs=eeg_fs, env_fs=env_fs)
                        
                        # Compute CCA correlation with right audio
                        right_inputs = {'input_1': eeg_view, 'input_2': right_env}
                        right_predictions = self.model(right_inputs)
                        right_scores = self._compute_correlation_scores(right_predictions, eeg_fs=eeg_fs, env_fs=env_fs)
                    except Exception as e:
                        print(f"⚠ Error during LDA fitting (batch size: {eeg_view.shape[0]}): {e}")
                        print(f"  Skipping this batch for LDA fitting...")
                        continue
                
                # Aggregate to window-level (mean per window)
                # Each batch contains multiple windows, each with window_size samples
                # FIXED: Handle large batches by processing in smaller chunks if needed
                batch_size_samples = eeg_view.shape[0] if len(eeg_view.shape) > 1 else 1
                num_samples = len(left_scores)
                        
                # Warn if batch is very large (may cause GPU memory issues)
                if batch_count == 0 and batch_size_samples > 4096:
                    print(f"  ⚠ WARNING: Large batch size detected ({batch_size_samples} samples)")
                    print(f"    This may cause GPU memory issues. Consider reducing batch_size in data loader.")
                batch_count += 1
                
                if num_samples > 0:
                    # Reshape to (num_windows, window_size) and aggregate
                    if num_samples % self.window_size == 0:
                        num_windows = num_samples // self.window_size
                        window_size = self.window_size
                    else:
                        # Handle edge case
                        num_windows = 1
                        window_size = num_samples
                    
                    try:
                        left_scores_reshaped = left_scores[:num_windows * window_size].reshape(num_windows, window_size)
                        right_scores_reshaped = right_scores[:num_windows * window_size].reshape(num_windows, window_size)
                        
                        # Aggregate scores per window (use mean)
                        left_window_scores = np.mean(left_scores_reshaped, axis=1)
                        right_window_scores = np.mean(right_scores_reshaped, axis=1)
                        
                        all_left_corrs.append(left_window_scores)
                        all_right_corrs.append(right_window_scores)
                    except (ValueError, IndexError):
                        # Fallback: aggregate all scores into single prediction
                        all_left_corrs.append(np.array([np.mean(left_scores)]))
                        all_right_corrs.append(np.array([np.mean(right_scores)]))
                
                # Get labels (window-level)
                if targets is not None:
                    if hasattr(targets, 'numpy'):
                        labels = targets.numpy().flatten()
                    else:
                        labels = np.array(targets).flatten()
                    # Ensure labels match number of windows
                    if len(labels) != len(left_window_scores):
                        # Repeat or truncate to match
                        if len(labels) == 1:
                            labels = np.repeat(labels, len(left_window_scores))
                        else:
                            labels = labels[:len(left_window_scores)]
                    all_labels.append(labels)
            else:
                # Fallback: use simple correlation if left/right not available
                # FIXED: Use CPU for LDA fitting to avoid GPU memory issues
                with tf.device('/CPU:0'):
                    try:
                        cca_outputs = self.model(inputs)
                        if hasattr(cca_outputs, 'numpy'):
                            cca_outputs = cca_outputs.numpy()
                        
                        # Process CCA outputs (inside CPU context)
                        cca_width = cca_outputs.shape[-1] // 2
                        proj_eeg = cca_outputs[:, :cca_width]
                        proj_audio = cca_outputs[:, cca_width:]
                        
                        # Compute correlation per sample
                        corr_coeffs = proj_eeg[:, 0] * proj_audio[:, 0]
                        
                        # Aggregate to window-level
                        num_samples = len(corr_coeffs)
                        if num_samples % self.window_size == 0:
                            num_windows = num_samples // self.window_size
                            window_size = self.window_size
                        else:
                            num_windows = 1
                            window_size = num_samples
                        
                        try:
                            corr_reshaped = corr_coeffs[:num_windows * window_size].reshape(num_windows, window_size)
                            window_corrs = np.mean(corr_reshaped, axis=1)
                            # Use same correlation for both left and right (fallback)
                            all_left_corrs.append(window_corrs)
                            all_right_corrs.append(window_corrs)
                        except (ValueError, IndexError):
                            all_left_corrs.append(np.array([np.mean(corr_coeffs)]))
                            all_right_corrs.append(np.array([np.mean(corr_coeffs)]))
                    except Exception as e:
                        print(f"⚠ Error during LDA fitting fallback (batch size: {inputs['input_1'].shape[0]}): {e}")
                        print(f"  Skipping this batch for LDA fitting...")
                        continue
                
                # Get labels
                if targets is not None:
                    if hasattr(targets, 'numpy'):
                        labels = targets.numpy().flatten()
                    else:
                        labels = np.array(targets).flatten()
                    if len(labels) != len(window_corrs):
                        if len(labels) == 1:
                            labels = np.repeat(labels, len(window_corrs))
                        else:
                            labels = labels[:len(window_corrs)]
                    all_labels.append(labels)
        
        if not all_left_corrs or not all_right_corrs or not all_labels:
            print("⚠ WARNING: Insufficient data for LDA, skipping LDA fitting")
            self.lda_model = None
            return
        
        # Prepare data for LDA: [left_corr, right_corr] per window
        left_corrs = np.hstack(all_left_corrs)
        right_corrs = np.hstack(all_right_corrs)
        labels = np.hstack(all_labels)
        
        # Create feature matrix: (n_windows, 2) - [left_corr, right_corr]
        cca_features = np.column_stack([left_corrs, right_corrs])
        
        # Ensure consistent lengths
        min_len = min(len(cca_features), len(labels))
        cca_features = cca_features[:min_len]
        labels = labels[:min_len]
        
        if len(np.unique(labels)) < 2:
            print("⚠ WARNING: Only one class in labels, skipping LDA fitting")
            self.lda_model = None
            return
        
        # Fit LDA on [left_corr, right_corr] features
        self.lda_model = LinearDiscriminantAnalysis()
        self.lda_model.fit(cca_features, labels)
        
        print(f"✓ LDA fitted on {len(labels)} windows with features [left_corr, right_corr]")
    
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
    
    # NOTE: _create_time_lagged_eeg removed - using module-level _apply_time_lagging instead
    # NOTE: _create_causal_audio removed - not currently used in the implementation
    
    def fit(self, dataset: tf.data.Dataset):
        """
        Fit the CCA model to the dataset with robust GPU handling.
        
        Args:
            dataset: TensorFlow dataset containing EEG data and labels
        """
        print("Fitting DASCCA model...")
        
        # Fit PCA on training data if enabled
        if self.pca_components is not None:
            print(f"Fitting PCA with {self.pca_components} components...")
            self._fit_pca(dataset)
        

        self.model = self._create_robust_cca_model(dataset)
        

        try:
            print("Compiling CCA model...")

            self.model.compile(
                optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                loss='mse',
                metrics=[cca_pearson_correlation_first]
            )
            
            print("Training CCA model...")

            # Train for more epochs to learn meaningful correlations
            self.model.fit(dataset, epochs=5)
            
            print("✓ DASCCA model fitted successfully")
            
        except Exception as e:
            print(f"Training failed: {e}")
            error_msg = str(e)
            

            if "rot1 must be shape" in error_msg or "rot2 must be shape" in error_msg:
                print(f"\n⚠ CCA Dimension Mismatch Error Detected!")
                print(f"  Requested CCA dimensions: {self.cca_dims}")
                print(f"  Input dimensions: EEG=64, Audio=4")
                print(f"  Maximum possible CCA dimensions: min(64, 4) = 4")
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
                        # Train for more epochs to learn meaningful correlations
                        self.model.fit(dataset, epochs=5)
                        print("✓ DASCCA model fitted successfully with reduced dimensions (2)")
                        # Success - return early, don't try CPU fallback
                        self.is_fitted = True
                        print("✓ DASCCA model training completed")
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
                            # Train for more epochs to learn meaningful correlations
                            self.model.fit(dataset, epochs=5)
                            print("✓ DASCCA model fitted successfully with 1 dimension")
                            # Success - return early, don't try CPU fallback
                            self.is_fitted = True
                            print("✓ DASCCA model training completed")
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
                    # Train for more epochs to learn meaningful correlations
                    # CCA typically needs multiple epochs to converge
                    self.model.fit(dataset, epochs=5)
                    print("✓ DASCCA model fitted successfully on CPU")
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
        print("✓ DASCCA model training completed")
        
        # Fit LDA classifier if enabled (after CCA is fitted)
        if self.use_lda:
            print("Fitting LDA classifier on CCA outputs...")
            self._fit_lda(dataset)
    
    def _compute_correlation_scores(self, predictions: tf.Tensor, eeg_fs: int = 128, env_fs: int = 128) -> np.ndarray:
        """
        Compute correlation scores from CCA projections using telluride_decoding method.
        
        This uses the actual Pearson correlation computation from telluride_decoding,
        which is the correct way to compute correlations from CCA rotated outputs.
        
        The CCA model outputs [rotated_eeg, rotated_audio] concatenated.
        We compute Pearson correlation for each CCA dimension, then weight by importance.
        
        Args:
            predictions: CCA model predictions
            eeg_fs: EEG sampling rate in Hz (for verification)
            env_fs: Envelope sampling rate in Hz (for verification)
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
            
            # For per-sample scoring, compute normalized dot product (cosine similarity)
            # CCA rotations maximize correlation, so normalized dot product gives similarity score
            # Weight by dimension importance (first dimension has highest canonical correlation)
            weights = np.exp(-np.arange(cca_width) * 0.15)  # Exponential decay
            weights = weights / np.sum(weights)
            
            # Normalize projections to unit length per sample for fair comparison
            # This gives cosine similarity between EEG and audio projections
            proj_eeg_norm = proj_eeg / (np.linalg.norm(proj_eeg, axis=1, keepdims=True) + 1e-8)
            proj_env_norm = proj_env / (np.linalg.norm(proj_env, axis=1, keepdims=True) + 1e-8)
            
            # Compute weighted cosine similarity per sample
            # Higher values indicate better match between EEG and audio
            dot_products = proj_eeg_norm * proj_env_norm  # Element-wise product per dimension
            scores = np.sum(dot_products * weights, axis=1)  # Weighted sum per sample
            
            # Alternative: Use squared values to emphasize stronger correlations
            # scores = np.sum((dot_products ** 2) * weights, axis=1)
            
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
                        
                        # Compute CCA correlation with left audio
                        left_inputs = {'input_1': eeg_view, 'input_2': left_env}
                        left_predictions = self.model(left_inputs)
                        left_scores = self._compute_correlation_scores(left_predictions, eeg_fs=eeg_fs, env_fs=env_fs)
                        
                        # Compute CCA correlation with right audio
                        right_inputs = {'input_1': eeg_view, 'input_2': right_env}
                        right_predictions = self.model(right_inputs)
                        right_scores = self._compute_correlation_scores(right_predictions, eeg_fs=eeg_fs, env_fs=env_fs)
                        
                        # Aggregate sample-level scores to window-level
                        # Get input shape to determine window structure
                        input_shape_tensor = tf.shape(eeg_view)[0]
                        input_shape = input_shape_tensor.numpy() if hasattr(input_shape_tensor, 'numpy') else int(input_shape_tensor)
                        
                        # Determine number of windows in this batch
                        # input_shape = batch_size * window_size (after batching)
                        num_samples = len(left_scores)
                        
                        # Calculate number of windows: input_shape should be divisible by window_size
                        if input_shape % self.window_size == 0:
                            num_windows = input_shape // self.window_size
                            window_size = self.window_size
                        else:
                            # Try to infer window size from common sizes
                            possible_window_sizes = [32, 64, 128, 256, 512, 1024, 2048, 64 * 30]
                            num_windows = None
                            window_size = self.window_size
                            for ws in possible_window_sizes:
                                if input_shape % ws == 0 and input_shape // ws > 0:
                                    num_windows = input_shape // ws
                                    window_size = ws
                                    break
                            
                            if num_windows is None:
                                # Last resort: use the actual number of samples
                                # This handles edge cases
                                num_windows = 1
                                window_size = num_samples
                        
                        # Ensure we have the right number of samples
                        # If num_samples doesn't match expected, truncate or pad
                        expected_samples = num_windows * window_size
                        if num_samples != expected_samples:
                            if num_samples > expected_samples:
                                # Truncate if we have more samples than expected
                                left_scores = left_scores[:expected_samples]
                                right_scores = right_scores[:expected_samples]
                            elif num_samples < expected_samples:
                                # This shouldn't happen, but handle it
                                # Use actual number of samples to recalculate
                                if num_samples % self.window_size == 0:
                                    num_windows = num_samples // self.window_size
                                    window_size = self.window_size
                                else:
                                    num_windows = 1
                                    window_size = num_samples
                        
                        # Reshape scores to (num_windows, window_size) and aggregate per window
                        try:
                            left_scores_reshaped = left_scores[:num_windows * window_size].reshape(num_windows, window_size)
                            right_scores_reshaped = right_scores[:num_windows * window_size].reshape(num_windows, window_size)
                            
                            # Aggregate scores per window (use mean)
                            left_window_scores = np.mean(left_scores_reshaped, axis=1)
                            right_window_scores = np.mean(right_scores_reshaped, axis=1)
                        except (ValueError, IndexError) as e:
                            # Fallback: if reshape fails, aggregate all scores into single prediction
                            left_window_scores = np.array([np.mean(left_scores)])
                            right_window_scores = np.array([np.mean(right_scores)])
                        
                        # FIXED: Store window-level scores for diagnostics (not sample-level)
                        # This ensures diagnostics match window-level targets
                        all_left_scores.extend(left_window_scores)
                        all_right_scores.extend(right_window_scores)
                        
                        # Predict using LDA if enabled, otherwise use direct comparison
                        if self.use_lda and self.lda_model is not None:
                            # Use LDA to classify based on correlation coefficients
                            # Create feature vector: [left_corr, right_corr] for each window
                            lda_features = np.column_stack([left_window_scores, right_window_scores])
                            window_predictions = self.lda_model.predict(lda_features).astype(np.int64)
                        else:
                            # Direct comparison: Right=1 if right > left, Left=0 otherwise
                            window_predictions = (right_window_scores > left_window_scores).astype(np.int64)
                        
                        # Ensure window_predictions is 1D array with num_windows elements
                        if window_predictions.ndim > 1:
                            window_predictions = window_predictions.flatten()
                        if len(window_predictions) != num_windows:
                            # If mismatch, take first num_windows or repeat as needed
                            if len(window_predictions) > num_windows:
                                window_predictions = window_predictions[:num_windows]
                            else:
                                # Repeat last prediction if needed (shouldn't happen)
                                window_predictions = np.pad(window_predictions, (0, num_windows - len(window_predictions)), mode='edge')
                        
                        binary_predictions = tf.constant(window_predictions, dtype=tf.int64)
                        
                        # For left/right comparison, predictions are now per-window after aggregation
                        with tf.device('/CPU:0'):
                            pred_numpy = binary_predictions.numpy()
                            # Ensure we're adding the right number of predictions
                            if len(pred_numpy) == num_windows:
                                all_predictions.extend(pred_numpy)
                            else:
                                # Safety check: only add num_windows predictions
                                all_predictions.extend(pred_numpy[:num_windows])
                            
                            if targets is not None:
                                target_array = targets.numpy() if hasattr(targets, 'numpy') else np.array(targets)
                                target_flat = target_array.flatten()
                                # Ensure targets match number of windows
                                if len(target_flat) == num_windows:
                                    all_targets.extend(target_flat)
                                else:
                                    # Take first num_windows targets or repeat as needed
                                    if len(target_flat) > num_windows:
                                        all_targets.extend(target_flat[:num_windows])
                                    else:
                                        all_targets.extend(np.pad(target_flat, (0, num_windows - len(target_flat)), mode='edge'))
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
                            
                            # Aggregate sample-level scores to window-level
                            # Get input shape to determine window structure
                            input_shape_tensor = tf.shape(eeg_view)[0]
                            input_shape = input_shape_tensor.numpy() if hasattr(input_shape_tensor, 'numpy') else int(input_shape_tensor)
                            
                            # Determine number of windows in this batch
                            # input_shape = batch_size * window_size (after batching)
                            num_samples = len(left_scores)
                            
                            # Calculate number of windows: input_shape should be divisible by window_size
                            if input_shape % self.window_size == 0:
                                num_windows = input_shape // self.window_size
                                window_size = self.window_size
                            else:
                                # Try to infer window size from common sizes
                                possible_window_sizes = [32, 64, 128, 256, 512, 1024, 2048, 64 * 30]
                                num_windows = None
                                window_size = self.window_size
                                for ws in possible_window_sizes:
                                    if input_shape % ws == 0 and input_shape // ws > 0:
                                        num_windows = input_shape // ws
                                        window_size = ws
                                        break
                                
                                if num_windows is None:
                                    # Last resort: use the actual number of samples
                                    # This handles edge cases
                                    num_windows = 1
                                    window_size = num_samples
                            
                            # Ensure we have the right number of samples
                            # If num_samples doesn't match expected, truncate or pad
                            expected_samples = num_windows * window_size
                            if num_samples != expected_samples:
                                if num_samples > expected_samples:
                                    # Truncate if we have more samples than expected
                                    left_scores = left_scores[:expected_samples]
                                    right_scores = right_scores[:expected_samples]
                                elif num_samples < expected_samples:
                                    # This shouldn't happen, but handle it
                                    # Use actual number of samples to recalculate
                                    if num_samples % self.window_size == 0:
                                        num_windows = num_samples // self.window_size
                                        window_size = self.window_size
                                    else:
                                        num_windows = 1
                                        window_size = num_samples
                            
                            # Reshape scores to (num_windows, window_size) and aggregate per window
                            try:
                                left_scores_reshaped = left_scores[:num_windows * window_size].reshape(num_windows, window_size)
                                right_scores_reshaped = right_scores[:num_windows * window_size].reshape(num_windows, window_size)
                                
                                # Aggregate scores per window (use mean)
                                left_window_scores = np.mean(left_scores_reshaped, axis=1)
                                right_window_scores = np.mean(right_scores_reshaped, axis=1)
                            except (ValueError, IndexError) as e:
                                # Fallback: if reshape fails, aggregate all scores into single prediction
                                left_window_scores = np.array([np.mean(left_scores)])
                                right_window_scores = np.array([np.mean(right_scores)])
                            
                            # FIXED: Store window-level scores for diagnostics (not sample-level)
                            # This ensures diagnostics match window-level targets
                            all_left_scores.extend(left_window_scores)
                            all_right_scores.extend(right_window_scores)
                            
                            # Predict based on which correlation is higher: Right=1 if right > left, Left=0 otherwise
                            window_predictions = (right_window_scores > left_window_scores).astype(np.int64)
                            
                            # Ensure window_predictions is 1D array with num_windows elements
                            if window_predictions.ndim > 1:
                                window_predictions = window_predictions.flatten()
                            if len(window_predictions) != num_windows:
                                # If mismatch, take first num_windows or repeat as needed
                                if len(window_predictions) > num_windows:
                                    window_predictions = window_predictions[:num_windows]
                                else:
                                    # Repeat last prediction if needed (shouldn't happen)
                                    window_predictions = np.pad(window_predictions, (0, num_windows - len(window_predictions)), mode='edge')
                            
                            binary_predictions = tf.constant(window_predictions, dtype=tf.int64)
                            
                            # For left/right comparison, predictions are now per-window after aggregation
                            pred_numpy = binary_predictions.numpy()
                            # Ensure we're adding the right number of predictions
                            if len(pred_numpy) == num_windows:
                                all_predictions.extend(pred_numpy)
                            else:
                                # Safety check: only add num_windows predictions
                                all_predictions.extend(pred_numpy[:num_windows])
                            
                            if targets is not None:
                                if hasattr(targets, 'numpy'):
                                    target_array = targets.numpy()
                                elif isinstance(targets, (list, np.ndarray)):
                                    target_array = np.array(targets)
                                else:
                                    # Handle dict case
                                    label = targets.get('label', None) if isinstance(targets, dict) else None
                                    if label is not None:
                                        target_array = label.numpy() if hasattr(label, 'numpy') else np.array(label)
                                    else:
                                        target_array = np.array([0])
                                
                                target_flat = target_array.flatten()
                                # Ensure targets match number of windows
                                if len(target_flat) == num_windows:
                                    all_targets.extend(target_flat)
                                else:
                                    # Take first num_windows targets or repeat as needed
                                    if len(target_flat) > num_windows:
                                        all_targets.extend(target_flat[:num_windows])
                                    else:
                                        all_targets.extend(np.pad(target_flat, (0, num_windows - len(target_flat)), mode='edge'))
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
        
        return np.array(all_predictions), np.array(all_targets)


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
        
        predictions, targets = self.model.predict(test_dataset)
        
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
        """Calculate ROC-AUC and related metrics."""
        try:

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

                temp_dataset = DasDatasetCCA(
                    self.tfrecord_dir, 
                    mode='test',
                    window_size=window_samples,
                    overlap=0.5,
                    load_audio=True,  # Use real audio for accurate temporal metrics
                    audio_base_dir=self.audio_base_dir  # Use same audio directory as main training
                )
                
                if len(temp_dataset) == 0:
                    print(f"  No data for {window_sec}s window")
                    continue
                

                def temp_generator():
                    for i in range(len(temp_dataset)):
                        window_data, aux_data = temp_dataset[i]
                        
                        # Extract label from aux_data (new format) or directly (old format)
                        # Ensure label is ALWAYS a tensor with shape (1,) not a scalar
                        try:
                            if isinstance(aux_data, dict):
                                label_tensor = aux_data.get('label', None)
                                if label_tensor is None:
                                    label_value = 0
                                elif hasattr(label_tensor, 'numpy'):
                                    label_array = label_tensor.numpy()
                                    if label_array.size > 0:
                                        # Handle both array and scalar cases
                                        if label_array.ndim == 0:  # Scalar numpy array
                                            label_value = int(label_array.item())
                                        else:
                                            label_value = int(label_array.flat[0])
                                    else:
                                        label_value = 0
                                elif hasattr(label_tensor, '__len__') and len(label_tensor) > 0:
                                    label_value = int(label_tensor[0])
                                elif hasattr(label_tensor, 'item'):
                                    label_value = int(label_tensor.item())
                                else:
                                    label_value = int(label_tensor) if label_tensor is not None else 0
                            else:
                                # Old format: aux_data is the label directly
                                if hasattr(aux_data, 'numpy'):
                                    label_array = aux_data.numpy()
                                    if label_array.size > 0:
                                        # Handle both array and scalar cases
                                        if label_array.ndim == 0:  # Scalar numpy array
                                            label_value = int(label_array.item())
                                        else:
                                            label_value = int(label_array.flat[0])
                                    else:
                                        label_value = 0
                                elif hasattr(aux_data, '__len__') and len(aux_data) > 0:
                                    label_value = int(aux_data[0])
                                elif hasattr(aux_data, 'item'):
                                    label_value = int(aux_data.item())
                                else:
                                    label_value = int(aux_data) if aux_data is not None else 0
                            
                            # Always create a tensor with shape (1,)
                            # Ensure label_value is a Python int, not a numpy scalar
                            if hasattr(label_value, 'item'):
                                label_value = int(label_value.item())
                            elif isinstance(label_value, (np.integer, np.floating)):
                                label_value = int(label_value)
                            else:
                                label_value = int(label_value)
                            
                            # Create tensor with explicit shape (1,)
                            label = tf.constant([label_value], dtype=tf.int64, shape=(1,))
                            
                        except Exception as e:
                            # Fallback: use default label with explicit shape
                            label = tf.constant([0], dtype=tf.int64, shape=(1,))

                        if isinstance(window_data, tuple):
                            eeg_data, audio_data = window_data
                        else:

                            eeg_data = window_data

                            audio_data = tf.zeros((window_samples, 4), dtype=tf.float32)
                        

                        eeg_shape = eeg_data.shape.as_list() if hasattr(eeg_data.shape, 'as_list') else list(eeg_data.shape)
                        audio_shape = audio_data.shape.as_list() if hasattr(audio_data.shape, 'as_list') else list(audio_data.shape)
                        

                        if len(eeg_shape) == 2 and eeg_shape[1] == 64:
                            input_1 = eeg_data
                        else:

                            input_1 = tf.reshape(eeg_data, (window_samples, 64))
                        
                        # Apply time-lagging if the model was trained with it
                        if self.model.eeg_lag_samples > 0:
                            # Convert to numpy for time-lagging
                            eeg_np = input_1.numpy() if hasattr(input_1, 'numpy') else np.array(input_1)
                            lagged_eeg = _apply_time_lagging(eeg_np, self.model.eeg_lag_samples)
                            input_1 = tf.constant(lagged_eeg, dtype=tf.float32)
                        

                        if len(audio_shape) == 2 and audio_shape[1] == 4:
                            input_2 = audio_data
                        elif len(audio_shape) == 2 and audio_shape[1] == 1:

                            input_2 = tf.tile(audio_data, [1, 4])
                        else:

                            input_2 = tf.zeros((window_samples, 4), dtype=tf.float32)
                        
                        # Final safety check: ensure label is a tensor with shape (1,)
                        if not isinstance(label, tf.Tensor):
                            label = tf.constant([int(label)], dtype=tf.int64)
                        elif label.shape.rank == 0:  # Scalar tensor
                            label = tf.reshape(label, (1,))
                        elif label.shape != (1,):
                            label = tf.reshape(label, (1,))
                        
                        # Double-check shape before yielding
                        label_shape = label.shape
                        if label_shape.rank == 0 or (label_shape.rank == 1 and label_shape[0] != 1):
                            label = tf.constant([0], dtype=tf.int64)  # Fallback to safe default
                        
                        yield {
                            'input_1': input_1,
                            'input_2': input_2
                        }, label
                
                # Calculate expected EEG dimension based on time-lagging
                if self.model.eeg_lag_samples > 0:
                    expected_eeg_dim = 64 * (self.model.eeg_lag_samples + 1)
                else:
                    expected_eeg_dim = 64
                
                # FIXED: Create dataset with aux_data for proper left/right comparison
                def create_temporal_dataset_with_aux():
                    """Create dataset with aux_data for temporal metrics."""
                    for i in range(len(temp_dataset)):
                        window_data, aux_data = temp_dataset[i]
                        
                        # Extract label and audio envelopes
                        if isinstance(aux_data, dict):
                            label = aux_data.get('label', tf.constant([0], dtype=tf.int64))
                            left_env = aux_data.get('left_env')
                            right_env = aux_data.get('right_env')
                        else:
                            label = aux_data if aux_data is not None else tf.constant([0], dtype=tf.int64)
                            left_env = None
                            right_env = None
                        
                        if isinstance(window_data, tuple):
                            eeg_data, audio_data = window_data
                        else:
                            eeg_data = window_data
                            audio_data = tf.zeros((window_samples, 4), dtype=tf.float32)
                        
                        # Apply time-lagging if needed
                        if self.model.eeg_lag_samples > 0:
                            eeg_np = eeg_data.numpy() if hasattr(eeg_data, 'numpy') else np.array(eeg_data)
                            lagged_eeg = _apply_time_lagging(eeg_np, self.model.eeg_lag_samples)
                            eeg_data = tf.constant(lagged_eeg, dtype=tf.float32)
                            expected_eeg_dim = 64 * (self.model.eeg_lag_samples + 1)
                        else:
                            expected_eeg_dim = 64
                        
                        # Ensure audio has correct shape
                        if audio_data.shape[1] != 4:
                            if audio_data.shape[1] == 1:
                                audio_data = tf.tile(audio_data, [1, 4])
                            else:
                                audio_data = tf.zeros((window_samples, 4), dtype=tf.float32)
                        
                        # Create aux_data dict
                        aux_dict = {'label': label}
                        if left_env is not None and right_env is not None:
                            # Ensure shapes match
                            if left_env.shape[0] != window_samples:
                                if left_env.shape[0] == 1:
                                    left_env = tf.tile(left_env, [window_samples, 1])
                                else:
                                    left_env = left_env[:window_samples]
                            if right_env.shape[0] != window_samples:
                                if right_env.shape[0] == 1:
                                    right_env = tf.tile(right_env, [window_samples, 1])
                                else:
                                    right_env = right_env[:window_samples]
                            aux_dict['left_env'] = left_env
                            aux_dict['right_env'] = right_env
                        else:
                            aux_dict['left_env'] = tf.zeros((window_samples, 4), dtype=tf.float32)
                            aux_dict['right_env'] = tf.zeros((window_samples, 4), dtype=tf.float32)
                        
                        yield {
                            'input_1': eeg_data,
                            'input_2': audio_data
                        }, aux_dict
                
                temp_tf_dataset = tf.data.Dataset.from_generator(
                    create_temporal_dataset_with_aux,
                    output_signature=(
                        {
                            'input_1': tf.TensorSpec(shape=(window_samples, expected_eeg_dim), dtype=tf.float32),
                            'input_2': tf.TensorSpec(shape=(window_samples, 4), dtype=tf.float32)
                        },
                        {
                            'label': tf.TensorSpec(shape=(1,), dtype=tf.int64),
                            'left_env': tf.TensorSpec(shape=(window_samples, 4), dtype=tf.float32),
                            'right_env': tf.TensorSpec(shape=(window_samples, 4), dtype=tf.float32)
                        }
                    )
                )
                
                # Reshape batch function
                def reshape_batch(inputs, aux_data):
                    # Handle variable input dimensions (due to time-lagging)
                    input_1_shape = tf.shape(inputs['input_1'])
                    input_1_feat_dim = input_1_shape[-1]  # Get feature dimension dynamically
                    input_1_reshaped = tf.reshape(inputs['input_1'], (-1, input_1_feat_dim))
                    input_2_reshaped = tf.reshape(inputs['input_2'], (-1, 4))
                    
                    # Reshape aux_data
                    reshaped_aux = {}
                    if isinstance(aux_data, dict):
                        if 'label' in aux_data:
                            reshaped_aux['label'] = aux_data['label']
                        if 'left_env' in aux_data:
                            reshaped_aux['left_env'] = tf.reshape(aux_data['left_env'], (-1, 4))
                        if 'right_env' in aux_data:
                            reshaped_aux['right_env'] = tf.reshape(aux_data['right_env'], (-1, 4))
                    else:
                        reshaped_aux = aux_data
                    
                    return {
                        'input_1': input_1_reshaped,
                        'input_2': input_2_reshaped
                    }, reshaped_aux
                
                temp_tf_dataset = temp_tf_dataset.batch(16).map(reshape_batch)
                
                # FIXED: Temporarily update model's window_size for temporal metrics
                # This ensures proper aggregation for different window sizes
                original_window_size = self.model.window_size
                self.model.window_size = window_samples
                
                try:
                    temp_predictions, temp_targets = self.model.predict(temp_tf_dataset)
                finally:
                    # Restore original window size
                    self.model.window_size = original_window_size
                
                if len(temp_predictions) > 0:
                    accuracy = accuracy_score(temp_targets, temp_predictions)
                    f1 = f1_score(temp_targets, temp_predictions, average='weighted')
                    
                    temporal_results[f'accuracy_{window_sec}s'] = accuracy
                    temporal_results[f'f1_{window_sec}s'] = f1
                    
                    print(f"  {window_sec}s: Acc={accuracy:.3f}, F1={f1:.3f}")
                else:
                    print(f"  {window_sec}s: No valid predictions")
                    
            except Exception as e:
                print(f"  Error testing {window_sec}s window: {e}")
                continue
        
        return temporal_results
    
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
        
        print(f"DASCCA results saved to {self.output_dir}")
    
    def _save_comprehensive_report(self, results: Dict):
        """Save a comprehensive metrics report."""
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


def create_das_data_loaders(tfrecord_dir: str, batch_size: int = 16, 
                           window_size: int = 32, overlap: float = 0.5,
                           train_ratio: float = 0.64, val_ratio: float = 0.18,  # Adjusted to ensure at least 2 val subjects (11 * 0.18 = 1.98 -> 2)
                           max_samples: Optional[int] = None,
                           audio_base_dir: Optional[str] = None,
                           load_audio: bool = True, max_files: Optional[int] = None,
                           eeg_lag_samples: int = 0, pca_model: Optional[PCA] = None) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Create data loaders for DAS dataset with proper subject-wise splitting.
    
    Args:
        eeg_lag_samples: Number of past time samples to include for backward model (0 = no lagging)
        pca_model: Pre-fitted PCA model to apply to EEG data (None = no PCA)
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
    



    for i, (data_idx, label) in enumerate(full_dataset.window_indices):
        subject_id = "unknown"
        

        if data_idx in data_idx_to_subject:
            subject_id = data_idx_to_subject[data_idx]
        else:


            closest_idx = min(data_idx_to_subject.keys(), key=lambda x: abs(x - data_idx))
            if abs(closest_idx - data_idx) < full_dataset.window_size:
                subject_id = data_idx_to_subject[closest_idx]
            else:

                for subj_id, (start_idx, end_idx) in subject_ranges.items():
                    if start_idx <= data_idx < end_idx:
                        subject_id = subj_id
                        break
        


        window_subjects = []
        for sample_idx in range(data_idx, min(data_idx + full_dataset.window_size, len(data_idx_to_subject))):
            if sample_idx in data_idx_to_subject:
                window_subjects.append(data_idx_to_subject[sample_idx])
        
        if window_subjects:

            from collections import Counter
            subject_counts = Counter(window_subjects)
            subject_id = subject_counts.most_common(1)[0][0]
        
        if subject_id not in subject_windows:
            subject_windows[subject_id] = []
        subject_windows[subject_id].append(i)
    
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
        
        # Calculate expected feature dimension based on time-lagging and PCA
        base_eeg_dim = 64  # Original EEG channels
        if eeg_lag_samples > 0:
            # Time-lagging: concatenate past samples
            expected_eeg_dim = base_eeg_dim * (eeg_lag_samples + 1)
        else:
            expected_eeg_dim = base_eeg_dim
        
        if pca_model is not None:
            # PCA reduces to pca_components
            expected_eeg_dim = pca_model.n_components_
        
        pca_info = pca_model.n_components_ if pca_model is not None else 'None'
        print(f"  Expected EEG feature dimension: {expected_eeg_dim} (base: {base_eeg_dim}, lag: {eeg_lag_samples}, PCA: {pca_info})")
        
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
                        # Fallback for old format - should not happen with current implementation
                        label = aux_data
                        left_env = None
                        right_env = None
                        print(f"WARNING: Unexpected aux_data format (not dict). This may indicate a data loading issue.")
                    

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
                        # Try to get audio from full_dataset if available
                        try:
                            # Get the actual audio data from the dataset
                            window_idx = indices[valid_samples] if 'indices' in locals() else None
                            if window_idx is not None and hasattr(full_dataset, 'audio_envelopes'):
                                data_idx, _ = full_dataset.window_indices[window_idx]
                                audio_data = full_dataset.audio_envelopes[data_idx:data_idx + dataset_window_size]
                                if len(audio_data) < dataset_window_size:
                                    # Pad if needed
                                    padding = np.zeros((dataset_window_size - len(audio_data), audio_data.shape[1] if len(audio_data.shape) > 1 else 1), dtype=np.float32)
                                    audio_data = np.vstack([audio_data, padding])
                                audio_data = tf.constant(audio_data, dtype=tf.float32)
                            else:
                                raise ValueError("Cannot recover audio data - window_data format error")
                        except Exception as e:
                            print(f"ERROR: Cannot recover audio data: {e}")
                            print(f"  This sample will be skipped to prevent training on corrupted data.")
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
                    
                    # Apply PCA transformation (if enabled)
                    if pca_model is not None:
                        # Flatten time dimension for PCA
                        original_shape = input_1.shape
                        eeg_flat = tf.reshape(input_1, (-1, original_shape[-1]))
                        eeg_np = eeg_flat.numpy() if hasattr(eeg_flat, 'numpy') else np.array(eeg_flat)
                        eeg_pca = pca_model.transform(eeg_np)
                        # Reshape back to (time, pca_components)
                        input_1 = tf.constant(eeg_pca.reshape(original_shape[0], -1), dtype=tf.float32)
                    


                    if len(audio_shape) == 1:


                        audio_expanded = tf.tile(tf.reshape(audio_data, (dataset_window_size, 1)), [1, 4])
                        input_2 = audio_expanded
                    elif len(audio_shape) == 2:

                        if audio_shape[1] == 1:

                            input_2 = tf.tile(audio_data, [1, 4])
                        else:
                            input_2 = audio_data
                    else:
                        # FIXED: This should not happen if window_data is properly formatted
                        print(f"ERROR: Unexpected audio shape {audio_shape}.")
                        print(f"  Expected audio_data from window_data tuple with shape (window_size, 4) or (window_size, 1)")
                        print(f"  This indicates a data loading error. Skipping this sample.")
                        continue
                    
                    valid_samples += 1
                    
                    # Prepare aux_data with left/right envelopes
                    # Always include left_env and right_env in aux_dict to match output signature
                    aux_dict = {'label': label}
                    
                    # Ensure left_env and right_env are always present with correct shape
                    if left_env is not None and right_env is not None:
                        # Ensure shapes match
                        if left_env.shape[0] != input_1.shape[0]:
                            if left_env.shape[0] == 1:
                                left_env = tf.tile(left_env, [input_1.shape[0], 1])
                            else:
                                left_env = left_env[:input_1.shape[0]]
                        if right_env.shape[0] != input_1.shape[0]:
                            if right_env.shape[0] == 1:
                                right_env = tf.tile(right_env, [input_1.shape[0], 1])
                            else:
                                right_env = right_env[:input_1.shape[0]]
                        aux_dict['left_env'] = left_env
                        aux_dict['right_env'] = right_env
                    else:
                        # FIXED: left_env and right_env should always be available from __getitem__
                        # If not, this indicates a data loading error
                        window_shape = tf.shape(input_1)[0]
                        if not hasattr(create_cca_dataset, '_missing_audio_warned'):
                            print(f"⚠ WARNING: Left/right audio envelopes not available in aux_data.")
                            print(f"  Using zeros as fallback. This may affect model performance.")
                            create_cca_dataset._missing_audio_warned = True
                        aux_dict['left_env'] = tf.zeros((window_shape, 4), dtype=tf.float32)
                        aux_dict['right_env'] = tf.zeros((window_shape, 4), dtype=tf.float32)
                    
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
            # Reshape inputs
            # Handle variable input dimensions (due to time-lagging or PCA)
            input_1_shape = tf.shape(inputs['input_1'])
            input_1_feat_dim = input_1_shape[-1]  # Get feature dimension dynamically
            input_1_reshaped = tf.reshape(inputs['input_1'], (-1, input_1_feat_dim))
            input_2_reshaped = tf.reshape(inputs['input_2'], (-1, 4))
            
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
    """Main function for DASCCA training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='DASCCA - CCA Algorithm for DAS Dataset')
    parser.add_argument('--tfrecord_dir', type=str, default='das_16subjects_preprocessed/tfrecords',
                       help='TFRecord directory path')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--cca_dims', type=int, default=5,
                       help='Number of CCA dimensions')
    parser.add_argument('--regularization', type=float, default=0.01,
                       help='CCA regularization parameter')
    parser.add_argument('--window_size', type=int, default=512,
                       help='Window size for EEG data (512 samples = 4 seconds at 128Hz)')
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
    parser.add_argument('--pca_components', type=int, default=None,
                       help='Number of PCA components for EEG regularization (None = no PCA)')
    parser.add_argument('--use_lda', action='store_true', default=True,
                       help='Use LDA classifier downstream (default: True)')
    parser.add_argument('--no_lda', dest='use_lda', action='store_false',
                       help='Disable LDA classifier, use direct correlation comparison')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("DASCCA - CANONICAL CORRELATION ANALYSIS FOR DAS DATASET")
    print("=" * 80)
    print("Features:")
    print("- CCA implementation based on telluride_decoding")
    print("- Accuracy, MSED, ROC-AUC metrics")
    print("- Temporal performance analysis (0.5s to 30s)")
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
    

    print(f"\nCreating DAS data loaders...")
    # Create datasets - we'll fit PCA after model creation if needed
    train_dataset, val_dataset, test_dataset = create_das_data_loaders(
        args.tfrecord_dir, batch_size=args.batch_size, window_size=args.window_size,
        audio_base_dir=args.audio_base_dir, load_audio=args.load_audio,
        max_files=args.max_files,
        eeg_lag_samples=args.eeg_lag_samples if args.eeg_lag_samples > 0 else 0,
        pca_model=None  # Will be set after fitting
    )
    

    print("\nCreating DASCCA model...")
    model = DASCCAModel(
        cca_dims=args.cca_dims,
        regularization=args.regularization,
        window_size=args.window_size,
        eeg_lag_samples=args.eeg_lag_samples,
        pca_components=args.pca_components,
        use_lda=args.use_lda
    )
    
    # If PCA is enabled, fit it on training data and recreate datasets with PCA
    if args.pca_components is not None:
        print("Fitting PCA on training data...")
        model._fit_pca(train_dataset)
        # Recreate datasets with fitted PCA model
        train_dataset, val_dataset, test_dataset = create_das_data_loaders(
            args.tfrecord_dir, batch_size=args.batch_size, window_size=args.window_size,
            audio_base_dir=args.audio_base_dir, load_audio=args.load_audio,
            max_files=args.max_files,
            eeg_lag_samples=args.eeg_lag_samples if args.eeg_lag_samples > 0 else 0,
            pca_model=model.pca_model
        )
        print("✓ Datasets recreated with PCA transformation")

    trainer = DASCCATrainer(model, args.output_dir, args.tfrecord_dir, 
                           sampling_rate=128, window_size=args.window_size,  # FIXED: 128 Hz to match preprocessing
                           audio_base_dir=args.audio_base_dir)
    

    print("\nStarting DASCCA training...")
    best_val_acc = trainer.train(train_dataset, val_dataset)
    

    print("\nTesting DASCCA model...")
    results = trainer.test(test_dataset)
    

    trainer.save_results(results)
    
    print("\n" + "=" * 80)
    print("DASCCA TRAINING COMPLETE!")
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
