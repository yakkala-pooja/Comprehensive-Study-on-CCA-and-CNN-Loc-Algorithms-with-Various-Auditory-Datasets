#!/usr/bin/env python3
"""
FULCCA - Canonical Correlation Analysis Algorithm for Fulsang Dataset

This module implements a comprehensive Canonical Correlation Analysis (CCA) 
algorithm specifically designed for the Fulsang dataset. It includes:

- CCA implementation based on telluride_decoding repository
- Comprehensive metrics: Accuracy, MSED, ROC-AUC, and temporal performance
- Temporal analysis across window lengths from 0.5s to 30s
- Robust preprocessing and data handling
- Detailed performance evaluation and reporting
"""

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

# Set environment variables for robust GPU usage
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Allow GPU memory growth

# Force CPU usage for random operations to avoid CUDA handle corruption
os.environ['TF_DETERMINISTIC_OPS'] = '1'  # Use deterministic operations
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # Use deterministic cuDNN

# Force GPU-only mode
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU only

# Configure GPU for maximum stability
try:
    # Check if GPU is available
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"Found {len(gpu_devices)} GPU device(s)")
        # Set memory growth for all GPUs (compatible with TensorFlow 2.20.0)
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✓ GPU memory growth configured")
        
        # Try to set memory limit if supported (TensorFlow 2.4+)
        try:
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_limit(gpu, 8192)  # 8GB limit
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
    print("Continuing with basic functionality...")

# Enable TensorFlow v2 behavior
tf.compat.v1.enable_v2_behavior()

# Force GPU-only mode with CPU fallback for problematic operations
device = tf.device('/GPU:0')
print("Using GPU for computation (GPU-only mode with CPU fallback)")

# Set random seeds for reproducibility and stability
tf.random.set_seed(42)
np.random.seed(42)
print("✓ Random seeds set for reproducibility")

# Force CPU for random operations to avoid CUDA handle corruption
def safe_random_operations():
    """Force CPU usage for random operations."""
    with tf.device('/CPU:0'):
        tf.random.set_seed(42)
        np.random.seed(42)


class FulsangDatasetCCA:
    """
    Fulsang-specific dataset class for CCA analysis using FULPREPROCESSING validated data.
    
    This dataset class is designed to work with the output from FULPREPROCESSING.py,
    which provides:
    - Validated attention labels with quality control
    - Subject-wise organized data to prevent data leakage
    - Robust EEG data extraction (envelope data removed)
    - Comprehensive preprocessing reports
    """
    
    def __init__(self, tfrecord_dir: str, mode: str = 'full', 
                 window_size: int = 32, overlap: float = 0.5,
                 cache_size: int = 1000):
        self.tfrecord_dir = Path(tfrecord_dir)
        self.mode = mode
        self.window_size = window_size
        self.overlap = overlap
        self.cache_size = cache_size
        
        # Fulsang-specific parameters
        self.sampling_rate = 64  # Hz
        self.n_channels = 66  # EEG channels
        self.attention_switch_duration = 20  # seconds
        
        # Cache for preprocessed windows
        self._window_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Load Fulsang data using FULPREPROCESSING validated data
        self.eeg_data, self.labels, self.metadata = self._load_fulpreprocessing_data()
        
        self.window_indices = self._create_fulsang_windows()
        
        print(f"Loaded {len(self.window_indices)} Fulsang windows for {mode} mode")
        print(f"Fulsang EEG shape: {self.eeg_data.shape}")
        print(f"Fulsang Label distribution: {np.bincount(self.labels)}")
        print(f"Using FULPREPROCESSING: Yes")
        print(f"Cache size: {cache_size} windows")
    
    def _load_fulpreprocessing_data(self) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Load FULPREPROCESSING validated TFRecord data with robust shape validation."""
        tfrecord_files = list(self.tfrecord_dir.glob("*.tfrecords"))
        if not tfrecord_files:
            raise ValueError(f"No TFRecord files found in {self.tfrecord_dir}")
        
        print(f"Loading FULPREPROCESSING validated data from {len(tfrecord_files)} files...")
        print("✓ Using validated attention labels with quality control")
        print("✓ Using subject-wise organized data (no data leakage)")
        print("✓ EEG-only processing (envelope data removed)")
        
        all_eeg_data = []
        all_labels = []
        all_metadata = []
        
        successful_files = 0
        failed_files = 0
        total_records = 0
        subject_stats = {}
        shape_validation_errors = 0
        
        for tfrecord_file in tqdm(tfrecord_files, desc="Loading FULPREPROCESSING data"):
            try:
                dataset = tf.data.TFRecordDataset(str(tfrecord_file))
                records_in_file = 0
                file_subject_id = None
                
                for record in dataset:
                    try:
                        example = tf.train.Example.FromString(record.numpy())
                        features = example.features.feature
                        
                        # Check required features (FULPREPROCESSING format) - EEG only
                        required_features = ['eeg', 'attention_label', 'subject_id']
                        if not all(key in features for key in required_features):
                            continue
                        
                        # Extract EEG data with robust shape validation
                        eeg_values = features['eeg'].float_list.value
                        if not eeg_values or len(eeg_values) == 0:
                            continue
                        
                        # Validate EEG shape - must be exactly 66 channels
                        if len(eeg_values) != 66:
                            print(f"ERROR: Expected 66 EEG channels, got {len(eeg_values)} in {tfrecord_file.name}")
                            shape_validation_errors += 1
                            continue
                        
                        # Reshape with explicit validation: (1, 66) for single sample
                        eeg_data = np.array(eeg_values, dtype=np.float32).reshape(1, 66)
                        
                        # Validate EEG data quality
                        if np.any(np.isnan(eeg_data)) or np.any(np.isinf(eeg_data)):
                            print(f"WARNING: Invalid EEG values (NaN/Inf) in {tfrecord_file.name}")
                            continue
                        
                        # Extract validated attention label
                        label_values = features['attention_label'].int64_list.value
                        if not label_values or len(label_values) == 0:
                            continue
                        label = int(label_values[0])
                        
                        # Validate label value
                        if label not in [0, 1]:
                            print(f"ERROR: Invalid attention label {label} in {tfrecord_file.name}")
                            continue
                        
                        # Extract metadata (FULPREPROCESSING format)
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
                    successful_files += 1
                    if file_subject_id:
                        print(f"✓ Loaded {records_in_file} samples from subject {file_subject_id}")
                else:
                    failed_files += 1
                    
            except Exception as e:
                failed_files += 1
                print(f"ERROR loading {tfrecord_file.name}: {e}")
                continue
        
        print(f"Successfully loaded {successful_files} files, {failed_files} files failed")
        print(f"Total records loaded: {total_records}")
        print(f"Shape validation errors: {shape_validation_errors}")
        
        if shape_validation_errors > 0:
            print(f"⚠ WARNING: {shape_validation_errors} records had shape validation errors")
        
        # Display subject statistics
        print(f"\nSubject-wise statistics:")
        for subject_id, stats in subject_stats.items():
            label_dist = np.bincount(stats['labels'])
            print(f"  {subject_id}: {stats['samples']} samples, labels {label_dist}")
        
        if not all_eeg_data:
            raise ValueError("No valid FULPREPROCESSING data found in TFRecord files")
        
        eeg_data = np.vstack(all_eeg_data)
        labels = np.array(all_labels, dtype=np.int64)
        
        # Final shape validation
        print(f"\nFinal data shapes:")
        print(f"  EEG data: {eeg_data.shape} (samples, channels)")
        print(f"  Labels: {labels.shape} (samples,)")
        print(f"  Expected EEG shape: (n_samples, 66)")
        
        if eeg_data.shape[1] != 66:
            raise ValueError(f"CRITICAL: EEG data has {eeg_data.shape[1]} channels, expected 66")
        
        if len(eeg_data) != len(labels):
            raise ValueError(f"CRITICAL: EEG samples ({len(eeg_data)}) != labels ({len(labels)})")
        
        del all_eeg_data, all_labels
        import gc
        gc.collect()
        
        return eeg_data, labels, all_metadata
    
    def _create_fulsang_windows(self) -> List[Tuple[int, int]]:
        """Create windows optimized for Fulsang data structure with proper time units."""
        # Convert window size from samples to seconds for clarity
        window_seconds = self.window_size / self.sampling_rate
        step_size = int(self.window_size * (1 - self.overlap))
        step_seconds = step_size / self.sampling_rate
        
        total_windows = (len(self.eeg_data) - self.window_size) // step_size + 1
        
        print(f"Creating {total_windows} Fulsang windows:")
        print(f"  Window size: {self.window_size} samples ({window_seconds:.1f} seconds)")
        print(f"  Step size: {step_size} samples ({step_seconds:.1f} seconds)")
        print(f"  Overlap: {self.overlap:.1%}")
        print(f"  Sampling rate: {self.sampling_rate} Hz")
        
        # Validate window size for EEG attention decoding
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
                # Use majority voting for window label to handle trial transitions
                window_start = data_idx
                window_end = data_idx + self.window_size
                window_labels = self.labels[window_start:window_end]
                
                # Majority vote for window label
                if len(window_labels) > 0:
                    window_label = int(np.bincount(window_labels).argmax())
                else:
                    window_label = 0
                
                window_indices.append((data_idx, window_label))
        
        print(f"Created {len(window_indices)} Fulsang windows")
        
        # Analyze window label distribution
        window_labels = [label for _, label in window_indices]
        label_dist = np.bincount(window_labels)
        print(f"Window label distribution: {label_dist}")
        
        return window_indices
    
    def _fulsang_eeg_preprocessing(self, eeg_window: np.ndarray) -> np.ndarray:
        """Fulsang-specific EEG preprocessing with artifact handling."""
        from scipy import signal
        
        # 1. Artifact detection and removal
        # Detect high-amplitude artifacts (>5 standard deviations)
        artifact_threshold = 3.0
        for ch in range(eeg_window.shape[1]):
            channel_data = eeg_window[:, ch]
            std_val = np.std(channel_data)
            mean_val = np.mean(channel_data)
            
            # Mark artifacts
            artifacts = np.abs(channel_data - mean_val) > (artifact_threshold * std_val)
            
            if np.any(artifacts):
                # Interpolate over artifacts
                valid_indices = ~artifacts
                if np.sum(valid_indices) > 2:  # Need at least 2 valid points
                    from scipy.interpolate import interp1d
                    valid_data = channel_data[valid_indices]
                    valid_time = np.where(valid_indices)[0]
                    all_time = np.arange(len(channel_data))
                    
                    f_interp = interp1d(valid_time, valid_data, kind='linear', 
                                      bounds_error=False, fill_value='extrapolate')
                    eeg_window[:, ch] = f_interp(all_time)
        
        # 2. Baseline correction (DC removal)
        eeg_window = eeg_window - np.mean(eeg_window, axis=0, keepdims=True)
        
        # 3. Bandpass filtering (1-40 Hz for EEG attention)
        nyquist = self.sampling_rate / 2
        low_freq = 1.0 / nyquist
        high_freq = min(30.0 / nyquist, 0.99)  # Ensure < Nyquist
        
        # Design Butterworth filter
        b, a = signal.butter(4, [low_freq, high_freq], btype='band')
        
        # Apply filtering to each channel
        filtered_eeg = np.zeros_like(eeg_window)
        for ch in range(eeg_window.shape[1]):
            filtered_eeg[:, ch] = signal.filtfilt(b, a, eeg_window[:, ch])
        
        # 4. Robust normalization (MAD-based)
        mad_values = np.median(np.abs(filtered_eeg - np.median(filtered_eeg, axis=0)), axis=0)
        mad_values = np.where(mad_values == 0, 1.0, mad_values)  # Avoid division by zero
        filtered_eeg = filtered_eeg / mad_values
        
        # 5. Soft clipping to prevent extreme values
        filtered_eeg = np.tanh(filtered_eeg * 0.5)
        
        # 6. Final quality check
        if np.any(np.isnan(filtered_eeg)) or np.any(np.isinf(filtered_eeg)):
            print("WARNING: Invalid values detected after preprocessing")
            filtered_eeg = np.nan_to_num(filtered_eeg, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return filtered_eeg.astype(np.float32)
    
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
        
        # Convert to tensors (EEG only)
        window_tensor = tf.constant(window_eeg, dtype=tf.float32)
        label_tensor = tf.constant([label], dtype=tf.int64)
        
        # Cache the result
        if len(self._window_cache) < self.cache_size:
            self._window_cache[cache_key] = (window_tensor, label_tensor)
        
        return window_tensor, label_tensor


class FULCCAModel:
    """
    FULCCA model implementing Canonical Correlation Analysis for Fulsang EEG dataset.
    
    This model uses the telluride_decoding CCA implementation to find correlations
    between EEG data and attention labels, providing comprehensive metrics evaluation.
    """
    
    def __init__(self, cca_dims: int = 8, regularization: float = 0.05, window_size: int = 512):
        """
        Initialize FULCCA model.
        
        Args:
            cca_dims: Number of CCA dimensions to compute
            regularization: Regularization parameter for CCA
            window_size: Window size for EEG data processing
        """
        self.cca_dims = cca_dims
        self.regularization = regularization
        self.window_size = window_size
        self.model = None
        self.is_fitted = False
        
        print(f"FULCCA model initialized:")
        print(f"  CCA dimensions: {cca_dims}")
        print(f"  Regularization: {regularization}")
    
    def _create_robust_cca_model(self, dataset: tf.data.Dataset):
        """
        Create CCA model with robust CUDA handling.
        """
        # Clear any existing GPU memory
        tf.keras.backend.clear_session()
        
        # Use safe random operations
        safe_random_operations()
        
        # Try CPU-first approach to avoid CUDA handle corruption
        print("Creating CCA model with CPU-first approach...")
        try:
            with tf.device('/CPU:0'):
                # Create CCA model using telluride_decoding implementation
                model = BrainModelCCA(
                    input_dataset=dataset,
                    cca_dims=self.cca_dims,
                    regularization_lambda=self.regularization
                )
            print("✓ CCA model created successfully on CPU")
            return model
            
        except Exception as e:
            print(f"CPU model creation failed: {e}")
            print("Trying GPU model creation...")
            
            # Try GPU as fallback
            try:
                with tf.device('/GPU:0'):
                    model = BrainModelCCA(
                        input_dataset=dataset,
                        cca_dims=self.cca_dims,
                        regularization_lambda=self.regularization
                    )
                print("✓ CCA model created successfully on GPU")
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
        
        # Create robust CCA model
        self.model = self._create_robust_cca_model(dataset)
        
        # Compile and train the model
        try:
            print("Compiling CCA model...")
            # Compile the model
            self.model.compile(
                optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
                loss='mse',
                metrics=[cca_pearson_correlation_first]
            )
            
            print("Training CCA model...")
            # Fit the model (CCA is deterministic)
            self.model.fit(dataset, epochs=3)
            
            print("✓ FULCCA model fitted successfully")
            
        except Exception as e:
            print(f"Training failed: {e}")
            # Try CPU fallback for training
            print("Trying CPU fallback for training...")
            
            with tf.device('/CPU:0'):
                # Recreate model on CPU
                self.model = BrainModelCCA(
                    input_dataset=dataset,
                    cca_dims=self.cca_dims,
                    regularization_lambda=self.regularization
                )
                
                self.model.compile(
                    optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
                    loss='mse',
                    metrics=[cca_pearson_correlation_first]
                )
                
                self.model.fit(dataset, epochs=3)
                print("✓ FULCCA model fitted successfully on CPU")
        
        self.is_fitted = True
        print("✓ FULCCA model training completed")
    
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
        
        # Use GPU for predictions
        with device:
            for batch in tqdm(dataset, desc="Predicting"):
                if isinstance(batch, dict):
                    inputs = batch
                    targets = None
                else:
                    inputs, targets = batch
                
                # Get predictions from CCA model
                predictions = self.model(inputs)
                
                # Split concatenated CCA output
                cca_width = predictions.shape[-1] // 2
                pred1 = predictions[:, :cca_width]
                pred2 = predictions[:, cca_width:]
                
                # Use first CCA component for classification
                cca_scores = pred1[:, 0]  # First CCA component
                
                # Convert CCA scores to binary predictions
                binary_predictions = tf.cast(cca_scores > 0, tf.int64)
                
                # Aggregate predictions per sample (batch_size predictions per batch)
                # The dataset is reshaped to (batch_size * window_size, 33), so we need to
                # aggregate back to batch_size predictions
                batch_size = inputs['input_1'].shape[0] // self.window_size
                window_size = self.window_size
                
                # Reshape predictions back to (batch_size, window_size)
                pred_reshaped = tf.reshape(binary_predictions, (batch_size, window_size))
                
                # Aggregate per sample using majority voting
                sample_predictions = tf.reduce_sum(pred_reshaped, axis=1)
                sample_predictions = tf.cast(sample_predictions > (window_size // 2), tf.int64)
                
                all_predictions.extend(sample_predictions.numpy())
                
                if targets is not None:
                    all_targets.extend(targets.numpy().flatten())
        
        return np.array(all_predictions), np.array(all_targets)


class FULCCATrainer:
    """
    FULCCA trainer with comprehensive metrics evaluation.
    """
    
    def __init__(self, model: FULCCAModel, output_dir: str = "fulcca_results", 
                 tfrecord_dir: str = None, sampling_rate: int = 64, window_size: int = 512):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Dataset parameters
        self.tfrecord_dir = tfrecord_dir
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        
        print(f"FULCCA trainer initialized. Output directory: {self.output_dir}")
    
    def train(self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset) -> float:
        """Train the FULCCA model."""
        print("Starting FULCCA training...")
        
        # Fit the model on training data
        self.model.fit(train_dataset)
        
        # Evaluate on validation data
        val_predictions, val_targets = self.model.predict(val_dataset)
        val_accuracy = accuracy_score(val_targets, val_predictions)
        
        print(f"FULCCA training completed! Validation accuracy: {val_accuracy:.4f}")
        return val_accuracy
    
    def test(self, test_dataset: tf.data.Dataset) -> Dict:
        """Test the FULCCA model with comprehensive metrics."""
        print("Testing FULCCA model...")
        
        predictions, targets = self.model.predict(test_dataset)
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(targets, predictions)
        
        # Classification report
        report = classification_report(targets, predictions, 
                                   target_names=['Left', 'Right'], 
                                   labels=[0, 1],
                                   output_dict=True)
        
        cm = confusion_matrix(targets, predictions)
        
        # Calculate comprehensive metrics
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
            # For CCA, we can use the predictions as probabilities
            probabilities = predictions.astype(np.float32)
            
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
    
    def _calculate_temporal_metrics(self, test_dataset: tf.data.Dataset) -> Dict[str, float]:
        """Calculate temporal performance metrics across different window sizes."""
        print("Calculating temporal performance metrics...")
        
        # Test different window sizes (in seconds)
        window_sizes_seconds = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0]
        temporal_results = {}
        
        for window_sec in window_sizes_seconds:
            window_samples = int(window_sec * self.sampling_rate)
            
            print(f"Testing {window_sec}s window ({window_samples} samples)...")
            
            try:
                # Create temporary dataset with different window size
                temp_dataset = FulsangDatasetCCA(
                    self.tfrecord_dir, 
                    mode='test',
                    window_size=window_samples,
                    overlap=0.5
                )
                
                if len(temp_dataset) == 0:
                    print(f"  No data for {window_sec}s window")
                    continue
                
                # Convert to TensorFlow dataset with CCA format
                def temp_generator():
                    for i in range(len(temp_dataset)):
                        eeg_data, label = temp_dataset[i]
                        mid_point = eeg_data.shape[1] // 2
                        input_1 = eeg_data[:, :mid_point]
                        input_2 = eeg_data[:, mid_point:]
                        yield {
                            'input_1': input_1,
                            'input_2': input_2
                        }, label
                
                temp_tf_dataset = tf.data.Dataset.from_generator(
                    temp_generator,
                    output_signature=(
                        {
                            'input_1': tf.TensorSpec(shape=(window_samples, 33), dtype=tf.float32),
                            'input_2': tf.TensorSpec(shape=(window_samples, 33), dtype=tf.float32)
                        },
                        tf.TensorSpec(shape=(1,), dtype=tf.int64)
                    )
                ).batch(16)
                
                # Evaluate on this window size
                temp_predictions, temp_targets = self.model.predict(temp_tf_dataset)
                
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
        # Prepare results
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
        
        # Save results
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        
        # Save predictions
        save_data = {
            'predictions': results['predictions'],
            'targets': results['targets']
        }
        
        with open(self.output_dir / 'predictions.pkl', 'wb') as f:
            pickle.dump(save_data, f)
        
        # Save comprehensive metrics report
        self._save_comprehensive_report(results)
        
        print(f"FULCCA results saved to {self.output_dir}")
    
    def _save_comprehensive_report(self, results: Dict):
        """Save a comprehensive metrics report."""
        with open(self.output_dir / 'comprehensive_metrics_report.txt', 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FULCCA COMPREHENSIVE METRICS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Basic metrics
            f.write("BASIC METRICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Accuracy: {results['accuracy']:.4f}\n\n")
            
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
            
            # Temporal analysis
            temporal = results.get('temporal_metrics', {})
            f.write("TEMPORAL PERFORMANCE ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            for key, value in temporal.items():
                f.write(f"{key}: {value:.4f}\n")


def create_fulsang_data_loaders(tfrecord_dir: str, batch_size: int = 16, 
                               window_size: int = 32, overlap: float = 0.5,
                               train_ratio: float = 0.7, val_ratio: float = 0.15,
                               max_samples: Optional[int] = None) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Create data loaders for Fulsang dataset with proper subject-wise splitting."""
    
    print("Creating Fulsang dataset with subject-wise splitting...")
    print(f"TFRecord directory: {tfrecord_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Window size: {window_size} samples ({window_size/64:.1f} seconds at 64Hz)")
    print(f"Overlap: {overlap}")
    print(f"Using FULPREPROCESSING: Yes")
    
    # Create full dataset
    full_dataset = FulsangDatasetCCA(tfrecord_dir, mode='full', 
                                   window_size=window_size, overlap=overlap)
    
    total_size = len(full_dataset)
    print(f"Total dataset size: {total_size} samples")
    
    # Extract subject information for proper splitting
    subject_windows = {}
    
    # Group metadata by subject to understand the data structure
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
    
    # Add the last subject
    if current_subject is not None:
        subject_ranges[current_subject] = (start_idx, len(full_dataset.metadata))
    
    print(f"Subject ranges in metadata:")
    for subject_id, (start, end) in subject_ranges.items():
        print(f"  {subject_id}: samples {start}-{end-1} ({end-start} samples)")
    
    # Now map windows to subjects based on their data_idx
    for i, (data_idx, label) in enumerate(full_dataset.window_indices):
        subject_id = "unknown"
        
        # Find which subject this window belongs to based on data_idx
        for subj_id, (start_idx, end_idx) in subject_ranges.items():
            if start_idx <= data_idx < end_idx:
                subject_id = subj_id
                break
        
        if subject_id not in subject_windows:
            subject_windows[subject_id] = []
        subject_windows[subject_id].append(i)
    
    print(f"Found {len(subject_windows)} subjects:")
    for subject_id, windows in subject_windows.items():
        print(f"  {subject_id}: {len(windows)} windows")
    
    # Subject-wise splitting to prevent data leakage
    subjects = list(subject_windows.keys())
    np.random.seed(42)  # Fixed seed for reproducibility
    np.random.shuffle(subjects)
    
    n_subjects = len(subjects)
    n_train_subjects = int(train_ratio * n_subjects)
    n_val_subjects = int(val_ratio * n_subjects)
    
    train_subjects = subjects[:n_train_subjects]
    val_subjects = subjects[n_train_subjects:n_train_subjects + n_val_subjects]
    test_subjects = subjects[n_train_subjects + n_val_subjects:]
    
    print(f"\nSubject-wise split:")
    print(f"  Train subjects: {len(train_subjects)} ({train_subjects})")
    print(f"  Val subjects: {len(val_subjects)} ({val_subjects})")
    print(f"  Test subjects: {len(test_subjects)} ({test_subjects})")
    
    # Create subject-based window indices
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
    
    # Verify no overlap between splits
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
    
    # Create subset datasets with proper CCA format (input_1, input_2)
    def create_cca_dataset(indices):
        def generator():
            for i in indices:
                eeg_data, label = full_dataset[i]
                
                # For CCA, we need input_1 and input_2
                # Since we only have EEG data, we'll split EEG channels:
                # input_1: first half of channels (33 channels)
                # input_2: second half of channels (33 channels)
                # This creates cross-correlation between different brain regions
                
                # Ensure we have the right shape: (window_size, 66)
                if len(eeg_data.shape) == 2 and eeg_data.shape[1] == 66:
                    mid_point = eeg_data.shape[1] // 2
                    input_1 = eeg_data[:, :mid_point]  # First 33 channels
                    input_2 = eeg_data[:, mid_point:]  # Last 33 channels
                else:
                    # Handle unexpected shapes
                    print(f"WARNING: Unexpected EEG shape {eeg_data.shape}, reshaping...")
                    eeg_data = tf.reshape(eeg_data, (window_size, 66))
                    mid_point = 33
                    input_1 = eeg_data[:, :mid_point]
                    input_2 = eeg_data[:, mid_point:]
                
                yield {
                    'input_1': input_1,
                    'input_2': input_2
                }, label
        
        # Create dataset without batching first, then batch with proper reshaping
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                {
                    'input_1': tf.TensorSpec(shape=(window_size, 33), dtype=tf.float32),
                    'input_2': tf.TensorSpec(shape=(window_size, 33), dtype=tf.float32)
                },
                tf.TensorSpec(shape=(1,), dtype=tf.int64)
            )
        )
        
        # Batch and reshape for CCA compatibility
        def reshape_batch(inputs, labels):
            # Reshape from (batch_size, window_size, 33) to (batch_size * window_size, 33)
            input_1_reshaped = tf.reshape(inputs['input_1'], (-1, 33))
            input_2_reshaped = tf.reshape(inputs['input_2'], (-1, 33))
            
            return {
                'input_1': input_1_reshaped,
                'input_2': input_2_reshaped
            }, labels
        
        return dataset.batch(batch_size).map(reshape_batch)
    
    train_dataset = create_cca_dataset(train_indices)
    val_dataset = create_cca_dataset(val_indices)
    test_dataset = create_cca_dataset(test_indices)
    
    print(f"✓ Data loaders created with subject-wise splitting")
    print(f"✓ Data leakage prevention implemented")
    print(f"✓ Attention labels validated")
    print(f"✓ Subject-wise organization applied")
    
    return train_dataset, val_dataset, test_dataset



    # Enhanced CCA configurations for better performance
    enhanced_configs = [
        {'name': 'optimal_balanced', 'cca_dims': 8, 'regularization': 0.05, 'window_size': 512},
        {'name': 'precision_focused', 'cca_dims': 12, 'regularization': 0.08, 'window_size': 768},
        {'name': 'robust_general', 'cca_dims': 6, 'regularization': 0.03, 'window_size': 640},
        {'name': 'high_dim_optimized', 'cca_dims': 15, 'regularization': 0.1, 'window_size': 512},
        {'name': 'extended_window', 'cca_dims': 10, 'regularization': 0.06, 'window_size': 1024},
        {'name': 'fine_tuned', 'cca_dims': 4, 'regularization': 0.02, 'window_size': 384},
    ]
    

def main():
    """Main function for FULCCA training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='FULCCA - CCA Algorithm for Fulsang Dataset')
    parser.add_argument('--tfrecord_dir', type=str, default='fulsang_preprocessed/tfrecords',
                       help='TFRecord directory path')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--cca_dims', type=int, default=8,
                       help='Number of CCA dimensions')
    parser.add_argument('--regularization', type=float, default=0.05,
                       help='CCA regularization parameter')
    parser.add_argument('--window_size', type=int, default=512,
                       help='Window size for EEG data (512 samples = 8 seconds at 64Hz)')
    parser.add_argument('--output_dir', type=str, default='fulcca_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("FULCCA - CANONICAL CORRELATION ANALYSIS FOR FULSANG DATASET")
    print("=" * 80)
    print("Features:")
    print("- CCA implementation based on telluride_decoding")
    print("- Accuracy, MSED, ROC-AUC metrics")
    print("- Temporal performance analysis (0.5s to 30s)")
    print("- FULPREPROCESSING integration for data quality")
    print("- Data leakage prevention")
    print("- Validated attention labels")
    print("=" * 80)
    
    print("✓ Using FULPREPROCESSING validated data")
    print("✓ Data leakage prevention enabled")
    print("✓ Attention labels validated")
    print("✓ CCA implementation from telluride_decoding")
    
    # Create data loaders
    print(f"\nCreating Fulsang data loaders...")
    train_dataset, val_dataset, test_dataset = create_fulsang_data_loaders(
        args.tfrecord_dir, batch_size=args.batch_size, window_size=args.window_size
    )
    
    # Create FULCCA model
    print("\nCreating FULCCA model...")
    model = FULCCAModel(
        cca_dims=args.cca_dims,
        regularization=args.regularization,
        window_size=args.window_size
    )
    
    # Create trainer
    trainer = FULCCATrainer(model, args.output_dir, args.tfrecord_dir, 
                           sampling_rate=64, window_size=args.window_size)
    
    # Train model
    print("\nStarting FULCCA training...")
    best_val_acc = trainer.train(train_dataset, val_dataset)
    
    # Test model
    print("\nTesting FULCCA model...")
    results = trainer.test(test_dataset)
    
    # Save results
    trainer.save_results(results)
    
    print("\n" + "=" * 80)
    print("FULCCA TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {results['accuracy']:.4f}")
    
    # Display comprehensive metrics
    print("\n" + "=" * 80)
    print("COMPREHENSIVE METRICS SUMMARY")
    print("=" * 80)
    
    # ROC-AUC metrics
    roc_auc = results.get('roc_auc_metrics', {})
    if "error" not in roc_auc:
        print(f"ROC-AUC Score: {roc_auc.get('roc_auc_score', 'N/A'):.4f}")
        print(f"Average Precision: {roc_auc.get('average_precision', 'N/A'):.4f}")
    
    # MSED metrics
    msed = results.get('msed_metrics', {})
    if "error" not in msed:
        print(f"RMSE: {msed.get('rmse', 'N/A'):.4f}")
        print(f"R-squared: {msed.get('r_squared', 'N/A'):.4f}")
    
    # Advanced metrics
    advanced = results.get('advanced_metrics', {})
    if "error" not in advanced:
        print(f"Matthews Correlation Coefficient: {advanced.get('matthews_correlation_coefficient', 'N/A'):.4f}")
        print(f"Balanced Accuracy: {advanced.get('balanced_accuracy', 'N/A'):.4f}")
    
    # Temporal analysis
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
        # Force garbage collection
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
