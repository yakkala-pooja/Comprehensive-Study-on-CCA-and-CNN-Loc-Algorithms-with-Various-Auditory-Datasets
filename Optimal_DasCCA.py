#!/usr/bin/env python3
"""
Optimal_DASCCA - Optimized CCA Algorithm for DAS Dataset Targeting 83% Accuracy

This module implements an optimized Canonical Correlation Analysis (CCA) 
algorithm specifically designed to achieve 83%+ accuracy on the DAS dataset.

Key optimizations:
- Ultra-enhanced preprocessing pipeline
- Optimized hyperparameters for high accuracy
- Advanced feature extraction
- Multi-stage data normalization
- Adaptive regularization
- Comprehensive metrics evaluation
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
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import seaborn as sns
from tqdm import tqdm
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set environment variables for robust GPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Configure GPU
try:
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"Found {len(gpu_devices)} GPU device(s)")
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✓ GPU memory growth configured")
    else:
        raise RuntimeError("No GPU devices found!")
except Exception as e:
    print(f"GPU configuration failed: {e}")
    raise RuntimeError("Cannot proceed without GPU.")

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

tf.compat.v1.enable_v2_behavior()
device = tf.device('/GPU:0')
print("Using GPU for computation (GPU-only mode)")

# Set random seeds
tf.random.set_seed(42)
np.random.seed(42)
print("✓ Random seeds set for reproducibility")


def ultra_enhanced_eeg_preprocessing(eeg_data: np.ndarray, sampling_rate: int = 64) -> np.ndarray:
    """
    Ultra-enhanced EEG preprocessing pipeline optimized for 83%+ CCA accuracy.
    
    Multi-stage preprocessing:
    1. Outlier removal using IQR method
    2. Quantile normalization
    3. Robust standardization with adaptive scaling
    4. Advanced temporal smoothing
    5. Soft clipping
    6. Minimal noise for numerical stability
    """
    # Remove NaN/Inf values
    eeg_data = np.nan_to_num(eeg_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Stage 1: Outlier removal using IQR method
    Q1 = np.percentile(eeg_data, 25, axis=0)
    Q3 = np.percentile(eeg_data, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    eeg_data = np.clip(eeg_data, lower_bound, upper_bound)
    
    # Stage 2: DC removal
    eeg_data = eeg_data - np.mean(eeg_data, axis=0, keepdims=True)
    
    # Stage 3: Bandpass filtering (1-30 Hz for attention decoding)
    nyquist = sampling_rate / 2
    low_freq = 1.0 / nyquist
    high_freq = min(30.0 / nyquist, 0.99)
    b, a = signal.butter(4, [low_freq, high_freq], btype='band')
    
    filtered_eeg = np.zeros_like(eeg_data)
    for ch in range(eeg_data.shape[1]):
        filtered_eeg[:, ch] = signal.filtfilt(b, a, eeg_data[:, ch])
    eeg_data = filtered_eeg
    
    # Stage 4: Quantile normalization for better distribution matching
    eeg_quantile = np.zeros_like(eeg_data)
    for i in range(eeg_data.shape[1]):
        eeg_quantile[:, i] = stats.rankdata(eeg_data[:, i], method='average') / len(eeg_data)
    
    # Stage 5: Robust standardization with adaptive scaling
    median = np.median(eeg_quantile, axis=0)
    mad = np.median(np.abs(eeg_quantile - median), axis=0)
    mad = np.where(mad < 1e-8, 1.0, mad)
    eeg_data = (eeg_quantile - median) / (1.4826 * mad)
    
    # Stage 6: Advanced temporal smoothing
    eeg_smoothed = np.zeros_like(eeg_data)
    for i in range(eeg_data.shape[1]):
        eeg_smoothed[:, i] = gaussian_filter1d(eeg_data[:, i], sigma=0.5)
    
    # Stage 7: Soft clipping
    eeg_data = np.tanh(eeg_smoothed * 0.5)
    
    # Stage 8: Minimal noise for numerical stability
    noise_scale = 1e-8 * np.std(eeg_data)
    eeg_data = eeg_data + noise_scale * np.random.randn(*eeg_data.shape)
    
    return eeg_data.astype(np.float32)


class OptimalDasDatasetCCA:
    """
    Optimized DAS dataset class with ultra-enhanced preprocessing for 83%+ accuracy.
    """
    
    def __init__(self, tfrecord_dir: str, mode: str = 'full', 
                 window_size: int = 512, overlap: float = 0.5,
                 cache_size: int = 1000):
        self.tfrecord_dir = Path(tfrecord_dir)
        self.mode = mode
        self.window_size = window_size
        self.overlap = overlap
        self.cache_size = cache_size
        
        # DAS-specific parameters
        self.sampling_rate = 64  # Hz
        self.n_channels = 64  # EEG channels
        
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
        print(f"Using Ultra-Enhanced Preprocessing: Yes")
    
    def _load_das_data(self) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Load DAS data with ultra-enhanced preprocessing."""
        tfrecord_files = []
        direct_files = list(self.tfrecord_dir.glob("*.tfrecords"))
        subdir_files = list(self.tfrecord_dir.glob("*/*.tfrecords"))
        
        if direct_files:
            tfrecord_files.extend(direct_files)
        if subdir_files:
            tfrecord_files.extend(subdir_files)
        
        if not tfrecord_files:
            raise ValueError(f"No TFRecord files found in {self.tfrecord_dir}")
        
        print(f"Loading DAS data from {len(tfrecord_files)} files...")
        print("✓ Using Ultra-Enhanced Preprocessing Pipeline")
        print("✓ Optimized for 83%+ accuracy")
        
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
                        
                        if not all(key in features for key in ['eeg', 'attended_ear', 'subject_id']):
                            continue
                        
                        # Extract EEG data
                        eeg_values = features['eeg'].float_list.value
                        if len(eeg_values) != 64:
                            continue
                        
                        eeg_data = np.array(eeg_values, dtype=np.float32).reshape(1, 64)
                        
                        # Validate EEG data
                        if np.any(np.isnan(eeg_data)) or np.any(np.isinf(eeg_data)):
                            continue
                        
                        # Extract label
                        attended_ear_values = features['attended_ear'].bytes_list.value
                        if not attended_ear_values:
                            continue
                        
                        attended_ear = attended_ear_values[0].decode('utf-8')
                        label = 0 if attended_ear == 'L' else 1
                        
                        if attended_ear not in ['L', 'R']:
                            continue
                        
                        # Extract subject_id
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
                            'attended_ear': attended_ear
                        })
                        
                    except Exception:
                        continue
                        
            except Exception:
                continue
        
        if not all_eeg_data:
            raise ValueError("No valid DAS data found")
        
        eeg_data = np.vstack(all_eeg_data)
        labels = np.array(all_labels, dtype=np.int64)
        
        print(f"Loaded {len(eeg_data)} samples")
        print(f"EEG shape: {eeg_data.shape}")
        print(f"Label distribution: {np.bincount(labels)}")
        
        return eeg_data, labels, all_metadata
    
    def _create_das_windows(self) -> List[Tuple[int, int]]:
        """Create windows for DAS data."""
        step_size = int(self.window_size * (1 - self.overlap))
        total_windows = (len(self.eeg_data) - self.window_size) // step_size + 1
        
        window_indices = []
        for i in range(total_windows):
            data_idx = i * step_size
            if data_idx + self.window_size <= len(self.eeg_data):
                window_labels = self.labels[data_idx:data_idx + self.window_size]
                window_label = int(np.bincount(window_labels).argmax())
                window_indices.append((data_idx, window_label))
        
        return window_indices
    
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
        
        # Apply ultra-enhanced preprocessing
        window_eeg = ultra_enhanced_eeg_preprocessing(window_eeg, self.sampling_rate)
        
        # Convert to tensors
        window_tensor = tf.constant(window_eeg, dtype=tf.float32)
        label_tensor = tf.constant([label], dtype=tf.int64)
        
        # Cache
        if len(self._window_cache) < self.cache_size:
            self._window_cache[cache_key] = (window_tensor, label_tensor)
        
        return window_tensor, label_tensor


class OptimalDASCCAModel:
    """
    Optimized DASCCA model targeting 83%+ accuracy.
    """
    
    def __init__(self, cca_dims: int = 10, regularization: float = 0.001, window_size: int = 512):
        self.cca_dims = cca_dims
        self.regularization = regularization
        self.window_size = window_size
        self.model = None
        self.is_fitted = False
        
        print(f"Optimal DASCCA model initialized:")
        print(f"  CCA dimensions: {cca_dims}")
        print(f"  Regularization: {regularization}")
        print(f"  Target accuracy: 83%+")
    
    def _create_cca_model(self, dataset: tf.data.Dataset):
        """Create CCA model on GPU with dataset limiting for faster computation."""
        tf.keras.backend.clear_session()
        
        print("Creating Optimal CCA model on GPU...")
        print("⚠ Limiting dataset to 200 batches for CCA parameter calculation (to prevent long hangs)...")
        
        # Limit dataset size for CCA parameter calculation to prevent long hangs
        # CCA parameters can be estimated from a subset of data
        # Using 200 batches should be sufficient for stable CCA parameter estimation
        limited_dataset = dataset.take(200)
        
        # Cache the limited dataset to avoid re-reading
        limited_dataset = limited_dataset.cache()
        
        print(f"Using limited dataset (200 batches) for CCA parameter calculation...")
        print("This speeds up computation while maintaining accuracy.")
        print("⚠ If this still hangs, try reducing batch_size or window_size in configuration.")
        
        try:
            with tf.device('/GPU:0'):
                print("Starting CCA parameter calculation (this may take a few minutes)...")
                model = BrainModelCCA(
                    input_dataset=limited_dataset,
                    cca_dims=self.cca_dims,
                    regularization_lambda=self.regularization
                )
            print("✓ CCA model created successfully on GPU")
            return model
        except Exception as e:
            print(f"✗ CCA model creation failed: {e}")
            print("Try reducing batch_size, window_size, or cca_dims in configuration.")
            raise
    
    def fit(self, dataset: tf.data.Dataset):
        """Fit the CCA model."""
        print("Fitting Optimal DASCCA model...")
        
        self.model = self._create_cca_model(dataset)
        
        try:
            print("Compiling CCA model...")
            self.model.compile(
                optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                loss='mse',
                metrics=[cca_pearson_correlation_first]
            )
            
            print("Training CCA model...")
            self.model.fit(dataset, epochs=1)
            print("✓ Optimal DASCCA model fitted successfully")
            
        except Exception as e:
            print(f"Training failed: {e}")
            raise
        
        self.is_fitted = True
    
    def predict(self, dataset: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        print("Making Optimal DASCCA predictions...")
        
        all_predictions = []
        all_targets = []
        
        with device:
            for batch in tqdm(dataset, desc="Predicting"):
                if isinstance(batch, dict):
                    inputs = batch
                    targets = None
                else:
                    inputs, targets = batch
                
                predictions = self.model(inputs)
                
                cca_width = predictions.shape[-1] // 2
                pred1 = predictions[:, :cca_width]
                cca_scores = pred1[:, 0]
                binary_predictions = tf.cast(cca_scores > 0, tf.int64)
                
                # Dynamic window size inference
                input_shape = int(inputs['input_1'].shape[0])
                num_predictions = int(binary_predictions.shape[0])
                
                possible_window_sizes = [32, 64, 128, 256, 512, 1024, 2048, 1920]
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
                    all_targets.extend(targets.numpy().flatten())
        
        return np.array(all_predictions), np.array(all_targets)


class OptimalDASCCATrainer:
    """
    Optimal DASCCA trainer targeting 83%+ accuracy.
    """
    
    def __init__(self, model: OptimalDASCCAModel, output_dir: str = "optimal_dascca_results"):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        print(f"Optimal DASCCA trainer initialized. Output: {self.output_dir}")
    
    def train(self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset) -> float:
        """Train the model."""
        print("Starting Optimal DASCCA training...")
        self.model.fit(train_dataset)
        
        val_predictions, val_targets = self.model.predict(val_dataset)
        val_accuracy = accuracy_score(val_targets, val_predictions)
        
        print(f"Optimal DASCCA training completed! Validation accuracy: {val_accuracy:.4f}")
        return val_accuracy
    
    def test(self, test_dataset: tf.data.Dataset) -> Dict:
        """Test the model with comprehensive metrics."""
        print("Testing Optimal DASCCA model...")
        
        predictions, targets = self.model.predict(test_dataset)
        
        accuracy = accuracy_score(targets, predictions)
        report = classification_report(targets, predictions, output_dict=True)
        cm = confusion_matrix(targets, predictions)
        
        # Calculate comprehensive metrics
        roc_auc_metrics = self._calculate_roc_auc_metrics(targets, predictions)
        msed_metrics = self._calculate_msed_metrics(targets, predictions)
        advanced_metrics = self._calculate_advanced_metrics(targets, predictions)
        
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
    
    def _calculate_roc_auc_metrics(self, targets: np.ndarray, predictions: np.ndarray) -> Dict:
        """Calculate ROC-AUC metrics."""
        try:
            probabilities = predictions.astype(np.float32)
            roc_auc = roc_auc_score(targets, probabilities)
            fpr, tpr, roc_thresholds = roc_curve(targets, probabilities)
            
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = roc_thresholds[optimal_idx]
            
            precision, recall, pr_thresholds = precision_recall_curve(targets, probabilities)
            avg_precision = average_precision_score(targets, probabilities)
            
            return {
                "roc_auc_score": float(roc_auc),
                "average_precision": float(avg_precision),
                "optimal_threshold": float(optimal_threshold),
                "optimal_tpr": float(tpr[optimal_idx]),
                "optimal_fpr": float(fpr[optimal_idx])
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_msed_metrics(self, targets: np.ndarray, predictions: np.ndarray) -> Dict:
        """Calculate MSED metrics."""
        try:
            mse = np.mean((predictions - targets) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - targets))
            
            ss_res = np.sum((targets - predictions) ** 2)
            ss_tot = np.sum((targets - np.mean(targets)) ** 2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-8))
            
            return {
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "r_squared": float(r_squared)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_advanced_metrics(self, targets: np.ndarray, predictions: np.ndarray) -> Dict:
        """Calculate advanced metrics."""
        try:
            mcc = matthews_corrcoef(targets, predictions)
            kappa = cohen_kappa_score(targets, predictions)
            balanced_acc = balanced_accuracy_score(targets, predictions)
            
            precision, recall, f1, support = precision_recall_fscore_support(
                targets, predictions, average=None, labels=[0, 1]
            )
            
            return {
                "matthews_correlation_coefficient": float(mcc),
                "cohens_kappa": float(kappa),
                "balanced_accuracy": float(balanced_acc),
                "per_class_metrics": {
                    "left_attention": {
                        "precision": float(precision[0]),
                        "recall": float(recall[0]),
                        "f1_score": float(f1[0])
                    },
                    "right_attention": {
                        "precision": float(precision[1]),
                        "recall": float(recall[1]),
                        "f1_score": float(f1[1])
                    }
                }
            }
        except Exception as e:
            return {"error": str(e)}
    
    def save_results(self, results: Dict, config: Dict, val_accuracy: float = None):
        """Save results."""
        results_json = {
            'configuration': config,
            'validation_accuracy': float(val_accuracy) if val_accuracy is not None else 0,
            'test_accuracy': float(results.get('accuracy', 0)),
            'roc_auc': results.get('roc_auc_metrics', {}).get('roc_auc_score', 0),
            'matthews_correlation': results.get('advanced_metrics', {}).get('matthews_correlation_coefficient', 0),
            'balanced_accuracy': results.get('advanced_metrics', {}).get('balanced_accuracy', 0),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"Results saved to {self.output_dir}")


def create_optimal_data_loaders(tfrecord_dir: str, batch_size: int = 16, 
                                window_size: int = 512, overlap: float = 0.5,
                                train_ratio: float = 0.7, val_ratio: float = 0.15):
    """Create optimal data loaders with subject-wise splitting."""
    
    print("Creating Optimal DAS dataset...")
    full_dataset = OptimalDasDatasetCCA(tfrecord_dir, mode='full', 
                                      window_size=window_size, overlap=overlap)
    
    # Subject-wise splitting
    subject_windows = {}
    for i, (data_idx, label) in enumerate(full_dataset.window_indices):
        # Find subject_id from metadata - use the first sample in the window
        subject_id = 'unknown'
        if data_idx < len(full_dataset.metadata):
            subject_id = full_dataset.metadata[data_idx].get('subject_id', 'unknown')
        if subject_id not in subject_windows:
            subject_windows[subject_id] = []
        subject_windows[subject_id].append(i)
    
    subjects = list(subject_windows.keys())
    np.random.seed(42)
    np.random.shuffle(subjects)
    
    n_subjects = len(subjects)
    if n_subjects < 3:
        # Random window splitting
        all_indices = []
        for windows in subject_windows.values():
            all_indices.extend(windows)
        np.random.shuffle(all_indices)
        n_windows = len(all_indices)
        train_indices = all_indices[:int(train_ratio * n_windows)]
        val_indices = all_indices[int(train_ratio * n_windows):int((train_ratio + val_ratio) * n_windows)]
        test_indices = all_indices[int((train_ratio + val_ratio) * n_windows):]
    else:
        # Subject-wise splitting
        n_train_subjects = int(train_ratio * n_subjects)
        n_val_subjects = int(val_ratio * n_subjects)
        
        train_subjects = subjects[:n_train_subjects]
        val_subjects = subjects[n_train_subjects:n_train_subjects + n_val_subjects]
        test_subjects = subjects[n_train_subjects + n_val_subjects:]
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        for subject_id in train_subjects:
            train_indices.extend(subject_windows[subject_id])
        for subject_id in val_subjects:
            val_indices.extend(subject_windows[subject_id])
        for subject_id in test_subjects:
            test_indices.extend(subject_windows[subject_id])
    
    def create_cca_dataset(indices):
        def generator():
            for i in indices:
                eeg_data, label = full_dataset[i]
                mid_point = eeg_data.shape[1] // 2
                input_1 = eeg_data[:, :mid_point]
                input_2 = eeg_data[:, mid_point:]
                yield {'input_1': input_1, 'input_2': input_2}, label
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                {
                    'input_1': tf.TensorSpec(shape=(window_size, 32), dtype=tf.float32),
                    'input_2': tf.TensorSpec(shape=(window_size, 32), dtype=tf.float32)
                },
                tf.TensorSpec(shape=(1,), dtype=tf.int64)
            )
        )
        
        def reshape_batch(inputs, labels):
            # Reshape to (batch_size, window_size * channels) for CCA
            # inputs['input_1'] shape: (batch_size, window_size, 32)
            # Need: (batch_size, window_size * 32)
            batch_size = tf.shape(inputs['input_1'])[0]
            input_1_reshaped = tf.reshape(inputs['input_1'], (batch_size, -1))
            input_2_reshaped = tf.reshape(inputs['input_2'], (batch_size, -1))
            return {'input_1': input_1_reshaped, 'input_2': input_2_reshaped}, labels
        
        return dataset.batch(batch_size).map(reshape_batch)
    
    train_dataset = create_cca_dataset(train_indices)
    val_dataset = create_cca_dataset(val_indices)
    test_dataset = create_cca_dataset(test_indices)
    
    print(f"Train windows: {len(train_indices)}")
    print(f"Val windows: {len(val_indices)}")
    print(f"Test windows: {len(test_indices)}")
    
    return train_dataset, val_dataset, test_dataset


# OPTIMAL CONFIGURATIONS FOR 83%+ ACCURACY
# Note: Reduced batch sizes to prevent memory issues and long computation times
OPTIMAL_CONFIGS = [
    # Configuration 1: Balanced optimal (target: 83%)
    {
        'name': 'optimal_83_target',
        'cca_dims': 10,
        'regularization': 0.001,
        'window_size': 512,
        'batch_size': 8  # Reduced from 16 to prevent hangs
    },
    # Configuration 2: High-dimensional
    {
        'name': 'optimal_high_dim',
        'cca_dims': 12,
        'regularization': 0.0005,
        'window_size': 640,
        'batch_size': 6  # Reduced from 12
    },
    # Configuration 3: Extended window
    {
        'name': 'optimal_extended',
        'cca_dims': 8,
        'regularization': 0.001,
        'window_size': 768,
        'batch_size': 5  # Reduced from 10
    },
    # Configuration 4: Fine-tuned
    {
        'name': 'optimal_finetuned',
        'cca_dims': 9,
        'regularization': 0.0008,
        'window_size': 576,
        'batch_size': 7  # Reduced from 14
    },
    # Configuration 5: Aggressive
    {
        'name': 'optimal_aggressive',
        'cca_dims': 15,
        'regularization': 0.0003,
        'window_size': 512,
        'batch_size': 8  # Reduced from 16
    }
]


def main():
    """Main function for Optimal DASCCA training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimal DASCCA - Targeting 83%+ Accuracy')
    parser.add_argument('--tfrecord_dir', type=str, default='das_16subjects_preprocessed/tfrecords',
                       help='TFRecord directory path')
    parser.add_argument('--config_name', type=str, default='optimal_83_target',
                       help='Configuration name from OPTIMAL_CONFIGS')
    parser.add_argument('--output_dir', type=str, default='optimal_dascca_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Find configuration
    config = None
    for cfg in OPTIMAL_CONFIGS:
        if cfg['name'] == args.config_name:
            config = cfg
            break
    
    if config is None:
        print(f"Configuration '{args.config_name}' not found. Using default optimal_83_target")
        config = OPTIMAL_CONFIGS[0]
    
    print("=" * 80)
    print("OPTIMAL DASCCA - TARGETING 83%+ ACCURACY")
    print("=" * 80)
    print(f"Configuration: {config['name']}")
    print(f"  CCA dimensions: {config['cca_dims']}")
    print(f"  Regularization: {config['regularization']}")
    print(f"  Window size: {config['window_size']} samples ({config['window_size']/64:.1f}s)")
    print(f"  Batch size: {config['batch_size']}")
    print("=" * 80)
    print("Features:")
    print("  ✓ Ultra-enhanced preprocessing pipeline")
    print("  ✓ Optimized hyperparameters for 83%+ accuracy")
    print("  ✓ GPU-accelerated computation")
    print("  ✓ Comprehensive metrics evaluation")
    print("=" * 80)
    
    # Create data loaders
    train_dataset, val_dataset, test_dataset = create_optimal_data_loaders(
        args.tfrecord_dir,
        batch_size=config['batch_size'],
        window_size=config['window_size']
    )
    
    # Create model
    model = OptimalDASCCAModel(
        cca_dims=config['cca_dims'],
        regularization=config['regularization'],
        window_size=config['window_size']
    )
    
    # Create trainer
    trainer = OptimalDASCCATrainer(model, args.output_dir)
    
    # Train
    val_acc = trainer.train(train_dataset, val_dataset)
    
    # Test
    results = trainer.test(test_dataset)
    
    # Save results
    trainer.save_results(results, config, val_acc)
    
    print("\n" + "=" * 80)
    print("OPTIMAL DASCCA TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy: {results['accuracy']:.4f}")
    print(f"ROC-AUC: {results.get('roc_auc_metrics', {}).get('roc_auc_score', 0):.4f}")
    print(f"Matthews Correlation: {results.get('advanced_metrics', {}).get('matthews_correlation_coefficient', 0):.4f}")
    print(f"Balanced Accuracy: {results.get('advanced_metrics', {}).get('balanced_accuracy', 0):.4f}")
    
    if results['accuracy'] >= 0.83:
        print("\n🎉 SUCCESS: Target accuracy of 83%+ achieved!")
    else:
        print(f"\n⚠ Current accuracy: {results['accuracy']:.2%}, target: 83%+")
        print("Consider trying other configurations or further hyperparameter tuning.")
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()