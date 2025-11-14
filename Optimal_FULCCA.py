#!/usr/bin/env python3
"""
FULCCA Optimal Configuration - opt_3 (70.00% accuracy)
Detailed metrics and temporal performance analysis.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
import seaborn as sns

# Add telluride_decoding to path
sys.path.append('/home/py9363/telluride_decoding')
from telluride_decoding.cca import BrainModelCCA, cca_pearson_correlation_first

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
try:
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        try:
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_limit(gpu, 8192)
        except (AttributeError, Exception):
            pass
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
except Exception:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.compat.v1.enable_v2_behavior()
device = tf.device('/GPU:0')
tf.random.set_seed(42)
np.random.seed(42)

# Optimal Configuration (opt_3)
OPTIMAL_CONFIG = {
    'name': 'opt_3',
    'cca_dims': 12,
    'regularization': 0.08,
    'window_size': 1280,  # 20.0 seconds
    'batch_size': 6
}

def run_optimal_fulcca_analysis(tfrecord_dir: str = "/home/py9363/telluride_decoding/fulsang_preprocessed/tfrecords", 
                               output_dir: str = "optimal_fulcca_results"):
    """
    Run optimal FULCCA analysis with detailed metrics and temporal performance.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    train_dataset, val_dataset, test_dataset = create_optimal_data_loaders(
        tfrecord_dir, 
        batch_size=OPTIMAL_CONFIG['batch_size'],
        window_size=OPTIMAL_CONFIG['window_size']
    )
    
    model = OptimalFULCCAModel(
        cca_dims=OPTIMAL_CONFIG['cca_dims'],
        regularization=OPTIMAL_CONFIG['regularization'],
        window_size=OPTIMAL_CONFIG['window_size']
    )
    
    trainer = OptimalFULCCATrainer(model, str(output_path))
    val_accuracy = trainer.train(train_dataset, val_dataset)
    results = trainer.test(test_dataset)
    detailed_metrics = calculate_detailed_metrics(results['predictions'], results['targets'])
    temporal_metrics = calculate_temporal_performance(tfrecord_dir, model)
    generate_comprehensive_report(results, detailed_metrics, temporal_metrics, val_accuracy, output_path)
    
    return results, detailed_metrics, temporal_metrics

def create_optimal_data_loaders(tfrecord_dir: str, batch_size: int = 6, 
                              window_size: int = 1280) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Create data loaders for optimal configuration."""
    from FULCCA import create_fulsang_data_loaders
    
    correct_tfrecord_dir = "/home/py9363/telluride_decoding/fulsang_preprocessed/tfrecords"
    train_dataset, val_dataset, test_dataset = create_fulsang_data_loaders(
        correct_tfrecord_dir, 
        batch_size=batch_size, 
        window_size=window_size,
        train_ratio=0.60,
        val_ratio=0.25
    )
    
    return train_dataset, val_dataset, test_dataset

class OptimalFULCCAModel:
    """
    Optimal FULCCA model with opt_3 configuration.
    """
    
    def __init__(self, cca_dims: int = 12, regularization: float = 0.08, window_size: int = 1280):
        self.cca_dims = cca_dims
        self.regularization = regularization
        self.window_size = window_size
        self.model = None
        self.is_fitted = False
    
    def fit(self, dataset: tf.data.Dataset):
        """Fit the optimal CCA model with class balancing."""
        try:
            class_weights = self._calculate_class_weights(dataset)
        except Exception:
            class_weights = {0: 1.0, 1: 1.0}
        
        self.model = self._create_optimal_cca_model(dataset)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=self._create_weighted_loss(class_weights),
            metrics=[cca_pearson_correlation_first]
        )
        self.model.fit(dataset, epochs=2)
        self.is_fitted = True
    
    def _create_optimal_cca_model(self, dataset: tf.data.Dataset):
        """Create optimal CCA model using telluride_decoding implementation."""
        with tf.device('/CPU:0'):
            cca_model = BrainModelCCA(
                input_dataset=dataset,
                cca_dims=self.cca_dims,
                regularization_lambda=self.regularization
            )
        return cca_model
    
    def _calculate_class_weights(self, dataset: tf.data.Dataset) -> Dict[int, float]:
        """Calculate class weights for imbalanced data."""
        all_labels = []
        for batch in dataset:
            if isinstance(batch, dict) and 'label' in batch:
                labels = batch['label'].numpy().flatten()
                all_labels.extend(labels)
            elif isinstance(batch, tuple) and len(batch) == 2:
                _, labels = batch
                all_labels.extend(labels.numpy().flatten())
        
        if not all_labels:
            return {0: 1.0, 1: 1.0}
        
        unique_classes, class_counts = np.unique(all_labels, return_counts=True)
        total_samples = len(all_labels)
        n_classes = len(unique_classes)
        
        class_weights = {}
        for i, class_id in enumerate(unique_classes):
            if class_counts[i] > 0:
                weight = total_samples / (n_classes * class_counts[i])
                class_weights[int(class_id)] = float(weight)
            else:
                class_weights[int(class_id)] = 1.0
        
        if 0 not in class_weights:
            class_weights[0] = 1.0
        if 1 not in class_weights:
            class_weights[1] = 1.0
            
        return class_weights
    
    def _create_weighted_loss(self, class_weights: Dict[int, float]):
        """Create weighted loss function for class balancing."""
        def weighted_binary_crossentropy_loss(y_true, y_pred):
            cca_width = y_pred.shape[-1] // 2
            pred1 = y_pred[:, :cca_width]
            cca_scores = pred1[:, 0]
            y_pred_prob = (tf.nn.tanh(cca_scores) + 1.0) / 2.0
            
            weights = tf.where(y_true == 0, 
                             tf.constant(class_weights[0], dtype=tf.float32),
                             tf.constant(class_weights[1], dtype=tf.float32))
            
            bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred_prob)
            weighted_loss = bce_loss * weights
            return tf.reduce_mean(weighted_loss)
        
        return weighted_binary_crossentropy_loss
    
    def predict(self, dataset: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Optimal prediction."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
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
                
                if isinstance(inputs, dict) and 'input_1' in inputs:
                    batch_size = inputs['input_1'].shape[0] // self.window_size
                else:
                    batch_size = binary_predictions.shape[0]
                
                if batch_size > 0 and binary_predictions.shape[0] % batch_size == 0:
                    pred_reshaped = tf.reshape(binary_predictions, (batch_size, -1))
                    window_size = pred_reshaped.shape[1]
                    threshold = tf.cast(window_size * 0.70, tf.int64)
                    prediction_counts = tf.reduce_sum(pred_reshaped, axis=1)
                    sample_predictions = tf.cast(prediction_counts > threshold, tf.int64)
                else:
                    sample_predictions = binary_predictions
                
                all_predictions.extend(sample_predictions.numpy())
                
                if targets is not None:
                    all_targets.extend(targets.numpy().flatten())
        
        return np.array(all_predictions), np.array(all_targets)

class OptimalFULCCATrainer:
    """Optimal trainer with comprehensive analysis."""
    
    def __init__(self, model: OptimalFULCCAModel, output_dir: str):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def train(self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset) -> float:
        """Train the optimal model."""
        self.model.fit(train_dataset)
        val_predictions, val_targets = self.model.predict(val_dataset)
        val_accuracy = accuracy_score(val_targets, val_predictions)
        return val_accuracy
    
    def test(self, test_dataset: tf.data.Dataset) -> Dict:
        """Test with comprehensive metrics."""
        predictions, targets = self.model.predict(test_dataset)
        accuracy = accuracy_score(targets, predictions)
        
        try:
            roc_auc = roc_auc_score(targets, predictions)
            avg_precision = average_precision_score(targets, predictions)
        except ValueError:
            roc_auc = 0.5
            avg_precision = 0.5
        
        mcc = matthews_corrcoef(targets, predictions)
        balanced_acc = balanced_accuracy_score(targets, predictions)
        
        results = {
            'accuracy': accuracy,
            'roc_auc_metrics': {'roc_auc_score': roc_auc, 'average_precision': avg_precision},
            'advanced_metrics': {
                'matthews_correlation_coefficient': mcc,
                'balanced_accuracy': balanced_acc
            },
            'predictions': predictions,
            'targets': targets
        }
        
        return results

def calculate_detailed_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict:
    """Calculate comprehensive detailed metrics."""
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='binary')
    recall = recall_score(targets, predictions, average='binary')
    f1 = f1_score(targets, predictions, average='binary')
    
    # Advanced metrics
    mcc = matthews_corrcoef(targets, predictions)
    balanced_acc = balanced_accuracy_score(targets, predictions)
    
    # ROC-AUC metrics
    try:
        roc_auc = roc_auc_score(targets, predictions)
        avg_precision = average_precision_score(targets, predictions)
    except ValueError:
        roc_auc = 0.5
        avg_precision = 0.5
    
    # Confusion matrix
    cm = confusion_matrix(targets, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    class_report = classification_report(targets, predictions, 
                                       target_names=['Left Attention', 'Right Attention'], 
                                       labels=[0, 1],
                                       output_dict=True)
    
    detailed_metrics = {
        'basic_metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'npv': npv,
            'ppv': ppv
        },
        'advanced_metrics': {
            'matthews_correlation_coefficient': mcc,
            'balanced_accuracy': balanced_acc,
            'roc_auc_score': roc_auc,
            'average_precision': avg_precision
        },
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        },
        'classification_report': class_report
    }
    
    return detailed_metrics

def calculate_temporal_performance(tfrecord_dir: str, base_model: OptimalFULCCAModel) -> Dict:
    """Calculate temporal performance across different window sizes."""
    from FULCCA import create_fulsang_data_loaders
    
    window_sizes = [256, 512, 1024, 1280, 1536, 2048]
    temporal_results = {}
    
    for window_size in window_sizes:
        try:
            train_dataset, val_dataset, test_dataset = create_fulsang_data_loaders(
                "/home/py9363/telluride_decoding/fulsang_preprocessed/tfrecords",
                batch_size=6, window_size=window_size,
                train_ratio=0.60,
                val_ratio=0.25
            )
            
            temp_model = OptimalFULCCAModel(
                cca_dims=OPTIMAL_CONFIG['cca_dims'],
                regularization=OPTIMAL_CONFIG['regularization'],
                window_size=window_size
            )
            
            temp_trainer = OptimalFULCCATrainer(temp_model, f"temp_{window_size}")
            val_accuracy = temp_trainer.train(train_dataset, val_dataset)
            results = temp_trainer.test(test_dataset)
            
            temp_accuracy = results['accuracy']
            temporal_results[f"{window_size/64:.1f}s_window"] = temp_accuracy
            temporal_results[f"{window_size/64:.1f}s_window_val"] = val_accuracy
            
            cleanup_gpu_memory()
        except Exception:
            temporal_results[f"{window_size/64:.1f}s_window"] = 0.0
    
    return temporal_results

def generate_comprehensive_report(results: Dict, detailed_metrics: Dict, temporal_metrics: Dict, 
                                val_accuracy: float, output_path: Path):
    """Generate comprehensive analysis report."""
    print(f"Configuration: {OPTIMAL_CONFIG['name']}")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    save_comprehensive_results(results, detailed_metrics, temporal_metrics, val_accuracy, output_path)

def save_comprehensive_results(results: Dict, detailed_metrics: Dict, temporal_metrics: Dict, 
                             val_accuracy: float, output_path: Path):
    """Save comprehensive results to files."""
    results_to_save = {
        'configuration': OPTIMAL_CONFIG,
        'validation_accuracy': val_accuracy,
        'test_accuracy': results['accuracy'],
        'roc_auc': detailed_metrics['advanced_metrics']['roc_auc_score'],
        'matthews_correlation': detailed_metrics['advanced_metrics']['matthews_correlation_coefficient'],
        'balanced_accuracy': detailed_metrics['advanced_metrics']['balanced_accuracy']
    }
    
    with open(output_path / "comprehensive_results.json", 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    with open(output_path / "detailed_metrics.json", 'w') as f:
        json.dump(detailed_metrics, f, indent=2)
    
    with open(output_path / "temporal_performance.json", 'w') as f:
        json.dump(temporal_metrics, f, indent=2)
    
    with open(output_path / "classification_report.txt", 'w') as f:
        f.write("FULCCA Optimal Configuration Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Configuration: {OPTIMAL_CONFIG['name']}\n")
        f.write(f"Test Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Validation Accuracy: {val_accuracy:.4f}\n\n")
        
        class_report = detailed_metrics['classification_report']
        f.write("Per-Class Metrics:\n")
        f.write(f"Left Attention (0):\n")
        f.write(f"  Precision: {class_report['0']['precision']:.4f}\n")
        f.write(f"  Recall: {class_report['0']['recall']:.4f}\n")
        f.write(f"  F1-Score: {class_report['0']['f1-score']:.4f}\n")
        f.write(f"Right Attention (1):\n")
        f.write(f"  Precision: {class_report['1']['precision']:.4f}\n")
        f.write(f"  Recall: {class_report['1']['recall']:.4f}\n")
        f.write(f"  F1-Score: {class_report['1']['f1-score']:.4f}\n")

def cleanup_gpu_memory():
    """Clean up GPU memory."""
    try:
        tf.keras.backend.clear_session()
        gc.collect()
    except Exception:
        pass

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FULCCA Optimal Configuration Analysis")
    parser.add_argument("--tfrecord_dir", type=str, default="/home/py9363/telluride_decoding/fulsang_preprocessed/tfrecords", help="Path to TFRecord directory")
    parser.add_argument("--output_dir", type=str, default="optimal_fulcca_results", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        results, detailed_metrics, temporal_metrics = run_optimal_fulcca_analysis(args.tfrecord_dir, args.output_dir)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        cleanup_gpu_memory()
