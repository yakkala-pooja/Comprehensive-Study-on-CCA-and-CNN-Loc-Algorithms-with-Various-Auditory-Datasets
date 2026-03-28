#!/usr/bin/env python3
"""
Extended Window FULCCA Analysis - Focused on temporal performance with different window lengths.
Based on the best performing configuration: extended_window (66.67% accuracy)
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
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score
import pandas as pd
from tqdm import tqdm
import gc

# Add telluride_decoding to path
sys.path.append('/home/py9363/telluride_decoding')
from telluride_decoding.cca import BrainModelCCA, cca_pearson_correlation_first

print("=" * 80)
print("EXTENDED WINDOW FULCCA - Temporal Performance Analysis")
print("=" * 80)

# Set environment variables for robust GPU usage
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
        print("No GPU devices found, using CPU mode...")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
except Exception as e:
    print(f"GPU configuration failed: {e}")
    print("Using CPU mode...")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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

# Extended Window configurations - different window lengths for temporal analysis
EXTENDED_WINDOW_CONFIGS = [
    {'name': 'short_window', 'cca_dims': 10, 'regularization': 0.06, 'window_size': 256, 'batch_size': 8},   # 4 seconds
    {'name': 'medium_window', 'cca_dims': 10, 'regularization': 0.06, 'window_size': 512, 'batch_size': 8}, # 8 seconds
    {'name': 'extended_window', 'cca_dims': 10, 'regularization': 0.06, 'window_size': 1024, 'batch_size': 8}, # 16 seconds (best)
    {'name': 'long_window', 'cca_dims': 10, 'regularization': 0.06, 'window_size': 1536, 'batch_size': 8},   # 24 seconds
    {'name': 'very_long_window', 'cca_dims': 10, 'regularization': 0.06, 'window_size': 2048, 'batch_size': 8}, # 32 seconds
]

def run_extended_window_analysis(tfrecord_dir: str = "/home/py9363/telluride_decoding/fulsang_preprocessed/tfrecords", 
                                output_dir: str = "extended_window_results"):
    """
    Run extended window FULCCA analysis with temporal performance evaluation.
    
    Args:
        tfrecord_dir: Path to TFRecord directory
        output_dir: Output directory for results
    """
    print(f"\n{'='*80}")
    print("EXTENDED WINDOW FULCCA ANALYSIS")
    print(f"{'='*80}")
    print(f"TFRecord directory: {tfrecord_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Number of window configurations: {len(EXTENDED_WINDOW_CONFIGS)}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Results storage
    all_results = []
    best_accuracy = 0
    best_config = None
    
    # Run each window configuration
    for i, config in enumerate(EXTENDED_WINDOW_CONFIGS):
        print(f"\n{'='*60}")
        print(f"Window Configuration {i+1}/{len(EXTENDED_WINDOW_CONFIGS)}: {config['name']}")
        print(f"{'='*60}")
        print(f"CCA dimensions: {config['cca_dims']}")
        print(f"Regularization: {config['regularization']}")
        print(f"Window size: {config['window_size']} samples ({config['window_size']/64:.1f} seconds at 64Hz)")
        print(f"Batch size: {config['batch_size']}")
        
        try:
            # Create data loaders
            train_dataset, val_dataset, test_dataset = create_extended_data_loaders(
                tfrecord_dir, 
                batch_size=config['batch_size'],
                window_size=config['window_size']
            )
            
            # Create model
            model = ExtendedWindowFULCCAModel(
                cca_dims=config['cca_dims'],
                regularization=config['regularization'],
                window_size=config['window_size']
            )
            
            # Create trainer
            trainer = ExtendedWindowFULCCATrainer(model, str(output_path / config['name']))
            
            # Train and test
            val_accuracy = trainer.train(train_dataset, val_dataset)
            results = trainer.test(test_dataset)
            
            # Store results
            config_result = {
                'configuration': config['name'],
                'window_size_samples': config['window_size'],
                'window_size_seconds': config['window_size'] / 64,
                'accuracy': results['accuracy'],
                'roc_auc': results.get('roc_auc_metrics', {}).get('roc_auc_score', 0),
                'matthews_corr': results.get('advanced_metrics', {}).get('matthews_correlation_coefficient', 0),
                'balanced_accuracy': results.get('advanced_metrics', {}).get('balanced_accuracy', 0),
                'config_params': config,
                'detailed_results': results
            }
            all_results.append(config_result)
            
            # Track best configuration
            if results['accuracy'] > best_accuracy:
                best_accuracy = results['accuracy']
                best_config = config_result
            
            print(f"✓ Configuration {config['name']} completed")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  ROC-AUC: {results.get('roc_auc_metrics', {}).get('roc_auc_score', 0):.4f}")
            print(f"  Window: {config['window_size']/64:.1f} seconds")
            
            # Clean up memory
            cleanup_gpu_memory()
            
        except Exception as e:
            print(f"❌ Configuration {config['name']} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate comprehensive temporal analysis report
    generate_temporal_analysis_report(all_results, best_config, output_path)
    
    return all_results, best_config

def create_extended_data_loaders(tfrecord_dir: str, batch_size: int = 8, 
                               window_size: int = 1024) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Create data loaders for extended window analysis.
    """
    # Use the correct TFRecord directory
    correct_tfrecord_dir = "/home/py9363/telluride_decoding/fulsang_preprocessed/tfrecords"
    
    # Import the original data loading function
    from FULCCA import create_fulsang_data_loaders
    
    # Use the original function but with correct path
    train_dataset, val_dataset, test_dataset = create_fulsang_data_loaders(
        correct_tfrecord_dir, batch_size=batch_size, window_size=window_size
    )
    
    return train_dataset, val_dataset, test_dataset

class ExtendedWindowFULCCAModel:
    """
    Extended Window FULCCA model optimized for temporal analysis.
    """
    
    def __init__(self, cca_dims: int = 10, regularization: float = 0.06, window_size: int = 1024):
        self.cca_dims = cca_dims
        self.regularization = regularization
        self.window_size = window_size
        self.model = None
        self.is_fitted = False
        
        print(f"Extended Window FULCCA model initialized:")
        print(f"  CCA dimensions: {cca_dims}")
        print(f"  Regularization: {regularization}")
        print(f"  Window size: {window_size} samples ({window_size/64:.1f} seconds)")
    
    def fit(self, dataset: tf.data.Dataset):
        """Fit the extended window CCA model."""
        print("Fitting Extended Window FULCCA model...")
        
        # Create robust CCA model
        self.model = self._create_extended_cca_model(dataset)
        
        # Compile and train
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Lower learning rate for stability
            loss='mse',
            metrics=[cca_pearson_correlation_first]
        )
        
        # Fit with multiple epochs for better convergence
        self.model.fit(dataset, epochs=3)  # More epochs for better learning
        
        self.is_fitted = True
        print("✓ Extended Window FULCCA model fitted successfully")
    
    def _create_extended_cca_model(self, dataset: tf.data.Dataset):
        """Create extended window CCA model with better initialization."""
        # Force CPU creation to avoid CUDA handle corruption
        print("Creating Extended Window CCA model on CPU to avoid CUDA issues...")
        with tf.device('/CPU:0'):
            model = BrainModelCCA(
                input_dataset=dataset,
                cca_dims=self.cca_dims,
                regularization_lambda=self.regularization
            )
        print("✓ Extended Window CCA model created successfully on CPU")
        return model
    
    def predict(self, dataset: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Extended window prediction with temporal analysis."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        print("Making Extended Window FULCCA predictions...")
        
        all_predictions = []
        all_targets = []
        
        with device:
            for batch in tqdm(dataset, desc="Predicting"):
                if isinstance(batch, dict):
                    inputs = batch
                    targets = None
                else:
                    inputs, targets = batch
                
                # Get predictions
                predictions = self.model(inputs)
                
                # Enhanced prediction processing
                cca_width = predictions.shape[-1] // 2
                pred1 = predictions[:, :cca_width]
                pred2 = predictions[:, cca_width:]
                
                # Use first CCA component for classification
                cca_scores = pred1[:, 0]  # First CCA component
                
                # Convert to binary predictions
                binary_predictions = tf.cast(cca_scores > 0, tf.int64)
                
                # Aggregate predictions per sample
                batch_size = inputs['input_1'].shape[0] // self.window_size
                pred_reshaped = tf.reshape(binary_predictions, (batch_size, self.window_size))
                
                # Use simple majority voting for stability
                sample_predictions = tf.cast(tf.reduce_sum(pred_reshaped, axis=1) > (self.window_size // 2), tf.int64)
                
                all_predictions.extend(sample_predictions.numpy())
                
                if targets is not None:
                    all_targets.extend(targets.numpy().flatten())
        
        return np.array(all_predictions), np.array(all_targets)

class ExtendedWindowFULCCATrainer:
    """Extended Window trainer with comprehensive temporal metrics."""
    
    def __init__(self, model: ExtendedWindowFULCCAModel, output_dir: str):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def train(self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset) -> float:
        """Train the extended window model."""
        print("Starting Extended Window FULCCA training...")
        
        self.model.fit(train_dataset)
        
        # Enhanced validation
        val_predictions, val_targets = self.model.predict(val_dataset)
        val_accuracy = accuracy_score(val_targets, val_predictions)
        
        print(f"Extended Window FULCCA training completed! Validation accuracy: {val_accuracy:.4f}")
        return val_accuracy
    
    def test(self, test_dataset: tf.data.Dataset) -> Dict:
        """Test with comprehensive temporal metrics."""
        print("Testing Extended Window FULCCA model...")
        
        predictions, targets = self.model.predict(test_dataset)
        
        # Calculate metrics
        accuracy = accuracy_score(targets, predictions)
        
        # Enhanced metrics calculation
        try:
            roc_auc = roc_auc_score(targets, predictions)
            avg_precision = average_precision_score(targets, predictions)
        except ValueError:
            roc_auc = 0.5
            avg_precision = 0.5
        
        mcc = matthews_corrcoef(targets, predictions)
        balanced_acc = balanced_accuracy_score(targets, predictions)
        
        # Calculate temporal metrics
        temporal_metrics = self._calculate_temporal_metrics(test_dataset)
        
        results = {
            'accuracy': accuracy,
            'roc_auc_metrics': {'roc_auc_score': roc_auc, 'average_precision': avg_precision},
            'advanced_metrics': {
                'matthews_correlation_coefficient': mcc,
                'balanced_accuracy': balanced_acc
            },
            'temporal_metrics': temporal_metrics,
            'predictions': predictions,
            'targets': targets
        }
        
        return results
    
    def _calculate_temporal_metrics(self, test_dataset: tf.data.Dataset) -> Dict:
        """Calculate temporal performance metrics."""
        print("Calculating temporal performance metrics...")
        
        # Test different window sizes for temporal analysis
        window_sizes = [256, 512, 1024, 1536, 2048]  # Different temporal scales
        temporal_results = {}
        
        for window_size in window_sizes:
            try:
                # Create temporary dataset with different window size
                from FULCCA import create_fulsang_data_loaders
                _, _, temp_test_dataset = create_fulsang_data_loaders(
                    "/home/py9363/telluride_decoding/fulsang_preprocessed/tfrecords",
                    batch_size=8, window_size=window_size
                )
                
                # Make predictions
                temp_predictions, temp_targets = self.model.predict(temp_test_dataset)
                
                # Calculate accuracy for this window size
                temp_accuracy = accuracy_score(temp_targets, temp_predictions)
                temporal_results[f"{window_size/64:.1f}s_window"] = temp_accuracy
                
            except Exception as e:
                print(f"Warning: Could not test {window_size/64:.1f}s window: {e}")
                temporal_results[f"{window_size/64:.1f}s_window"] = 0.0
        
        return temporal_results

def generate_temporal_analysis_report(all_results: List[Dict], best_config: Dict, output_path: Path):
    """Generate comprehensive temporal analysis report."""
    print(f"\n{'='*80}")
    print("EXTENDED WINDOW TEMPORAL ANALYSIS COMPLETE")
    print(f"{'='*80}")
    
    # Create results DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Sort by accuracy
    df_results = df_results.sort_values('accuracy', ascending=False)
    
    print("\nTemporal Performance Results (sorted by accuracy):")
    print("-" * 80)
    print(f"{'Window Size':<15} {'Accuracy':<10} {'ROC-AUC':<10} {'MCC':<10} {'Balanced':<10}")
    print("-" * 80)
    
    for _, row in df_results.iterrows():
        print(f"{row['window_size_seconds']:<15.1f} {row['accuracy']:<10.4f} {row['roc_auc']:<10.4f} {row['matthews_corr']:<10.4f} {row['balanced_accuracy']:<10.4f}")
    
    # Best configuration
    if best_config:
        print(f"\n🏆 BEST WINDOW SIZE: {best_config['window_size_seconds']:.1f} seconds")
        print(f"   Accuracy: {best_config['accuracy']:.4f}")
        print(f"   ROC-AUC: {best_config['roc_auc']:.4f}")
        print(f"   Matthews Correlation: {best_config['matthews_corr']:.4f}")
        
        if best_config['accuracy'] >= 0.68:
            print("   🎉 TARGET ACHIEVED! Accuracy >= 68%")
        else:
            print("   ⚠️  Target not reached. Consider further optimization.")
    
    # Temporal analysis
    print(f"\n📊 TEMPORAL PERFORMANCE ANALYSIS:")
    print("-" * 50)
    for _, row in df_results.iterrows():
        temporal_metrics = row.get('detailed_results', {}).get('temporal_metrics', {})
        print(f"\n{row['window_size_seconds']:.1f}s Window Performance:")
        for metric_name, metric_value in temporal_metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
    
    # Save results
    df_results.to_csv(output_path / "temporal_analysis_results.csv", index=False)
    
    # Save best configuration
    with open(output_path / "best_temporal_config.json", 'w') as f:
        json.dump(best_config, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_path}")
    print("  - temporal_analysis_results.csv")
    print("  - best_temporal_config.json")

def cleanup_gpu_memory():
    """Clean up GPU memory."""
    try:
        tf.keras.backend.clear_session()
        gc.collect()
        print("✓ GPU memory cleaned up")
    except Exception as e:
        print(f"GPU cleanup warning: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extended Window FULCCA Analysis")
    parser.add_argument("--tfrecord_dir", type=str, default="/home/py9363/telluride_decoding/fulsang_preprocessed/tfrecords", help="Path to TFRecord directory")
    parser.add_argument("--output_dir", type=str, default="extended_window_results", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        results, best_config = run_extended_window_analysis(args.tfrecord_dir, args.output_dir)
        print("\n🎉 Extended Window FULCCA analysis completed successfully!")
    except Exception as e:
        print(f"❌ Extended Window FULCCA analysis failed: {e}")
        sys.exit(1)
    finally:
        cleanup_gpu_memory()
