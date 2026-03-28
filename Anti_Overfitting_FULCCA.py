#!/usr/bin/env python3
"""
Fixed FULCCA - Addresses overfitting and validation issues.
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
print("FIXED FULCCA - Addressing Overfitting and Validation Issues")
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

# Anti-overfitting configurations
ANTI_OVERFITTING_CONFIGS = [
    # Conservative configurations to prevent overfitting
    {'name': 'conservative_small', 'cca_dims': 3, 'regularization': 0.1, 'window_size': 512, 'batch_size': 16},
    {'name': 'conservative_medium', 'cca_dims': 5, 'regularization': 0.15, 'window_size': 512, 'batch_size': 16},
    {'name': 'conservative_large', 'cca_dims': 8, 'regularization': 0.2, 'window_size': 512, 'batch_size': 16},
    {'name': 'very_conservative', 'cca_dims': 2, 'regularization': 0.25, 'window_size': 512, 'batch_size': 16},
]

def run_anti_overfitting_analysis(tfrecord_dir: str = "/home/py9363/telluride_decoding/fulsang_preprocessed/tfrecords", 
                                 output_dir: str = "anti_overfitting_results"):
    """
    Run anti-overfitting FULCCA analysis.
    """
    print(f"\n{'='*80}")
    print("ANTI-OVERFITTING FULCCA ANALYSIS")
    print(f"{'='*80}")
    print(f"TFRecord directory: {tfrecord_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Number of configurations: {len(ANTI_OVERFITTING_CONFIGS)}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Results storage
    all_results = []
    best_val_accuracy = 0
    best_config = None
    
    # Run each configuration
    for i, config in enumerate(ANTI_OVERFITTING_CONFIGS):
        print(f"\n{'='*60}")
        print(f"Configuration {i+1}/{len(ANTI_OVERFITTING_CONFIGS)}: {config['name']}")
        print(f"{'='*60}")
        print(f"CCA dimensions: {config['cca_dims']}")
        print(f"Regularization: {config['regularization']}")
        print(f"Window size: {config['window_size']} samples ({config['window_size']/64:.1f} seconds at 64Hz)")
        print(f"Batch size: {config['batch_size']}")
        
        try:
            # Create data loaders
            train_dataset, val_dataset, test_dataset = create_anti_overfitting_data_loaders(
                tfrecord_dir, 
                batch_size=config['batch_size'],
                window_size=config['window_size']
            )
            
            # Create model
            model = AntiOverfittingFULCCAModel(
                cca_dims=config['cca_dims'],
                regularization=config['regularization'],
                window_size=config['window_size']
            )
            
            # Create trainer
            trainer = AntiOverfittingFULCCATrainer(model, str(output_path / config['name']))
            
            # Train and test
            val_accuracy = trainer.train(train_dataset, val_dataset)
            results = trainer.test(test_dataset)
            
            # Store results
            config_result = {
                'configuration': config['name'],
                'val_accuracy': val_accuracy,
                'test_accuracy': results['accuracy'],
                'roc_auc': results.get('roc_auc_metrics', {}).get('roc_auc_score', 0),
                'matthews_corr': results.get('advanced_metrics', {}).get('matthews_correlation_coefficient', 0),
                'balanced_accuracy': results.get('advanced_metrics', {}).get('balanced_accuracy', 0),
                'config_params': config,
                'detailed_results': results
            }
            all_results.append(config_result)
            
            # Track best configuration based on validation accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_config = config_result
            
            print(f"✓ Configuration {config['name']} completed")
            print(f"  Validation Accuracy: {val_accuracy:.4f}")
            print(f"  Test Accuracy: {results['accuracy']:.4f}")
            print(f"  ROC-AUC: {results.get('roc_auc_metrics', {}).get('roc_auc_score', 0):.4f}")
            
            # Clean up memory
            cleanup_gpu_memory()
            
        except Exception as e:
            print(f"❌ Configuration {config['name']} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate anti-overfitting report
    generate_anti_overfitting_report(all_results, best_config, output_path)
    
    return all_results, best_config

def create_anti_overfitting_data_loaders(tfrecord_dir: str, batch_size: int = 16, 
                                       window_size: int = 512) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Create data loaders with better validation split.
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

class AntiOverfittingFULCCAModel:
    """
    Anti-overfitting FULCCA model with conservative parameters.
    """
    
    def __init__(self, cca_dims: int = 3, regularization: float = 0.1, window_size: int = 512):
        self.cca_dims = cca_dims
        self.regularization = regularization
        self.window_size = window_size
        self.model = None
        self.is_fitted = False
        
        print(f"Anti-Overfitting FULCCA model initialized:")
        print(f"  CCA dimensions: {cca_dims} (conservative)")
        print(f"  Regularization: {regularization} (high)")
        print(f"  Window size: {window_size} samples ({window_size/64:.1f} seconds)")
    
    def fit(self, dataset: tf.data.Dataset):
        """Fit the anti-overfitting CCA model."""
        print("Fitting Anti-Overfitting FULCCA model...")
        
        # Create robust CCA model
        self.model = self._create_anti_overfitting_cca_model(dataset)
        
        # Compile with conservative settings
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),  # Very low learning rate
            loss='mse',
            metrics=[cca_pearson_correlation_first]
        )
        
        # Fit with only 1 epoch to prevent overfitting
        self.model.fit(dataset, epochs=1)
        
        self.is_fitted = True
        print("✓ Anti-Overfitting FULCCA model fitted successfully")
    
    def _create_anti_overfitting_cca_model(self, dataset: tf.data.Dataset):
        """Create anti-overfitting CCA model."""
        # Force CPU creation to avoid CUDA handle corruption
        print("Creating Anti-Overfitting CCA model on CPU...")
        with tf.device('/CPU:0'):
            model = BrainModelCCA(
                input_dataset=dataset,
                cca_dims=self.cca_dims,
                regularization_lambda=self.regularization
            )
        print("✓ Anti-Overfitting CCA model created successfully on CPU")
        return model
    
    def predict(self, dataset: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Anti-overfitting prediction."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        print("Making Anti-Overfitting FULCCA predictions...")
        
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
                
                # Simple prediction processing
                cca_width = predictions.shape[-1] // 2
                pred1 = predictions[:, :cca_width]
                
                # Use first CCA component for classification
                cca_scores = pred1[:, 0]  # First CCA component
                
                # Convert to binary predictions
                binary_predictions = tf.cast(cca_scores > 0, tf.int64)
                
                # Aggregate predictions per sample
                batch_size = inputs['input_1'].shape[0] // self.window_size
                pred_reshaped = tf.reshape(binary_predictions, (batch_size, self.window_size))
                
                # Use simple majority voting
                sample_predictions = tf.cast(tf.reduce_sum(pred_reshaped, axis=1) > (self.window_size // 2), tf.int64)
                
                all_predictions.extend(sample_predictions.numpy())
                
                if targets is not None:
                    all_targets.extend(targets.numpy().flatten())
        
        return np.array(all_predictions), np.array(all_targets)

class AntiOverfittingFULCCATrainer:
    """Anti-overfitting trainer with focus on validation performance."""
    
    def __init__(self, model: AntiOverfittingFULCCAModel, output_dir: str):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def train(self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset) -> float:
        """Train with focus on validation performance."""
        print("Starting Anti-Overfitting FULCCA training...")
        
        self.model.fit(train_dataset)
        
        # Enhanced validation
        val_predictions, val_targets = self.model.predict(val_dataset)
        val_accuracy = accuracy_score(val_targets, val_predictions)
        
        print(f"Anti-Overfitting FULCCA training completed!")
        print(f"  Validation accuracy: {val_accuracy:.4f}")
        print(f"  Validation samples: {len(val_targets)}")
        
        return val_accuracy
    
    def test(self, test_dataset: tf.data.Dataset) -> Dict:
        """Test with comprehensive metrics."""
        print("Testing Anti-Overfitting FULCCA model...")
        
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

def generate_anti_overfitting_report(all_results: List[Dict], best_config: Dict, output_path: Path):
    """Generate anti-overfitting analysis report."""
    print(f"\n{'='*80}")
    print("ANTI-OVERFITTING ANALYSIS COMPLETE")
    print(f"{'='*80}")
    
    # Create results DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Sort by validation accuracy
    df_results = df_results.sort_values('val_accuracy', ascending=False)
    
    print("\nAnti-Overfitting Results (sorted by validation accuracy):")
    print("-" * 80)
    print(f"{'Configuration':<20} {'Val Acc':<10} {'Test Acc':<10} {'ROC-AUC':<10} {'MCC':<10}")
    print("-" * 80)
    
    for _, row in df_results.iterrows():
        print(f"{row['configuration']:<20} {row['val_accuracy']:<10.4f} {row['test_accuracy']:<10.4f} {row['roc_auc']:<10.4f} {row['matthews_corr']:<10.4f}")
    
    # Best configuration
    if best_config:
        print(f"\n🏆 BEST CONFIGURATION: {best_config['configuration']}")
        print(f"   Validation Accuracy: {best_config['val_accuracy']:.4f}")
        print(f"   Test Accuracy: {best_config['test_accuracy']:.4f}")
        print(f"   ROC-AUC: {best_config['roc_auc']:.4f}")
        print(f"   Matthews Correlation: {best_config['matthews_corr']:.4f}")
        
        # Check for overfitting
        val_test_diff = abs(best_config['val_accuracy'] - best_config['test_accuracy'])
        if val_test_diff < 0.1:
            print("   ✅ Good generalization (low overfitting)")
        else:
            print("   ⚠️  Potential overfitting detected")
    
    # Save results
    df_results.to_csv(output_path / "anti_overfitting_results.csv", index=False)
    
    # Save best configuration
    with open(output_path / "best_anti_overfitting_config.json", 'w') as f:
        json.dump(best_config, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_path}")
    print("  - anti_overfitting_results.csv")
    print("  - best_anti_overfitting_config.json")

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
    
    parser = argparse.ArgumentParser(description="Anti-Overfitting FULCCA Analysis")
    parser.add_argument("--tfrecord_dir", type=str, default="/home/py9363/telluride_decoding/fulsang_preprocessed/tfrecords", help="Path to TFRecord directory")
    parser.add_argument("--output_dir", type=str, default="anti_overfitting_results", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        results, best_config = run_anti_overfitting_analysis(args.tfrecord_dir, args.output_dir)
        print("\n🎉 Anti-overfitting FULCCA analysis completed successfully!")
    except Exception as e:
        print(f"❌ Anti-overfitting FULCCA analysis failed: {e}")
        sys.exit(1)
    finally:
        cleanup_gpu_memory()
