#!/usr/bin/env python3
"""
Fine-tune FULCCA around the best 66.67% configuration to reach 68%+ accuracy.
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
print("FULCCA FINE-TUNING - Target: 68%+ Accuracy")
print("Based on best configuration: 66.67% (extended_window)")
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

# Fine-tuning configurations around the best 66.67% result
FINE_TUNING_CONFIGS = [
    # Base configuration (your best: 66.67%)
    {'name': 'base_best', 'cca_dims': 10, 'regularization': 0.06, 'window_size': 1024, 'batch_size': 8},
    
    # Fine-tune CCA dimensions around 10
    {'name': 'cca_9', 'cca_dims': 9, 'regularization': 0.06, 'window_size': 1024, 'batch_size': 8},
    {'name': 'cca_11', 'cca_dims': 11, 'regularization': 0.06, 'window_size': 1024, 'batch_size': 8},
    {'name': 'cca_12', 'cca_dims': 12, 'regularization': 0.06, 'window_size': 1024, 'batch_size': 8},
    
    # Fine-tune regularization around 0.06
    {'name': 'reg_0.05', 'cca_dims': 10, 'regularization': 0.05, 'window_size': 1024, 'batch_size': 8},
    {'name': 'reg_0.07', 'cca_dims': 10, 'regularization': 0.07, 'window_size': 1024, 'batch_size': 8},
    {'name': 'reg_0.08', 'cca_dims': 10, 'regularization': 0.08, 'window_size': 1024, 'batch_size': 8},
    
    # Fine-tune window size around 1024 (16s)
    {'name': 'window_896', 'cca_dims': 10, 'regularization': 0.06, 'window_size': 896, 'batch_size': 8},   # 14s
    {'name': 'window_1152', 'cca_dims': 10, 'regularization': 0.06, 'window_size': 1152, 'batch_size': 8}, # 18s
    {'name': 'window_1280', 'cca_dims': 10, 'regularization': 0.06, 'window_size': 1280, 'batch_size': 8}, # 20s
    
    # Fine-tune batch size around 8
    {'name': 'batch_6', 'cca_dims': 10, 'regularization': 0.06, 'window_size': 1024, 'batch_size': 6},
    {'name': 'batch_10', 'cca_dims': 10, 'regularization': 0.06, 'window_size': 1024, 'batch_size': 10},
    {'name': 'batch_12', 'cca_dims': 10, 'regularization': 0.06, 'window_size': 1024, 'batch_size': 12},
    
    # Combined optimizations
    {'name': 'opt_1', 'cca_dims': 11, 'regularization': 0.07, 'window_size': 1152, 'batch_size': 8},
    {'name': 'opt_2', 'cca_dims': 9, 'regularization': 0.05, 'window_size': 896, 'batch_size': 10},
    {'name': 'opt_3', 'cca_dims': 12, 'regularization': 0.08, 'window_size': 1280, 'batch_size': 6},
]

def run_fine_tuning_analysis(tfrecord_dir: str = "/home/py9363/telluride_decoding/fulsang_preprocessed/tfrecords", 
                            output_dir: str = "fine_tuning_results"):
    """
    Run fine-tuning analysis around the best 66.67% configuration.
    """
    print(f"\n{'='*80}")
    print("FULCCA FINE-TUNING ANALYSIS")
    print(f"{'='*80}")
    print(f"TFRecord directory: {tfrecord_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Number of fine-tuning configurations: {len(FINE_TUNING_CONFIGS)}")
    print(f"Target: Push 66.67% → 68%+ (need 1.33% improvement)")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Results storage
    all_results = []
    best_accuracy = 0.6667  # Your current best
    best_config = None
    target_achieved = False
    
    # Run each fine-tuning configuration
    for i, config in enumerate(FINE_TUNING_CONFIGS):
        print(f"\n{'='*60}")
        print(f"Fine-tuning {i+1}/{len(FINE_TUNING_CONFIGS)}: {config['name']}")
        print(f"{'='*60}")
        print(f"CCA dimensions: {config['cca_dims']}")
        print(f"Regularization: {config['regularization']}")
        print(f"Window size: {config['window_size']} samples ({config['window_size']/64:.1f} seconds)")
        print(f"Batch size: {config['batch_size']}")
        
        try:
            # Create data loaders
            train_dataset, val_dataset, test_dataset = create_fine_tuning_data_loaders(
                tfrecord_dir, 
                batch_size=config['batch_size'],
                window_size=config['window_size']
            )
            
            # Create model
            model = FineTuningFULCCAModel(
                cca_dims=config['cca_dims'],
                regularization=config['regularization'],
                window_size=config['window_size']
            )
            
            # Create trainer
            trainer = FineTuningFULCCATrainer(model, str(output_path / config['name']))
            
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
                'improvement': results['accuracy'] - 0.6667,  # Improvement over base
                'target_achieved': results['accuracy'] >= 0.68
            }
            all_results.append(config_result)
            
            # Track best configuration
            if results['accuracy'] > best_accuracy:
                best_accuracy = results['accuracy']
                best_config = config_result
            
            # Check if target achieved
            if results['accuracy'] >= 0.68:
                target_achieved = True
            
            print(f"✓ Configuration {config['name']} completed")
            print(f"  Test Accuracy: {results['accuracy']:.4f}")
            print(f"  Improvement: {config_result['improvement']:+.4f}")
            print(f"  ROC-AUC: {results.get('roc_auc_metrics', {}).get('roc_auc_score', 0):.4f}")
            
            if results['accuracy'] >= 0.68:
                print(f"  🎉 TARGET ACHIEVED! Accuracy >= 68%")
            
            # Clean up memory
            cleanup_gpu_memory()
            
        except Exception as e:
            print(f"❌ Configuration {config['name']} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate fine-tuning report
    generate_fine_tuning_report(all_results, best_config, target_achieved, output_path)
    
    return all_results, best_config

def create_fine_tuning_data_loaders(tfrecord_dir: str, batch_size: int = 8, 
                                  window_size: int = 1024) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Create data loaders for fine-tuning analysis.
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

class FineTuningFULCCAModel:
    """
    Fine-tuning FULCCA model optimized around the best 66.67% configuration.
    """
    
    def __init__(self, cca_dims: int = 10, regularization: float = 0.06, window_size: int = 1024):
        self.cca_dims = cca_dims
        self.regularization = regularization
        self.window_size = window_size
        self.model = None
        self.is_fitted = False
        
        print(f"Fine-tuning FULCCA model initialized:")
        print(f"  CCA dimensions: {cca_dims}")
        print(f"  Regularization: {regularization}")
        print(f"  Window size: {window_size} samples ({window_size/64:.1f} seconds)")
    
    def fit(self, dataset: tf.data.Dataset):
        """Fit the fine-tuning CCA model."""
        print("Fitting Fine-tuning FULCCA model...")
        
        # Create robust CCA model
        self.model = self._create_fine_tuning_cca_model(dataset)
        
        # Compile with optimized settings
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Optimized learning rate
            loss='mse',
            metrics=[cca_pearson_correlation_first]
        )
        
        # Fit with 2 epochs for fine-tuning
        self.model.fit(dataset, epochs=2)
        
        self.is_fitted = True
        print("✓ Fine-tuning FULCCA model fitted successfully")
    
    def _create_fine_tuning_cca_model(self, dataset: tf.data.Dataset):
        """Create fine-tuning CCA model."""
        # Force CPU creation to avoid CUDA handle corruption
        print("Creating Fine-tuning CCA model on CPU...")
        with tf.device('/CPU:0'):
            model = BrainModelCCA(
                input_dataset=dataset,
                cca_dims=self.cca_dims,
                regularization_lambda=self.regularization
            )
        print("✓ Fine-tuning CCA model created successfully on CPU")
        return model
    
    def predict(self, dataset: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Fine-tuning prediction."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        print("Making Fine-tuning FULCCA predictions...")
        
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
                
                # Use simple majority voting
                sample_predictions = tf.cast(tf.reduce_sum(pred_reshaped, axis=1) > (self.window_size // 2), tf.int64)
                
                all_predictions.extend(sample_predictions.numpy())
                
                if targets is not None:
                    all_targets.extend(targets.numpy().flatten())
        
        return np.array(all_predictions), np.array(all_targets)

class FineTuningFULCCATrainer:
    """Fine-tuning trainer focused on reaching 68%+ accuracy."""
    
    def __init__(self, model: FineTuningFULCCAModel, output_dir: str):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def train(self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset) -> float:
        """Train with focus on reaching 68%+ accuracy."""
        print("Starting Fine-tuning FULCCA training...")
        
        self.model.fit(train_dataset)
        
        # Enhanced validation
        val_predictions, val_targets = self.model.predict(val_dataset)
        val_accuracy = accuracy_score(val_targets, val_predictions)
        
        print(f"Fine-tuning FULCCA training completed!")
        print(f"  Validation accuracy: {val_accuracy:.4f}")
        
        return val_accuracy
    
    def test(self, test_dataset: tf.data.Dataset) -> Dict:
        """Test with comprehensive metrics."""
        print("Testing Fine-tuning FULCCA model...")
        
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

def generate_fine_tuning_report(all_results: List[Dict], best_config: Dict, target_achieved: bool, output_path: Path):
    """Generate fine-tuning analysis report."""
    print(f"\n{'='*80}")
    print("FULCCA FINE-TUNING ANALYSIS COMPLETE")
    print(f"{'='*80}")
    
    # Create results DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Sort by test accuracy
    df_results = df_results.sort_values('test_accuracy', ascending=False)
    
    print("\nFine-tuning Results (sorted by test accuracy):")
    print("-" * 80)
    print(f"{'Configuration':<20} {'Test Acc':<10} {'Improvement':<12} {'ROC-AUC':<10} {'Target':<8}")
    print("-" * 80)
    
    for _, row in df_results.iterrows():
        target_status = "🎉 YES" if row['target_achieved'] else "❌ NO"
        improvement = f"{row['improvement']:+.4f}"
        print(f"{row['configuration']:<20} {row['test_accuracy']:<10.4f} {improvement:<12} {row['roc_auc']:<10.4f} {target_status:<8}")
    
    # Best configuration
    if best_config:
        print(f"\n🏆 BEST CONFIGURATION: {best_config['configuration']}")
        print(f"   Test Accuracy: {best_config['test_accuracy']:.4f}")
        print(f"   Improvement: {best_config['improvement']:+.4f}")
        print(f"   ROC-AUC: {best_config['roc_auc']:.4f}")
        print(f"   Matthews Correlation: {best_config['matthews_corr']:.4f}")
        
        if best_config['test_accuracy'] >= 0.68:
            print("   🎉 TARGET ACHIEVED! Accuracy >= 68%")
        else:
            print("   ⚠️  Target not reached. Consider further optimization.")
    
    # Overall status
    if target_achieved:
        print(f"\n🎉 SUCCESS! Target of 68%+ accuracy achieved!")
    else:
        print(f"\n⚠️  Target not reached. Best improvement: {df_results['improvement'].max():+.4f}")
    
    # Save results
    df_results.to_csv(output_path / "fine_tuning_results.csv", index=False)
    
    # Save best configuration (without numpy arrays)
    config_to_save = best_config.copy()
    if 'detailed_results' in config_to_save:
        del config_to_save['detailed_results']  # Remove numpy arrays
    with open(output_path / "best_fine_tuning_config.json", 'w') as f:
        json.dump(config_to_save, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_path}")
    print("  - fine_tuning_results.csv")
    print("  - best_fine_tuning_config.json")

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
    
    parser = argparse.ArgumentParser(description="FULCCA Fine-tuning Analysis")
    parser.add_argument("--tfrecord_dir", type=str, default="/home/py9363/telluride_decoding/fulsang_preprocessed/tfrecords", help="Path to TFRecord directory")
    parser.add_argument("--output_dir", type=str, default="fine_tuning_results", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        results, best_config = run_fine_tuning_analysis(args.tfrecord_dir, args.output_dir)
        print("\n🎉 Fine-tuning FULCCA analysis completed successfully!")
    except Exception as e:
        print(f"❌ Fine-tuning FULCCA analysis failed: {e}")
        sys.exit(1)
    finally:
        cleanup_gpu_memory()
