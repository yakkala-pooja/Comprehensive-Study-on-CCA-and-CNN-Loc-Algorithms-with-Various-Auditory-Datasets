#!/usr/bin/env python3
"""
Quick Hyperparameter Tuning for FULCNN
Focuses on the most impactful parameters for EEG attention decoding.
"""

import subprocess
import json
import os
from pathlib import Path
from datetime import datetime

def run_training(config):
    """Run FULCNN training with given configuration."""
    cmd = [
        'python3', 'FULCNN.py',
        '--tfrecord_dir', config['tfrecord_dir'],
        '--batch_size', str(config['batch_size']),
        '--num_epochs', str(config['num_epochs']),
        '--learning_rate', str(config['learning_rate']),
        '--window_size', str(config['window_size']),
        '--weight_decay', str(config['weight_decay']),
        '--dropout_rate', str(config['dropout_rate']),
        '--label_smoothing', str(config['label_smoothing']),
        '--output_dir', config['output_dir']
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        results_file = Path(config['output_dir']) / 'results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            return {
                'success': True,
                'test_accuracy': results.get('accuracy', 0),
                'val_accuracy': results.get('best_val_acc', 0),
                'roc_auc': results.get('roc_auc_metrics', {}).get('roc_auc_score', 0),
                'rmse': results.get('msed_metrics', {}).get('rmse', float('inf')),
                'mcc': results.get('advanced_metrics', {}).get('matthews_correlation_coefficient', 0)
            }
    
    return {
        'success': False,
        'test_accuracy': 0,
        'val_accuracy': 0,
        'roc_auc': 0,
        'rmse': float('inf'),
        'mcc': 0
    }

def quick_tuning():
    """Quick tuning focusing on most impactful parameters."""
    
    # Define configurations to test (most impactful parameters)
    configurations = [
        # Configuration 1: Conservative (lower learning rate, more regularization)
        {
            'name': 'Conservative',
            'batch_size': 32,
            'num_epochs': 100,
            'learning_rate': 1e-4,
            'window_size': 512,
            'weight_decay': 1e-4,
            'dropout_rate': 0.4,
            'label_smoothing': 0.1
        },
        
        # Configuration 2: Aggressive (higher learning rate, less regularization)
        {
            'name': 'Aggressive',
            'batch_size': 64,
            'num_epochs': 150,
            'learning_rate': 5e-4,
            'window_size': 512,
            'weight_decay': 1e-5,
            'dropout_rate': 0.2,
            'label_smoothing': 0.05
        },
        
        # Configuration 3: Balanced (middle ground)
        {
            'name': 'Balanced',
            'batch_size': 32,
            'num_epochs': 100,
            'learning_rate': 2e-4,
            'window_size': 512,
            'weight_decay': 1e-4,
            'dropout_rate': 0.3,
            'label_smoothing': 0.1
        },
        
        # Configuration 4: Large window (longer temporal context)
        {
            'name': 'Large_Window',
            'batch_size': 16,
            'num_epochs': 100,
            'learning_rate': 1e-4,
            'window_size': 1024,  # 16 seconds
            'weight_decay': 1e-4,
            'dropout_rate': 0.4,
            'label_smoothing': 0.1
        },
        
        # Configuration 5: Small window (shorter temporal context)
        {
            'name': 'Small_Window',
            'batch_size': 64,
            'num_epochs': 100,
            'learning_rate': 2e-4,
            'window_size': 256,  # 4 seconds
            'weight_decay': 1e-4,
            'dropout_rate': 0.3,
            'label_smoothing': 0.1
        },
        
        # Configuration 6: High regularization (prevent overfitting)
        {
            'name': 'High_Regularization',
            'batch_size': 32,
            'num_epochs': 100,
            'learning_rate': 1e-4,
            'window_size': 512,
            'weight_decay': 1e-3,
            'dropout_rate': 0.5,
            'label_smoothing': 0.15
        }
    ]
    
    # Create output directory
    tuning_dir = Path('quick_tuning_results')
    tuning_dir.mkdir(exist_ok=True)
    
    best_score = -float('inf')
    best_config = None
    results = []
    
    for i, config in enumerate(configurations):
        print(f"\n{'='*80}")
        print(f"TESTING CONFIGURATION {i+1}: {config['name']}")
        print(f"{'='*80}")
        
        # Add required parameters
        config['tfrecord_dir'] = 'fulsang_preprocessed/tfrecords'
        config['output_dir'] = str(tuning_dir / f"config_{i+1}_{config['name']}")
        
        print(f"Parameters:")
        for key, value in config.items():
            if key not in ['name', 'tfrecord_dir', 'output_dir']:
                print(f"  {key}: {value}")
        
        # Run training
        result = run_training(config)
        
        if result['success']:
            # Calculate composite score
            score = (
                0.4 * result['test_accuracy'] +
                0.3 * result['val_accuracy'] +
                0.2 * result['roc_auc'] +
                0.1 * max(0, 1 - result['rmse'])  # Ensure non-negative
            )
            
            print(f"\nResults:")
            print(f"  Test Accuracy: {result['test_accuracy']:.4f}")
            print(f"  Val Accuracy: {result['val_accuracy']:.4f}")
            print(f"  ROC-AUC: {result['roc_auc']:.4f}")
            print(f"  RMSE: {result['rmse']:.4f}")
            print(f"  MCC: {result['mcc']:.4f}")
            print(f"  Composite Score: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_config = config.copy()
                print(f"  🎯 NEW BEST CONFIGURATION!")
            
            results.append({
                'config': config,
                'results': result,
                'score': score
            })
        else:
            print(f"  ❌ Training failed!")
            results.append({
                'config': config,
                'results': result,
                'score': -float('inf')
            })
    
    # Save results
    results_file = tuning_dir / 'quick_tuning_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'best_config': best_config,
            'best_score': best_score,
            'all_results': results
        }, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"QUICK TUNING COMPLETE")
    print(f"{'='*80}")
    print(f"Best configuration: {best_config['name']}")
    print(f"Parameters:")
    for key, value in best_config.items():
        if key not in ['name', 'tfrecord_dir', 'output_dir']:
            print(f"  {key}: {value}")
    print(f"Best score: {best_score:.4f}")
    print(f"Results saved to: {results_file}")
    
    return best_config, best_score

if __name__ == "__main__":
    print("FULCNN Quick Hyperparameter Tuning")
    print("="*50)
    print("Testing 6 key configurations focusing on most impactful parameters")
    
    best_config, best_score = quick_tuning()
    
    print(f"\n{'='*80}")
    print(f"RECOMMENDED NEXT STEPS")
    print(f"{'='*80}")
    print(f"1. Use the best configuration for production training")
    print(f"2. If results are still poor, consider:")
    print(f"   - Data augmentation techniques")
    print(f"   - Different architecture (e.g., LSTM, Transformer)")
    print(f"   - Feature engineering (e.g., different frequency bands)")
    print(f"   - Ensemble methods")
    print(f"3. Run full hyperparameter tuning with: python3 hyperparameter_tuning.py")
