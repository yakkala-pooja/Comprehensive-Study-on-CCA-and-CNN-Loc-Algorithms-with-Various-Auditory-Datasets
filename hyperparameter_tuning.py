#!/usr/bin/env python3
"""
Hyperparameter Tuning Script for FULCNN
This script performs systematic hyperparameter optimization for the Fulsang CNN-LOC model.
"""

import subprocess
import json
import os
import itertools
from pathlib import Path
import numpy as np
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
        # Parse results from the output directory
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

def grid_search():
    """Perform grid search over hyperparameter space."""
    
    # Define hyperparameter search space
    param_grid = {
        'tfrecord_dir': ['fulsang_preprocessed/tfrecords'],
        'batch_size': [16, 32, 64],
        'num_epochs': [50, 100, 150],
        'learning_rate': [1e-4, 2e-4, 5e-4, 1e-3],
        'window_size': [256, 512, 1024],  # 4s, 8s, 16s at 64Hz
        'weight_decay': [1e-5, 1e-4, 1e-3],
        'dropout_rate': [0.2, 0.3, 0.4, 0.5],
        'label_smoothing': [0.05, 0.1, 0.15]
    }
    
    # Create output directory for tuning results
    tuning_dir = Path('hyperparameter_tuning_results')
    tuning_dir.mkdir(exist_ok=True)
    
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    best_score = -float('inf')
    best_config = None
    results = []
    
    total_combinations = np.prod([len(v) for v in values])
    print(f"Total combinations to test: {total_combinations}")
    
    # Sample a subset for initial exploration (random sampling)
    np.random.seed(42)
    n_samples = min(50, total_combinations)  # Test up to 50 configurations
    
    sampled_indices = np.random.choice(total_combinations, n_samples, replace=False)
    
    for i, idx in enumerate(sampled_indices):
        # Convert index to parameter combination
        config = {}
        temp_idx = idx
        for j, key in enumerate(keys):
            config[key] = values[j][temp_idx % len(values[j])]
            temp_idx //= len(values[j])
        
        # Create unique output directory for this run
        config['output_dir'] = str(tuning_dir / f"run_{i+1:03d}")
        
        print(f"\n{'='*80}")
        print(f"HYPERPARAMETER TUNING RUN {i+1}/{n_samples}")
        print(f"{'='*80}")
        print(f"Configuration:")
        for key, value in config.items():
            if key != 'output_dir':
                print(f"  {key}: {value}")
        
        # Run training
        result = run_training(config)
        
        if result['success']:
            # Calculate composite score (weighted combination of metrics)
            score = (
                0.4 * result['test_accuracy'] +
                0.3 * result['val_accuracy'] +
                0.2 * result['roc_auc'] +
                0.1 * (1 - result['rmse'])  # Lower RMSE is better
            )
            
            print(f"Results:")
            print(f"  Test Accuracy: {result['test_accuracy']:.4f}")
            print(f"  Val Accuracy: {result['val_accuracy']:.4f}")
            print(f"  ROC-AUC: {result['roc_auc']:.4f}")
            print(f"  RMSE: {result['rmse']:.4f}")
            print(f"  MCC: {result['mcc']:.4f}")
            print(f"  Composite Score: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_config = config.copy()
                print(f"  🎯 NEW BEST SCORE!")
            
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
    results_file = tuning_dir / 'tuning_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'best_config': best_config,
            'best_score': best_score,
            'all_results': results
        }, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER TUNING COMPLETE")
    print(f"{'='*80}")
    print(f"Best configuration:")
    for key, value in best_config.items():
        if key != 'output_dir':
            print(f"  {key}: {value}")
    print(f"Best score: {best_score:.4f}")
    print(f"Results saved to: {results_file}")
    
    return best_config, best_score

def targeted_search():
    """Perform targeted search around promising configurations."""
    
    # Start with the best configuration from grid search
    tuning_dir = Path('hyperparameter_tuning_results')
    results_file = tuning_dir / 'tuning_results.json'
    
    if not results_file.exists():
        print("No previous tuning results found. Run grid_search() first.")
        return None, None
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    best_config = data['best_config']
    
    # Define targeted search around best config
    targeted_params = {
        'learning_rate': [best_config['learning_rate'] * 0.5, best_config['learning_rate'], best_config['learning_rate'] * 2],
        'weight_decay': [best_config['weight_decay'] * 0.5, best_config['weight_decay'], best_config['weight_decay'] * 2],
        'dropout_rate': [max(0.1, best_config['dropout_rate'] - 0.1), best_config['dropout_rate'], min(0.6, best_config['dropout_rate'] + 0.1)],
        'label_smoothing': [max(0.01, best_config['label_smoothing'] - 0.05), best_config['label_smoothing'], min(0.2, best_config['label_smoothing'] + 0.05)]
    }
    
    print(f"Performing targeted search around best configuration...")
    print(f"Best config: {best_config}")
    
    # Test targeted variations
    best_score = data['best_score']
    best_config_final = best_config.copy()
    
    for param, values in targeted_params.items():
        print(f"\nTesting {param}: {values}")
        
        for value in values:
            config = best_config.copy()
            config[param] = value
            config['output_dir'] = str(tuning_dir / f"targeted_{param}_{value}")
            
            result = run_training(config)
            
            if result['success']:
                score = (
                    0.4 * result['test_accuracy'] +
                    0.3 * result['val_accuracy'] +
                    0.2 * result['roc_auc'] +
                    0.1 * (1 - result['rmse'])
                )
                
                print(f"  {param}={value}: Score={score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_config_final = config.copy()
                    print(f"    🎯 NEW BEST!")
    
    return best_config_final, best_score

if __name__ == "__main__":
    print("FULCNN Hyperparameter Tuning")
    print("="*50)
    
    # Run grid search
    print("Phase 1: Grid Search")
    best_config, best_score = grid_search()
    
    # Run targeted search
    print("\nPhase 2: Targeted Search")
    final_config, final_score = targeted_search()
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Best configuration:")
    for key, value in final_config.items():
        if key != 'output_dir':
            print(f"  {key}: {value}")
    print(f"Best score: {final_score:.4f}")
    
    # Save final best configuration
    final_config_file = Path('best_hyperparameters.json')
    with open(final_config_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'best_config': final_config,
            'best_score': final_score
        }, f, indent=2)
    
    print(f"Final configuration saved to: {final_config_file}")
