#!/usr/bin/env python3
"""
Test different hyperparameters for FULCNNLOC with optimal window size
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

# Import from FULCNNLOC
from FULCNNLOC import FULCNNLOCDataset, CNNLOCTrainer, split_dataset

def run_training_hyperparams(window_size_seconds, batch_size, learning_rate, 
                             num_epochs, tfrecord_dir='fulsang_preprocessed/tfrecords',
                             output_dir="hyperparam_results", overlap=0.5):
    """
    Run training with specific hyperparameters and return results.
    """
    window_size_samples = int(window_size_seconds * 64)  # 64 Hz sampling rate
    
    print(f"\nTesting: WS={window_size_seconds}s, BS={batch_size}, LR={learning_rate}, Epochs={num_epochs}")
    
    try:
        # Create datasets
        full_dataset = FULCNNLOCDataset(
            tfrecord_dir=tfrecord_dir,
            window_size=window_size_samples,
            overlap=overlap,
            transform_eeg=True,
            mode='train'
        )
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = split_dataset(full_dataset, train_ratio=0.7, val_ratio=0.15)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        # Calculate class weights
        train_labels = [train_dataset[i][1].item() for i in range(len(train_dataset))]
        unique, counts = np.unique(train_labels, return_counts=True)
        class_weights = torch.tensor(
            [len(train_labels) / (len(unique) * count) for count in counts],
            dtype=torch.float32
        )
        
        # Create model and trainer
        trainer = CNNLOCTrainer(
            n_channels=66,
            n_time=32,
            n_freq=4,
            n_classes=2,
            dropout_rate=0.3,
            output_dir=Path(output_dir) / f"ws{window_size_seconds}s_bs{batch_size}_lr{learning_rate}"
        )
        
        # Train
        best_val_acc = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            class_weights=class_weights
        )
        
        # Test
        test_results = trainer.test(test_loader)
        
        # Save results
        output_path = Path(output_dir) / f"ws{window_size_seconds}s_bs{batch_size}_lr{learning_rate}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_dict = {
            'window_size_seconds': window_size_seconds,
            'window_size_samples': window_size_samples,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'overlap': overlap,
            'test_accuracy': test_results.get('accuracy', None),
            'test_roc_auc': test_results.get('roc_auc', None),
            'test_f1_score': test_results.get('f1_score', None),
            'test_balanced_accuracy': test_results.get('balanced_accuracy', None),
            'test_precision': test_results.get('precision', None),
            'test_recall': test_results.get('recall', None),
            'best_val_accuracy': test_results.get('best_val_acc', None),
            'status': 'success'
        }
        
        with open(output_path / "results.json", 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"✓ Acc={results_dict['test_accuracy']:.4f}, ROC-AUC={results_dict['test_roc_auc']:.4f if results_dict['test_roc_auc'] else 'N/A'}")
        
        return results_dict
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'window_size_seconds': window_size_seconds,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'status': 'error',
            'error': str(e)
        }


def test_hyperparameters():
    """Test different hyperparameter combinations."""
    
    # Optimal window size from previous testing (or use default)
    optimal_window_size = 8  # Start with 8s, can be updated after window size testing
    
    # Hyperparameter combinations to test
    hyperparams = [
        # Batch sizes
        {'batch_size': 16, 'learning_rate': 0.001, 'num_epochs': 30},
        {'batch_size': 32, 'learning_rate': 0.001, 'num_epochs': 30},
        {'batch_size': 64, 'learning_rate': 0.001, 'num_epochs': 30},
        {'batch_size': 128, 'learning_rate': 0.001, 'num_epochs': 30},
        
        # Learning rates
        {'batch_size': 32, 'learning_rate': 0.0001, 'num_epochs': 30},
        {'batch_size': 32, 'learning_rate': 0.0005, 'num_epochs': 30},
        {'batch_size': 32, 'learning_rate': 0.001, 'num_epochs': 30},
        {'batch_size': 32, 'learning_rate': 0.002, 'num_epochs': 30},
        {'batch_size': 32, 'learning_rate': 0.005, 'num_epochs': 30},
        
        # Epochs (with optimal batch/lr)
        {'batch_size': 32, 'learning_rate': 0.001, 'num_epochs': 20},
        {'batch_size': 32, 'learning_rate': 0.001, 'num_epochs': 30},
        {'batch_size': 32, 'learning_rate': 0.001, 'num_epochs': 50},
    ]
    
    all_results = []
    
    print("="*80)
    print("FULCNNLOC Hyperparameter Testing")
    print("="*80)
    print(f"Window Size: {optimal_window_size}s (fixed)")
    print(f"Testing {len(hyperparams)} hyperparameter combinations")
    print("="*80)
    
    for hp_idx, hp in enumerate(hyperparams):
        print(f"\n[{hp_idx+1}/{len(hyperparams)}] ", end="")
        
        result = run_training_hyperparams(
            window_size_seconds=optimal_window_size,
            **hp,
            output_dir="hyperparam_results"
        )
        
        all_results.append(result)
        
        # Save intermediate results
        df = pd.DataFrame(all_results)
        df.to_csv("hyperparam_results_all.csv", index=False)
    
    # Save final results
    df = pd.DataFrame(all_results)
    df.to_csv("hyperparam_results_final.csv", index=False)
    
    # Create summary report
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    successful_results = df[df['status'] == 'success'].copy()
    
    if len(successful_results) > 0:
        # Best by batch size
        print("\nBest Results by Batch Size:")
        print("-" * 80)
        for bs in [16, 32, 64, 128]:
            bs_results = successful_results[successful_results['batch_size'] == bs]
            if len(bs_results) > 0:
                best = bs_results.loc[bs_results['test_accuracy'].idxmax()]
                print(f"  BS={bs:3d}: Acc={best['test_accuracy']:.4f}, "
                      f"ROC-AUC={best['test_roc_auc']:.4f if best['test_roc_auc'] else 'N/A'}, "
                      f"LR={best['learning_rate']}")
        
        # Best by learning rate
        print("\nBest Results by Learning Rate:")
        print("-" * 80)
        for lr in [0.0001, 0.0005, 0.001, 0.002, 0.005]:
            lr_results = successful_results[successful_results['learning_rate'] == lr]
            if len(lr_results) > 0:
                best = lr_results.loc[lr_results['test_accuracy'].idxmax()]
                print(f"  LR={lr:.4f}: Acc={best['test_accuracy']:.4f}, "
                      f"ROC-AUC={best['test_roc_auc']:.4f if best['test_roc_auc'] else 'N/A'}, "
                      f"BS={best['batch_size']}")
        
        # Overall best
        print("\nOverall Best Result:")
        print("-" * 80)
        best_overall = successful_results.loc[successful_results['test_accuracy'].idxmax()]
        print(f"  Batch Size: {best_overall['batch_size']}")
        print(f"  Learning Rate: {best_overall['learning_rate']}")
        print(f"  Epochs: {best_overall['num_epochs']}")
        print(f"  Test Accuracy: {best_overall['test_accuracy']:.4f}")
        print(f"  Test ROC-AUC: {best_overall['test_roc_auc']:.4f if best_overall['test_roc_auc'] else 'N/A'}")
        print(f"  Test F1-Score: {best_overall['test_f1_score']:.4f if best_overall['test_f1_score'] else 'N/A'}")
    
    print(f"\nResults saved to:")
    print(f"  - hyperparam_results_final.csv")
    
    return df


if __name__ == "__main__":
    results_df = test_hyperparameters()
    print("\n✓ Hyperparameter testing completed!")

