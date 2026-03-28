#!/usr/bin/env python3
"""
Test different window sizes (1s to 30s) and hyperparameters for FULCNNLOC
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

def run_training_window_size(window_size_seconds, batch_size=32, learning_rate=0.001, 
                             num_epochs=30, tfrecord_dir='fulsang_preprocessed/tfrecords',
                             output_dir="window_size_results", overlap=0.5):
    """
    Run training with a specific window size and return results.
    
    Args:
        window_size_seconds: Window size in seconds
        batch_size: Batch size
        learning_rate: Learning rate
        num_epochs: Number of epochs
        tfrecord_dir: TFRecord directory
        output_dir: Output directory for results
        overlap: Window overlap fraction
    
    Returns:
        Dictionary with results
    """
    window_size_samples = int(window_size_seconds * 64)  # 64 Hz sampling rate
    
    print(f"\n{'='*80}")
    print(f"Testing Window Size: {window_size_seconds}s ({window_size_samples} samples)")
    print(f"{'='*80}")
    
    try:
        # Create datasets
        print("Loading dataset...")
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
        print("Initializing model...")
        trainer = CNNLOCTrainer(
            n_channels=66,
            n_time=32,  # Time frames after transformation
            n_freq=4,   # Frequency bins after transformation
            n_classes=2,
            dropout_rate=0.3,
            output_dir=Path(output_dir) / f"window_{window_size_seconds}s"
        )
        
        # Train
        print("Training...")
        best_val_acc = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            class_weights=class_weights
        )
        
        # Test
        print("Testing...")
        test_results = trainer.test(test_loader)
        
        # Save results
        output_path = Path(output_dir) / f"window_{window_size_seconds}s"
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
            'n_train_samples': len(train_dataset),
            'n_val_samples': len(val_dataset),
            'n_test_samples': len(test_dataset),
            'status': 'success'
        }
        
        with open(output_path / "results.json", 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"✓ Window {window_size_seconds}s: Acc={results_dict['test_accuracy']:.4f}, "
              f"ROC-AUC={results_dict['test_roc_auc']:.4f if results_dict['test_roc_auc'] else 'N/A'}")
        
        return results_dict
        
    except Exception as e:
        print(f"✗ Error for window size {window_size_seconds}s: {e}")
        import traceback
        traceback.print_exc()
        return {
            'window_size_seconds': window_size_seconds,
            'window_size_samples': window_size_samples,
            'status': 'error',
            'error': str(e)
        }


def test_window_sizes(quick_mode=False):
    """Test different window sizes from 1s to 30s.
    
    Args:
        quick_mode: If True, test fewer window sizes with fewer epochs for faster results
    """
    
    if quick_mode:
        # Quick mode: test fewer window sizes with fewer epochs
        window_sizes = [1, 2, 4, 6, 8, 10, 15, 20, 30]
        num_epochs = 20
        print("QUICK MODE: Testing 9 window sizes with 20 epochs each")
    else:
        # Full mode: test all window sizes
        window_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 25, 30]
        num_epochs = 30
        print("FULL MODE: Testing 16 window sizes with 30 epochs each")
    
    # Default hyperparameters
    default_hp = {
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': num_epochs,
        'overlap': 0.5
    }
    
    all_results = []
    
    print("="*80)
    print("FULCNNLOC Window Size Testing (1s to 30s)")
    print("="*80)
    print(f"Testing {len(window_sizes)} window sizes")
    print(f"Hyperparameters: {default_hp}")
    print("="*80)
    
    for ws_idx, window_size in enumerate(window_sizes):
        print(f"\n[{ws_idx+1}/{len(window_sizes)}] ", end="")
        
        result = run_training_window_size(
            window_size_seconds=window_size,
            **default_hp,
            output_dir="window_size_results"
        )
        
        all_results.append(result)
        
        # Save intermediate results
        df = pd.DataFrame(all_results)
        df.to_csv("window_size_results_all.csv", index=False)
    
    # Save final results
    df = pd.DataFrame(all_results)
    df.to_csv("window_size_results_final.csv", index=False)
    
    # Create summary report
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    successful_results = df[df['status'] == 'success'].copy()
    
    if len(successful_results) > 0:
        # Results by window size
        print("\nResults by Window Size:")
        print("-" * 80)
        print(f"{'Window':<10} {'Accuracy':<12} {'ROC-AUC':<12} {'F1-Score':<12} {'Balanced Acc':<12}")
        print("-" * 80)
        
        for _, row in successful_results.iterrows():
            ws = f"{row['window_size_seconds']}s"
            acc = f"{row['test_accuracy']:.4f}" if row['test_accuracy'] else "N/A"
            roc = f"{row['test_roc_auc']:.4f}" if row['test_roc_auc'] else "N/A"
            f1 = f"{row['test_f1_score']:.4f}" if row['test_f1_score'] else "N/A"
            bal = f"{row['test_balanced_accuracy']:.4f}" if row['test_balanced_accuracy'] else "N/A"
            print(f"{ws:<10} {acc:<12} {roc:<12} {f1:<12} {bal:<12}")
        
        # Overall best
        print("\nOverall Best Result:")
        print("-" * 80)
        best_overall = successful_results.loc[successful_results['test_accuracy'].idxmax()]
        print(f"  Window Size: {best_overall['window_size_seconds']}s ({best_overall['window_size_samples']} samples)")
        print(f"  Test Accuracy: {best_overall['test_accuracy']:.4f}")
        print(f"  Test ROC-AUC: {best_overall['test_roc_auc']:.4f if best_overall['test_roc_auc'] else 'N/A'}")
        print(f"  Test F1-Score: {best_overall['test_f1_score']:.4f if best_overall['test_f1_score'] else 'N/A'}")
        print(f"  Test Balanced Accuracy: {best_overall['test_balanced_accuracy']:.4f if best_overall['test_balanced_accuracy'] else 'N/A'}")
        print(f"  Best Validation Accuracy: {best_overall['best_val_accuracy']:.4f if best_overall['best_val_accuracy'] else 'N/A'}")
        
        # Create summary statistics
        summary_by_window = successful_results.groupby('window_size_seconds').agg({
            'test_accuracy': ['max', 'mean', 'std'],
            'test_roc_auc': ['max', 'mean', 'std'],
            'test_f1_score': ['max', 'mean', 'std']
        }).round(4)
        
        print("\nSummary Statistics by Window Size:")
        print("-" * 80)
        print(summary_by_window)
        
        summary_by_window.to_csv("window_size_summary.csv")
        
        # Create visualization-ready CSV
        viz_df = successful_results[['window_size_seconds', 'test_accuracy', 'test_roc_auc', 
                                     'test_f1_score', 'test_balanced_accuracy']].copy()
        viz_df = viz_df.sort_values('window_size_seconds')
        viz_df.to_csv("window_size_results_viz.csv", index=False)
    
    print(f"\nResults saved to:")
    print(f"  - window_size_results_final.csv (all results)")
    print(f"  - window_size_summary.csv (summary by window size)")
    print(f"  - window_size_results_viz.csv (visualization data)")
    
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test window sizes for FULCNNLOC')
    parser.add_argument('--quick', action='store_true', help='Quick test mode (fewer window sizes, fewer epochs)')
    args = parser.parse_args()
    
    results_df = test_window_sizes(quick_mode=args.quick)
    print("\n✓ Testing completed!")
