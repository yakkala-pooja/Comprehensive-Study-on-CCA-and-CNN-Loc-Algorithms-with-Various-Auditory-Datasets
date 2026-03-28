#!/usr/bin/env python3
"""
Training script for STAnet on DAS dataset.

Usage:
    python run_STAnet.py --tfrecord_dir <path_to_tfrecords> [options]
"""

import argparse
import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import torch
from torch.utils.data import DataLoader

# Import STAnet module
# First, verify STAnet.py exists
stanet_path = Path(__file__).parent / "STAnet.py"
if not stanet_path.exists():
    # Try current working directory
    stanet_path = Path.cwd() / "STAnet.py"
    if stanet_path.exists():
        sys.path.insert(0, str(stanet_path.parent))

try:
    from STAnet import STAnetModel, STAnetDataset, STAnetTrainer, device
except ImportError as e:
    print(f"Error importing STAnet: {e}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print(f"Looking for STAnet.py in: {stanet_path}")
    raise

def main():
    parser = argparse.ArgumentParser(description='Train STAnet on DAS dataset')
    parser.add_argument('--tfrecord_dir', type=str, required=True,
                       help='Directory containing TFRecord files')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--window_size', type=int, default=32,
                       help='Window size for EEG data')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Overlap ratio for sliding windows')
    parser.add_argument('--num_channels', type=int, default=64,
                       help='Number of EEG channels')
    parser.add_argument('--time_steps', type=int, default=32,
                       help='Number of time steps')
    parser.add_argument('--num_features', type=int, default=5,
                       help='Number of frequency features')
    parser.add_argument('--gcn_hidden', type=int, default=64,
                       help='GCN hidden dimension')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--output_dir', type=str, default='stanet_results',
                       help='Output directory for results')
    parser.add_argument('--save_model', action='store_true',
                       help='Save the trained model')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("STAnet Training on DAS Dataset")
    print("=" * 80)
    print(f"TFRecord directory: {args.tfrecord_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Device: {device}")
    print("=" * 80)
    
    # Load datasets
    print("\nLoading datasets...")
    try:
        train_dataset = STAnetDataset(
            tfrecord_dir=args.tfrecord_dir,
            mode='train',
            window_size=args.window_size,
            overlap=args.overlap,
            transform_eeg=True
        )
        
        val_dataset = STAnetDataset(
            tfrecord_dir=args.tfrecord_dir,
            mode='val',
            window_size=args.window_size,
            overlap=args.overlap,
            transform_eeg=True
        )
        
        test_dataset = STAnetDataset(
            tfrecord_dir=args.tfrecord_dir,
            mode='test',
            window_size=args.window_size,
            overlap=args.overlap,
            transform_eeg=True
        )
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Trying with mode='full'...")
        # Fallback: use full dataset and split manually
        full_dataset = STAnetDataset(
            tfrecord_dir=args.tfrecord_dir,
            mode='full',
            window_size=args.window_size,
            overlap=args.overlap,
            transform_eeg=True
        )
        
        # Split dataset indices manually
        total_size = len(full_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        
        indices = torch.randperm(total_size).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model
    print("\nCreating STAnet model...")
    model = STAnetModel(
        num_channels=args.num_channels,
        time_steps=args.time_steps,
        num_features=args.num_features,
        gcn_hidden=args.gcn_hidden,
        num_classes=2,
        dropout_rate=args.dropout_rate
    )
    
    # Create trainer
    trainer = STAnetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        device=device
    )
    
    # Train model
    print("\n" + "=" * 80)
    print("STARTING STAnet TRAINING")
    print("=" * 80)
    test_metrics = trainer.train()
    
    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL TRAINING SUMMARY")
    print("=" * 80)
    
    if trainer.train_accs:
        print(f"\nTraining Metrics:")
        print(f"  - Final Train Loss: {trainer.train_losses[-1]:.4f}")
        print(f"  - Final Train Accuracy: {trainer.train_accs[-1]:.2f}%")
        print(f"  - Best Train Accuracy: {max(trainer.train_accs):.2f}%")
    
    if trainer.val_accs:
        print(f"\nValidation Metrics:")
        print(f"  - Final Val Loss: {trainer.val_losses[-1]:.4f}")
        print(f"  - Final Val Accuracy: {trainer.val_accs[-1]:.2f}%")
        print(f"  - Best Val Accuracy: {max(trainer.val_accs):.2f}%")
    
    if test_metrics:
        print(f"\nTest Metrics:")
        print(f"  - Test Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
        print(f"  - Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
        print(f"  - F1 Score: {test_metrics['f1_score']:.4f}")
        print(f"  - Precision: {test_metrics['precision_weighted']:.4f}")
        print(f"  - Recall: {test_metrics['recall_weighted']:.4f}")
        print(f"  - ROC-AUC: {test_metrics['roc_auc']:.4f}")
        print(f"  - Matthews Correlation: {test_metrics['matthews_corrcoef']:.4f}")
        print(f"  - Cohen's Kappa: {test_metrics['cohens_kappa']:.4f}")
    
    # Save model if requested
    if args.save_model:
        model_path = output_dir / 'stanet_model.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'num_channels': args.num_channels,
                'time_steps': args.time_steps,
                'num_features': args.num_features,
                'gcn_hidden': args.gcn_hidden,
                'num_classes': 2,
                'dropout_rate': args.dropout_rate
            },
            'test_metrics': test_metrics,
            'train_history': {
                'train_losses': trainer.train_losses,
                'val_losses': trainer.val_losses,
                'train_accs': trainer.train_accs,
                'val_accs': trainer.val_accs
            }
        }, model_path)
        print(f"\nModel saved to {model_path}")
    
    # Save results
    results = {
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'train_accs': trainer.train_accs,
        'val_accs': trainer.val_accs,
        'test_metrics': test_metrics,
        'final_summary': {
            'best_train_acc': max(trainer.train_accs) if trainer.train_accs else None,
            'best_val_acc': max(trainer.val_accs) if trainer.val_accs else None,
            'final_train_acc': trainer.train_accs[-1] if trainer.train_accs else None,
            'final_val_acc': trainer.val_accs[-1] if trainer.val_accs else None,
            'test_accuracy': test_metrics['accuracy'] if test_metrics else None,
            'test_balanced_accuracy': test_metrics['balanced_accuracy'] if test_metrics else None,
            'test_f1_score': test_metrics['f1_score'] if test_metrics else None,
            'test_roc_auc': test_metrics['roc_auc'] if test_metrics else None
        }
    }
    
    import json
    results_path = output_dir / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    print("\n" + "=" * 80)
    print("STAnet TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()

