#!/usr/bin/env python3
"""
Regenerate Fulsang TFRecords using FULPREPROCESSING
Stores TFRecords in Preprocessed_FulsangNorm directory
"""

import sys
from pathlib import Path
from FULPREPROCESSING import FulsangPreprocessor

def main():
    """Regenerate TFRecords with correct labels and 66 channels."""
    
    # Set up directories
    data_dir = "Data/Fulsang"  # Original data directory
    output_dir = "Preprocessed_FulsangNorm"  # New output directory
    
    print("=" * 80)
    print("REGENERATING FULSANG TFRECORDS")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Expected channels: 66")
    print(f"Label structure: Alternating every 1280 samples (20 seconds)")
    print("=" * 80)
    
    # Create preprocessor with new output directory
    preprocessor = FulsangPreprocessor(
        data_dir=data_dir,
        output_dir=output_dir
    )
    
    # Run preprocessing
    print("\nStarting preprocessing...")
    success = preprocessor.preprocess_dataset()
    
    if success:
        print("\n" + "=" * 80)
        print("SUCCESS: TFRecords regenerated successfully!")
        print(f"Output directory: {output_dir}/tfrecords")
        print("=" * 80)
        
        # Print statistics
        stats = preprocessor.processing_stats
        print(f"\nProcessing Statistics:")
        print(f"  Total files processed: {stats['total_files']}")
        print(f"  Successful files: {stats['successful_files']}")
        print(f"  Failed files: {stats['failed_files']}")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Validation failures: {stats['validation_failures']}")
        
        # Verify channel count
        print(f"\nVerifying channel count...")
        tfrecord_dir = Path(output_dir) / "tfrecords"
        tfrecord_files = list(tfrecord_dir.glob("*.tfrecords"))
        
        if tfrecord_files:
            import tensorflow as tf
            sample_file = tfrecord_files[0]
            dataset = tf.data.TFRecordDataset(str(sample_file))
            
            for record in dataset.take(1):
                example = tf.train.Example.FromString(record.numpy())
                features = example.features.feature
                eeg_values = features['eeg'].float_list.value
                print(f"  Sample from {sample_file.name}: {len(eeg_values)} channels")
                
                if len(eeg_values) == 66:
                    print(f"  Channel count is correct: 66")
                else:
                    print(f"  WARNING: Expected 66 channels, got {len(eeg_values)}")
                break
    else:
        print("\n" + "=" * 80)
        print("ERROR: Preprocessing failed!")
        print("=" * 80)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

