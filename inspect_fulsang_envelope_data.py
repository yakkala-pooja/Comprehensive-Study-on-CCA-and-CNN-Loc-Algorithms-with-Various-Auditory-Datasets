#!/usr/bin/env python3
"""
Fulsang Envelope Data Inspector
Checks what envelope data is available in TFRecord files to see if multi-band envelopes exist.
"""

import tensorflow as tf
import numpy as np
import sys
from pathlib import Path

def inspect_envelope_data(tfrecord_file):
    """Inspect envelope data in a single TFRecord file."""
    print(f"\nInspecting envelope data in: {tfrecord_file}")
    
    try:
        dataset = tf.data.TFRecordDataset(str(tfrecord_file))
        record_count = 0
        
        for record in dataset:
            try:
                example = tf.train.Example.FromString(record.numpy())
                features = example.features.feature
                
                print(f"\nRecord {record_count}:")
                print(f"  Available features: {list(features.keys())}")
                
                # Check for envelope-related features
                envelope_features = [key for key in features.keys() if 'envelope' in key.lower() or 'wav' in key.lower() or 'audio' in key.lower()]
                print(f"  Envelope-related features: {envelope_features}")
                
                # Inspect each envelope feature
                for feature_name in envelope_features:
                    feature = features[feature_name]
                    
                    if hasattr(feature, 'float_list'):
                        values = feature.float_list.value
                        print(f"  {feature_name} (float_list): length={len(values)}")
                        if len(values) > 0:
                            print(f"    Range: [{min(values):.4f}, {max(values):.4f}]")
                            print(f"    Mean: {np.mean(values):.4f}, Std: {np.std(values):.4f}")
                    elif hasattr(feature, 'bytes_list'):
                        values = feature.bytes_list.value
                        print(f"  {feature_name} (bytes_list): length={len(values)}")
                    elif hasattr(feature, 'int64_list'):
                        values = feature.int64_list.value
                        print(f"  {feature_name} (int64_list): length={len(values)}")
                
                # Check EEG data for comparison
                if 'eeg' in features:
                    eeg_values = features['eeg'].float_list.value
                    print(f"  EEG data: length={len(eeg_values)}")
                
                record_count += 1
                if record_count >= 3:  # Check first 3 records
                    break
                    
            except Exception as e:
                print(f"Error parsing record {record_count}: {e}")
                record_count += 1
                if record_count >= 3:
                    break
                    
    except Exception as e:
        print(f"Error reading file: {e}")

def main():
    """Main function to inspect envelope data."""
    tfrecord_dir = Path("fulsang_analysis_results_final/tfrecords")
    
    if not tfrecord_dir.exists():
        print(f"Directory not found: {tfrecord_dir}")
        return
    
    tfrecord_files = list(tfrecord_dir.glob("*.tfrecords"))
    print(f"Found {len(tfrecord_files)} TFRecord files")
    
    if not tfrecord_files:
        print("No TFRecord files found!")
        return
    
    # Inspect first 5 files
    files_to_inspect = tfrecord_files[:5]
    print(f"Inspecting first {len(files_to_inspect)} files...")
    
    for tfrecord_file in files_to_inspect:
        inspect_envelope_data(tfrecord_file)
    
    print("\n" + "="*80)
    print("ENVELOPE DATA ANALYSIS SUMMARY")
    print("="*80)
    
    # Check all files for envelope features
    all_envelope_features = set()
    for tfrecord_file in tfrecord_files[:20]:  # Check first 20 files
        try:
            dataset = tf.data.TFRecordDataset(str(tfrecord_file))
            for record in dataset:
                example = tf.train.Example.FromString(record.numpy())
                features = example.features.feature
                
                # Find envelope-related features
                envelope_features = [key for key in features.keys() if 'envelope' in key.lower() or 'wav' in key.lower() or 'audio' in key.lower()]
                all_envelope_features.update(envelope_features)
                break  # Just check first record
        except:
            continue
    
    print(f"Envelope-related features found across files: {sorted(all_envelope_features)}")
    
    if all_envelope_features:
        print("\n✓ ENVELOPE DATA AVAILABLE!")
        print("You can potentially use envelope data to improve accuracy.")
        print("\nRecommendations:")
        print("1. Modify CNN-LOC to include envelope features")
        print("2. Use envelope data as additional input")
        print("3. This should improve attention decoding accuracy")
    else:
        print("\n✗ NO ENVELOPE DATA FOUND")
        print("The TFRecord files only contain EEG data.")
        print("To get envelope data, you would need to:")
        print("1. Re-run preprocessing with envelope extraction")
        print("2. Use the original audio files to create envelopes")
        print("3. Modify the TFRecord creation process")

if __name__ == "__main__":
    main()
