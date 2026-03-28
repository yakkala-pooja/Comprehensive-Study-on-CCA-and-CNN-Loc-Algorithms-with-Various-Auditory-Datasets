#!/usr/bin/env python3
"""
Debug script to test DAS TFRecord loading
"""

import tensorflow as tf
import numpy as np
from pathlib import Path

def debug_das_tfrecord_loading():
    """Debug DAS TFRecord loading to find the issue."""
    tfrecord_dir = Path('das_analysis_results_final/tfrecords')
    
    # Find TFRecord files
    tfrecord_files = []
    direct_files = list(tfrecord_dir.glob("*.tfrecords"))
    subdir_files = list(tfrecord_dir.glob("*/*.tfrecords"))
    
    if direct_files:
        tfrecord_files.extend(direct_files)
    if subdir_files:
        tfrecord_files.extend(subdir_files)
    
    print(f"Found {len(tfrecord_files)} TFRecord files")
    
    if not tfrecord_files:
        print("ERROR: No TFRecord files found!")
        return
    
    # Test first few files
    successful_files = 0
    failed_files = 0
    total_records = 0
    
    for i, tfrecord_file in enumerate(tfrecord_files[:5]):  # Test first 5 files
        print(f"\nTesting file {i+1}: {tfrecord_file}")
        
        try:
            dataset = tf.data.TFRecordDataset(str(tfrecord_file))
            records_in_file = 0
            
            for record in dataset:
                try:
                    example = tf.train.Example.FromString(record.numpy())
                    features = example.features.feature
                    
                    print(f"  Record {records_in_file}:")
                    print(f"    Available features: {list(features.keys())}")
                    
                    # Check required features
                    required_features = ['eeg', 'attended_ear']
                    missing_features = [f for f in required_features if f not in features]
                    if missing_features:
                        print(f"    ✗ Missing features: {missing_features}")
                        continue
                    
                    # Check EEG data
                    eeg_values = features['eeg'].float_list.value
                    print(f"    EEG values: {len(eeg_values)} values")
                    
                    if len(eeg_values) != 64:
                        print(f"    ✗ Wrong EEG channels: {len(eeg_values)}, expected 64")
                        continue
                    
                    # Check attended_ear
                    attended_ear_values = features['attended_ear'].bytes_list.value
                    if not attended_ear_values:
                        print(f"    ✗ No attended_ear values")
                        continue
                    
                    attended_ear = attended_ear_values[0].decode('utf-8')
                    print(f"    Attended ear: {attended_ear}")
                    
                    if attended_ear not in ['L', 'R']:
                        print(f"    ✗ Invalid attended_ear: {attended_ear}")
                        continue
                    
                    print(f"    ✓ Valid record")
                    records_in_file += 1
                    total_records += 1
                    
                    if records_in_file >= 3:  # Test first 3 records per file
                        break
                        
                except Exception as record_error:
                    print(f"    ✗ Record error: {record_error}")
                    continue
            
            if records_in_file > 0:
                successful_files += 1
                print(f"  ✓ File loaded successfully: {records_in_file} records")
            else:
                failed_files += 1
                print(f"  ✗ File failed to load any records")
                
        except Exception as e:
            failed_files += 1
            print(f"  ✗ File error: {e}")
    
    print(f"\nSummary:")
    print(f"  Successful files: {successful_files}")
    print(f"  Failed files: {failed_files}")
    print(f"  Total records: {total_records}")
    
    if total_records == 0:
        print("\n⚠ CRITICAL: No valid records found!")
        print("This suggests the TFRecord format is not compatible with DASCCA.")
    else:
        print(f"\n✓ Found {total_records} valid records - DASCCA should work!")

if __name__ == "__main__":
    debug_das_tfrecord_loading()
