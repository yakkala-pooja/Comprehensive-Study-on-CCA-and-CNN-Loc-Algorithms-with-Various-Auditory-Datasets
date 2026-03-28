#!/usr/bin/env python3
"""
Diagnostic script to identify CCA analysis issues in the DAS dataset.
This script will help identify why correlations are so low.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('telluride_decoding')

from das_preprocessor import DasPreprocessor
from run_das_cca_analysis_final import DasCCAAnalyzer

def diagnose_data_quality():
    """Diagnose data quality issues that could cause poor CCA results."""
    print("=" * 60)
    print("DIAGNOSING CCA ISSUES")
    print("=" * 60)
    
    # 1. Check preprocessing parameters
    print("\n1. CURRENT PREPROCESSING PARAMETERS:")
    print("-" * 40)
    preprocessor = DasPreprocessor()
    params = preprocessor.params
    
    print(f"EEG rereferencing: {params['rereference']}")
    print(f"Bandpass filter: {params['highpass']}-{params['lowpass']} Hz")
    print(f"Target sample rate: {params['targetSampleRate']} Hz")
    print(f"Gammatone filters: {len(params['freqs'])}")
    print(f"Frequency range: {params['freqs'][0]:.1f} - {params['freqs'][-1]:.1f} Hz")
    print(f"Power law exponent: {params['power']}")
    
    # 2. Check if TFRecord data exists
    print("\n2. CHECKING TFRecord DATA:")
    print("-" * 40)
    tfrecord_dir = Path("das_analysis_results_final/tfrecords")
    if tfrecord_dir.exists():
        tfrecord_files = list(tfrecord_dir.glob("*.tfrecords"))
        print(f"Found {len(tfrecord_files)} TFRecord files")
        
        if tfrecord_files:
            # Load a sample to check data quality
            import tensorflow as tf
            dataset = tf.data.TFRecordDataset(str(tfrecord_files[0]))
            
            sample_count = 0
            eeg_shapes = []
            envelope_shapes = []
            
            for record in dataset.take(10):  # Check first 10 samples
                example = tf.train.Example.FromString(record.numpy())
                
                # Check EEG data
                if 'eeg' in example.features.feature:
                    eeg_data = np.array(example.features.feature['eeg'].float_list.value)
                    eeg_shapes.append(eeg_data.shape)
                
                # Check envelope data
                if 'attended_envelope_full' in example.features.feature:
                    envelope_data = np.array(example.features.feature['attended_envelope_full'].float_list.value)
                    envelope_shapes.append(envelope_data.shape)
                elif 'intensity' in example.features.feature:
                    intensity_data = np.array(example.features.feature['intensity'].float_list.value)
                    envelope_shapes.append(intensity_data.shape)
                
                sample_count += 1
            
            print(f"Sample shapes - EEG: {set(eeg_shapes)}, Envelope: {set(envelope_shapes)}")
            
            # Check for data quality issues
            if eeg_shapes:
                eeg_sample = np.array(example.features.feature['eeg'].float_list.value)
                print(f"EEG data stats: mean={np.mean(eeg_sample):.4f}, std={np.std(eeg_sample):.4f}")
                print(f"EEG data range: [{np.min(eeg_sample):.4f}, {np.max(eeg_sample):.4f}]")
                
                # Check for constant or near-constant data
                if np.std(eeg_sample) < 1e-6:
                    print("⚠️  WARNING: EEG data appears to be constant or near-constant!")
                
            if envelope_shapes:
                if 'attended_envelope_full' in example.features.feature:
                    env_sample = np.array(example.features.feature['attended_envelope_full'].float_list.value)
                else:
                    env_sample = np.array(example.features.feature['intensity'].float_list.value)
                print(f"Envelope data stats: mean={np.mean(env_sample):.4f}, std={np.std(env_sample):.4f}")
                print(f"Envelope data range: [{np.min(env_sample):.4f}, {np.max(env_sample):.4f}]")
                
                # Check for constant or near-constant data
                if np.std(env_sample) < 1e-6:
                    print("⚠️  WARNING: Envelope data appears to be constant or near-constant!")
    else:
        print("❌ TFRecord directory not found!")
        return False
    
    return True

def suggest_fixes():
    """Suggest specific fixes for the CCA issues."""
    print("\n3. SUGGESTED FIXES:")
    print("-" * 40)
    
    print("🔧 EEG PREPROCESSING FIXES:")
    print("   - Try different rereferencing: 'mean' instead of 'Cz'")
    print("   - Widen bandpass filter: 0.5-12 Hz instead of 1-9 Hz")
    print("   - Increase sample rate: 64 Hz instead of 32 Hz")
    print("   - Check for electrode artifacts and bad channels")
    
    print("\n🔧 AUDIO PREPROCESSING FIXES:")
    print("   - Verify gammatone filterbank is working correctly")
    print("   - Check if power law exponent (0.6) is appropriate")
    print("   - Ensure temporal alignment between EEG and audio")
    print("   - Verify stimulus files are being loaded correctly")
    
    print("\n🔧 CCA ANALYSIS FIXES:")
    print("   - Use regularization λ = 0.1-0.5 (not 0.001 or 10.0)")
    print("   - Increase CCA dimensions to 10-20")
    print("   - Check temporal context windows (pre/post context)")
    print("   - Verify data is properly normalized")
    
    print("\n🔧 DATA QUALITY CHECKS:")
    print("   - Ensure EEG and audio are temporally synchronized")
    print("   - Check for missing or corrupted data")
    print("   - Verify attention labels are correct")
    print("   - Check if data contains sufficient variability")

def create_improved_analysis():
    """Create an improved analysis script with better parameters."""
    print("\n4. CREATING IMPROVED ANALYSIS:")
    print("-" * 40)
    
    improved_params = {
        'rereference': 'mean',  # Instead of 'Cz'
        'highpass': 0.5,        # Instead of 1 Hz
        'lowpass': 12,          # Instead of 9 Hz
        'targetSampleRate': 64, # Instead of 32 Hz
        'power': 0.6,           # Keep same
        'spacing': 1.5,         # Keep same
    }
    
    print("Improved parameters:")
    for key, value in improved_params.items():
        print(f"  {key}: {value}")
    
    return improved_params

if __name__ == "__main__":
    print("Starting CCA diagnostic analysis...")
    
    # Run diagnostics
    data_ok = diagnose_data_quality()
    
    if data_ok:
        suggest_fixes()
        improved_params = create_improved_analysis()
        
        print("\n" + "=" * 60)
        print("DIAGNOSTIC COMPLETE")
        print("=" * 60)
        print("Next steps:")
        print("1. Implement the suggested preprocessing fixes")
        print("2. Re-run the analysis with improved parameters")
        print("3. Check for temporal alignment issues")
        print("4. Verify data quality and synchronization")
    else:
        print("❌ Cannot proceed - data issues detected")
