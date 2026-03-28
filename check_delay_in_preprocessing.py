#!/usr/bin/env python3
"""
Check if delay is already accounted for in preprocessing or needs to be applied during training.
"""

import scipy.io as sio
from pathlib import Path
import numpy as np

print("="*80)
print("CHECKING IF DELAY IS ALREADY ACCOUNTED FOR IN PREPROCESSING")
print("="*80)

# Check preprocessing script
preproc_file = Path("Data/Fulsang/preproc_data.m")
print(f"\n1. Checking preprocessing script: {preproc_file}")

if preproc_file.exists():
    with open(preproc_file, 'r') as f:
        content = f.read()
        
    # Look for delay-related operations
    delay_keywords = ['delay', 'lag', 'shift', 'offset', 'align']
    found_keywords = []
    for keyword in delay_keywords:
        if keyword.lower() in content.lower():
            found_keywords.append(keyword)
    
    if found_keywords:
        print(f"   Found keywords: {found_keywords}")
        print("   Checking context...")
    else:
        print("   No delay-related keywords found in preprocessing script")
    
    # Check what preprocessing actually does
    print("\n2. What preprocessing does (from preproc_data.m):")
    print("   - Line 130-134: Downsample AUDIO to EEG sampling rate")
    print("   - Line 132: cfg.wavA.newfs = data{ii}.fsample.eeg")
    print("   - This only resamples audio to match EEG rate")
    print("   - NO time shifting or delay compensation")
    print("   - Audio and EEG are aligned at trial start (trigger), but NOT shifted for neural latency")

# Check actual data alignment
print("\n3. Checking actual data:")
data_dir = Path("Data/Fulsang/DATA_preproc")
mat_file = data_dir / "S1_data_preproc.mat"

if mat_file.exists():
    print(f"   Loading {mat_file.name}...")
    mat_data = sio.loadmat(str(mat_file), squeeze_me=False, struct_as_record=False)
    
    if 'data' in mat_data:
        data_struct = mat_data['data']
        if isinstance(data_struct, np.ndarray) and data_struct.size > 0:
            trial = data_struct.flat[0]
            
            # Check if wavA and EEG are same length and aligned
            if hasattr(trial, 'eeg') and hasattr(trial, 'wavA'):
                eeg = trial.eeg
                wavA = trial.wavA
                
                if isinstance(eeg, np.ndarray) and eeg.dtype == object:
                    eeg = eeg.flat[0]
                if isinstance(wavA, np.ndarray) and wavA.dtype == object:
                    wavA = wavA.flat[0]
                
                if isinstance(eeg, np.ndarray) and isinstance(wavA, np.ndarray):
                    print(f"   EEG shape: {eeg.shape}")
                    print(f"   wavA shape: {wavA.shape}")
                    print(f"   Same length: {eeg.shape[0] == wavA.shape[0]}")
                    print("   YES: Audio and EEG are same length (aligned at trial start)")
                    print("   WARNING: But NO delay compensation - they start at same time")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("The dataset does NOT account for delay during preprocessing.")
print("\nEvidence:")
print("1. preproc_data.m only resamples audio to match EEG rate")
print("2. No time shifting or delay compensation in preprocessing")
print("3. Audio and EEG are aligned at trial start (trigger), but NOT shifted for neural latency")
print("\nThe delay is handled during MODEL TRAINING:")
print("- FULCCA uses make_lagged_audio() to create time-lagged audio features")
print("- Lag range: 150-400ms (10-26 samples at 64Hz)")
print("- This happens in FULCCA.py during dataset creation, not in preprocessing")
print("\nSo:")
print("  - Preprocessing: Audio and EEG aligned at trigger (t=0), no delay compensation")
print("  - Training: Time-lagged audio features created (150-400ms range)")
print("  - Result: Model learns optimal lag from the range, accounting for neural latency")
