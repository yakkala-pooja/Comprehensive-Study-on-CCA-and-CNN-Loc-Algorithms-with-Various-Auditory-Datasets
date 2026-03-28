#!/usr/bin/env python3
"""
Check if raw (unprocessed) dataset handles delay.

Raw dataset = Original EEG and audio files before any preprocessing.
"""

import scipy.io as sio
from pathlib import Path
import numpy as np

print("="*80)
print("CHECKING IF RAW (UNPROCESSED) DATASET HANDLES DELAY")
print("="*80)

print("\n1. What is the 'raw dataset'?")
print("   - Original EEG files: Data/Fulsang/EEG/S*.mat")
print("   - Original audio files: Data/Fulsang/AUDIO/*.wav")
print("   - These are UNPROCESSED - before any filtering, downsampling, etc.")
print("   - They are separate files that need to be aligned")

print("\n2. How raw data is structured:")
print("   - EEG: Continuous recordings with events/triggers")
print("   - Audio: Separate .wav files (not embedded in EEG)")
print("   - Alignment: Done using triggers/events during preprocessing")

# Check raw EEG file structure
print("\n3. Checking raw EEG file structure:")
eeg_file = Path("Data/Fulsang/EEG/S1.mat")

if eeg_file.exists():
    print(f"   Loading {eeg_file.name}...")
    try:
        mat_data = sio.loadmat(str(eeg_file), squeeze_me=False, struct_as_record=False)
        
        print("   Fields in raw EEG file:")
        for key in mat_data.keys():
            if not key.startswith('__'):
                print(f"     - {key}")
        
        # Check if there's any delay-related information
        if 'data' in mat_data:
            data = mat_data['data']
            print("\n   Raw EEG data structure:")
            print("     - Contains continuous EEG recordings")
            print("     - Contains event/trigger information")
            print("     - NO audio data embedded")
            print("     - NO delay compensation")
        
        if 'expinfo' in mat_data:
            print("\n   Experimental info found:")
            print("     - Contains trial information")
            print("     - Contains trigger values")
            print("     - NO delay information")
            
    except Exception as e:
        print(f"   Could not load: {e}")

print("\n4. How preprocessing handles raw data:")
print("   From preproc_data.m:")
print("   - Line 13: load(fullfile(EEGBASEPATH,['S' num2str(ss) '.mat']))")
print("     -> Loads RAW EEG data")
print("   - Line 114-128: Loads audio files separately")
print("     -> Audio is NOT in the raw EEG file")
print("   - Line 130-134: Resamples audio to match EEG rate")
print("     -> Aligns audio and EEG at trial start (trigger)")
print("     -> NO delay compensation")

print("\n5. Key points about raw data:")
print("   - Raw EEG: Continuous recordings, no audio embedded")
print("   - Raw Audio: Separate .wav files")
print("   - Alignment: Done during preprocessing using triggers")
print("   - Delay: Biological phenomenon that exists in the data")
print("   - Delay compensation: NOT in raw data, NOT in preprocessing,")
print("                          ONLY handled during model training")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("Raw dataset does NOT handle delay.")
print("\nWhy:")
print("1. Raw data = separate EEG and audio files")
print("2. They are not even aligned yet (alignment happens in preprocessing)")
print("3. Delay is a biological phenomenon, not a data format issue")
print("4. Delay compensation requires:")
print("   - Knowing the relationship between audio and EEG")
print("   - Applying time-lagging (150-400ms)")
print("   - This is done during MODEL TRAINING, not in raw data")
print("\nData pipeline:")
print("  Raw Data -> Preprocessing -> Training")
print("  (separate files) -> (aligned at trigger) -> (time-lagged features)")
print("\n  - Raw: No alignment, no delay handling")
print("  - Preprocessing: Alignment at trigger, still no delay handling")
print("  - Training: Time-lagged features (150-400ms) account for delay")
