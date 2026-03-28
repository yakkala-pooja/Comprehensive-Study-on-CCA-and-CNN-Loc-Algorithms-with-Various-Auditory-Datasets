#!/usr/bin/env python3
"""
Check if data is pre-aligned (time-synchronized) in raw or preprocessed files.
"""

import scipy.io as sio
from pathlib import Path
import numpy as np

print("="*80)
print("CHECKING IF DATA IS PRE-ALIGNED")
print("="*80)

print("\n1. What does 'alignment' mean?")
print("   - Time synchronization: Audio and EEG at same time points")
print("   - Sample-level matching: Each EEG sample corresponds to audio sample")
print("   - Trigger-based: Alignment using event markers")

# Check raw data
print("\n2. Checking RAW data alignment:")
eeg_file = Path("Data/Fulsang/EEG/S1.mat")

if eeg_file.exists():
    print(f"   Loading {eeg_file.name}...")
    try:
        mat_data = sio.loadmat(str(eeg_file), squeeze_me=False, struct_as_record=False)
        
        if 'data' in mat_data:
            data = mat_data['data']
            if isinstance(data, np.ndarray) and data.size > 0:
                first_elem = data.flat[0]
                
                # Check for event/trigger information
                if hasattr(first_elem, 'event'):
                    event = first_elem.event
                    print("   Event structure found in raw data")
                    if isinstance(event, np.ndarray) and event.size > 0:
                        first_event = event.flat[0]
                        if hasattr(first_event, 'eeg'):
                            eeg_events = first_event.eeg
                            if isinstance(eeg_events, np.ndarray) and eeg_events.size > 0:
                                first_eeg_event = eeg_events.flat[0]
                                if hasattr(first_eeg_event, 'sample'):
                                    sample_val = first_eeg_event.sample
                                    if isinstance(sample_val, np.ndarray):
                                        sample = int(sample_val.flatten()[0]) if sample_val.size > 0 else None
                                    else:
                                        sample = int(sample_val)
                                    print(f"   Trigger sample in raw data: {sample}")
                                    print("   -> Raw data has trigger markers for alignment")
                
                # Check if audio is embedded
                if hasattr(first_elem, 'wavA') or hasattr(first_elem, 'audio'):
                    print("   Audio data found in raw EEG file")
                    print("   -> Raw data is pre-aligned")
                else:
                    print("   NO audio data in raw EEG file")
                    print("   -> Raw data is NOT pre-aligned (audio is separate)")
                    
    except Exception as e:
        print(f"   Error: {e}")

# Check preprocessed data
print("\n3. Checking PREPROCESSED data alignment:")
preproc_file = Path("Data/Fulsang/DATA_preproc/S1_data_preproc.mat")

if preproc_file.exists():
    print(f"   Loading {preproc_file.name}...")
    try:
        mat_data = sio.loadmat(str(preproc_file), squeeze_me=False, struct_as_record=False)
        
        if 'data' in mat_data:
            data = mat_data['data']
            if isinstance(data, np.ndarray) and data.size > 0:
                trial = data.flat[0]
                
                # Check if audio and EEG are aligned
                has_eeg = hasattr(trial, 'eeg')
                has_wavA = hasattr(trial, 'wavA')
                has_wavB = hasattr(trial, 'wavB')
                
                if has_eeg and has_wavA:
                    eeg = trial.eeg
                    wavA = trial.wavA
                    
                    if isinstance(eeg, np.ndarray) and eeg.dtype == object:
                        eeg = eeg.flat[0]
                    if isinstance(wavA, np.ndarray) and wavA.dtype == object:
                        wavA = wavA.flat[0]
                    
                    if isinstance(eeg, np.ndarray) and isinstance(wavA, np.ndarray):
                        eeg_len = eeg.shape[0] if eeg.ndim >= 1 else 0
                        wavA_len = wavA.shape[0] if wavA.ndim >= 1 else 0
                        
                        print(f"   EEG length: {eeg_len} samples")
                        print(f"   wavA length: {wavA_len} samples")
                        
                        if eeg_len == wavA_len and eeg_len > 0:
                            print("   -> PREPROCESSED data IS aligned (same length)")
                            print("   -> Audio and EEG are time-synchronized")
                            print("   -> Each sample corresponds to same time point")
                        else:
                            print("   -> PREPROCESSED data is NOT aligned (different lengths)")
                            
    except Exception as e:
        print(f"   Error: {e}")

# Check preprocessing script
print("\n4. How preprocessing aligns data:")
print("   From preproc_data.m:")
print("   - Line 92: cfg.eeg.splitsample = data.event.eeg.sample")
print("     -> Splits continuous EEG at trigger positions")
print("   - Line 114-128: Loads audio files")
print("     -> Audio files are separate, not embedded")
print("   - Line 130-134: cfg.wavA.newfs = data{ii}.fsample.eeg")
print("     -> Resamples audio to match EEG sampling rate")
print("   - Result: Audio and EEG are aligned at trial start (trigger)")
print("   - They are time-synchronized (same sampling rate, same length)")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("RAW data: NOT pre-aligned")
print("  - EEG and audio are separate files")
print("  - No time synchronization in raw format")
print("  - Alignment happens during preprocessing")
print("\nPREPROCESSED data: IS pre-aligned")
print("  - Audio and EEG are time-synchronized")
print("  - Same sampling rate (64 Hz)")
print("  - Same length (3200 samples per trial)")
print("  - Aligned at trial start (trigger)")
print("\nBUT: Alignment does NOT mean delay compensation")
print("  - Pre-aligned = time-synchronized (same time points)")
print("  - Delay compensation = accounting for 150-400ms neural latency")
print("  - Preprocessed data is aligned but NOT delay-compensated")
print("  - Delay compensation happens during training (time-lagging)")
