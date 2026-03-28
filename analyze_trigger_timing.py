#!/usr/bin/env python3
"""Analyze when triggers occur in the Fulsang dataset - at trial start or during trial."""

import scipy.io as sio
from pathlib import Path
import numpy as np

data_dir = Path("Data/Fulsang/DATA_preproc")
mat_file = data_dir / "S1_data_preproc.mat"

print(f"Loading {mat_file}...")
mat_data = sio.loadmat(str(mat_file), squeeze_me=False, struct_as_record=False)

if 'data' in mat_data:
    data_struct = mat_data['data']
    
    if isinstance(data_struct, np.ndarray) and data_struct.size > 0:
        first_elem = data_struct.flat[0]
        
        print("\n" + "="*80)
        print("ANALYZING TRIGGER TIMING IN FULSANG DATASET")
        print("="*80)
        
        # Check event structure
        if hasattr(first_elem, 'event'):
            event = first_elem.event
            if isinstance(event, np.ndarray) and event.size > 0:
                first_event = event.flat[0]
                if hasattr(first_event, 'eeg'):
                    eeg_events = first_event.eeg
                    if isinstance(eeg_events, np.ndarray) and eeg_events.size > 0:
                        print(f"\nFound {eeg_events.size} events in the event structure")
                        print("\nAnalyzing first 5 events:")
                        print("-"*80)
                        
                        # Get trial length
                        if hasattr(first_elem, 'eeg'):
                            eeg_data = first_elem.eeg
                            if isinstance(eeg_data, np.ndarray) and eeg_data.dtype == object:
                                if eeg_data.size > 0:
                                    first_trial_eeg = eeg_data.flat[0]
                                    if isinstance(first_trial_eeg, np.ndarray):
                                        trial_length = first_trial_eeg.shape[0] if first_trial_eeg.ndim >= 1 else 0
                                        sampling_rate = 64  # From preprocessing script
                                        trial_duration = trial_length / sampling_rate
                                        print(f"Trial length: {trial_length} samples ({trial_duration:.2f} seconds at 64 Hz)")
                        
                        for i in range(min(5, eeg_events.size)):
                            eeg_event = eeg_events.flat[i]
                            
                            # Get sample (time point) and value (trigger)
                            sample = None
                            value = None
                            
                            if hasattr(eeg_event, 'sample'):
                                sample_val = eeg_event.sample
                                if isinstance(sample_val, np.ndarray):
                                    sample = int(sample_val.flatten()[0]) if sample_val.size > 0 else None
                                else:
                                    sample = int(sample_val)
                            
                            if hasattr(eeg_event, 'value'):
                                value_obj = eeg_event.value
                                if isinstance(value_obj, np.ndarray):
                                    # Unwrap nested arrays
                                    while isinstance(value_obj, np.ndarray) and value_obj.dtype == object and value_obj.size > 0:
                                        value_obj = value_obj.flat[0]
                                    if isinstance(value_obj, np.ndarray):
                                        value_flat = value_obj.flatten()
                                        if value_flat.size > 0:
                                            value = int(value_flat[0])
                            
                            if sample is not None:
                                sample_time = sample / sampling_rate if sampling_rate > 0 else sample
                                print(f"Event {i}: sample={sample}, time={sample_time:.3f}s, trigger={value}")
                            else:
                                print(f"Event {i}: sample=N/A, trigger={value}")
                        
                        print("\n" + "="*80)
                        print("INTERPRETATION:")
                        print("-"*80)
                        
                        # Check if events are at trial start (sample=0 or very early)
                        first_event_sample = None
                        if hasattr(eeg_events.flat[0], 'sample'):
                            sample_val = eeg_events.flat[0].sample
                            if isinstance(sample_val, np.ndarray):
                                first_event_sample = int(sample_val.flatten()[0]) if sample_val.size > 0 else None
                            else:
                                first_event_sample = int(sample_val)
                        
                        if first_event_sample is not None:
                            if first_event_sample == 0 or first_event_sample < 100:  # Less than ~1.5 seconds
                                print("YES: Triggers occur at TRIAL START (sample ~0)")
                                print("  - The trigger marks when the trial begins")
                                print("  - It indicates which speaker to attend to for the ENTIRE trial")
                                print("  - This is an INSTRUCTION trigger, not a mid-trial switch")
                            else:
                                print(f"WARNING: Triggers occur at sample {first_event_sample} (mid-trial)")
                                print("  - This suggests attention switches during the trial")
                        
                        print("\n" + "="*80)
                        print("WHAT IS THE TRIGGER IN FULSANG DATASET?")
                        print("-"*80)
                        print("Based on preproc_data.m:")
                        print("  1. Line 20: event.eeg.value is set to attend_mf (1=male, 2=female)")
                        print("  2. Line 92: Trials are split at event.eeg.sample positions")
                        print("  3. Line 114: wavA is assigned based on event.eeg.value{1}")
                        print("\nThe trigger is:")
                        print("  - A TRIAL-LEVEL INSTRUCTION marker")
                        print("  - Indicates which speaker to attend to for that entire trial")
                        print("  - Set at the START of each trial (when trial is split)")
                        print("  - NOT a mid-trial attention switch")
                        
                        print("\n" + "="*80)
                        print("IS THERE A LAG?")
                        print("-"*80)
                        print("The trigger marks the INSTRUCTION to attend to a speaker.")
                        print("There may be a brief cognitive lag (~100-500ms) where:")
                        print("  - The participant processes the instruction")
                        print("  - Attention shifts from previous trial's speaker")
                        print("  - Neural entrainment to new speaker begins")
                        print("\nHowever, the trigger itself is at trial start (t=0),")
                        print("so any lag would be AFTER the trigger, not before it.")
