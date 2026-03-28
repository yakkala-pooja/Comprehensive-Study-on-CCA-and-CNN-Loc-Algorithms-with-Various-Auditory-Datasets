#!/usr/bin/env python3
"""Check what triggers exist in Das dataset."""

import scipy.io as sio
from pathlib import Path
import numpy as np

data_dir = Path("Data/Das/4004271")
mat_file = data_dir / "S1.mat"

print(f"Loading {mat_file}...")
mat_data = sio.loadmat(str(mat_file), squeeze_me=True, struct_as_record=False)

if 'trials' in mat_data:
    trials = mat_data['trials']
    
    if not isinstance(trials, np.ndarray):
        trials = [trials]
    else:
        trials = trials.flatten()
    
    print(f"\nFound {len(trials)} trials")
    print("\n" + "="*80)
    print("ANALYZING DAS DATASET TRIGGERS")
    print("="*80)
    
    print("\nFirst 5 trials structure:")
    print("-"*80)
    
    for i in range(min(5, len(trials))):
        trial = trials[i]
        print(f"\nTrial {i}:")
        print(f"  Attributes: {[k for k in dir(trial) if not k.startswith('_')]}")
        
        # Check for attended_ear
        attended_ear = None
        if hasattr(trial, 'attended_ear'):
            attended_ear = trial.attended_ear
            if isinstance(attended_ear, np.ndarray):
                attended_ear = attended_ear.item() if attended_ear.size == 1 else str(attended_ear[0])
        
        # Check for trigger
        trigger = None
        if hasattr(trial, 'trigger'):
            trigger = trial.trigger
        elif hasattr(trial, 'Trigger'):
            trigger = trial.Trigger
        
        # Check for event structure
        event = None
        if hasattr(trial, 'event'):
            event = trial.event
        elif hasattr(trial, 'Event'):
            event = trial.Event
        
        # Check RawData for triggers
        rawdata_trigger = None
        if hasattr(trial, 'RawData'):
            rawdata = trial.RawData
            if hasattr(rawdata, 'trigger'):
                rawdata_trigger = rawdata.trigger
            elif hasattr(rawdata, 'Trigger'):
                rawdata_trigger = rawdata.Trigger
        
        print(f"  attended_ear: {attended_ear}")
        print(f"  trigger (trial level): {trigger}")
        print(f"  trigger (RawData): {rawdata_trigger}")
        print(f"  event: {event}")
        
        # Check stimuli
        if hasattr(trial, 'stimuli'):
            stimuli = trial.stimuli
            print(f"  stimuli: {stimuli}")
    
    print("\n" + "="*80)
    print("CONCLUSION:")
    print("-"*80)
    print("Das dataset uses 'attended_ear' (L/R) instead of numeric triggers.")
    print("The trigger in Das dataset is SPATIAL (left/right ear), not gender-based.")
    print("  - 'L' = Left ear attended")
    print("  - 'R' = Right ear attended")
    print("\nUnlike Fulsang (trigger = attend_mf: 1=male, 2=female),")
    print("Das uses spatial attention (left vs right ear) as the 'trigger'.")
