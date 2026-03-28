#!/usr/bin/env python3
"""Verify trigger values across multiple trials to see the pattern."""

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
        # data_struct might be (1,1) with a cell array inside
        # Check if first element is a cell array of trials
        first_elem = data_struct.flat[0]
        if isinstance(first_elem, np.ndarray) and first_elem.dtype == object:
            # It's a cell array
            n_trials = first_elem.size
            print(f"\nFound {n_trials} trials (cell array structure)")
            trials_array = first_elem
        else:
            # Direct array structure
            n_trials = data_struct.size
            print(f"\nFound {n_trials} trials (direct structure)")
            trials_array = data_struct
        print("\nTrigger values for first 20 trials:")
        print("-" * 60)
        
        triggers = []
        for i in range(min(20, n_trials)):
            if isinstance(trials_array, np.ndarray) and trials_array.dtype == object:
                trial = trials_array.flat[i]
            else:
                trial = trials_array.flat[i]
            trigger = None
            
            # Extract trigger from event structure
            if hasattr(trial, 'event'):
                event = trial.event
                if isinstance(event, np.ndarray) and event.size > 0:
                    first_event = event.flat[0]
                    if hasattr(first_event, 'eeg'):
                        eeg_events = first_event.eeg
                        if isinstance(eeg_events, np.ndarray) and eeg_events.size > 0:
                            first_eeg_event = eeg_events.flat[0]
                            if hasattr(first_eeg_event, 'value'):
                                value = first_eeg_event.value
                                if isinstance(value, np.ndarray):
                                    # Unwrap nested arrays
                                    while isinstance(value, np.ndarray) and value.dtype == object and value.size > 0:
                                        value = value.flat[0]
                                    if isinstance(value, np.ndarray):
                                        value_flat = value.flatten()
                                        if value_flat.size > 0:
                                            trigger = int(value_flat[0])
            
            triggers.append(trigger)
            print(f"Trial {i:2d}: Trigger = {trigger}")
        
        print("-" * 60)
        print(f"\nTrigger value distribution:")
        unique_triggers = set(triggers)
        for trig in sorted(unique_triggers):
            count = triggers.count(trig)
            print(f"  Trigger {trig}: {count} trials")
        
        print(f"\nTrial 0 trigger value: {triggers[0]}")
        print(f"Is trigger value 2 correct for trial 0? {'YES' if triggers[0] == 2 else 'NO - should be ' + str(triggers[0])}")
