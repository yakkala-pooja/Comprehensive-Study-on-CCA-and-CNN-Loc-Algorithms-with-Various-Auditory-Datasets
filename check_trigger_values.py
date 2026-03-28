#!/usr/bin/env python3
"""Quick script to check what trigger values exist in the raw MATLAB files."""

import scipy.io as sio
from pathlib import Path
import numpy as np

data_dir = Path("Data/Fulsang/DATA_preproc")
mat_file = data_dir / "S1_data_preproc.mat"

print(f"Loading {mat_file}...")
mat_data = sio.loadmat(str(mat_file), squeeze_me=False, struct_as_record=False)

print("\nKeys in mat_data:")
for key in mat_data.keys():
    if not key.startswith('__'):
        print(f"  {key}")

if 'data' in mat_data:
    data_struct = mat_data['data']
    print(f"\ndata_struct type: {type(data_struct)}")
    print(f"data_struct shape: {data_struct.shape if hasattr(data_struct, 'shape') else 'N/A'}")
    
    if isinstance(data_struct, np.ndarray) and data_struct.size > 0:
        first_trial = data_struct.flat[0]
        print(f"\nFirst trial type: {type(first_trial)}")
        print(f"First trial attributes: {[k for k in dir(first_trial) if not k.startswith('_')]}")
        
        # Check for expinfo
        if 'expinfo' in mat_data:
            expinfo = mat_data['expinfo']
            print(f"\nexpinfo type: {type(expinfo)}")
            if hasattr(expinfo, 'dtype') and expinfo.dtype.names:
                print(f"expinfo fields: {expinfo.dtype.names}")
                if 'trigger' in expinfo.dtype.names:
                    trigger_val = expinfo['trigger']
                    print(f"\nTrigger value in expinfo: {trigger_val}")
                    print(f"Trigger type: {type(trigger_val)}")
                    print(f"Trigger shape: {trigger_val.shape if hasattr(trigger_val, 'shape') else 'N/A'}")
                    if isinstance(trigger_val, np.ndarray):
                        print(f"Trigger array: {trigger_val}")
                        if trigger_val.size > 0:
                            print(f"First few trigger values: {trigger_val.flatten()[:10]}")
        
        # Check for event structure in first trial
        print("\n" + "="*60)
        print("Checking event structure...")
        if hasattr(first_trial, 'event'):
            event = first_trial.event
            print(f"Event structure found in first trial")
            print(f"Event type: {type(event)}")
            print(f"Event shape: {event.shape if hasattr(event, 'shape') else 'N/A'}")
            
            # Check event structure more deeply
            if isinstance(event, np.ndarray):
                if event.size > 0:
                    first_event = event.flat[0]
                    print(f"First event type: {type(first_event)}")
                    print(f"First event attributes: {[k for k in dir(first_event) if not k.startswith('_')]}")
                    
                    if hasattr(first_event, 'eeg'):
                        eeg_event = first_event.eeg
                        print(f"\nEvent.eeg type: {type(eeg_event)}")
                        print(f"Event.eeg shape: {eeg_event.shape if hasattr(eeg_event, 'shape') else 'N/A'}")
                        
                        if isinstance(eeg_event, np.ndarray) and eeg_event.size > 0:
                            first_eeg = eeg_event.flat[0]
                            print(f"First eeg event type: {type(first_eeg)}")
                            print(f"First eeg event attributes: {[k for k in dir(first_eeg) if not k.startswith('_')]}")
                            
                            if hasattr(first_eeg, 'value'):
                                value = first_eeg.value
                                print(f"\nEvent.eeg.value: {value}")
                                print(f"Event.eeg.value type: {type(value)}")
                                if isinstance(value, np.ndarray):
                                    print(f"Event.eeg.value shape: {value.shape}")
                                    if value.dtype == object and value.size > 0:
                                        print(f"First value element: {value.flat[0]}")
                                        print(f"First value element type: {type(value.flat[0])}")
                            
                            # Check for trigger in eeg event
                            if hasattr(first_eeg, 'trigger'):
                                print(f"\nEvent.eeg.trigger: {first_eeg.trigger}")
                            
                            # Check for type field (might contain trigger info)
                            if hasattr(first_eeg, 'type'):
                                print(f"Event.eeg.type: {first_eeg.type}")
        
        # Check if trigger is in trial structure directly
        print("\n" + "="*60)
        print("Checking trial structure for trigger...")
        if hasattr(first_trial, 'trigger'):
            print(f"Trial has 'trigger' attribute: {first_trial.trigger}")
        elif hasattr(first_trial, 'dtype') and hasattr(first_trial.dtype, 'names'):
            print(f"Trial dtype names: {first_trial.dtype.names}")
            if 'trigger' in first_trial.dtype.names:
                print(f"Trial has 'trigger' field: {first_trial['trigger']}")
        
        # Check multiple trials for trigger patterns
        print("\n" + "="*60)
        print("Checking triggers across multiple trials...")
        n_trials_to_check = min(5, data_struct.size)
        for i in range(n_trials_to_check):
            trial = data_struct.flat[i]
            trigger_found = None
            
            # Try event structure
            if hasattr(trial, 'event'):
                event = trial.event
                if isinstance(event, np.ndarray) and event.size > 0:
                    first_event = event.flat[0]
                    if hasattr(first_event, 'eeg'):
                        eeg_event = first_event.eeg
                        if isinstance(eeg_event, np.ndarray) and eeg_event.size > 0:
                            first_eeg = eeg_event.flat[0]
                            # Check various possible trigger locations
                            for attr in ['trigger', 'type', 'value']:
                                if hasattr(first_eeg, attr):
                                    val = getattr(first_eeg, attr)
                                    if val is not None:
                                        trigger_found = (attr, val)
                                        break
            
            print(f"Trial {i}: trigger = {trigger_found}")
