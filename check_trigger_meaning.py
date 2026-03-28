#!/usr/bin/env python3
"""Check what trigger value 2 means by comparing with attend_mf and other experimental conditions."""

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
        
        # Extract expinfo
        expinfo = {}
        if 'expinfo' in mat_data:
            expinfo_raw = mat_data['expinfo']
            if hasattr(expinfo_raw, 'dtype') and expinfo_raw.dtype.names:
                for field in expinfo_raw.dtype.names:
                    value = expinfo_raw[field]
                    if isinstance(value, np.ndarray):
                        if value.size == 1:
                            expinfo[field] = value.item()
                        else:
                            expinfo[field] = value.flatten()
                    else:
                        expinfo[field] = value
        
        print("\nExperimental Information (expinfo):")
        print("-" * 60)
        for key, value in expinfo.items():
            if isinstance(value, np.ndarray) and value.size <= 10:
                print(f"  {key}: {value}")
            elif isinstance(value, np.ndarray):
                print(f"  {key}: array of {value.size} values (first 5: {value[:5]})")
            else:
                print(f"  {key}: {value}")
        
        # Check trigger values and compare with attend_mf
        print("\n" + "=" * 60)
        print("Comparing trigger values with attend_mf for first 10 trials:")
        print("-" * 60)
        
        triggers = []
        attend_mf_values = []
        
        # Get attend_mf from expinfo
        attend_mf = expinfo.get('attend_mf', None)
        if attend_mf is not None:
            if isinstance(attend_mf, np.ndarray):
                attend_mf_values = attend_mf.flatten().tolist()
            else:
                attend_mf_values = [attend_mf]
        
        # Extract triggers from event structure
        if hasattr(first_elem, 'event'):
            event = first_elem.event
            if isinstance(event, np.ndarray) and event.size > 0:
                first_event = event.flat[0]
                if hasattr(first_event, 'eeg'):
                    eeg_events = first_event.eeg
                    if isinstance(eeg_events, np.ndarray) and eeg_events.size > 0:
                        n_trials = min(10, eeg_events.size)
                        for i in range(n_trials):
                            eeg_event = eeg_events.flat[i]
                            if hasattr(eeg_event, 'value'):
                                value = eeg_event.value
                                if isinstance(value, np.ndarray):
                                    # Unwrap nested arrays
                                    while isinstance(value, np.ndarray) and value.dtype == object and value.size > 0:
                                        value = value.flat[0]
                                    if isinstance(value, np.ndarray):
                                        value_flat = value.flatten()
                                        if value_flat.size > 0:
                                            trigger = int(value_flat[0])
                                            triggers.append(trigger)
                                            
                                            attend_mf_val = attend_mf_values[i] if i < len(attend_mf_values) else None
                                            match = "YES" if trigger == attend_mf_val else "NO"
                                            
                                            print(f"Trial {i:2d}: Trigger={trigger:2d}, attend_mf={attend_mf_val}, Match: {match}")
        
        print("\n" + "=" * 60)
        print("CONCLUSION:")
        print("-" * 60)
        if triggers and attend_mf_values:
            matches = sum(1 for i, t in enumerate(triggers) if i < len(attend_mf_values) and t == attend_mf_values[i])
            total = min(len(triggers), len(attend_mf_values))
            print(f"Trigger values match attend_mf: {matches}/{total} trials")
            
            if matches == total:
                print("\n✓ Trigger value = attend_mf (attended speaker gender)")
                print("  - Trigger 1 = Male speaker attended (attend_mf=1)")
                print("  - Trigger 2 = Female speaker attended (attend_mf=2)")
                print(f"\n  So trigger value 2 means: FEMALE SPEAKER IS ATTENDED")
            else:
                print("\n⚠ Trigger values do NOT directly match attend_mf")
                print("  Triggers may encode other experimental conditions")
        
        # Check other conditions
        print("\n" + "=" * 60)
        print("Other experimental conditions for context:")
        print("-" * 60)
        if 'attend_lr' in expinfo:
            attend_lr = expinfo['attend_lr']
            if isinstance(attend_lr, np.ndarray) and attend_lr.size <= 10:
                print(f"  attend_lr (spatial): {attend_lr}")
        if 'acoustic_condition' in expinfo:
            acoustic = expinfo['acoustic_condition']
            if isinstance(acoustic, np.ndarray) and acoustic.size <= 10:
                print(f"  acoustic_condition: {acoustic}")
        if 'n_speakers' in expinfo:
            n_speakers = expinfo['n_speakers']
            print(f"  n_speakers: {n_speakers}")
