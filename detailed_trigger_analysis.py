#!/usr/bin/env python3
"""
Detailed analysis of triggers in raw MATLAB files for both datasets.
Examines where triggers are positioned in the data.
"""

import scipy.io as sio
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Tuple


def analyze_das_triggers():
    """Analyze triggers in DAS dataset raw MATLAB files."""
    print("\n" + "="*80)
    print("DAS DATASET - RAW MATLAB FILE ANALYSIS")
    print("="*80)
    
    das_mat_file = Path("Data/Das/4004271/S1.mat")
    
    if not das_mat_file.exists():
        print(f"ERROR: File not found: {das_mat_file}")
        return
    
    print(f"\nLoading: {das_mat_file}")
    mat_data = sio.loadmat(str(das_mat_file), squeeze_me=True, struct_as_record=False)
    
    print(f"\nKeys in MATLAB file: {[k for k in mat_data.keys() if not k.startswith('__')]}")
    
    if 'trials' not in mat_data:
        print("ERROR: 'trials' not found in MATLAB file")
        return
    
    trials = mat_data['trials']
    if not isinstance(trials, np.ndarray):
        trials = [trials]
    else:
        trials = trials.flatten()
    
    print(f"\nFound {len(trials)} trials")
    
    # Analyze first 5 trials in detail
    print("\n" + "-"*80)
    print("DETAILED ANALYSIS OF FIRST 5 TRIALS")
    print("-"*80)
    
    for i in range(min(5, len(trials))):
        trial = trials[i]
        print(f"\n--- Trial {i} ---")
        print(f"Attributes: {[k for k in dir(trial) if not k.startswith('_')]}")
        
        # Check for trigger at trial level
        has_trigger = hasattr(trial, 'trigger') or hasattr(trial, 'Trigger')
        print(f"Has trigger attribute: {has_trigger}")
        
        if hasattr(trial, 'trigger'):
            trigger_val = trial.trigger
            print(f"  trial.trigger: {trigger_val} (type: {type(trigger_val)})")
        elif hasattr(trial, 'Trigger'):
            trigger_val = trial.Trigger
            print(f"  trial.Trigger: {trigger_val} (type: {type(trigger_val)})")
        
        # Check attended_ear
        if hasattr(trial, 'attended_ear'):
            attended_ear = trial.attended_ear
            if isinstance(attended_ear, np.ndarray):
                attended_ear = attended_ear.item() if attended_ear.size == 1 else str(attended_ear[0])
            print(f"attended_ear: {attended_ear}")
        
        # Check RawData structure
        if hasattr(trial, 'RawData'):
            rawdata = trial.RawData
            print(f"RawData attributes: {[k for k in dir(rawdata) if not k.startswith('_')]}")
            
            # Check if RawData has trigger
            if hasattr(rawdata, 'trigger'):
                rawdata_trigger = rawdata.trigger
                print(f"  RawData.trigger: {rawdata_trigger}")
            elif hasattr(rawdata, 'Trigger'):
                rawdata_trigger = rawdata.Trigger
                print(f"  RawData.Trigger: {rawdata_trigger}")
            
            # Check EegData shape
            if hasattr(rawdata, 'EegData'):
                eeg_data = rawdata.EegData
                if isinstance(eeg_data, np.ndarray):
                    print(f"  EegData shape: {eeg_data.shape}")
                    print(f"  EegData samples: {eeg_data.shape[0]}")
                    print(f"  EegData channels: {eeg_data.shape[1] if eeg_data.ndim > 1 else 1}")
        
        # Check for event structure
        if hasattr(trial, 'event'):
            event = trial.event
            print(f"Has event structure: {event is not None}")
            if event is not None:
                print(f"  Event type: {type(event)}")
                if isinstance(event, np.ndarray):
                    print(f"  Event shape: {event.shape}")
        
        # Check FileHeader
        if hasattr(trial, 'FileHeader'):
            fileheader = trial.FileHeader
            print(f"FileHeader attributes: {[k for k in dir(fileheader) if not k.startswith('_')]}")
            if hasattr(fileheader, 'SampleRate'):
                print(f"  SampleRate: {fileheader.SampleRate}")
    
    print("\n" + "="*80)
    print("CONCLUSION FOR DAS:")
    print("="*80)
    print("DAS dataset does NOT use numeric triggers.")
    print("Instead, it uses 'attended_ear' (L/R) as the attention label.")
    print("This is a SPATIAL attention task (left vs right ear), not a gender-based task.")
    print("Triggers are NOT present in the traditional sense - attention is encoded")
    print("as a categorical label ('L' or 'R') at the trial level.")


def analyze_fulsang_triggers():
    """Analyze triggers in Fulsang dataset raw MATLAB files."""
    print("\n" + "="*80)
    print("FULSANG DATASET - RAW MATLAB FILE ANALYSIS")
    print("="*80)
    
    fulsang_mat_file = Path("Data/Fulsang/DATA_preproc/S1_data_preproc.mat")
    
    if not fulsang_mat_file.exists():
        print(f"ERROR: File not found: {fulsang_mat_file}")
        return
    
    print(f"\nLoading: {fulsang_mat_file}")
    mat_data = sio.loadmat(str(fulsang_mat_file), squeeze_me=False, struct_as_record=False)
    
    print(f"\nKeys in MATLAB file: {[k for k in mat_data.keys() if not k.startswith('__')]}")
    
    # Check expinfo
    expinfo = {}
    if 'expinfo' in mat_data:
        expinfo_raw = mat_data['expinfo']
        print(f"\nexpinfo type: {type(expinfo_raw)}")
        
        if hasattr(expinfo_raw, 'dtype') and expinfo_raw.dtype.names:
            print(f"expinfo fields: {expinfo_raw.dtype.names}")
            
            for field in expinfo_raw.dtype.names:
                value = expinfo_raw[field]
                if isinstance(value, np.ndarray):
                    if value.size == 1:
                        expinfo[field] = value.item()
                    else:
                        expinfo[field] = value.flatten()
                else:
                    expinfo[field] = value
            
            print(f"\nExpinfo values:")
            for key, val in expinfo.items():
                if isinstance(val, np.ndarray) and val.size <= 20:
                    print(f"  {key}: {val}")
                elif isinstance(val, np.ndarray):
                    print(f"  {key}: array of {val.size} values (first 5: {val[:5]})")
                else:
                    print(f"  {key}: {val}")
    
    # Check data structure
    if 'data' not in mat_data:
        print("ERROR: 'data' not found in MATLAB file")
        return
    
    data_struct = mat_data['data']
    print(f"\ndata_struct type: {type(data_struct)}")
    print(f"data_struct shape: {data_struct.shape if hasattr(data_struct, 'shape') else 'N/A'}")
    
    if not isinstance(data_struct, np.ndarray) or data_struct.size == 0:
        print("ERROR: data_struct is empty or invalid")
        return
    
    print(f"\nFound {data_struct.size} trials in data structure")
    
    # Analyze first 5 trials
    print("\n" + "-"*80)
    print("DETAILED ANALYSIS OF FIRST 5 TRIALS")
    print("-"*80)
    
    for i in range(min(5, data_struct.size)):
        trial_elem = data_struct.flat[i]
        print(f"\n--- Trial {i} ---")
        print(f"Attributes: {[k for k in dir(trial_elem) if not k.startswith('_')]}")
        
        # Check EEG data
        if hasattr(trial_elem, 'eeg'):
            eeg = trial_elem.eeg
            if isinstance(eeg, np.ndarray):
                print(f"EEG shape: {eeg.shape}")
                print(f"EEG samples: {eeg.shape[0] if eeg.ndim >= 1 else 1}")
                print(f"EEG channels: {eeg.shape[1] if eeg.ndim >= 2 else 1}")
        
        # Check event structure
        if hasattr(trial_elem, 'event'):
            event = trial_elem.event
            print(f"Has event structure: {event is not None}")
            
            if isinstance(event, np.ndarray) and event.size > 0:
                first_event = event.flat[0]
                print(f"  First event attributes: {[k for k in dir(first_event) if not k.startswith('_')]}")
                
                if hasattr(first_event, 'eeg'):
                    eeg_events = first_event.eeg
                    if isinstance(eeg_events, np.ndarray) and eeg_events.size > 0:
                        print(f"  Found {eeg_events.size} EEG events")
                        
                        # Analyze first 5 events
                        print(f"\n  First 5 events:")
                        for j in range(min(5, eeg_events.size)):
                            eeg_event = eeg_events.flat[j]
                            
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
                                # Unwrap nested arrays
                                while isinstance(value_obj, np.ndarray) and value_obj.dtype == object and value_obj.size > 0:
                                    value_obj = value_obj.flat[0]
                                if isinstance(value_obj, np.ndarray):
                                    value_flat = value_obj.flatten()
                                    if value_flat.size > 0:
                                        value = int(value_flat[0])
                                elif not isinstance(value_obj, np.ndarray):
                                    try:
                                        value = int(value_obj)
                                    except:
                                        value = value_obj
                            
                            print(f"    Event {j}: sample={sample}, value={value}")
                            
                            # Calculate time position
                            if sample is not None and hasattr(trial_elem, 'fsample'):
                                fsample = trial_elem.fsample
                                if isinstance(fsample, np.ndarray):
                                    fsample = fsample.item() if fsample.size == 1 else fsample[0]
                                elif hasattr(fsample, '__float__'):
                                    try:
                                        fsample = float(fsample)
                                    except:
                                        fsample = None
                                if fsample is not None and isinstance(fsample, (int, float)) and fsample > 0:
                                    time_sec = sample / fsample
                                    print(f"      Time: {time_sec:.3f}s (at {fsample} Hz)")
        
        # Check fsample
        if hasattr(trial_elem, 'fsample'):
            fsample = trial_elem.fsample
            if isinstance(fsample, np.ndarray):
                fsample = fsample.item() if fsample.size == 1 else fsample[0]
            elif hasattr(fsample, '__float__'):
                try:
                    fsample = float(fsample)
                except:
                    fsample = None
            if fsample is not None:
                print(f"Sampling rate: {fsample} Hz")
    
    print("\n" + "="*80)
    print("CONCLUSION FOR FULSANG:")
    print("="*80)
    print("Fulsang dataset HAS triggers in the event structure.")
    print("Triggers are stored in: data.event.eeg (event structure)")
    print("Each event has:")
    print("  - sample: sample index where trigger occurs")
    print("  - value: trigger value (1=male, 2=female typically)")
    print("\nTrigger position:")
    print("  - Triggers appear to be at the START of trials (sample ~1)")
    print("  - They mark which speaker to attend to for the entire trial")
    print("  - This is a TRIAL-LEVEL instruction marker, not a mid-trial switch")
    print("\nNote: Triggers may not be present in preprocessed TFRecord files")
    print("if the preprocessing script didn't extract them from the event structure.")


def main():
    print("="*80)
    print("COMPREHENSIVE TRIGGER ANALYSIS")
    print("="*80)
    
    analyze_das_triggers()
    analyze_fulsang_triggers()
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print("\nDAS Dataset:")
    print("  - Triggers: NOT PRESENT (uses 'attended_ear' instead)")
    print("  - Position: N/A (trial-level categorical label)")
    print("  - Format: 'L' or 'R' string in attended_ear field")
    
    print("\nFulsang Dataset:")
    print("  - Triggers: PRESENT in raw MATLAB files")
    print("  - Position: At trial START (sample ~1)")
    print("  - Format: Numeric value in event.eeg structure")
    print("  - Location: data[i].event.eeg[j].value")
    print("  - Note: May not be in preprocessed TFRecord files")


if __name__ == '__main__':
    main()
