#!/usr/bin/env python3
"""
Verify if FULPRE.py actually writes triggers to TFRecord files.
Checks the code logic and tests actual files.
"""

import tensorflow as tf
from pathlib import Path
import scipy.io as sio
import numpy as np

def check_fulpre_logic():
    """Check the logic in FULPRE.py for trigger extraction."""
    print("="*80)
    print("ANALYZING FULPRE.PY TRIGGER LOGIC")
    print("="*80)
    
    print("\n1. Trigger Extraction from Event Structure:")
    print("   - FULPRE.py extracts from event.eeg.value{1} (lines 1331-1375)")
    print("   - Stores in expinfo['label_original_values'] (line 1397)")
    print("   - These are the original 1/2 values (1=male, 2=female)")
    
    print("\n2. Trigger Writing to TFRecord:")
    print("   - Code tries: trigger_val = get_expinfo_scalar('trigger') (line 1670)")
    print("   - This looks for expinfo['trigger'] or expinfo.trigger")
    print("   - BUT: expinfo['trigger'] is extracted from MATLAB expinfo structure")
    print("   - NOT from event.eeg.value{1}")
    
    print("\n3. The Issue:")
    print("   - Trigger values ARE extracted (as label_original_values)")
    print("   - But they're NOT written as 'trigger' feature")
    print("   - Instead, they're used for 'attend_mf_raw' feature")
    
    print("\n4. What Should Happen:")
    print("   - Use label_original_values[trial_idx] to write trigger")
    print("   - This would preserve the original trigger values from event.eeg.value{1}")

def check_actual_file():
    """Check if any FULPRE-created files have triggers."""
    print("\n" + "="*80)
    print("CHECKING ACTUAL FULPRE FILES")
    print("="*80)
    
    # Check default output directory
    fulpre_dir = Path("fulsang_preprocessed/tfrecords")
    if fulpre_dir.exists():
        files = list(fulpre_dir.glob("fulsang_S*.tfrecords"))
        if files:
            print(f"\nFound {len(files)} files in {fulpre_dir}")
            test_file = files[0]
            print(f"\nChecking: {test_file.name}")
            
            try:
                dataset = tf.data.TFRecordDataset(str(test_file))
                example = next(iter(dataset))
                features = tf.train.Example.FromString(example.numpy()).features.feature
                
                has_trigger = 'trigger' in features
                has_attend_mf_raw = 'attend_mf_raw' in features
                
                print(f"  Has 'trigger' feature: {has_trigger}")
                print(f"  Has 'attend_mf_raw' feature: {has_attend_mf_raw}")
                
                if has_trigger:
                    trigger_val = features['trigger'].int64_list.value[0]
                    print(f"  Trigger value: {trigger_val}")
                
                if has_attend_mf_raw:
                    attend_mf_val = features['attend_mf_raw'].int64_list.value[0]
                    print(f"  attend_mf_raw value: {attend_mf_val}")
                    print(f"  Note: This contains the trigger value (1 or 2) from event.eeg.value{{1}}")
                
                print(f"\n  All features: {sorted(features.keys())}")
                
            except Exception as e:
                print(f"  Error: {e}")
    else:
        print(f"\nDirectory not found: {fulpre_dir}")

def check_raw_matlab_trigger():
    """Check if triggers exist in raw MATLAB file expinfo."""
    print("\n" + "="*80)
    print("CHECKING RAW MATLAB FILE FOR TRIGGERS IN EXPINFO")
    print("="*80)
    
    mat_file = Path("Data/Fulsang/DATA_preproc/S1_data_preproc.mat")
    if not mat_file.exists():
        print(f"File not found: {mat_file}")
        return
    
    try:
        mat_data = sio.loadmat(str(mat_file), squeeze_me=False, struct_as_record=False)
        
        # Check expinfo
        if 'expinfo' in mat_data:
            expinfo = mat_data['expinfo']
            print(f"\nexpinfo type: {type(expinfo)}")
            
            if hasattr(expinfo, 'dtype') and expinfo.dtype.names:
                print(f"expinfo fields: {expinfo.dtype.names}")
                
                if 'trigger' in expinfo.dtype.names:
                    trigger_val = expinfo['trigger']
                    print(f"\n✓ expinfo.trigger EXISTS")
                    print(f"  Type: {type(trigger_val)}")
                    if isinstance(trigger_val, np.ndarray):
                        print(f"  Shape: {trigger_val.shape}")
                        print(f"  First 5 values: {trigger_val.flatten()[:5]}")
                else:
                    print(f"\n✗ expinfo.trigger NOT FOUND")
                    print(f"  Available fields: {expinfo.dtype.names}")
        else:
            print("\n✗ 'expinfo' not found in MATLAB file")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    check_fulpre_logic()
    check_actual_file()
    check_raw_matlab_trigger()
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("\nFULPRE.py DOES extract trigger values from event.eeg.value{1}")
    print("BUT they are stored as 'label_original_values' and used for 'attend_mf_raw'")
    print("They are NOT written as a 'trigger' feature in TFRecord files.")
    print("\nTo preserve triggers, the code should use label_original_values[trial_idx]")
    print("instead of expinfo.get('trigger') when writing the trigger feature.")
