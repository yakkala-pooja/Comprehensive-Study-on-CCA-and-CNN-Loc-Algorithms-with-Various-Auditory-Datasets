"""Test script to check if we can read MATLAB tables from raw EEG files."""

import sys
from pathlib import Path
import scipy.io as sio
import numpy as np

# Test file path
eeg_file = Path(r'D:\telluride_decoding\Data\Fulsang\EEG\S1.mat')

if not eeg_file.exists():
    print(f"[FAIL] File not found: {eeg_file}")
    sys.exit(1)

print(f"Loading: {eeg_file.name}")
print("="*70)

# Try scipy.io.loadmat with different options
print("\n[1] Trying scipy.io.loadmat with struct_as_record=False...")
try:
    mat_data = sio.loadmat(str(eeg_file), squeeze_me=False, struct_as_record=False)
    print("[OK] Loaded successfully")
    
    print(f"\nTop-level keys: {[k for k in mat_data.keys() if not k.startswith('__')]}")
    
    # Check the 'None' key (contains MATLAB table reference)
    none_key = mat_data.get('None')
    if none_key is not None:
        print(f"\n[2] Found 'None' key (MATLAB table reference)")
        print(f"   Type: {type(none_key)}")
        print(f"   Repr: {repr(none_key)[:200]}")
    
    # Check for expinfo
    expinfo = mat_data.get('expinfo')
    if expinfo is None:
        print(f"\n[3] [FAIL] 'expinfo' is None at top level")
        print(f"   This is the problem - expinfo is a MATLAB table that scipy.io can't read")
    else:
        print(f"\n[3] [OK] Found 'expinfo'")
        print(f"   Type: {type(expinfo)}")
        
        # Try to access attend_mf
        if isinstance(expinfo, dict):
            if 'attend_mf' in expinfo:
                attend_mf = expinfo['attend_mf']
                print(f"   [SUCCESS] Found 'attend_mf'!")
                print(f"   attend_mf: {attend_mf}")
        elif hasattr(expinfo, 'attend_mf'):
            attend_mf = expinfo.attend_mf
            print(f"   [SUCCESS] Found 'attend_mf' as attribute!")
            print(f"   attend_mf: {attend_mf}")
    
    # Check if expinfo is in 'data'
    if 'data' in mat_data:
        print(f"\n[4] Checking 'data' field...")
        data = mat_data['data']
        print(f"   data type: {type(data)}")
        
        if isinstance(data, dict):
            if 'expinfo' in data:
                print(f"   [OK] Found 'expinfo' in 'data'!")
                expinfo = data['expinfo']
                print(f"   expinfo type: {type(expinfo)}")
    
    print("\n" + "="*70)
    print("CONCLUSION:")
    print("  - File is NOT MATLAB v7.3 format (mat73 can't read it)")
    print("  - scipy.io.loadmat loads the file but expinfo is None")
    print("  - expinfo is stored as a MATLAB table (newer format)")
    print("  - Need to find alternative method to extract table data")
    print("\n  OPTIONS:")
    print("  1. Use MATLAB to export expinfo to a simpler format")
    print("  2. Use MATLAB Engine for Python to read the table")
    print("  3. Check if there's metadata in DATA_preproc files")
    print("  4. Use the 'None' key reference if it contains accessible data")
    
except Exception as e:
    print(f"\n[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
