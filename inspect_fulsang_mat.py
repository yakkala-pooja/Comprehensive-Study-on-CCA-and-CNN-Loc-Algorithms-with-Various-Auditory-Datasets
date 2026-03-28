#!/usr/bin/env python3
"""
Inspect Fulsang .mat file contents to verify expinfo (attend_mf, attend_lr) and data structure.
Run from repo root, e.g.:
  python inspect_fulsang_mat.py
  python inspect_fulsang_mat.py Data/Fulsang/EEG
  python inspect_fulsang_mat.py path/to/S1.mat
"""
import sys
from pathlib import Path
import numpy as np
import scipy.io as sio

def _describe(obj, name="", max_elems=5):
    """Return a short description of an object for printing."""
    if obj is None:
        return "None"
    if hasattr(obj, '_fieldnames'):
        return f"mat_struct with _fieldnames: {obj._fieldnames}"
    if hasattr(obj, 'dtype') and hasattr(obj.dtype, 'names') and obj.dtype.names:
        return f"structured array fields: {obj.dtype.names}"
    if isinstance(obj, np.ndarray):
        shp = obj.shape
        return f"ndarray shape {shp}"
    if isinstance(obj, (list, tuple)):
        return f"list/tuple len {len(obj)}"
    if hasattr(obj, '__dict__'):
        keys = [k for k in obj.__dict__.keys() if not k.startswith('_')]
        return f"object with keys: {keys}"
    return type(obj).__name__

def get_struct_fields(obj):
    """Get field names from a MATLAB struct (mat_struct or structured array)."""
    if obj is None:
        return []
    if hasattr(obj, '_fieldnames'):
        return list(obj._fieldnames)
    if hasattr(obj, 'dtype') and hasattr(obj.dtype, 'names') and obj.dtype.names:
        return list(obj.dtype.names)
    if hasattr(obj, '__dict__'):
        return [k for k in obj.__dict__.keys() if not k.startswith('_')]
    return []

def get_field_value(obj, field, default=None):
    """Safely get a field from struct (attribute or dict)."""
    if obj is None:
        return default
    try:
        if hasattr(obj, field):
            return getattr(obj, field)
        if isinstance(obj, dict) and field in obj:
            return obj[field]
        if hasattr(obj, 'dtype') and obj.dtype.names and field in obj.dtype.names:
            return obj[field]
    except Exception:
        pass
    return default


def _print_expinfo_content(exp):
    """Print attend_mf, attend_lr, etc. from an expinfo-like struct."""
    fields = get_struct_fields(exp)
    if fields:
        print(f"  Fields: {fields}")
    for f in ['attend_mf', 'attend_lr', 'attendMf', 'attendLr', 'wavfile_male', 'wavfile_female', 'trigger']:
        v = get_field_value(exp, f)
        if v is not None:
            arr = np.asarray(v).flatten()
            if arr.size <= 10:
                print(f"    {f}: {arr}")
            else:
                print(f"    {f}: array size {arr.size}, first 5: {arr[:5]}")
            # Raw counts for attend_lr / attend_mf (1 vs 2) so user can verify uneven vs even
            if f in ('attend_mf', 'attend_lr', 'attendMf', 'attendLr') and arr.size > 0:
                uniq, counts = np.unique(arr, return_counts=True)
                cnt_str = ", ".join(f"value {u}: {c}" for u, c in zip(uniq, counts))
                print(f"      -> counts: {cnt_str}")


def inspect_mat(filepath: Path):
    print(f"\n{'='*60}")
    print(f"File: {filepath}")
    print(f"Exists: {filepath.exists()}")
    if not filepath.exists():
        return
    try:
        mat = sio.loadmat(str(filepath), squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"Load error: {e}")
        return
    # All keys (include __globals__ which lists variable names saved in the .mat)
    all_keys = list(mat.keys())
    print(f"All keys: {all_keys}")
    if "__globals__" in mat:
        print(f"  __globals__ (variable names in file): {mat['__globals__']}")
    keys = [k for k in all_keys if not k.startswith("__")]
    print(f"Top-level variables: {keys}")
    # Raw counts for sidecar *_expinfo.mat (attend_mf, attend_lr at top level)
    for var in ['attend_lr', 'attend_mf']:
        if var not in mat:
            continue
        arr = np.asarray(mat[var]).flatten()
        uniq, counts = np.unique(arr, return_counts=True)
        cnt_str = ", ".join(f"{int(u)}: {c}" for u, c in zip(uniq, counts))
        print(f"  {var} (top-level) counts: {cnt_str}  (attend_lr: 1=Left, 2=Right; attend_mf: 1=Male, 2=Female)")
    # expinfo by known name
    for key in ['expinfo', 'exp_info', 'experiment_info', 'info']:
        if key not in mat:
            continue
        exp = mat[key]
        print(f"\n  '{key}' present: {_describe(exp)}")
        _print_expinfo_content(exp)
    # scipy loadmat sometimes stores the second variable under key 'None' (e.g. expinfo)
    if "None" in mat:
        val = mat["None"]
        fields = get_struct_fields(val)
        print(f"\n  Variable under key 'None': {_describe(val)}")
        if fields:
            print(f"  Fields: {fields}")
            if any(f in fields for f in ["attend_mf", "attend_lr", "attendMf", "attendLr", "wavfile_male"]):
                print("  >>> This is expinfo (use key 'None' in Python when loading)")
                _print_expinfo_content(val)
    # Scan every top-level variable for expinfo-like struct
    for key in keys:
        if key in ["expinfo", "exp_info", "experiment_info", "info", "None"]:
            continue
        val = mat[key]
        fields = get_struct_fields(val)
        if not fields:
            continue
        if any(f in fields for f in ["attend_mf", "attend_lr", "attendMf", "attendLr", "wavfile_male"]):
            print(f"\n  >>> '{key}' looks like expinfo: {fields}")
            _print_expinfo_content(val)
    # data (brief)
    if 'data' in mat:
        d = mat['data']
        print(f"\n  'data' present: {_describe(d)}")
        if hasattr(d, 'eeg'):
            print(f"    data.eeg: {_describe(d.eeg)}")
        if hasattr(d, 'fsample'):
            print(f"    data.fsample: {_describe(d.fsample)}")
    print()

def main():
    repo = Path(__file__).resolve().parent
    if len(sys.argv) >= 2:
        arg = Path(sys.argv[1])
        if arg.suffix == '.mat':
            inspect_mat(arg)
            return
        base = arg
    else:
        base = repo / "Data" / "Fulsang" / "EEG"
    base = Path(base)
    print(f"Base path: {base}")
    # Try raw S1.mat
    for subj in [1, 2]:
        for name in [f"S{subj}.mat", f"S{subj}_data_preproc.mat"]:
            f = base / name
            if f.exists():
                inspect_mat(f)
                return
        # Also try eeg/ subfolder
        for name in [f"S{subj}.mat", f"S{subj}_data_preproc.mat"]:
            f = base / "eeg" / name
            if f.exists():
                inspect_mat(f)
                return
    print("No S1.mat or S2.mat or S1_data_preproc.mat found under base path.")
    print("Usage: python inspect_fulsang_mat.py [path_to_dir_or_S1.mat]")

if __name__ == "__main__":
    main()
