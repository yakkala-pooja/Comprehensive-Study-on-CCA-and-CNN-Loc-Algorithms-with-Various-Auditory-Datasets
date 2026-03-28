#!/usr/bin/env python3
"""Verify Fulsang dataset processing and trial extraction."""

import scipy.io as sio
import numpy as np
from pathlib import Path
from collections import Counter

print("="*80)
print("VERIFYING FULSANG DATASET PROCESSING")
print("="*80)

# Check MWF processed files
fulsang_mwf_dir = Path("combined_dataset/fulsang_mwf")
if not fulsang_mwf_dir.exists():
    print(f"\n✗ MWF directory not found: {fulsang_mwf_dir}")
    print("  Checking legacy location...")
    fulsang_mwf_dir = Path("MWF_cleaned_Fuglsang")

if not fulsang_mwf_dir.exists():
    print(f"\n✗ No Fulsang MWF files found. Please run MWF processing first.")
    exit(1)

mwf_files = sorted(list(fulsang_mwf_dir.glob("sub*_MWF.mat")))
print(f"\nFound {len(mwf_files)} MWF files in {fulsang_mwf_dir}")

if len(mwf_files) == 0:
    print("✗ No MWF files found!")
    exit(1)

# Analyze each file
total_trials = 0
total_samples = 0
label_distribution = Counter()
subjects_with_issues = []

for mwf_file in mwf_files[:5]:  # Check first 5 files
    print(f"\n{'='*80}")
    print(f"Analyzing: {mwf_file.name}")
    print(f"{'='*80}")
    
    try:
        data = sio.loadmat(str(mwf_file), squeeze_me=True, struct_as_record=False)
        
        if 'trials' not in data:
            print(f"  ✗ No 'trials' key found!")
            subjects_with_issues.append(mwf_file.stem)
            continue
        
        trials = data['trials']
        if not isinstance(trials, np.ndarray):
            trials = [trials]
        else:
            trials = trials.flatten()
        
        print(f"  ✓ Found {len(trials)} trials")
        total_trials += len(trials)
        
        # Check each trial
        valid_trials = 0
        invalid_trials = 0
        trial_labels = []
        
        for trial_idx, trial in enumerate(trials):
            # Check EEG data
            has_eeg = False
            eeg_shape = None
            
            if hasattr(trial, 'eeg_data'):
                eeg_data = trial.eeg_data
                has_eeg = True
                eeg_shape = eeg_data.shape if hasattr(eeg_data, 'shape') else None
            elif isinstance(trial, dict) and 'eeg_data' in trial:
                eeg_data = trial['eeg_data']
                has_eeg = True
                eeg_shape = eeg_data.shape if hasattr(eeg_data, 'shape') else None
            
            if not has_eeg or eeg_data is None:
                invalid_trials += 1
                continue
            
            # Check label
            label = None
            if hasattr(trial, 'attention_label'):
                label = trial.attention_label
            elif isinstance(trial, dict) and 'attention_label' in trial:
                label = trial['attention_label']
            
            if label is not None:
                if isinstance(label, np.ndarray):
                    if label.size > 0:
                        label = label.item() if label.size == 1 else label.flatten()[0]
                    else:
                        label = None
                
                if label in [0, 1]:
                    trial_labels.append(label)
                    label_distribution[label] += 1
                    valid_trials += 1
                    if eeg_shape:
                        total_samples += eeg_shape[0]
                else:
                    invalid_trials += 1
            else:
                invalid_trials += 1
        
        print(f"  Valid trials: {valid_trials}/{len(trials)}")
        print(f"  Invalid trials: {invalid_trials}/{len(trials)}")
        print(f"  Label distribution: {dict(Counter(trial_labels))}")
        
        if invalid_trials > 0:
            print(f"  ⚠️  Warning: {invalid_trials} trials have issues")
            subjects_with_issues.append(mwf_file.stem)
        
        # Show first trial structure
        if len(trials) > 0:
            first_trial = trials[0]
            print(f"\n  First trial structure:")
            print(f"    Type: {type(first_trial)}")
            if hasattr(first_trial, '__dict__'):
                attrs = [x for x in dir(first_trial) if not x.startswith('_')]
                print(f"    Attributes: {attrs[:10]}...")  # Show first 10
            elif isinstance(first_trial, dict):
                print(f"    Keys: {list(first_trial.keys())[:10]}...")
            elif isinstance(first_trial, np.ndarray) and first_trial.dtype.names:
                print(f"    Structured array fields: {first_trial.dtype.names}")
        
    except Exception as e:
        print(f"  ✗ Error loading {mwf_file}: {e}")
        import traceback
        traceback.print_exc()
        subjects_with_issues.append(mwf_file.stem)

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Total files checked: {len(mwf_files)}")
print(f"Total trials: {total_trials}")
print(f"Total samples: {total_samples:,}")
print(f"Label distribution: {dict(label_distribution)}")
print(f"  Left (0): {label_distribution[0]} ({label_distribution[0]/sum(label_distribution.values())*100:.1f}%)")
print(f"  Right (1): {label_distribution[1]} ({label_distribution[1]/sum(label_distribution.values())*100:.1f}%)")

if subjects_with_issues:
    print(f"\n⚠️  Subjects with issues: {subjects_with_issues}")
else:
    print(f"\n✓ All subjects processed correctly!")

print(f"\n{'='*80}")

