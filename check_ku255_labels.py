#!/usr/bin/env python3
"""
Verify KU255 dataset labels: ensure Left=0, Right=1 is consistent with
preprocessed .mat files and with PREPROCESS255 / Curry filename convention.
Run from project root:
  python check_ku255_labels.py
  python check_ku255_labels.py --preprocessed_dir kuleuven_255_preprocessed
"""

import argparse
import numpy as np
import scipy.io as sio
from pathlib import Path


def parse_kuleuven_filename(filename: str) -> dict:
    """Parse KU Leuven 255 filename for subject (and optionally trial/ear)."""
    base = Path(filename).stem.replace('_preprocessed', '').strip()
    import re
    # S<number>_AAD_<trial><L|R>
    m = re.match(r'^(S\d+)_AAD_(\d+)([LR])$', base)
    if m:
        return {'subject_id': m.group(1), 'trial_id': m.group(2), 'attention_ear': m.group(3)}
    # S<number> only
    m = re.match(r'^\s*(S\d+)\s*$', base)
    if m:
        return {'subject_id': m.group(1), 'trial_id': None, 'attention_ear': None}
    return None


def run_check(preprocessed_dir: str, max_files: int = 5) -> int:
    """Verify labels in preprocessed_dir. Returns 0 if OK, 1 if mismatches or error."""
    preprocessed_dir = Path(preprocessed_dir)
    if not preprocessed_dir.exists():
        print(f"ERROR: Directory not found: {preprocessed_dir}")
        return 1

    mat_files = sorted(preprocessed_dir.glob("*_preprocessed.mat"))
    if not mat_files:
        print(f"ERROR: No *_preprocessed.mat files in {preprocessed_dir}")
        return 1

    print(f"KU255 label verification (Left=0, Right=1)")
    print(f"Preprocessed dir: {preprocessed_dir}")
    print(f"Found {len(mat_files)} .mat files\n")
    print("Convention: L / Left -> label 0; R / Right -> label 1")
    print("-" * 60)

    total_trials = 0
    mismatches = []
    label_counts = {0: 0, 1: 0}
    n_inspect = max_files if max_files > 0 else len(mat_files)

    for mi, mat_file in enumerate(mat_files):
        if mi >= n_inspect:
            break
        try:
            data = sio.loadmat(str(mat_file), squeeze_me=True, struct_as_record=False)
        except Exception as e:
            print(f"  Skip {mat_file.name}: load failed: {e}")
            continue

        if 'trials' not in data:
            print(f"  {mat_file.name}: no 'trials' key")
            continue

        trials = data['trials']
        if not isinstance(trials, np.ndarray):
            trials = [trials]
        else:
            trials = trials.flatten()

        filename_info = parse_kuleuven_filename(mat_file.name)
        subject_id = filename_info['subject_id'] if filename_info else '?'

        print(f"\n  File: {mat_file.name} (subject {subject_id}, {len(trials)} trials)")

        for trial_idx, trial in enumerate(trials):
            total_trials += 1
            # Raw from .mat (PREPROCESS255 saves attended_ear, attention_label)
            attended_ear_raw = None
            attention_label_mat = None
            trial_code = None
            if hasattr(trial, 'attended_ear'):
                attended_ear_raw = trial.attended_ear
            elif isinstance(trial, dict):
                attended_ear_raw = trial.get('attended_ear')
            if hasattr(trial, 'attention_label'):
                attention_label_mat = trial.attention_label
            elif isinstance(trial, dict):
                attention_label_mat = trial.get('attention_label')
            if hasattr(trial, 'trial_code'):
                trial_code = trial.trial_code
            elif isinstance(trial, dict):
                trial_code = trial.get('trial_code')

            # Normalize for comparison
            if isinstance(attended_ear_raw, np.ndarray) and attended_ear_raw.size > 0:
                attended_ear_raw = str(attended_ear_raw.item() if attended_ear_raw.size == 1 else attended_ear_raw.flatten()[0])
            elif attended_ear_raw is not None:
                attended_ear_raw = str(attended_ear_raw).strip()
            if attention_label_mat is not None and isinstance(attention_label_mat, np.ndarray):
                attention_label_mat = int(attention_label_mat.item() if attention_label_mat.size == 1 else attention_label_mat.flatten()[0])
            elif attention_label_mat is not None:
                attention_label_mat = int(attention_label_mat)

            # Our convention: L -> 0, R -> 1
            if attended_ear_raw:
                derived_label = 0 if attended_ear_raw.upper() in ('L', 'LEFT') else 1
            else:
                derived_label = None

            # Check consistency
            if attention_label_mat is not None and derived_label is not None and attention_label_mat != derived_label:
                mismatches.append({
                    'file': mat_file.name,
                    'trial_idx': trial_idx,
                    'trial_code': trial_code,
                    'attended_ear_raw': attended_ear_raw,
                    'attention_label_mat': attention_label_mat,
                    'derived_label': derived_label,
                })
            if derived_label is not None:
                label_counts[derived_label] = label_counts.get(derived_label, 0) + 1

            # Print first 2 trials per file in detail
            if trial_idx < 2:
                ear_str = repr(attended_ear_raw) if attended_ear_raw is not None else "N/A"
                mat_str = str(attention_label_mat) if attention_label_mat is not None else "N/A"
                der_str = str(derived_label) if derived_label is not None else "N/A"
                code_str = str(trial_code) if trial_code is not None else "N/A"
                ok = "OK" if (attention_label_mat == derived_label or attention_label_mat is None) else "MISMATCH"
                print(f"    trial {trial_idx}: trial_code={code_str}, attended_ear={ear_str}, "
                      f"attention_label(.mat)={mat_str}, derived(0=L,1=R)={der_str}  [{ok}]")
        if len(trials) > 2:
            print(f"    ... ({len(trials)} trials total)")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total trials checked: {total_trials}")
    print(f"  Label distribution: 0 (Left) = {label_counts.get(0, 0)}, 1 (Right) = {label_counts.get(1, 0)}")
    if mismatches:
        print(f"  MISMATCHES: {len(mismatches)}")
        for m in mismatches[:10]:
            print(f"    {m['file']} trial {m['trial_idx']}: attended_ear={m['attended_ear_raw']}, "
                  f".mat label={m['attention_label_mat']}, derived={m['derived_label']}")
        if len(mismatches) > 10:
            print(f"    ... and {len(mismatches) - 10} more")
        print("\n  If .mat attention_label disagrees with attended_ear, fix PREPROCESS255 or loading.")
    else:
        print("  No mismatches: .mat attention_label and L->0/R->1 derivation agree.")
    print("\n  Filename convention (Curry): Sx_AAD_1L = attend Left (0), Sx_AAD_1R = attend Right (1).")
    print("  If your model's inverted accuracy > main accuracy, try --flip_labels at test or verify")
    print("  that the experiment design matches this convention (e.g. some datasets swap L/R).")
    return 0 if not mismatches else 1


def main():
    parser = argparse.ArgumentParser(description='Verify KU255 preprocessed labels (Left=0, Right=1).')
    parser.add_argument('--preprocessed_dir', type=str, default='kuleuven_255_preprocessed',
                        help='Directory containing *_preprocessed.mat files')
    parser.add_argument('--max_files', type=int, default=5,
                        help='Max .mat files to inspect in detail (0 = all)')
    args = parser.parse_args()
    return run_check(args.preprocessed_dir, args.max_files)


if __name__ == '__main__':
    exit(main())
