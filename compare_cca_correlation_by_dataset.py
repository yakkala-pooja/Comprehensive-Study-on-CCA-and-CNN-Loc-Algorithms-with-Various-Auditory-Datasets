#!/usr/bin/env python3
"""
Compare first canonical correlation (ρ₁) when CCA is trained on:
  1. Das only
  2. Fulsang only
  3. Combined (Das + Fulsang)

Uses the same preprocessing and CCA setup as CombinedCCA. Run from repo root.
Bandpass 2-8 Hz (delta/theta) and --use_hilbert_envelope can improve ρ₁ for speech-brain.
Example: python compare_cca_correlation_by_dataset.py
         python compare_cca_correlation_by_dataset.py --bandpass_low_hz 2 --bandpass_high_hz 8 --use_hilbert_envelope
         python compare_cca_correlation_by_dataset.py --max_windows_per_group 2000
"""

import argparse
import numpy as np
from pathlib import Path

# Same lag/preprocess helpers as CombinedCCA (no TF dependency for data build)
def make_lagged_audio(audio: np.ndarray, lag_samples: np.ndarray, fs: float = 128.0) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    T, B = audio.shape
    num_lags = len(lag_samples)
    lagged_features = []
    for lag in lag_samples:
        shifted = np.roll(audio, int(lag), axis=0)
        if lag > 0:
            shifted[: int(lag), :] = 0
        lagged_features.append(shifted)
    return np.concatenate(lagged_features, axis=1).astype(np.float32)


def make_lagged_eeg(eeg: np.ndarray, L: int) -> np.ndarray:
    T, C = eeg.shape
    if L <= 1:
        return np.asarray(eeg, dtype=np.float32)
    out = np.zeros((T, C * L), dtype=np.float32)
    for t in range(T):
        segs = []
        for lag in range(L):
            idx = t - lag
            if idx >= 0:
                segs.append(eeg[idx, :])
            else:
                segs.append(np.zeros(C, dtype=eeg.dtype))
        out[t, :] = np.concatenate(segs, axis=0)
    return out


def preprocess_window(eeg_window: np.ndarray) -> np.ndarray:
    """Linear per-channel standardization only (no tanh) for CCA."""
    eeg_window = eeg_window - np.mean(eeg_window, axis=0, keepdims=True)
    std_vals = np.std(eeg_window, axis=0, keepdims=True)
    std_vals = np.where(std_vals == 0, 1.0, std_vals)
    eeg_window = eeg_window / std_vals
    eeg_window = np.nan_to_num(eeg_window, nan=0.0, posinf=0.0, neginf=0.0)
    return eeg_window.astype(np.float32)


def first_cca_correlation(X: np.ndarray, Y: np.ndarray, regularization: float = 0.01, eps_eig: float = 1e-12) -> float:
    """Compute first canonical correlation between X (n, dx) and Y (n, dy). Same math as CombinedCCA."""
    n = X.shape[0]
    mean_x = np.mean(X, axis=0, keepdims=True)
    mean_y = np.mean(Y, axis=0, keepdims=True)
    X_c = X - mean_x
    Y_c = Y - mean_y
    n1 = max(1, n - 1)
    cov_xx = (X_c.T @ X_c) / n1 + regularization * np.eye(X.shape[1])
    cov_yy = (Y_c.T @ Y_c) / n1 + regularization * np.eye(Y.shape[1])
    cov_xy = (X_c.T @ Y_c) / n1

    x_vals, x_vecs = np.linalg.eigh(cov_xx)
    y_vals, y_vecs = np.linalg.eigh(cov_yy)
    idx1 = np.where(x_vals > eps_eig)[0]
    idx2 = np.where(y_vals > eps_eig)[0]
    x_vals, x_vecs = x_vals[idx1], x_vecs[:, idx1]
    y_vals, y_vecs = y_vals[idx2], y_vecs[:, idx2]

    k11 = x_vecs @ np.diag(1.0 / np.sqrt(x_vals)) @ x_vecs.T
    k22 = y_vecs @ np.diag(1.0 / np.sqrt(y_vals)) @ y_vecs.T
    t = k11 @ cov_xy @ k22
    u, e, vh = np.linalg.svd(t, full_matrices=False)
    e1 = e[0] if len(e) > 0 else 0.0
    rho1 = np.sqrt(np.clip(e1, 0.0, 1.0))
    return float(rho1)


def _classify_dataset(dataset_tag: str, subject_id: str) -> str:
    tag = str(dataset_tag).strip().lower()
    if tag in ('das', 'das-mwf', 'combined_das', 'das-preprocessed'):
        return 'das'
    if tag in ('fulsang', 'fulsang-mwf', 'fuglsang'):
        return 'fulsang'
    sid = str(subject_id).strip()
    if not sid:
        return 'das'
    if __import__('re').match(r'^S?\d{1}$', sid) or (sid.isdigit() and 1 <= int(sid) <= 9):
        return 'das'
    return 'fulsang'


def _float_or_default(default: float):
    """Argparse type: use default when value is empty (avoids invalid float value: '')."""
    def _parse(s):
        if s is None or (isinstance(s, str) and s.strip() == ''):
            return default
        return float(s)
    return _parse


def main():
    parser = argparse.ArgumentParser(description='Compare CCA ρ₁: Das only vs Fulsang only vs Combined')
    parser.add_argument('--combined_dataset_dir', type=str, default='combined_dataset')
    parser.add_argument('--das_data_dir', type=str, default='das_16subjects_preprocessed')
    parser.add_argument('--das_preprocessing_type', type=str, default='COMBINED_DAS')
    parser.add_argument('--das_original_dir', type=str, default='Data/Das/4004271')
    parser.add_argument('--das_audio_dir', type=str, default='Data/Das/4004271/stimuli/stimuli')
    parser.add_argument('--fulsang_raw_dir', type=str, default='Data/Fulsang')
    parser.add_argument('--fulsang_audio_dir', type=str, default='Data/Fulsang/AUDIO')
    parser.add_argument('--fulsang_mwf_dir', type=str, default='MWF_cleaned_Fuglsang')
    parser.add_argument('--fulsang_expinfo_dir', type=str, default='Exp_Info')
    parser.add_argument('--bandpass_low_hz', type=_float_or_default(2.0), default=2.0,
                        help='Butterworth bandpass low cutoff (Hz). 2-8 Hz often better for speech-brain. Default 2.')
    parser.add_argument('--bandpass_high_hz', type=_float_or_default(8.0), default=8.0,
                        help='Butterworth bandpass high cutoff (Hz). Default 8 (delta/theta).')
    parser.add_argument('--bandpass_order', type=int, default=1)
    parser.add_argument('--use_hilbert_envelope', action='store_true',
                        help='Use Hilbert envelope for audio (better for speech-brain).')
    parser.add_argument('--window_size', type=int, default=512)
    parser.add_argument('--overlap', type=float, default=0.25)
    parser.add_argument('--target_sampling_rate', type=int, default=64)
    parser.add_argument('--eeg_lag_taps', type=int, default=12)
    parser.add_argument('--min_lag_ms', type=float, default=0.0)
    parser.add_argument('--max_lag_ms', type=float, default=250.0)
    parser.add_argument('--max_windows_per_group', type=int, default=1200,
                        help='Max windows per dataset group (limits memory and time)')
    parser.add_argument('--max_windows_total', type=int, default=5000,
                        help='Max windows to load from combined dataset')
    parser.add_argument('--regularization', type=float, default=0.01)
    parser.add_argument('--pca_components', type=int, default=128)
    args = parser.parse_args()

    # Import after args (CombinedDataset may log)
    from CombinedDataset import CombinedDataset

    fs = args.target_sampling_rate
    min_lag_s = int(args.min_lag_ms * fs / 1000.0)
    max_lag_s = int(args.max_lag_ms * fs / 1000.0)
    lag_samples = np.arange(min_lag_s, max_lag_s + 1)
    num_lags = len(lag_samples)

    print("Loading combined dataset (same as CombinedCCA)...")
    kwargs = {
        'das_data_dir': args.das_data_dir,
        'das_preprocessing_type': args.das_preprocessing_type,
        'das_original_dir': args.das_original_dir,
        'das_audio_dir': args.das_audio_dir,
        'fulsang_raw_dir': args.fulsang_raw_dir,
        'fulsang_audio_dir': args.fulsang_audio_dir,
        'fulsang_mwf_output_dir': args.fulsang_mwf_dir,
        'fulsang_expinfo_dir': args.fulsang_expinfo_dir.strip() or None,
        'combined_dataset_dir': args.combined_dataset_dir,
        'window_size': args.window_size,
        'overlap': args.overlap,
        'target_sampling_rate': fs,
    }
    try:
        sig = __import__('inspect').signature(CombinedDataset.__init__)
        if 'bandpass_low_hz' in sig.parameters:
            kwargs['bandpass_low_hz'] = args.bandpass_low_hz
            kwargs['bandpass_high_hz'] = args.bandpass_high_hz
            kwargs['bandpass_order'] = args.bandpass_order
        if 'use_hilbert_envelope' in sig.parameters:
            kwargs['use_hilbert_envelope'] = args.use_hilbert_envelope
    except Exception:
        pass

    combined = CombinedDataset(**kwargs)
    window_indices = combined.get_window_indices(max_windows=args.max_windows_total)
    print(f"  Total windows: {len(window_indices)}")

    # Verification: label/envelope convention and alignment (same as CombinedCCA)
    print()
    print("  Label and envelope verification:")
    print("    Labels: 0 = attend left, 1 = attend right (attend_lr 1->0, 2->1).")
    print("    Envelopes: left = speaker on left position, right = speaker on right position.")
    print("    Fulsang: when only attend_lr is available, left=Aske (male), right=Marianne (female).")
    print("    Envelope timing: resampled to EEG length per trial; CCA uses 0-250 ms lags (no constant offset).")
    print("    Bandpass: {}-{} Hz (EEG and envelopes). Envelope: {}.".format(
          args.bandpass_low_hz, args.bandpass_high_hz,
          "Hilbert" if args.use_hilbert_envelope else "abs + smoothing"))

    # Split by dataset
    das_indices = []
    fulsang_indices = []
    for i, win in enumerate(window_indices):
        start_idx, end_idx, label, subject_id, trial_idx, dataset_tag = win[0], win[1], win[2], win[3], win[4], win[5]
        kind = _classify_dataset(dataset_tag, subject_id)
        if kind == 'das':
            das_indices.append(i)
        else:
            fulsang_indices.append(i)

    print(f"  Das windows: {len(das_indices)}, Fulsang windows: {len(fulsang_indices)}")

    def collect_xy(indices, max_windows):
        X_list, Y_list = [], []
        n_take = min(max_windows, len(indices))
        for idx in indices[:n_take]:
            win = window_indices[idx]
            start_idx, end_idx, label = win[0], win[1], win[2]
            eeg = combined.eeg_data[start_idx:end_idx]
            eeg = preprocess_window(eeg)
            eeg = make_lagged_eeg(eeg, args.eeg_lag_taps)
            left_env, right_env = combined.get_envelope_window(start_idx, end_idx)
            if left_env is None or right_env is None:
                continue
            if left_env.ndim == 1:
                left_env = left_env.reshape(-1, 1)
            if right_env.ndim == 1:
                right_env = right_env.reshape(-1, 1)
            left_env = make_lagged_audio(left_env, lag_samples, fs)
            right_env = make_lagged_audio(right_env, lag_samples, fs)
            att = left_env if label == 0 else right_env
            X_list.append(eeg)
            Y_list.append(att)
        if not X_list:
            return None, None, 0
        X = np.vstack(X_list).astype(np.float32)
        Y = np.vstack(Y_list).astype(np.float32)
        return X, Y, len(X_list)

    from sklearn.decomposition import PCA

    results = []
    combined_indices = list(range(len(window_indices)))
    for name, indices in [('Das only', das_indices), ('Fulsang only', fulsang_indices), ('Combined', combined_indices)]:
        if not indices:
            print(f"  {name}: no windows, skipping")
            results.append((name, 0, 0, float('nan')))
            continue
        print(f"  Collecting data for {name} (up to {args.max_windows_per_group} windows)...")
        X, Y, n_win = collect_xy(indices, args.max_windows_per_group)
        if X is None:
            print(f"  {name}: no valid envelopes, skipping")
            results.append((name, 0, 0, float('nan')))
            continue
        n_time = X.shape[0]
        # PCA on EEG to match CombinedCCA (768 -> 128)
        n_comp = min(args.pca_components, X.shape[1], X.shape[0] - 1)
        if n_comp < 2:
            results.append((name, n_win, n_time, float('nan')))
            continue
        pca = PCA(n_components=n_comp, random_state=42).fit(X)
        Xp = pca.transform(X).astype(np.float32)
        rho1 = first_cca_correlation(Xp, Y, regularization=args.regularization)
        results.append((name, n_win, n_time, rho1))
        print(f"    Windows: {n_win}, time points: {n_time}, first canonical correlation ρ₁ = {rho1:.4f}")

    print()
    print("=" * 70)
    print("First canonical correlation (ρ₁) by training data")
    print("=" * 70)
    print(f"{'Dataset':<14} {'Windows':>10} {'Time points':>12} {'ρ₁':>10}")
    print("-" * 70)
    for name, n_win, n_time, rho1 in results:
        rho_str = f"{rho1:.4f}" if not np.isnan(rho1) else "N/A"
        print(f"{name:<14} {n_win:>10} {n_time:>12} {rho_str:>10}")
    print("=" * 70)
    print("\nInterpretation: If ρ₁ is higher for Das only or Fulsang only than for Combined,")
    print("the combined CCA is a compromise and loses correlation (different datasets, one direction).")
    print("\nTips to increase ρ₁:")
    print("  • --use_hilbert_envelope     (Hilbert envelope for audio; often better for speech-brain)")
    print("  • --max_windows_per_group N  (more data, e.g. 2000–3000; you have 1411 Das, 3586 Fulsang)")
    print("  • --min_lag_ms / --max_lag_ms (e.g. 50–200 ms or 0–300 ms; speech-brain often peaks ~100–200 ms)")
    print("  • --regularization R          (try 0.001 or 0.1; default 0.01)")
    print("  • --bandpass_low_hz / --bandpass_high_hz (e.g. 1–10 or 4–8 Hz)")
    print("  • --eeg_lag_taps L            (e.g. 8 or 16; default 12)")
    print("  • --pca_components N          (e.g. 64 or 256; default 128)")


if __name__ == '__main__':
    main()
