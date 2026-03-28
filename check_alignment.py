#!/usr/bin/env python3
"""
Quick sanity check for EEG–envelope alignment and envelope mismatch in the combined dataset.

Runs a simple trial/window-level correlation analysis between EEG and both
attended and unattended envelopes, over a small random subset of windows.
"""

import argparse
import numpy as np
from pathlib import Path

from CombinedDataset import CombinedDataset


def best_corr_1d(x: np.ndarray, y: np.ndarray, max_lag_samples: int) -> tuple[float, int]:
    """Return (best_corr, best_lag_samples) for two 1D signals over ±max_lag_samples."""
    x = np.asarray(x, dtype=np.float32).flatten()
    y = np.asarray(y, dtype=np.float32).flatten()
    if len(x) != len(y) or len(x) < 10:
        return 0.0, 0

    x = x - x.mean()
    y = y - y.mean()
    best_r, best_lag = 0.0, 0
    for lag in range(-max_lag_samples, max_lag_samples + 1):
        if lag > 0:
            xs = x[lag:]
            ys = y[:-lag]
        elif lag < 0:
            xs = x[:lag]
            ys = y[-lag:]
        else:
            xs = x
            ys = y
        if len(xs) < 10:
            continue
        num = float(np.sum(xs * ys))
        den = float(np.sqrt(np.sum(xs ** 2) * np.sum(ys ** 2)) + 1e-8)
        r = num / den
        if abs(r) > abs(best_r):
            best_r, best_lag = r, lag
    return best_r, best_lag


def main():
    parser = argparse.ArgumentParser(description="Check EEG–envelope alignment and mismatch on a subset of windows.")
    parser.add_argument("--combined_dataset_dir", type=str, default="combined_dataset")
    parser.add_argument("--das_data_dir", type=str, default="das_16subjects_preprocessed")
    parser.add_argument("--das_preprocessing_type", type=str, default="COMBINED_DAS")
    parser.add_argument("--das_original_dir", type=str, default="Data/Das/4004271")
    parser.add_argument("--das_audio_dir", type=str, default="Data/Das/4004271/stimuli/stimuli")
    parser.add_argument("--fulsang_raw_dir", type=str, default="Data/Fulsang")
    parser.add_argument("--fulsang_audio_dir", type=str, default="Data/Fulsang/AUDIO")
    parser.add_argument("--fulsang_mwf_dir", type=str, default="MWF_cleaned_Fuglsang")
    parser.add_argument("--fulsang_expinfo_dir", type=str, default="Exp_Info")
    parser.add_argument("--bandpass_low_hz", type=float, default=2.0)
    parser.add_argument("--bandpass_high_hz", type=float, default=8.0)
    parser.add_argument("--bandpass_order", type=int, default=1)
    parser.add_argument("--use_hilbert_envelope", action="store_true", default=True)
    parser.add_argument("--window_size", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--target_sampling_rate", type=int, default=64)
    parser.add_argument("--max_windows", type=int, default=200)
    parser.add_argument("--max_lag_ms", type=float, default=400.0)
    args = parser.parse_args()

    fs = float(args.target_sampling_rate)
    max_lag_samples = int(args.max_lag_ms * fs / 1000.0)

    print("=" * 80)
    print("EEG–ENVELOPE ALIGNMENT CHECK")
    print("=" * 80)

    ds = CombinedDataset(
        das_data_dir=args.das_data_dir,
        das_preprocessing_type=args.das_preprocessing_type,
        das_original_dir=args.das_original_dir,
        das_audio_dir=args.das_audio_dir,
        fulsang_raw_dir=args.fulsang_raw_dir,
        fulsang_audio_dir=args.fulsang_audio_dir,
        fulsang_expinfo_dir=args.fulsang_expinfo_dir,
        fulsang_mwf_output_dir=args.fulsang_mwf_dir,
        combined_dataset_dir=args.combined_dataset_dir,
        window_size=args.window_size,
        overlap=args.overlap,
        target_sampling_rate=args.target_sampling_rate,
        bandpass_low_hz=args.bandpass_low_hz,
        bandpass_high_hz=args.bandpass_high_hz,
        bandpass_order=args.bandpass_order,
        use_hilbert_envelope=args.use_hilbert_envelope,
    )

    window_indices = ds.get_window_indices()
    n_windows = len(window_indices)
    print(f"Total windows in combined dataset: {n_windows}")

    rng = np.random.RandomState(0)
    take = min(args.max_windows, n_windows)
    idxs = rng.choice(n_windows, size=take, replace=False)

    best_att_corrs = []
    best_unatt_corrs = []
    best_att_lags = []
    best_unatt_lags = []

    for idx in idxs:
        start, end, label, subject_id, trial_idx, dataset_tag = window_indices[idx][:6]
        eeg = ds.eeg_data[start:end]  # (T, C)
        # simple summary: mean across channels
        eeg_pc = eeg.mean(axis=1)
        left_env, right_env = ds.get_envelope_window(start, end)
        if left_env is None or right_env is None:
            continue
        left_env = left_env[:, 0]
        right_env = right_env[:, 0]

        if label == 0:
            att = left_env
            unatt = right_env
        else:
            att = right_env
            unatt = left_env

        r_att, lag_att = best_corr_1d(eeg_pc, att, max_lag_samples)
        r_unatt, lag_unatt = best_corr_1d(eeg_pc, unatt, max_lag_samples)

        best_att_corrs.append(r_att)
        best_unatt_corrs.append(r_unatt)
        best_att_lags.append(lag_att / fs)
        best_unatt_lags.append(lag_unatt / fs)

    if not best_att_corrs:
        print("No valid windows for alignment check.")
        return

    att_corrs = np.array(best_att_corrs)
    unatt_corrs = np.array(best_unatt_corrs)
    att_lags = np.array(best_att_lags)
    unatt_lags = np.array(best_unatt_lags)

    print("\nBest correlation over ±{:.0f} ms between EEG(mean over channels) and envelopes:".format(args.max_lag_ms))
    print("Attended envelope:")
    print("  Median r:  {:.4f}".format(np.median(att_corrs)))
    print("  Mean r:    {:.4f} ± {:.4f}".format(att_corrs.mean(), att_corrs.std()))
    print("  Median lag: {:.3f} s".format(np.median(att_lags)))
    print("Unattended envelope:")
    print("  Median r:  {:.4f}".format(np.median(unatt_corrs)))
    print("  Mean r:    {:.4f} ± {:.4f}".format(unatt_corrs.mean(), unatt_corrs.std()))
    print("  Median lag: {:.3f} s".format(np.median(unatt_lags)))

    diff = att_corrs - unatt_corrs
    print("\nAttended − unattended best-corr difference across windows:")
    print("  Median: {:.4f}".format(np.median(diff)))
    print("  Mean:   {:.4f} ± {:.4f}".format(diff.mean(), diff.std()))

    print("\nInterpretation:")
    print("  - If attended r is clearly > unattended r (e.g., median diff > 0.02) and lags cluster around 0.1–0.2s,")
    print("    alignment is likely OK and tracking exists.")
    print("  - If both attended and unattended correlations are ~0 and similar, EEG–envelope tracking is weak or misaligned.")
    print("  - If unattended r ≈ attended r or larger, check label/envelope mapping and experiment design.")


if __name__ == "__main__":
    main()

