#!/usr/bin/env python3
"""
PaperCCA — EEG–audio CCA pipeline matching the described paper setup.

Default Das source matches DASCCA.py: das_16subjects_preprocessed/tfrecords @ 128 Hz
(das_preprocessing_16subjects). Override with --das-data-dir / --das-preprocessing-type / --fs-intermediate.

Pipeline summary:
  1) EEG: bandpass (general 1–32 Hz via CombinedDataset), then 1–9 Hz @ intermediate fs,
     resample intermediate Hz → 20 Hz for linear/CCA (default intermediate = 128 to match DASCCA data).
  2) Audio: gammatone subband envelopes (via CombinedDataset when available), per-band
     compression x^0.6, sum to broadband; same resampling to CCA rate.
  3) Time lags: EEG backward model = future taps x_c(t),…,x_c(t+L−1); speech forward
     model = past taps s(t),…,s(t−La+1).
  4) PCA on lagged EEG (regularization); paper note: keeping all components is often optimal.
  5) Multi-component CCA between PCA–EEG and lagged attended speech (training).
  6) Per window / speaker: ρ_j = corr(U_j, V_{i,j}); feature f = ρ_left − ρ_right (binary AAD).
  7) LDA on f (vector of length J).
  8) Decision windows: accuracy = correct windows / total windows × 100.
  9) Nested CV: outer leave-one-segment-out (trial = segment by default; optional subject LOSO),
     inner CV to tune number of canonical components J.

Requires: numpy, scipy, scikit-learn; optional gammatone package for cochlear envelopes.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, filtfilt, resample_poly
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Repo imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from CombinedDataset import CombinedDataset  # noqa: E402


# ---------------------------------------------------------------------------
# Preprocessing (paper-style)
# ---------------------------------------------------------------------------


def butter_bandpass_matrix(x: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    """Bandpass each column of x (samples × channels)."""
    x = np.asarray(x, dtype=np.float64)
    nyq = fs / 2.0
    lo = max(low_hz / nyq, 1e-4)
    hi = min(high_hz / nyq, 0.99)
    if hi <= lo:
        return x.astype(np.float32)
    b, a = butter(order, [lo, hi], btype="band")
    out = np.empty_like(x)
    for c in range(x.shape[1]):
        out[:, c] = filtfilt(b, a, x[:, c])
    return out.astype(np.float32)


def resample_matrix(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Resample along time (axis 0) with ratio fs_out/fs_in = up/down (exact integers)."""
    if abs(fs_in - fs_out) < 1e-6:
        return np.asarray(x, dtype=np.float32)
    fi = int(round(fs_in))
    fo = int(round(fs_out))
    if fi <= 0 or fo <= 0:
        n_out = max(1, int(round(x.shape[0] * fs_out / fs_in)))
        from scipy.signal import resample

        return resample(x, n_out, axis=0).astype(np.float32)
    g = math.gcd(fo, fi)
    up = fo // g
    down = fi // g
    y0 = resample_poly(x[:, 0].astype(np.float64), up, down)
    n_out = y0.shape[0]
    y = np.zeros((n_out, x.shape[1]), dtype=np.float64)
    y[:, 0] = y0
    for c in range(1, x.shape[1]):
        y[:, c] = resample_poly(x[:, c].astype(np.float64), up, down)
    return y.astype(np.float32)


def paper_envelope_compression(env: np.ndarray, power: float = 0.6) -> np.ndarray:
    """Non-negative envelope: apply x^power (paper: 0.6) per column, then sum bands if multi-column."""
    env = np.asarray(env, dtype=np.float32)
    if env.ndim == 1:
        env = env.reshape(-1, 1)
    comp = np.power(np.maximum(env, 0.0), power).astype(np.float32)
    if comp.shape[1] == 1:
        return comp
    return np.sum(comp, axis=1, keepdims=True).astype(np.float32)


# ---------------------------------------------------------------------------
# Time-lagged representations (paper orientation)
# ---------------------------------------------------------------------------


def lag_eeg_future(eeg: np.ndarray, L: int) -> np.ndarray:
    """
    Backward model / EEG side: X(t) stacks future samples per channel
    [x_1(t), x_1(t+1), …, x_1(t+L−1), x_2(t), …, x_C(t+L−1)].
    eeg: (T, C)
    returns: (T, C * L)
    """
    eeg = np.asarray(eeg, dtype=np.float32)
    T, C = eeg.shape
    if L <= 1:
        return eeg.copy()
    pad = np.zeros((L - 1, C), dtype=np.float32)
    ep = np.vstack([eeg, pad])
    sw = sliding_window_view(ep, L, axis=0)
    assert sw.shape[0] >= T
    sw = sw[:T]
    return sw.reshape(T, C * L).astype(np.float32)


def lag_speech_past(s: np.ndarray, La: int) -> np.ndarray:
    """
    Forward model / speech side: S(t) = [s(t), s(t−1), …, s(t−La+1)].
    s: (T,) or (T, 1)
    returns: (T, La)
    """
    s = np.asarray(s, dtype=np.float32).reshape(-1)
    T = s.shape[0]
    if La <= 1:
        return s.reshape(T, 1)
    pad = np.zeros(La - 1, dtype=np.float32)
    saug = np.concatenate([pad, s])
    sw = sliding_window_view(saug, La)
    return sw[:, ::-1].copy().astype(np.float32)


# ---------------------------------------------------------------------------
# CCA + paper feature (ρ vector, LDA on difference)
# ---------------------------------------------------------------------------


def eeg_canonical_scores(X: np.ndarray, cca: CCA) -> np.ndarray:
    """EEG-side canonical variates (same filters for any stimulus)."""
    Xs = (X - cca._x_mean) / cca._x_std
    return (Xs @ cca.x_rotations_).astype(np.float32)


def speech_canonical_scores(Y: np.ndarray, cca: CCA) -> np.ndarray:
    """Speech-side variates using CCA-trained stimulus scaling (fixed across speakers)."""
    Ys = (Y - cca._y_mean) / cca._y_std
    return (Ys @ cca.y_rotations_).astype(np.float32)


def corr_per_component(U: np.ndarray, V: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Pearson correlation per column (canonical component) across time."""
    J = U.shape[1]
    rho = np.zeros(J, dtype=np.float64)
    for j in range(J):
        u = U[:, j].astype(np.float64)
        v = V[:, j].astype(np.float64)
        du = np.std(u)
        dv = np.std(v)
        if du < eps or dv < eps:
            rho[j] = 0.0
        else:
            r = np.corrcoef(u, v)[0, 1]
            rho[j] = 0.0 if np.isnan(r) else float(r)
    return rho


def rho_feature_for_window(
    X_win: np.ndarray,
    S_left_win: np.ndarray,
    S_right_win: np.ndarray,
    pca: PCA,
    cca: CCA,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns rho_left, rho_right, f = rho_left - rho_right for one time window.
    X_win: (T, C*L), S_*: (T, La) already lagged.
    """
    Xp = pca.transform(X_win)
    U = eeg_canonical_scores(Xp, cca)
    Vl = speech_canonical_scores(S_left_win, cca)
    Vr = speech_canonical_scores(S_right_win, cca)
    rho_l = corr_per_component(U, Vl)
    rho_r = corr_per_component(U, Vr)
    f = rho_l - rho_r
    return rho_l.astype(np.float32), rho_r.astype(np.float32), f.astype(np.float32)


@dataclass
class TrialPack:
    start: int
    end: int
    label: int
    subject_id: str
    trial_idx: int
    dataset: str


def build_trial_packs(ds: CombinedDataset) -> List[TrialPack]:
    packs = []
    for i, ((a, b), lab) in enumerate(zip(ds.trial_boundaries, ds.trial_labels)):
        meta = ds.trial_meta[i] if i < len(ds.trial_meta) else {}
        packs.append(
            TrialPack(
                start=a,
                end=b,
                label=int(lab),
                subject_id=str(meta.get("subject_id", "unknown")),
                trial_idx=i,
                dataset=str(meta.get("dataset", "unknown")),
            )
        )
    return packs


def slice_trial(
    eeg: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    pack: TrialPack,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sl = slice(pack.start, pack.end)
    return eeg[sl], left[sl], right[sl]


def fit_pca_cca_from_trials(
    packs: Sequence[TrialPack],
    eeg_full: np.ndarray,
    left_full: np.ndarray,
    right_full: np.ndarray,
    L: int,
    La: int,
    n_cca: int,
    pca_components: Optional[int],
) -> Tuple[PCA, CCA]:
    X_parts: List[np.ndarray] = []
    S_parts: List[np.ndarray] = []
    for pack in packs:
        e, le, re = slice_trial(eeg_full, left_full, right_full, pack)
        if e.shape[0] < max(L, La) + 2:
            continue
        Xlag = lag_eeg_future(e, L)
        s_att = le if pack.label == 0 else re
        Slag = lag_speech_past(s_att, La)
        X_parts.append(Xlag)
        S_parts.append(Slag)
    if not X_parts:
        raise ValueError("No training data for PCA/CCA (trials too short?)")
    X = np.vstack(X_parts)
    S = np.vstack(S_parts)
    n_comp = pca_components
    if n_comp is None or n_comp <= 0:
        n_comp = min(X.shape[0], X.shape[1])
    pca = PCA(n_components=n_comp, svd_solver="full")
    Xp = pca.fit_transform(X)
    cca = CCA(n_components=n_cca, max_iter=2000, tol=1e-6)
    cca.fit(Xp, S)
    return pca, cca


def collect_lda_training_features(
    packs: Sequence[TrialPack],
    eeg_full: np.ndarray,
    left_full: np.ndarray,
    right_full: np.ndarray,
    L: int,
    La: int,
    pca: PCA,
    cca: CCA,
    window_samples: int,
    step_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slide windows inside trials; label = 0 left attended, 1 right attended."""
    Xf: List[np.ndarray] = []
    y: List[int] = []
    for pack in packs:
        e, le, re = slice_trial(eeg_full, left_full, right_full, pack)
        n = e.shape[0]
        if n < window_samples + max(L, La):
            continue
        Xlag = lag_eeg_future(e, L)
        Sl = lag_speech_past(le, La)
        Sr = lag_speech_past(re, La)
        for start in range(0, n - window_samples + 1, step_samples):
            end = start + window_samples
            _, _, fvec = rho_feature_for_window(Xlag[start:end], Sl[start:end], Sr[start:end], pca, cca)
            Xf.append(fvec)
            y.append(pack.label)
    if not Xf:
        raise ValueError("No LDA training windows generated.")
    return np.stack(Xf, axis=0), np.array(y, dtype=np.int64)


def eval_trial_windows(
    pack: TrialPack,
    eeg_full: np.ndarray,
    left_full: np.ndarray,
    right_full: np.ndarray,
    L: int,
    La: int,
    pca: PCA,
    cca: CCA,
    lda: LinearDiscriminantAnalysis,
    window_samples: int,
    step_samples: int,
) -> Tuple[int, int]:
    """Returns (n_correct, n_total) for this trial."""
    e, le, re = slice_trial(eeg_full, left_full, right_full, pack)
    n = e.shape[0]
    if n < window_samples + max(L, La):
        return 0, 0
    Xlag = lag_eeg_future(e, L)
    Sl = lag_speech_past(le, La)
    Sr = lag_speech_past(re, La)
    correct = 0
    total = 0
    for start in range(0, n - window_samples + 1, step_samples):
        end = start + window_samples
        _, _, fvec = rho_feature_for_window(Xlag[start:end], Sl[start:end], Sr[start:end], pca, cca)
        pred = int(lda.predict(fvec.reshape(1, -1))[0])
        if pred == pack.label:
            correct += 1
        total += 1
    return correct, total


def inner_tune_J(
    train_packs: Sequence[TrialPack],
    val_packs: Sequence[TrialPack],
    eeg: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    L: int,
    La: int,
    j_candidates: Sequence[int],
    pca_components: Optional[int],
    window_samples: int,
    step_samples: int,
) -> int:
    best_j = int(j_candidates[0])
    best_acc = -1.0
    for J in j_candidates:
        try:
            pca, cca = fit_pca_cca_from_trials(train_packs, eeg, left, right, L, La, J, pca_components)
            X_lda, y_lda = collect_lda_training_features(
                train_packs, eeg, left, right, L, La, pca, cca, window_samples, step_samples
            )
            lda = LinearDiscriminantAnalysis(solver="svd")
            lda.fit(X_lda, y_lda)
            c_tot = t_tot = 0
            for pk in val_packs:
                c, t = eval_trial_windows(pk, eeg, left, right, L, La, pca, cca, lda, window_samples, step_samples)
                c_tot += c
                t_tot += t
            acc = c_tot / max(1, t_tot)
            if acc > best_acc:
                best_acc = acc
                best_j = int(J)
        except Exception:
            continue
    return best_j


def run_outer_cv(
    all_packs: List[TrialPack],
    eeg: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    L: int,
    La: int,
    j_candidates: Sequence[int],
    pca_components: Optional[int],
    window_sec: float,
    fs_cca: float,
    outer_mode: str,
    inner_frac: float,
    rng: np.random.Generator,
) -> Dict:
    step_samples = max(1, int(round(window_sec * fs_cca)))
    window_samples = step_samples  # decision window length = hop = contiguous non-overlap windows

    results_outer = []

    if outer_mode == "subject":
        subjects = sorted({p.subject_id for p in all_packs})
        outer_units = subjects
    else:
        outer_units = list(range(len(all_packs)))

    for unit in outer_units:
        if outer_mode == "subject":
            test_packs = [p for p in all_packs if p.subject_id == unit]
            train_pool = [p for p in all_packs if p.subject_id != unit]
        else:
            test_packs = [all_packs[unit]]
            train_pool = [p for j, p in enumerate(all_packs) if j != unit]

        if len(train_pool) < 3:
            continue

        idx = np.arange(len(train_pool))
        rng.shuffle(idx)
        n_in = max(1, int(len(train_pool) * (1.0 - inner_frac)))
        tr_idx = set(idx[:n_in].tolist())
        inner_train = [train_pool[i] for i in range(len(train_pool)) if i in tr_idx]
        inner_val = [train_pool[i] for i in range(len(train_pool)) if i not in tr_idx]
        if len(inner_val) == 0:
            inner_val = [inner_train.pop()]

        best_J = inner_tune_J(
            inner_train,
            inner_val,
            eeg,
            left,
            right,
            L,
            La,
            j_candidates,
            pca_components,
            window_samples,
            step_samples,
        )

        pca, cca = fit_pca_cca_from_trials(train_pool, eeg, left, right, L, La, best_J, pca_components)
        X_lda, y_lda = collect_lda_training_features(
            train_pool, eeg, left, right, L, La, pca, cca, window_samples, step_samples
        )
        lda = LinearDiscriminantAnalysis(solver="svd")
        lda.fit(X_lda, y_lda)

        c_tot = t_tot = 0
        for pk in test_packs:
            c, t = eval_trial_windows(pk, eeg, left, right, L, La, pca, cca, lda, window_samples, step_samples)
            c_tot += c
            t_tot += t
        acc = 100.0 * c_tot / max(1, t_tot)
        results_outer.append(
            {
                "outer_unit": str(unit),
                "best_J": best_J,
                "correct": c_tot,
                "total_windows": t_tot,
                "accuracy_pct": acc,
            }
        )

    return {
        "window_sec": window_sec,
        "fs_cca": fs_cca,
        "L_eeg": L,
        "La_speech": La,
        "outer_mode": outer_mode,
        "folds": results_outer,
        "mean_accuracy_pct": float(np.mean([r["accuracy_pct"] for r in results_outer])) if results_outer else 0.0,
    }


def prepare_arrays_after_load(
    ds: CombinedDataset,
    fs_intermediate: float,
    fs_cca: float,
    eeg_bp_low: float,
    eeg_bp_high: float,
    compress_power: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Bandpass EEG, compress envelopes, resample all to fs_cca."""
    eeg = np.asarray(ds.eeg_data, dtype=np.float32)
    left = np.asarray(ds.left_envelope_stream, dtype=np.float32)
    right = np.asarray(ds.right_envelope_stream, dtype=np.float32)
    fs0 = float(ds.sampling_rate)

    eeg = butter_bandpass_matrix(eeg, fs0, eeg_bp_low, eeg_bp_high)
    left = paper_envelope_compression(left, compress_power)
    right = paper_envelope_compression(right, compress_power)

    eeg = resample_matrix(eeg, fs0, fs_intermediate)
    left = resample_matrix(left, fs0, fs_intermediate)
    right = resample_matrix(right, fs0, fs_intermediate)

    # Narrow band again at intermediate rate (linear track 1–9 Hz)
    eeg = butter_bandpass_matrix(eeg, fs_intermediate, eeg_bp_low, min(eeg_bp_high, fs_intermediate / 2 - 1))

    eeg = resample_matrix(eeg, fs_intermediate, fs_cca)
    left = resample_matrix(left, fs_intermediate, fs_cca)
    right = resample_matrix(right, fs_intermediate, fs_cca)

    # Length alignment
    n = min(eeg.shape[0], left.shape[0], right.shape[0])
    eeg = eeg[:n]
    left = left[:n]
    right = right[:n]
    if left.ndim == 1:
        left = left.reshape(-1, 1)
    if right.ndim == 1:
        right = right.reshape(-1, 1)
    # Multi-band: compression+sum already applied in paper_envelope_compression

    return eeg, left, right, fs_cca


def scale_trial_boundaries(ds: CombinedDataset, n_new: int) -> List[Tuple[int, int]]:
    """Map original trial boundaries to new length n_new (proportional)."""
    n_old = len(ds.eeg_data)
    if n_old <= 0:
        return []
    out = []
    for (a, b) in ds.trial_boundaries:
        a2 = int(round(a * n_new / n_old))
        b2 = int(round(b * n_new / n_old))
        b2 = max(a2 + 1, min(b2, n_new))
        out.append((a2, b2))
    return out


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Paper-style CCA + LDA AAD pipeline")
    p.add_argument(
        "--das-data-dir",
        type=str,
        default="das_16subjects_preprocessed",
        help="Parent of tfrecords/ (same convention as DASCCA --tfrecord_dir parent).",
    )
    p.add_argument(
        "--das-preprocessing-type",
        type=str,
        default="COMBINED_DAS",
        help="COMBINED_DAS loads das_data_dir/tfrecords (DASCCA-compatible TFRecords).",
    )
    p.add_argument("--das-original-dir", type=str, default="Data/Das/4004271")
    p.add_argument("--das-audio-dir", type=str, default="Data/Das/4004271/stimuli/stimuli")
    p.add_argument("--fulsang-raw-dir", type=str, default="Data/Fulsang/EEG")
    p.add_argument("--fulsang-audio-dir", type=str, default="Data/Fulsang/AUDIO")
    p.add_argument("--fulsang-expinfo-dir", type=str, default="Exp_Info")
    p.add_argument("--combined-dataset-dir", type=str, default="combined_dataset")
    p.add_argument(
        "--das-mwf-dir",
        type=str,
        default="",
        help="Directory with Das S*_MWF.mat when using --das-preprocessing-type MWF (e.g. MWF_cleaned_DAS).",
    )
    p.add_argument("--overlap", type=float, default=0.25)
    p.add_argument(
        "--fs-intermediate",
        type=float,
        default=128.0,
        help="Sampling rate (Hz) of loaded Das data; 128 matches DASCCA / das_16subjects_preprocessed.",
    )
    p.add_argument("--fs-cca", type=float, default=20.0, help="Linear / CCA rate (Hz)")
    p.add_argument("--eeg-bp-general-low", type=float, default=1.0)
    p.add_argument("--eeg-bp-general-high", type=float, default=32.0)
    p.add_argument("--eeg-bp-linear-low", type=float, default=1.0)
    p.add_argument("--eeg-bp-linear-high", type=float, default=9.0)
    p.add_argument("--compress-power", type=float, default=0.6)
    p.add_argument("--use-gammatone", action="store_true", help="Use gammatone in CombinedDataset envelope extraction")
    p.add_argument("--L-eeg", type=int, default=0, help="EEG future lags; 0 = auto from --eeg-lag-ms")
    p.add_argument("--eeg-lag-ms", type=float, default=350.0, help="If L-eeg=0, L = round(ms * fs_cca / 1000), clamped 250–420 ms style")
    p.add_argument("--La-speech", type=int, default=0, help="Speech past lags; 0 = auto from --encoder-sec")
    p.add_argument("--encoder-sec", type=float, default=1.25, help="Speech stack length in seconds if La-speech=0")
    p.add_argument("--pca-components", type=int, default=0, help="0 = keep all (min(n_samples, n_features))")
    p.add_argument("--j-candidates", type=str, default="2,3,4,5,6,8,10")
    p.add_argument("--window-seconds", type=str, default="1,5,10,30")
    p.add_argument("--outer-mode", type=str, choices=["trial", "subject"], default="trial")
    p.add_argument("--inner-val-frac", type=float, default=0.2)
    p.add_argument("--max-trials", type=int, default=0, help="0 = use all trials")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-json", type=str, default="paper_cca_results.json")
    p.add_argument(
        "--das-only",
        action="store_true",
        help="Load only Das via CombinedDataset(include_fulsang=False); no Fulsang trials.",
    )
    args = p.parse_args()

    # DASCCA uses --tfrecord_dir=.../tfrecords; CombinedDataset expects parent with tfrecords/ inside.
    _dd = Path(args.das_data_dir)
    if _dd.is_dir() and _dd.name == "tfrecords":
        args.das_data_dir = str(_dd.parent)

    rng = np.random.default_rng(args.seed)

    fs_cca = float(args.fs_cca)
    L = int(args.L_eeg)
    if L <= 0:
        L = int(round(args.eeg_lag_ms * fs_cca / 1000.0))
        L = max(2, min(L, int(round(420 * fs_cca / 1000.0))))
    La = int(args.La_speech)
    if La <= 0:
        La = max(2, int(round(args.encoder_sec * fs_cca)))

    pca_comp = int(args.pca_components)
    pca_arg = None if pca_comp <= 0 else pca_comp
    j_list = parse_int_list(args.j_candidates)
    windows = parse_float_list(args.window_seconds)

    win_samples_ds = max(64, int(round(args.fs_intermediate * max(windows))))

    ds_kw: Dict = dict(
        das_data_dir=args.das_data_dir,
        das_preprocessing_type=args.das_preprocessing_type,
        das_original_dir=args.das_original_dir,
        das_audio_dir=args.das_audio_dir,
        fulsang_raw_dir=args.fulsang_raw_dir,
        fulsang_audio_dir=args.fulsang_audio_dir,
        fulsang_expinfo_dir=args.fulsang_expinfo_dir,
        combined_dataset_dir=args.combined_dataset_dir,
        window_size=win_samples_ds,
        overlap=args.overlap,
        target_sampling_rate=int(round(args.fs_intermediate)),
        bandpass_low_hz=args.eeg_bp_general_low,
        bandpass_high_hz=args.eeg_bp_general_high,
        bandpass_order=4,
        use_hilbert_envelope=not args.use_gammatone,
        use_gammatone_filter=args.use_gammatone,
        envelope_normalize="scale_only",
        balance_envelope_energy=True,
    )
    if "include_fulsang" in inspect.signature(CombinedDataset.__init__).parameters:
        ds_kw["include_fulsang"] = not args.das_only
    _mwf = (args.das_mwf_dir or "").strip()
    if _mwf and "das_mwf_dir" in inspect.signature(CombinedDataset.__init__).parameters:
        ds_kw["das_mwf_dir"] = _mwf
    ds = CombinedDataset(**ds_kw)

    eeg, left, right, fs_out = prepare_arrays_after_load(
        ds,
        fs_intermediate=float(args.fs_intermediate),
        fs_cca=fs_cca,
        eeg_bp_low=args.eeg_bp_linear_low,
        eeg_bp_high=args.eeg_bp_linear_high,
        compress_power=args.compress_power,
    )
    assert abs(fs_out - fs_cca) < 0.01

    new_bounds = scale_trial_boundaries(ds, eeg.shape[0])
    labels = list(ds.trial_labels)
    metas = list(ds.trial_meta) if ds.trial_meta else [{}] * len(labels)
    all_packs: List[TrialPack] = []
    for i, ((a, b), lab) in enumerate(zip(new_bounds, labels)):
        m = metas[i] if i < len(metas) else {}
        all_packs.append(
            TrialPack(
                start=a,
                end=b,
                label=int(lab),
                subject_id=str(m.get("subject_id", "unknown")),
                trial_idx=i,
                dataset=str(m.get("dataset", "unknown")),
            )
        )
    if args.max_trials > 0:
        all_packs = all_packs[: args.max_trials]

    report: Dict = {
        "config": {
            "das_only": bool(args.das_only),
            "das_data_dir": args.das_data_dir,
            "das_preprocessing_type": args.das_preprocessing_type,
            "das_original_dir": args.das_original_dir,
            "das_audio_dir": args.das_audio_dir,
            "das_mwf_dir": (args.das_mwf_dir or "").strip() or None,
            "fs_intermediate": args.fs_intermediate,
            "fs_cca": fs_cca,
            "L_eeg": L,
            "La_speech": La,
            "pca_components": "all" if pca_arg is None else pca_arg,
            "j_candidates": j_list,
            "outer_mode": args.outer_mode,
            "use_gammatone": args.use_gammatone,
        },
        "windows": [],
    }

    for wsec in windows:
        res = run_outer_cv(
            all_packs,
            eeg,
            left,
            right,
            L,
            La,
            j_list,
            pca_arg,
            wsec,
            fs_cca,
            args.outer_mode,
            args.inner_val_frac,
            rng,
        )
        report["windows"].append(res)
        print(f"\n=== Decision window {wsec}s ===")
        print(f"Mean accuracy (%): {res['mean_accuracy_pct']:.2f}")
        for fold in res["folds"][:8]:
            print(f"  unit={fold['outer_unit']} J={fold['best_J']} acc={fold['accuracy_pct']:.2f}% ({fold['correct']}/{fold['total_windows']})")
        if len(res["folds"]) > 8:
            print(f"  ... {len(res['folds']) - 8} more folds")

    out_path = Path(args.output_json)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
