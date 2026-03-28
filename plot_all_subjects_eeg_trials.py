#!/usr/bin/env python3
"""
Per-trial EEG-only plots for all subjects (no audio, no dependency on other repo modules).

**Stages (default: all four):** for each trial, PNGs include:

1. **raw** — loaded from .mat (no preprocessing).
2. **filtered** — Butterworth bandpass (default 0.5–40 Hz, order 4), optional notch.
3. **normalized** — z-score per channel after filtering.
4. **averaged** — **one figure**, channel mean ``mean(eeg, axis=1)`` for raw / filtered / normalized as **stacked line subplots** (easier quick view).

Use ``--stages`` to select a subset. Multichannel plot style: ``--plot-style stacked`` (default), ``butterfly``, or ``heatmap`` (averaged is always line plots).

- Das: ``trials[].RawData.EegData`` (time × channels)
- Fulsang: ``data.eeg`` cells (with the same orientation fix as elsewhere when needed)

``--max-time-samples`` optionally subsamples along time **for drawing** only.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio
from scipy import signal


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _default_path(linux: Path, fallback: Path) -> Path:
    return linux if linux.exists() else fallback


def _s_index(stem: str) -> int:
    m = re.match(r"^S(\d+)", stem, re.IGNORECASE)
    return int(m.group(1)) if m else 0


def _subject_s_mat_files(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    rx = re.compile(r"^S\d+\.mat$", re.IGNORECASE)
    out = [p for p in folder.glob("S*.mat") if rx.match(p.name)]
    return sorted(out, key=lambda p: _s_index(p.stem))


def _preproc_mat_files(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    files = list(folder.glob("S*_data_preproc.mat"))
    return sorted(files, key=lambda p: _s_index(p.stem))


def _flatten_trials(trials_obj: Any) -> List[Any]:
    if trials_obj is None:
        return []
    if isinstance(trials_obj, np.ndarray):
        return list(trials_obj.flatten())
    return [trials_obj]


def _plot_this_trial(trial_idx: int, plot_only: Optional[int], plot_dir: Optional[Path]) -> bool:
    if plot_dir is None:
        return False
    if plot_only is None:
        return True
    return trial_idx == plot_only


_STAGE_ORDER = ("raw", "filtered", "normalized", "averaged")


def _coerce_stages(stages: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if stages is None or len(stages) == 0:
        return _STAGE_ORDER
    want = set(stages)
    return tuple(s for s in _STAGE_ORDER if s in want)


def _bandpass_eeg(
    x: np.ndarray,
    fs_hz: float,
    low_hz: float,
    high_hz: float,
    order: int,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.size == 0:
        return x
    if not np.isfinite(fs_hz) or fs_hz <= 0:
        return x.copy()
    nyq = fs_hz / 2.0
    hi_eff = min(high_hz, nyq * 0.99)
    lo = low_hz / nyq
    hi = hi_eff / nyq
    lo = float(np.clip(lo, 0.001, 0.98))
    hi = float(np.clip(hi, lo + 0.001, 0.99))
    try:
        b, a = signal.butter(order, [lo, hi], btype="band")
    except ValueError:
        return x.copy()
    n = x.shape[0]
    padlen = 3 * max(len(a), len(b))
    if n <= padlen:
        return x.copy()
    out = np.empty_like(x)
    for c in range(x.shape[1]):
        try:
            out[:, c] = signal.filtfilt(b, a, x[:, c])
        except Exception:
            out[:, c] = x[:, c]
    return out


def _notch_eeg(x: np.ndarray, fs_hz: float, f0_hz: float, q: float = 30.0) -> np.ndarray:
    if f0_hz <= 0 or not np.isfinite(f0_hz):
        return x
    x = np.asarray(x, dtype=np.float64)
    w0 = f0_hz / (fs_hz / 2.0)
    if w0 <= 0 or w0 >= 1.0:
        return x
    try:
        b, a = signal.iirnotch(w0, q)
    except Exception:
        return x
    padlen = 3 * max(len(a), len(b))
    if x.shape[0] <= padlen:
        return x.copy()
    out = np.empty_like(x)
    for c in range(x.shape[1]):
        try:
            out[:, c] = signal.filtfilt(b, a, x[:, c])
        except Exception:
            out[:, c] = x[:, c]
    return out


def _zscore_per_channel(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = np.mean(x, axis=0, keepdims=True)
    s = np.std(x, axis=0, keepdims=True)
    s = np.maximum(s, eps)
    return (x - m) / s


def _prepare_stage_arrays(
    raw: np.ndarray,
    fs_hz: float,
    coerced: Sequence[str],
    bandpass_low_hz: float,
    bandpass_high_hz: float,
    bandpass_order: int,
    notch_hz: float,
) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, np.ndarray]]]:
    raw = np.asarray(raw, dtype=np.float64)
    st_mult = {s for s in coerced if s != "averaged"}
    want_avg = "averaged" in coerced
    need_filt = ("filtered" in st_mult) or ("normalized" in st_mult) or want_avg
    filtered: Optional[np.ndarray] = None
    normalized: Optional[np.ndarray] = None
    if need_filt and np.isfinite(fs_hz) and fs_hz > 0:
        hi_eff = min(bandpass_high_hz, (fs_hz / 2.0) * 0.99)
        filtered = _bandpass_eeg(raw, fs_hz, bandpass_low_hz, hi_eff, bandpass_order)
        if notch_hz > 0:
            filtered = _notch_eeg(filtered, fs_hz, notch_hz)
        normalized = _zscore_per_channel(filtered)
    out: Dict[str, np.ndarray] = {}
    if "raw" in st_mult:
        out["raw"] = raw
    if "filtered" in st_mult and filtered is not None:
        out["filtered"] = filtered
    if "normalized" in st_mult and normalized is not None:
        out["normalized"] = normalized
    avg_bundle: Optional[Dict[str, np.ndarray]] = None
    if want_avg and filtered is not None and normalized is not None:
        avg_bundle = {"raw": raw, "filtered": filtered, "normalized": normalized}
    return out, avg_bundle


def _stage_title_line(
    stage: str,
    bandpass_low_hz: float,
    bandpass_high_hz: float,
    notch_hz: float,
) -> str:
    if stage == "raw":
        return "1) Raw EEG (no preprocessing)"
    if stage == "filtered":
        t = f"2) After preprocessing: bandpass {bandpass_low_hz:g}-{bandpass_high_hz:g} Hz"
        if notch_hz > 0:
            t += f", notch {notch_hz:g} Hz"
        return t
    if stage == "normalized":
        return "3) After normalization: z-score per channel (mean 0, unit variance)"
    if stage == "averaged":
        return "4) Channel-averaged (mean over channels)"
    return stage


def _get_fulsang_fsample(first_elem: Any) -> float:
    if not hasattr(first_elem, "fsample"):
        return 64.0
    fs = first_elem.fsample
    if hasattr(fs, "eeg"):
        return float(np.asarray(fs.eeg).flatten()[0])
    return float(np.asarray(fs).flatten()[0])


def _eeg_trial_list(first_elem: Any) -> List[np.ndarray]:
    if not hasattr(first_elem, "eeg"):
        return []
    cell = first_elem.eeg
    if not isinstance(cell, np.ndarray) or cell.dtype != object:
        return []
    out = []
    for i in range(cell.size):
        t = np.asarray(cell.flat[i], dtype=np.float32)
        if t.ndim == 2:
            if t.shape[0] < t.shape[1] and t.shape[0] <= 128:
                t = t.T
        elif t.ndim == 1:
            t = t.reshape(-1, 1)
        out.append(t)
    return out


def _time_axis_and_eeg_for_plot(
    eeg: np.ndarray,
    fs_hz: float,
    max_time_samples: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Return (t_seconds, eeg_draw, note) for plotting; optional time stride."""
    n = eeg.shape[0]
    if max_time_samples > 0 and n > max_time_samples:
        step = int(np.ceil(n / max_time_samples))
        idx = np.arange(0, n, step, dtype=np.int64)
        eeg_d = eeg[idx, :]
        t = idx.astype(np.float64) / float(fs_hz)
        note = f" | drawn every {step} sample ({n} total)"
    else:
        eeg_d = eeg
        t = np.arange(n, dtype=np.float64) / float(fs_hz)
        note = ""
    return t, eeg_d, note


def _time_axis_for_1d(
    y: np.ndarray,
    fs_hz: float,
    max_time_samples: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(y.size)
    if n == 0:
        return np.array([]), y, ""
    if max_time_samples > 0 and n > max_time_samples:
        step = int(np.ceil(n / max_time_samples))
        idx = np.arange(0, n, step, dtype=np.int64)
        yp = y[idx]
        t = idx.astype(np.float64) / float(fs_hz)
        note = f" | drawn every {step} sample ({n} total)"
    else:
        yp = y
        t = np.arange(n, dtype=np.float64) / float(fs_hz)
        note = ""
    return t, yp, note


def _save_eeg_signal_stacked(
    eeg: np.ndarray,
    fs_hz: float,
    out: Path,
    suptitle: str,
    max_time_samples: int,
    verbose: bool,
) -> Optional[Path]:
    """Stacked offset line traces — one waveform per channel (raw units)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eeg = np.asarray(eeg, dtype=np.float64)
    if eeg.ndim != 2 or eeg.size == 0 or not np.isfinite(fs_hz) or fs_hz <= 0:
        return None

    n, n_ch = eeg.shape
    t, eeg_d, note = _time_axis_and_eeg_for_plot(eeg, fs_hz, max_time_samples)

    if n_ch == 1:
        fig, ax = plt.subplots(figsize=(14, 3.5), constrained_layout=True)
        ax.plot(t, eeg_d[:, 0], color="0.1", lw=0.45)
        ax.set_ylabel("ch 0 (.mat units)")
    else:
        ptp = np.ptp(eeg_d, axis=0)
        pos = ptp[np.isfinite(ptp) & (ptp > 0)]
        spacing = float(np.median(pos)) if pos.size else 0.0
        if spacing <= 0 or not np.isfinite(spacing):
            spacing = float(np.nanstd(eeg_d)) * 2.5
        if spacing <= 0 or not np.isfinite(spacing):
            spacing = 1.0
        spacing *= 1.12

        fig_h = max(5.0, min(36.0, 0.32 * n_ch + 2.5))
        fig, ax = plt.subplots(figsize=(14, fig_h), constrained_layout=True)
        for c in range(n_ch):
            off = (n_ch - 1 - c) * spacing
            ax.plot(t, eeg_d[:, c] + off, color="0.12", lw=0.32, solid_capstyle="round")
        ymin = -0.55 * spacing
        ymax = (n_ch - 0.45) * spacing
        ax.set_ylim(ymin, ymax)
        tick_step = max(1, n_ch // 20)
        yticks = [(n_ch - 1 - c) * spacing for c in range(0, n_ch, tick_step)]
        ylabels = [str(c) for c in range(0, n_ch, tick_step)]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=8)
        ax.set_ylabel("channel # (offset traces, 0 at top)")

    ax.set_xlabel(f"Time (s)  |  fs = {fs_hz:.6g} Hz  |  {n} × {n_ch}")
    fig.suptitle(suptitle + note, fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    if verbose:
        print(f"  Saved: {out.resolve()}")
    return out.resolve()


def _save_eeg_signal_butterfly(
    eeg: np.ndarray,
    fs_hz: float,
    out: Path,
    suptitle: str,
    max_time_samples: int,
    verbose: bool,
) -> Optional[Path]:
    """All channels overlaid on one y-axis (raw units)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eeg = np.asarray(eeg, dtype=np.float64)
    if eeg.ndim != 2 or eeg.size == 0 or not np.isfinite(fs_hz) or fs_hz <= 0:
        return None

    n, n_ch = eeg.shape
    t, eeg_d, note = _time_axis_and_eeg_for_plot(eeg, fs_hz, max_time_samples)

    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    alpha = min(0.85, 18.0 / max(n_ch, 1))
    for c in range(n_ch):
        ax.plot(t, eeg_d[:, c], color="k", lw=0.28, alpha=alpha)
    ax.set_xlabel(f"Time (s)  |  fs = {fs_hz:.6g} Hz  |  {n} × {n_ch}")
    ax.set_ylabel("amplitude (.mat units)")
    fig.suptitle(suptitle + note + " | butterfly overlay", fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    if verbose:
        print(f"  Saved: {out.resolve()}")
    return out.resolve()


def _save_eeg_native_heatmap(
    eeg: np.ndarray,
    fs_hz: float,
    out: Path,
    suptitle: str,
    max_time_samples: int = 0,
    verbose: bool = False,
) -> Optional[Path]:
    """Raw values as a time × channel image; vmin/vmax from data (no normalization)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eeg = np.asarray(eeg, dtype=np.float64)
    if eeg.ndim != 2 or eeg.size == 0 or not np.isfinite(fs_hz) or fs_hz <= 0:
        return None

    n, n_ch = eeg.shape
    duration_s = n / float(fs_hz)
    _, eeg_plot, stride_note = _time_axis_and_eeg_for_plot(eeg, fs_hz, max_time_samples)
    note = stride_note + f" | span {duration_s:.3f}s"

    vmin = float(np.nanmin(eeg_plot))
    vmax = float(np.nanmax(eeg_plot))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = -1.0, 1.0

    fig_h = max(4.0, min(28.0, 0.2 * n_ch + 2.5))
    fig, ax = plt.subplots(figsize=(14, fig_h), constrained_layout=True)
    im = ax.imshow(
        eeg_plot.T,
        aspect="auto",
        origin="upper",
        extent=[0.0, duration_s, float(n_ch), 0.0],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_ylabel("channel (column index in .mat, 0 = top)")
    ax.set_xlabel(f"Time (s)  |  fs = {fs_hz:.6g} Hz  |  {n} samples × {n_ch} ch")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="amplitude (.mat units)")
    fig.suptitle(suptitle + note, fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    if verbose:
        print(f"  Saved: {out.resolve()}")
    return out.resolve()


def _save_eeg_figure(
    eeg: np.ndarray,
    fs_hz: float,
    out: Path,
    suptitle: str,
    max_time_samples: int,
    plot_style: str,
    verbose: bool,
) -> Optional[Path]:
    if plot_style == "heatmap":
        return _save_eeg_native_heatmap(eeg, fs_hz, out, suptitle, max_time_samples, verbose)
    if plot_style == "butterfly":
        return _save_eeg_signal_butterfly(eeg, fs_hz, out, suptitle, max_time_samples, verbose)
    return _save_eeg_signal_stacked(eeg, fs_hz, out, suptitle, max_time_samples, verbose)


def _save_channel_average_figure(
    bundle: Dict[str, np.ndarray],
    fs_hz: float,
    out: Path,
    suptitle: str,
    max_time_samples: int,
    bandpass_low_hz: float,
    bandpass_high_hz: float,
    notch_hz: float,
    verbose: bool,
) -> Optional[Path]:
    """Three subplots: mean(eeg, axis=1) for raw, filtered, normalized."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not np.isfinite(fs_hz) or fs_hz <= 0:
        return None

    order_keys = ("raw", "filtered", "normalized")
    rows: List[Tuple[str, np.ndarray, str]] = []
    for key in order_keys:
        if key not in bundle or bundle[key] is None:
            continue
        arr = np.asarray(bundle[key], dtype=np.float64)
        if arr.ndim != 2 or arr.size == 0:
            continue
        mean_1d = np.mean(arr, axis=1)
        label = _stage_title_line(key, bandpass_low_hz, bandpass_high_hz, notch_hz)
        rows.append((key, mean_1d, label))

    if not rows:
        return None

    fig_h = max(4.0, 2.6 * len(rows))
    fig, axes = plt.subplots(len(rows), 1, figsize=(14, fig_h), sharex=True, constrained_layout=True)
    if len(rows) == 1:
        axes = np.array([axes])
    last_note = ""
    for ax, (_k, mean_1d, label) in zip(axes, rows):
        t, yp, last_note = _time_axis_for_1d(mean_1d, fs_hz, max_time_samples)
        if t.size == 0:
            continue
        ax.plot(t, yp, color="0.15", lw=0.55)
        ax.set_ylabel("mean(ch)", fontsize=9)
        ax.set_title(label, fontsize=9, loc="left")
    axes[-1].set_xlabel(f"Time (s)  |  fs = {fs_hz:.6g} Hz{last_note}")
    fig.suptitle(suptitle + " | easier view: average over channels", fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    if verbose:
        print(f"  Saved: {out.resolve()}")
    return out.resolve()


def process_das_mat(
    das_mat: Path,
    max_trials: Optional[int],
    plot_output_dir: Optional[Path],
    plot_only_trial: Optional[int],
    verbose: bool,
    max_time_samples: int,
    plot_style: str,
    stages: Sequence[str],
    bandpass_low_hz: float,
    bandpass_high_hz: float,
    bandpass_order: int,
    notch_hz: float,
) -> Tuple[int, List[Path]]:
    saved: List[Path] = []
    issues = 0
    if not das_mat.exists():
        if verbose:
            print(f"  missing file: {das_mat}")
        return 1, saved

    mat = sio.loadmat(str(das_mat), squeeze_me=True, struct_as_record=False)
    if "trials" not in mat:
        print(f"  {das_mat.name}: no 'trials' key")
        return 1, saved

    trials = _flatten_trials(mat["trials"])
    n = len(trials) if max_trials is None else min(len(trials), max_trials)

    for i in range(n):
        tr = trials[i]
        try:
            eeg = np.asarray(tr.RawData.EegData, dtype=np.float64)
        except Exception as e:
            print(f"  Trial {i}: RawData.EegData: {e}")
            issues += 1
            continue
        if eeg.ndim != 2:
            print(f"  Trial {i}: EegData shape {eeg.shape} (need 2D)")
            issues += 1
            continue
        sr = np.asarray(tr.FileHeader.SampleRate).flatten()
        fs_eeg = float(sr[0]) if sr.size else float("nan")

        if verbose:
            print(f"  Trial {i}: {eeg.shape[0]} x {eeg.shape[1]} @ {fs_eeg:.2f} Hz")

        if not _plot_this_trial(i, plot_only_trial, plot_output_dir):
            continue
        assert plot_output_dir is not None
        stage_arrays, avg_bundle = _prepare_stage_arrays(
            eeg,
            fs_eeg,
            stages,
            bandpass_low_hz,
            bandpass_high_hz,
            bandpass_order,
            notch_hz,
        )
        for stage in stages:
            if stage == "averaged":
                if avg_bundle is not None:
                    out_avg = plot_output_dir / f"{das_mat.stem}_trial{i}_das_eeg_averaged.png"
                    tag_avg = f"Das {das_mat.stem} trial {i}"
                    rp = _save_channel_average_figure(
                        avg_bundle,
                        fs_eeg,
                        out_avg,
                        tag_avg,
                        max_time_samples,
                        bandpass_low_hz,
                        bandpass_high_hz,
                        notch_hz,
                        verbose,
                    )
                    if rp is not None:
                        saved.append(Path(rp))
                elif verbose:
                    print(f"  Trial {i}: skipped averaged (missing filtered/normalized)")
                continue
            arr = stage_arrays.get(stage)
            if arr is None:
                continue
            suf = stage
            out_png = plot_output_dir / f"{das_mat.stem}_trial{i}_das_eeg_{suf}.png"
            tag = f"{_stage_title_line(stage, bandpass_low_hz, bandpass_high_hz, notch_hz)} | Das {das_mat.stem} trial {i}"
            rp = _save_eeg_figure(arr, fs_eeg, out_png, tag, max_time_samples, plot_style, verbose)
            if rp is not None:
                saved.append(Path(rp))

    return issues, saved


def process_fulsang_mat(
    ful_mat: Path,
    max_trials: Optional[int],
    plot_output_dir: Optional[Path],
    plot_only_trial: Optional[int],
    verbose: bool,
    title_prefix: str,
    max_time_samples: int,
    plot_style: str,
    stages: Sequence[str],
    bandpass_low_hz: float,
    bandpass_high_hz: float,
    bandpass_order: int,
    notch_hz: float,
) -> Tuple[int, List[Path]]:
    saved: List[Path] = []
    issues = 0
    if not ful_mat.exists():
        if verbose:
            print(f"  missing file: {ful_mat}")
        return 1, saved

    mat = sio.loadmat(str(ful_mat), squeeze_me=True, struct_as_record=False)
    if "data" not in mat:
        print(f"  {ful_mat.name}: no 'data' key")
        return 1, saved

    data_struct = mat["data"]
    first = data_struct.flat[0] if isinstance(data_struct, np.ndarray) else data_struct
    fs_eeg = _get_fulsang_fsample(first)
    eeg_trials = _eeg_trial_list(first)
    if not eeg_trials:
        print(f"  {ful_mat.name}: no EEG trials in data.eeg")
        return 1, saved

    n_eeg = len(eeg_trials)
    n = n_eeg if max_trials is None else min(n_eeg, max_trials)
    if verbose:
        print(f"  fsample: {fs_eeg:g} Hz, trials: {n_eeg}, plotting: {n}")

    for i in range(n):
        e = eeg_trials[i]
        if verbose:
            print(f"  Trial {i}: {e.shape}")

        if not _plot_this_trial(i, plot_only_trial, plot_output_dir):
            continue
        assert plot_output_dir is not None
        raw = e.astype(np.float64)
        stage_arrays, avg_bundle = _prepare_stage_arrays(
            raw,
            fs_eeg,
            stages,
            bandpass_low_hz,
            bandpass_high_hz,
            bandpass_order,
            notch_hz,
        )
        for stage in stages:
            if stage == "averaged":
                if avg_bundle is not None:
                    out_avg = plot_output_dir / f"{ful_mat.stem}_trial{i}_fulsang_eeg_averaged.png"
                    tag_avg = f"{title_prefix} {ful_mat.stem} trial {i}"
                    rp = _save_channel_average_figure(
                        avg_bundle,
                        fs_eeg,
                        out_avg,
                        tag_avg,
                        max_time_samples,
                        bandpass_low_hz,
                        bandpass_high_hz,
                        notch_hz,
                        verbose,
                    )
                    if rp is not None:
                        saved.append(Path(rp))
                elif verbose:
                    print(f"  Trial {i}: skipped averaged (missing filtered/normalized)")
                continue
            arr = stage_arrays.get(stage)
            if arr is None:
                continue
            out_png = plot_output_dir / f"{ful_mat.stem}_trial{i}_fulsang_eeg_{stage}.png"
            tag = (
                f"{_stage_title_line(stage, bandpass_low_hz, bandpass_high_hz, notch_hz)} | "
                f"{title_prefix} {ful_mat.stem} trial {i}"
            )
            rp = _save_eeg_figure(arr, fs_eeg, out_png, tag, max_time_samples, plot_style, verbose)
            if rp is not None:
                saved.append(Path(rp))

    return issues, saved


def main() -> int:
    repo = _repo_root()
    linux_das = Path("/home/py9363/telluride_decoding/Data/Das/4004271")
    linux_ful_eeg = Path("/home/py9363/telluride_decoding/Data/Fulsang/EEG")
    linux_ful_pre = Path("/home/py9363/telluride_decoding/Data/Fulsang/DATA_preproc")

    p = argparse.ArgumentParser(description="Save per-trial EEG-only PNGs (Das + Fulsang).")
    p.add_argument(
        "--das-dir",
        type=str,
        default=str(_default_path(linux_das, repo / "Data" / "Das" / "4004271")),
    )
    p.add_argument(
        "--fulsang-eeg-dir",
        type=str,
        default=str(_default_path(linux_ful_eeg, repo / "Data" / "Fulsang" / "EEG")),
    )
    p.add_argument(
        "--fulsang-preproc-dir",
        type=str,
        default=str(_default_path(linux_ful_pre, repo / "Data" / "Fulsang" / "DATA_preproc")),
    )
    p.add_argument(
        "--plot-out",
        type=str,
        default=str(repo / "eeg_trial_figures_3stage"),
    )
    p.add_argument("--max-trials", type=int, default=0, help="0 = all trials per subject")
    p.add_argument("--plot-only-trial", type=int, default=None)
    p.add_argument("--skip-das", action="store_true")
    p.add_argument("--skip-fulsang-eeg", action="store_true")
    p.add_argument("--skip-fulsang-preproc", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--no-save-plots", action="store_true")
    p.add_argument(
        "--max-time-samples",
        type=int,
        default=0,
        help="If >0 and trial is longer, draw every k-th sample along time (fewer points); "
        "0 = plot every sample.",
    )
    p.add_argument(
        "--plot-style",
        type=str,
        choices=("stacked", "butterfly", "heatmap"),
        default="stacked",
        help="stacked = offset waveform per channel (default); butterfly = overlay; heatmap = time×ch image",
    )
    p.add_argument(
        "--stages",
        nargs="*",
        default=None,
        choices=("raw", "filtered", "normalized", "averaged"),
        metavar="STAGE",
        help="Pipeline stages to save (default: all four). 'averaged' = one figure, mean over channels (3 subplots).",
    )
    p.add_argument("--bandpass-low-hz", type=float, default=0.5, help="Bandpass low cutoff (Hz)")
    p.add_argument("--bandpass-high-hz", type=float, default=40.0, help="Bandpass high cutoff (Hz); clipped to Nyquist")
    p.add_argument("--bandpass-order", type=int, default=4, help="Butterworth order for bandpass")
    p.add_argument(
        "--notch-hz",
        type=float,
        default=0.0,
        help="Line notch frequency (e.g. 50 or 60); 0 = disabled. Applied after bandpass on filtered/normalized path.",
    )
    args = p.parse_args()

    das_dir = Path(args.das_dir)
    ful_eeg = Path(args.fulsang_eeg_dir)
    ful_pre = Path(args.fulsang_preproc_dir)
    plot_root = Path(args.plot_out)
    max_tr: Optional[int] = args.max_trials if args.max_trials > 0 else None
    plot_dir: Optional[Path] = None if args.no_save_plots else plot_root
    stages = _coerce_stages(args.stages)

    total_png = 0
    total_issues = 0

    if not args.skip_das:
        das_files = _subject_s_mat_files(das_dir)
        if not das_files:
            print(f"Das: no S*.mat under {das_dir}")
        for mat in das_files:
            sub_out = None if plot_dir is None else plot_dir / "Das" / mat.stem
            print(f"Das: {mat.name} -> {sub_out or '(no plots)'}")
            iss, pngs = process_das_mat(
                mat,
                max_tr,
                sub_out,
                args.plot_only_trial,
                args.verbose,
                args.max_time_samples,
                args.plot_style,
                stages,
                args.bandpass_low_hz,
                args.bandpass_high_hz,
                args.bandpass_order,
                args.notch_hz,
            )
            total_issues += iss
            total_png += len(pngs)

    if not args.skip_fulsang_eeg:
        files = _subject_s_mat_files(ful_eeg)
        if not files:
            print(f"Fulsang EEG: no S*.mat under {ful_eeg}")
        for mat in files:
            sub_out = None if plot_dir is None else plot_root / "Fulsang_EEG" / mat.stem
            print(f"Fulsang raw: {mat.name} -> {sub_out or '(no plots)'}")
            iss, pngs = process_fulsang_mat(
                mat,
                max_tr,
                sub_out,
                args.plot_only_trial,
                args.verbose,
                "Fulsang raw",
                args.max_time_samples,
                args.plot_style,
                stages,
                args.bandpass_low_hz,
                args.bandpass_high_hz,
                args.bandpass_order,
                args.notch_hz,
            )
            total_issues += iss
            total_png += len(pngs)

    if not args.skip_fulsang_preproc:
        files = _preproc_mat_files(ful_pre)
        if not files:
            print(f"Fulsang preproc: no S*_data_preproc.mat under {ful_pre}")
        for mat in files:
            stem = mat.stem.replace("_data_preproc", "")
            sub_out = None if plot_dir is None else plot_root / "Fulsang_preproc" / stem
            print(f"Fulsang preproc: {mat.name} -> {sub_out or '(no plots)'}")
            iss, pngs = process_fulsang_mat(
                mat,
                max_tr,
                sub_out,
                args.plot_only_trial,
                args.verbose,
                "Fulsang preproc",
                args.max_time_samples,
                args.plot_style,
                stages,
                args.bandpass_low_hz,
                args.bandpass_high_hz,
                args.bandpass_order,
                args.notch_hz,
            )
            total_issues += iss
            total_png += len(pngs)

    print("\n" + "=" * 72)
    if plot_dir:
        print(f"Finished: {total_png} PNG(s) under {plot_root.resolve()}")
    else:
        print("Finished (no plots saved).")
    if total_issues:
        print(f"Load/structure issues: {total_issues}")
    print("=" * 72)
    return 0 if total_issues == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
