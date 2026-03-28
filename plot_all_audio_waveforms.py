#!/usr/bin/env python3
"""
Plot every audio file as its own PDF: full duration on the time axis, fixed y-limits
for clipping checks, and min–max decimation so peaks are not missed visually.

Reads WAV via scipy (integer PCM) or soundfile/librosa for other formats.
Processes long files in chunks so memory stays bounded.

Usage:
  python plot_all_audio_waveforms.py --output-dir audio_waveform_pdfs \\
      --das-audio-dir Data/Das/4004271/stimuli/stimuli \\
      --fulsang-audio-dir Data/Fulsang/AUDIO

  # Single directory, recursive:
  python plot_all_audio_waveforms.py --audio-dir /path/to/audio --recursive

Environment (optional):
  DAS_AUDIO_DIR, FULSANG_AUDIO_DIR — used as defaults if CLI args omitted.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Audio I/O
# -----------------------------------------------------------------------------


def _collect_files(
    directory: Path,
    extensions: Tuple[str, ...] = (".wav", ".WAV", ".mp3", ".MP3"),
    recursive: bool = False,
) -> List[Path]:
    if not directory.exists():
        return []
    out: List[Path] = []
    if recursive:
        for ext in extensions:
            out.extend(directory.rglob(f"*{ext}"))
    else:
        for ext in extensions:
            out.extend(directory.glob(f"*{ext}"))
    return sorted(set(out))


def _safe_pdf_name(src: Path, output_dir: Path) -> Path:
    stem = re.sub(r'[^\w\-]+', "_", src.stem, flags=re.UNICODE).strip("_") or "audio"
    base = f"{stem}_waveform.pdf"
    out = output_dir / base
    if not out.exists():
        return out
    # Disambiguate (same filename in different folders)
    parent = re.sub(r'[^\w\-]+', "_", src.parent.name, flags=re.UNICODE).strip("_") or "dir"
    return output_dir / f"{parent}__{stem}_waveform.pdf"


def _integer_clip_limits(data: np.ndarray) -> Tuple[float, float]:
    """Symmetric y-limits for integer PCM (clipping at dtype extremes)."""
    if not np.issubdtype(data.dtype, np.integer):
        raise ValueError("expected integer array")
    info = np.iinfo(data.dtype)
    return float(info.min), float(info.max)


def _float_clip_limits() -> Tuple[float, float]:
    return -1.0, 1.0


def _format_hms(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = int(round((seconds % 1.0) * 1000))
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}.{ms:03d}"
    return f"{m:d}:{s:02d}.{ms:03d}"


def _time_tick_step(duration_s: float) -> Tuple[float, float]:
    """
    Return (major_step_s, minor_step_s) so ticks stay readable.
    Every second gets a minor tick when duration allows.
    """
    if duration_s <= 1.0:
        return 0.2, 0.05
    if duration_s <= 10.0:
        return 1.0, 0.2
    if duration_s <= 60.0:
        return 5.0, 1.0
    if duration_s <= 120:
        return 10.0, 1.0
    if duration_s <= 600:
        return 30.0, 5.0
    if duration_s <= 3600:
        return 60.0, 10.0
    if duration_s <= 4 * 3600:
        return 300.0, 60.0
    return 600.0, 60.0


def read_audio_scipy_wav(path: Path) -> Tuple[int, np.ndarray]:
    from scipy.io import wavfile

    fs, data = wavfile.read(str(path))
    if data.ndim == 1:
        data = data[:, np.newaxis]
    return int(fs), data


def try_soundfile_open(path: Path):
    try:
        import soundfile as sf

        return sf.SoundFile(str(path))
    except ImportError:
        return None


def read_audio_librosa(path: Path) -> Tuple[int, np.ndarray]:
    import librosa

    data, fs = librosa.load(str(path), sr=None, mono=False)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    else:
        data = data.T
    return int(fs), data.astype(np.float64, copy=False)


def process_with_soundfile(
    path: Path,
    n_bins: int,
    chunk_frames: int,
) -> Tuple[
    int,
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    str,
    float,
    float,
    int,
    int,
    int,
]:
    """
    Stream file with soundfile (float64 samples, typically normalized PCM to ±1).
    Clipping is counted at ±1.0 (digital full scale after decode/normalize).

    Returns:
      fs, n_frames, bin_min (n_bins, ch), bin_max (n_bins, ch), rms_per_ch,
      dtype_kind, peak_linear, crest_db, n_clip_pos, n_clip_neg, n_total_samples
    """
    sf_file = try_soundfile_open(path)
    if sf_file is None:
        raise RuntimeError("soundfile not available")

    clip_hi = 1.0
    clip_lo = -1.0
    eps = 1e-6

    with sf_file as f:
        fs = int(f.samplerate)
        n_frames = int(f.frames)
        ch = int(f.channels)
        subtype = str(f.subtype)
        dtype_kind = f"soundfile subtype={subtype}"

        if n_frames <= 0 or ch <= 0:
            raise ValueError("empty or invalid audio file")

        bin_min = np.full((n_bins, ch), np.inf, dtype=np.float64)
        bin_max = np.full((n_bins, ch), -np.inf, dtype=np.float64)
        sum_sq = np.zeros(ch, dtype=np.float64)
        peak = np.zeros(ch, dtype=np.float64)
        n_clip_pos = np.zeros(ch, dtype=np.int64)
        n_clip_neg = np.zeros(ch, dtype=np.int64)
        n_count = 0

        while f.tell() < n_frames:
            block = f.read(chunk_frames, dtype="float64", always_2d=True)
            if block.size == 0:
                break
            n = block.shape[0]
            if block.ndim == 1:
                block = block.reshape(-1, 1)
            b = block

            sum_sq += np.sum(b * b, axis=0)
            peak = np.maximum(peak, np.max(np.abs(b), axis=0))
            n_count += n

            n_clip_pos += np.sum(b >= clip_hi - eps, axis=0)
            n_clip_neg += np.sum(b <= clip_lo + eps, axis=0)

            end = f.tell()
            global_idx = np.arange(end - n, end, dtype=np.int64)
            bin_idx = (global_idx * n_bins) // max(n_frames, 1)
            bin_idx = np.minimum(bin_idx, n_bins - 1)

            for c in range(ch):
                col = b[:, c]
                np.minimum.at(bin_min[:, c], bin_idx, col)
                np.maximum.at(bin_max[:, c], bin_idx, col)

        rms = np.sqrt(sum_sq / max(n_count, 1))
        peak_linear = float(np.max(peak))
        crest_db = 20.0 * np.log10((peak_linear + 1e-20) / (float(np.max(rms)) + 1e-20))

        return (
            fs,
            n_frames,
            bin_min,
            bin_max,
            rms,
            dtype_kind,
            peak_linear,
            float(crest_db),
            int(np.sum(n_clip_pos)),
            int(np.sum(n_clip_neg)),
            n_count * ch,
        )


def process_numpy_in_memory(
    fs: int,
    data: np.ndarray,
    n_bins: int,
) -> Tuple[
    int,
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    str,
    float,
    float,
    int,
    int,
    int,
]:
    """Fallback when whole array is already in memory (scipy wav read)."""
    if data.ndim == 1:
        data = data[:, np.newaxis]
    n_frames, ch = data.shape
    work = data.astype(np.float64, copy=False)

    dtype_kind = str(data.dtype)
    is_int = np.issubdtype(data.dtype, np.integer)
    if is_int:
        info = np.iinfo(data.dtype)
        clip_hi = float(info.max)
        clip_lo = float(info.min)
        n_clip_pos = int(np.sum(work >= clip_hi - 0.5))
        n_clip_neg = int(np.sum(work <= clip_lo + 0.5))
    else:
        clip_hi, clip_lo = 1.0, -1.0
        eps = 1e-6
        n_clip_pos = int(np.sum(work >= clip_hi - eps))
        n_clip_neg = int(np.sum(work <= clip_lo + eps))

    bin_min = np.full((n_bins, ch), np.inf, dtype=np.float64)
    bin_max = np.full((n_bins, ch), -np.inf, dtype=np.float64)
    global_idx = np.arange(n_frames, dtype=np.int64)
    bin_idx = (global_idx * n_bins) // max(n_frames, 1)
    bin_idx = np.minimum(bin_idx, n_bins - 1)

    for c in range(ch):
        col = work[:, c]
        np.minimum.at(bin_min[:, c], bin_idx, col)
        np.maximum.at(bin_max[:, c], bin_idx, col)

    sum_sq = np.sum(work * work, axis=0)
    rms = np.sqrt(sum_sq / max(n_frames, 1))
    peak = np.max(np.abs(work), axis=0)
    peak_linear = float(np.max(peak))
    crest_db = 20.0 * np.log10((peak_linear + 1e-20) / (float(np.max(rms)) + 1e-20))

    return (
        fs,
        n_frames,
        bin_min,
        bin_max,
        rms,
        dtype_kind,
        peak_linear,
        float(crest_db),
        n_clip_pos,
        n_clip_neg,
        n_frames * ch,
    )


def y_limits_for_plot(data_dtype: np.dtype, data_sample: np.ndarray) -> Tuple[float, float]:
    if np.issubdtype(data_dtype, np.integer):
        return _integer_clip_limits(data_sample)
    return _float_clip_limits()


def plot_audio_pdf(
    out_path: Path,
    title_name: str,
    fs: int,
    n_frames: int,
    bin_min: np.ndarray,
    bin_max: np.ndarray,
    rms: np.ndarray,
    dtype_label: str,
    peak_linear: float,
    crest_db: float,
    n_clip_pos: int,
    n_clip_neg: int,
    n_total_samples: int,
    y_lim: Tuple[float, float],
    dpi: float,
    fig_width: float,
    fig_height_per_channel: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    from matplotlib.collections import LineCollection

    duration = n_frames / max(fs, 1)
    n_bins, n_ch = bin_min.shape
    t_edges = np.linspace(0.0, duration, n_bins + 1)
    t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])

    n_rows = max(1, n_ch)
    fig_h = max(3.5, fig_height_per_channel * n_rows + 1.2)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(fig_width, fig_h),
        sharex=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()

    def x_formatter(x, _pos):
        return _format_hms(float(x))

    major_step, minor_step = _time_tick_step(duration)

    clip_pct = 100.0 * (n_clip_pos + n_clip_neg) / max(n_total_samples, 1)

    for c in range(n_rows):
        ax = axes_flat[c]
        lo = bin_min[:, c] if c < n_ch else bin_min[:, 0]
        hi = bin_max[:, c] if c < n_ch else bin_max[:, 0]
        finite = np.isfinite(lo) & np.isfinite(hi)
        t_c = t_centers[finite]
        lo = lo[finite]
        hi = hi[finite]

        nseg = len(t_c)
        segments = np.empty((nseg, 2, 2), dtype=np.float64)
        segments[:, 0, 0] = t_c
        segments[:, 0, 1] = lo
        segments[:, 1, 0] = t_c
        segments[:, 1, 1] = hi
        lc = LineCollection(
            segments,
            colors="#1f4e79",
            linewidths=0.35,
            alpha=0.92,
        )
        ax.add_collection(lc)
        ax.axhline(0.0, color="#888888", linewidth=0.6, linestyle="--", zorder=0)
        for clip_y in (y_lim[0], y_lim[1]):
            ax.axhline(clip_y, color="#c00000", linewidth=0.9, linestyle=":", zorder=0, alpha=0.85)

        ax.set_ylim(y_lim[0], y_lim[1])
        ax.set_xlim(0.0, duration)
        ax.set_ylabel(f"Amplitude\n(ch {c + 1})" if n_ch > 1 else "Amplitude", fontsize=10)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=9, symmetric=True))
        ax.grid(True, which="major", axis="y", linestyle="-", linewidth=0.4, alpha=0.35)
        ax.grid(True, which="major", axis="x", linestyle="-", linewidth=0.35, alpha=0.25)

        ch_rms = float(rms[c]) if c < len(rms) else float(rms[0])
        ax.text(
            0.01,
            0.98,
            f"RMS ≈ {ch_rms:.6g}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.9),
        )

    ax_bot = axes_flat[-1]
    ax_bot.set_xlabel("Time (full file, hh:mm:ss.ms)", fontsize=11, labelpad=6)
    for ax in axes_flat:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(major_step))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(minor_step))
        ax.grid(True, which="minor", axis="x", linestyle=":", linewidth=0.25, alpha=0.18)
    ax_bot.xaxis.set_major_formatter(ticker.FuncFormatter(x_formatter))

    info_lines = [
        f"File: {title_name}",
        f"Sample rate: {fs} Hz  |  Frames: {n_frames:,}  |  Duration: {_format_hms(duration)} ({duration:.6f} s)",
        f"Channels: {n_ch}  |  Storage dtype: {dtype_label}",
        f"Peak |x|: {peak_linear:.8g}  |  Crest (max ch): {crest_db:.2f} dB",
        f"Clip-level hits (all ch): +clip={n_clip_pos:,}, −clip={n_clip_neg:,}  ({clip_pct:.4f}% of samples)",
        f"Y-limits fixed to full-scale for clipping check: [{y_lim[0]:.6g}, {y_lim[1]:.6g}]",
        f"Waveform: min–max envelope over {n_bins:,} time bins (true peaks preserved; no seconds omitted on axis).",
    ]
    fig.suptitle("\n".join(info_lines), fontsize=9, family="monospace", ha="left", x=0.02, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", dpi=dpi, bbox_inches="tight", metadata={"Creator": "plot_all_audio_waveforms.py"})
    plt.close(fig)


def process_file(
    path: Path,
    output_dir: Path,
    n_bins: int,
    chunk_frames: int,
    dpi: float,
    fig_width: float,
    fig_height_per_channel: float,
    force_librosa: bool,
    max_wav_ram_mb: float,
) -> Optional[str]:
    ext = path.suffix.lower()
    err: Optional[str] = None

    try:
        use_streaming = False
        if ext == ".wav" and not force_librosa and path.exists():
            mb = path.stat().st_size / (1024 * 1024)
            use_streaming = mb > max_wav_ram_mb

        scipy_ok = False
        if ext == ".wav" and not force_librosa and not use_streaming:
            try:
                fs, data = read_audio_scipy_wav(path)
                scipy_ok = int(data.size) > 0
            except Exception:
                scipy_ok = False

        if ext == ".wav" and not force_librosa and not use_streaming and scipy_ok:
            (
                fs,
                n_frames,
                bin_min,
                bin_max,
                rms,
                dtype_kind,
                peak_linear,
                crest_db,
                n_clip_pos,
                n_clip_neg,
                n_total,
            ) = process_numpy_in_memory(fs, data, n_bins)
            y_lim = y_limits_for_plot(data.dtype, data)
        else:
            used_sf = False
            try:
                (
                    fs,
                    n_frames,
                    bin_min,
                    bin_max,
                    rms,
                    dtype_kind,
                    peak_linear,
                    crest_db,
                    n_clip_pos,
                    n_clip_neg,
                    n_total,
                ) = process_with_soundfile(path, n_bins, chunk_frames)
                used_sf = True
            except Exception:
                used_sf = False
            if used_sf:
                y_lim = _float_clip_limits()
            else:
                fs, data = read_audio_librosa(path)
                if data.size == 0:
                    return f"{path}: empty audio"
                (
                    fs,
                    n_frames,
                    bin_min,
                    bin_max,
                    rms,
                    dtype_kind,
                    peak_linear,
                    crest_db,
                    n_clip_pos,
                    n_clip_neg,
                    n_total,
                ) = process_numpy_in_memory(fs, data, n_bins)
                y_lim = y_limits_for_plot(data.dtype, data)

        out_pdf = _safe_pdf_name(path, output_dir)
        plot_audio_pdf(
            out_pdf,
            path.name,
            fs,
            n_frames,
            bin_min,
            bin_max,
            rms,
            dtype_kind,
            peak_linear,
            crest_db,
            n_clip_pos,
            n_clip_neg,
            n_total,
            y_lim,
            dpi=dpi,
            fig_width=fig_width,
            fig_height_per_channel=fig_height_per_channel,
        )
    except Exception as e:
        err = f"{path}: {e}"
    return err


def main() -> int:
    root = Path(__file__).resolve().parent
    default_das = os.environ.get("DAS_AUDIO_DIR", str(root / "Data/Das/4004271/stimuli/stimuli"))
    default_ful = os.environ.get("FULSANG_AUDIO_DIR", str(root / "Data/Fulsang/AUDIO"))

    p = argparse.ArgumentParser(description="Plot each audio file to its own PDF (full duration, clip-aware y-axis).")
    p.add_argument("--output-dir", type=str, default=str(root / "audio_waveform_pdfs"))
    p.add_argument("--das-audio-dir", type=str, default=default_das, help="DAS stimuli directory (optional)")
    p.add_argument("--fulsang-audio-dir", type=str, default=default_ful, help="Fulsang AUDIO directory (optional)")
    p.add_argument(
        "--audio-dir",
        type=str,
        default="",
        help="If set, only this directory is used (use with --recursive for trees).",
    )
    p.add_argument("--recursive", action="store_true", help="Search subdirectories for audio files")
    p.add_argument("--skip-das", action="store_true")
    p.add_argument("--skip-fulsang", action="store_true")
    p.add_argument("--dpi", type=float, default=300.0)
    p.add_argument("--fig-width", type=float, default=14.0)
    p.add_argument("--fig-height-per-channel", type=float, default=2.8)
    p.add_argument(
        "--n-bins",
        type=int,
        default=0,
        help="Horizontal resolution (min–max bins). 0 = auto from dpi and fig-width.",
    )
    p.add_argument("--chunk-frames", type=int, default=1_048_576, help="Frames per read when using soundfile")
    p.add_argument("--force-librosa-wav", action="store_true", help="Do not use scipy for .wav (librosa/soundfile only)")
    p.add_argument(
        "--max-wav-ram-mb",
        type=float,
        default=256.0,
        help="If a .wav is larger than this, stream with soundfile instead of scipy (full file still plotted).",
    )
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    n_bins = int(args.n_bins)
    if n_bins <= 0:
        n_bins = max(4000, int(args.dpi * args.fig_width * 1.2))

    files: List[Path] = []
    if args.audio_dir:
        files.extend(_collect_files(Path(args.audio_dir), recursive=args.recursive))
    else:
        if not args.skip_das:
            files.extend(_collect_files(Path(args.das_audio_dir), recursive=args.recursive))
        if not args.skip_fulsang:
            files.extend(_collect_files(Path(args.fulsang_audio_dir), recursive=args.recursive))

    files = sorted(set(files))
    if not files:
        print("No audio files found. Set --audio-dir or check DAS/Fulsang paths.", file=sys.stderr)
        return 1

    print(f"Output directory: {output_dir.resolve()}")
    print(f"Files to plot: {len(files)}  |  n_bins={n_bins:,}")

    errors: List[str] = []
    for i, fpath in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {fpath}")
        err = process_file(
            fpath,
            output_dir,
            n_bins=n_bins,
            chunk_frames=args.chunk_frames,
            dpi=args.dpi,
            fig_width=args.fig_width,
            fig_height_per_channel=args.fig_height_per_channel,
            force_librosa=args.force_librosa_wav,
            max_wav_ram_mb=args.max_wav_ram_mb,
        )
        if err:
            print(f"  ERROR: {err}", file=sys.stderr)
            errors.append(err)

    if errors:
        print(f"\nCompleted with {len(errors)} error(s).", file=sys.stderr)
        return 2
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
