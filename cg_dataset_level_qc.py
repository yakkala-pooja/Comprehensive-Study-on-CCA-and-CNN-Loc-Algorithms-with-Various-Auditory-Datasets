#!/usr/bin/env python3
"""
CG dataset-level QC checks for Das/Fulsang.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.io as sio
from scipy import signal


def _flatten_trials(trials_obj: Any) -> List[Any]:
    if trials_obj is None:
        return []
    if isinstance(trials_obj, np.ndarray):
        return list(trials_obj.flatten())
    return [trials_obj]


def _subject_index(stem: str) -> int:
    s = stem.lower().replace("s", "")
    digits = "".join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else 0


def _das_subject_files(das_dir: Path) -> List[Path]:
    out = [p for p in das_dir.glob("S*.mat") if p.name.lower().endswith(".mat")]
    return sorted(out, key=lambda p: _subject_index(p.stem))


def _fulsang_preproc_files(ful_dir: Path) -> List[Path]:
    out = list(ful_dir.glob("S*_data_preproc.mat"))
    return sorted(out, key=lambda p: _subject_index(p.stem))


def _to_volts(eeg: np.ndarray) -> Tuple[np.ndarray, str]:
    eeg = np.asarray(eeg, dtype=np.float64)
    mx = float(np.nanmax(np.abs(eeg))) if eeg.size else 0.0
    if mx > 1e-2:
        return eeg * 1e-6, "uV->V"
    return eeg, "V"


def _bandpass_eeg(x: np.ndarray, fs_hz: float, low_hz: float, high_hz: float, order: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.size == 0 or not np.isfinite(fs_hz) or fs_hz <= 0:
        return x.copy()
    nyq = fs_hz / 2.0
    hi_eff = min(high_hz, nyq * 0.99)
    lo = float(np.clip(low_hz / nyq, 0.001, 0.98))
    hi = float(np.clip(hi_eff / nyq, lo + 0.001, 0.99))
    b, a = signal.butter(order, [lo, hi], btype="band")
    padlen = 3 * max(len(a), len(b))
    if x.shape[0] <= padlen:
        return x.copy()
    out = np.empty_like(x)
    for c in range(x.shape[1]):
        out[:, c] = signal.filtfilt(b, a, x[:, c])
    return out


def _range_stats(arr: np.ndarray) -> Dict[str, float]:
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    return {
        "min": float(np.nanmin(flat)),
        "max": float(np.nanmax(flat)),
        "abs_max": float(np.nanmax(np.abs(flat))),
        "p001": float(np.nanpercentile(flat, 0.1)),
        "p999": float(np.nanpercentile(flat, 99.9)),
    }


def _stream_range_stats(chunks: List[np.ndarray], sample_per_chunk: int = 20000) -> Dict[str, float]:
    """
    Memory-safe robust stats over many arrays.
    Uses exact min/max/abs_max and percentile estimates from uniform subsamples.
    """
    gmin = float("inf")
    gmax = float("-inf")
    gabs = 0.0
    samples: List[np.ndarray] = []
    rng = np.random.default_rng(123)
    for ch in chunks:
        flat = np.asarray(ch, dtype=np.float64).reshape(-1)
        if flat.size == 0:
            continue
        gmin = min(gmin, float(np.nanmin(flat)))
        gmax = max(gmax, float(np.nanmax(flat)))
        gabs = max(gabs, float(np.nanmax(np.abs(flat))))
        if flat.size <= sample_per_chunk:
            samples.append(flat)
        else:
            idx = rng.choice(flat.size, size=sample_per_chunk, replace=False)
            samples.append(flat[idx])
    if not samples:
        return {"min": np.nan, "max": np.nan, "abs_max": np.nan, "p001": np.nan, "p999": np.nan}  # type: ignore[return-value]
    s = np.concatenate(samples, axis=0)
    return {
        "min": gmin,
        "max": gmax,
        "abs_max": gabs,
        "p001": float(np.nanpercentile(s, 0.1)),
        "p999": float(np.nanpercentile(s, 99.9)),
    }


def _suggest_ylim_volts(stats: Dict[str, float]) -> Tuple[float, float]:
    # Pure data-driven range (symmetric), no imposed floor.
    lim = max(abs(stats["p001"]), abs(stats["p999"]))
    if not np.isfinite(lim) or lim <= 0:
        lim = max(abs(stats["min"]), abs(stats["max"]))
    if not np.isfinite(lim) or lim <= 0:
        lim = 1.0
    return -lim, lim


def _load_das_eeg_trials(path: Path) -> Tuple[List[np.ndarray], float]:
    mat = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    trials = _flatten_trials(mat.get("trials", []))
    out: List[np.ndarray] = []
    fs = np.nan
    for tr in trials:
        try:
            eeg = np.asarray(tr.RawData.EegData, dtype=np.float64)
            sr = np.asarray(tr.FileHeader.SampleRate).flatten()
            fs = float(sr[0]) if sr.size else fs
            if eeg.ndim == 2:
                out.append(eeg)
        except Exception:
            continue
    return out, fs


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
    out: List[np.ndarray] = []
    for i in range(cell.size):
        t = np.asarray(cell.flat[i], dtype=np.float64)
        if t.ndim == 2 and t.shape[0] < t.shape[1] and t.shape[0] <= 128:
            t = t.T
        elif t.ndim == 1:
            t = t.reshape(-1, 1)
        out.append(t)
    return out


def _load_fulsang_trials(path: Path) -> Tuple[List[np.ndarray], float]:
    mat = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    d = mat.get("data")
    if d is None:
        return [], np.nan
    first = d.flat[0] if isinstance(d, np.ndarray) else d
    return _eeg_trial_list(first), _get_fulsang_fsample(first)


def _dataset_audio_files(dataset: str, das_stimuli: Path, ful_audio: Path) -> List[Path]:
    return sorted(das_stimuli.glob("*.wav")) if dataset == "das" else sorted(ful_audio.glob("*.wav"))


def _wav_float(path: Path) -> Tuple[np.ndarray, int]:
    from scipy.io import wavfile

    sr, x = wavfile.read(str(path))
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        x = np.asarray(x, dtype=np.float64) / max(abs(info.min), abs(info.max))
    else:
        x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    return x, int(sr)


def _audio_loudness(files: List[Path], max_files: int = 100) -> Dict[str, Any]:
    picks = files[:max_files]
    lufs_vals: List[float] = []
    dbfs_rms: List[float] = []
    used_lufs = False
    meter = None
    try:
        import pyloudnorm as pyln  # type: ignore

        used_lufs = True
    except Exception:
        pyln = None  # type: ignore

    for pth in picks:
        try:
            x, sr = _wav_float(pth)
            if x.size < sr:
                continue
            rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)
            dbfs_rms.append(20.0 * np.log10(rms))
            if used_lufs:
                if meter is None or getattr(meter, "rate", None) != sr:
                    meter = pyln.Meter(sr)
                loud = float(meter.integrated_loudness(x.astype(np.float64)))
                if np.isfinite(loud):
                    lufs_vals.append(loud)
        except Exception:
            continue

    out: Dict[str, Any] = {
        "n_files_scanned": len(picks),
        "n_valid": len(dbfs_rms),
        "metric": "LUFS" if used_lufs and len(lufs_vals) > 0 else "RMS_dBFS",
    }
    arr = np.asarray(lufs_vals if (used_lufs and len(lufs_vals) > 0) else dbfs_rms, dtype=np.float64)
    if arr.size:
        out.update(
            {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        )
    return out


def _mwf_check(repo: Path) -> Dict[str, Any]:
    das_dir = repo / "MWF_cleaned_DAS"
    ful_dir = repo / "MWF_cleaned_Fuglsang"
    das_files = sorted(das_dir.glob("S*_MWF.mat")) if das_dir.exists() else []
    ful_files = sorted(ful_dir.glob("sub*_MWF.mat")) if ful_dir.exists() else []
    out: Dict[str, Any] = {
        "das_mwf_dir": str(das_dir),
        "fulsang_mwf_dir": str(ful_dir),
        "das_mwf_files": len(das_files),
        "fulsang_mwf_files": len(ful_files),
    }
    sample = das_files[:1] + ful_files[:1]
    if sample:
        pth = sample[0]
        try:
            m = sio.loadmat(str(pth), squeeze_me=True, struct_as_record=False)
            out["sample_file"] = str(pth)
            out["sample_keys"] = [k for k in m.keys() if not k.startswith("__")]
            out["seems_working"] = True
        except Exception as e:
            out["sample_file"] = str(pth)
            out["seems_working"] = False
            out["error"] = str(e)
    else:
        out["seems_working"] = False
        out["error"] = "No MWF output files found"
    return out


def main() -> int:
    repo = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="CG QC checks for Das/Fulsang audio + EEG + MWF")
    p.add_argument("--das-dir", default=str(repo / "Data" / "Das" / "4004271"))
    p.add_argument("--das-stimuli-dir", default=str(repo / "Data" / "Das" / "4004271" / "stimuli" / "stimuli"))
    p.add_argument("--ful-preproc-dir", default=str(repo / "Data" / "Fulsang" / "DATA_preproc"))
    p.add_argument("--ful-audio-dir", default=str(repo / "Data" / "Fulsang" / "AUDIO"))
    p.add_argument("--target-fs", type=float, default=64.0)
    p.add_argument("--bp-low", type=float, default=0.5)
    p.add_argument("--bp-high", type=float, default=40.0)
    p.add_argument("--bp-order", type=int, default=4)
    p.add_argument("--out-json", default=str(repo / "cg_dataset_level_qc_report.json"))
    args = p.parse_args()

    das_dir = Path(args.das_dir)
    das_stimuli = Path(args.das_stimuli_dir)
    ful_dir = Path(args.ful_preproc_dir)
    ful_audio = Path(args.ful_audio_dir)

    report: Dict[str, Any] = {
        "professor_rules": {
            "audio": "single gain coefficient per dataset",
            "eeg": "store in volts, common sampling rate",
        },
        "settings": {
            "target_fs_hz": args.target_fs,
            "bandpass_hz": [args.bp_low, args.bp_high],
            "bandpass_order": args.bp_order,
        },
    }

    for ds_name in ("das", "fulsang"):
        files = _das_subject_files(das_dir) if ds_name == "das" else _fulsang_preproc_files(ful_dir)
        all_raw_v: List[np.ndarray] = []
        all_bp_v: List[np.ndarray] = []
        trial_durations: List[float] = []
        fs_values: List[float] = []
        unit_tags: List[str] = []

        for f in files:
            trials, fs = _load_das_eeg_trials(f) if ds_name == "das" else _load_fulsang_trials(f)
            if np.isfinite(fs):
                fs_values.append(float(fs))
            for tr in trials:
                v, tag = _to_volts(tr)
                unit_tags.append(tag)
                all_raw_v.append(v)
                if np.isfinite(fs) and fs > 0:
                    all_bp_v.append(_bandpass_eeg(v, fs, args.bp_low, args.bp_high, args.bp_order))
                    trial_durations.append(v.shape[0] / float(fs))

        if all_raw_v:
            raw_stats = _stream_range_stats(all_raw_v)
            bp_stats = _stream_range_stats(all_bp_v if all_bp_v else all_raw_v)
            report[f"eeg_{ds_name}"] = {
                "n_subject_files": len(files),
                "n_trials_total": len(all_raw_v),
                "fs_hz_unique": sorted({round(x, 6) for x in fs_values}),
                "unit_conversion_counts": {k: unit_tags.count(k) for k in sorted(set(unit_tags))},
                "duration_sec": {
                    "mean": float(np.mean(trial_durations)) if trial_durations else None,
                    "min": float(np.min(trial_durations)) if trial_durations else None,
                    "max": float(np.max(trial_durations)) if trial_durations else None,
                },
                "raw_volts_stats": raw_stats,
                "bandpass_volts_stats": bp_stats,
                "suggested_ylim_raw_volts": _suggest_ylim_volts(raw_stats),
                "suggested_ylim_bandpass_volts": _suggest_ylim_volts(bp_stats),
            }
        else:
            report[f"eeg_{ds_name}"] = {"error": "No trials loaded"}

    aud: Dict[str, Any] = {}
    aud["das"] = _audio_loudness(_dataset_audio_files("das", das_stimuli, ful_audio))
    aud["fulsang"] = _audio_loudness(_dataset_audio_files("fulsang", das_stimuli, ful_audio))

    if "mean" in aud["das"] and "mean" in aud["fulsang"]:
        target = float((aud["das"]["mean"] + aud["fulsang"]["mean"]) / 2.0)
        g_das = float(10 ** ((target - aud["das"]["mean"]) / 20.0))
        g_ful = float(10 ** ((target - aud["fulsang"]["mean"]) / 20.0))
        aud["normalization_coefficients_suggestion"] = {
            "metric": aud["das"].get("metric", "RMS_dBFS"),
            "target_level": target,
            "das_gain_c": g_das,
            "fulsang_gain_c": g_ful,
        }
    report["audio"] = aud

    report["mwf"] = _mwf_check(repo)

    outp = Path(args.out_json)
    outp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"CG dataset QC complete. Report: {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
