#!/usr/bin/env python3
"""
Create matched DAS/Fulsang EEG pair plots with fixed y-limits in microvolts.

Outputs:
- qc_professor_checks/pair_cg_trial0_ch0_ylim_pm300uV.png
- qc_professor_checks/pair_cg_trial0_ch0_ylim_pm15uV.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import scipy.io as sio


def _flatten_trials(trials_obj: Any) -> List[Any]:
    if trials_obj is None:
        return []
    if isinstance(trials_obj, np.ndarray):
        return list(trials_obj.flatten())
    return [trials_obj]


def _load_das_trial0(repo: Path) -> Tuple[np.ndarray, float]:
    p = repo / "Data" / "Das" / "4004271" / "S1.mat"
    mat = sio.loadmat(str(p), squeeze_me=True, struct_as_record=False)
    trials = _flatten_trials(mat.get("trials"))
    tr = trials[0]
    eeg = np.asarray(tr.RawData.EegData, dtype=np.float64)
    sr = np.asarray(tr.FileHeader.SampleRate).flatten()
    fs = float(sr[0]) if sr.size else 128.0
    return eeg, fs


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
        t = np.asarray(cell.flat[i], dtype=np.float64)
        if t.ndim == 2 and t.shape[0] < t.shape[1] and t.shape[0] <= 128:
            t = t.T
        elif t.ndim == 1:
            t = t.reshape(-1, 1)
        out.append(t)
    return out


def _load_fulsang_trial0(repo: Path) -> Tuple[np.ndarray, float]:
    p = repo / "Data" / "Fulsang" / "DATA_preproc" / "S1_data_preproc.mat"
    mat = sio.loadmat(str(p), squeeze_me=True, struct_as_record=False)
    d = mat["data"]
    first = d.flat[0] if isinstance(d, np.ndarray) else d
    fs = _get_fulsang_fsample(first)
    trials = _eeg_trial_list(first)
    return trials[0], fs


def _to_uv(eeg: np.ndarray) -> np.ndarray:
    # heuristic: if values look like volts already, convert; else assume uV-scale already
    mx = float(np.nanmax(np.abs(eeg))) if eeg.size else 0.0
    if mx < 1e-2:
        return eeg * 1e6
    return eeg


def _plot_pair(das_eeg_uv: np.ndarray, fs_das: float, ful_eeg_uv: np.ndarray, fs_ful: float, ylim_uv: float, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_d = int(min(das_eeg_uv.shape[0], round(50.0 * fs_das)))
    n_f = int(min(ful_eeg_uv.shape[0], round(50.0 * fs_ful)))
    t_d = np.arange(n_d, dtype=np.float64) / fs_das
    t_f = np.arange(n_f, dtype=np.float64) / fs_ful

    fig, ax = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True, sharey=True)
    ax[0].plot(t_d, das_eeg_uv[:n_d, 0], lw=0.5, color="tab:blue")
    ax[0].set_title("DAS S1 trial0 ch0")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Amplitude (uV)")
    ax[0].set_ylim(-ylim_uv, ylim_uv)
    ax[0].grid(alpha=0.25)

    ax[1].plot(t_f, ful_eeg_uv[:n_f, 0], lw=0.5, color="tab:orange")
    ax[1].set_title("Fulsang S1 trial0 ch0")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylim(-ylim_uv, ylim_uv)
    ax[1].grid(alpha=0.25)

    fig.suptitle(f"CG pair view (50 s) | fixed y-limit: +/-{ylim_uv:.0f} uV")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    repo = Path(__file__).resolve().parent
    out_dir = repo / "qc_professor_checks"
    das, fs_d = _load_das_trial0(repo)
    ful, fs_f = _load_fulsang_trial0(repo)
    das_uv = _to_uv(das)
    ful_uv = _to_uv(ful)

    _plot_pair(das_uv, fs_d, ful_uv, fs_f, 300.0, out_dir / "pair_cg_trial0_ch0_ylim_pm300uV.png")
    _plot_pair(das_uv, fs_d, ful_uv, fs_f, 15.0, out_dir / "pair_cg_trial0_ch0_ylim_pm15uV.png")
    print("Saved fixed-range pair plots under qc_professor_checks/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
