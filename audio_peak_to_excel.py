#!/usr/bin/env python3
"""
Compute absolute peak for every audio file in DAS and Fulsang datasets
and write results to an Excel file.

Usage:
  python audio_peak_to_excel.py
  python audio_peak_to_excel.py --das_audio_dir "Data/Das/4004271/stimuli/stimuli" --fulsang_audio_dir "Data/Fulsang/AUDIO" --output audio_peaks.xlsx
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional: openpyxl for .xlsx export (pip install openpyxl)
try:
    import openpyxl  # noqa: F401
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False


def get_audio_files(directory: Path, extensions: tuple = ('.wav', '.WAV', '.mp3', '.MP3')) -> List[Path]:
    """Collect all audio files in directory (non-recursive by default)."""
    if not directory.exists():
        return []
    files = []
    for ext in extensions:
        files.extend(directory.glob(f'*{ext}'))
    return sorted(set(files))


def read_audio_wav(filepath: Path):
    """Read WAV with scipy; return (sample_rate, data). Data may be int or float."""
    from scipy.io import wavfile
    fs, data = wavfile.read(str(filepath))
    return fs, data


def read_audio_other(filepath: Path):
    """Read MP3 or other format via soundfile or librosa if available."""
    try:
        import soundfile as sf
        data, fs = sf.read(str(filepath), dtype='float64')
        return fs, data
    except ImportError:
        pass
    try:
        import librosa
        data, fs = librosa.load(str(filepath), sr=None, mono=False)
        if data.ndim == 2:
            data = data.T  # (samples, channels)
        return fs, data
    except ImportError:
        pass
    return None, None


def compute_absolute_peak(filepath: Path) -> Tuple[Optional[float], Optional[int], Optional[str]]:
    """
    Read audio file and return (absolute_peak, sample_rate, error_message).
    Absolute peak = max(abs(samples)) over all channels.
    """
    ext = filepath.suffix.lower()
    fs, data = None, None

    if ext == '.wav':
        try:
            fs, data = read_audio_wav(filepath)
        except Exception as e:
            return None, None, str(e)
    else:
        fs, data = read_audio_other(filepath)
        if data is None:
            return None, None, "MP3/other format requires soundfile or librosa (pip install soundfile or librosa)"

    if data is None or data.size == 0:
        return None, fs, "Empty or unreadable file"

    # Flatten (mono or stereo) and take absolute peak
    if data.ndim > 1:
        data = data.reshape(-1)
    peak = float(np.max(np.abs(data)))
    return peak, fs, None


def main():
    parser = argparse.ArgumentParser(
        description='Compute absolute peak for DAS and Fulsang audio files and export to Excel.'
    )
    parser.add_argument(
        '--das_audio_dir',
        type=str,
        default='Data/Das/4004271/stimuli/stimuli',
        help='Directory containing DAS audio files',
    )
    parser.add_argument(
        '--fulsang_audio_dir',
        type=str,
        default='Data/Fulsang/AUDIO',
        help='Directory containing Fulsang audio files',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='audio_absolute_peaks.xlsx',
        help='Output Excel file path',
    )
    parser.add_argument(
        '--no_excel',
        action='store_true',
        help='If set, write CSV instead of Excel (no openpyxl needed)',
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    das_dir = root / args.das_audio_dir
    fulsang_dir = root / args.fulsang_audio_dir

    rows = []

    # DAS
    das_files = get_audio_files(das_dir)
    print(f"DAS audio dir: {das_dir} (exists: {das_dir.exists()})")
    print(f"  Found {len(das_files)} audio file(s).")
    for f in das_files:
        peak, sr, err = compute_absolute_peak(f)
        rows.append({
            'dataset': 'DAS',
            'filename': f.name,
            'filepath': str(f.resolve()),
            'absolute_peak': peak if peak is not None else '',
            'sample_rate': sr if sr is not None else '',
            'error': err or '',
        })

    # Fulsang
    fulsang_files = get_audio_files(fulsang_dir)
    print(f"Fulsang audio dir: {fulsang_dir} (exists: {fulsang_dir.exists()})")
    print(f"  Found {len(fulsang_files)} audio file(s).")
    for f in fulsang_files:
        peak, sr, err = compute_absolute_peak(f)
        rows.append({
            'dataset': 'Fulsang',
            'filename': f.name,
            'filepath': str(f.resolve()),
            'absolute_peak': peak if peak is not None else '',
            'sample_rate': sr if sr is not None else '',
            'error': err or '',
        })

    if not rows:
        print("No audio files found. Check --das_audio_dir and --fulsang_audio_dir.")
        sys.exit(1)

    df = pd.DataFrame(rows)

    out_path = root / args.output
    if args.no_excel or not HAS_OPENPYXL:
        out_path = out_path.with_suffix('.csv')
        df.to_csv(out_path, index=False)
        print(f"Wrote {len(df)} rows to {out_path}")
        if not HAS_OPENPYXL and not args.no_excel:
            print("Install openpyxl for Excel output: pip install openpyxl")
    else:
        df.to_excel(out_path, index=False, engine='openpyxl')
        print(f"Wrote {len(df)} rows to {out_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
