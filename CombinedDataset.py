#!/usr/bin/env python3
"""
CombinedDataset - Combined Das and Fulsang Dataset with MWF Filtering

This module creates a combined dataset from:
- Das: Already MWF filtered (from MWF_cleaned_DAS)
- Fulsang: Raw data that needs MWF filtering (applied on-the-fly)

The dataset aligns channels, sampling rates, and preprocessing to make both
datasets compatible for training with FULCCA and FULCNN architectures.
"""

import os
import sys
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
import zipfile
import tempfile
warnings.filterwarnings('ignore')

# Import MWF processing
sys.path.append('.')
try:
    from mwf_artifact_removal import FuglsangDatasetMWF, MultiChannelWienerFilter
except ImportError as e:
    print(f"Warning: Could not import MWF modules: {e}")
    print("MWF filtering for Fulsang will be skipped")


class CombinedDataset:
    """
    Combined dataset class for Das (MWF-cleaned) and Fulsang (raw, needs MWF) data.
    
    This class:
    1. Loads Das MWF-cleaned data (already processed)
    2. Loads Fulsang raw data and applies MWF filtering
    3. Aligns channels (Das: 64, Fulsang: 66 -> use 64 common channels)
    4. Aligns sampling rates (both to 128 Hz)
    5. Creates windows compatible with FULCCA and FULCNN
    """
    
    def __init__(self, 
                 das_data_dir: str = "das_combined_preprocessed",
                 das_preprocessing_type: str = "COMBINED_DAS",
                 das_original_dir: str = "Data/Das/4004271",  # Original Das .mat files (with stimuli info for envelope mapping)
                 das_audio_dir: str = "Data/Das/4004271/stimuli/stimuli",  # Das audio files
                 fulsang_raw_dir: str = "/home/py9363/telluride_decoding/Data/Fulsang/EEG",
                 fulsang_audio_dir: str = "/home/py9363/telluride_decoding/Data/Fulsang/AUDIO",
                 fulsang_expinfo_dir: Optional[str] = "Exp_Info",
                 fulsang_mwf_output_dir: str = "MWF_cleaned_Fuglsang",
                 combined_dataset_dir: str = "combined_dataset",  # Centralized output directory
                 das_mwf_dir: Optional[str] = None,
                 window_size: int = 512,  # samples at 128 Hz = 4 seconds
                 overlap: float = 0.25,
                 target_channels: int = 64,
                 target_sampling_rate: int = 128,
                 run_das_preprocessing_if_missing: bool = True,
                 bandpass_low_hz: Optional[float] = 2.0,
                 bandpass_high_hz: Optional[float] = 8.0,
                 bandpass_order: int = 1,
                 use_hilbert_envelope: bool = True,
                 envelope_normalize: str = 'zscore',
                 balance_envelope_energy: bool = True,
                 use_gammatone_filter: bool = False,
                 include_fulsang: bool = True):
        """
        Initialize combined dataset.

        Args:
            das_data_dir: Directory containing Das preprocessed data (COMBINED_DAS from das_preprocessing_combined.py)
            das_preprocessing_type: Use "COMBINED_DAS" (run: python das_preprocessing_combined.py)
            das_original_dir: Directory containing original Das .mat files (for envelope extraction)
            das_audio_dir: Directory containing Das audio files (for envelope extraction)
            fulsang_raw_dir: Directory containing Fulsang raw EEG data
            fulsang_audio_dir: Directory containing Fulsang audio data
            fulsang_expinfo_dir: Directory containing S*_expinfo.mat (attend left/right). If set, searched first for labels.
            fulsang_mwf_output_dir: Output directory for Fulsang MWF processing (legacy, for backward compatibility)
            combined_dataset_dir: Centralized directory for all processed files (default: "combined_dataset")
            das_mwf_dir: If set, directory containing Das S*_MWF.mat when using das_preprocessing_type MWF
                (e.g. repo-root MWF_cleaned_DAS). Default: combined_dataset/das_mwf
            window_size: Window size in samples
            overlap: Window overlap fraction
            target_channels: Target number of channels (64 for Das compatibility)
            target_sampling_rate: Target sampling rate (128 Hz)
            run_das_preprocessing_if_missing: If True (default), run das_preprocessing_combined when TFRecords are missing and DAS .mat files exist.
            bandpass_low_hz: Low cutoff for Butterworth bandpass (Hz). None to skip.
            bandpass_high_hz: High cutoff for Butterworth bandpass (Hz). None or <= low to skip.
            bandpass_order: Butterworth filter order (default 1, match Fulsang).
            use_hilbert_envelope: If True, use Hilbert envelope for 1-band audio (better for speech-brain). Default True.
            envelope_normalize: 'zscore' = (env-mean)/std (can remove slow structure); 'scale_only' = divide by RMS only (preserves slow structure for CCA). Default 'zscore'.
            balance_envelope_energy: If True, scale right stream so total energy matches left (reduces left/right bias in CCA). Default True.
            include_fulsang: If False, load only Das (no Fulsang MWF or concatenation). Default True.
        """
        self.include_fulsang = bool(include_fulsang)
        self.das_data_dir = Path(das_data_dir)
        self.das_preprocessing_type = das_preprocessing_type.upper()
        self.das_original_dir = Path(das_original_dir)  # Original Das .mat files for envelope extraction
        self.das_audio_dir = Path(das_audio_dir) if das_audio_dir else None  # Das audio files
        # Resolve Fulsang raw dir: support relative paths (e.g. Data/Fulsang) from cwd
        _fulsang_raw = Path(fulsang_raw_dir)
        if not _fulsang_raw.is_absolute():
            _fulsang_raw = (Path.cwd() / _fulsang_raw).resolve()
        self.fulsang_raw_dir = _fulsang_raw
        self.fulsang_audio_dir = Path(fulsang_audio_dir) if fulsang_audio_dir else None
        if fulsang_expinfo_dir:
            p = Path(fulsang_expinfo_dir)
            if not p.is_absolute():
                # Try cwd first (run from repo root), then script dir
                for base in (Path.cwd(), Path(__file__).resolve().parent):
                    candidate = (base / p).resolve()
                    if candidate.exists() and candidate.is_dir():
                        p = candidate
                        break
                else:
                    p = (Path(__file__).resolve().parent / p).resolve()
            self.fulsang_expinfo_dir = p
        else:
            self.fulsang_expinfo_dir = None
        
        # Centralized combined dataset directory
        self.combined_dataset_dir = Path(combined_dataset_dir)
        self.combined_dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories for processed files (Das MWF mats often live in MWF_cleaned_DAS at repo root)
        if das_mwf_dir is not None and str(das_mwf_dir).strip():
            _dmw = Path(str(das_mwf_dir).strip())
            if not _dmw.is_absolute():
                _dmw = (Path.cwd() / _dmw).resolve()
            self.das_mwf_dir = _dmw
        else:
            self.das_mwf_dir = self.combined_dataset_dir / "das_mwf"
            self.das_mwf_dir.mkdir(parents=True, exist_ok=True)
        self.fulsang_mwf_dir = self.combined_dataset_dir / "fulsang_mwf"
        self.fulsang_mwf_dir.mkdir(parents=True, exist_ok=True)
        self.window_size = window_size
        self.overlap = overlap
        self.target_channels = target_channels
        self.target_sampling_rate = target_sampling_rate
        self.run_das_preprocessing_if_missing = run_das_preprocessing_if_missing
        self.bandpass_low_hz = bandpass_low_hz
        self.bandpass_high_hz = bandpass_high_hz
        self.bandpass_order = int(bandpass_order)
        self.use_hilbert_envelope = bool(use_hilbert_envelope)
        self.envelope_normalize = str(envelope_normalize).strip().lower() if envelope_normalize else 'zscore'
        if self.envelope_normalize not in ('zscore', 'scale_only'):
            self.envelope_normalize = 'zscore'
        self.balance_envelope_energy = bool(balance_envelope_energy)
        self.use_gammatone_filter = bool(use_gammatone_filter)
        if das_preprocessing_type.upper() == "COMBINED_DAS" and target_sampling_rate != 64:
            print("  Note: Use target_sampling_rate=64 for combined DAS+Fulsang.")
        
        # Parameters
        self.sampling_rate = target_sampling_rate
        self.n_channels = target_channels
        
        # Load original Das files for envelope extraction (used by both MWF and DASPREPROCESS)
        self.original_das_files = self._load_original_das_files()
        
        # Set up raw EEG directory for Fulsang (for extracting true attention labels)
        # Same logic as FULPRE.py - try to find raw EEG files for true labels
        self.fulsang_eeg_raw_path = None
        self.fulsang_eeg_raw_is_zip = False
        self._fulsang_raw_labels_cache: Dict[str, Optional[np.ndarray]] = {}
        
        # Try to auto-detect Fulsang raw EEG files (multiple conventions)
        eeg_zip = self.fulsang_raw_dir / "EEG.zip" if self.fulsang_raw_dir else None
        eeg_dir = self.fulsang_raw_dir / "EEG" if self.fulsang_raw_dir else None
        if eeg_zip and eeg_zip.exists():
            self.fulsang_eeg_raw_path = eeg_zip
            self.fulsang_eeg_raw_is_zip = True
        elif eeg_dir and eeg_dir.exists():
            self.fulsang_eeg_raw_path = eeg_dir
            self.fulsang_eeg_raw_is_zip = False
        else:
            # Convention: fulsang_raw_dir can be the EEG folder itself (e.g. Data/Fulsang/EEG with S*.mat inside)
            if self.fulsang_raw_dir and self.fulsang_raw_dir.exists() and self.fulsang_raw_dir.is_dir():
                zip_here = self.fulsang_raw_dir / "EEG.zip"
                if zip_here.exists():
                    self.fulsang_eeg_raw_path = zip_here
                    self.fulsang_eeg_raw_is_zip = True
                elif any(self.fulsang_raw_dir.glob("S*.mat")):
                    self.fulsang_eeg_raw_path = self.fulsang_raw_dir
                    self.fulsang_eeg_raw_is_zip = False
        
        # Initialize envelope tracking attributes
        self.left_envelope_stream = None
        self.right_envelope_stream = None
        self._real_envelope_frames = 0
        self._total_frames = 0
        
        print("="*80)
        print("COMBINED DATASET - Das + Fulsang")
        print("="*80)
        print(f"Das preprocessing: {self.das_preprocessing_type}")
        print(f"Combined dataset directory: {self.combined_dataset_dir}")
        print(f"  - Das MWF files: {self.das_mwf_dir}")
        print(f"  - Fulsang MWF files: {self.fulsang_mwf_dir}")
        
        # Report Fulsang raw EEG availability (for label extraction)
        if self.fulsang_eeg_raw_path is not None:
            print(f"  ✓ Fulsang raw EEG files available: {self.fulsang_eeg_raw_path}")
            print(f"    Will extract true attention labels (expinfo.attend_mf) from raw EEG files")
        else:
            print(f"  ⚠ Fulsang raw EEG files not found")
            print(f"    Will fall back to labels from MWF files (may be less accurate)")
            print(f"    Expected: (1) {self.fulsang_raw_dir / 'EEG.zip'} or (2) {self.fulsang_raw_dir / 'EEG'} or (3) {self.fulsang_raw_dir} containing S*.mat")
        repo_root = Path(__file__).resolve().parent
        cwd = Path.cwd()
        expinfo_dirs = [repo_root / "Exp_Info", repo_root / "exp_info", cwd / "Exp_Info", cwd / "exp_info"]
        if self.fulsang_raw_dir is not None:
            expinfo_dirs = [
                self.fulsang_raw_dir, self.fulsang_raw_dir / "EEG", self.fulsang_raw_dir / "eeg",
                self.fulsang_raw_dir / "Exp_Info", self.fulsang_raw_dir / "exp_info"
            ] + expinfo_dirs
            if hasattr(self.fulsang_raw_dir, 'parent') and self.fulsang_raw_dir.parent:
                expinfo_dirs.append(self.fulsang_raw_dir.parent / "Exp_Info")
                expinfo_dirs.append(self.fulsang_raw_dir.parent / "exp_info")
            if self.fulsang_eeg_raw_path is not None and not getattr(self, 'fulsang_eeg_raw_is_zip', True):
                expinfo_dirs.insert(0, self.fulsang_eeg_raw_path)
                expinfo_dirs.insert(1, self.fulsang_eeg_raw_path.parent / "Exp_Info")
                expinfo_dirs.insert(2, self.fulsang_eeg_raw_path.parent / "exp_info")
        seen = set()
        uniq = [str(d) for d in expinfo_dirs if d is not None and str(d) not in seen and not seen.add(str(d))]
        if self.fulsang_expinfo_dir is not None:
            exists = self.fulsang_expinfo_dir.exists() and self.fulsang_expinfo_dir.is_dir()
            print(f"  S*_expinfo.mat sidecar: {self.fulsang_expinfo_dir} (exists: {exists})")
            if exists:
                try:
                    files = list(self.fulsang_expinfo_dir.glob("*.mat"))
                    print(f"    Found {len(files)} .mat files (e.g. {[f.name for f in files[:3]]})")
                except Exception:
                    pass
        print(f"  S*_expinfo.mat sidecar (labels + left/right): looked for in {uniq[:8]}")
        
        # Load DAS data (COMBINED_DAS: from das_preprocessing_combined.py TFRecords; MWF: from S*_MWF.mat)
        das_load_type = self.das_preprocessing_type
        if das_load_type == "MWF":
            mwf_files = sorted(list(self.das_mwf_dir.glob("S*_MWF.mat")))
            if not mwf_files:
                print(f"  No MWF-cleaned Das files in {self.das_mwf_dir}; falling back to COMBINED_DAS (TFRecords).")
                das_load_type = "COMBINED_DAS"
                # If current das_data_dir was for MWF (e.g. MWF_cleaned_DAS), it has no tfrecords; use default TFRecord location
                default_tfrecord_parent = Path("das_combined_preprocessed")
                tfrecord_dir = self.das_data_dir / "tfrecords"
                if not tfrecord_dir.exists() and (default_tfrecord_parent / "tfrecords").exists():
                    self.das_data_dir = default_tfrecord_parent
                    print(f"  Using DAS TFRecords from: {self.das_data_dir / 'tfrecords'}")
                elif not tfrecord_dir.exists():
                    self.das_data_dir = default_tfrecord_parent
                    print(f"  DAS TFRecords will be read from: {self.das_data_dir / 'tfrecords'} (run das_preprocessing_combined.py if missing)")
        print(f"\nLoading DAS data ({das_load_type})...")
        if das_load_type == "MWF":
            das_eeg, das_labels, das_metadata, das_trial_lengths, das_left_envs, das_right_envs = self._load_das_mwf_data()
        elif das_load_type == "DASPREPROCESS":
            das_eeg, das_labels, das_metadata, das_trial_lengths, das_left_envs, das_right_envs = self._load_das_preprocessed_data()
        else:
            # COMBINED_DAS (default or fallback)
            das_eeg, das_labels, das_metadata, das_trial_lengths, das_left_envs, das_right_envs = self._load_das_combined_data()
        
        # Load Fulsang raw data and apply MWF (optional)
        if self.include_fulsang:
            print("\nLoading Fulsang raw data and applying MWF filtering...")
            fulsang_eeg, fulsang_labels, fulsang_metadata, fulsang_trial_lengths, fulsang_left_envs, fulsang_right_envs = self._load_fulsang_and_apply_mwf()
        else:
            print("\nSkipping Fulsang (Das-only; include_fulsang=False).")
            fulsang_eeg = np.zeros((0, das_eeg.shape[1]), dtype=np.float32)
            fulsang_labels = np.array([], dtype=np.int32)
            fulsang_metadata = []
            fulsang_trial_lengths = []
            fulsang_left_envs = []
            fulsang_right_envs = []
        
        # Normalize channel count BEFORE combining
        if fulsang_eeg.shape[0] > 0 and das_eeg.shape[1] != fulsang_eeg.shape[1]:
            print(f"\n⚠️  WARNING: Channel mismatch - Das: {das_eeg.shape[1]}, Fulsang: {fulsang_eeg.shape[1]}")
            print(f"Aligning to {self.target_channels} channels (keeping all Das channels)")
            print(f"⚠️  NOTE: This assumes first {self.target_channels} Fulsang channels correspond to Das channels.")
            print(f"   If channel ordering differs, this will cause domain mismatch and degraded learning.")
            print(f"   Verify channel montage/names match between datasets if possible.")
            
            # Pad Fulsang if needed
            if fulsang_eeg.shape[1] < self.target_channels:
                padding = self.target_channels - fulsang_eeg.shape[1]
                pad_data = np.zeros((fulsang_eeg.shape[0], padding), dtype=fulsang_eeg.dtype)
                fulsang_eeg = np.hstack([fulsang_eeg, pad_data])
            elif fulsang_eeg.shape[1] > self.target_channels:
                # Trim Fulsang to match Das
                fulsang_eeg = fulsang_eeg[:, :self.target_channels]
            
            # Ensure Das has correct channels
            if das_eeg.shape[1] < self.target_channels:
                padding = self.target_channels - das_eeg.shape[1]
                pad_data = np.zeros((das_eeg.shape[0], padding), dtype=das_eeg.dtype)
                das_eeg = np.hstack([das_eeg, pad_data])
            elif das_eeg.shape[1] > self.target_channels:
                das_eeg = das_eeg[:, :self.target_channels]
        
        # Combine datasets. EEG is left as-loaded; per-window normalization happens in the CCA pipeline (_preprocess_window).
        print("\nCombining datasets...")
        if fulsang_eeg.shape[0] > 0:
            self.eeg_data = np.vstack([das_eeg, fulsang_eeg])
            self.labels = np.hstack([das_labels, fulsang_labels])
            self.metadata = das_metadata + fulsang_metadata
        else:
            self.eeg_data = das_eeg
            self.labels = np.asarray(das_labels)
            self.metadata = list(das_metadata)
            print("  Das-only: using Das EEG/labels only (no Fulsang concatenation).")

        # Unified Butterworth bandpass (EEG) so any dataset gets same preprocessing
        self._bandpass_b, self._bandpass_a = None, None
        if (self.bandpass_low_hz is not None and self.bandpass_high_hz is not None
                and self.bandpass_low_hz > 0 and self.bandpass_high_hz > self.bandpass_low_hz):
            fs = float(self.target_sampling_rate)
            nyq = fs / 2.0
            low = np.clip(self.bandpass_low_hz / nyq, 0.001, 0.99)
            high = np.clip(self.bandpass_high_hz / nyq, low + 0.001, 0.99)
            self._bandpass_b, self._bandpass_a = butter(self.bandpass_order, [low, high], btype='band')
            for ch in range(self.eeg_data.shape[1]):
                self.eeg_data[:, ch] = filtfilt(self._bandpass_b, self._bandpass_a, self.eeg_data[:, ch].astype(np.float64), axis=0).astype(np.float32)
            print(f"  Applied Butterworth bandpass {self.bandpass_low_hz}-{self.bandpass_high_hz} Hz (order {self.bandpass_order}) to EEG.")
        
        # Combine envelope streams (1-band Fulsang-style for both DAS and Fulsang)
        def ensure_2d_nbands(env_list, n_bands: int = 1):
            """Ensure all envelopes are 2D (samples x n_bands) for proper vstack. Both DAS and Fulsang use 1-band."""
            result = []
            for env in env_list:
                if env is None:
                    continue
                env = np.asarray(env, dtype=np.float32)
                if len(env.shape) == 1:
                    env = env.reshape(-1, 1)
                elif len(env.shape) > 2:
                    env = env.reshape(env.shape[0], -1)
                if env.shape[1] == 1 and n_bands > 1:
                    env = np.tile(env, (1, n_bands))
                elif env.shape[1] > n_bands:
                    env = env[:, :n_bands]
                elif env.shape[1] < n_bands:
                    pad = np.zeros((env.shape[0], n_bands - env.shape[1]), dtype=np.float32)
                    env = np.hstack([env, pad])
                result.append(env)
            return result
        
        n_bands = self.ENVELOPE_BANDS
        das_left_envs_2d = ensure_2d_nbands(das_left_envs, n_bands) if das_left_envs else []
        das_right_envs_2d = ensure_2d_nbands(das_right_envs, n_bands) if das_right_envs else []
        fulsang_left_envs_2d = ensure_2d_nbands(fulsang_left_envs, n_bands) if fulsang_left_envs else []
        fulsang_right_envs_2d = ensure_2d_nbands(fulsang_right_envs, n_bands) if fulsang_right_envs else []
        
        # Envelope streams: normalize per dataset. 'scale_only' preserves slow structure (better for CCA); 'zscore' removes mean.
        def _norm(x: np.ndarray) -> np.ndarray:
            if getattr(self, 'envelope_normalize', 'zscore') == 'scale_only':
                rms = np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=0, keepdims=True)) + 1e-8
                return (x / rms).astype(np.float32)
            mean = np.mean(x, axis=0, keepdims=True)
            std = np.std(x, axis=0, keepdims=True) + 1e-8
            return ((x - mean) / std).astype(np.float32)

        def _stack_and_standardize_per_dataset(left_das, right_das, left_ful, right_ful):
            left_parts, right_parts = [], []
            if left_das:
                left_parts.append(_norm(np.vstack(left_das).astype(np.float32)))
            if left_ful:
                left_parts.append(_norm(np.vstack(left_ful).astype(np.float32)))
            if right_das:
                right_parts.append(_norm(np.vstack(right_das).astype(np.float32)))
            if right_ful:
                right_parts.append(_norm(np.vstack(right_ful).astype(np.float32)))
            left_stream = np.vstack(left_parts) if left_parts else None
            right_stream = np.vstack(right_parts) if right_parts else None
            return left_stream, right_stream
        left_env_stream, right_env_stream = _stack_and_standardize_per_dataset(
            das_left_envs_2d, das_right_envs_2d, fulsang_left_envs_2d, fulsang_right_envs_2d)

        # Balance left/right envelope energy so CCA is not biased toward the louder stream
        if (left_env_stream is not None and right_env_stream is not None
                and getattr(self, 'balance_envelope_energy', True)):
            left_energy = np.sum(left_env_stream.astype(np.float64) ** 2) + 1e-12
            right_energy = np.sum(right_env_stream.astype(np.float64) ** 2) + 1e-12
            scale = np.sqrt(left_energy / right_energy)
            right_env_stream = (right_env_stream * scale).astype(np.float32)

        if left_env_stream is not None and right_env_stream is not None:
            self.left_envelope_stream = left_env_stream.astype(np.float32)
            self.right_envelope_stream = right_env_stream.astype(np.float32)
            self.envelope_bands = self.left_envelope_stream.shape[1]
            self._total_frames = self.left_envelope_stream.shape[0]
            if (self._bandpass_b is not None and self._bandpass_a is not None
                    and not getattr(self, '_gammatone_in_use', False)):
                for b in range(self.left_envelope_stream.shape[1]):
                    self.left_envelope_stream[:, b] = filtfilt(
                        self._bandpass_b, self._bandpass_a,
                        self.left_envelope_stream[:, b].astype(np.float64), axis=0).astype(np.float32)
                    self.right_envelope_stream[:, b] = filtfilt(
                        self._bandpass_b, self._bandpass_a,
                        self.right_envelope_stream[:, b].astype(np.float64), axis=0).astype(np.float32)
                print(f"  Applied same Butterworth bandpass to envelope streams.")
            elif self._bandpass_b is not None and self._bandpass_a is not None and getattr(self, '_gammatone_in_use', False):
                print("  Skipping envelope Butterworth bandpass because gammatone envelope was used.")
            if getattr(self, 'envelope_normalize', 'zscore') == 'scale_only':
                print(f"  Envelope: scale_only (RMS), no mean subtraction (preserves slow structure for CCA).")
            if getattr(self, 'balance_envelope_energy', True):
                print(f"  Envelope: left/right energy balanced.")

            # Diagnostic: Check envelope statistics
            left_nonzero = np.count_nonzero(self.left_envelope_stream)
            right_nonzero = np.count_nonzero(self.right_envelope_stream)
            left_mean = np.mean(np.abs(self.left_envelope_stream))
            right_mean = np.mean(np.abs(self.right_envelope_stream))
            left_energy = np.sum(self.left_envelope_stream ** 2)
            right_energy = np.sum(self.right_envelope_stream ** 2)
            
            print(f"\nEnvelope Statistics:")
            print(f"  Left envelope - Non-zero samples: {left_nonzero}/{self._total_frames} ({100*left_nonzero/self._total_frames:.1f}%)")
            print(f"  Left envelope - Mean abs: {left_mean:.6f}, Energy: {left_energy:.6f}")
            print(f"  Right envelope - Non-zero samples: {right_nonzero}/{self._total_frames} ({100*right_nonzero/self._total_frames:.1f}%)")
            print(f"  Right envelope - Mean abs: {right_mean:.6f}, Energy: {right_energy:.6f}")
            
            coverage = (self._real_envelope_frames / max(1, self._total_frames)) * 100.0
            print(f"\nEnvelope coverage: {coverage:.2f}% of samples use real stimulus envelopes.")
            if coverage < 1.0:
                print(f"⚠️  WARNING: Only {coverage:.2f}% of samples have real envelopes!")
                print(f"   This will severely impact CCA performance.")
                print(f"   Checking envelope extraction...")
                # Diagnostic: Check a few samples
                if das_left_envs_2d:
                    das_left_nonzero = sum(1 for env in das_left_envs_2d[:min(10, len(das_left_envs_2d))] if np.any(env != 0))
                    das_left_total = min(10, len(das_left_envs_2d))
                    print(f"   Das left envelopes - Non-zero: {das_left_nonzero}/{das_left_total}")
                if fulsang_left_envs_2d:
                    fulsang_left_nonzero = sum(1 for env in fulsang_left_envs_2d[:min(10, len(fulsang_left_envs_2d))] if np.any(env != 0))
                    fulsang_left_total = min(10, len(fulsang_left_envs_2d))
                    print(f"   Fulsang left envelopes - Non-zero: {fulsang_left_nonzero}/{fulsang_left_total}")
        else:
            self.envelope_bands = n_bands
            self.left_envelope_stream = np.zeros((len(self.eeg_data), n_bands), dtype=np.float32)
            self.right_envelope_stream = np.zeros((len(self.eeg_data), n_bands), dtype=np.float32)
            self._total_frames = len(self.eeg_data)
            print(f"\n⚠️  WARNING: No envelope streams created - using zero envelopes!")
            print(f"   This will cause CCA to fail. Check envelope extraction in data files.")
        
        self.n_channels = self.target_channels
        
        # Track trial boundaries for label mapping and grouping info for splitting
        self.trial_boundaries = []
        self.trial_labels = []
        self.trial_meta = []  # Store metadata for each trial (for group-based splitting)
        current_idx = 0
        trial_idx = 0
        
        # Track Das trial boundaries
        for label, trial_length in zip(das_labels, das_trial_lengths):
            self.trial_boundaries.append((current_idx, current_idx + trial_length))
            self.trial_labels.append(label)
            # Store metadata for this trial (for group-based splitting)
            if trial_idx < len(das_metadata):
                self.trial_meta.append(das_metadata[trial_idx])
            else:
                self.trial_meta.append({'subject_id': 'unknown', 'trial_idx': trial_idx, 'dataset': 'Das'})
            current_idx += trial_length
            trial_idx += 1
        
        # Track Fulsang trial boundaries
        if fulsang_eeg.shape[0] > 0:
            fulsang_trial_idx = 0
            for label, trial_length in zip(fulsang_labels, fulsang_trial_lengths):
                self.trial_boundaries.append((current_idx, current_idx + trial_length))
                self.trial_labels.append(label)
                # Store metadata for this trial (for group-based splitting)
                if fulsang_trial_idx < len(fulsang_metadata):
                    self.trial_meta.append(fulsang_metadata[fulsang_trial_idx])
                else:
                    self.trial_meta.append({'subject_id': 'unknown', 'trial_idx': fulsang_trial_idx, 'dataset': 'Fulsang'})
                current_idx += trial_length
                fulsang_trial_idx += 1
        
        print(f"\n✓ Combined dataset loaded:")
        print(f"  Total samples: {len(self.eeg_data)}")
        print(f"  EEG shape: {self.eeg_data.shape}")
        print(f"  Channels: {self.n_channels}")
        print(f"  Sampling rate: {self.sampling_rate} Hz")
        print(f"  Label distribution: {np.bincount(self.labels)}")
        print(f"  Das trials: {len(das_labels)}")
        print(f"  Fulsang trials: {len(fulsang_labels)}")
    
    def _ensure_column_vector(self, array: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if array is None:
            return None
        arr = np.asarray(array).astype(np.float32)
        if arr.ndim == 0:
            return None
        if arr.ndim > 1:
            # Average across feature dimension if needed
            arr = np.mean(arr, axis=1)
        arr = arr.reshape(-1, 1)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    
    def _compute_signal_envelope(self, signal: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if signal is None:
            return None
        arr = self._ensure_column_vector(signal)
        if arr is None:
            return None
        envelope = np.abs(arr)
        if envelope.shape[0] > 9:
            kernel = np.ones((9, 1), dtype=np.float32) / 9.0
            envelope = np.convolve(envelope.flatten(), kernel.flatten(), mode='same').reshape(-1, 1)
        return envelope.astype(np.float32)
    
    def _align_envelope_length(self, envelope: Optional[np.ndarray], target_len: int) -> Optional[np.ndarray]:
        if envelope is None:
            return None
        if envelope.shape[0] == target_len:
            return envelope.astype(np.float32)
        if envelope.shape[0] <= 1:
            return np.full((target_len, 1), float(envelope.squeeze() if envelope.size else 0.0), dtype=np.float32)
        src = envelope.flatten()
        src_idx = np.linspace(0.0, 1.0, num=src.shape[0])
        dst_idx = np.linspace(0.0, 1.0, num=target_len)
        aligned = np.interp(dst_idx, src_idx, src).astype(np.float32)
        return aligned.reshape(-1, 1)
    
    def _load_original_das_files(self) -> Dict[str, Dict]:
        """Load original Das .mat files for envelope extraction.
        
        These files contain stimuli information needed to map trials to audio files.
        Returns a dictionary mapping subject_id -> loaded .mat file data.
        """
        original_das_files = {}
        if self.das_original_dir.exists():
            for orig_file in self.das_original_dir.glob("S*.mat"):
                subject_id = orig_file.stem
                if subject_id not in original_das_files:
                    try:
                        orig_data = sio.loadmat(str(orig_file), squeeze_me=True, struct_as_record=False)
                        original_das_files[subject_id] = orig_data
                    except Exception as e:
                        # Silently skip files that can't be loaded
                        pass
        return original_das_files
    
    def _fallback_envelopes(self, length: int, label: int) -> Tuple[np.ndarray, np.ndarray]:
        ramp = np.linspace(0.0, 1.0, num=length, dtype=np.float32).reshape(-1, 1)
        zeros = np.zeros_like(ramp)
        if label == 0:
            return ramp, zeros
        return zeros, ramp
    
    def _extract_das_envelopes_from_original(self, original_trials, trial_idx: int, target_length: int, subject_id: str, use_4band: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract envelopes from original Das file or audio files. Combined pipeline uses 1-band (Fulsang-style); use_4band=True for 4-band."""
        if original_trials is None or trial_idx >= len(original_trials):
            return None, None
        
        try:
            original_trial = original_trials[trial_idx]
            
            # Try to get stimuli from original trial (this is the key to mapping to audio files)
            # Das files typically have stimuli in trial.stimuli or trial.RawData.stimuli
            stimuli = None
            if hasattr(original_trial, 'stimuli'):
                stimuli = original_trial.stimuli
            elif isinstance(original_trial, dict) and 'stimuli' in original_trial:
                stimuli = original_trial['stimuli']
            # Also check RawData structure (Das files often have RawData.stimuli)
            elif hasattr(original_trial, 'RawData') and hasattr(original_trial.RawData, 'stimuli'):
                stimuli = original_trial.RawData.stimuli
            
            if stimuli is None:
                return None, None
            
            # Convert to list if needed
            if isinstance(stimuli, np.ndarray):
                stimuli = stimuli.flatten().tolist()
            elif not isinstance(stimuli, (list, tuple)):
                stimuli = [stimuli]
            
            if len(stimuli) < 2:
                return None, None
            
            # Get stimulus names
            left_stim = str(stimuli[0]) if len(stimuli) > 0 else None
            right_stim = str(stimuli[1]) if len(stimuli) > 1 else None
            
            # Extract envelopes from audio files
            left_env = None
            right_env = None
            
            if self.das_audio_dir and self.das_audio_dir.exists():
                if left_stim:
                    # Try different audio file extensions
                    audio_file = None
                    for ext in ['.wav', '.WAV', '.mp3', '.MP3']:
                        candidate = self.das_audio_dir / f"{left_stim}{ext}"
                        if candidate.exists():
                            audio_file = candidate
                            break
                    
                    # Try pattern matching if direct match failed
                    if audio_file is None:
                        for f in self.das_audio_dir.glob(f"*{left_stim}*"):
                            if f.suffix.lower() in ['.wav', '.mp3']:
                                audio_file = f
                                break
                    
                    if audio_file and audio_file.exists():
                        extract_fn = self._extract_envelope_from_audio_4band if use_4band else self._extract_envelope_from_audio
                        left_env = extract_fn(audio_file, target_length, self.target_sampling_rate)
                        # Diagnostic: Log successful extraction
                        if not hasattr(self, '_das_left_extracted'):
                            self._das_left_extracted = 0
                        if self._das_left_extracted < 3:
                            print(f"  ✓ Das {subject_id} trial {trial_idx}: Extracted left envelope from '{left_stim}' -> {audio_file.name}")
                            self._das_left_extracted += 1
                    else:
                        # Diagnostic: Log missing left audio file
                        if not hasattr(self, '_das_left_audio_missing'):
                            self._das_left_audio_missing = 0
                        if self._das_left_audio_missing < 3:
                            print(f"  ⚠️  Das {subject_id} trial {trial_idx}: Left audio file not found for stimulus '{left_stim}'")
                            print(f"      Searched in: {self.das_audio_dir}")
                            # List available files for debugging
                            available_files = list(self.das_audio_dir.glob('*'))[:5]
                            if available_files:
                                print(f"      Sample available files: {[f.name for f in available_files]}")
                            self._das_left_audio_missing += 1
                
                if right_stim:
                    audio_file = None
                    for ext in ['.wav', '.WAV', '.mp3', '.MP3']:
                        candidate = self.das_audio_dir / f"{right_stim}{ext}"
                        if candidate.exists():
                            audio_file = candidate
                            break
                    
                    if audio_file is None:
                        for f in self.das_audio_dir.glob(f"*{right_stim}*"):
                            if f.suffix.lower() in ['.wav', '.mp3']:
                                audio_file = f
                                break
                    
                    if audio_file and audio_file.exists():
                        extract_fn = self._extract_envelope_from_audio_4band if use_4band else self._extract_envelope_from_audio
                        right_env = extract_fn(audio_file, target_length, self.target_sampling_rate)
                        # Diagnostic: Log successful extraction
                        if not hasattr(self, '_das_right_extracted'):
                            self._das_right_extracted = 0
                        if self._das_right_extracted < 3:
                            print(f"  ✓ Das {subject_id} trial {trial_idx}: Extracted right envelope from '{right_stim}' -> {audio_file.name}")
                            self._das_right_extracted += 1
                    else:
                        # Diagnostic: Log missing right audio file
                        if not hasattr(self, '_das_right_audio_missing'):
                            self._das_right_audio_missing = 0
                        if self._das_right_audio_missing < 3:
                            print(f"  ⚠️  Das {subject_id} trial {trial_idx}: Right audio file not found for stimulus '{right_stim}'")
                            self._das_right_audio_missing += 1
            
            return left_env, right_env
            
        except Exception as e:
            return None, None
    
    def _extract_fulsang_envelopes_from_audio(self, subject_id: str, trial_idx: int, target_length: int, 
                                               trial_data: Optional[object] = None,
                                               subject_num: Optional[str] = None,
                                               attend_lr: Optional[int] = None,
                                               attend_mf_label: Optional[int] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract envelopes from Fulsang audio files.
        
        Fulsang: two speakers per trial (Aske=male, Marianne=female). Which is left/right
        depends on exp_info: attend_mf (1=male, 2=female), attend_lr (1=left, 2=right).
        Left channel = speaker on left position; right = speaker on right position.
        When attend_lr and attend_mf_label are provided, left_is_male = (attend_mf_label==0)==(attend_lr==1).
        Otherwise falls back to aske=left, marianne=right.
        
        Args:
            subject_id: Subject ID (e.g., 'S1' or 'sub01')
            trial_idx: Trial index
            target_length: Target envelope length
            trial_data: Optional trial data object
            attend_lr: 1=left, 2=right (spatial position of attended speaker) for this trial
            attend_mf_label: 0=male attended, 1=female attended (our label) for this trial
        """
        if not self.fulsang_audio_dir or not self.fulsang_audio_dir.exists():
            return None, None
        
        try:
            # Method 1: Try to get audio file names from trial data (if saved during MWF processing)
            audio_file_male = None
            audio_file_female = None
            
            if trial_data is not None:
                if hasattr(trial_data, 'audio_file_male'):
                    audio_file_male = trial_data.audio_file_male
                elif isinstance(trial_data, dict) and 'audio_file_male' in trial_data:
                    audio_file_male = trial_data['audio_file_male']
                
                if hasattr(trial_data, 'audio_file_female'):
                    audio_file_female = trial_data.audio_file_female
                elif isinstance(trial_data, dict) and 'audio_file_female' in trial_data:
                    audio_file_female = trial_data['audio_file_female']
            
            # Method 2: Use sequential mapping based on sorted audio files
            # Fulsang has two speakers: aske (male) and marianne (female)
            # Each story/trial combination has both speakers
            if audio_file_male is None or audio_file_female is None:
                # Get all audio files sorted by name
                all_audio_files = sorted(list(self.fulsang_audio_dir.glob('*.wav')))
                
                # Diagnostic: Log found audio files (first time only)
                if not hasattr(self, '_fulsang_audio_files_logged'):
                    print(f"\n  Fulsang audio directory: {self.fulsang_audio_dir}")
                    print(f"  Found {len(all_audio_files)} WAV files")
                    if len(all_audio_files) > 0:
                        print(f"  Sample files: {[f.name for f in all_audio_files[:5]]}")
                    self._fulsang_audio_files_logged = True
                
                # Separate by speaker
                aske_files = sorted([f for f in all_audio_files if 'aske' in f.name.lower()])
                marianne_files = sorted([f for f in all_audio_files if 'marianne' in f.name.lower()])
                
                # Diagnostic: Log speaker file counts
                if not hasattr(self, '_fulsang_speaker_files_logged'):
                    print(f"  Aske (male) files: {len(aske_files)}")
                    print(f"  Marianne (female) files: {len(marianne_files)}")
                    if len(aske_files) > 0:
                        print(f"    Sample aske files: {[f.name for f in aske_files[:3]]}")
                    if len(marianne_files) > 0:
                        print(f"    Sample marianne files: {[f.name for f in marianne_files[:3]]}")
                    self._fulsang_speaker_files_logged = True
                
                # Map trial_idx to audio files (sequential mapping)
                # Each trial has both speakers, so we need to find matching story/trial pairs
                # Group files by story and trial number
                aske_by_trial = {}
                marianne_by_trial = {}
                
                def extract_story_trial(filename):
                    """Extract (story_num, trial_num) from filename.
                    
                    Handles formats:
                    - aske_story1_trial_1.wav -> (1, 1)
                    - aske_story1_trial1.wav -> (1, 1)
                    - aske_story_1_trial_1.wav -> (1, 1)
                    """
                    stem = filename.stem.lower()
                    parts = stem.split('_')
                    
                    story_num = None
                    trial_num = None
                    
                    # Find story number
                    for i, part in enumerate(parts):
                        if 'story' in part:
                            # Extract number from 'story1' or 'story_1'
                            num_str = part.replace('story', '').replace('_', '')
                            if num_str.isdigit():
                                story_num = int(num_str)
                            elif i + 1 < len(parts) and parts[i + 1].isdigit():
                                story_num = int(parts[i + 1])
                            break
                    
                    # Find trial number
                    for i, part in enumerate(parts):
                        if 'trial' in part:
                            # Extract number from 'trial1' or 'trial_1'
                            num_str = part.replace('trial', '').replace('_', '')
                            if num_str.isdigit():
                                trial_num = int(num_str)
                            elif i + 1 < len(parts) and parts[i + 1].isdigit():
                                trial_num = int(parts[i + 1])
                            break
                    
                    return story_num, trial_num
                
                for f in aske_files:
                    story_num, trial_num = extract_story_trial(f)
                    if story_num is not None and trial_num is not None:
                        key = (story_num, trial_num)
                        if key not in aske_by_trial:
                            aske_by_trial[key] = []
                        aske_by_trial[key].append(f)
                
                for f in marianne_files:
                    story_num, trial_num = extract_story_trial(f)
                    if story_num is not None and trial_num is not None:
                        key = (story_num, trial_num)
                        if key not in marianne_by_trial:
                            marianne_by_trial[key] = []
                        marianne_by_trial[key].append(f)
                
                # Diagnostic: Log parsed file mapping (first time only)
                if not hasattr(self, '_fulsang_file_mapping_logged'):
                    print(f"  Parsed file mapping:")
                    print(f"    Aske files grouped: {len(aske_by_trial)} unique (story, trial) combinations")
                    print(f"    Marianne files grouped: {len(marianne_by_trial)} unique (story, trial) combinations")
                    if len(aske_by_trial) > 0:
                        sample_keys = sorted(list(aske_by_trial.keys()))[:3]
                        print(f"    Sample aske keys: {sample_keys}")
                    if len(marianne_by_trial) > 0:
                        sample_keys = sorted(list(marianne_by_trial.keys()))[:3]
                        print(f"    Sample marianne keys: {sample_keys}")
                    self._fulsang_file_mapping_logged = True
                
                # Get all unique (story, trial) combinations, sorted
                # Sort by story first, then by trial number
                all_combinations = sorted(set(list(aske_by_trial.keys()) + list(marianne_by_trial.keys())))
                
                # Map trial_idx to combination
                if trial_idx < len(all_combinations):
                    story_num, trial_num = all_combinations[trial_idx]
                    key = (story_num, trial_num)
                    
                    if key in aske_by_trial and len(aske_by_trial[key]) > 0:
                        audio_file_male = aske_by_trial[key][0]  # Use first matching file
                    if key in marianne_by_trial and len(marianne_by_trial[key]) > 0:
                        audio_file_female = marianne_by_trial[key][0]
                else:
                    # Fallback: use simple sequential mapping
                    if trial_idx < len(aske_files):
                        audio_file_male = aske_files[trial_idx]
                    if trial_idx < len(marianne_files):
                        audio_file_female = marianne_files[trial_idx]
                    
                    # Diagnostic: Log fallback usage
                    if not hasattr(self, '_fulsang_fallback_warnings'):
                        self._fulsang_fallback_warnings = set()
                    if (subject_id, trial_idx) not in self._fulsang_fallback_warnings and len(self._fulsang_fallback_warnings) < 5:
                        print(f"  ⚠️  Fulsang {subject_id} trial {trial_idx}: Using fallback sequential mapping (trial_idx >= {len(all_combinations)} combinations)")
                        self._fulsang_fallback_warnings.add((subject_id, trial_idx))
            
            # Left/right from exp_info: attend_lr (1=left, 2=right), attend_mf (1=male, 2=female).
            # left_is_male = (attend_mf_label==0)==(attend_lr==1). If unknown, fallback: aske=left, marianne=right.
            left_is_male = None
            if attend_lr is not None and attend_mf_label is not None and attend_lr in (1, 2):
                left_is_male = (attend_mf_label == 0) == (attend_lr == 1)
            male_env = None
            female_env = None
            left_env = None
            right_env = None
            
            if audio_file_male:
                if isinstance(audio_file_male, (str, Path)):
                    p = Path(audio_file_male)
                    # Avoid doubling path: glob returns paths that may include parent dir (e.g. Data/Fulsang/AUDIO/file.wav)
                    if p.is_absolute() and p.exists():
                        audio_path = p
                    else:
                        audio_path = self.fulsang_audio_dir / p.name
                else:
                    audio_path = audio_file_male
                if audio_path.exists():
                    male_env = self._extract_envelope_from_audio(audio_path, target_length, self.target_sampling_rate)
                    if not hasattr(self, '_fulsang_left_extracted'):
                        self._fulsang_left_extracted = 0
                    if self._fulsang_left_extracted < 3:
                        print(f"  ✓ Fulsang {subject_id} trial {trial_idx}: Extracted male (Aske) envelope from {audio_path.name}")
                        self._fulsang_left_extracted += 1
                else:
                    if not hasattr(self, '_fulsang_left_missing'):
                        self._fulsang_left_missing = 0
                    if self._fulsang_left_missing < 3:
                        print(f"  ⚠️  Fulsang {subject_id} trial {trial_idx}: Male audio file not found: {audio_path}")
                        self._fulsang_left_missing += 1
            if audio_file_female:
                if isinstance(audio_file_female, (str, Path)):
                    p = Path(audio_file_female)
                    if p.is_absolute() and p.exists():
                        audio_path = p
                    else:
                        audio_path = self.fulsang_audio_dir / p.name
                else:
                    audio_path = audio_file_female
                if audio_path.exists():
                    female_env = self._extract_envelope_from_audio(audio_path, target_length, self.target_sampling_rate)
                    if not hasattr(self, '_fulsang_right_extracted'):
                        self._fulsang_right_extracted = 0
                    if self._fulsang_right_extracted < 3:
                        print(f"  ✓ Fulsang {subject_id} trial {trial_idx}: Extracted female (Marianne) envelope from {audio_path.name}")
                        self._fulsang_right_extracted += 1
                else:
                    if not hasattr(self, '_fulsang_right_missing'):
                        self._fulsang_right_missing = 0
                    if self._fulsang_right_missing < 3:
                        print(f"  ⚠️  Fulsang {subject_id} trial {trial_idx}: Female audio file not found: {audio_path}")
                        self._fulsang_right_missing += 1
            if left_is_male is True:
                left_env = male_env
                right_env = female_env
            elif left_is_male is False:
                left_env = female_env
                right_env = male_env
            else:
                left_env = male_env
                right_env = female_env
            
            # If we still don't have files, try fallback only when trial_idx is beyond (story,trial) mapping.
            # When trial_idx < len(all_combinations), we already used the correct (story, trial) and one
            # speaker may simply not have that key — do NOT use filename-sorted index (would pair wrong files).
            if left_env is None or right_env is None:
                aske_files = sorted(list(self.fulsang_audio_dir.glob('aske_*.wav')))
                marianne_files = sorted(list(self.fulsang_audio_dir.glob('marianne_*.wav')))
                try:
                    n_combinations = len(all_combinations)
                    use_index_fallback = (trial_idx >= n_combinations)
                except NameError:
                    n_combinations = 0
                    use_index_fallback = True
                if use_index_fallback:
                    if trial_idx < len(aske_files) and left_env is None:
                        left_env = self._extract_envelope_from_audio(aske_files[trial_idx], target_length, self.target_sampling_rate)
                        if not hasattr(self, '_fulsang_fallback_direct'):
                            self._fulsang_fallback_direct = set()
                        if (subject_id, trial_idx, 'left') not in self._fulsang_fallback_direct and len(self._fulsang_fallback_direct) < 5:
                            print(f"  ⚠️  Fulsang {subject_id} trial {trial_idx}: Using index fallback for left (trial_idx >= {n_combinations} combinations)")
                            self._fulsang_fallback_direct.add((subject_id, trial_idx, 'left'))
                    if trial_idx < len(marianne_files) and right_env is None:
                        right_env = self._extract_envelope_from_audio(marianne_files[trial_idx], target_length, self.target_sampling_rate)
                        if not hasattr(self, '_fulsang_fallback_direct'):
                            self._fulsang_fallback_direct = set()
                        if (subject_id, trial_idx, 'right') not in self._fulsang_fallback_direct and len(self._fulsang_fallback_direct) < 5:
                            print(f"  ⚠️  Fulsang {subject_id} trial {trial_idx}: Using index fallback for right (trial_idx >= {n_combinations} combinations)")
                            self._fulsang_fallback_direct.add((subject_id, trial_idx, 'right'))
            
            return left_env, right_env
            
        except Exception as e:
            return None, None
    
    # 1-band envelope (Fulsang original processing) - same for both DAS and Fulsang in combined pipeline
    ENVELOPE_BANDS = 1

    def _compute_4band_envelope(self, audio_data: np.ndarray, fs: float) -> np.ndarray:
        """Compute 4-band filterbank envelopes (Telluride/DASCCA-style). Returns (N, 4)."""
        from scipy.signal import butter, filtfilt, hilbert
        from scipy.ndimage import uniform_filter1d
        nyquist = fs / 2
        bands = [
            (max(1, 0), min(500, nyquist * 0.9)),
            (max(1, 500), min(1500, nyquist * 0.9)),
            (max(1, 1500), min(4000, nyquist * 0.9)),
            (max(1, 4000), min(8000, nyquist * 0.9)),
        ]
        envelopes = []
        for low, high in bands:
            try:
                if low <= 0:
                    low = 1.0
                if high >= nyquist:
                    high = nyquist * 0.95
                if low >= high:
                    envelope = np.abs(hilbert(audio_data))
                else:
                    low_norm = max(0.001, min(low / nyquist, 0.99))
                    high_norm = max(0.001, min(high / nyquist, 0.99))
                    if low_norm >= high_norm:
                        low_norm, high_norm = 0.001, min(0.99, 0.1)
                    b, a = butter(4, [low_norm, high_norm], btype='band')
                    filtered = filtfilt(b, a, audio_data)
                    envelope = np.abs(hilbert(filtered))
            except Exception:
                envelope = np.abs(hilbert(audio_data))
            if len(envelope) > 9:
                envelope = uniform_filter1d(envelope.astype(np.float64), size=9, mode='nearest')
            envelopes.append(envelope.astype(np.float32))
        result = np.column_stack(envelopes)
        for i in range(4):
            if np.max(np.abs(result[:, i])) > 0:
                result[:, i] = result[:, i] / (np.max(np.abs(result[:, i])) + 1e-8)
        return result.astype(np.float32)

    def _extract_envelope_from_audio_4band(self, audio_file: Path, target_length: int, target_fs: int = 128) -> Optional[np.ndarray]:
        """Extract 4-band envelope from audio file (DASCCA-style). Returns (target_length, 4) or None."""
        try:
            from scipy.io import wavfile
            from scipy import signal
            if not audio_file.exists():
                return None
            fs, audio_data = wavfile.read(str(audio_file))
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            if fs != target_fs:
                num_samples = int(len(audio_data) * target_fs / fs)
                audio_data = signal.resample(audio_data, num_samples)
            envelope_4band = self._compute_4band_envelope(audio_data, float(target_fs))
            if len(envelope_4band) != target_length:
                # Resize each band via interpolation
                out = np.zeros((target_length, 4), dtype=np.float32)
                for b in range(4):
                    src = envelope_4band[:, b]
                    src_idx = np.linspace(0.0, 1.0, num=len(src))
                    dst_idx = np.linspace(0.0, 1.0, num=target_length)
                    out[:, b] = np.interp(dst_idx, src_idx, src)
                envelope_4band = out
            return envelope_4band
        except Exception as e:
            if not hasattr(self, '_envelope_4band_errors'):
                self._envelope_4band_errors = 0
            if self._envelope_4band_errors < 3:
                print(f"  ⚠️  Error extracting 4-band envelope from {audio_file}: {e}")
                self._envelope_4band_errors += 1
            return None

    def _extract_envelope_from_audio(self, audio_file: Path, target_length: int, target_fs: int = 128) -> Optional[np.ndarray]:
        """Extract envelope from audio file (simple 1-band). Returns (target_length, 1). Used for Fulsang."""
        try:
            from scipy.io import wavfile
            from scipy import signal
            use_gammatone = getattr(self, 'use_gammatone_filter', False)
            # Track whether we successfully used gammatone for this dataset run.
            # Used later to decide whether to apply the additional envelope bandpass.
            if not hasattr(self, '_gammatone_in_use'):
                self._gammatone_in_use = False
            
            if not audio_file.exists():
                return None
            
            # Load audio
            fs, audio_data = wavfile.read(str(audio_file))
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Normalize
            audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Resample to target frequency
            if fs != target_fs:
                num_samples = int(len(audio_data) * target_fs / fs)
                audio_data = signal.resample(audio_data, num_samples)
            
            # Option 1: gammatone filterbank envelope (more cochlear-like)
            if use_gammatone:
                try:
                    from gammatone.filters import make_erb_filters, erb_filterbank
                    from scipy.signal import hilbert

                    # Simple 4-band gammatone filterbank, speech-relevant range
                    center_freqs = [150, 300, 600, 1200]
                    erb_filters = make_erb_filters(target_fs, center_freqs)
                    gt_out = erb_filterbank(audio_data, erb_filters)  # (n_bands, n_samples)
                    gt_out = gt_out.astype(np.float32)

                    # Hilbert envelope per band
                    envs = np.abs(hilbert(gt_out, axis=1)).astype(np.float32)
                    # Smooth each band with moving average
                    if envs.shape[1] > 9:
                        kernel = np.ones(9, dtype=np.float32) / 9.0
                        envs = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 1, envs)
                    # Collapse to single band by averaging across bands
                    envelope = envs.mean(axis=0)
                    self._gammatone_in_use = True
                except Exception as e:
                    if not hasattr(self, '_gammatone_warning_shown'):
                        print(f"  ⚠️  Gammatone filterbank failed or not installed (gammatone). Falling back to Hilbert envelope. Error: {e}")
                        self._gammatone_warning_shown = True
                    # Fallback to Hilbert/simple envelope below
                    use_gammatone = False
                    self._gammatone_in_use = False

            # Option 2: classic Hilbert / abs envelope
            if not use_gammatone:
                if getattr(self, 'use_hilbert_envelope', False):
                    from scipy.signal import hilbert
                    envelope = np.abs(hilbert(audio_data.astype(np.float64)))
                else:
                    envelope = np.abs(audio_data)
            
            # Smooth with moving average (9 samples window)
            if len(envelope) > 9:
                kernel = np.ones(9) / 9.0
                envelope = np.convolve(envelope, kernel, mode='same')
            
            # Resize to target length
            if len(envelope) != target_length:
                src_idx = np.linspace(0.0, 1.0, num=len(envelope))
                dst_idx = np.linspace(0.0, 1.0, num=target_length)
                envelope = np.interp(dst_idx, src_idx, envelope)
            
            envelope_2d = envelope.reshape(-1, 1).astype(np.float32)
            # Diagnostic: Log if envelope is essentially zero
            if np.max(np.abs(envelope_2d)) < 1e-6:
                if not hasattr(self, '_zero_envelope_warnings'):
                    self._zero_envelope_warnings = 0
                if self._zero_envelope_warnings < 3:
                    print(f"  ⚠️  Warning: Extracted envelope from {audio_file.name} is essentially zero (max={np.max(np.abs(envelope_2d)):.2e})")
                    self._zero_envelope_warnings += 1
            return envelope_2d
            
        except Exception as e:
            if not hasattr(self, '_envelope_extraction_errors'):
                self._envelope_extraction_errors = 0
            if self._envelope_extraction_errors < 3:
                print(f"  ⚠️  Error extracting envelope from {audio_file}: {e}")
                self._envelope_extraction_errors += 1
            return None
    
    def _extract_trial_envelopes(self, trial, dataset_name: str = 'Das') -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        def _maybe_get(obj, key):
            if hasattr(obj, key):
                return getattr(obj, key)
            if isinstance(obj, dict) and key in obj:
                return obj[key]
            if isinstance(obj, np.ndarray) and obj.dtype.names and key in obj.dtype.names:
                return obj[key]
            return None
        
        left_candidates = ['left_envelope', 'envelope_left', 'wavA', 'audio_left', 'stimulus_left', 
                          'left_audio', 'left_stimulus', 'envelopeL', 'envelope_l']
        right_candidates = ['right_envelope', 'envelope_right', 'wavB', 'audio_right', 'stimulus_right',
                           'right_audio', 'right_stimulus', 'envelopeR', 'envelope_r']
        
        left_raw = None
        right_raw = None
        found_field = None
        
        for l_key, r_key in zip(left_candidates, right_candidates):
            l_val = _maybe_get(trial, l_key)
            r_val = _maybe_get(trial, r_key)
            if l_val is not None and r_val is not None:
                left_raw = l_val
                right_raw = r_val
                found_field = (l_key, r_key)
                break
        
        if left_raw is None or right_raw is None:
            stimuli = _maybe_get(trial, 'stimuli')
            if stimuli is not None:
                try:
                    stimuli_list = list(stimuli)
                except TypeError:
                    stimuli_list = [stimuli]
                if len(stimuli_list) >= 2:
                    left_raw = stimuli_list[0]
                    right_raw = stimuli_list[1]
                    found_field = 'stimuli'
        
        # Debug: Log what fields are available if envelopes not found
        # Skip for Fulsang-MWF: MWF .mat trials don't have envelope fields; envelopes are loaded from WAV
        if (left_raw is None or right_raw is None) and dataset_name != 'Fulsang-MWF':
            # Only log for first few trials to avoid spam
            if not hasattr(self, '_envelope_debug_count'):
                self._envelope_debug_count = 0
            
            if self._envelope_debug_count < 3:
                available_fields = []
                if hasattr(trial, '__dict__'):
                    available_fields.extend(trial.__dict__.keys())
                elif isinstance(trial, dict):
                    available_fields.extend(trial.keys())
                elif isinstance(trial, np.ndarray) and trial.dtype.names:
                    available_fields.extend(trial.dtype.names)
                
                print(f"  ⚠️  {dataset_name}: Envelopes not found in trial. Available fields: {available_fields[:10]}")
                self._envelope_debug_count += 1
        
        left_env = self._compute_signal_envelope(left_raw)
        right_env = self._compute_signal_envelope(right_raw)
        return left_env, right_env
    
    def _prepare_trial_envelopes(self, trial, target_length: int, label: int, dataset_name: str) -> Tuple[np.ndarray, np.ndarray, bool]:
        left_env, right_env = self._extract_trial_envelopes(trial, dataset_name)
        have_real = left_env is not None and right_env is not None
        
        # Align envelope lengths
        if left_env is not None:
            left_env = self._align_envelope_length(left_env, target_length)
        if right_env is not None:
            right_env = self._align_envelope_length(right_env, target_length)
        
        # Check if alignment succeeded
        if left_env is None or right_env is None:
            left_env, right_env = self._fallback_envelopes(target_length, label)
            have_real = False
        else:
            # Only count as real if both envelopes are valid arrays with data
            if (isinstance(left_env, np.ndarray) and isinstance(right_env, np.ndarray) and
                left_env.size > 0 and right_env.size > 0 and
                np.any(left_env != 0) and np.any(right_env != 0)):  # Check they're not all zeros
                self._real_envelope_frames += target_length
            else:
                have_real = False
                left_env, right_env = self._fallback_envelopes(target_length, label)
        
        return left_env.astype(np.float32), right_env.astype(np.float32), have_real
    
    def _load_das_mwf_data(self) -> Tuple[np.ndarray, np.ndarray, List[Dict], List[int]]:
        """Load MWF-cleaned Das dataset from centralized location."""
        # Only check centralized location
        mwf_files = sorted(list(self.das_mwf_dir.glob("S*_MWF.mat")))
        
        if not mwf_files:
            raise ValueError(f"No MWF-cleaned Das files found in {self.das_mwf_dir}\n"
                           f"Expected files: S1_MWF.mat, S2_MWF.mat, etc.\n"
                           f"Please run MWF processing first. Files should be in: {self.das_mwf_dir}")
        
        all_eeg = []
        all_labels = []
        all_metadata = []
        trial_lengths = []
        all_left_env = []
        all_right_env = []
        
        # Use pre-loaded original Das files (loaded in __init__)
        if self.original_das_files:
            print(f"Using {len(self.original_das_files)} pre-loaded original Das files for envelope extraction...")
            for subject_id, orig_data in self.original_das_files.items():
                # Check if this file has stimuli info
                has_stimuli = False
                if 'trials' in orig_data:
                    trials = orig_data['trials']
                    if not isinstance(trials, np.ndarray):
                        trials = [trials]
                    else:
                        trials = trials.flatten()
                    if len(trials) > 0:
                        first_trial = trials[0]
                        if hasattr(first_trial, 'stimuli') or (isinstance(first_trial, dict) and 'stimuli' in first_trial):
                            has_stimuli = True
                if has_stimuli:
                    print(f"  ✓ {subject_id}: Found stimuli information")
                else:
                    print(f"  ⚠️  {subject_id}: No stimuli information found (will try to extract from audio files)")
        
        for mwf_file in tqdm(mwf_files, desc="Loading Das MWF data"):
            try:
                data = sio.loadmat(str(mwf_file), squeeze_me=True, struct_as_record=False)
                subject_id = mwf_file.stem.replace('_MWF', '')
                
                # Get original trial data for envelope extraction
                original_trials = None
                if subject_id in self.original_das_files:
                    orig_data = self.original_das_files[subject_id]
                    if 'trials' in orig_data:
                        original_trials = orig_data['trials']
                        if not isinstance(original_trials, np.ndarray):
                            original_trials = [original_trials]
                        else:
                            original_trials = original_trials.flatten()
                
                if 'trials' in data:
                    trials = data['trials']
                    if not isinstance(trials, np.ndarray):
                        trials = [trials]
                    else:
                        trials = trials.flatten()
                    
                    for trial_idx, trial in enumerate(trials):
                        if hasattr(trial, 'eeg_data'):
                            eeg_data = trial.eeg_data
                        elif isinstance(trial, dict):
                            eeg_data = trial.get('eeg_data', None)
                        else:
                            continue
                        
                        if eeg_data is None:
                            continue
                        
                        # Ensure eeg_data is 2D (samples x channels)
                        if len(eeg_data.shape) == 1:
                            eeg_data = eeg_data.reshape(-1, 1)
                        elif len(eeg_data.shape) > 2:
                            eeg_data = eeg_data.reshape(eeg_data.shape[0], -1)
                        
                        # Get attended ear label
                        if hasattr(trial, 'attended_ear'):
                            attended_ear = trial.attended_ear
                        elif isinstance(trial, dict):
                            attended_ear = trial.get('attended_ear', 'L')
                        else:
                            attended_ear = 'L'
                        
                        # Convert numpy array to scalar if needed (avoid boolean ambiguity error)
                        if isinstance(attended_ear, np.ndarray):
                            if attended_ear.size == 0:
                                attended_ear = 'L'
                            else:
                                attended_ear = str(attended_ear.item() if attended_ear.ndim == 0 else attended_ear.flat[0])
                        else:
                            attended_ear = str(attended_ear)
                        
                        # Convert to label (L=0, R=1)
                        label = 0 if attended_ear.upper() == 'L' else 1
                        
                        # Extract envelopes from original file or audio files
                        left_env, right_env = self._extract_das_envelopes_from_original(
                            original_trials, trial_idx, eeg_data.shape[0], subject_id
                        )
                        
                        # Verify mapping: Check if attended_ear matches the stimuli assignment (only once per subject)
                        if not hasattr(self, '_das_mapping_verified'):
                            self._das_mapping_verified = set()
                        # Check original_trials properly (avoid numpy array boolean ambiguity)
                        has_original_trials = (original_trials is not None and 
                                              ((isinstance(original_trials, (list, tuple)) and len(original_trials) > 0) or
                                               (isinstance(original_trials, np.ndarray) and original_trials.size > 0)))
                        if subject_id not in self._das_mapping_verified and has_original_trials and trial_idx < len(original_trials):
                            original_trial = original_trials[trial_idx]
                            stimuli = None
                            if hasattr(original_trial, 'stimuli'):
                                stimuli = original_trial.stimuli
                            elif isinstance(original_trial, dict) and 'stimuli' in original_trial:
                                stimuli = original_trial['stimuli']
                            elif hasattr(original_trial, 'RawData') and hasattr(original_trial.RawData, 'stimuli'):
                                stimuli = original_trial.RawData.stimuli
                            
                            if stimuli is not None:
                                if isinstance(stimuli, np.ndarray):
                                    stimuli = stimuli.flatten().tolist()
                                elif not isinstance(stimuli, (list, tuple)):
                                    stimuli = [stimuli]
                                
                                if len(stimuli) >= 2:
                                    left_stim_name = str(stimuli[0]) if len(stimuli) > 0 else None
                                    right_stim_name = str(stimuli[1]) if len(stimuli) > 1 else None
                                    print(f"  ✓ Das {subject_id} trial {trial_idx}: Mapping verified")
                                    print(f"      Attended ear: {attended_ear}, Label: {label} (L=0, R=1)")
                                    print(f"      Left stimulus: {left_stim_name}, Right stimulus: {right_stim_name}")
                                    self._das_mapping_verified.add(subject_id)
                        
                        # Diagnostic: Check if envelopes are valid
                        if left_env is not None and right_env is not None:
                            left_env = np.asarray(left_env)
                            right_env = np.asarray(right_env)
                            # Ensure 2D shape
                            if len(left_env.shape) == 1:
                                left_env = left_env.reshape(-1, 1)
                            if len(right_env.shape) == 1:
                                right_env = right_env.reshape(-1, 1)
                            
                            # Check if envelopes have actual data
                            if np.any(left_env != 0) and np.any(right_env != 0):
                                self._real_envelope_frames += eeg_data.shape[0]
                            else:
                                # Envelopes are zeros, try fallback
                                if not hasattr(self, '_das_zero_env_count'):
                                    self._das_zero_env_count = 0
                                if self._das_zero_env_count < 3:
                                    print(f"  ⚠️  Das {subject_id} trial {trial_idx}: Extracted envelopes are zeros, using fallback")
                                    self._das_zero_env_count += 1
                                left_env, right_env, _ = self._prepare_trial_envelopes(trial, eeg_data.shape[0], label, dataset_name='Das-MWF')
                        else:
                            # Fallback to trying from trial object
                            left_env, right_env, _ = self._prepare_trial_envelopes(trial, eeg_data.shape[0], label, dataset_name='Das-MWF')
                        
                        all_eeg.append(eeg_data)
                        all_labels.append(label)
                        trial_lengths.append(eeg_data.shape[0])
                        all_metadata.append({
                            'subject_id': subject_id,
                            'trial_idx': trial_idx,
                            'dataset': 'Das',
                            'attended_ear': attended_ear,
                            'preprocessing': 'MWF'
                        })
                        all_left_env.append(left_env)
                        all_right_env.append(right_env)
            except Exception as e:
                print(f"Error loading {mwf_file}: {e}")
                continue
        
        if not all_eeg:
            raise ValueError("No valid Das MWF data loaded")
        
        # Normalize channel count within Das data
        channel_counts = [eeg.shape[1] for eeg in all_eeg]
        max_channels = max(channel_counts)
        
        if len(set(channel_counts)) > 1:
            print(f"Warning: Das data has inconsistent channels: {set(channel_counts)}")
            print(f"Padding to {max_channels} channels")
            normalized_eeg = []
            for eeg in all_eeg:
                if eeg.shape[1] < max_channels:
                    padding = max_channels - eeg.shape[1]
                    pad_data = np.zeros((eeg.shape[0], padding), dtype=eeg.dtype)
                    eeg = np.hstack([eeg, pad_data])
                normalized_eeg.append(eeg)
            all_eeg = normalized_eeg
        
        eeg_data = np.vstack(all_eeg)
        labels = np.array(all_labels)
        
        return eeg_data, labels, all_metadata, trial_lengths, all_left_env, all_right_env

    def _run_das_preprocessing_combined(self) -> bool:
        """Run DAS combined preprocessing if .mat files exist. Returns True if preprocessing ran and produced TFRecords."""
        das_mat_files = list(self.das_original_dir.glob("S*.mat")) if self.das_original_dir.exists() else []
        if not das_mat_files:
            return False
        try:
            from das_preprocessing_combined import DasPreprocessorCombined
        except ImportError:
            return False
        audio_dir = str(self.das_audio_dir) if self.das_audio_dir else None
        preprocessor = DasPreprocessorCombined(
            data_dir=str(self.das_original_dir),
            output_dir=str(self.das_data_dir),
            audio_dir=audio_dir or "Data/Das/4004271/stimuli/stimuli",
        )
        n = preprocessor.create_tfrecord_data()
        return n > 0

    def _load_das_combined_data(self) -> Tuple[np.ndarray, np.ndarray, List[Dict], List[int]]:
        """Load DAS from das_preprocessing_combined.py (128→64 Hz). Use das_data_dir='das_combined_preprocessed', target_sampling_rate=64."""
        import tensorflow as tf
        tfrecord_dir = self.das_data_dir / "tfrecords"
        if not tfrecord_dir.exists():
            if self.run_das_preprocessing_if_missing:
                print(f"DAS TFRecords not found at {tfrecord_dir}. Running DAS combined preprocessing by default...")
                self._run_das_preprocessing_combined()
            if not tfrecord_dir.exists():
                raise ValueError(
                    f"DAS combined TFRecord directory does not exist: {tfrecord_dir}\n"
                    "Please run: python das_preprocessing_combined.py\n"
                    "Requirements: DAS .mat files (S*.mat) in Data/Das/4004271 (or set --data_dir).\n"
                    "Alternatively use Das MWF: das_preprocessing_type='MWF' and das_mwf_dir='MWF_cleaned_DAS' (S*_MWF.mat)."
                )
        tfrecord_files = list(tfrecord_dir.glob("*.tfrecords")) or list(tfrecord_dir.glob("*/*.tfrecords"))
        if not tfrecord_files and self.run_das_preprocessing_if_missing:
            print(f"No TFRecord files in {tfrecord_dir}. Running DAS combined preprocessing by default...")
            if self._run_das_preprocessing_combined():
                tfrecord_files = list(tfrecord_dir.glob("*.tfrecords")) or list(tfrecord_dir.glob("*/*.tfrecords"))
        if not tfrecord_files:
            raise ValueError(
                f"No TFRecord files in {tfrecord_dir}\n"
                "Please run: python das_preprocessing_combined.py\n"
                "Requirements: DAS .mat files (S*.mat) in Data/Das/4004271 (or set --data_dir).\n"
                "Alternatively use Das MWF: das_preprocessing_type='MWF' and das_mwf_dir='MWF_cleaned_DAS' (S*_MWF.mat)."
            )
        all_eeg, all_labels, all_metadata, trial_lengths, all_left_env, all_right_env = [], [], [], [], [], []
        current_trial_samples, current_trial_label = [], None
        current_trial_id, current_subject_id = None, None
        current_left_audio_file, current_right_audio_file = None, None
        current_trial_index = 0

        def _resolve(path_str: Optional[str]) -> Optional[Path]:
            if not path_str: return None
            p = Path(path_str)
            if p.exists(): return p
            if self.das_audio_dir and self.das_audio_dir.exists():
                # Try exact basename, then path_str as relative, then stem match
                for alt in [self.das_audio_dir / p.name, self.das_audio_dir / path_str]:
                    if alt.exists(): return alt
                # Try stem with common extensions (TFRecord may store path without extension)
                stem = p.stem
                for ext in ['.wav', '.WAV', '.mp3', '.MP3', '']:
                    cand = self.das_audio_dir / (stem + ext)
                    if cand.exists(): return cand
                # Glob by stem in case filename differs
                for f in self.das_audio_dir.glob(f"*{stem}*"):
                    if f.suffix.lower() in ['.wav', '.mp3']: return f
            return None

        _das_fallback_count = [0]  # use list so inner function can mutate

        def _get_env(trial_length: int, label: int, left_path: Optional[str], right_path: Optional[str],
                     subject_id: Optional[str] = None, trial_idx: Optional[int] = None):
            left_p, right_p = _resolve(left_path), _resolve(right_path)
            left_env = self._extract_envelope_from_audio(left_p, trial_length, self.target_sampling_rate) if left_p else None
            right_env = self._extract_envelope_from_audio(right_p, trial_length, self.target_sampling_rate) if right_p else None
            if left_env is not None and right_env is not None:
                self._real_envelope_frames += trial_length
                return left_env, right_env
            if subject_id is not None and trial_idx is not None and self.original_das_files:
                orig = self.original_das_files.get(subject_id)
                if orig is None and subject_id.isdigit():
                    orig = self.original_das_files.get('S' + subject_id)
                if orig is None and subject_id.startswith('S'):
                    orig = self.original_das_files.get(subject_id[1:])
                if orig and 'trials' in orig:
                    ot = orig['trials']
                    ot = np.array(ot).flatten().tolist() if isinstance(ot, np.ndarray) else list(ot)
                    if trial_idx < len(ot):
                        le, re = self._extract_das_envelopes_from_original(ot, trial_idx, trial_length, subject_id, use_4band=False)
                        if le is not None and re is not None:
                            self._real_envelope_frames += trial_length
                            return le, re
            if left_env is not None and right_env is None:
                right_env = left_env
                self._real_envelope_frames += trial_length
                return left_env, right_env
            if right_env is not None and left_env is None:
                left_env = right_env
                self._real_envelope_frames += trial_length
                return left_env, right_env
            _das_fallback_count[0] += 1
            return self._fallback_envelopes(trial_length, label)

        for tfrecord_file in tqdm(tfrecord_files, desc="Loading Das TFRecords (COMBINED_DAS / 16-subjects)"):
            try:
                for record in tf.data.TFRecordDataset(str(tfrecord_file)):
                    try:
                        ex = tf.train.Example.FromString(record.numpy())
                        f = ex.features.feature
                        if 'eeg' not in f or 'attended_ear' not in f: continue
                        ev = f['eeg'].float_list.value
                        if not ev or len(ev) != 64: continue
                        eeg_sample = np.array(ev, dtype=np.float32).reshape(1, 64)
                        left_audio_path = None
                        right_audio_path = None
                        if 'left_audio_file' in f and f['left_audio_file'].bytes_list.value:
                            left_audio_path = f['left_audio_file'].bytes_list.value[0].decode('utf-8')
                        if 'right_audio_file' in f and f['right_audio_file'].bytes_list.value:
                            right_audio_path = f['right_audio_file'].bytes_list.value[0].decode('utf-8')
                        # Match DASCCA / das_16subjects: normalize left=track1, right=track2
                        swapped_left_right = False
                        if left_audio_path and right_audio_path:
                            if 'track2' in left_audio_path and 'track1' in right_audio_path:
                                left_audio_path, right_audio_path = right_audio_path, left_audio_path
                                swapped_left_right = True
                        label = 0 if f['attended_ear'].bytes_list.value[0].decode('utf-8').upper() == 'L' else 1
                        if swapped_left_right:
                            label = 1 - label
                        subject_id = f['subject_id'].bytes_list.value[0].decode('utf-8') if 'subject_id' in f else "unknown"
                        if 'trial_id' in f and f['trial_id'].int64_list.value:
                            trial_id = f['trial_id'].int64_list.value[0]
                        elif 'trial_index' in f and f['trial_index'].int64_list.value:
                            trial_id = f['trial_index'].int64_list.value[0]
                        else:
                            trial_id = 0
                        trial_index_0based = f['trial_index'].int64_list.value[0] if 'trial_index' in f and f['trial_index'].int64_list.value else (int(trial_id) - 1 if 1 <= int(trial_id) <= 20 else 0)
                        if current_trial_id != trial_id or current_subject_id != subject_id:
                            if current_trial_samples:
                                te = np.vstack(current_trial_samples)
                                all_eeg.append(te)
                                all_labels.append(current_trial_label)
                                trial_lengths.append(len(current_trial_samples))
                                all_metadata.append({'subject_id': current_subject_id, 'trial_idx': current_trial_id, 'dataset': 'Das', 'attended_ear': 'L' if current_trial_label == 0 else 'R', 'preprocessing': 'COMBINED_DAS'})
                                le, re = _get_env(te.shape[0], current_trial_label, current_left_audio_file, current_right_audio_file, current_subject_id, current_trial_index)
                                all_left_env.append(le)
                                all_right_env.append(re)
                            current_trial_samples, current_trial_label = [eeg_sample], label
                            current_trial_id, current_subject_id = trial_id, subject_id
                            current_trial_index = int(trial_index_0based)
                            current_left_audio_file, current_right_audio_file = left_audio_path, right_audio_path
                        else:
                            current_trial_samples.append(eeg_sample)
                    except Exception: continue
            except Exception as e:
                if "OUT_OF_RANGE" not in str(e) and "End of sequence" not in str(e):
                    print(f"Warning: {tfrecord_file.name}: {e}")
            if current_trial_samples:
                te = np.vstack(current_trial_samples)
                all_eeg.append(te)
                all_labels.append(current_trial_label)
                trial_lengths.append(len(current_trial_samples))
                all_metadata.append({'subject_id': current_subject_id, 'trial_idx': current_trial_id, 'dataset': 'Das', 'attended_ear': 'L' if current_trial_label == 0 else 'R', 'preprocessing': 'COMBINED_DAS'})
                le, re = _get_env(te.shape[0], current_trial_label, current_left_audio_file, current_right_audio_file, current_subject_id, current_trial_index)
                all_left_env.append(le)
                all_right_env.append(re)
                current_trial_samples = []
        if not all_eeg:
            raise ValueError("No valid DAS combined data. Run: python das_preprocessing_combined.py")
        if _das_fallback_count[0] > 0:
            print(f"  ⚠ Das (COMBINED_DAS): {_das_fallback_count[0]} trial(s) used fallback envelopes (audio not read). "
                  f"Check left_audio_file/right_audio_file in TFRecords and das_audio_dir={self.das_audio_dir}")
        max_ch = max(eeg.shape[1] for eeg in all_eeg)
        if any(eeg.shape[1] != max_ch for eeg in all_eeg):
            all_eeg = [np.hstack([e, np.zeros((e.shape[0], max_ch - e.shape[1]), dtype=e.dtype)]) if e.shape[1] < max_ch else e for e in all_eeg]
        return np.vstack(all_eeg), np.array(all_labels), all_metadata, trial_lengths, all_left_env, all_right_env

    def _load_das_preprocessed_data(self) -> Tuple[np.ndarray, np.ndarray, List[Dict], List[int]]:
        """Load DASPREPROCESS Das dataset."""
        if not self.das_data_dir.exists():
            raise ValueError(f"Das preprocessed directory does not exist: {self.das_data_dir}\n"
                           f"Please run DASPREPROCESS first: python3 unified_preprocessing.py --dataset das")
        
        preprocessed_files = sorted(list(self.das_data_dir.glob("S*_preprocessed.mat")))
        if not preprocessed_files:
            raise ValueError(f"No preprocessed Das files found in {self.das_data_dir}\n"
                           f"Expected files: S1_preprocessed.mat, S2_preprocessed.mat, etc.\n"
                           f"Please run DASPREPROCESS first")
        
        all_eeg = []
        all_labels = []
        all_metadata = []
        trial_lengths = []
        all_left_env = []
        all_right_env = []
        
        for preprocessed_file in tqdm(preprocessed_files, desc="Loading Das preprocessed data"):
            try:
                data = sio.loadmat(str(preprocessed_file), squeeze_me=True, struct_as_record=False)
                subject_id = preprocessed_file.stem.replace('_preprocessed', '')
                
                # Get original trial data for envelope extraction
                original_trials = None
                if subject_id in self.original_das_files:
                    orig_data = self.original_das_files[subject_id]
                    if 'trials' in orig_data:
                        original_trials = orig_data['trials']
                        if not isinstance(original_trials, np.ndarray):
                            original_trials = [original_trials]
                        else:
                            original_trials = original_trials.flatten()
                
                if 'trials' in data:
                    trials = data['trials']
                    if not isinstance(trials, np.ndarray):
                        trials = [trials]
                    else:
                        trials = trials.flatten()
                    
                    for trial_idx, trial in enumerate(trials):
                        if hasattr(trial, 'eeg_data'):
                            eeg_data = trial.eeg_data
                        elif isinstance(trial, dict):
                            eeg_data = trial.get('eeg_data', None)
                        else:
                            continue
                        
                        if eeg_data is None:
                            continue
                        
                        # Ensure eeg_data is 2D (samples x channels)
                        if len(eeg_data.shape) == 1:
                            eeg_data = eeg_data.reshape(-1, 1)
                        elif len(eeg_data.shape) > 2:
                            eeg_data = eeg_data.reshape(eeg_data.shape[0], -1)
                        
                        # Get attended ear label
                        if hasattr(trial, 'attended_ear'):
                            attended_ear = trial.attended_ear
                        elif isinstance(trial, dict):
                            attended_ear = trial.get('attended_ear', 'L')
                        else:
                            attended_ear = 'L'
                        
                        # Convert numpy array to scalar if needed (avoid boolean ambiguity error)
                        if isinstance(attended_ear, np.ndarray):
                            if attended_ear.size == 0:
                                attended_ear = 'L'
                            else:
                                attended_ear = str(attended_ear.item() if attended_ear.ndim == 0 else attended_ear.flat[0])
                        else:
                            attended_ear = str(attended_ear)
                        
                        # Convert to label (L=0, R=1)
                        label = 0 if attended_ear.upper() == 'L' else 1
                        
                        # Extract envelopes from original file or audio files
                        left_env, right_env = self._extract_das_envelopes_from_original(
                            original_trials, trial_idx, eeg_data.shape[0], subject_id
                        )
                        
                        if left_env is None or right_env is None:
                            # Fallback to trying from trial object
                            left_env, right_env, _ = self._prepare_trial_envelopes(trial, eeg_data.shape[0], label, dataset_name='Das-Preprocessed')
                        else:
                            # Count as real envelopes
                            self._real_envelope_frames += eeg_data.shape[0]
                        
                        all_eeg.append(eeg_data)
                        all_labels.append(label)
                        trial_lengths.append(eeg_data.shape[0])
                        all_metadata.append({
                            'subject_id': subject_id,
                            'trial_idx': trial_idx,
                            'dataset': 'Das',
                            'attended_ear': attended_ear,
                            'preprocessing': 'DASPREPROCESS'
                        })
                        all_left_env.append(left_env)
                        all_right_env.append(right_env)
            except Exception as e:
                print(f"Error loading {preprocessed_file}: {e}")
                continue
        
        if not all_eeg:
            raise ValueError("No valid Das preprocessed data loaded")
        
        # Normalize channel count within Das data
        channel_counts = [eeg.shape[1] for eeg in all_eeg]
        max_channels = max(channel_counts)
        
        if len(set(channel_counts)) > 1:
            print(f"Warning: Das data has inconsistent channels: {set(channel_counts)}")
            print(f"Padding to {max_channels} channels")
            normalized_eeg = []
            for eeg in all_eeg:
                if eeg.shape[1] < max_channels:
                    padding = max_channels - eeg.shape[1]
                    pad_data = np.zeros((eeg.shape[0], padding), dtype=eeg.dtype)
                    eeg = np.hstack([eeg, pad_data])
                normalized_eeg.append(eeg)
            all_eeg = normalized_eeg
        
        eeg_data = np.vstack(all_eeg)
        labels = np.array(all_labels)
        
        return eeg_data, labels, all_metadata, trial_lengths, all_left_env, all_right_env
    
    def _load_fulsang_expinfo_sidecar(self, subject_id: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load expinfo from sidecar files S{n}_expinfo.mat (from MATLAB save_expinfo_only.m).
        Returns (attend_mf_array, attend_lr_array) for labels and left/right; 0/1 for attend_mf.
        """
        try:
            subject_num = int(subject_id.replace('S', '').replace('s', '').replace('sub', ''))
        except Exception:
            return None, None
        # Look for S{n}_expinfo.mat: explicit fulsang_expinfo_dir first, then EEG dir, Fulsang raw dir, repo-root Exp_Info
        search_dirs = []
        if self.fulsang_expinfo_dir is not None and self.fulsang_expinfo_dir.exists() and self.fulsang_expinfo_dir.is_dir():
            search_dirs.append(self.fulsang_expinfo_dir)
        repo_root = Path(__file__).resolve().parent
        for name in ("Exp_Info", "exp_info"):
            d = repo_root / name
            if d.exists() and d.is_dir() and d not in search_dirs:
                search_dirs.append(d)
        cwd = Path.cwd()
        for name in ("Exp_Info", "exp_info"):
            d = cwd / name
            if d.exists() and d.is_dir() and d not in search_dirs:
                search_dirs.append(d)
        if self.fulsang_eeg_raw_path is not None and not getattr(self, 'fulsang_eeg_raw_is_zip', True):
            if hasattr(self.fulsang_eeg_raw_path, 'is_dir') and self.fulsang_eeg_raw_path.is_dir():
                search_dirs.append(self.fulsang_eeg_raw_path)
            if hasattr(self.fulsang_eeg_raw_path, 'parent') and self.fulsang_eeg_raw_path.parent:
                for sub in ("Exp_Info", "exp_info"):
                    d = self.fulsang_eeg_raw_path.parent / sub
                    if d.exists() and d.is_dir() and d not in search_dirs:
                        search_dirs.append(d)
        if self.fulsang_raw_dir is not None:
            for sub in ("EEG", "eeg", "Exp_Info", "exp_info", ""):
                d = self.fulsang_raw_dir / sub if sub else self.fulsang_raw_dir
                if d.exists() and d.is_dir() and d not in search_dirs:
                    search_dirs.append(d)
            # When FULSANG_RAW_DIR is the EEG folder (e.g. Data/Fulsang/EEG), Exp_Info may be sibling: Data/Fulsang/Exp_Info
            if hasattr(self.fulsang_raw_dir, 'parent') and self.fulsang_raw_dir.parent:
                for sub in ("Exp_Info", "exp_info"):
                    d = self.fulsang_raw_dir.parent / sub
                    if d.exists() and d.is_dir() and d not in search_dirs:
                        search_dirs.append(d)
        # One-time diagnostic: where we look for S*_expinfo and whether we find them
        if not getattr(self, '_fulsang_sidecar_diagnostic_logged', False):
            self._fulsang_sidecar_diagnostic_logged = True
            print(f"\n  S*_expinfo: Searching for sidecar files (S<n>_expinfo.mat) in {len(search_dirs)} directories:")
            for i, base in enumerate(search_dirs):
                print(f"    [{i+1}] {base}")
            print(f"  S*_expinfo: For subject {subject_id}, checking S{subject_num}_expinfo.mat and S{subject_id}_expinfo.mat in each dir:")
            for base in search_dirs[:5]:
                for name in (f"S{subject_num}_expinfo.mat", f"S{subject_id}_expinfo.mat"):
                    p = base / name
                    print(f"    {p} -> exists={p.exists()}")
        for base in search_dirs:
            for name in (f"S{subject_num}_expinfo.mat", f"S{subject_id}_expinfo.mat"):
                path = base / name
                if not path.exists():
                    continue
                try:
                    mat = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
                    # Top-level keys (skip __header__ etc.)
                    mat_keys = [k for k in mat.keys() if not k.startswith('__')]
                    def _from_mat(obj, *names):
                        """Extract 1d array from mat var; handle nested struct and 0-d arrays."""
                        for n in names:
                            val = obj.get(n) if isinstance(obj, dict) else (getattr(obj, n, None) if hasattr(obj, n) else None)
                            if val is None and isinstance(obj, dict):
                                continue
                            if val is not None:
                                if isinstance(val, np.ndarray) and val.size > 0 and np.issubdtype(val.dtype, np.object_):
                                    val = val.flat[0]
                                val = np.atleast_1d(np.asarray(val).flatten())
                                if val.size > 0:
                                    return val
                        return None
                    attend_mf = _from_mat(mat, 'attend_mf', 'attend_MF', 'Attend_mf')
                    attend_lr = _from_mat(mat, 'attend_lr', 'attend_LR', 'Attend_lr')
                    if attend_mf is None or attend_lr is None:
                        exp = mat.get('expinfo') or mat.get('exp_info')
                        if exp is not None:
                            if isinstance(exp, np.ndarray) and exp.size > 0 and np.issubdtype(exp.dtype, np.object_):
                                exp = exp.flat[0]
                            def _from_exp(e, *names):
                                for n in names:
                                    v = e.get(n) if isinstance(e, dict) else getattr(e, n, None)
                                    if v is not None:
                                        v = np.atleast_1d(np.asarray(v).flatten())
                                        if v.size > 0:
                                            return v
                                return None
                            if attend_mf is None:
                                attend_mf = _from_exp(exp, 'attend_mf', 'attend_MF', 'Attend_mf')
                            if attend_lr is None:
                                attend_lr = _from_exp(exp, 'attend_lr', 'attend_LR', 'Attend_lr')
                    if attend_mf is not None:
                        attend_mf = np.atleast_1d(np.asarray(attend_mf).flatten())
                        attend_mf = np.array([0 if int(x) == 1 else 1 for x in attend_mf], dtype=np.int64)
                    if attend_lr is not None:
                        attend_lr = np.atleast_1d(np.asarray(attend_lr).flatten())
                    if attend_mf is not None or attend_lr is not None:
                        if not getattr(self, '_fulsang_sidecar_file_logged', False):
                            self._fulsang_sidecar_file_logged = True
                            print(f"  S*_expinfo: FOUND at {path}")
                            print(f"  S*_expinfo: Sidecar file {path.name}: keys={mat_keys}, attend_lr shape={attend_lr.shape if attend_lr is not None else None}, attend_mf shape={attend_mf.shape if attend_mf is not None else None}")
                        return attend_mf, attend_lr
                except Exception:
                    continue
        # One-time message when no S*_expinfo.mat was found in any search directory
        if not getattr(self, '_fulsang_sidecar_not_found_logged', False):
            self._fulsang_sidecar_not_found_logged = True
            print(f"  S*_expinfo: NOT FOUND — no S{subject_num}_expinfo.mat (or S{subject_id}_expinfo.mat) in any of the {len(search_dirs)} search directories.")
        return None, None

    def _load_fulsang_labels_from_raw_eeg(self, subject_id: str) -> Optional[np.ndarray]:
        """
        Load attention labels (expinfo.attend_mf) from raw EEG files.
        
        This follows the same logic as FULPRE.py - PRIMARY source of true attention labels.
        Raw EEG files contain expinfo.attend_mf which correctly encodes attended speaker (1=male, 2=female).
        
        Args:
            subject_id: Subject ID (e.g., 'S1' or 'sub01')
            
        Returns:
            Array of trial-level labels (0=male, 1=female) or None if not available
        """
        # Extract numeric subject ID
        try:
            subject_num = int(subject_id.replace('S', '').replace('s', '').replace('sub', ''))
        except:
            return None
        
        # Check cache first
        cache_key = f"S{subject_num}"
        if cache_key in self._fulsang_raw_labels_cache:
            return self._fulsang_raw_labels_cache[cache_key]
        
        # If no raw EEG path configured, return None
        if self.fulsang_eeg_raw_path is None:
            self._fulsang_raw_labels_cache[cache_key] = None
            return None
        
        # PRIORITY 1: Check for converted expinfo_struct.mat file (from MATLAB conversion script)
        if self.fulsang_eeg_raw_is_zip:
            zip_dir = self.fulsang_eeg_raw_path.parent
            converted_file = zip_dir / f"S{subject_num}_expinfo_struct.mat"
            if not converted_file.exists():
                converted_file = zip_dir / "EEG" / f"S{subject_num}_expinfo_struct.mat"
        else:
            converted_file = self.fulsang_eeg_raw_path / f"S{subject_num}_expinfo_struct.mat"
        
        if converted_file.exists():
            try:
                mat_data = sio.loadmat(str(converted_file), squeeze_me=True, struct_as_record=False)
                expinfo_struct = mat_data.get('expinfo_struct')
                if expinfo_struct is not None:
                    attend_mf = None
                    if isinstance(expinfo_struct, dict):
                        attend_mf = expinfo_struct.get('attend_mf')
                    elif hasattr(expinfo_struct, 'attend_mf'):
                        attend_mf = expinfo_struct.attend_mf
                    elif hasattr(expinfo_struct, 'dtype') and hasattr(expinfo_struct.dtype, 'names') and 'attend_mf' in expinfo_struct.dtype.names:
                        attend_mf = expinfo_struct['attend_mf']
                    
                    if attend_mf is not None:
                        if isinstance(attend_mf, np.ndarray):
                            attend_mf_list = attend_mf.flatten().tolist()
                        elif isinstance(attend_mf, (list, tuple)):
                            attend_mf_list = list(attend_mf)
                        else:
                            attend_mf_list = [attend_mf]
                        
                        # Convert 1=male->0, 2=female->1
                        labels = np.array([0 if v == 1 else 1 for v in attend_mf_list], dtype=np.int64)
                        
                        # Validate
                        unique_vals = set(attend_mf_list)
                        if unique_vals.issubset({1, 2}):
                            print(f"    ✓ Loaded {len(labels)} labels from converted expinfo file for {subject_id}")
                            self._fulsang_raw_labels_cache[cache_key] = labels
                            return labels
            except Exception as e:
                pass
        
        # PRIORITY 2: Try to load from original raw EEG file
        try:
            if self.fulsang_eeg_raw_is_zip:
                with zipfile.ZipFile(self.fulsang_eeg_raw_path, 'r') as zip_ref:
                    mat_filename = f"S{subject_num}.mat"
                    if mat_filename not in zip_ref.namelist():
                        possible_names = [f"EEG/S{subject_num}.mat", f"S{subject_num}/S{subject_num}.mat", f"eeg/S{subject_num}.mat"]
                        mat_filename = None
                        for name in possible_names:
                            if name in zip_ref.namelist():
                                mat_filename = name
                                break
                        if mat_filename is None:
                            self._fulsang_raw_labels_cache[cache_key] = None
                            return None
                    
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        tmp_dir_path = Path(tmp_dir)
                        zip_ref.extract(mat_filename, tmp_dir_path)
                        extracted_path = tmp_dir_path / Path(mat_filename).name
                        if not extracted_path.exists():
                            extracted_path = tmp_dir_path / mat_filename
                        
                        if extracted_path.exists():
                            mat_data = sio.loadmat(str(extracted_path), squeeze_me=False, struct_as_record=False)
                            expinfo = mat_data.get('expinfo')
                            if expinfo is not None:
                                attend_mf = None
                                if isinstance(expinfo, dict):
                                    attend_mf = expinfo.get('attend_mf')
                                elif hasattr(expinfo, 'attend_mf'):
                                    attend_mf = expinfo.attend_mf
                                
                                if attend_mf is not None:
                                    if isinstance(attend_mf, np.ndarray):
                                        attend_mf_list = attend_mf.flatten().tolist()
                                    elif isinstance(attend_mf, (list, tuple)):
                                        attend_mf_list = list(attend_mf)
                                    else:
                                        attend_mf_list = [attend_mf]
                                    
                                    labels = np.array([0 if v == 1 else 1 for v in attend_mf_list], dtype=np.int64)
                                    unique_vals = set(attend_mf_list)
                                    if unique_vals.issubset({1, 2}):
                                        print(f"    ✓ Loaded {len(labels)} labels from raw EEG (ZIP) for {subject_id}")
                                        self._fulsang_raw_labels_cache[cache_key] = labels
                                        return labels
            else:
                eeg_file = self.fulsang_eeg_raw_path / f"S{subject_num}.mat"
                if not eeg_file.exists():
                    possible_paths = [self.fulsang_eeg_raw_path / "EEG" / f"S{subject_num}.mat",
                                    self.fulsang_eeg_raw_path / "eeg" / f"S{subject_num}.mat"]
                    for path in possible_paths:
                        if path.exists():
                            eeg_file = path
                            break
                
                if eeg_file.exists():
                    mat_data = sio.loadmat(str(eeg_file), squeeze_me=False, struct_as_record=False)
                    expinfo = mat_data.get('expinfo')
                    if expinfo is not None:
                        attend_mf = None
                        if isinstance(expinfo, dict):
                            attend_mf = expinfo.get('attend_mf')
                        elif hasattr(expinfo, 'attend_mf'):
                            attend_mf = expinfo.attend_mf
                        
                        if attend_mf is not None:
                            if isinstance(attend_mf, np.ndarray):
                                attend_mf_list = attend_mf.flatten().tolist()
                            elif isinstance(attend_mf, (list, tuple)):
                                attend_mf_list = list(attend_mf)
                            else:
                                attend_mf_list = [attend_mf]
                            
                            labels = np.array([0 if v == 1 else 1 for v in attend_mf_list], dtype=np.int64)
                            unique_vals = set(attend_mf_list)
                            if unique_vals.issubset({1, 2}):
                                print(f"    ✓ Loaded {len(labels)} labels from raw EEG (directory) for {subject_id}")
                                self._fulsang_raw_labels_cache[cache_key] = labels
                                return labels
        except Exception as e:
            pass
        
        self._fulsang_raw_labels_cache[cache_key] = None
        return None
    
    def _load_fulsang_and_apply_mwf(self) -> Tuple[np.ndarray, np.ndarray, List[Dict], List[int]]:
        """Load Fulsang raw data and apply MWF filtering to centralized location."""
        # Only check centralized location
        existing_mwf_files = list(self.fulsang_mwf_dir.glob("sub*_MWF.mat"))
        
        # If MWF files exist, use them directly and skip raw file requirement
        if existing_mwf_files:
            print(f"  Found {len(existing_mwf_files)} existing MWF files in {self.fulsang_mwf_dir}, using them directly")
            mwf_files = sorted(list(self.fulsang_mwf_dir.glob("sub*_MWF.mat")))
            if not mwf_files:
                raise ValueError(f"No MWF-cleaned Fulsang files found. MWF processing may have failed.")
            # Skip to loading MWF files (code continues below after the MWF processing block)
        else:
            # MWF files don't exist, need raw files for processing
            if not self.fulsang_raw_dir.exists():
                raise ValueError(f"Fulsang raw directory does not exist: {self.fulsang_raw_dir}")
            
            # Load raw files
            raw_files = sorted(list(self.fulsang_raw_dir.glob("S*.mat")))
            if not raw_files:
                raise ValueError(f"No Fulsang raw files found in {self.fulsang_raw_dir}")
            
            # Apply MWF if not already done
            if len(existing_mwf_files) < len(raw_files):
                print(f"Applying MWF filtering to Fulsang data...")
                print(f"  Found {len(existing_mwf_files)} existing MWF files")
                print(f"  Found {len(raw_files)} raw files")
                print(f"  Processing missing files...")
                
                try:
                    from mwf_artifact_removal import FuglsangDatasetMWF
                    # Use centralized directory for new processing
                    mwf_processor = FuglsangDatasetMWF(
                        eeg_base_path=str(self.fulsang_raw_dir),
                        audio_base_path=str(self.fulsang_audio_dir) if self.fulsang_audio_dir else None,
                        output_dir=str(self.fulsang_mwf_dir)  # Save to centralized location
                    )
                    
                    # Process all subjects
                    for raw_file in raw_files:
                        subject_id = int(raw_file.stem.replace('S', ''))
                        mwf_file = self.fulsang_mwf_dir / f"sub{subject_id:02d}_MWF.mat"
                        
                        if not mwf_file.exists():
                            print(f"  Processing subject {subject_id}...")
                            try:
                                processed_data = mwf_processor.process_subject(subject_id)
                                if processed_data:
                                    mwf_processor.save_cleaned_data(processed_data)
                                    print(f"  ✓ Saved MWF data for subject {subject_id}")
                            except Exception as e:
                                print(f"  Warning: Failed to process subject {subject_id}: {e}")
                                import traceback
                                traceback.print_exc()
                                continue
                    
                    print("✓ MWF processing completed")
                except Exception as e:
                    print(f"Warning: MWF processing failed: {e}")
                    print("  Continuing with raw data (no MWF filtering)")
            
            # Get MWF files after processing (only from centralized location)
            mwf_files = sorted(list(self.fulsang_mwf_dir.glob("sub*_MWF.mat")))
        
        # Load MWF-cleaned data (mwf_files already set above if MWF files existed, or set in else block)
        if not mwf_files:
            raise ValueError(f"No MWF-cleaned Fulsang files found. MWF processing may have failed.")
        
        all_eeg = []
        all_labels = []
        all_metadata = []
        trial_lengths = []
        all_left_env = []
        all_right_env = []
        
        for mwf_file in tqdm(mwf_files, desc="Loading Fulsang MWF data"):
            try:
                data = sio.loadmat(str(mwf_file), squeeze_me=True, struct_as_record=False)
                # One-time diagnostic: show what keys are in Fulsang MWF files
                if not getattr(self, '_fulsang_mwf_keys_logged', False):
                    data_keys = [k for k in data.keys() if not k.startswith('__')]
                    print(f"  Fulsang MWF file keys (first file): {data_keys}")
                    if 'expinfo' in data:
                        ex = data['expinfo']
                        print(f"  MWF expinfo diagnostic: type={type(ex).__name__}, shape={getattr(ex, 'shape', 'N/A')}, dtype={getattr(ex, 'dtype', 'N/A')}")
                        if isinstance(ex, np.ndarray) and ex.size > 0:
                            ex0 = ex.flat[0]
                            print(f"    expinfo.flat[0]: type={type(ex0).__name__}, dir={[x for x in dir(ex0) if not x.startswith('_')][:20]}")
                            for attr in ('attend_lr', 'attend_mf', 'attend_LR', 'attend_MF'):
                                if hasattr(ex0, attr):
                                    v = getattr(ex0, attr)
                                    print(f"    expinfo.flat[0].{attr}: {type(v).__name__} {getattr(v, 'shape', v)}")
                    self._fulsang_mwf_keys_logged = True
                subject_id = mwf_file.stem.replace('_MWF', '').replace('sub', 'S')
                subject_num = subject_id.replace('S', '').replace('sub', '')
                
                if 'trials' in data:
                    trials = data['trials']
                    if not isinstance(trials, np.ndarray):
                        trials = [trials]
                    else:
                        trials = trials.flatten()
                    
                    # Comprehensive label extraction function
                    def extract_label_from_trial(trial, trial_idx=0, debug=False):
                        """
                        Extract attention label from trial structure.
                        Checks all possible field names and formats.
                        Returns (label_value, found, field_used) where found is True if label was extracted.
                        """
                        # List of possible field names to check (in order of preference)
                        # Fulsang MWF files use 'attention_label' (0 or 1)
                        # Das MWF files use 'attended_ear' ('L' or 'R')
                        possible_fields = [
                            'attention_label',  # Fulsang MWF format
                            'attended_ear',     # Das MWF format
                            'label', 'attention',
                            'event', 'events', 'event_code', 'eventcode',
                            'trial_label', 'trial_attention', 'attention_direction'
                        ]
                        
                        # Method 1: Check as object attributes
                        for field_name in possible_fields:
                            if hasattr(trial, field_name):
                                try:
                                    value = getattr(trial, field_name)
                                    label = _convert_to_label(value)
                                    if label is not None:
                                        if debug:
                                            print(f"    Trial {trial_idx}: Found label in attribute '{field_name}' = {value} -> {label}")
                                        return label, True, field_name
                                except Exception as e:
                                    if debug:
                                        print(f"    Trial {trial_idx}: Error reading attribute '{field_name}': {e}")
                        
                        # Method 2: Check as dictionary
                        if isinstance(trial, dict):
                            for field_name in possible_fields:
                                if field_name in trial:
                                    try:
                                        value = trial[field_name]
                                        label = _convert_to_label(value)
                                        if label is not None:
                                            if debug:
                                                print(f"    Trial {trial_idx}: Found label in dict key '{field_name}' = {value} -> {label}")
                                            return label, True, field_name
                                    except Exception as e:
                                        if debug:
                                            print(f"    Trial {trial_idx}: Error reading dict key '{field_name}': {e}")
                        
                        # Method 3: Check structured array fields (prioritize attention-related fields)
                        if isinstance(trial, np.ndarray) and trial.dtype.names:
                            # First check attention-related fields
                            for field_name in trial.dtype.names:
                                if any(keyword in field_name.lower() for keyword in ['attend', 'label', 'event']):
                                    try:
                                        value = trial[field_name]
                                        label = _convert_to_label(value)
                                        if label is not None:
                                            if debug:
                                                print(f"    Trial {trial_idx}: Found label in structured array field '{field_name}' = {value} -> {label}")
                                            return label, True, field_name
                                    except Exception as e:
                                        if debug:
                                            print(f"    Trial {trial_idx}: Error reading structured array field '{field_name}': {e}")
                            
                            # Then check all other fields
                            for field_name in trial.dtype.names:
                                if not any(keyword in field_name.lower() for keyword in ['attend', 'label', 'event']):
                                    try:
                                        value = trial[field_name]
                                        label = _convert_to_label(value)
                                        if label is not None:
                                            if debug:
                                                print(f"    Trial {trial_idx}: Found label in structured array field '{field_name}' = {value} -> {label}")
                                            return label, True, field_name
                                    except:
                                        pass
                        
                        # Method 4: If trial is a structured array, show available fields for debugging
                        if isinstance(trial, np.ndarray) and trial.dtype.names:
                            if debug:
                                print(f"    Trial {trial_idx}: Available structured array fields: {trial.dtype.names}")
                        
                        return None, False, None
                    
                    def _convert_to_label(value):
                        """
                        Convert various label formats to binary label (0 or 1).
                        
                        Handles:
                        - Fulsang MWF: attention_label as integer (0 or 1) or event codes (191, 192)
                        - Das MWF: attended_ear as string ('L' or 'R')
                        - Das TFRecord: attended_ear as bytes (b'L' or b'R')
                        """
                        if value is None:
                            return None
                        
                        # Handle numpy arrays
                        if isinstance(value, np.ndarray):
                            if value.size == 0:
                                return None
                            elif value.size == 1:
                                value = value.item()
                            else:
                                # Take first non-zero value or first value
                                flat = value.flatten()
                                for v in flat:
                                    if v != 0:
                                        value = v
                                        break
                                else:
                                    value = flat[0]
                        
                        # Convert to numeric
                        try:
                            # Handle string/bytes (Das format: 'L' or 'R')
                            if isinstance(value, (str, bytes)):
                                if isinstance(value, bytes):
                                    value_str = value.decode('utf-8').upper().strip()
                                else:
                                    value_str = str(value).upper().strip()
                                
                                if value_str in ['L', 'LEFT', '0']:
                                    return 0
                                elif value_str in ['R', 'RIGHT', '1']:
                                    return 1
                                else:
                                    return None
                            
                            # Handle numeric (Fulsang format: 0, 1, or event codes 191, 192, or data_preproc codes 1, 2)
                            elif isinstance(value, (int, float, np.integer, np.floating)):
                                value_int = int(value)
                                # Map event codes to binary labels (Fulsang event codes)
                                # Fulsang MWF files use event codes:
                                #   191 = Right attention (attend right)
                                #   192 = Left attention (attend left)
                                # Other codes (131, 132, 133, 135, 137, 141, 150, 160, 224, 240, 248, 252, 254, etc.)
                                # are non-attention trials (practice, rest, etc.) and should be skipped
                                if value_int == 192:
                                    return 0  # Left attention
                                elif value_int == 191:
                                    return 1  # Right attention
                                elif value_int in [0, 1]:
                                    # Direct binary labels (already correct)
                                    return value_int
                                # Handle data_preproc event codes (1 and 2) - UNVERIFIED, return None
                                # These codes need verification against expinfo to ensure correct mapping
                                elif value_int in [1, 2]:
                                    # Return None until verified - don't silently flip labels
                                    return None
                                else:
                                    # All other event codes are non-attention trials - skip them
                                    # This includes: 131, 132, 133, 135, 137, 141, 150, 160, 224, 240, 248, 252, 254, etc.
                                    return None
                        except Exception as e:
                            return None
                        
                        return None
                    
                    # Label priority: 1) Sidecar (Exp_Info S*_expinfo.mat), 2) Raw EEG, 3) MWF expinfo, 4) event, 5) trial-by-trial
                    raw_labels = None
                    label_source = 'unknown'
                    attend_lr_list = None
                    expinfo_labels = None
                    
                    # Priority 1: Sidecar S*_expinfo.mat in Exp_Info (first choice)
                    sidecar_mf, sidecar_lr = self._load_fulsang_expinfo_sidecar(subject_id)
                    n_trials = len(trials)
                    n_trials_use = n_trials  # may be reduced to match sidecar so labels and EEG trials match 1:1
                    if sidecar_lr is not None and len(sidecar_lr) > 0:
                        arr = np.array([0 if int(x) == 1 else 1 for x in sidecar_lr], dtype=np.int64)
                        if len(arr) >= n_trials:
                            expinfo_labels = arr[:n_trials].copy()
                            attend_lr_list = np.asarray(sidecar_lr).flatten()[:n_trials].astype(np.int64)
                        else:
                            # Sidecar has fewer labels than EEG trials: use only trials that have labels (no padding)
                            n_trials_use = len(sidecar_lr)
                            expinfo_labels = arr.copy()
                            attend_lr_list = np.asarray(sidecar_lr).flatten().astype(np.int64)
                        label_source = 'sidecar_expinfo.attend_lr'
                        if not getattr(self, '_fulsang_sidecar_first_logged', False):
                            if n_trials_use < n_trials:
                                print(f"  ✓ {subject_id}: Using labels from sidecar Exp_Info/S*_expinfo.mat attend_lr (priority 1) - {n_trials_use} trials (EEG had {n_trials}; truncated to match sidecar for 1:1 alignment)")
                            else:
                                print(f"  ✓ {subject_id}: Using labels from sidecar Exp_Info/S*_expinfo.mat attend_lr (priority 1) - {n_trials_use} trials (sidecar had {len(sidecar_lr)})")
                            self._fulsang_sidecar_first_logged = True
                    elif sidecar_mf is not None and len(sidecar_mf) > 0:
                        arr = np.asarray(sidecar_mf).flatten()
                        if len(arr) >= n_trials:
                            expinfo_labels = np.array([0 if int(x) == 1 else 1 for x in arr[:n_trials]], dtype=np.int64)
                        else:
                            n_trials_use = len(arr)
                            expinfo_labels = np.array([0 if int(x) == 1 else 1 for x in arr], dtype=np.int64)
                        label_source = 'sidecar_expinfo'
                        if not getattr(self, '_fulsang_sidecar_first_logged', False):
                            if n_trials_use < n_trials:
                                print(f"  ✓ {subject_id}: Using labels from sidecar Exp_Info/S*_expinfo.mat attend_mf - {n_trials_use} trials (EEG had {n_trials}; truncated to match sidecar)")
                            else:
                                print(f"  ✓ {subject_id}: Using labels from sidecar Exp_Info/S*_expinfo.mat attend_mf - {n_trials_use} trials")
                            self._fulsang_sidecar_first_logged = True
                    
                    # Priority 2: Raw EEG files (expinfo.attend_mf)
                    if expinfo_labels is None and self.fulsang_eeg_raw_path is not None:
                        raw_labels = self._load_fulsang_labels_from_raw_eeg(subject_id)
                        if raw_labels is not None and len(raw_labels) == len(trials):
                            print(f"  ✓ {subject_id}: Using labels from raw EEG files (expinfo.attend_mf) - {len(raw_labels)} trials")
                            label_source = 'raw_eeg.expinfo.attend_mf'
                    
                    # Priority 3: MWF file expinfo.attend_mf / attend_lr (only if sidecar and raw didn't provide labels)
                    if expinfo_labels is None and raw_labels is None and 'expinfo' in data:
                        try:
                            expinfo = data['expinfo']
                            # scipy.loadmat often wraps struct in 0-d or 1-d object array
                            if isinstance(expinfo, np.ndarray) and expinfo.size > 0 and np.issubdtype(expinfo.dtype, np.object_):
                                expinfo = expinfo.flat[0]
                            attend_mf = None
                            attend_lr = None
                            def _get_field(obj, *keys):
                                for k in keys:
                                    if isinstance(obj, dict) and k in obj:
                                        return obj[k]
                                    if hasattr(obj, k):
                                        return getattr(obj, k)
                                    if isinstance(obj, np.ndarray) and hasattr(obj, 'dtype') and obj.dtype.names and k in obj.dtype.names:
                                        return obj[k]
                                return None
                            attend_mf = _get_field(expinfo, 'attend_mf', 'attend_MF', 'Attend_mf')
                            attend_lr = _get_field(expinfo, 'attend_lr', 'attend_LR', 'Attend_lr')
                            # Flatten nested 0-d arrays from MATLAB
                            for name, val in [('attend_mf', attend_mf), ('attend_lr', attend_lr)]:
                                if val is not None and isinstance(val, np.ndarray) and val.size == 1 and val.ndim >= 1:
                                    v = val.flatten()[0]
                                    if name == 'attend_mf':
                                        attend_mf = v
                                    else:
                                        attend_lr = v
                            
                            if attend_mf is not None:
                                if isinstance(attend_mf, np.ndarray):
                                    attend_mf_list = attend_mf.flatten().tolist()
                                elif isinstance(attend_mf, (list, tuple)):
                                    attend_mf_list = list(attend_mf)
                                else:
                                    attend_mf_list = [attend_mf]
                                expinfo_labels = np.array([0 if v == 1 else 1 for v in attend_mf_list], dtype=np.int64)
                                if len(expinfo_labels) == len(trials):
                                    print(f"  ✓ {subject_id}: Using labels from MWF expinfo.attend_mf - {len(expinfo_labels)} trials")
                                    label_source = 'mwf.expinfo.attend_mf'
                            if attend_lr is not None:
                                if isinstance(attend_lr, np.ndarray):
                                    _arr = np.array(attend_lr.flatten().tolist())
                                elif isinstance(attend_lr, (list, tuple)):
                                    _arr = np.array(list(attend_lr))
                                else:
                                    _arr = np.array([attend_lr])
                                if _arr.shape[0] == len(trials):
                                    attend_lr_list = _arr
                                    if not hasattr(self, '_fulsang_attend_lr_logged'):
                                        print(f"  ✓ {subject_id}: Using expinfo.attend_lr for left/right speaker assignment ({attend_lr_list.shape[0]} trials)")
                                        self._fulsang_attend_lr_logged = True
                            # Use attend_lr as labels (1=left=0, 2=right=1) when attend_mf not in MWF
                            if expinfo_labels is None and attend_lr_list is not None and len(attend_lr_list) == len(trials):
                                expinfo_labels = np.array([0 if int(x) == 1 else 1 for x in attend_lr_list], dtype=np.int64)
                                label_source = 'mwf.expinfo.attend_lr'
                                if not getattr(self, '_fulsang_mwf_attend_lr_labels_logged', False):
                                    print(f"  ✓ {subject_id}: Using labels from MWF expinfo.attend_lr (left=0, right=1) - {len(expinfo_labels)} trials")
                                    self._fulsang_mwf_attend_lr_labels_logged = True
                        except Exception as e:
                            pass
                    
                    # Load attend_lr from MWF expinfo whenever expinfo exists (for left/right envelope assignment).
                    # This runs even when labels came from raw EEG or trial-by-trial, so left/right follows exp_info.
                    if attend_lr_list is None:
                        for expinfo_key in ('expinfo', 'exp_info'):
                            if attend_lr_list is not None:
                                break
                            if expinfo_key not in data:
                                continue
                            try:
                                expinfo = data[expinfo_key]
                                if isinstance(expinfo, np.ndarray) and expinfo.size > 0 and np.issubdtype(expinfo.dtype, np.object_):
                                    expinfo = expinfo.flat[0]
                                attend_lr = None
                                if isinstance(expinfo, dict):
                                    attend_lr = expinfo.get('attend_lr')
                                elif hasattr(expinfo, 'attend_lr'):
                                    attend_lr = getattr(expinfo, 'attend_lr', None)
                                elif isinstance(expinfo, np.ndarray) and hasattr(expinfo, 'dtype') and expinfo.dtype.names and 'attend_lr' in expinfo.dtype.names:
                                    attend_lr = expinfo['attend_lr']
                                if attend_lr is not None:
                                    if isinstance(attend_lr, np.ndarray):
                                        arr = np.array(attend_lr.flatten().tolist())
                                    elif isinstance(attend_lr, (list, tuple)):
                                        arr = np.array(list(attend_lr))
                                    else:
                                        arr = np.array([attend_lr])
                                    if arr.shape[0] == len(trials):
                                        attend_lr_list = arr
                                        if not hasattr(self, '_fulsang_attend_lr_logged'):
                                            print(f"  ✓ {subject_id}: Using {expinfo_key}.attend_lr for left/right speaker assignment ({attend_lr_list.shape[0]} trials)")
                                            self._fulsang_attend_lr_logged = True
                                        break
                            except Exception:
                                pass
                    
                    # Load expinfo from sidecar files S{n}_expinfo.mat (from MATLAB save_expinfo_only.m)
                    if (attend_lr_list is None or (raw_labels is None and expinfo_labels is None)):
                        sidecar_mf, sidecar_lr = self._load_fulsang_expinfo_sidecar(subject_id)
                        if sidecar_lr is not None and attend_lr_list is None and len(sidecar_lr) == len(trials):
                            attend_lr_list = sidecar_lr
                            if not hasattr(self, '_fulsang_attend_lr_logged'):
                                print(f"  ✓ {subject_id}: Using attend_lr from sidecar S*_expinfo.mat ({len(attend_lr_list)} trials)")
                                self._fulsang_attend_lr_logged = True
                        if sidecar_mf is not None and raw_labels is None and expinfo_labels is None and len(sidecar_mf) == len(trials):
                            expinfo_labels = sidecar_mf
                            label_source = 'sidecar_expinfo'
                            if not hasattr(self, '_fulsang_sidecar_labels_logged'):
                                print(f"  ✓ {subject_id}: Using labels from sidecar S*_expinfo.mat ({len(expinfo_labels)} trials)")
                                self._fulsang_sidecar_labels_logged = True
                        # Use attend_lr for left/right labels when no other labels yet (left=0, right=1)
                        if sidecar_lr is not None and raw_labels is None and expinfo_labels is None and len(sidecar_lr) == len(trials):
                            expinfo_labels = np.array([0 if int(x) == 1 else 1 for x in sidecar_lr], dtype=np.int64)
                            label_source = 'sidecar_expinfo.attend_lr'
                            if not hasattr(self, '_fulsang_sidecar_lr_labels_logged'):
                                print(f"  ✓ {subject_id}: Using left/right labels from sidecar S*_expinfo.mat attend_lr ({len(expinfo_labels)} trials)")
                                self._fulsang_sidecar_lr_labels_logged = True
                    
                    # Priority 3: Try event.eeg.value{1} from MWF file structure (like FULPRE.py)
                    event_labels = None
                    if raw_labels is None and expinfo_labels is None and 'data' in data:
                        try:
                            data_struct = data['data']
                            if isinstance(data_struct, np.ndarray) and data_struct.size > 0:
                                main_data = data_struct.flat[0]
                                if hasattr(main_data, 'event'):
                                    event = main_data.event
                                    if isinstance(event, np.ndarray) and event.dtype == object and event.size > 0:
                                        event_struct = event.flat[0]
                                        if hasattr(event_struct, 'eeg'):
                                            event_eeg = event_struct.eeg
                                            if isinstance(event_eeg, np.ndarray) and event_eeg.dtype == object and event_eeg.size >= len(trials):
                                                label_values = []
                                                for i in range(len(trials)):
                                                    trial_event = event_eeg.flat[i]
                                                    if hasattr(trial_event, 'value'):
                                                        value_cell = trial_event.value
                                                        if isinstance(value_cell, np.ndarray) and value_cell.dtype == object and value_cell.size > 0:
                                                            label_val = value_cell.flat[0]
                                                            if isinstance(label_val, np.ndarray):
                                                                label_val = label_val.flatten()[0] if label_val.size > 0 else None
                                                            if label_val is not None:
                                                                label_val = int(label_val)
                                                                if label_val in [1, 2]:
                                                                    label_values.append(0 if label_val == 1 else 1)
                                                                    continue
                                                    label_values = None
                                                    break
                                                
                                                if label_values is not None and len(label_values) == len(trials):
                                                    event_labels = np.array(label_values, dtype=np.int64)
                                                    print(f"  ✓ {subject_id}: Using labels from event.eeg.value{{1}} - {len(event_labels)} trials")
                                                    label_source = 'event.eeg.value{1}'
                        except Exception as e:
                            pass
                    
                    # Priority 4: Fall back to trial-by-trial extraction
                    label_lookup = {}  # Maps trial_idx -> (label, field_used)
                    skipped_trials = []
                    skipped_values = {}  # Track what values were in skipped trials
                    
                    # Use the best available labels
                    if raw_labels is not None:
                        # Use raw EEG labels (best quality)
                        for trial_idx in range(len(trials)):
                            if trial_idx < len(raw_labels):
                                label_lookup[trial_idx] = (raw_labels[trial_idx], 'raw_eeg')
                    elif expinfo_labels is not None:
                        # Use expinfo labels
                        for trial_idx in range(len(trials)):
                            if trial_idx < len(expinfo_labels):
                                label_lookup[trial_idx] = (expinfo_labels[trial_idx], 'expinfo')
                    elif event_labels is not None:
                        # Use event labels
                        for trial_idx in range(len(trials)):
                            if trial_idx < len(event_labels):
                                label_lookup[trial_idx] = (event_labels[trial_idx], 'event')
                    else:
                        # Fall back to trial-by-trial extraction
                        if not getattr(self, '_fulsang_sidecar_fallback_logged', False):
                            # One-time: try sidecar and report why we didn't use it
                            sidecar_mf, sidecar_lr = self._load_fulsang_expinfo_sidecar(subject_id)
                            if sidecar_lr is not None and len(sidecar_lr) == len(trials):
                                print(f"  ⚠ Sidecar has attend_lr for {subject_id} but was not used (check order of checks). Using it now.")
                                expinfo_labels = np.array([0 if int(x) == 1 else 1 for x in sidecar_lr], dtype=np.int64)
                                label_source = 'sidecar_expinfo.attend_lr'
                                for t in range(len(trials)):
                                    label_lookup[t] = (expinfo_labels[t], 'expinfo')
                            else:
                                print(f"  ⚠ {subject_id}: No labels from MWF expinfo or sidecar (sidecar_lr={sidecar_lr is not None}, len={len(sidecar_lr) if sidecar_lr is not None else 0}, trials={len(trials)}). Falling back to trial-by-trial.")
                            self._fulsang_sidecar_fallback_logged = True
                        if not label_lookup:
                            print(f"  ⚠ {subject_id}: Falling back to trial-by-trial label extraction")
                            # Debug: show structure of first trial
                            if len(trials) > 0:
                                first_trial = trials[0]
                                print(f"    Inspecting {subject_id}: First trial type: {type(first_trial)}")
                                if hasattr(first_trial, '__dict__'):
                                    print(f"      Attributes: {list(first_trial.__dict__.keys())}")
                                elif isinstance(first_trial, dict):
                                    print(f"      Dict keys: {list(first_trial.keys())}")
                                elif isinstance(first_trial, np.ndarray) and first_trial.dtype.names:
                                    print(f"      Structured array fields: {first_trial.dtype.names}")
                            for trial_idx, trial in enumerate(trials):
                                label_value, found, field_used = extract_label_from_trial(trial, trial_idx, debug=(trial_idx < 3))
                                if found and label_value is not None:
                                    label_lookup[trial_idx] = (label_value, field_used)
                                else:
                                    skipped_trials.append(trial_idx)
                                    if hasattr(trial, 'attention_label'):
                                        raw_val = trial.attention_label
                                        if isinstance(raw_val, np.ndarray) and raw_val.size > 0:
                                            raw_val = raw_val.item() if raw_val.size == 1 else raw_val.flatten()[0]
                                        skipped_values[trial_idx] = raw_val
                                    elif isinstance(trial, dict) and 'attention_label' in trial:
                                        skipped_values[trial_idx] = trial['attention_label']
                            if skipped_trials:
                                unique_skipped = {}
                                for idx in skipped_trials[:20]:
                                    if idx in skipped_values:
                                        val = skipped_values[idx]
                                        if val not in unique_skipped:
                                            unique_skipped[val] = 0
                                        unique_skipped[val] += 1
                                if unique_skipped:
                                    print(f"  {subject_id}: Skipped {len(skipped_trials)} trials. Sample skipped values: {dict(list(unique_skipped.items())[:5])}")
                                else:
                                    print(f"  {subject_id}: Skipped {len(skipped_trials)} trials without extractable labels")
                    
                    # Second pass: process trials and use label lookup with smart fallback (only trials with valid labels)
                    for trial_idx, trial in enumerate(trials[:n_trials_use]):
                        if hasattr(trial, 'eeg_data'):
                            eeg_data = trial.eeg_data
                        elif isinstance(trial, dict):
                            eeg_data = trial.get('eeg_data', None)
                        else:
                            continue
                        
                        if eeg_data is None:
                            continue
                        
                        # Ensure eeg_data is 2D (samples x channels)
                        if len(eeg_data.shape) == 1:
                            eeg_data = eeg_data.reshape(-1, 1)
                        elif len(eeg_data.shape) > 2:
                            eeg_data = eeg_data.reshape(eeg_data.shape[0], -1)
                        
                        # Extract correct EEG channels: first 64 channels only
                        # Fulsang has 64+8 channels (64 EEG + 8 EOG), we need only the first 64 EEG channels
                        n_channels = eeg_data.shape[1]
                        if n_channels > 64:
                            # Take only first 64 channels (EEG channels, excluding EOG)
                            eeg_data = eeg_data[:, :64]
                        elif n_channels < 64:
                            # Pad with zeros if fewer than 64 channels
                            padding = np.zeros((eeg_data.shape[0], 64 - n_channels), dtype=eeg_data.dtype)
                            eeg_data = np.hstack([eeg_data, padding])
                        
                        # Ensure exactly 64 channels
                        if eeg_data.shape[1] != 64:
                            print(f"Warning: Could not normalize to 64 channels for {subject_id} trial {trial_idx}: {eeg_data.shape[1]} channels")
                            continue
                        
                        # Get attention label from lookup - only use extracted labels, skip if not found
                        if trial_idx in label_lookup:
                            label, field_used = label_lookup[trial_idx]
                        else:
                            # Try to extract label again in case it wasn't found in first pass
                            label_value, found, field_used = extract_label_from_trial(trial, trial_idx)
                            if found and label_value is not None:
                                label = label_value
                            else:
                                # Skip this trial - no label found
                                continue
                        
                        # Ensure label is 0 or 1
                        label = 0 if label == 0 else 1
                        
                        # Extract envelopes from Fulsang audio files
                        # Pass subject_id and trial_idx for per-subject mapping; attend_lr/attend_mf_label for left/right from exp_info
                        attend_lr = None
                        if attend_lr_list is not None and trial_idx < len(attend_lr_list):
                            attend_lr = int(attend_lr_list[trial_idx])
                        else:
                            # Per-trial fallback: some MWF files store attend_lr on each trial
                            if hasattr(trial, 'attend_lr'):
                                attend_lr = getattr(trial, 'attend_lr', None)
                            elif isinstance(trial, dict) and 'attend_lr' in trial:
                                attend_lr = trial['attend_lr']
                            if attend_lr is not None:
                                attend_lr = int(attend_lr) if attend_lr in (1, 2) else None
                        left_env, right_env = self._extract_fulsang_envelopes_from_audio(
                            subject_id, trial_idx, eeg_data.shape[0], 
                            trial_data=trial, subject_num=subject_num,
                            attend_lr=attend_lr, attend_mf_label=int(label)
                        )
                        
                        # Diagnostic: Check if envelopes are valid
                        if left_env is not None and right_env is not None:
                            left_env = np.asarray(left_env)
                            right_env = np.asarray(right_env)
                            # Ensure 2D shape
                            if len(left_env.shape) == 1:
                                left_env = left_env.reshape(-1, 1)
                            if len(right_env.shape) == 1:
                                right_env = right_env.reshape(-1, 1)
                            
                            # Check if envelopes have actual data
                            if not (np.any(left_env != 0) and np.any(right_env != 0)):
                                # Envelopes are zeros, log warning
                                if not hasattr(self, '_fulsang_zero_env_count'):
                                    self._fulsang_zero_env_count = 0
                                if self._fulsang_zero_env_count < 3:
                                    print(f"  ⚠️  Fulsang {subject_id} trial {trial_idx}: Extracted envelopes are zeros")
                                    self._fulsang_zero_env_count += 1
                        
                        if left_env is None and right_env is not None:
                            left_env = right_env
                        elif right_env is None and left_env is not None:
                            right_env = left_env
                        if left_env is None or right_env is None:
                            left_env, right_env, _ = self._prepare_trial_envelopes(trial, eeg_data.shape[0], label, dataset_name='Fulsang-MWF')
                        if left_env is not None and right_env is not None:
                            self._real_envelope_frames += eeg_data.shape[0]
                        
                        all_eeg.append(eeg_data)
                        all_labels.append(label)
                        trial_lengths.append(eeg_data.shape[0])
                        all_metadata.append({
                            'subject_id': subject_id,
                            'trial_idx': trial_idx,
                            'dataset': 'Fulsang',
                            'attention_label': label,
                            'preprocessing': 'MWF',
                            'label_source': label_source  # Track where labels came from (raw_eeg, expinfo, event, or trial-by-trial)
                        })
                        all_left_env.append(left_env)
                        all_right_env.append(right_env)
            except Exception as e:
                print(f"Error loading {mwf_file}: {e}")
                continue
        
        if not all_eeg:
            raise ValueError("No valid Fulsang MWF data loaded")
        
        # Verify all Fulsang data has exactly 64 channels (already normalized above)
        channel_counts = [eeg.shape[1] for eeg in all_eeg]
        if len(set(channel_counts)) > 1:
            print(f"Warning: Fulsang data still has inconsistent channels after normalization: {set(channel_counts)}")
            print(f"Normalizing all to 64 channels")
            normalized_eeg = []
            for eeg in all_eeg:
                if eeg.shape[1] < 64:
                    padding = 64 - eeg.shape[1]
                    pad_data = np.zeros((eeg.shape[0], padding), dtype=eeg.dtype)
                    eeg = np.hstack([eeg, pad_data])
                elif eeg.shape[1] > 64:
                    eeg = eeg[:, :64]  # Take first 64 channels
                normalized_eeg.append(eeg)
            all_eeg = normalized_eeg
        elif max(channel_counts) != 64:
            print(f"Warning: Fulsang data has {max(channel_counts)} channels, expected 64")
            print(f"Normalizing all to 64 channels")
            normalized_eeg = []
            for eeg in all_eeg:
                if eeg.shape[1] < 64:
                    padding = 64 - eeg.shape[1]
                    pad_data = np.zeros((eeg.shape[0], padding), dtype=eeg.dtype)
                    eeg = np.hstack([eeg, pad_data])
                elif eeg.shape[1] > 64:
                    eeg = eeg[:, :64]  # Take first 64 channels
                normalized_eeg.append(eeg)
            all_eeg = normalized_eeg
        
        eeg_data = np.vstack(all_eeg)
        labels = np.array(all_labels)
        
        # Validate and report label distribution
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        print(f"\n✓ Fulsang MWF data loaded successfully:")
        print(f"  Total trials: {len(all_eeg)}")
        print(f"  Total samples: {eeg_data.shape[0]}")
        print(f"  Channels: {eeg_data.shape[1]} (normalized to 64)")
        print(f"  Label distribution:")
        for label_val, count in zip(unique_labels, label_counts):
            label_name = "Left (0)" if label_val == 0 else "Right (1)"
            percentage = 100.0 * count / len(labels)
            print(f"    {label_name}: {count} trials ({percentage:.1f}%)")
        
        # Check for label balance
        if len(unique_labels) == 2:
            balance_ratio = min(label_counts) / max(label_counts)
            if balance_ratio < 0.7:
                print(f"  ⚠ Warning: Label imbalance detected (ratio: {balance_ratio:.2f})")
            else:
                print(f"  ✓ Labels are reasonably balanced (ratio: {balance_ratio:.2f})")
        elif len(unique_labels) == 1:
            print(f"  ⚠ Warning: Only one label class found: {unique_labels[0]}")
        
        # Check label source distribution (like FULPRE.py)
        label_sources = {}
        for meta in all_metadata:
            source = meta.get('label_source', 'unknown')
            label_sources[source] = label_sources.get(source, 0) + 1
        
        print(f"\n  Label source distribution:")
        for source, count in sorted(label_sources.items()):
            percentage = 100.0 * count / len(all_metadata) if all_metadata else 0
            print(f"    {source}: {count} trials ({percentage:.1f}%)")
        
        # Warn if using lower-priority sources
        if 'raw_eeg.expinfo.attend_mf' not in label_sources:
            if 'mwf.expinfo.attend_mf' not in label_sources:
                if 'sidecar_expinfo' not in label_sources and 'sidecar_expinfo.attend_lr' not in label_sources:
                    if 'event.eeg.value{1}' not in label_sources:
                        print(f"  ⚠ Warning: No high-priority label sources found!")
                        print(f"     Using trial-by-trial extraction (may be less accurate)")
                        print(f"     To fix: (1) Set FULSANG_RAW_DIR to Data/Fulsang (or path with EEG.zip / EEG/ or S*.mat)")
                        print(f"            (2) Put S*_expinfo.mat in Exp_Info/ (next to Data/Fulsang or in repo root)")
                        print(f"            (3) Or ensure MWF .mat files contain expinfo.attend_mf / attend_lr")
                elif label_sources.get('sidecar_expinfo') or label_sources.get('sidecar_expinfo.attend_lr'):
                    print(f"  ✓ Using sidecar S*_expinfo.mat labels (Exp_Info)")
                else:
                    print(f"  ✓ Using event.eeg.value{{1}} labels (from preprocessing pipeline)")
            else:
                print(f"  ✓ Using MWF expinfo.attend_mf labels")
        else:
            print(f"  ✓ Using raw EEG labels (best quality - expinfo.attend_mf)")
        
        return eeg_data, labels, all_metadata, trial_lengths, all_left_env, all_right_env
    
    def get_window_indices(self, max_windows: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> List[Tuple[int, int, int, str, int, str]]:
        """
        Create sliding windows with labels, ensuring windows don't cross trial boundaries.

        CombinedDataset uses a single window_size (default 512) for both DAS and Fulsang.
        That yields many more windows per trial than DASCCA (512) or FULCCA (1920) alone,
        so total windows can be 10000+ instead of ~5000 + ~2000. Use max_windows to cap,
        or pass a larger window_size when constructing CombinedDataset to reduce count.

        Args:
            max_windows: If set, subsample to at most this many windows (keeps subject
                proportions roughly intact). Use e.g. 7000 to get ~5000 DAS + ~2000 Fulsang.
            rng: Random generator for subsampling when max_windows is set (default: np.random.default_rng(42)).

        Returns:
            List of (start_idx, end_idx, label, subject_id, trial_idx, dataset) tuples.
            Grouping info (subject_id, trial_idx, dataset) enables group-based splitting
            to prevent data leakage from overlapping windows.
        """
        window_indices = []
        step_size = int(self.window_size * (1 - self.overlap))
        if step_size < 1:
            step_size = 1

        # Generate windows per trial to avoid crossing boundaries
        for trial_idx, ((trial_start, trial_end), label) in enumerate(zip(self.trial_boundaries, self.trial_labels)):
            # Get metadata for this trial
            meta = self.trial_meta[trial_idx] if trial_idx < len(self.trial_meta) else {}
            subject_id = meta.get('subject_id', 'unknown')
            dataset = meta.get('dataset', 'unknown')
            
            # Generate windows within this trial (don't cross boundaries)
            for start_idx in range(trial_start, trial_end - self.window_size + 1, step_size):
                end_idx = start_idx + self.window_size
                # Ensure window doesn't exceed trial boundary
                if end_idx <= trial_end:
                    window_indices.append((start_idx, end_idx, label, subject_id, trial_idx, dataset))
        
        # Optionally subsample to cap total windows (preserve subject balance)
        if max_windows is not None and len(window_indices) > max_windows:
            rng = rng if rng is not None else np.random.default_rng(42)
            # Group by subject to keep proportions
            by_subject: Dict[str, List[int]] = {}
            for i, w in enumerate(window_indices):
                sid = w[3]
                if sid not in by_subject:
                    by_subject[sid] = []
                by_subject[sid].append(i)
            chosen = []
            for sid, indices in by_subject.items():
                n_take = max(1, int(round(max_windows * len(indices) / len(window_indices))))
                n_take = min(n_take, len(indices))
                chosen.extend(rng.choice(indices, size=n_take, replace=False).tolist())
            # If we're still over (rounding), trim randomly
            if len(chosen) > max_windows:
                chosen = rng.choice(chosen, size=max_windows, replace=False).tolist()
            window_indices = [window_indices[i] for i in sorted(chosen)]
            print(f"  Subsampled to max_windows={max_windows}: {len(window_indices)} windows (subject proportions preserved)")
        
        return window_indices

    def get_envelope_window(self, start_idx: int, end_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return left/right stimulus envelopes aligned with the EEG window. Shape (window_len, envelope_bands)."""
        n_bands = getattr(self, 'envelope_bands', self.ENVELOPE_BANDS)
        length = end_idx - start_idx
        
        if self.left_envelope_stream is None or self.right_envelope_stream is None:
            left, right = self._fallback_envelopes(length, 0)
            if n_bands > 1 and left.shape[1] == 1:
                left = np.tile(left, (1, n_bands))
                right = np.tile(right, (1, n_bands))
            return left.astype(np.float32), right.astype(np.float32)
        
        if start_idx < 0 or end_idx > len(self.left_envelope_stream):
            left, right = self._fallback_envelopes(length, 0)
            if n_bands > 1 and left.shape[1] == 1:
                left = np.tile(left, (1, n_bands))
                right = np.tile(right, (1, n_bands))
            return left.astype(np.float32), right.astype(np.float32)
        
        left_window = self.left_envelope_stream[start_idx:end_idx]
        right_window = self.right_envelope_stream[start_idx:end_idx]
        return left_window.astype(np.float32), right_window.astype(np.float32)
    
    def validate_timing_synchronization(self, n_samples: int = 50) -> Dict:
        """Validate that envelopes are properly synchronized with EEG data.
        
        Returns:
            Dictionary with validation results including:
            - valid: bool indicating if synchronization is valid
            - status: str describing the status
            - envelope_length: length of envelope stream
            - eeg_length: length of EEG data
            - mismatch_samples: number of samples where length doesn't match
        """
        result = {
            'valid': True,
            'status': 'OK',
            'envelope_length': 0,
            'eeg_length': len(self.eeg_data),
            'mismatch_samples': 0,
            'envelope_coverage': 0.0
        }
        
        if self.left_envelope_stream is None or self.right_envelope_stream is None:
            result['valid'] = False
            result['status'] = 'No envelope streams available'
            return result
        
        result['envelope_length'] = len(self.left_envelope_stream)
        
        # Check length match
        if result['envelope_length'] != result['eeg_length']:
            result['valid'] = False
            result['status'] = f'Length mismatch: EEG={result["eeg_length"]}, Envelopes={result["envelope_length"]}'
            result['mismatch_samples'] = abs(result['envelope_length'] - result['eeg_length'])
            return result
        
        # Sample random windows to check alignment
        np.random.seed(42)
        sample_indices = np.random.choice(
            max(1, len(self.eeg_data) - self.window_size), 
            min(n_samples, max(1, len(self.eeg_data) - self.window_size)), 
            replace=False
        )
        
        alignment_issues = 0
        for idx in sample_indices:
            start_idx = int(idx)
            end_idx = start_idx + self.window_size
            
            try:
                left_env, right_env = self.get_envelope_window(start_idx, end_idx)
                if left_env.shape[0] != self.window_size or right_env.shape[0] != self.window_size:
                    alignment_issues += 1
            except Exception:
                alignment_issues += 1
        
        if alignment_issues > 0:
            result['valid'] = False
            result['status'] = f'Alignment issues in {alignment_issues}/{len(sample_indices)} sampled windows'
        
        # Calculate envelope coverage
        if hasattr(self, '_real_envelope_frames') and hasattr(self, '_total_frames'):
            result['envelope_coverage'] = (self._real_envelope_frames / max(1, self._total_frames)) * 100.0
        
        return result


if __name__ == '__main__':
    # Test the dataset
    dataset = CombinedDataset()
    print(f"\n✓ Dataset created successfully")
    print(f"  Total windows: {len(dataset.get_window_indices())}")

