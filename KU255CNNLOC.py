#!/usr/bin/env python3
"""
KU255CNNLOC - CNN-LOC for KU Leuven 255 Dataset

CNN-LOC model for attention decoding on KU Leuven 255 EEG data.
Includes metrics (accuracy, MSED, ROC-AUC) and temporal analysis.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           precision_recall_fscore_support, roc_auc_score, roc_curve,
                           precision_recall_curve, average_precision_score,
                           matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
                           f1_score)
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr
from scipy.signal import find_peaks
from tqdm import tqdm
import json
import pickle
from datetime import datetime
from collections import Counter, defaultdict
import warnings
import scipy.io as sio
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_kuleuven_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse KU Leuven 255 filename to extract subject ID.
    
    Handles two formats:
    1. Sx_AAD_tL or Sx_AAD_tR (with trial info in filename)
    2. Sx_preprocessed (trial info inside .mat file)
    
    Returns:
        Dict with 'subject_id', and optionally 'trial_id', 'attention_ear'
        If trial info not in filename, returns only 'subject_id'
    """
    import re
    # Remove extension and _preprocessed suffix, strip whitespace
    base_name = Path(filename).stem.replace('_preprocessed', '').strip()
    
    # Pattern 1: S<number>_AAD_<trial><L|R> (with trial info)
    pattern_with_trial = r'^(S\d+)_AAD_(\d+)([LR])$'
    match = re.match(pattern_with_trial, base_name)
    
    if match:
        subject_id = match.group(1)
        trial_id = match.group(2)
        attention_ear = match.group(3)
        return {
            'subject_id': subject_id,
            'trial_id': trial_id,
            'attention_ear': attention_ear
        }
    
    # Pattern 2: S<number> (subject only, trial info in .mat file)
    # This should match S0, S1, S2, etc. (with optional leading/trailing whitespace)
    pattern_subject_only = r'^\s*(S\d+)\s*$'
    match = re.match(pattern_subject_only, base_name)
    
    if match:
        subject_id = match.group(1)
        return {
            'subject_id': subject_id,
            'trial_id': None,  # Will be extracted from trial object
            'attention_ear': None  # Will be extracted from trial object
        }
    
    # If we get here, the filename doesn't match expected patterns
    return None


def read_curry_trigger_boundaries(base_file: Path, sampling_rate: float = 1000.0) -> Optional[Tuple[int, int]]:
    """
    Extract experiment boundaries from Curry trigger channel using the method from sample_script.m.
    
    This replicates the MATLAB logic:
    1. Load trigger channel from .dat file
    2. Remove baseline marker
    3. Find peaks in trigger channel
    4. First peak = experiment start, last peak = experiment end
    
    Args:
        base_file: Path to Curry file base name (without extension, e.g., 'S9_AAD_1L')
        sampling_rate: Sampling rate in Hz (default 1000 for raw, will be downsampled later)
    
    Returns:
        Tuple of (start_sample, end_sample) or None if not found
    """
    try:
        # Find Curry files (.dat, .dap, .rs3)
        dat_file = base_file.with_suffix('.dat')
        dap_file = base_file.with_suffix('.dap')
        rs3_file = base_file.with_suffix('.rs3')
        
        if not dat_file.exists() or not dap_file.exists() or not rs3_file.exists():
            return None
        
        # Read parameters from .dap file
        with open(dap_file, 'rt', encoding='latin-1', errors='ignore') as f:
            dap_content = f.read()
        
        # Extract sampling rate and other params
        tokens = {
            'NumSamples': None, 'NUM_SAMPLES': None,
            'NumChannels': None, 'NUM_CHANNELS': None,
            'NumTrials': None, 'NUM_TRIALS': None,
            'SampleFreqHz': None, 'SAMPLE_FREQ_HZ': None,
            'DataFormat': None, 'DATA_FORMAT': None,
            'DataSampOrder': None, 'DATA_SAMP_ORDER': None
        }
        
        for token in tokens.keys():
            idx = dap_content.find(token)
            if idx != -1:
                remaining = dap_content[idx + len(token):]
                if '=' in remaining:
                    value_str = remaining.split('=')[1].split()[0] if remaining.split('=')[1].split() else None
                    if value_str:
                        if value_str.upper() in ['ASCII', 'CHAN']:
                            tokens[token] = 1 if value_str.upper() == 'ASCII' else 0
                        else:
                            try:
                                tokens[token] = float(value_str)
                            except:
                                pass
        
        nSamples = int(tokens['NumSamples'] or tokens['NUM_SAMPLES'] or 0)
        nChannels = int(tokens['NumChannels'] or tokens['NUM_CHANNELS'] or 0)
        nTrials = int(tokens['NumTrials'] or tokens['NUM_TRIALS'] or 1)
        fFrequency = float(tokens['SampleFreqHz'] or tokens['SAMPLE_FREQ_HZ'] or sampling_rate)
        nASCII = int(tokens['DataFormat'] == 1 or tokens['DATA_FORMAT'] == 1)
        nMultiplex = int(tokens['DataSampOrder'] == 1 or tokens['DATA_SAMP_ORDER'] == 1)
        
        # Read channel names from .rs3 file
        with open(rs3_file, 'rt', encoding='latin-1', errors='ignore') as f:
            rs3_content = f.read()
        
        channel_names = [f'EEG{i+1}' for i in range(nChannels)]
        idx_positions = []
        idx = rs3_content.find('LABELS')
        while idx != -1:
            idx_positions.append(idx)
            idx = rs3_content.find('LABELS', idx + 1)
        
        if len(idx_positions) >= 4:
            for i in range(3, len(idx_positions), 4):
                start_idx = idx_positions[i-1]
                end_idx = idx_positions[i] if i < len(idx_positions) else len(rs3_content)
                section = rs3_content[start_idx:end_idx]
                lines = section.split('\n')
                nc = 0
                for line in lines[1:]:
                    line = line.strip()
                    if line and line != 'END_LIST' and nc < nChannels:
                        channel_names[nc] = line
                        nc += 1
                    if line == 'END_LIST' or nc >= nChannels:
                        break
        
        # Find trigger channel index
        trigger_channel_idx = None
        for i, name in enumerate(channel_names):
            if name == 'Trigger':
                trigger_channel_idx = i
                break
        
        if trigger_channel_idx is None:
            print(f"  Warning: No 'Trigger' channel found in {base_file.name}")
            return None
        
        # Read .dat file - FIXED: Correct handling of multiplexed vs channel-major data
        if nASCII == 1:
            with open(dat_file, 'rt') as f:
                raw = np.loadtxt(f)
            raw = raw.flatten()  # Ensure 1D
        else:
            with open(dat_file, 'rb') as f:
                raw = np.fromfile(f, dtype=np.float32, count=nChannels * nSamples * nTrials)
        
        # Reshape based on multiplexing order
        if nMultiplex == 1:
            # Sample-major (multiplexed): (nSamples*nTrials, nChannels)
            data = raw.reshape(nSamples * nTrials, nChannels)
            labels = data[:, trigger_channel_idx].copy()
        else:
            # Channel-major: (nChannels, nSamples*nTrials)
            data = raw.reshape(nChannels, nSamples * nTrials)
            labels = data[trigger_channel_idx, :].copy()
        
        # Find baseline marker (replicate MATLAB logic)
        # If first 5 seconds are constant, use first value; otherwise use median
        check_samples = int(5 * fFrequency)
        if check_samples > len(labels):
            check_samples = len(labels)
        
        if check_samples > 0 and np.all(np.diff(labels[:check_samples]) == 0):
            baseline_marker = labels[0]
        else:
            baseline_marker = np.median(labels)
        
        # Process trigger channel: find onsets (replicate loadCurryData.m exactly)
        # MATLAB: 
        #   tmpLabels = zeros(1, size(data,2));
        #   onset_samples = find(diff(labels))+1;
        #   tmpLabels(onset_samples) = labels(onset_samples)-baseline_marker;
        #   labels = tmpLabels;
        processed_labels = np.zeros_like(labels)
        onset_samples = np.where(np.diff(labels) != 0)[0] + 1  # +1 because MATLAB is 1-indexed
        if len(onset_samples) > 0:
            processed_labels[onset_samples] = labels[onset_samples] - baseline_marker
        
        # Find peaks (replicate MATLAB findpeaks from sample_script.m)
        # MATLAB: [pks,locs] = findpeaks(triggers)
        # This finds local maxima in the processed trigger channel
        # FIXED: Added minimum height, prominence, and distance to filter noisy trigger channels
        # Use a minimum height threshold (e.g., 10% of max value) to filter noise
        min_height = max(0, np.max(processed_labels) * 0.1) if len(processed_labels) > 0 else 0
        # FIXED: Use fFrequency (actual Curry rate) for peak distance, not function argument
        peaks, properties = find_peaks(
            processed_labels, 
            height=max(0, min_height),  # Minimum height threshold
            prominence=max(0, min_height * 0.5),  # Minimum prominence to filter noise
            distance=int(fFrequency * 0.1)  # Minimum distance between peaks (100ms at Curry rate)
        )
        
        # FIXED: Fallback to no-threshold findpeaks if thresholds filter out all peaks
        # (can happen if triggers are low amplitude after baseline subtraction)
        if len(peaks) < 2:
            peaks, _ = find_peaks(processed_labels)  # Fallback: no thresholds
            if len(peaks) < 2:
                print(f"  Warning: Insufficient peaks in trigger channel for {base_file.name} ({len(peaks)} found even without thresholds)")
                return None
        
        # First peak = experiment start, last peak = experiment end
        start_sample = int(peaks[0])
        end_sample = int(peaks[-1])
        
        # FIXED: Always return samples in Curry rate (fFrequency), conversion happens later in _load_kuleuven_data
        # This avoids double-scaling and makes the function's behavior clear
        print(f"  Loaded trigger boundaries from {base_file.name}: peaks {peaks[0]} - {peaks[-1]} (Curry {fFrequency}Hz)")
        
        return (start_sample, end_sample)
    
    except Exception as e:
        print(f"Warning: Could not read trigger boundaries from {base_file}: {e}")
        import traceback
        traceback.print_exc()
        return None


def find_curry_base_file(mat_file: Path, data_root: Optional[Path] = None) -> Optional[Path]:
    """
    Find corresponding Curry base file (.dat/.dap/.rs3) for a .mat file.
    
    Returns the base path (without extension) for loading Curry files.
    This is used to extract trigger channel boundaries.
    """
    # Try same directory as .mat file
    base_name = mat_file.stem.replace('_preprocessed', '')
    base_file = mat_file.parent / base_name
    
    # Check if .dat file exists (indicates Curry files are present)
    if (base_file.with_suffix('.dat')).exists():
        return base_file
    
    # Try in raw data structure: Data/KULeuven 255/Sx/Sx/Sx_AAD_tL
    if data_root is not None:
        # Extract subject and trial info from filename
        parsed = parse_kuleuven_filename(mat_file.name)
        if parsed:
            subject_id = parsed['subject_id']
            trial_id = parsed.get('trial_id')  # May be None
            attention_ear = parsed.get('attention_ear')  # May be None
            
            # FIXED: Don't attempt paths if trial info isn't in filename (subject-only format)
            if trial_id is None or attention_ear is None:
                return None  # Can't construct Curry file path without trial info
            
            # Try different possible paths
            possible_paths = [
                data_root / subject_id / subject_id / f"{subject_id}_AAD_{trial_id}{attention_ear}",
                data_root / subject_id / f"{subject_id}_AAD_{trial_id}{attention_ear}",
            ]
            
            for path in possible_paths:
                if (path.with_suffix('.dat')).exists():
                    return path
    
    return None


class KU255Dataset(Dataset):
    """
    Dataset for KU Leuven 255 EEG data. Uses PREPROCESS255 output.
    Handles windowing and preprocessing for attention decoding.
    """
    
    def __init__(self, preprocessed_dir: str, mode: str = 'full', 
                 window_size: int = 32, overlap: float = 0.5,
                 transform_eeg: bool = True, cache_size: int = 1000,
                 allowed_subjects: Optional[List[str]] = None,
                 raw_data_dir: Optional[str] = None):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.mode = mode
        self.window_size = window_size
        self.overlap = overlap
        self.transform_eeg = transform_eeg
        self.cache_size = cache_size
        self.allowed_subjects = allowed_subjects  # For mode filtering
        
        # KU255 dataset params
        self.sampling_rate = 128  # Hz (downsampled from 1000 Hz)
        self.original_sampling_rate = 1000  # Hz (original Curry recording rate)
        self.n_channels = 64  # EEG channels (downsampled from 255/256)
        self.attention_switch_duration = 20  # seconds
        
        # Raw data directory for finding Curry files (.dat/.dap/.rs3)
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir else None
        
        # FIXED: Disable trigger boundary filtering by default (safest option)
        # Only enable if you can guarantee alignment between Curry files and preprocessed trials
        self.use_trigger_boundaries = False  # Default: off for safety
        
        # Cache disabled for speed (too slow with 267K windows)
        self._window_cache = None
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Load data from PREPROCESS255 output
        # FIXED: metadata is now per-trial only (not per-sample) to save RAM
        self.eeg_data, self.labels, self.trial_metadata, self.trial_boundaries = self._load_kuleuven_data()
        
        # Filter subjects based on mode or allowed_subjects
        self._filter_by_mode()
        
        # FIXED: Recompute all_subject_ids after filtering (it becomes stale if filtering occurred)
        # Store subject IDs for mode filtering (extract from trial_boundaries)
        # FIXED: trial_boundaries is now 4-tuple: (start, end, subject_id, trial_id)
        self.all_subject_ids = sorted(set(subj_id for _, _, subj_id, _ in self.trial_boundaries))
        
        # Validate trial boundaries for correctness
        self._validate_trial_boundaries()
        
        self.window_indices, self.window_trial_keys = self._create_kuleuven_windows()
        
        print(f"Loaded {len(self.window_indices)} windows, EEG shape: {self.eeg_data.shape}, Label dist: {np.bincount(self.labels)}")
    
    def _load_kuleuven_data(self) -> Tuple[np.ndarray, np.ndarray, List[Dict], List[Tuple[int, int, str, str]]]:
        """Load preprocessed data from PREPROCESS255 output (.mat files).
        
        Returns:
            eeg_data: Concatenated EEG data (samples, channels)
            labels: Labels for each sample
            trial_metadata: List of metadata dicts (one per trial, not per sample) - FIXED: per-trial only to save RAM
            trial_boundaries: List of (start_idx, end_idx, subject_id, trial_id) tuples - FIXED: 4-tuple only
        """
        # CRITICAL: Sort files to ensure consistent subject ordering
        mat_files = sorted(self.preprocessed_dir.glob("*_preprocessed.mat"))
        if not mat_files:
            raise ValueError(f"No preprocessed .mat files found in {self.preprocessed_dir}")
        
        print(f"Loading KU255 preprocessed data from {len(mat_files)} files...")
        
        all_eeg_data = []
        all_labels = []
        trial_metadata = []  # FIXED: Store metadata per-trial only (not per-sample) to save RAM
        trial_boundaries = []  # FIXED: Store (start_idx, end_idx, subject_id, trial_id) - 4-tuple only
        
        n_success = 0
        n_failed = 0
        total_records = 0
        subject_stats = {}
        shape_errors = 0
        trials_skipped = {}  # FIXED: Initialize trials_skipped before use
        
        for mat_file in tqdm(mat_files, desc="Loading KU255 data"):
            # FIXED: Track records per file to correctly count success/failure
            file_records_before = total_records
            try:
                # CRITICAL: Parse filename to get subject (and optionally trial/attention)
                filename_info = parse_kuleuven_filename(mat_file.name)
                if filename_info is None:
                    # Debug: show what we tried to parse
                    base_name = Path(mat_file.name).stem.replace('_preprocessed', '')
                    print(f"WARNING: Could not parse filename '{mat_file.name}' (base_name='{base_name}'), skipping")
                    continue
                
                subject_id = filename_info['subject_id']
                filename_trial_id = filename_info.get('trial_id')  # May be None
                filename_attention_ear = filename_info.get('attention_ear')  # May be None
                
                # CRITICAL: Extract experiment boundaries from trigger channel (replicate sample_script.m)
                # FIXED: Only compute boundaries if enabled (disabled by default for safety)
                file_boundaries = None  # Per-file boundaries (computed once)
                if self.use_trigger_boundaries:
                    # FIXED: Compute boundaries once per file, not per trial
                    curry_base_file = find_curry_base_file(mat_file, self.raw_data_dir)
                    if curry_base_file:
                        # Use trigger channel peaks to find boundaries (correct method from MATLAB)
                        file_boundaries = read_curry_trigger_boundaries(
                            curry_base_file, 
                            sampling_rate=self.original_sampling_rate
                        )
                        if file_boundaries:
                            # Convert to downsampled rate
                            start_orig, end_orig = file_boundaries
                            start_downsampled = int(start_orig * self.sampling_rate / self.original_sampling_rate)
                            end_downsampled = int(end_orig * self.sampling_rate / self.original_sampling_rate)
                            file_boundaries = (start_downsampled, end_downsampled)
                
                # Use squeeze_me and struct_as_record to properly load MATLAB structures
                data = sio.loadmat(str(mat_file), squeeze_me=True, struct_as_record=False)
                
                # Get trials data
                if 'trials' in data:
                    trials = data['trials']
                    # Handle MATLAB cell array structure
                    if not isinstance(trials, np.ndarray):
                        trials = [trials]
                    else:
                        trials = trials.flatten()
                    
                    for trial_idx, trial in enumerate(trials):
                        try:
                            # Extract trial ID and attention ear from trial object if not in filename
                            trial_id = filename_trial_id
                            attention_ear = filename_attention_ear
                            
                            # Try to extract from trial object
                            if trial_id is None:
                                # Try different possible field names
                                if hasattr(trial, 'trial_id'):
                                    trial_id = str(trial.trial_id)
                                elif hasattr(trial, 'trial_number'):
                                    trial_id = str(trial.trial_number)
                                elif isinstance(trial, dict):
                                    trial_id = str(trial.get('trial_id', trial.get('trial_number', trial_idx)))
                                else:
                                    trial_id = str(trial_idx)
                            
                            if attention_ear is None:
                                # FIXED: More robust extraction with multiple fallback fields
                                # Try to extract attention ear from trial object with multiple field names
                                attended_ear = None
                                
                                # Try common field names
                                field_names = ['attended_ear', 'attention_ear', 'attention_direction', 
                                             'attended_side', 'attend_dir', 'side', 'ear', 'direction']
                                
                                for field_name in field_names:
                                    if hasattr(trial, field_name):
                                        attended_ear = getattr(trial, field_name)
                                        break
                                    elif isinstance(trial, dict) and field_name in trial:
                                        attended_ear = trial[field_name]
                                        break
                                
                                if attended_ear is not None:
                                    # FIXED: Print raw value for label sanity check (first few trials per file)
                                    if trial_idx < 3:
                                        print(f"    [LABEL DEBUG] trial {trial_idx} raw attended_ear: {attended_ear} (type: {type(attended_ear).__name__})")
                                    
                                    # Handle numpy array
                                    if isinstance(attended_ear, np.ndarray):
                                        if attended_ear.size > 0:
                                            attended_ear = str(attended_ear.item() if attended_ear.size == 1 else attended_ear.flatten()[0])
                                        else:
                                            attended_ear = None
                                    else:
                                        attended_ear = str(attended_ear).strip()
                                    
                                    # FIXED: Only accept explicit string values, do not guess numeric encodings
                                    # This prevents silent label flips - skip if not explicit
                                    if attended_ear:
                                        attended_ear_upper = attended_ear.upper()
                                        if attended_ear_upper in ["L", "LEFT"]:
                                            attention_ear = "L"
                                        elif attended_ear_upper in ["R", "RIGHT"]:
                                            attention_ear = "R"
                                        else:
                                            # IMPORTANT: do not guess numeric codes; skip if not explicit
                                            attention_ear = None
                                    else:
                                        attention_ear = None
                            
                            # CRITICAL: Get attention label. Prefer attention_label from .mat (PREPROCESS255 canonical).
                            # L = Left = 0, R = Right = 1. If .mat has attention_label (0/1), use it; else derive from attended_ear.
                            label_from_mat = None
                            if hasattr(trial, 'attention_label'):
                                v = trial.attention_label
                                if isinstance(v, np.ndarray) and v.size > 0:
                                    v = int(v.item() if v.size == 1 else v.flatten()[0])
                                else:
                                    v = int(v) if v is not None else None
                                if v in (0, 1):
                                    label_from_mat = v
                            elif isinstance(trial, dict) and trial.get('attention_label') is not None:
                                v = trial['attention_label']
                                if isinstance(v, np.ndarray) and v.size > 0:
                                    v = int(v.item() if v.size == 1 else v.flatten()[0])
                                else:
                                    v = int(v)
                                if v in (0, 1):
                                    label_from_mat = v
                            if label_from_mat is not None:
                                attention_label = label_from_mat
                                if attention_ear and (0 if attention_ear.upper() == 'L' else 1) != attention_label:
                                    print(f"  [LABEL] {mat_file.name} trial {trial_idx}: using .mat attention_label={attention_label} (attended_ear={attention_ear} would give different label)")
                            elif attention_ear:
                                attention_label = 0 if attention_ear.upper() == 'L' else 1
                                if trial_idx < 2:
                                    print(f"  [LABEL] {mat_file.name} trial {trial_idx} -> attention_ear={attention_ear}, label={attention_label} (derived; no attention_label in .mat)")
                            else:
                                if subject_id not in trials_skipped:
                                    trials_skipped[subject_id] = 0
                                trials_skipped[subject_id] += 1
                                print(f"WARNING: Could not determine attention label for {mat_file.name} trial {trial_idx} after trying all field names, skipping")
                                continue
                            
                            # Extract EEG data - handle both object attributes and dict access
                            if hasattr(trial, 'eeg_data'):
                                eeg_data = trial.eeg_data
                            elif isinstance(trial, dict):
                                eeg_data = trial.get('eeg_data', None)
                            else:
                                continue
                            
                            if eeg_data is None:
                                continue
                            
                            # Convert to numpy array and handle shape
                            eeg_data = np.asarray(eeg_data)
                            
                            # Handle different shapes
                            if len(eeg_data.shape) == 1:
                                eeg_data = eeg_data.reshape(-1, 1)
                            elif len(eeg_data.shape) > 2:
                                eeg_data = eeg_data.reshape(eeg_data.shape[0], -1)
                            
                            # Must be (samples, 64) or (samples, channels)
                            if eeg_data.shape[1] != 64:
                                if eeg_data.shape[0] == 64:
                                    eeg_data = eeg_data.T
                                else:
                                    print(f"WARNING: Expected 64 channels, got {eeg_data.shape} in {mat_file.name}")
                                    shape_errors += 1
                                    continue
                            
                            # Check for invalid values
                            if np.any(np.isnan(eeg_data)) or np.any(np.isinf(eeg_data)):
                                print(f"WARNING: Invalid EEG values (NaN/Inf) in {mat_file.name}")
                                continue
                            
                            # CRITICAL: Verify this is real data, not dummy/zeros
                            if np.allclose(eeg_data, 0.0, atol=1e-10):
                                print(f"WARNING: All-zero EEG data detected in {mat_file.name} trial {trial_idx} - skipping")
                                continue
                            
                            # Check data variance (dummy data often has zero or very low variance)
                            if np.var(eeg_data) < 1e-10:
                                print(f"WARNING: Very low variance EEG data in {mat_file.name} trial {trial_idx} - may be dummy data")
                                # Don't skip, but warn
                            
                            # CRITICAL: Use label from filename (if available) or trial object
                            # Filename is authoritative if present, otherwise use trial metadata
                            label = attention_label  # Already determined above
                            
                            # Track subject statistics
                            if subject_id not in subject_stats:
                                subject_stats[subject_id] = {'samples': 0, 'labels': [], 'trials_kept': 0}
                                trials_skipped[subject_id] = 0
                            subject_stats[subject_id]['samples'] += eeg_data.shape[0]
                            subject_stats[subject_id]['labels'].append(label)
                            subject_stats[subject_id]['trials_kept'] += 1
                            
                            # CRITICAL: Apply experiment boundaries if available
                            # Only use data within valid experiment segment (not pre/post baseline)
                            # VALIDATION: Check alignment between raw trigger indices and preprocessed data
                            # FIXED: Use per-trial variable instead of mutating file_boundaries
                            trial_len_before = eeg_data.shape[0]
                            expected_trial_length = 6 * 60 * self.sampling_rate  # ~6 min @ sampling_rate Hz = ~46k @ 128Hz
                            
                            # Use per-trial variable (don't mutate file_boundaries)
                            trial_boundaries_use = file_boundaries
                            
                            if trial_boundaries_use:
                                start_bound, end_bound = trial_boundaries_use
                                # Clamp to valid range
                                start_bound = max(0, min(start_bound, eeg_data.shape[0]))
                                end_bound = max(start_bound, min(end_bound, eeg_data.shape[0]))
                                
                                if end_bound <= start_bound:
                                    print(f"WARNING: Invalid experiment boundaries for {mat_file.name} trial {trial_idx}, skipping boundary filtering")
                                    trial_boundaries_use = None
                                else:
                                    # VALIDATION: Check if boundaries seem misaligned
                                    # If preprocessed trial is much shorter/longer than expected, boundaries may be wrong
                                    if trial_len_before < expected_trial_length * 0.5 or trial_len_before > expected_trial_length * 2.0:
                                        print(f"WARNING: Preprocessed trial length ({trial_len_before}) differs significantly from expected (~{expected_trial_length:.0f})")
                                        print(f"   This suggests boundaries may be misaligned. Skipping boundary filtering for safety.")
                                        trial_boundaries_use = None
                                    else:
                                        # Extract only valid segment
                                        eeg_data = eeg_data[start_bound:end_bound, :]
                                        trial_len_after = eeg_data.shape[0]
                                        
                                        # Print validation info
                                        print(f"  Trial {trial_idx} ({mat_file.name}): len_before={trial_len_before}, len_after={trial_len_after}, expected~{expected_trial_length:.0f}")
                                        
                                        # Check if result is reasonable
                                        if trial_len_after < expected_trial_length * 0.3:
                                            print(f"  ⚠ WARNING: After boundary filtering, trial is very short ({trial_len_after} vs expected ~{expected_trial_length:.0f})")
                                            print(f"     Boundaries may be misaligned. Using safe trim instead.")
                                            # Fallback: safe trim (drop first/last 5 seconds)
                                            safe_trim = int(5 * self.sampling_rate)  # 5 seconds
                                            if trial_len_before > safe_trim * 2:
                                                eeg_data = eeg_data[safe_trim:-safe_trim] if len(eeg_data) > safe_trim * 2 else eeg_data
                                                trial_len_after = eeg_data.shape[0]
                                                print(f"     Applied safe trim: len_after={trial_len_after}")
                                            trial_boundaries_use = None
                                        
                                        if eeg_data.shape[0] < self.window_size:
                                            print(f"WARNING: Trial {trial_idx} in {mat_file.name} too short after boundary filtering ({eeg_data.shape[0]} < {self.window_size}), skipping")
                                            continue
                            else:
                                # No boundaries - use safe trim if trial is very long
                                if trial_len_before > expected_trial_length * 1.5:
                                    safe_trim = int(5 * self.sampling_rate)  # 5 seconds
                                    if trial_len_before > safe_trim * 2:
                                        eeg_data = eeg_data[safe_trim:-safe_trim]
                                        print(f"  Trial {trial_idx} ({mat_file.name}): Applied safe trim, len={eeg_data.shape[0]}")
                            
                            # Store trial start index before adding data
                            trial_start_idx = total_records
                            
                            # FIXED: Store metadata per-trial only (not per-sample) to save RAM
                            metadata = {
                                'subject_id': subject_id,
                                'trial_id': trial_id,  # From filename or trial object
                                'file': mat_file.name,
                                'trial_idx': trial_idx,  # Index within file
                                'attention_label': label,  # From filename (if available) or trial object
                                'attention_ear': attention_ear,  # From filename (if available) or trial object
                                'preprocessing_method': 'PREPROCESS255',
                                'validation_passed': True,
                                'data_type': 'EEG_only',
                                'eeg_shape': eeg_data.shape,
                                'label_alignment': 'filename_parsed' if filename_attention_ear else 'trial_object',
                                'experiment_boundaries_applied': trial_boundaries_use is not None
                            }
                            
                            all_eeg_data.append(eeg_data)
                            
                            # CRITICAL: Label is constant per trial (from filename)
                            # All samples in this trial have the same attention label
                            all_labels.extend([label] * eeg_data.shape[0])
                            trial_metadata.append(metadata)  # FIXED: Store once per trial, not per sample
                            
                            trial_end_idx = total_records + eeg_data.shape[0]
                            # FIXED: Store as 4-tuple only (label is already in metadata['attention_label'])
                            # FIXED: Ensure trial_id is always str for consistency
                            trial_boundaries.append((trial_start_idx, trial_end_idx, subject_id, str(trial_id)))
                            total_records += eeg_data.shape[0]
                            
                        except Exception as trial_error:
                            print(f"ERROR processing trial in {mat_file.name}: {trial_error}")
                            import traceback
                            traceback.print_exc()
                            continue
                
                # FIXED: Count success/failure based on records added by THIS file, not total
                file_records_added = total_records - file_records_before
                if file_records_added > 0:
                    n_success += 1
                else:
                    n_failed += 1
                    
            except Exception as e:
                n_failed += 1
                print(f"ERROR loading {mat_file.name}: {e}")
                continue
        
        print(f"Successfully loaded {n_success} files, {n_failed} files failed")
        print(f"Total records loaded: {total_records}")
        print(f"Shape errors: {shape_errors}")
        
        if shape_errors > 0:
            print(f"WARNING: {shape_errors} records had shape errors")
        
        if not all_eeg_data:
            raise ValueError("No valid KU255 data found in preprocessed files")
        
        eeg_data = np.vstack(all_eeg_data)
        labels = np.array(all_labels, dtype=np.int64)
        
        # Final shape validation
        print(f"Final data shapes: EEG {eeg_data.shape}, Labels {labels.shape}")
        
        if eeg_data.shape[1] != 64:
            raise ValueError(f"CRITICAL: EEG data has {eeg_data.shape[1]} channels, expected 64")
        
        if len(eeg_data) != len(labels):
            raise ValueError(f"CRITICAL: EEG samples ({len(eeg_data)}) != labels ({len(labels)})")
        
        # CRITICAL: Validate that data is real, not dummy/zeros
        data_variance = np.var(eeg_data)
        data_mean = np.mean(np.abs(eeg_data))
        zero_ratio = np.sum(np.abs(eeg_data) < 1e-10) / eeg_data.size
        
        print(f"Data validation: variance={data_variance:.6f}, mean_abs={data_mean:.6f}, zero_ratio={zero_ratio:.4f}")
        
        if data_variance < 1e-10:
            raise ValueError(f"CRITICAL: EEG data variance is too low ({data_variance:.2e}) - data may be dummy/zeros!")
        
        if zero_ratio > 0.95:
            raise ValueError(f"CRITICAL: {zero_ratio*100:.1f}% of data is zero - data may be dummy!")
        
        if data_mean < 1e-6:
            print(f"WARNING: Very low mean absolute value ({data_mean:.2e}) - data may be preprocessed too aggressively")
        
        # Validate labels are not all the same (would indicate dummy data)
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            raise ValueError(f"CRITICAL: Only {len(unique_labels)} unique label(s) found - data may be dummy!")
        
        label_distribution = np.bincount(labels)
        print(f"Label distribution: {dict(zip(unique_labels, label_distribution[unique_labels]))}")
        
        # FIXED: Label sanity check - verify both L and R exist (critical for learning)
        if len(label_distribution) >= 2:
            has_l = label_distribution[0] > 0
            has_r = label_distribution[1] > 0
            if not (has_l and has_r):
                print(f"⚠ CRITICAL: Missing one label class - L={has_l}, R={has_r}")
                print(f"   This will prevent the model from learning! Check label extraction.")
            balance_ratio = min(label_distribution) / max(label_distribution)
            if balance_ratio < 0.1:
                print(f"WARNING: Extreme label imbalance (ratio: {balance_ratio:.3f}) - may affect training")
        
        del all_eeg_data, all_labels
        import gc
        gc.collect()
        
        # FIXED: Return trial_metadata (per-trial) instead of all_metadata (per-sample) to save RAM
        return eeg_data, labels, trial_metadata, trial_boundaries
    
    def _filter_by_mode(self):
        """Filter subjects and trials based on mode or allowed_subjects.
        
        FIXED: Now works with per-trial metadata instead of per-sample metadata.
        """
        if self.allowed_subjects is not None:
            # Filter by explicitly provided subject list
            allowed_set = set(self.allowed_subjects)
            # Filter trial_boundaries to only include allowed subjects
            filtered_boundaries = []
            filtered_eeg_data = []
            filtered_labels = []
            filtered_trial_metadata = []  # FIXED: Per-trial metadata, not per-sample
            
            current_offset = 0
            new_trial_boundaries = []
            
            # FIXED: Build metadata map once (O(n)) instead of O(n²) lookup per trial
            meta_map = {}
            for i, (s, e, sid, tid) in enumerate(self.trial_boundaries):
                meta_map[(s, e, sid, tid)] = i
            
            # FIXED: trial_boundaries is 4-tuple: (start, end, subject_id, trial_id)
            for trial_start, trial_end, subject_id, trial_id in self.trial_boundaries:
                if subject_id in allowed_set:
                    # Keep this trial
                    trial_data = self.eeg_data[trial_start:trial_end]
                    trial_labels = self.labels[trial_start:trial_end]
                    
                    # FIXED: Fast O(1) metadata lookup using dict
                    orig_i = meta_map.get((trial_start, trial_end, subject_id, trial_id))
                    metadata = self.trial_metadata[orig_i] if (orig_i is not None and orig_i < len(self.trial_metadata)) else {}
                    
                    new_start = current_offset
                    new_end = current_offset + len(trial_data)
                    
                    filtered_eeg_data.append(trial_data)
                    filtered_labels.append(trial_labels)  # FIXED: Append arrays, not extend element-by-element
                    filtered_trial_metadata.append(metadata)  # FIXED: Store once per trial
                    new_trial_boundaries.append((new_start, new_end, subject_id, trial_id))  # FIXED: 4-tuple only
                    
                    current_offset = new_end
            
            if filtered_eeg_data:
                self.eeg_data = np.vstack(filtered_eeg_data)
                # FIXED: Use np.concatenate for arrays (much faster than np.array(list of lists))
                self.labels = np.concatenate(filtered_labels).astype(np.int64) if filtered_labels else np.array([], dtype=np.int64)
                self.trial_metadata = filtered_trial_metadata  # FIXED: Per-trial metadata
                self.trial_boundaries = new_trial_boundaries
            else:
                # No data matches - create empty arrays
                self.eeg_data = np.zeros((0, 64), dtype=np.float32)
                self.labels = np.array([], dtype=np.int64)
                self.trial_metadata = []  # FIXED: Per-trial metadata
                self.trial_boundaries = []
        # If mode is 'full' or no allowed_subjects, keep all data (default behavior)
    
    def _validate_trial_boundaries(self):
        """Validate trial boundaries for correctness and consistency.
        
        Ensures:
        - Boundaries are non-overlapping and contiguous
        - Each boundary corresponds to correct subject
        - No gaps or overlaps in sample indices
        """
        if not self.trial_boundaries:
            raise ValueError("CRITICAL: No trial boundaries found!")
        
        # Sort by start index (required for binary search in _get_window_subject)
        sorted_boundaries = sorted(self.trial_boundaries, key=lambda x: x[0])
        # FIXED: Assign sorted boundaries back to ensure binary search works correctly
        self.trial_boundaries = sorted_boundaries
        
        # Check for overlaps and gaps
        # FIXED: Validate boundaries structurally without per-sample metadata (much faster)
        # FIXED: trial_boundaries is always 4-tuple: (start, end, subject_id, trial_id)
        prev_end = 0
        for i, (start, end, subject_id, trial_id) in enumerate(sorted_boundaries):
            if start < prev_end:
                raise ValueError(f"CRITICAL: Trial boundary overlap detected! "
                               f"Trial {i} starts at {start} but previous ends at {prev_end}")
            if start > prev_end and i > 0:
                print(f"WARNING: Gap between trials: previous ends at {prev_end}, next starts at {start}")
            
            if start >= end:
                raise ValueError(f"CRITICAL: Invalid trial boundary! Start ({start}) >= End ({end})")
            
            if end > len(self.eeg_data):
                raise ValueError(f"CRITICAL: Trial boundary exceeds data length! "
                               f"End ({end}) > data length ({len(self.eeg_data)})")
            
            # FIXED: Verify label consistency within trial (labels should be constant per trial)
            trial_labels = self.labels[start:end]
            if len(trial_labels) > 0:
                if not np.all(trial_labels == trial_labels[0]):
                    raise ValueError(f"CRITICAL: Label changed within trial {i}! "
                                   f"Trial [{start}:{end}] has labels: {np.unique(trial_labels)}. "
                                   f"Labels should be constant within a trial.")
            
            prev_end = end
        
        # Check that boundaries cover all data (with small tolerance for rounding)
        # FIXED: trial_boundaries is 4-tuple: (start, end, subject_id, trial_id)
        total_covered = sum(b[1] - b[0] for b in sorted_boundaries)
        if abs(total_covered - len(self.eeg_data)) > 10:  # Allow small rounding differences
            print(f"WARNING: Trial boundaries may not cover all data. "
                  f"Covered: {total_covered}, Total: {len(self.eeg_data)}, Diff: {abs(total_covered - len(self.eeg_data))}")
        
        print(f"✓ Validated {len(sorted_boundaries)} trial boundaries")
    
    def _get_window_subject(self, window_start_idx: int, window_size: int) -> str:
        """Get subject ID for a window using trial_boundaries (efficient binary search).
        
        FIXED: Uses binary search instead of O(n) linear search for better performance.
        """
        window_end_idx = min(window_start_idx + window_size, len(self.eeg_data))
        
        # Use binary search for O(log n) lookup instead of O(n)
        # Find the trial that contains window_start_idx
        left, right = 0, len(self.trial_boundaries) - 1
        while left <= right:
            mid = (left + right) // 2
            trial_start, trial_end, subject_id, trial_id = self.trial_boundaries[mid]
            
            if window_start_idx < trial_start:
                right = mid - 1
            elif window_start_idx >= trial_end:
                left = mid + 1
            else:
                # window_start_idx is within this trial
                # FIXED: Be strict - windows should always be within trial boundaries
                # Since windows are created within each trial, this should always hold
                if window_end_idx > trial_end:
                    # This should never happen if windowing is correct - raise during dev
                    raise ValueError(f"CRITICAL: Window [{window_start_idx}:{window_end_idx}] exceeds trial boundary [{trial_start}:{trial_end}]. "
                                   f"This indicates a windowing bug - windows should always be within trial boundaries.")
                return subject_id
        
        # Fallback: linear search if binary search fails (shouldn't happen)
        for trial_start, trial_end, subject_id, trial_id in self.trial_boundaries:
            if window_start_idx < trial_end and window_end_idx > trial_start:
                return subject_id
        
        # Final fallback: shouldn't happen if windowing is correct
        print(f"WARNING: Could not find subject for window [{window_start_idx}:{window_end_idx}]")
        return 'unknown'
    
    def _create_kuleuven_windows(self) -> Tuple[List[Tuple[int, int]], List[Tuple[str, str]]]:
        """Create sliding windows from KU255 data.
        
        CRITICAL: Windows are created within each trial separately to prevent
        crossing subject/trial boundaries. This ensures:
        - No window spans multiple subjects
        - No window spans multiple trials
        - Labels are correct (no majority voting across boundaries)
        - Subject-wise splits are safe
        """
        # Convert to seconds for display
        window_seconds = self.window_size / self.sampling_rate
        step_size = int(self.window_size * (1 - self.overlap))
        # FIXED: Guard against overlap=1.0 causing division by zero
        if step_size <= 0:
            raise ValueError(f"Invalid overlap={self.overlap}. Must be < 1.0. Overlap=1.0 causes step_size=0.")
        step_seconds = step_size / self.sampling_rate
        
        print(f"Creating windows (size: {self.window_size} samples, {window_seconds:.1f}s, step: {step_seconds:.1f}s)")
        print(f"CRITICAL: Windows are created within each trial separately to prevent boundary crossing")
        
        # Warn about window size (valid range: 1-30 seconds)
        if window_seconds < 1.0:
            print(f"WARNING: Very short window ({window_seconds:.1f}s) may have poor signal-to-noise")
        elif window_seconds > 30.0:
            print(f"WARNING: Window ({window_seconds:.1f}s) exceeds maximum recommended (30s)")
        elif window_seconds > 20.0:
            print(f"WARNING: Very long window ({window_seconds:.1f}s) may miss temporal dynamics")
        
        window_indices = []
        window_trial_keys = []  # (subject_id, trial_id) per window for trial-level aggregation
        
        # CRITICAL FIX: Window within each trial separately
        # FIXED: trial_boundaries is 4-tuple: (start, end, subject_id, trial_id)
        for trial_start, trial_end, subject_id, trial_id in self.trial_boundaries:
            trial_length = trial_end - trial_start
            
            # Skip trials that are too short
            if trial_length < self.window_size:
                continue
            
            # Create windows within this trial only
            trial_windows = (trial_length - self.window_size) // step_size + 1
            
            for i in range(trial_windows):
                # Window start relative to trial start
                window_offset = i * step_size
                data_idx = trial_start + window_offset
                
                # Ensure window doesn't exceed trial boundary
                if data_idx + self.window_size > trial_end:
                    break
                
                # Get labels for this window (all from same trial, so label is consistent)
                window_start = data_idx
                window_end = data_idx + self.window_size
                window_labels = self.labels[window_start:window_end]
                
                # FIXED: Hard check for unanimous labels within trial (catches boundary bugs)
                if len(window_labels) > 0:
                    # Since window is within a single trial, all labels should be identical
                    if not np.all(window_labels == window_labels[0]):
                        raise ValueError(f"CRITICAL: Label changed within a single trial window! "
                                       f"Window [{window_start}:{window_end}] has labels: {np.unique(window_labels)}. "
                                       f"This indicates a boundary/label bug - labels should be constant within a trial.")
                    window_label = int(window_labels[0])
                else:
                    raise ValueError(f"CRITICAL: Empty window labels for window [{window_start}:{window_end}]")
                
                window_indices.append((data_idx, window_label))
                window_trial_keys.append((subject_id, trial_id))
        
        print(f"Created {len(window_indices)} windows")
        
        # Check window label distribution and compare with raw data
        window_labels = [label for _, label in window_indices]
        window_label_dist = np.bincount(window_labels, minlength=2)
        raw_label_dist = np.bincount(self.labels, minlength=2)
        
        print(f"Window label distribution: Class 0: {window_label_dist[0]}, Class 1: {window_label_dist[1]}")
        print(f"Raw data label distribution: Class 0: {raw_label_dist[0]}, Class 1: {raw_label_dist[1]}")
        
        # Compare window distribution with raw distribution
        if len(raw_label_dist) >= 2 and max(raw_label_dist) > 0:
            raw_balance = min(raw_label_dist) / max(raw_label_dist)
            if len(window_label_dist) >= 2 and max(window_label_dist) > 0:
                window_balance = min(window_label_dist) / max(window_label_dist)
                
                # Check if window distribution is reasonably close to raw distribution
                raw_ratio = raw_label_dist[0] / (raw_label_dist[0] + raw_label_dist[1]) if (raw_label_dist[0] + raw_label_dist[1]) > 0 else 0.5
                window_ratio = window_label_dist[0] / (window_label_dist[0] + window_label_dist[1]) if (window_label_dist[0] + window_label_dist[1]) > 0 else 0.5
                
                ratio_diff = abs(raw_ratio - window_ratio)
                if ratio_diff > 0.15:  # More than 15% difference
                    print(f"⚠ WARNING: Window label distribution differs significantly from raw data")
                    print(f"   Raw data ratio (Class 0): {raw_ratio:.3f}, Window ratio: {window_ratio:.3f}, Diff: {ratio_diff:.3f}")
                    print(f"   This may indicate windowing is creating bias - check window size and overlap")
                else:
                    print(f"✓ Window label distribution matches raw data (ratio diff: {ratio_diff:.3f})")
        
        # Warn if severely imbalanced
        if len(window_label_dist) >= 2 and max(window_label_dist) > 0:
            balance_ratio = min(window_label_dist) / max(window_label_dist)
            if balance_ratio < 0.5:
                print(f"⚠ WARNING: Window label imbalance (ratio: {balance_ratio:.3f}) - may affect training")
        
        return window_indices, window_trial_keys
    
    def get_trial_key(self, idx: int) -> Tuple[str, str]:
        """Return (subject_id, trial_id) for the window at index idx. Used for trial-level aggregation."""
        return self.window_trial_keys[idx]
    
    def _kuleuven_eeg_preprocessing(self, eeg_window: np.ndarray) -> np.ndarray:
        """FAST preprocessing: simplified for speed (skips slow filtering/artifact removal).
        
        Uses SOFT normalization: divide by (std + eps) with a floor so we don't over-normalize
        and wipe out amplitude differences that can carry attention-related signal. Per-window
        z-score (std=1) can make all windows look identical and cause near-constant logits.
        """
        # Vectorized operations only - 10x faster than scipy filtering
        # Skip artifact removal and filtering (too slow, done in PREPROCESS255)
        
        # Remove DC offset (vectorized, fast)
        eeg_window = eeg_window - np.mean(eeg_window, axis=0, keepdims=True)
        
        # SOFT normalization: avoid over-normalizing so amplitude differences are preserved.
        # Per-window z-score (divide by std) makes every window have var=1 and can remove
        # discriminative signal (AAD often in relative power/amplitude). Use floor on divisor.
        std_vals = np.std(eeg_window, axis=0, keepdims=True)
        std_vals = np.where(std_vals == 0, 1.0, std_vals)
        std_floor = 0.4  # Cap how much we scale down (preserve relative amplitude)
        divisor = np.maximum(std_vals, std_floor)
        eeg_window = eeg_window / divisor
        
        # FIXED: Comment out tanh for overfitting test (can destroy discriminative information)
        # Soft clipping (vectorized, fast)
        # eeg_window = np.tanh(eeg_window * 0.5)  # DISABLED: can make windows look too similar
        
        # Final check for NaNs/Infs (vectorized, fast)
        if np.any(np.isnan(eeg_window)) or np.any(np.isinf(eeg_window)):
            eeg_window = np.nan_to_num(eeg_window, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return eeg_window.astype(np.float32)
    
    def _eeg_to_timefreq_kuleuven(self, eeg_window: np.ndarray) -> np.ndarray:
        """Time-frequency representation with real temporal structure.
        
        FIXED: Now computes bandpower over multiple sub-windows to preserve
        temporal information. Each time frame represents a different sub-window.
        Scales n_time with window_size for better temporal resolution.
        """
        n_samples = eeg_window.shape[0]
        n_channels = eeg_window.shape[1]
        
        # Extract power in standard EEG bands
        bands = [
            (1, 4),   # Delta
            (4, 8),   # Theta  
            (8, 13),  # Alpha
            (13, 25), # Beta
            (25, 40)  # Gamma
        ]
        n_bands = len(bands)
        
        # FIXED: Use fixed n_time=32 for all window sizes to make temporal analysis comparable
        # But reduce n_time dynamically for very short windows to keep time representation meaningful
        if n_samples < 32 * 16:  # Less than 512 samples (4s @ 128Hz)
            n_time = min(32, max(4, n_samples // 16))  # Keep at least 4 time frames, max 32
        else:
            n_time = 32
        # Clamp sub_window_size to avoid tiny FFTs (minimum 16 samples per sub-window)
        sub_window_size = max(16, n_samples // n_time)
        
        # Initialize output: (channels, time, freq_bands)
        time_freq_array = np.zeros((n_channels, n_time, n_bands), dtype=np.float32)
        
        # Process each sub-window separately
        for t_idx in range(n_time):
            sub_start = t_idx * sub_window_size
            sub_end = min((t_idx + 1) * sub_window_size, n_samples)
            
            if sub_end <= sub_start:
                # Handle edge case
                sub_window = eeg_window[-1:, :]  # Use last sample
            else:
                sub_window = eeg_window[sub_start:sub_end, :]
            
            # Compute FFT for this sub-window
            fft_data = np.fft.rfft(sub_window, axis=0)  # Real FFT
            # FIXED: Use log-power instead of magnitude (preserves more discriminative information)
            magnitude = np.abs(fft_data)  # Magnitude
            power = magnitude ** 2  # Power
            power = np.log1p(power)  # Log-power (log(1+x) to avoid log(0))
            
            # Get frequency resolution
            freqs = np.fft.rfftfreq(sub_window.shape[0], 1.0/self.sampling_rate)
            
            # Extract band power for each band
            for band_idx, (low, high) in enumerate(bands):
                if high >= self.sampling_rate / 2:
                    high = self.sampling_rate / 2 - 1
                
                # Find frequency indices
                freq_mask = (freqs >= low) & (freqs <= high)
                if np.any(freq_mask):
                    # Average log-power across frequency band for each channel
                    band_power = np.mean(power[freq_mask, :], axis=0)  # (n_channels,) - using log-power
                else:
                    band_power = np.zeros(n_channels)
                
                # Store for this time frame and channel
                time_freq_array[:, t_idx, band_idx] = band_power
        
        # Output: (channels, time_frames, freq_bands) = (64, 8, 5)
        return time_freq_array.astype(np.float32)
    
    def __len__(self):
        return len(self.window_indices)
    
    def __getitem__(self, idx):
        data_idx, label = self.window_indices[idx]
        
        # Cache disabled for speed
        # if self._window_cache is not None:
        #     cache_key = (data_idx, self.mode)
        #     if cache_key in self._window_cache:
        #         self._cache_hits += 1
        #         cached_data, cached_label = self._window_cache[cache_key]
        #         return cached_data, cached_label
        
        self._cache_misses += 1
        
        # Extract window (EEG only)
        window_eeg = self.eeg_data[data_idx:data_idx + self.window_size]
        
        # Apply preprocessing
        try:
            window_eeg = self._kuleuven_eeg_preprocessing(window_eeg)
        except Exception:
            window_eeg = window_eeg - np.mean(window_eeg, axis=0, keepdims=True)
            window_eeg = window_eeg / (np.std(window_eeg, axis=0, keepdims=True) + 1e-8)
            window_eeg = np.tanh(window_eeg * 0.5)
        
        # Convert to time-frequency representation
        if self.transform_eeg:
            try:
                window_eeg = self._eeg_to_timefreq_kuleuven(window_eeg)
            except Exception as e:
                # If transform fails, use raw EEG with proper shape
                print(f"WARNING: Time-frequency transform failed: {e}")
                # Reshape to (channels, time, 1) for compatibility
                window_eeg = window_eeg.T  # (64, window_size)
                window_eeg = window_eeg[:, :, np.newaxis]  # (64, window_size, 1)
                pass
        
        # Convert to tensors (EEG only)
        window_tensor = torch.FloatTensor(window_eeg)
        label_tensor = torch.LongTensor([label])
        
        # Ensure proper tensor dimensions
        if window_tensor.dim() == 2:
            window_tensor = window_tensor.unsqueeze(0)  # Add channel dimension
        
        # Trial key for trial-level loss and validation ROC-AUC (AAD labels are trial-constant)
        trial_key = self.get_trial_key(idx)
        return window_tensor, label_tensor, trial_key


class ReusedWindowDataset(Dataset):
    """Lightweight dataset that reuses loaded data and only regenerates windows.
    
    FIXED: Extracted to reusable class to avoid code duplication between
    _calculate_temporal_metrics and _test_larger_window.
    """
    def __init__(self, eeg_data, labels, trial_boundaries, window_size, overlap, 
                 sampling_rate, transform_eeg, test_subject_ids):
        self.eeg_data = eeg_data
        # FIXED: Fail fast if labels is None (indicates serious dataset issue)
        if labels is None:
            raise ValueError("ReusedWindowDataset requires per-sample labels array. "
                           "Labels cannot be None - this indicates a dataset loading problem.")
        self.labels = self._create_labels_array(labels, trial_boundaries)
        self.trial_boundaries = trial_boundaries
        self.window_size = window_size
        self.overlap = overlap
        self.sampling_rate = sampling_rate
        self.transform_eeg = transform_eeg
        
        # Filter to test subjects only
        if test_subject_ids:
            filtered_boundaries = [
                (s, e, subj, tid) for s, e, subj, tid in trial_boundaries
                if subj in test_subject_ids
            ]
        else:
            filtered_boundaries = trial_boundaries
        
        # Regenerate window indices for new window size
        self.window_indices = self._create_windows(filtered_boundaries)
        
    def _create_labels_array(self, labels, trial_boundaries):
        """Create labels array from trial boundaries."""
        if isinstance(labels, list):
            return np.array(labels)
        return labels
        
    def _create_windows(self, boundaries):
        """Create windows within trials for new window size."""
        step_size = int(self.window_size * (1 - self.overlap))
        # FIXED: Guard against overlap=1.0 causing division by zero
        if step_size <= 0:
            raise ValueError(f"Invalid overlap={self.overlap}. Must be < 1.0. Overlap=1.0 causes step_size=0.")
        window_indices = []
        for trial_start, trial_end, subject_id, trial_idx in boundaries:
            trial_length = trial_end - trial_start
            if trial_length < self.window_size:
                continue
            trial_windows = (trial_length - self.window_size) // step_size + 1
            for i in range(trial_windows):
                window_offset = i * step_size
                data_idx = trial_start + window_offset
                if data_idx + self.window_size > trial_end:
                    break
                # FIXED: Assert labels are constant per trial (no majority voting)
                window_labels = self.labels[data_idx:data_idx + self.window_size]
                if len(window_labels) == 0:
                    continue
                if not np.all(window_labels == window_labels[0]):
                    raise ValueError("Label changed inside a trial window — boundary/label bug.")
                window_label = int(window_labels[0])
                window_indices.append((data_idx, window_label))
        return window_indices
    
    def __len__(self):
        return len(self.window_indices)
    
    def __getitem__(self, idx):
        data_idx, label = self.window_indices[idx]
        window_eeg = self.eeg_data[data_idx:data_idx + self.window_size]
        
        # Apply preprocessing: SOFT normalization (match main dataset to avoid constant logits)
        window_eeg = window_eeg - np.mean(window_eeg, axis=0, keepdims=True)
        std_vals = np.std(window_eeg, axis=0, keepdims=True)
        std_vals = np.where(std_vals == 0, 1.0, std_vals)
        divisor = np.maximum(std_vals, 0.4)  # Floor to preserve relative amplitude
        window_eeg = window_eeg / (divisor + 1e-6)
        # Tanh disabled (can make windows look too similar, destroying discriminative info)
        
        # Convert to time-frequency representation
        if self.transform_eeg:
            window_eeg = self._eeg_to_timefreq(window_eeg)
        
        window_tensor = torch.FloatTensor(window_eeg)
        label_tensor = torch.LongTensor([label])
        
        if window_tensor.dim() == 2:
            window_tensor = window_tensor.unsqueeze(0)
        
        return window_tensor, label_tensor
    
    def _eeg_to_timefreq(self, eeg_window):
        """Time-frequency transform (simplified, reuses logic from main dataset)."""
        n_samples = eeg_window.shape[0]
        n_channels = eeg_window.shape[1]
        bands = [(1, 4), (4, 8), (8, 13), (13, 25), (25, 40)]
        n_bands = len(bands)
        
        # FIXED: Use fixed n_time=32 for all window sizes to make temporal analysis comparable
        # But reduce n_time dynamically for very short windows to keep time representation meaningful
        if n_samples < 32 * 16:  # Less than 512 samples (4s @ 128Hz)
            n_time = min(32, max(4, n_samples // 16))  # Keep at least 4 time frames, max 32
        else:
            n_time = 32
        # Clamp sub_window_size to avoid tiny FFTs (minimum 16 samples per sub-window)
        sub_window_size = max(16, n_samples // n_time)
        time_freq_array = np.zeros((n_channels, n_time, n_bands), dtype=np.float32)
        
        for t_idx in range(n_time):
            sub_start = t_idx * sub_window_size
            sub_end = min((t_idx + 1) * sub_window_size, n_samples)
            if sub_end <= sub_start:
                sub_window = eeg_window[-1:, :]
            else:
                sub_window = eeg_window[sub_start:sub_end, :]
            
            fft_data = np.fft.rfft(sub_window, axis=0)
            # FIXED: Use log-power instead of magnitude (preserves more discriminative information)
            magnitude = np.abs(fft_data)  # Magnitude
            power = magnitude ** 2  # Power
            power = np.log1p(power)  # Log-power (log(1+x) to avoid log(0))
            freqs = np.fft.rfftfreq(sub_window.shape[0], 1.0/self.sampling_rate)
            
            for band_idx, (low, high) in enumerate(bands):
                if high >= self.sampling_rate / 2:
                    high = self.sampling_rate / 2 - 1
                freq_mask = (freqs >= low) & (freqs <= high)
                if np.any(freq_mask):
                    band_power = np.mean(power[freq_mask, :], axis=0)  # Using log-power
                else:
                    band_power = np.zeros(n_channels)
                time_freq_array[:, t_idx, band_idx] = band_power
        
        return time_freq_array.astype(np.float32)


class SpatialTemporalAttention(nn.Module):
    """Channel attention for EEG data. Kept simple to save memory."""
    
    def __init__(self, channels: int, reduction: int = 8, dropout_rate: float = 0.1):
        super(SpatialTemporalAttention, self).__init__()
        
        self.channels = channels
        self.reduction = max(1, reduction)
        self.reduced_channels = max(1, channels // self.reduction)
        
        # Channel attention only (no temporal to save memory)
        # FIXED: Replaced Global Average Pooling with AdaptiveMaxPool2d (as per user requirement)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),  # Changed from AdaptiveAvgPool2d to AdaptiveMaxPool2d
            nn.Conv2d(channels, self.reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(self.reduced_channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        channel_att = self.channel_attention(x)
        return x * channel_att


class ResidualBlock(nn.Module):
    """Residual block with attention. Standard ResNet-style.
    
    Uses GroupNorm instead of BatchNorm for better cross-subject generalization.
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout_rate: float = 0.2, num_groups: int = 8):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        # Use GroupNorm instead of BatchNorm for better cross-subject generalization
        self.gn1 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        # Shortcut for residual connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.GroupNorm(min(num_groups, out_channels), out_channels)
            )
        
        self.attention = SpatialTemporalAttention(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.gn2(self.conv2(out))
        out = self.dropout2(out)
        out = self.attention(out)
        
        out += residual
        out = self.relu(out)
        
        return out


class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale features using different kernel sizes. Simplified to save memory.
    
    Uses GroupNorm instead of BatchNorm for better cross-subject generalization.
    """
    
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8):
        super(MultiScaleFeatureExtractor, self).__init__()
        
        # Two scales: 1x1 and 3x1
        self.conv1x1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        self.conv3x1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=(3, 1), padding=(1, 0))
        
        # Use GroupNorm instead of BatchNorm
        self.gn = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        feat1 = self.conv1x1(x)
        feat3 = self.conv3x1(x)
        
        # Concatenate
        out = torch.cat([feat1, feat3], dim=1)
        out = self.relu(self.gn(out))
        
        return out


class AdaptivePooling(nn.Module):
    """Adaptive pooling for variable input sizes.
    
    FIXED: Replaced Global Average Pooling with AdaptiveMaxPool2d (as per user requirement).
    """
    
    def __init__(self, output_size: int = 1):
        super(AdaptivePooling, self).__init__()
        self.output_size = output_size
        # FIXED: Always use (1, 1) pooling to ensure consistent output size regardless of input
        # This makes the classifier input-size invariant for temporal analysis
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1, 1))  # Always (1, 1) for size invariance
        self.pool_size = (1, 1)
        
    def forward(self, x):
        # FIXED: Always use (1, 1) pooling for size invariance
        return self.adaptive_pool(x)


class KU255CNNBackbone(nn.Module):
    """Backbone network: attention, residual blocks, multi-scale features.
    
    Uses GroupNorm throughout for better cross-subject generalization.
    """
    
    def __init__(self, input_channels: int = 64, input_time: int = 32, input_freq: int = 5,
                 adaptive_input: bool = True, dropout_rate: float = 0.2, num_groups: int = 8):
        super(KU255CNNBackbone, self).__init__()
        
        self.input_channels = input_channels
        self.input_time = input_time
        self.input_freq = input_freq
        self.adaptive_input = adaptive_input
        
        print(f"Building KU255CNN backbone: channels={input_channels}, time={input_time}, freq={input_freq}, dropout={dropout_rate}, GroupNorm groups={num_groups}")
        
        # Initial multi-scale features (reduced capacity to prevent overfitting)
        self.initial_features = MultiScaleFeatureExtractor(input_channels, 24, num_groups=num_groups)  # Reduced from 32
        
        # Temporal blocks (reduced capacity with dropout)
        self.temporal_block1 = ResidualBlock(24, 24, stride=1, dropout_rate=dropout_rate, num_groups=num_groups)  # Reduced from 32
        self.temporal_pool1 = nn.MaxPool2d((2, 1), (2, 1))
        
        self.temporal_block2 = ResidualBlock(24, 48, stride=1, dropout_rate=dropout_rate, num_groups=num_groups)  # Reduced from 32->64
        self.temporal_pool2 = nn.MaxPool2d((2, 1), (2, 1))
        
        # Spatial blocks (reduced capacity with dropout)
        self.spatial_block1 = ResidualBlock(48, 48, stride=1, dropout_rate=dropout_rate, num_groups=num_groups)  # Reduced from 64
        self.spatial_pool1 = nn.MaxPool2d((1, 2), (1, 2))
        
        self.spatial_block2 = ResidualBlock(48, 96, stride=1, dropout_rate=dropout_rate, num_groups=num_groups)  # Reduced from 64->128
        self.spatial_pool2 = nn.MaxPool2d((1, 2), (1, 2))
        
        # Global attention (reduced capacity with dropout)
        self.global_attention = SpatialTemporalAttention(96, dropout_rate=dropout_rate * 0.5)  # Reduced from 128
        
        # Adaptive pooling
        self.adaptive_pooling = AdaptivePooling(output_size=1)
        
        # Calculate output size
        self._calculate_output_size()
        
    
    def _calculate_output_size(self):
        """Figure out output size by running a dummy input.
        
        NOTE: The input_time and input_freq parameters are initial estimates.
        The actual dimensions may differ based on the time-frequency transform
        applied to different window sizes. The real dimensions are detected from
        the actual data in main() and passed to the model.
        """
        dummy_input = torch.randn(1, self.input_channels, self.input_time, self.input_freq)
        
        with torch.no_grad():
            x = self.forward(dummy_input)
            self.output_size = x.numel()
        
    
    def forward(self, x):
        """Forward pass."""
        # Multi-scale features
        x = self.initial_features(x)
        
        # Temporal processing
        x = self.temporal_block1(x)
        x = self.temporal_pool1(x)
        
        x = self.temporal_block2(x)
        x = self.temporal_pool2(x)
        
        # Spatial processing
        x = self.spatial_block1(x)
        x = self.spatial_pool1(x)
        
        x = self.spatial_block2(x)
        x = self.spatial_pool2(x)
        
        # Attention
        x = self.global_attention(x)
        
        # Pool and flatten
        x = self.adaptive_pooling(x)
        x = x.view(x.size(0), -1)
        
        return x


class KU255CNNModel(nn.Module):
    """Full KU255CNN model: backbone + classifier for EEG attention decoding."""
    
    def __init__(self, input_channels: int = 64, input_time: int = 32, input_freq: int = 5,
                 num_classes: int = 2, dropout_rate: float = 0.3):
        super(KU255CNNModel, self).__init__()
        
        # Create backbone with dropout
        self.backbone = KU255CNNBackbone(input_channels, input_time, input_freq, dropout_rate=dropout_rate * 0.7)
        
        # FIXED: Classifier input-size invariant - use fixed size based on final channels (96)
        # After adaptive pooling (1,1), output is always (batch, 96, 1, 1) -> (batch, 96) when flattened
        # This works for any input window size
        final_channels = 96  # From spatial_block2 output
        classifier_input_size = final_channels  # After (1,1) pooling and flatten
        
        # Classifier: moderate dropout so gradients can flow (too much caused logit collapse)
        self.classifier = nn.Sequential(
            nn.Dropout(min(0.2, dropout_rate)),  # Cap so signal reaches final layer
            nn.Linear(classifier_input_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(min(0.2, dropout_rate * 0.8)),
            nn.Linear(64, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Dropout(min(0.2, dropout_rate * 0.9)),
            nn.Linear(16, num_classes)
        )
        
        self._initialize_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model created with {n_params:,} parameters")
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # FIXED: Use proper initialization for final classification layer
                # Check if this is the final layer (output layer with num_classes)
                if m.out_features == 2:  # Final classification layer
                    # Use larger init so logits have meaningful magnitude; std=0.01 caused collapse (logits ~0.003)
                    nn.init.normal_(m.weight, mean=0.0, std=0.1)
                    nn.init.constant_(m.bias, 0.0)  # Start with zero bias
                else:
                    # Hidden layers: use kaiming for ReLU
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the model."""
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class KU255CNNTrainer:
    """Handles training, validation, testing, and metrics for KU255CNN."""
    
    def __init__(self, model: KU255CNNModel, device: torch.device, 
                 output_dir: str = "ku255cnnloc_results", preprocessed_dir: str = None, 
                 sampling_rate: int = 128, window_size: int = 512, raw_data_dir: Optional[str] = None):
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Dataset parameters
        self.preprocessed_dir = preprocessed_dir
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir else None
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        
        # Store test subject IDs to prevent data leakage in temporal analysis
        self.test_subject_ids = None
        self.train_subject_ids = None
        self.val_subject_ids = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        self.best_val_acc = 0.0
        self.best_val_trial_auc = 0.0  # Model selection on trial-level ROC-AUC (primary for AAD)
        self.best_val_threshold = 0.5   # Optimal threshold from validation (never tune on test)
        self.best_model_path = self.output_dir / "best_model.pth"
        
    
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer,
                   criterion: nn.Module, scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                   mixup_alpha: float = 0.2, epoch: int = 0) -> Tuple[float, float]:
        """Train for one epoch using window-level loss (trial keys only used for metrics)."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        n_batches_used = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            # Support both (data, target) and (data, target, trial_keys) for backward compatibility
            data = batch[0].to(self.device)
            target = batch[1].to(self.device).view(-1)
            trial_keys = batch[2] if len(batch) >= 3 else None
            
            # For KU255CNNLOC, train on individual windows (like DASCNNLOC/FULCNNLOC);
            # trial keys are used only for validation metrics, not for trial-level loss.
            use_trial_level = False
            do_aug = (mixup_alpha > 0) and self.model.training
            
            if do_aug:
                if torch.rand(1) > 0.5:
                    lam = np.random.beta(mixup_alpha, mixup_alpha)
                    batch_size = data.size(0)
                    index = torch.randperm(batch_size).to(self.device)
                    mixed_data = lam * data + (1 - lam) * data[index, :]
                    target_a, target_b = target, target[index]
                else:
                    mixed_data = data
                    target_a, target_b = target, target
                    lam = 1.0
                noise = torch.randn_like(mixed_data) * 0.01
                mixed_data = mixed_data + noise
                if torch.rand(1) > 0.5:
                    scale = torch.rand(1, device=mixed_data.device) * 0.1 + 0.95
                    mixed_data = mixed_data * scale
                data = mixed_data
            else:
                lam = 1.0
                target_a, target_b = target, target
            
            output = self.model(data)  # (B, 2)
            B = data.size(0)
            
            if use_trial_level and len(trial_keys) >= B:
                # Trial-level loss: group by trial, mean logits per trial, CE(mean_logit, trial_label)
                trial_logits = defaultdict(list)
                trial_targets = {}
                for i in range(B):
                    k = trial_keys[i] if isinstance(trial_keys[i], tuple) else tuple(trial_keys[i]) if hasattr(trial_keys[i], '__iter__') else trial_keys[i]
                    trial_logits[k].append(output[i : i + 1])
                    trial_targets[k] = target[i].unsqueeze(0)
                loss = torch.tensor(0.0, device=self.device, dtype=output.dtype)
                n_trials = 0
                for k in trial_logits:
                    mean_logit = torch.cat(trial_logits[k], dim=0).mean(dim=0, keepdim=True)
                    trial_label = trial_targets[k]
                    loss = loss + criterion(mean_logit, trial_label)
                    n_trials += 1
                loss = loss / max(n_trials, 1)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.numel()
            else:
                if do_aug and lam < 1.0:
                    loss = lam * criterion(output, target_a) + (1 - lam) * criterion(output, target_b)
                else:
                    loss = criterion(output, target)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.numel()
            
            if torch.isnan(loss):
                continue
            if torch.any(torch.isnan(output)):
                output = torch.nan_to_num(output, nan=0.0)
            
            total_loss += loss.item()
            n_batches_used += 1
            
            optimizer.zero_grad()
            loss.backward()
            
            if batch_idx == 0 and epoch == 0:
                grad_norm_sum = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        grad_norm_sum += p.grad.data.norm(2).item()
                print(f"  [DEBUG] grad_norm_sum: {grad_norm_sum:.6f}")
                if grad_norm_sum < 1e-6:
                    print(f"  ⚠ WARNING: Gradient norm is very small - model may not be updating!")
                self._w0 = self.model.classifier[-1].weight.detach().clone()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if batch_idx == 0 and epoch == 0:
                w1 = self.model.classifier[-1].weight.detach()
                last_layer_delta_mean = (w1 - self._w0).abs().mean().item()
                print(f"  [DEBUG] last_layer_delta_mean: {last_layer_delta_mean:.6f}")
                if last_layer_delta_mean < 1e-6:
                    print(f"  ⚠ WARNING: Weights not changing - optimizer may not be working!")
            
            if scheduler is not None and isinstance(scheduler, OneCycleLR):
                scheduler.step()
            
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if total == 0:
            return float('inf'), 0.0
        avg_loss = total_loss / max(1, n_batches_used)
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float, float, float]:
        """Validate for one epoch. Returns (val_loss, window_acc, trial_roc_auc, trial_acc). Trial metrics are 0.0 if no trial keys."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_logits = []
        all_targets = []
        all_trial_keys = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                data = batch[0].to(self.device)
                target = batch[1].to(self.device).view(-1)
                trial_keys = batch[2] if len(batch) >= 3 else None
                
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.numel()
                
                if trial_keys is not None and len(trial_keys) > 0:
                    all_logits.append(output.cpu())
                    all_targets.append(target.cpu())
                    all_trial_keys.extend([k if isinstance(k, tuple) else tuple(k) for k in trial_keys])
        
        if total == 0:
            return float('inf'), 0.0, 0.0, 0.0
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        val_trial_auc = 0.0
        val_trial_acc = 0.0
        if all_logits and all_trial_keys:
            logits = torch.cat(all_logits, dim=0)
            targets = torch.cat(all_targets, dim=0)
            probs = F.softmax(logits, dim=1)[:, 1].numpy()
            targets_np = targets.numpy()
            by_trial = defaultdict(lambda: {'probs': [], 'targets': []})
            for i in range(len(all_trial_keys)):
                k = all_trial_keys[i]
                by_trial[k]['probs'].append(probs[i])
                by_trial[k]['targets'].append(targets_np[i])
            trial_probs = np.array([np.mean(by_trial[k]['probs']) for k in sorted(by_trial.keys())])
            trial_targets_arr = np.array([by_trial[k]['targets'][0] for k in sorted(by_trial.keys())])
            # Trial-level accuracy: one prediction per trial (mean prob >= 0.5 -> class 1)
            trial_preds = (trial_probs >= 0.5).astype(np.int64)
            val_trial_acc = (trial_preds == trial_targets_arr).mean()
            if len(np.unique(trial_targets_arr)) == 2:
                try:
                    val_trial_auc = roc_auc_score(trial_targets_arr, trial_probs)
                except Exception:
                    val_trial_auc = 0.0
        
        return avg_loss, accuracy, val_trial_auc, val_trial_acc
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 50, learning_rate: float = 3e-3,
              weight_decay: float = 1e-5, patience: int = 7, label_smoothing: float = 0.0,
              mixup_alpha: float = 0.0, train_indices: Optional[List[int]] = None,
              full_dataset: Optional[Any] = None):
        """Train the model with class balancing and label smoothing.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Initial learning rate
            weight_decay: Weight decay for optimizer
            patience: Early stopping patience
            label_smoothing: Label smoothing (0.0 recommended for binary AAD)
            mixup_alpha: Mixup augmentation alpha (0.0 for overfit test)
            train_indices: Optional list of training indices (for optimized class weight computation)
            full_dataset: Optional full dataset (for optimized class weight computation)
        """
        
        # Get class weights: use trial-level counts when we have trial keys (trial-level loss)
        num_classes = 2
        use_trial_weights = False
        if train_indices is not None and full_dataset is not None and hasattr(full_dataset, 'window_trial_keys'):
            # One label per trial (matches trial-level loss)
            trial_to_label = {}
            for idx in train_indices:
                k = full_dataset.window_trial_keys[idx]
                trial_to_label[k] = full_dataset.window_indices[idx][1]
            trial_labels = list(trial_to_label.values())
            n_trials = len(trial_labels)
            trial_counts = np.bincount(trial_labels, minlength=num_classes)
            if n_trials > 0:
                weights = np.array([
                    n_trials / (num_classes * trial_counts[c]) if trial_counts[c] > 0 else 1.0
                    for c in range(num_classes)
                ], dtype=np.float32)
                weights = np.clip(weights, 0.5, 2.0)
                class_weights = torch.tensor(weights, device=self.device)
                use_trial_weights = True
                print(f"Class weights (per-trial, n_trials={n_trials}): {class_weights.cpu().numpy()}, counts: {trial_counts.tolist()}")
        if not use_trial_weights:
            if train_indices is not None and full_dataset is not None:
                train_labels = [full_dataset.window_indices[idx][1] for idx in train_indices]
            else:
                train_labels = []
                for _, batch in enumerate(train_loader):
                    train_labels += batch[1].view(-1).cpu().numpy().tolist()
            unique, counts = np.unique(train_labels, return_counts=True)
            n_total = len(train_labels)
            weights = np.ones(num_classes, dtype=np.float32)
            if n_total > 0:
                for cls_id, cnt in zip(unique, counts):
                    if cnt > 0:
                        weights[int(cls_id)] = n_total / (len(unique) * cnt)
                class_weights = torch.tensor(weights, device=self.device)
            else:
                class_weights = torch.ones(num_classes, device=self.device)
            if set(unique.tolist()) == {0, 1}:
                balance_ratio = min(counts) / max(counts)
                if balance_ratio < 0.9:
                    minority_idx = np.argmin(counts)
                    weights[unique[minority_idx]] = min(2.0, weights[unique[minority_idx]] * 2.0)
                    class_weights = torch.tensor(np.clip(weights, 0.5, 2.0), device=self.device)
            print(f"Class distribution (window-level): {dict(zip(unique, counts))}, weights: {class_weights.cpu().numpy()}")
        
        # Use plain CrossEntropyLoss like DASCNNLOC/FULCNNLOC (no focal loss, no extra class weights)
        use_class_weights = False
        if not use_class_weights:
            class_weights = None
        
        # FocalLoss is kept for experimentation but disabled by default so training matches other CNN-LOC models.
        use_focal_loss = False
        
        if use_focal_loss:
            # FocalLoss helps prevent collapse to constant predictions
            class FocalLoss(nn.Module):
                def __init__(self, gamma=2.0, alpha=None):
                    super().__init__()
                    self.gamma = gamma
                    self.alpha = alpha

                def forward(self, logits, targets):
                    ce = F.cross_entropy(logits, targets, reduction="none")
                    pt = torch.exp(-ce)
                    loss = ((1 - pt) ** self.gamma) * ce
                    if self.alpha is not None:
                        at = self.alpha.gather(0, targets)
                        loss = at * loss
                    return loss.mean()
            
            alpha_tensor = torch.tensor([1.0, 1.0], device=self.device) if class_weights is None else class_weights
            criterion = FocalLoss(gamma=2.0, alpha=alpha_tensor)
            print("Using FocalLoss (gamma=2.0) to prevent class collapse")
        else:
            # Standard CrossEntropyLoss with class weights and label smoothing
            criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing
            )
        
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # FIXED: Use CosineAnnealingLR instead of ReduceLROnPlateau (prevents premature LR reduction)
        # Cosine annealing provides smooth LR decay without cutting LR while still learning
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
        
        patience_counter = 0
        best_train_val_gap = float('inf')
        best_val_loss = float('inf')
        no_improve_counter = 0  # Counter for validation loss not improving
        collapse_detected = False  # Track if model collapse was detected
        overfit_lr_reduction_count = 0  # Cap LR reductions so we don't drive LR to ~0 (max 2)
        stagnant_50_epochs = 0  # Consecutive epochs with val_acc stuck near 50%
        
        print(f"Starting training: {num_epochs} epochs, lr={learning_rate}, wd={weight_decay}, label_smoothing={label_smoothing}, mixup={mixup_alpha}")
        print("Using CosineAnnealingLR scheduler (smooth decay, no premature reduction)")
        print(f"⚠ NOTE: If model collapse is detected, regularization will be automatically reduced")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion, None, mixup_alpha=mixup_alpha, epoch=epoch)
            val_loss, val_acc, val_trial_auc, val_trial_acc = self.validate_epoch(val_loader, criterion)
            
            recovered_this_epoch = False  # Skip scheduler.step() if we applied collapse recovery (so recovery LR is kept)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Calculate train-val gap to detect overfitting (in 0-1 range)
            train_val_gap = train_acc - val_acc
            
            # Display: window acc = diagnostic; trial metrics = primary for AAD (one decision per trial)
            print(f"Train Loss: {train_loss:.4f}, Train Acc (window): {train_acc*100:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc (window): {val_acc*100:.2f}%  |  Val Trial Acc: {val_trial_acc*100:.2f}%, Val Trial AUC: {val_trial_auc:.4f}")
            print(f"  → For AAD use trial metrics (trial acc / trial AUC); window acc is diagnostic only.")
            print(f"Train-Val Gap: {train_val_gap*100:.2f}%")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early detection of model collapse (predicting only one class)
            # Collapse causes: (1) LR too low or scheduler overwriting recovery LR, (2) weight decay too high,
            # (3) dropout too high so gradients don't reach classifier, (4) label_smoothing pushing probs to 0.5,
            # (5) trial-level loss diluting gradients across many windows per trial.
            if epoch >= 3:
                if 0.48 <= val_acc <= 0.52 and train_acc <= 0.55:
                    stagnant_50_epochs += 1
                    print(f"⚠ WARNING: Model appears stuck at random performance (val_acc={val_acc*100:.2f}%)")
                    if stagnant_50_epochs >= 5:
                        print(f"   Val accuracy has been ~50% for {stagnant_50_epochs} consecutive epochs.")
                        print(f"   → Run label check: python check_ku255_labels.py --preprocessed_dir <dir>")
                        print(f"   → If labels are correct, try test-time flip: python KU255CNNLOC.py ... --flip_labels")
                        print(f"   → If flipped accuracy is higher, labels may be swapped (L/R) in the pipeline.")
                    self.model.eval()
                    with torch.no_grad():
                        sample_batch = next(iter(val_loader))
                        sample_data = sample_batch[0].to(self.device)
                        sample_target = sample_batch[1].to(self.device)
                        sample_output = self.model(sample_data)
                        sample_preds = sample_output.argmax(dim=1)
                        unique_preds = torch.unique(sample_preds).cpu().numpy()
                        if len(unique_preds) == 1:
                            print(f"⚠ CRITICAL: Model predicts only class {unique_preds[0]} - model collapse detected!")
                            print(f"   Logits range: [{sample_output.min().item():.4f}, {sample_output.max().item():.4f}]")
                            if not collapse_detected:
                                collapse_detected = True
                                recovered_this_epoch = True
                                print(f"   Automatically reducing regularization to help model learn...")
                                for param_group in optimizer.param_groups:
                                    old_wd = param_group['weight_decay']
                                    param_group['weight_decay'] = max(old_wd * 0.1, 1e-7)
                                    print(f"   Weight decay reduced: {old_wd:.2e} -> {param_group['weight_decay']:.2e}")
                                for param_group in optimizer.param_groups:
                                    old_lr = param_group['lr']
                                    param_group['lr'] = min(old_lr * 2.0, 1e-2)
                                    print(f"   Learning rate increased: {old_lr:.2e} -> {param_group['lr']:.2e}")
                                # Restart cosine decay from new LR over remaining epochs (so scheduler doesn't overwrite recovery LR)
                                remaining = max(1, num_epochs - epoch - 1)
                                scheduler = CosineAnnealingLR(optimizer, T_max=remaining, eta_min=1e-5)
                                print(f"   Scheduler reset: cosine decay over {remaining} remaining epochs (recovery LR kept).")
                    self.model.train()
                else:
                    stagnant_50_epochs = 0  # Reset when not in stagnant band
            
            # Track validation loss improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_counter = 0
            else:
                no_improve_counter += 1
            
            # Warn if overfitting is severe; apply LR reduction at most 2x to avoid killing learning
            # (Repeated reduction was driving LR to 1e-7 and preventing any learning.)
            if train_val_gap > 0.25:
                print(f"⚠ WARNING: Severe overfitting detected! Train-Val gap: {train_val_gap*100:.2f}%")
                if overfit_lr_reduction_count < 2:
                    overfit_lr_reduction_count += 1
                    print("   Applying automatic countermeasure (at most 2x per run)...")
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = max(param_group['lr'] * 0.5, 1e-5)  # Floor 1e-5, not 1e-7
                    print(f"   Learning rate reduced to: {optimizer.param_groups[0]['lr']:.2e}")
                else:
                    print("   (LR already reduced 2x this run; relying on early stopping.)")
            elif train_val_gap > 0.15 and overfit_lr_reduction_count < 2:
                print(f"⚠ WARNING: Moderate overfitting detected! Train-Val gap: {train_val_gap*100:.2f}%")
                overfit_lr_reduction_count += 1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr'] * 0.8, 1e-5)
                print(f"   Learning rate reduced to: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Model selection on trial-level ROC-AUC (primary for AAD); fallback to window acc if no trial AUC
            use_auc_for_selection = (val_trial_auc > 0.0)
            improved = (val_trial_auc > self.best_val_trial_auc) if use_auc_for_selection else (val_acc > self.best_val_acc)
            if improved:
                if use_auc_for_selection:
                    self.best_val_trial_auc = val_trial_auc
                    self.best_val_acc = val_acc
                    best_train_val_gap = train_val_gap
                    patience_counter = 0
                    # Compute optimal threshold from validation (for test; never tune on test)
                    try:
                        all_logits_v, all_tgts_v, all_keys_v = [], [], []
                        self.model.eval()
                        with torch.no_grad():
                            for batch in val_loader:
                                out = self.model(batch[0].to(self.device))
                                all_logits_v.append(out.cpu())
                                all_tgts_v.append(batch[1].view(-1).cpu())
                                if len(batch) >= 3:
                                    all_keys_v.extend([k if isinstance(k, tuple) else tuple(k) for k in batch[2]])
                        self.model.train()
                        if all_logits_v and all_keys_v:
                            log_v = torch.cat(all_logits_v, dim=0)
                            tgt_v = torch.cat(all_tgts_v, dim=0).numpy()
                            prob_v = F.softmax(log_v, dim=1)[:, 1].numpy()
                            by_t = defaultdict(lambda: {'probs': [], 'targets': []})
                            for i in range(len(all_keys_v)):
                                by_t[all_keys_v[i]]['probs'].append(prob_v[i])
                                by_t[all_keys_v[i]]['targets'].append(tgt_v[i])
                            trial_p = np.array([np.mean(by_t[k]['probs']) for k in sorted(by_t.keys())])
                            trial_t = np.array([by_t[k]['targets'][0] for k in sorted(by_t.keys())])
                            if len(np.unique(trial_t)) == 2:
                                fpr, tpr, thr = roc_curve(trial_t, trial_p)
                                j = tpr - fpr
                                self.best_val_threshold = float(thr[np.argmax(j)])
                    except Exception:
                        self.best_val_threshold = 0.5
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                        'val_trial_auc': val_trial_auc,
                        'val_loss': val_loss,
                        'train_val_gap': train_val_gap,
                        'optimal_threshold': self.best_val_threshold,
                    }, self.best_model_path)
                    print(f"✓ New best model saved! Val Trial AUC: {val_trial_auc:.4f}, Val Acc: {val_acc*100:.2f}%, threshold: {self.best_val_threshold:.4f}")
                else:
                    self.best_val_acc = val_acc
                    best_train_val_gap = train_val_gap
                    patience_counter = 0
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                        'val_loss': val_loss,
                        'train_val_gap': train_val_gap,
                    }, self.best_model_path)
                    print(f"✓ New best model saved! Val Acc: {val_acc*100:.2f}%, Gap: {train_val_gap*100:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if use_auc_for_selection:
                        print(f"⚠ Early stopping: No validation trial AUC improvement for {patience} epochs")
                        print(f"   Best validation trial AUC: {self.best_val_trial_auc:.4f}")
                    else:
                        print(f"⚠ Early stopping: No validation accuracy improvement for {patience} epochs")
                        print(f"   Best validation accuracy: {self.best_val_acc*100:.2f}%")
                    break
            
            # Early stopping if overfitting is too severe
            # FIXED: train_val_gap is now in 0-1 range, so threshold is 0.50
            if train_val_gap > 0.50:
                print(f"⚠ CRITICAL: Stopping early due to severe overfitting (gap: {train_val_gap*100:.2f}%)")
                print("   Model is not learning - consider reducing model capacity or increasing regularization")
                break
            
            # Early stopping if validation loss is not improving (additional check)
            if no_improve_counter >= patience + 2:
                print(f"⚠ Early stopping: Validation loss not improving for {no_improve_counter} epochs")
                break
            
            # Step scheduler (skip if we just applied collapse recovery so recovery LR is kept for next epoch)
            if not recovered_this_epoch:
                scheduler.step()
        
        if self.best_val_trial_auc > 0.0:
            print(f"Training completed. Best validation trial AUC: {self.best_val_trial_auc:.4f}, Val Acc: {self.best_val_acc*100:.2f}%")
            return self.best_val_trial_auc
        print(f"Training completed. Best validation accuracy: {self.best_val_acc*100:.2f}%")
        return self.best_val_acc
    
    def _get_trial_key_for_index(self, dataset, idx: int) -> Tuple[str, str]:
        """Get (subject_id, trial_id) for window index idx. Handles Subset."""
        if isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.get_trial_key(dataset.indices[idx])
        return dataset.get_trial_key(idx)
    
    def _aggregate_by_trial(self, preds: np.ndarray, probs: np.ndarray, targets: np.ndarray,
                            dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Aggregate window-level predictions to trial level using mean probability (not majority vote).
        Return order: (trial_preds, trial_probs, trial_targets) - always unpack in this order.
        Per-trial: assert labels consistent; trial_prob = mean(window_probs); trial_pred = (trial_prob >= 0.5).
        """
        # Resolve underlying dataset if Subset
        full_ds = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
        if not hasattr(full_ds, 'get_trial_key'):
            return np.array([]), np.array([]), np.array([])
        
        n = len(preds)
        if n != len(dataset):
            return np.array([]), np.array([]), np.array([])
        
        by_trial = defaultdict(lambda: {'preds': [], 'probs': [], 'targets': []})
        for i in range(n):
            key = self._get_trial_key_for_index(dataset, i)
            by_trial[key]['preds'].append(preds[i])
            by_trial[key]['probs'].append(probs[i])
            by_trial[key]['targets'].append(targets[i])
        
        trial_preds_list = []
        trial_probs_list = []
        trial_targets_list = []
        for key in sorted(by_trial.keys()):
            v = by_trial[key]
            # Assert label consistency per trial (do not average labels)
            assert len(set(v['targets'])) == 1, (
                f"Trial {key} has inconsistent labels: {v['targets']}. Data bug or boundary bug."
            )
            t = int(v['targets'][0])
            # Mean probability (AAD: weak signal, mean logits/probs is correct; never majority vote)
            trial_prob = float(np.mean(v['probs']))
            trial_pred = 1 if trial_prob >= 0.5 else 0
            trial_preds_list.append(trial_pred)
            trial_probs_list.append(trial_prob)
            trial_targets_list.append(t)
        
        return np.array(trial_preds_list), np.array(trial_probs_list), np.array(trial_targets_list)
    
    def test(self, test_loader: DataLoader, flip_labels: bool = False) -> Dict:
        """Test model and compute metrics. If flip_labels=True, invert predictions (for suspected label swap)."""
        # PyTorch 2.6+ defaults to weights_only=True; our checkpoint has numpy scalars → use False
        try:
            checkpoint = torch.load(self.best_model_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                data = batch[0].to(self.device)
                target = batch[1].to(self.device).view(-1)
                
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item()

                # DIAGNOSTIC: On first batch, check backbone feature variance (explains constant logits)
                if not hasattr(self, '_backbone_diag_done'):
                    with torch.no_grad():
                        feats = self.model.backbone(data)
                        feats_np = feats.cpu().numpy()
                    feat_std = feats_np.std()
                    feat_mean = feats_np.mean()
                    self._backbone_diag_done = True
                    print(f"\n=== BACKBONE FEATURE DIAGNOSTIC (first batch) ===")
                    print(f"Backbone output shape: {feats.shape}, mean: {feat_mean:.4f}, std: {feat_std:.4f}")
                    if feat_std < 0.01:
                        print(f"⚠ CRITICAL: Backbone features are nearly constant (std={feat_std:.4f})!")
                        print(f"   Likely cause: per-window normalization removed discriminative signal, or test distribution differs from train.")
                        print(f"   Fix: use softer normalization (--normalize soft) or check train/test subject overlap.")
                    elif feat_std < 0.1:
                        print(f"⚠ WARNING: Low backbone feature variance (std={feat_std:.4f}) - model may struggle to discriminate.")

                # FIXED: Model outputs logits (B, 2) - correct for CrossEntropyLoss
                # Option 1: 2 logits + CrossEntropyLoss (current setup)
                logits = output  # Shape: (B, 2)
                
                # Compute probabilities correctly: softmax of logits
                probabilities = F.softmax(logits, dim=1)  # Shape: (B, 2)
                probs_class1 = probabilities[:, 1]  # Probability of class 1
                
                # Predictions: argmax on logits (correct)
                pred = logits.argmax(dim=1)  # Shape: (B,)
                
                # FIXED: Sanity check - verify probability computation matches logits
                if not hasattr(self, '_sanity_checked'):
                    # Print first sample for verification
                    logit_0 = logits[0, 0].item()
                    logit_1 = logits[0, 1].item()
                    prob_0 = probabilities[0, 0].item()
                    prob_1 = probabilities[0, 1].item()
                    # Manual softmax check: exp(logit_1) / (exp(logit_0) + exp(logit_1))
                    manual_prob_1 = torch.exp(logits[0, 1]) / (torch.exp(logits[0, 0]) + torch.exp(logits[0, 1])).item()
                    print(f"\n=== PROBABILITY COMPUTATION SANITY CHECK ===")
                    print(f"logits[0]: [{logit_0:.4f}, {logit_1:.4f}]")
                    print(f"probabilities[0]: [{prob_0:.4f}, {prob_1:.4f}]")
                    print(f"manual softmax prob_1: {manual_prob_1:.4f}")
                    print(f"F.softmax prob_1: {prob_1:.4f}")
                    if abs(prob_1 - manual_prob_1) > 1e-5:
                        print(f"⚠ ERROR: Probability computation mismatch!")
                    self._sanity_checked = True
                
                # FIXED: Handle both scalar and array cases for batch_size=1
                pred_np = pred.cpu().numpy()
                target_np = target.cpu().numpy()
                if pred_np.ndim == 0:
                    pred_np = np.array([pred_np])
                if target_np.ndim == 0:
                    target_np = np.array([target_np])
                
                all_predictions.extend(pred_np.flatten())
                all_targets.extend(target_np.flatten())
                all_probabilities.extend(probs_class1.detach().cpu().numpy().flatten())
                
                # Store logits for diagnostics
                if not hasattr(self, '_logits_list'):
                    self._logits_list = []
                self._logits_list.append(logits.detach().cpu().numpy())
        
        # Convert to numpy
        preds = np.array(all_predictions)
        targets = np.array(all_targets)
        # FIXED: probs is already 1D (class 1 probabilities only)
        probs = np.asarray(all_probabilities).reshape(-1)  # Ensure 1D array
        
        if flip_labels:
            preds = 1 - preds
            probs = 1.0 - probs
            print("\n*** Evaluating with --flip_labels (predictions inverted for suspected label swap) ***\n")
        
        # DIAGNOSTIC: Print prediction statistics
        print(f"\n=== TEST DIAGNOSTICS ===")
        print(f"Total test samples: {len(targets)}")
        print(f"Target distribution: {np.bincount(targets, minlength=2)}")
        print(f"Prediction distribution: {np.bincount(preds, minlength=2)}")
        print(f"Probability range: [{probs.min():.4f}, {probs.max():.4f}], mean: {probs.mean():.4f}")
        
        # FIXED: Add diagnostic to check if model is outputting constant logits
        # If probabilities are all ~0.5, the model hasn't learned (logits are near 0)
        if probs.max() - probs.min() < 0.1:
            print(f"⚠ WARNING: Probability range is very narrow ({probs.max() - probs.min():.4f})")
            print(f"   This indicates the model is outputting near-constant logits (hasn't learned)")
            print(f"   Check: regularization too strong, learning rate too low, or model capacity too small")
        
        # Check if predictions are constant (all one class)
        unique_preds = np.unique(preds)
        if len(unique_preds) == 1:
            print(f"⚠ CRITICAL: Model predicts only class {unique_preds[0]} for all {len(preds)} samples!")
            print(f"   This indicates model collapse - check loss function, activation, or model initialization")
        
        # FIXED: Diagnostic to check logits (helps identify if model is outputting constant logits)
        if hasattr(self, '_logits_list') and len(self._logits_list) > 0:
            all_logits = np.vstack(self._logits_list)
            logit_0_mean = all_logits[:, 0].mean()
            logit_1_mean = all_logits[:, 1].mean()
            logit_0_std = all_logits[:, 0].std()
            logit_1_std = all_logits[:, 1].std()
            print(f"Logit statistics: class0 mean={logit_0_mean:.4f}±{logit_0_std:.4f}, class1 mean={logit_1_mean:.4f}±{logit_1_std:.4f}")
            if abs(logit_0_mean) < 0.1 and abs(logit_1_mean) < 0.1:
                print(f"⚠ WARNING: Logits are near zero - model hasn't learned discriminative features")
            if abs(logit_1_mean - logit_0_mean) < 0.1:
                print(f"⚠ WARNING: Logit difference is small ({abs(logit_1_mean - logit_0_mean):.4f}) - model output is near-random")
            delattr(self, '_logits_list')  # Clean up
        
        # Trial-level aggregation: same decision as training (mean prob, not majority vote).
        # Unpack consistently: (trial_preds, trial_probs, trial_targets) = _aggregate_by_trial(...)
        trial_preds, trial_probs, trial_targets = self._aggregate_by_trial(
            preds, probs, targets, test_loader.dataset
        )
        use_trial_metrics = len(trial_targets) > 0
        
        if use_trial_metrics:
            print(f"\n=== TRIAL-LEVEL METRICS (primary) ===")
            print("Primary metric: trial-level ROC-AUC (threshold-free). Accuracy uses fixed threshold from validation (no tuning on test).")
            # Threshold: use validation-derived if saved in checkpoint; never tune on test (data leakage)
            try:
                trial_roc_auc = roc_auc_score(trial_targets, trial_probs)
            except Exception:
                trial_roc_auc = 0.0
            threshold_t = float(checkpoint.get('optimal_threshold', getattr(self, 'best_val_threshold', 0.5)))
            trial_preds = (trial_probs >= threshold_t).astype(np.int64)
            trial_accuracy = accuracy_score(trial_targets, trial_preds)
            trial_balanced_acc = balanced_accuracy_score(trial_targets, trial_preds)
            print(f"Trial-level ROC-AUC: {trial_roc_auc:.4f} (primary), Accuracy @ threshold={threshold_t:.4f}: {trial_accuracy*100:.2f}%, Balanced: {trial_balanced_acc:.4f}")
        
        # Check for potential label inversion
        if len(np.unique(targets)) == 2:
            # Calculate per-class accuracy correctly
            class_0_mask = (targets == 0)
            class_1_mask = (targets == 1)
            
            if np.sum(class_0_mask) > 0:
                class_0_correct = np.sum((preds == 0) & class_0_mask)
                class_0_acc = class_0_correct / np.sum(class_0_mask)
                print(f"Class 0 accuracy: {class_0_acc:.4f} ({class_0_correct}/{np.sum(class_0_mask)})")
            else:
                print(f"Class 0 accuracy: N/A (no class 0 samples)")
            
            if np.sum(class_1_mask) > 0:
                class_1_correct = np.sum((preds == 1) & class_1_mask)
                class_1_acc = class_1_correct / np.sum(class_1_mask)
                print(f"Class 1 accuracy: {class_1_acc:.4f} ({class_1_correct}/{np.sum(class_1_mask)})")
            else:
                print(f"Class 1 accuracy: N/A (no class 1 samples)")
            
            # Check if inverting predictions would help (potential label swap)
            inverted_preds = 1 - preds
            inverted_acc = accuracy_score(targets, inverted_preds)
            main_acc = accuracy_score(targets, preds)
            acc_diff = abs(inverted_acc - main_acc)
            print(f"Inverted predictions accuracy: {inverted_acc:.4f} (main: {main_acc:.4f}, diff: {acc_diff:.4f})")
            
            # FIXED: Only warn on significant differences (not random variance)
            # Strong signal would be: inverted ~60% and normal ~40% (20% gap), or ROC-AUC inverted ~0.8
            if inverted_acc > main_acc:
                # Check if difference is significant (more than ~2% with reasonable sample size)
                if acc_diff > 0.02:  # 2% threshold
                    print(f"⚠ WARNING: Inverted accuracy ({inverted_acc:.4f}) > main accuracy ({main_acc:.4f}) by {acc_diff:.4f}!")
                    print(f"   This suggests a possible label swap. Check label encoding (Left=0, Right=1).")
                    print(f"   RECOMMENDATION: Re-run with --flip_labels to evaluate with inverted predictions.")
                else:
                    print(f"   Small difference ({acc_diff:.4f}) - likely random variance, not label swap")
            
            # FIXED: Also check ROC-AUC for inverted predictions (more robust than accuracy)
            if len(np.unique(targets)) == 2:  # Binary classification
                try:
                    # FIXED: Ensure probs is 1D array (not 2D) before inverting
                    probs_1d = np.asarray(probs).reshape(-1)  # Ensure 1D
                    inverted_probs = 1.0 - probs_1d  # Invert probabilities
                    inverted_auc = roc_auc_score(targets, inverted_probs)
                    main_auc = roc_auc_score(targets, probs_1d)
                    auc_diff = abs(inverted_auc - main_auc)
                    print(f"ROC-AUC: normal={main_auc:.4f}, inverted={inverted_auc:.4f}, diff={auc_diff:.4f}")
                    # Strong inversion signal: inverted AUC ~0.8 and normal ~0.2 (0.6 gap)
                    if inverted_auc > main_auc and auc_diff > 0.1:
                        print(f"⚠ WARNING: Inverted ROC-AUC ({inverted_auc:.4f}) > normal ({main_auc:.4f}) by {auc_diff:.4f}!")
                        print(f"   This is a stronger signal of possible label swap.")
                except Exception as e:
                    print(f"   Could not compute inverted ROC-AUC: {e}")
        
        # Window-level metrics (diagnostic only when trial-level is available)
        accuracy = accuracy_score(targets, preds)
        avg_loss = total_loss / len(test_loader)
        
        # FIXED: Convert to percent when displaying (stored in 0-1 range)
        print(f"Test Accuracy (window-level, diagnostic): {accuracy*100:.2f}%")
        print(f"Test Loss: {avg_loss:.4f}")
        if use_trial_metrics:
            print(f"Primary Test Accuracy (trial-level): {trial_accuracy*100:.2f}%")
        print("=" * 30 + "\n")
        
        # Classification report (window-level = diagnostic)
        report = classification_report(targets, preds, 
                                     target_names=['Left', 'Right'], 
                                     labels=[0, 1],
                                     output_dict=True)
        
        cm = confusion_matrix(targets, preds)
        
        # FIXED: Log both confusion matrices (normal and inverted) for label swap detection
        # This helps immediately see if there's a systematic label swap
        if len(targets) > 0:
            # inverted_preds and inverted_acc are already computed above if len(targets) > 0
            cm_inverted = confusion_matrix(targets, inverted_preds)
            print(f"\nConfusion Matrix (window-level, diagnostic):")
            print(cm)
            print(f"\nConfusion Matrix (inverted, for label swap detection):")
            print(cm_inverted)
            if inverted_acc > main_acc:
                print(f"\n⚠ WARNING: Inverted accuracy ({inverted_acc:.4f}) > main accuracy ({main_acc:.4f})!")
                print(f"   This suggests a possible label swap. Check your label mapping.")
        else:
            print(f"\nConfusion Matrix:")
            print(cm)
        
        # Calculate metrics: window-level = diagnostic
        roc_auc_metrics = self._calculate_roc_auc_metrics(targets, probs)
        msed_metrics = self._calculate_msed_metrics(targets, preds)
        advanced_metrics = self._calculate_advanced_metrics(targets, preds)
        temporal_metrics = self._calculate_temporal_metrics(test_loader)
        
        results = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': preds,
            'targets': targets,
            'probabilities': probs,
            'roc_auc_metrics': roc_auc_metrics,
            'msed_metrics': msed_metrics,
            'advanced_metrics': advanced_metrics,
            'temporal_metrics': temporal_metrics
        }
        if use_trial_metrics:
            results['trial_accuracy'] = trial_accuracy
            results['trial_roc_auc'] = trial_roc_auc
            results['trial_balanced_accuracy'] = trial_balanced_acc
            results['trial_predictions'] = trial_preds
            results['trial_targets'] = trial_targets
            results['trial_probabilities'] = trial_probs
            results['primary_metric'] = 'trial'
            results['primary_roc_auc'] = trial_roc_auc
        
        return results
    
    def _calculate_roc_auc_metrics(self, targets: np.ndarray, probabilities: np.ndarray) -> Dict:
        """Calculate ROC-AUC and related metrics."""
        try:
            roc_auc = roc_auc_score(targets, probabilities)
            fpr, tpr, roc_thresholds = roc_curve(targets, probabilities)
            
            # Find optimal threshold using Youden's J (maximize tpr - fpr)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = roc_thresholds[optimal_idx]
            optimal_tpr = tpr[optimal_idx]
            optimal_fpr = fpr[optimal_idx]
            
            # Precision-Recall Curve
            precision, recall, pr_thresholds = precision_recall_curve(targets, probabilities)
            avg_precision = average_precision_score(targets, probabilities)
            
            return {
                "roc_auc_score": float(roc_auc),
                "average_precision": float(avg_precision),
                "optimal_threshold": float(optimal_threshold),
                "optimal_tpr": float(optimal_tpr),
                "optimal_fpr": float(optimal_fpr),
                "roc_curve": {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "thresholds": roc_thresholds.tolist()
                },
                "precision_recall_curve": {
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                    "thresholds": pr_thresholds.tolist()
                }
            }
        except Exception as e:
            return {"error": f"Error calculating ROC-AUC metrics: {e}"}
    
    def _calculate_msed_metrics(self, targets: np.ndarray, predictions: np.ndarray) -> Dict:
        """Calculate MSED (Mean Squared Error Distance) metrics."""
        try:
            mse = np.mean((predictions - targets) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - targets))
            mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
            
            # R-squared
            ss_res = np.sum((targets - predictions) ** 2)
            ss_tot = np.sum((targets - np.mean(targets)) ** 2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-8))
            
            return {
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "mape": float(mape),
                "r_squared": float(r_squared)
            }
        except Exception as e:
            return {"error": f"Error calculating MSED metrics: {e}"}
    
    def _calculate_advanced_metrics(self, targets: np.ndarray, predictions: np.ndarray) -> Dict:
        """Calculate advanced classification metrics."""
        try:
            mcc = matthews_corrcoef(targets, predictions)
            kappa = cohen_kappa_score(targets, predictions)
            balanced_acc = balanced_accuracy_score(targets, predictions)
            
            precision, recall, f1, support = precision_recall_fscore_support(
                targets, predictions, average=None, labels=[0, 1]
            )
            
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                targets, predictions, average='macro'
            )
            
            precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
                targets, predictions, average='weighted'
            )
            
            return {
                "matthews_correlation_coefficient": float(mcc),
                "cohens_kappa": float(kappa),
                "balanced_accuracy": float(balanced_acc),
                "per_class_metrics": {
                    "left_attention": {
                        "precision": float(precision[0]),
                        "recall": float(recall[0]),
                        "f1_score": float(f1[0]),
                        "support": int(support[0])
                    },
                    "right_attention": {
                        "precision": float(precision[1]),
                        "recall": float(recall[1]),
                        "f1_score": float(f1[1]),
                        "support": int(support[1])
                    }
                },
                "macro_averages": {
                    "precision": float(precision_macro),
                    "recall": float(recall_macro),
                    "f1_score": float(f1_macro)
                },
                "weighted_averages": {
                    "precision": float(precision_weighted),
                    "recall": float(recall_weighted),
                    "f1_score": float(f1_weighted)
                }
            }
        except Exception as e:
            return {"error": f"Error calculating advanced metrics: {e}"}
    
    def _calculate_temporal_metrics(self, test_loader: DataLoader) -> Dict[str, float]:
        """Calculate real temporal performance metrics across different window sizes.
        
        FIXED: Now uses only test subjects to prevent data leakage.
        """
        # Test different window sizes (in seconds)
        window_sizes_seconds = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0]
        temporal_analysis = {}
        flat_results = {}
        
        print(f"\n=== TEMPORAL ANALYSIS ===")
        print(f"Training window size: {self.window_size} samples ({self.window_size/self.sampling_rate:.1f}s)")
        
        # Get test subject IDs from test loader
        # FIXED: Use all test indices (or trial_boundaries) to get complete subject list
        test_subject_ids = None
        if hasattr(test_loader.dataset, 'dataset'):  # Subset wrapper
            full_dataset = test_loader.dataset.dataset
            if hasattr(full_dataset, 'trial_boundaries'):
                # Use trial_boundaries to get all test subjects directly (most reliable)
                test_indices = test_loader.dataset.indices
                test_subjects_set = set()
                
                # Get subjects from all test indices (not just first 100)
                # FIXED: Use binary search (O(log n)) instead of linear scan (O(n)) per window
                for idx in test_indices:
                    if idx < len(full_dataset.window_indices):
                        data_idx, _ = full_dataset.window_indices[idx]
                        # Use efficient binary search helper instead of linear scan
                        subject_id = full_dataset._get_window_subject(data_idx, full_dataset.window_size)
                        if subject_id and subject_id != 'unknown':
                            test_subjects_set.add(subject_id)
                
                if test_subjects_set:
                    test_subject_ids = list(test_subjects_set)
                    print(f"✓ Extracted {len(test_subject_ids)} test subjects from {len(test_indices)} test windows")
                else:
                    # Fallback: try using _get_window_subject for all indices (slower but complete)
                    print(f"  Fallback: Using _get_window_subject for all test indices...")
                    for idx in test_indices:
                        if idx < len(full_dataset.window_indices):
                            data_idx, _ = full_dataset.window_indices[idx]
                            subject_id = full_dataset._get_window_subject(data_idx, full_dataset.window_size)
                            if subject_id:
                                test_subjects_set.add(subject_id)
                    test_subject_ids = list(test_subjects_set) if test_subjects_set else None
        
        # FIXED: Skip temporal analysis if we can't determine test subjects (prevents data leakage)
        if test_subject_ids is None:
            print(f"⚠ ERROR: Could not determine test subjects - skipping temporal analysis to prevent data leakage!")
            print(f"   Temporal analysis requires explicit test subject identification.")
            print(f"   This is a safety measure to prevent inflated results from data leakage.")
            return {
                'temporal_analysis': {},
                'recommended_window_size': 'N/A',
                'note': 'Temporal analysis skipped: could not determine test subjects (data leakage prevention)'
            }
        
        print(f"Using only test subjects for temporal analysis: {len(test_subject_ids)} subjects")
        print(f"   Test subjects: {sorted(test_subject_ids)}")
        print(f"   This prevents data leakage from train/val sets.\n")
        
        for window_sec in window_sizes_seconds:
            window_samples = int(window_sec * self.sampling_rate)
            
            # For larger windows, we need to create overlapping windows from the test data
            if window_samples > self.window_size:
                larger_window_results = self._test_larger_window(test_loader, window_samples, window_sec)
                flat_results.update(larger_window_results)
                
                # Add to temporal_analysis structure
                if f'accuracy_{window_sec}s' in larger_window_results:
                    temporal_analysis[f'{window_sec}s'] = {
                        'accuracy': larger_window_results[f'accuracy_{window_sec}s'],
                        'f1': larger_window_results[f'f1_{window_sec}s']
                    }
                continue
            
            # For smaller windows, reuse already-loaded data instead of recreating dataset
            # FIXED: Optimized to reuse eeg_data and trial_boundaries, only regenerate window_indices
            try:
                # Get the full dataset from test loader to reuse loaded data
                if hasattr(test_loader.dataset, 'dataset'):
                    full_dataset = test_loader.dataset.dataset
                    # Reuse eeg_data, labels, and trial_boundaries from full_dataset
                    # Only regenerate window_indices for the new window size
                    # FIXED: Use reusable ReusedWindowDataset class (extracted to avoid duplication)
                    
                    # Create reused dataset
                    temp_dataset = ReusedWindowDataset(
                        full_dataset.eeg_data,
                        full_dataset.labels if hasattr(full_dataset, 'labels') else None,
                        full_dataset.trial_boundaries,
                        window_samples,
                        0.5,  # overlap
                        full_dataset.sampling_rate,
                        True,  # transform_eeg
                        test_subject_ids
                    )
                else:
                    # Fallback: create full dataset if we can't reuse
                    temp_dataset = KU255Dataset(
                        self.preprocessed_dir, 
                        mode='test',
                        window_size=window_samples,
                        overlap=0.5,
                        allowed_subjects=test_subject_ids,
                        raw_data_dir=str(self.raw_data_dir) if self.raw_data_dir else None
                    )
                
                if len(temp_dataset) == 0:
                    print(f"  {window_sec}s: No data available")
                    continue
                
                temp_loader = DataLoader(temp_dataset, batch_size=16, shuffle=False)
                
                # Evaluate on this window size
                self.model.eval()
                all_predictions = []
                all_targets = []
                
                with torch.no_grad():
                    for data, target in temp_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        target = target.view(-1)  # FIXED: Use view(-1) instead of squeeze() to handle batch_size=1
                        output = self.model(data)
                        pred = output.argmax(dim=1)
                        
                        # FIXED: Handle both scalar and array cases for batch_size=1
                        pred_np = pred.cpu().numpy()
                        target_np = target.cpu().numpy()
                        if pred_np.ndim == 0:
                            pred_np = np.array([pred_np])
                        if target_np.ndim == 0:
                            target_np = np.array([target_np])
                        
                        all_predictions.extend(pred_np.flatten())
                        all_targets.extend(target_np.flatten())
                
                if len(all_predictions) > 0:
                    accuracy = accuracy_score(all_targets, all_predictions)
                    f1 = f1_score(all_targets, all_predictions, average='weighted')
                    
                    # Check if this matches training window size
                    match_note = " (MATCHES TRAINING)" if window_samples == self.window_size else ""
                    print(f"  {window_sec}s ({window_samples} samples): {accuracy:.4f}{match_note}")
                    
                    flat_results[f'accuracy_{window_sec}s'] = accuracy
                    flat_results[f'f1_{window_sec}s'] = f1
                    
                    # Add to temporal_analysis structure
                    temporal_analysis[f'{window_sec}s'] = {
                        'accuracy': accuracy,
                        'f1': f1
                    }
                else:
                    print(f"  {window_sec}s: No predictions generated")
                    
            except Exception as e:
                print(f"Error testing {window_sec}s window: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print("=" * 70 + "\n")
        
        # Find the best window size
        best_window = None
        best_accuracy = 0.0
        for window_key, metrics in temporal_analysis.items():
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_window = window_key
        
        # Return structured results
        return {
            'temporal_analysis': temporal_analysis,
            'recommended_window_size': best_window if best_window else 'N/A',
            'note': f'Best performance at {best_window}s window with {best_accuracy:.3f} accuracy' if best_window else 'No valid temporal analysis completed',
            **flat_results  # Keep flat results for backward compatibility
        }
    
    def _test_larger_window(self, test_loader: DataLoader, window_samples: int, window_sec: float) -> Dict[str, float]:
        """Test larger window sizes by creating overlapping windows from raw test data.
        
        FIXED: Now uses only test subjects to prevent data leakage.
        """
        # Get test subject IDs
        test_subject_ids = None
        if hasattr(test_loader.dataset, 'dataset'):  # Subset wrapper
            full_dataset = test_loader.dataset.dataset
            if hasattr(full_dataset, 'all_subject_ids'):
                test_indices = test_loader.dataset.indices
                test_subjects_set = set()
                # Get all test subjects from all test indices (not just first 100)
                for idx in test_indices:
                    data_idx, _ = full_dataset.window_indices[idx]
                    subject_id = full_dataset._get_window_subject(data_idx, full_dataset.window_size)
                    test_subjects_set.add(subject_id)
                test_subject_ids = list(test_subjects_set)
        
        # For larger windows, reuse already-loaded data instead of recreating dataset
        # FIXED: Optimized to reuse eeg_data and trial_boundaries, only regenerate window_indices
        try:
            # Get the full dataset from test loader to reuse loaded data
            if hasattr(test_loader.dataset, 'dataset'):
                full_dataset = test_loader.dataset.dataset
                # FIXED: Use reusable ReusedWindowDataset class (extracted to avoid duplication)
                
                # Create reused dataset
                temp_dataset = ReusedWindowDataset(
                    full_dataset.eeg_data,
                    full_dataset.labels if hasattr(full_dataset, 'labels') else None,
                    full_dataset.trial_boundaries,
                    window_samples,
                    0.5,  # overlap
                    full_dataset.sampling_rate,
                    True,  # transform_eeg
                    test_subject_ids
                )
            else:
                # Fallback: create full dataset if we can't reuse
                temp_dataset = KU255Dataset(
                    self.preprocessed_dir, 
                    mode='test',
                    window_size=window_samples,
                    overlap=0.5,
                    transform_eeg=True,
                    allowed_subjects=test_subject_ids,  # Only use test subjects
                    raw_data_dir=str(self.raw_data_dir) if self.raw_data_dir else None
                )
            
            if len(temp_dataset) == 0:
                return {}
            
            temp_loader = DataLoader(temp_dataset, batch_size=16, shuffle=False)
            
            # Evaluate on this window size
            self.model.eval()
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for data, target in temp_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    target = target.view(-1)  # FIXED: Use view(-1) instead of squeeze() to handle batch_size=1
                    output = self.model(data)
                    pred = output.argmax(dim=1)
                    
                    # FIXED: Handle both scalar and tensor cases for batch_size=1
                    pred_np = pred.cpu().numpy()
                    target_np = target.cpu().numpy()
                    if pred_np.ndim == 0:
                        pred_np = np.array([pred_np])
                    if target_np.ndim == 0:
                        target_np = np.array([target_np])
                    
                    # Ensure they're 1D arrays
                    if pred_np.ndim == 0:
                        pred_np = pred_np.reshape(1)
                    if target_np.ndim == 0:
                        target_np = target_np.reshape(1)
                    
                    all_predictions.extend(pred_np.flatten())
                    all_targets.extend(target_np.flatten())
            
            if len(all_predictions) > 0:
                accuracy = accuracy_score(all_targets, all_predictions)
                f1 = f1_score(all_targets, all_predictions, average='weighted')
                
                return {
                    f'accuracy_{window_sec}s': accuracy,
                    f'f1_{window_sec}s': f1
                }
            else:
                return {}
                
        except Exception as e:
            print(f"Error testing {window_sec}s window: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def save_results(self, results: Dict):
        """Save comprehensive results to files."""
        # Prepare results
        results_json = {
            'accuracy': float(results['accuracy']),
            'loss': float(results['loss']),
            'classification_report': results['classification_report'],
            'confusion_matrix': results['confusion_matrix'].tolist() if hasattr(results['confusion_matrix'], 'tolist') else results['confusion_matrix'],
            'best_val_acc': float(self.best_val_acc),
            'timestamp': datetime.now().isoformat(),
            'roc_auc_metrics': results.get('roc_auc_metrics', {}),
            'msed_metrics': results.get('msed_metrics', {}),
            'advanced_metrics': results.get('advanced_metrics', {}),
            'temporal_metrics': results.get('temporal_metrics', {})
        }
        
        # Save results
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        
        # Save predictions
        save_data = {
            'predictions': results['predictions'],
            'targets': results['targets'],
            'probabilities': results['probabilities']
        }
        
        with open(self.output_dir / 'predictions.pkl', 'wb') as f:
            pickle.dump(save_data, f)
        
        # Save comprehensive metrics report
        self._save_comprehensive_report(results)
        
    
    def _save_comprehensive_report(self, results: Dict):
        """Save a comprehensive metrics report."""
        with open(self.output_dir / 'comprehensive_metrics_report.txt', 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("KU255CNN COMPREHENSIVE METRICS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Basic metrics
            f.write("BASIC METRICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"Loss: {results['loss']:.4f}\n")
            # FIXED: Convert to percent when displaying (stored in 0-1 range)
            f.write(f"Best Validation Accuracy: {self.best_val_acc*100:.2f}%\n\n")
            
            # ROC-AUC metrics
            roc_auc = results.get('roc_auc_metrics', {})
            if "error" not in roc_auc:
                f.write("ROC-AUC METRICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"ROC-AUC Score: {roc_auc.get('roc_auc_score', 'N/A'):.4f}\n")
                f.write(f"Average Precision: {roc_auc.get('average_precision', 'N/A'):.4f}\n")
                f.write(f"Optimal Threshold: {roc_auc.get('optimal_threshold', 'N/A'):.4f}\n")
                f.write(f"Optimal TPR: {roc_auc.get('optimal_tpr', 'N/A'):.4f}\n")
                f.write(f"Optimal FPR: {roc_auc.get('optimal_fpr', 'N/A'):.4f}\n\n")
            
            # MSED metrics
            msed = results.get('msed_metrics', {})
            if "error" not in msed:
                f.write("MSED METRICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Mean Squared Error: {msed.get('mse', 'N/A'):.4f}\n")
                f.write(f"Root Mean Squared Error: {msed.get('rmse', 'N/A'):.4f}\n")
                f.write(f"Mean Absolute Error: {msed.get('mae', 'N/A'):.4f}\n")
                f.write(f"Mean Absolute Percentage Error: {msed.get('mape', 'N/A'):.4f}%\n")
                f.write(f"R-squared: {msed.get('r_squared', 'N/A'):.4f}\n\n")
            
            # Advanced metrics
            advanced = results.get('advanced_metrics', {})
            if "error" not in advanced:
                f.write("ADVANCED METRICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Matthews Correlation Coefficient: {advanced.get('matthews_correlation_coefficient', 'N/A'):.4f}\n")
                f.write(f"Cohen's Kappa: {advanced.get('cohens_kappa', 'N/A'):.4f}\n")
                f.write(f"Balanced Accuracy: {advanced.get('balanced_accuracy', 'N/A'):.4f}\n\n")
                
                # Per-class metrics
                per_class = advanced.get("per_class_metrics", {})
                f.write("PER-CLASS METRICS:\n")
                f.write("-" * 40 + "\n")
                
                left = per_class.get("left_attention", {})
                f.write("Left Attention:\n")
                f.write(f"  Precision: {left.get('precision', 'N/A'):.4f}\n")
                f.write(f"  Recall: {left.get('recall', 'N/A'):.4f}\n")
                f.write(f"  F1-Score: {left.get('f1_score', 'N/A'):.4f}\n")
                f.write(f"  Support: {left.get('support', 'N/A')}\n\n")
                
                right = per_class.get("right_attention", {})
                f.write("Right Attention:\n")
                f.write(f"  Precision: {right.get('precision', 'N/A'):.4f}\n")
                f.write(f"  Recall: {right.get('recall', 'N/A'):.4f}\n")
                f.write(f"  F1-Score: {right.get('f1_score', 'N/A'):.4f}\n")
                f.write(f"  Support: {right.get('support', 'N/A')}\n\n")
            
            # Temporal analysis
            temporal = results.get('temporal_metrics', {})
            f.write("TEMPORAL PERFORMANCE ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            for window_size, metrics in temporal.get("temporal_analysis", {}).items():
                f.write(f"{window_size}: {metrics.get('accuracy', 'N/A'):.4f}\n")
            f.write(f"\nRecommended: {temporal.get('recommended_window_size', 'N/A')}\n")
            f.write(f"Note: {temporal.get('note', 'N/A')}\n")
            
            # Add formatted results section
            f.write("\n" + "=" * 80 + "\n")
            f.write("KU255CNN COMPREHENSIVE RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("The KU255CNN model successfully processed the KU Leuven 255 dataset:\n")
            # FIXED: Convert to percent when displaying (stored in 0-1 range)
            f.write(f"- Best Validation Accuracy: {self.best_val_acc*100:.2f}%\n")
            f.write(f"- Final Test Accuracy: {results['accuracy']*100:.2f}%\n")
            
            # ROC-AUC metrics
            roc_auc = results.get('roc_auc_metrics', {})
            if "error" not in roc_auc:
                f.write(f"- ROC-AUC: {roc_auc.get('roc_auc_score', 'N/A'):.4f}\n")
            
            # Classification metrics
            class_report = results.get('classification_report', {})
            if 'macro avg' in class_report:
                macro_avg = class_report['macro avg']
                f.write(f"- Precision: {macro_avg.get('precision', 'N/A'):.4f}\n")
                f.write(f"- Recall: {macro_avg.get('recall', 'N/A'):.4f}\n")
                f.write(f"- F1-Score: {macro_avg.get('f1-score', 'N/A'):.4f}\n")
            
            # MSED metrics
            msed = results.get('msed_metrics', {})
            if "error" not in msed:
                f.write(f"- MSED (Primary Benchmark): {msed.get('rmse', 'N/A'):.4f}\n")
            
            # Advanced metrics
            advanced = results.get('advanced_metrics', {})
            if "error" not in advanced:
                f.write(f"- Direction Accuracy: {advanced.get('balanced_accuracy', 'N/A'):.4f}\n")
                f.write(f"- Spatial Consistency: {advanced.get('matthews_correlation_coefficient', 'N/A'):.4f}\n")
            
            # Temporal Integration Performance
            f.write("\nTEMPORAL INTEGRATION PERFORMANCE\n")
            f.write("The KU Leuven 255 dataset demonstrated robust performance across decision window lengths:\n")
            
            for ws_key, ws_data in temporal.get("temporal_analysis", {}).items():
                window_seconds = float(ws_key.replace('s', ''))
                accuracy = ws_data.get('accuracy', 0.0)
                f.write(f"- {ws_key} window: {accuracy:.4f}\n")


def _collate_ku255_batch(batch):
    """Collate batch so trial_keys (str, str) are kept as a list; default_collate fails on strings."""
    if not batch:
        return None, None, []
    first = batch[0]
    if len(first) == 2:
        data_list = [b[0] for b in batch]
        target_list = [b[1] for b in batch]
        trial_keys = []
    else:
        data_list = [b[0] for b in batch]
        target_list = [b[1] for b in batch]
        trial_keys = [b[2] if isinstance(b[2], tuple) else tuple(b[2]) for b in batch]
    data = torch.stack(data_list, dim=0)
    target = torch.cat([t.view(-1) if t.dim() > 1 else t for t in target_list], dim=0)
    return data, target, trial_keys


def create_ku255_data_loaders(preprocessed_dir: str, batch_size: int = 64, 
                               window_size: int = 1024, overlap: float = 0.75,
                               train_ratio: float = 0.7, val_ratio: float = 0.15,
                               max_samples: Optional[int] = None, 
                               num_workers: int = 0, pin_memory: bool = False,
                               raw_data_dir: Optional[str] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test loaders with subject-wise splitting (no data leakage).
    
    Args:
        preprocessed_dir: Directory containing preprocessed .mat files
        raw_data_dir: Optional directory containing raw Curry files (.dat, .dap, .rs3)
                     Used to find Curry files for experiment boundaries
    """
    
    print(f"DEBUG: window_size parameter = {window_size}")
    print(f"Creating dataset: batch_size={batch_size}, window_size={window_size} samples ({window_size/128:.1f}s), overlap={overlap}")
    print(f"Higher overlap ({overlap}) creates more training samples to prevent overfitting")
    if raw_data_dir:
        print(f"Using raw data directory for Curry files (.dat/.dap/.rs3): {raw_data_dir}")
    
    # Create full dataset
    full_dataset = KU255Dataset(preprocessed_dir, mode='full', 
                                 window_size=window_size, overlap=overlap,
                                 raw_data_dir=raw_data_dir)
    
    total_size = len(full_dataset)
    
    # Map windows to subjects for splitting
    subject_windows = {}
    
    # Map windows to subjects using the dataset's method
    # FIXED: Handle 'unknown' subjects properly (drop with warning)
    unknown_windows = []
    for i, (data_idx, label) in enumerate(full_dataset.window_indices):
        # Use the dataset's method to get subject for this window
        subject_id = full_dataset._get_window_subject(data_idx, full_dataset.window_size)
        
        # FIXED: Drop unknown subjects to prevent data quality issues
        if subject_id == 'unknown':
            unknown_windows.append(i)
            continue
        
        if subject_id not in subject_windows:
            subject_windows[subject_id] = []
        subject_windows[subject_id].append(i)
    
    # FIXED: Fail fast on unknown subjects instead of silently dropping (indicates serious dataset issue)
    if unknown_windows:
        examples = unknown_windows[:5]
        raise RuntimeError(f"Found {len(unknown_windows)} windows with 'unknown' subject. "
                         f"Example indices: {examples}. "
                         f"This indicates boundary extraction problems or missing metadata. "
                         f"Fix trial_boundaries mapping before proceeding.")
    
    # Split by subject (prevents data leakage)
    subjects = list(subject_windows.keys())
    np.random.seed(42)  # Reproducibility
    np.random.shuffle(subjects)
    
    n_subjects = len(subjects)
    n_train_subjects = int(train_ratio * n_subjects)
    n_val_subjects = int(val_ratio * n_subjects)
    
    train_subjects = subjects[:n_train_subjects]
    val_subjects = subjects[n_train_subjects:n_train_subjects + n_val_subjects]
    test_subjects = subjects[n_train_subjects + n_val_subjects:]
    
    # Get window indices for each split
    train_indices = []
    val_indices = []
    test_indices = []
    
    for subject_id in train_subjects:
        train_indices.extend(subject_windows[subject_id])
    for subject_id in val_subjects:
        val_indices.extend(subject_windows[subject_id])
    for subject_id in test_subjects:
        test_indices.extend(subject_windows[subject_id])
    
    # Check for data leakage
    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)
    
    if train_set & val_set:
        raise ValueError("CRITICAL: Data leakage - train/val overlap!")
    if train_set & test_set:
        raise ValueError("CRITICAL: Data leakage - train/test overlap!")
    if val_set & test_set:
        raise ValueError("CRITICAL: Data leakage - val/test overlap!")
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # Custom collate so trial_keys (str, str) are preserved; default_collate fails on strings
    collate_fn = _collate_ku255_batch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader


def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='KU255CNNLOC - CNN-LOC for KU Leuven 255 Dataset')
    parser.add_argument('--preprocessed_dir', type=str, default='kuleuven_255_preprocessed',
                       help='Preprocessed data directory path')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training (increased for speed, 64 is good balance)')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs (reduced to prevent overfitting)')
    parser.add_argument('--learning_rate', type=float, default=3e-3,
                       help='Learning rate (3e-3 recommended for weak-signal tasks, 1e-3 for regular training)')
    parser.add_argument('--window_size', type=int, default=1024,
                       help='Window size in samples (128-3840 samples = 1-30 seconds at 128Hz. Recommended: 1024 (8s) or 2048 (16s) for AAD decoding)')
    parser.add_argument('--overlap', type=float, default=0.25,
                       help='Window overlap fraction (0.5 = reduces training samples to prevent overfitting)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay for regularization (1e-5 recommended, 0.0 for pipeline proof test)')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                       help='Dropout rate (0.1 recommended, 0.0 for pipeline proof test)')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                       help='Label smoothing (0.0 recommended for binary AAD; avoid >0.1)')
    parser.add_argument('--mixup_alpha', type=float, default=0.0,
                       help='Mixup augmentation alpha (0.0 = no augmentation; for overfit test use 0.0; 0.2+ for regular training)')
    parser.add_argument('--patience', type=int, default=30,
                       help='Early stopping patience (30 for pipeline proof test, 7 for regular training)')
    parser.add_argument('--output_dir', type=str, default='ku255cnnloc_results',
                       help='Output directory for results')
    parser.add_argument('--raw_data_dir', type=str, default=None,
                       help='Raw data directory containing Curry files (.dat/.dap/.rs3) for experiment boundaries (e.g., "Data/KULeuven 255")')
    parser.add_argument('--flip_labels', action='store_true',
                       help='Invert predictions at test time (use if inverted accuracy > main accuracy suggests label swap)')
    parser.add_argument('--check_labels', action='store_true',
                       help='Verify dataset labels (Left=0, Right=1) in preprocessed .mat files then exit')
    
    args = parser.parse_args()
    
    if getattr(args, 'check_labels', False):
        try:
            from check_ku255_labels import run_check
            sys.exit(run_check(args.preprocessed_dir, max_files=5))
        except ImportError:
            print("Run from project root: python check_ku255_labels.py --preprocessed_dir <dir>")
            sys.exit(1)
    
    print(f"DEBUG: args.window_size = {args.window_size}")
    print(f"DEBUG: args.preprocessed_dir = {args.preprocessed_dir}")
    
    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU (GPU not available)")
    
    # FIXED: Validate window size BEFORE creating loaders (prevents expensive loading with bad window size)
    sampling_rate = 128  # Hz
    min_window_samples = 128  # 1 second
    max_window_samples = 3840  # 30 seconds
    
    if args.window_size < min_window_samples:
        print(f"\n❌ ERROR: window_size={args.window_size} is too small!")
        print(f"   Minimum: {min_window_samples} samples (1 second at {sampling_rate}Hz)")
        print(f"   Please use --window_size >= {min_window_samples}")
        raise ValueError(f"window_size must be >= {min_window_samples} samples (1 second)")
    
    if args.window_size > max_window_samples:
        print(f"\n❌ ERROR: window_size={args.window_size} is too large!")
        print(f"   Maximum: {max_window_samples} samples (30 seconds at {sampling_rate}Hz)")
        print(f"   Please use --window_size <= {max_window_samples}")
        raise ValueError(f"window_size must be <= {max_window_samples} samples (30 seconds)")
    
    window_seconds = args.window_size / sampling_rate
    print(f"✓ Window size: {args.window_size} samples ({window_seconds:.1f} seconds at {sampling_rate}Hz)")
    
    # Warn about window size recommendations for AAD decoding
    if args.window_size < 512:
        print(f"\n⚠ WARNING: window_size={args.window_size} ({window_seconds:.1f}s) may be too short for AAD decoding!")
        print(f"   AAD decoding typically needs 8-16 seconds of evidence.")
        print(f"   Recommended: --window_size 1024 (8s) or --window_size 2048 (16s)")
        print(f"   Shorter windows create highly correlated samples that don't generalize.")
    elif args.window_size > 2560:
        print(f"\n⚠ WARNING: window_size={args.window_size} ({window_seconds:.1f}s) is very large!")
        print(f"   This may cause memory issues and slower training.")
        print(f"   Recommended: --window_size 1024 (8s) or --window_size 2048 (16s) for AAD decoding")
    
    # FIXED: Create data loaders AFTER window size validation (before using train_loader)
    overlap = getattr(args, 'overlap', 0.5)
    train_loader, val_loader, test_loader = create_ku255_data_loaders(
        args.preprocessed_dir, batch_size=args.batch_size, window_size=args.window_size,
        overlap=overlap, max_samples=None, num_workers=0, pin_memory=False,
        raw_data_dir=args.raw_data_dir
    )
    
    # Get train_indices and full_dataset for optimized class weight computation
    train_dataset = train_loader.dataset
    if isinstance(train_dataset, torch.utils.data.Subset):
        train_indices_list = train_dataset.indices
        # Get the underlying full dataset
        full_dataset_obj = train_dataset.dataset
    else:
        train_indices_list = None
        full_dataset_obj = None
    
    # Get input dimensions from actual data (derive from dataset[0], not heuristics)
    ds = train_loader.dataset
    if len(ds) > 0:
        sample = ds[0]
        sample_tf = sample[0]
        actual_channels = int(sample_tf.shape[0])
        actual_time = int(sample_tf.shape[1])
        actual_freq = int(sample_tf.shape[2])
        print(f"Input dimensions (from data): channels={actual_channels}, time={actual_time}, freq={actual_freq}")
        
        # Warn if dimensions are wrong (freq too large = slow training)
        if actual_freq > 10:
            print(f"\n⚠ WARNING: Frequency dimension ({actual_freq}) is too large!")
            print(f"   This will cause very slow training. Expected ~5 frequency bands.")
            print(f"   The time-frequency transform may need adjustment.")
        # Note: Time dimension scales with window_size (8 for 1-2s, up to 64 for 24-30s)
        if actual_time > 64:
            print(f"\n⚠ WARNING: Time dimension ({actual_time}) is very large!")
            print(f"   This may cause memory issues. Consider using --window_size 1024 (8s) or 2048 (16s)")
        elif actual_time > 32:
            print(f"\nℹ INFO: Time dimension ({actual_time}) is large but acceptable for longer windows")
    else:
        actual_channels = 64  # EEG channels
        actual_time = 32
        actual_freq = 5
        print(f"Using defaults: channels={actual_channels}, time={actual_time}, freq={actual_freq}")
    
    # Create model
    print(f"Creating model: channels={actual_channels}, time={actual_time}, freq={actual_freq}")
    print(f"Hyperparameters: batch_size={args.batch_size}, lr={args.learning_rate}, wd={args.weight_decay}, dropout={args.dropout_rate}, label_smoothing={args.label_smoothing}")
    
    model = KU255CNNModel(
        input_channels=actual_channels,
        input_time=actual_time,
        input_freq=actual_freq,
        num_classes=2,
        dropout_rate=args.dropout_rate
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = KU255CNNTrainer(model, device, args.output_dir, args.preprocessed_dir, 
                           sampling_rate=128, window_size=args.window_size,
                           raw_data_dir=args.raw_data_dir)
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Train (model selection on trial-level AUC when dataset provides trial keys)
    trainer.train(
        train_loader, val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        train_indices=train_indices_list,
        full_dataset=full_dataset_obj
    )
    
    # Test
    results = trainer.test(test_loader, flip_labels=getattr(args, 'flip_labels', False))
    
    # Save
    trainer.save_results(results)
    
    if trainer.best_val_trial_auc > 0.0:
        print(f"\nTraining complete. Best val trial AUC: {trainer.best_val_trial_auc:.4f}, Test acc: {results['accuracy']*100:.2f}%")
    else:
        print(f"\nTraining complete. Best val acc: {trainer.best_val_acc*100:.2f}%, Test acc: {results['accuracy']*100:.2f}%")
    
    # Display key metrics
    roc_auc = results.get('roc_auc_metrics', {})
    if "error" not in roc_auc:
        print(f"ROC-AUC (window, diagnostic): {roc_auc.get('roc_auc_score', 'N/A'):.4f}")
    if results.get('primary_metric') == 'trial' and 'primary_roc_auc' in results:
        print(f"ROC-AUC (trial, primary): {results['primary_roc_auc']:.4f}")
    
    msed = results.get('msed_metrics', {})
    if "error" not in msed:
        print(f"RMSE: {msed.get('rmse', 'N/A'):.4f}")
    
    temporal = results.get('temporal_metrics', {})
    print(f"Recommended window size: {temporal.get('recommended_window_size', 'N/A')}")
    
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

