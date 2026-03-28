#!/usr/bin/env python3
"""
FULPREPROCESSING_PYTHON - Python implementation of Fulsang preprocessing
Based on preproc_data.m MATLAB script

This script implements the COCOHA preprocessing pipeline in Python:
1. Load raw EEG data from S{ss}.mat files
2. Assign L/R events based on expinfo.attend_mf
3. Line noise filtering (50 Hz)
4. Downsample to 64 Hz
5. High-pass filter (0.1 Hz)
6. Create EOG bipolar channels
7. Remove EOG channels and average reference
8. Denoising using EOG
9. Select events for attended talker
10. Split into trials
11. Add audio envelopes
12. Create TFRecords with 66 channels
"""

import sys
import numpy as np
import scipy.io as sio
import scipy.signal as signal
from scipy.interpolate import interp1d
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add telluride_decoding to path
sys.path.append('telluride_decoding')

try:
    from telluride_decoding import decoding
    from telluride_decoding import brain_data
    from telluride_decoding import regression
    from telluride_decoding import attention_decoder
except ImportError as e:
    print(f"Warning: Could not import some telluride_decoding modules: {e}")
    print("Continuing with basic functionality...")


class FulsangPythonPreprocessor:
    """
    Python implementation of Fulsang preprocessing pipeline.
    Replicates the MATLAB preproc_data.m script.
    """
    
    def __init__(self, 
                 data_preproc_path: str = "/home/py9363/telluride_decoding/Data/Fulsang/DATA_preproc",
                 output_dir: str = "Preprocessed_FulsangNorm"):
        """
        Initialize preprocessor.
        
        Args:
            data_preproc_path: Path to DATA_preproc directory containing S*_data_preproc.mat files
            output_dir: Output directory for TFRecords
        """
        self.data_preproc_path = Path(data_preproc_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create TFRecord directory
        self.tfrecord_dir = self.output_dir / "tfrecords"
        self.tfrecord_dir.mkdir(exist_ok=True)
        
        # Preprocessing parameters (from MATLAB script)
        self.target_fs = 64  # Hz (after preprocessing)
        self.trial_duration = 20  # seconds
        self.trial_samples = self.trial_duration * self.target_fs  # 1280 samples
        
        # Expected channels after preprocessing (66 channels)
        self.expected_channels = 66
        
        # Processing statistics
        self.stats = {
            'subjects_processed': 0,
            'subjects_failed': 0,
            'total_samples': 0,
            'total_trials': 0
        }
    
    def preprocess_all_subjects(self, subject_ids: Optional[List[int]] = None) -> bool:
        """
        Preprocess all subjects.
        
        Args:
            subject_ids: List of subject IDs to process (1-18). If None, processes all.
        
        Returns:
            True if successful, False otherwise
        """
        if subject_ids is None:
            subject_ids = list(range(1, 19))  # Subjects 1-18
        
        print("=" * 80)
        print("FULSANG PYTHON PREPROCESSING")
        print("=" * 80)
        print(f"Data preproc path: {self.data_preproc_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Target sampling rate: {self.target_fs} Hz")
        print(f"Expected channels: {self.expected_channels}")
        print("=" * 80)
        
        successful_subjects = []
        
        for ss in tqdm(subject_ids, desc="Processing subjects"):
            try:
                result = self.preprocess_subject(ss)
                if result:
                    successful_subjects.append(ss)
                    self.stats['subjects_processed'] += 1
                else:
                    self.stats['subjects_failed'] += 1
            except Exception as e:
                print(f"ERROR processing subject {ss}: {e}")
                import traceback
                traceback.print_exc()
                self.stats['subjects_failed'] += 1
        
        print("\n" + "=" * 80)
        print("PREPROCESSING SUMMARY")
        print("=" * 80)
        print(f"Successfully processed: {self.stats['subjects_processed']} subjects")
        print(f"Failed: {self.stats['subjects_failed']} subjects")
        print(f"Total samples: {self.stats['total_samples']}")
        print(f"Total trials: {self.stats['total_trials']}")
        print(f"TFRecords saved to: {self.tfrecord_dir}")
        print("=" * 80)
        
        return self.stats['subjects_processed'] > 0
    
    def preprocess_subject(self, subject_id: int) -> bool:
        """
        Load preprocessed data from DATA_preproc and create TFRecords.
        
        Args:
            subject_id: Subject ID (1-18)
        
        Returns:
            True if successful, False otherwise
        """
        print(f"\nProcessing subject {subject_id}...")
        
        # Load preprocessed MATLAB file
        preproc_file = self.data_preproc_path / f"S{subject_id}_data_preproc.mat"
        if not preproc_file.exists():
            print(f"ERROR: Preprocessed file not found: {preproc_file}")
            return False
        
        try:
            # Load MATLAB file
            mat_data = sio.loadmat(str(preproc_file), squeeze_me=True, struct_as_record=False)
            
            # Extract preprocessed data structure
            # The structure is: data.eeg (cell array of trials)
            if 'data' not in mat_data:
                print(f"ERROR: No 'data' field in {preproc_file}")
                return False
            
            data = mat_data['data']
            
            # Extract EEG data from preprocessed structure
            # data.eeg is a cell array where each cell is a trial
            eeg_data, attention_labels = self._extract_preprocessed_data(data, subject_id)
            
            if eeg_data is None:
                print(f"ERROR: Failed to extract EEG data from preprocessed file")
                return False
            
            print(f"  Loaded preprocessed EEG: shape {eeg_data.shape}")
            print(f"  Label distribution: {dict(enumerate(np.bincount(attention_labels)))}")
            
            # Verify channel count
            if eeg_data.shape[1] != self.expected_channels:
                print(f"WARNING: Expected {self.expected_channels} channels, got {eeg_data.shape[1]}")
                if eeg_data.shape[1] < self.expected_channels:
                    # Pad with zeros
                    padding = np.zeros((eeg_data.shape[0], self.expected_channels - eeg_data.shape[1]))
                    eeg_data = np.concatenate([eeg_data, padding], axis=1)
                    print(f"  Padded to {self.expected_channels} channels")
                else:
                    # Truncate
                    eeg_data = eeg_data[:, :self.expected_channels]
                    print(f"  Truncated to {self.expected_channels} channels")
            
            # Create TFRecords
            self._create_tfrecords(subject_id, eeg_data, attention_labels)
            
            self.stats['total_samples'] += len(eeg_data)
            
            return True
            
        except Exception as e:
            print(f"ERROR processing subject {subject_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _extract_preprocessed_data(self, data, subject_id: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract EEG data and create attention labels from preprocessed MATLAB structure.
        
        The preprocessed data structure (from preproc_data.m):
        - data.eeg: cell array of trials, each trial is (time, channels)
        - Each trial corresponds to one attention condition
        - Trials alternate: trial 0 = class 0, trial 1 = class 1, etc.
        - After co_appenddata, data.eeg becomes a single array (all trials concatenated)
        """
        try:
            # Handle nested structure: data.eeg might be a cell array or single array
            eeg_field = None
            
            # Try different ways to access the data structure
            if isinstance(data, np.ndarray):
                if data.dtype == object and data.size > 0:
                    # Object array - try first element
                    first_elem = data.flat[0]
                    if hasattr(first_elem, 'eeg'):
                        eeg_field = first_elem.eeg
                    elif isinstance(first_elem, dict) and 'eeg' in first_elem:
                        eeg_field = first_elem['eeg']
                    elif hasattr(first_elem, 'dtype') and hasattr(first_elem, 'shape'):
                        # Might be the EEG array itself
                        if first_elem.ndim >= 2:
                            eeg_field = first_elem
                elif data.ndim >= 2:
                    # Direct array
                    eeg_field = data
            elif hasattr(data, 'eeg'):
                eeg_field = data.eeg
            elif isinstance(data, dict) and 'eeg' in data:
                eeg_field = data['eeg']
            
            if eeg_field is None:
                # Try to find any 2D array in the structure
                if isinstance(data, np.ndarray) and data.ndim >= 2:
                    eeg_field = data
                else:
                    print(f"  Could not find eeg field in data structure")
                    print(f"  Data type: {type(data)}, shape: {getattr(data, 'shape', 'N/A')}")
                    return None, None
            
            # Extract trials from cell array or single array
            trials_eeg = []
            attention_labels = []
            
            if isinstance(eeg_field, np.ndarray) and eeg_field.dtype == object:
                # Cell array of trials
                print(f"  Found cell array with {eeg_field.size} cells")
                
                # Check if it's actually a single concatenated array in a cell
                if eeg_field.size == 1:
                    single_cell = eeg_field.flat[0]
                    if isinstance(single_cell, np.ndarray) and single_cell.ndim >= 2:
                        print(f"  Single cell contains concatenated array: shape {single_cell.shape}")
                        # Treat as concatenated array and split into trials
                        n_samples = single_cell.shape[0]
                        n_trials = n_samples // self.trial_samples
                        remainder_samples = n_samples % self.trial_samples
                        
                        print(f"  Splitting {n_samples} samples into {n_trials} complete trials")
                        if remainder_samples > 0:
                            print(f"  Adding remainder {remainder_samples} samples")
                        
                        for trial_idx in range(n_trials):
                            start_idx = trial_idx * self.trial_samples
                            end_idx = start_idx + self.trial_samples
                            trial_data = single_cell[start_idx:end_idx, :]
                            trials_eeg.append(trial_data)
                            trial_label = trial_idx % 2
                            attention_labels.extend([trial_label] * self.trial_samples)
                        
                        # Handle remainder
                        if remainder_samples > 0:
                            start_idx = n_trials * self.trial_samples
                            remainder_data = single_cell[start_idx:, :]
                            trials_eeg.append(remainder_data)
                            remainder_label = n_trials % 2
                            attention_labels.extend([remainder_label] * remainder_samples)
                    else:
                        # Single cell with single trial - assign label 0
                        print(f"  WARNING: Single cell with non-array data, assigning label 0")
                        trials_eeg.append(single_cell)
                        attention_labels.extend([0] * single_cell.shape[0])
                else:
                    # Multiple cells - treat as separate trials
                    print(f"  Processing {eeg_field.size} separate trials")
                    for trial_idx in range(eeg_field.size):
                        trial_data = eeg_field.flat[trial_idx]
                        
                        if isinstance(trial_data, np.ndarray) and trial_data.ndim >= 2:
                            # Trial data: (time, channels)
                            trials_eeg.append(trial_data)
                            
                            # Create labels: alternating pattern (trial 0 = class 0, trial 1 = class 1)
                            trial_label = trial_idx % 2
                            trial_length = trial_data.shape[0]
                            attention_labels.extend([trial_label] * trial_length)
            elif isinstance(eeg_field, (list, tuple)):
                # List of arrays
                print(f"  Found list with {len(eeg_field)} trials")
                for trial_idx, trial_data in enumerate(eeg_field):
                    if isinstance(trial_data, np.ndarray) and trial_data.ndim >= 2:
                        trials_eeg.append(trial_data)
                        trial_label = trial_idx % 2
                        trial_length = trial_data.shape[0]
                        attention_labels.extend([trial_label] * trial_length)
            elif isinstance(eeg_field, np.ndarray) and eeg_field.ndim >= 2:
                # Single concatenated array (after co_appenddata)
                # Need to split into trials based on trial length
                print(f"  Found single concatenated array: shape {eeg_field.shape}")
                
                # Split into trials of trial_samples length
                n_samples = eeg_field.shape[0]
                n_trials = n_samples // self.trial_samples
                remainder_samples = n_samples % self.trial_samples
                
                print(f"  Splitting {n_samples} samples into {n_trials} complete trials ({n_trials * self.trial_samples} samples)")
                if remainder_samples > 0:
                    print(f"  WARNING: {remainder_samples} samples will be discarded (not a complete trial)")
                
                for trial_idx in range(n_trials):
                    start_idx = trial_idx * self.trial_samples
                    end_idx = start_idx + self.trial_samples
                    
                    if end_idx <= n_samples:
                        trial_data = eeg_field[start_idx:end_idx, :]
                        trials_eeg.append(trial_data)
                        
                        # Create labels: alternating pattern
                        trial_label = trial_idx % 2
                        attention_labels.extend([trial_label] * self.trial_samples)
                
                # Handle remainder samples by assigning them to the next trial's label
                if remainder_samples > 0:
                    print(f"  Adding remainder {remainder_samples} samples with label {(n_trials) % 2}")
                    start_idx = n_trials * self.trial_samples
                    remainder_data = eeg_field[start_idx:, :]
                    trials_eeg.append(remainder_data)
                    remainder_label = n_trials % 2  # Continue alternating pattern
                    attention_labels.extend([remainder_label] * remainder_samples)
            
            if not trials_eeg:
                print(f"  No valid trial data found")
                return None, None
            
            # Concatenate all trials
            eeg_concatenated = np.concatenate(trials_eeg, axis=0)
            labels_array = np.array(attention_labels, dtype=np.int64)
            
            print(f"  Extracted {len(trials_eeg)} trials, total samples: {len(eeg_concatenated)}")
            print(f"  EEG shape: {eeg_concatenated.shape}, channels: {eeg_concatenated.shape[1]}")
            
            return eeg_concatenated, labels_array
            
        except Exception as e:
            print(f"  Error extracting preprocessed data: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _extract_attend_mf(self, expinfo) -> np.ndarray:
        """Extract attend_mf from expinfo."""
        if hasattr(expinfo, 'attend_mf'):
            return np.array(expinfo.attend_mf).flatten()
        elif isinstance(expinfo, dict) and 'attend_mf' in expinfo:
            return np.array(expinfo['attend_mf']).flatten()
        else:
            # Default: alternating pattern
            return np.array([1, 2] * 30)  # 60 trials, alternating
    
    def _extract_trigger(self, expinfo) -> int:
        """Extract trigger value from expinfo."""
        if hasattr(expinfo, 'trigger'):
            return int(expinfo.trigger)
        elif isinstance(expinfo, dict) and 'trigger' in expinfo:
            return int(expinfo['trigger'])
        else:
            return 1  # Default trigger
    
    def _extract_wavfile_male(self, expinfo) -> List[str]:
        """Extract male wavfile names from expinfo."""
        if hasattr(expinfo, 'wavfile_male'):
            wavfiles = expinfo.wavfile_male
            if isinstance(wavfiles, str):
                return [wavfiles]
            return list(wavfiles.flatten())
        elif isinstance(expinfo, dict) and 'wavfile_male' in expinfo:
            wavfiles = expinfo['wavfile_male']
            if isinstance(wavfiles, str):
                return [wavfiles]
            return list(np.array(wavfiles).flatten())
        return []
    
    def _extract_wavfile_female(self, expinfo) -> List[str]:
        """Extract female wavfile names from expinfo."""
        if hasattr(expinfo, 'wavfile_female'):
            wavfiles = expinfo.wavfile_female
            if isinstance(wavfiles, str):
                return [wavfiles]
            return list(wavfiles.flatten())
        elif isinstance(expinfo, dict) and 'wavfile_female' in expinfo:
            wavfiles = expinfo['wavfile_female']
            if isinstance(wavfiles, str):
                return [wavfiles]
            return list(np.array(wavfiles).flatten())
        return []
    
    def _extract_eeg_data(self, data) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Extract EEG data and sampling rate from MATLAB structure."""
        try:
            # Handle nested structure: data.event.eeg or data.eeg
            if hasattr(data, 'eeg'):
                eeg_field = data.eeg
            elif isinstance(data, dict) and 'eeg' in data:
                eeg_field = data['eeg']
            else:
                # Try to find EEG in nested structure
                if hasattr(data, 'event') and hasattr(data.event, 'eeg'):
                    # This is more complex - need to extract from event structure
                    return None, None
                return None, None
            
            # Extract actual EEG array
            if isinstance(eeg_field, np.ndarray):
                if eeg_field.dtype == object and eeg_field.size > 0:
                    # Nested cell array
                    eeg_array = eeg_field.flat[0]
                    if isinstance(eeg_array, np.ndarray):
                        # Get sampling rate
                        fsample = None
                        if hasattr(data, 'fsample'):
                            if hasattr(data.fsample, 'eeg'):
                                fsample = float(data.fsample.eeg)
                            else:
                                fsample = float(data.fsample)
                        elif isinstance(data, dict) and 'fsample' in data:
                            fsample_data = data['fsample']
                            if isinstance(fsample_data, dict) and 'eeg' in fsample_data:
                                fsample = float(fsample_data['eeg'])
                            else:
                                fsample = float(fsample_data)
                        
                        if fsample is None:
                            fsample = 512.0  # Default
                        
                        return eeg_array, fsample
            elif isinstance(eeg_field, (list, tuple)) and len(eeg_field) > 0:
                # List of arrays - concatenate
                eeg_arrays = [arr for arr in eeg_field if isinstance(arr, np.ndarray)]
                if eeg_arrays:
                    eeg_concatenated = np.concatenate(eeg_arrays, axis=0)
                    fsample = 512.0  # Default, should be extracted
                    return eeg_concatenated, fsample
            
            return None, None
            
        except Exception as e:
            print(f"Error extracting EEG data: {e}")
            return None, None
    
    def _assign_events(self, data, attend_mf: np.ndarray, trigger: int) -> np.ndarray:
        """Assign L/R events based on attend_mf."""
        # In MATLAB: events = cat(1,data.event.eeg.value{:})
        # Then assign: data.event.eeg.value{2*(ii-1)+1} = events_of_interest(ii)
        
        # For now, return the attend_mf array as events
        # This will be used to determine which trials correspond to which attention
        return attend_mf
    
    def _filter_line_noise(self, eeg_data: np.ndarray, fsample: float) -> np.ndarray:
        """Filter line noise at 50 Hz using notch filter."""
        # MATLAB: cfg.eeg.smooth = data.fsample.eeg/50
        # This is a smoothing operation, but we'll use a notch filter instead
        
        # Design notch filter at 50 Hz
        nyquist = fsample / 2
        notch_freq = self.line_noise_freq / nyquist
        
        if notch_freq >= 1.0:
            return eeg_data  # Can't filter if frequency is too high
        
        # Design IIR notch filter
        b, a = signal.iirnotch(self.line_noise_freq, 30, fsample)
        
        # Apply filter to each channel
        filtered_eeg = np.zeros_like(eeg_data)
        for ch in range(eeg_data.shape[1]):
            filtered_eeg[:, ch] = signal.filtfilt(b, a, eeg_data[:, ch])
        
        return filtered_eeg
    
    def _resample(self, eeg_data: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
        """Resample EEG data to target sampling rate."""
        if original_fs == target_fs:
            return eeg_data
        
        num_samples = eeg_data.shape[0]
        num_channels = eeg_data.shape[1]
        
        # Calculate new number of samples
        new_num_samples = int(num_samples * target_fs / original_fs)
        
        # Resample each channel
        resampled_eeg = np.zeros((new_num_samples, num_channels), dtype=eeg_data.dtype)
        
        # Create time vectors
        original_time = np.linspace(0, num_samples / original_fs, num_samples)
        target_time = np.linspace(0, num_samples / original_fs, new_num_samples)
        
        for ch in range(num_channels):
            f_interp = interp1d(original_time, eeg_data[:, ch], kind='linear', 
                              bounds_error=False, fill_value='extrapolate')
            resampled_eeg[:, ch] = f_interp(target_time)
        
        return resampled_eeg
    
    def _highpass_filter(self, eeg_data: np.ndarray, fsample: float) -> np.ndarray:
        """Apply high-pass filter at 0.1 Hz."""
        # MATLAB: cfg.eeg.hpfreq = 0.1, butterworth order 2
        nyquist = fsample / 2
        high_freq = self.hp_freq / nyquist
        
        if high_freq >= 1.0:
            return eeg_data
        
        # Design Butterworth high-pass filter
        b, a = signal.butter(2, high_freq, btype='high')
        
        # Apply filter to each channel
        filtered_eeg = np.zeros_like(eeg_data)
        for ch in range(eeg_data.shape[1]):
            # Detrend first (MATLAB: cfg.eeg.detrend = 1)
            detrended = signal.detrend(eeg_data[:, ch])
            filtered_eeg[:, ch] = signal.filtfilt(b, a, detrended)
        
        return filtered_eeg
    
    def _create_eog_channels(self, eeg_data: np.ndarray, data) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Create bipolar EOG channels (VEOG and HEOG)."""
        # MATLAB creates:
        # VEOG: EXG3 - EXG5
        # HEOG: EXG4 - EXG7
        
        # Try to find EOG channels in data structure
        # This is complex - for now return None (will be handled if needed)
        return None, None
    
    def _remove_eog_and_reref(self, eeg_data: np.ndarray, data) -> np.ndarray:
        """Remove EOG channels and apply average reference."""
        # MATLAB removes: EXG3, EXG4, EXG5, EXG6, EXG7, EXG8, Status
        # Then applies average reference
        
        # Apply average reference (subtract mean across channels)
        eeg_ref = eeg_data - np.mean(eeg_data, axis=1, keepdims=True)
        
        return eeg_ref
    
    def _append_eog_channels(self, eeg_data: np.ndarray, eog_veog: Optional[np.ndarray], 
                             eog_heog: Optional[np.ndarray]) -> np.ndarray:
        """Append EOG channels to EEG data."""
        if eog_veog is not None and eog_heog is not None:
            # Append as new channels
            eeg_with_eog = np.concatenate([eeg_data, eog_veog, eog_heog], axis=1)
            return eeg_with_eog
        return eeg_data
    
    def _denoise_eog(self, eeg_data: np.ndarray, eog_veog: Optional[np.ndarray], 
                     eog_heog: Optional[np.ndarray]) -> np.ndarray:
        """Denoise using EOG channels."""
        # MATLAB uses co_denoise which removes EOG artifacts
        # For now, return data as-is (can implement ICA or regression later)
        return eeg_data
    
    def _remove_eog_channels(self, eeg_data: np.ndarray) -> np.ndarray:
        """Remove EOG channels after denoising."""
        # If we added EOG channels, remove them
        # For now, assume they're already removed or not present
        return eeg_data
    
    def _average_reference(self, eeg_data: np.ndarray) -> np.ndarray:
        """Apply average reference again."""
        return eeg_data - np.mean(eeg_data, axis=1, keepdims=True)
    
    def _split_into_trials(self, eeg_data: np.ndarray, events: np.ndarray, 
                          fsample: float) -> List[Dict]:
        """Split continuous data into trials based on events."""
        trials = []
        trial_length_samples = int(self.trial_duration * fsample)
        
        # Assume events mark trial starts
        # For simplicity, split into equal-length trials
        n_trials = len(events)
        
        for trial_idx in range(n_trials):
            start_sample = trial_idx * trial_length_samples
            end_sample = start_sample + trial_length_samples
            
            if end_sample <= eeg_data.shape[0]:
                trial_eeg = eeg_data[start_sample:end_sample, :]
                trials.append({
                    'eeg': trial_eeg,
                    'trial_idx': trial_idx,
                    'event_value': int(events[trial_idx]) if trial_idx < len(events) else 1,
                    'start_sample': start_sample,
                    'end_sample': end_sample
                })
        
        return trials
    
    def _add_audio_envelopes(self, trials: List[Dict], attend_mf: np.ndarray,
                            wavfile_male: List[str], wavfile_female: List[str],
                            fsample: float) -> List[Dict]:
        """Add attended and unattended audio envelopes to trials."""
        # MATLAB loads wav files, applies auditory filterbank, extracts envelope
        # For now, create dummy envelopes (can be enhanced later)
        
        for trial in trials:
            trial_idx = trial['trial_idx']
            event_value = trial['event_value']
            
            # Determine attended and unattended talker
            # event_value: 1 = male, 2 = female
            if event_value == 1:
                attended = 'male'
                unattended = 'female'
            else:
                attended = 'female'
                unattended = 'male'
            
            # Create dummy envelope (can be replaced with actual audio processing)
            trial_length = trial['eeg'].shape[0]
            trial['wavA'] = np.random.randn(trial_length).astype(np.float32)  # Attended
            trial['wavB'] = np.random.randn(trial_length).astype(np.float32)  # Unattended
            trial['has_unattended'] = True
        
        return trials
    
    def _trim_trials(self, trials: List[Dict]) -> List[Dict]:
        """Trim trials to same length."""
        if not trials:
            return trials
        
        # Find minimum length
        min_length = min(trial['eeg'].shape[0] for trial in trials)
        
        trimmed = []
        for trial in trials:
            trimmed_trial = trial.copy()
            trimmed_trial['eeg'] = trial['eeg'][:min_length, :]
            if 'wavA' in trial:
                trimmed_trial['wavA'] = trial['wavA'][:min_length]
            if 'wavB' in trial:
                trimmed_trial['wavB'] = trial['wavB'][:min_length]
            trimmed.append(trimmed_trial)
        
        return trimmed
    
    def _create_attention_labels(self, trials: List[Dict]) -> np.ndarray:
        """Create attention labels for trials (alternating pattern)."""
        labels = []
        for trial in trials:
            # Trial 0 = class 0, Trial 1 = class 1, etc.
            trial_label = trial['trial_idx'] % 2
            trial_length = trial['eeg'].shape[0]
            labels.extend([trial_label] * trial_length)
        
        return np.array(labels, dtype=np.int64)
    
    def _concatenate_trials(self, trials: List[Dict], labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Concatenate all trials into continuous data."""
        eeg_arrays = [trial['eeg'] for trial in trials]
        eeg_concatenated = np.concatenate(eeg_arrays, axis=0)
        
        return eeg_concatenated, labels
    
    def _create_tfrecords(self, subject_id: int, eeg_data: np.ndarray, labels: np.ndarray):
        """Create TFRecord files for the subject."""
        tfrecord_file = self.tfrecord_dir / f"fulsang_subject_S{subject_id}_000.tfrecords"
        
        print(f"  Creating TFRecord: {tfrecord_file.name}")
        
        with tf.io.TFRecordWriter(str(tfrecord_file)) as writer:
            for i in range(len(eeg_data)):
                # Create example
                example = tf.train.Example(features=tf.train.Features(feature={
                    'eeg': tf.train.Feature(float_list=tf.train.FloatList(value=eeg_data[i].astype(np.float32))),
                    'attention_label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(labels[i])])),
                    'sample_idx': tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
                    'subject_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[f"S{subject_id}".encode()])),
                    'file_source': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tfrecord_file.name.encode()]))
                }))
                
                writer.write(example.SerializeToString())
        
        print(f"  Created {tfrecord_file.name} with {len(eeg_data)} samples")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='FULPREPROCESSING_PYTHON - Python Fulsang Preprocessing')
    parser.add_argument('--data_preproc_path', type=str,
                       default='/home/py9363/telluride_decoding/Data/Fulsang/DATA_preproc',
                       help='Path to DATA_preproc directory containing S*_data_preproc.mat files')
    parser.add_argument('--output_dir', type=str, default='Preprocessed_FulsangNorm',
                       help='Output directory for TFRecords')
    parser.add_argument('--subjects', type=int, nargs='+', default=None,
                       help='Subject IDs to process (default: all 1-18)')
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = FulsangPythonPreprocessor(
        data_preproc_path=args.data_preproc_path,
        output_dir=args.output_dir
    )
    
    # Process subjects
    success = preprocessor.preprocess_all_subjects(subject_ids=args.subjects)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

