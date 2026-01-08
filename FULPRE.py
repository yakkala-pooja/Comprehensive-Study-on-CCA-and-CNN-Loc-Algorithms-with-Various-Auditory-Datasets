#!/usr/bin/env python3
"""
FULPRE - Fulsang Dataset Preprocessing Script

This script preprocesses the Fulsang dataset from preprocessed MATLAB files.
Based on the MATLAB preprocessing script (preproc_data.m) and experimental design.

The script:
1. Loads preprocessed MATLAB files from Data/Fulsang/DATA_preproc/
2. Extracts EEG data (66 channels, 60 trials, 3200 samples per trial)
3. Extracts audio envelopes (wavA and wavB)
4. Extracts attention labels from event structure (attend_mf: 1=male, 2=female)
5. Creates TFRecord files for training
6. Handles all 18 subjects (S1-S18)

Experimental information (expinfo):
- attend_mf: attended speaker (1=male, 2=female)
- attend_lr: spatial position (1=left, 2=right)
- acoustic_condition: room type (1=anechoic, 2=mild reverb, 3=high reverb)
- n_speakers: number of speakers (1 or 2)
- wavfile_male: audio file for male speaker
- wavfile_female: audio file for female speaker
- trigger: trigger event value for each trial
"""

import sys
import numpy as np
import scipy.io as sio
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class FulsangPreprocessor:
    """
    Main preprocessor for Fulsang dataset.
    Loads preprocessed MATLAB files and converts them to TFRecord format.
    """
    
    # Preprocessing version for reproducibility
    PREPROCESSING_VERSION = "2.0.0"  # Major version bump for refactored code
    
    def __init__(self, 
                 data_dir: str = "Data/Fulsang",
                 output_dir: str = "fulsang_preprocessed",
                 sampling_rate: int = 64,
                 n_channels: int = 66,
                 trial_length: int = 3200,
                 filter_n_speakers: int = 2,
                 require_audio: bool = False):
        """
        Initialize the preprocessor.
        
        Args:
            data_dir: Directory containing Fulsang data
            output_dir: Directory to save preprocessed TFRecord files
            sampling_rate: EEG sampling rate (64 Hz for Fulsang)
            n_channels: Number of EEG channels (66 for Fulsang, or 72 if including EOG)
            trial_length: Number of samples per trial (3200 = 50 seconds at 64 Hz)
            filter_n_speakers: Only include trials with this many speakers (default: 2 for AAD)
            require_audio: If True, fail if audio extraction fails. If False, skip audio fields when missing.
        """
        self.data_dir = Path(data_dir)
        self.preproc_dir = self.data_dir / "DATA_preproc"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create TFRecord output directory
        self.tfrecord_dir = self.output_dir / "tfrecords"
        self.tfrecord_dir.mkdir(exist_ok=True)
        
        # Dataset parameters
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        self.trial_length = trial_length
        self.n_trials_per_subject = 60
        self.filter_n_speakers = filter_n_speakers
        self.require_audio = require_audio
        
        # Processing statistics
        self.stats = {
            'subjects_processed': 0,
            'subjects_failed': 0,
            'total_trials': 0,
            'total_trials_filtered': 0,
            'total_samples': 0,
            'start_time': None,
            'end_time': None
        }
    
    def preprocess_all_subjects(self, subject_ids: Optional[List[int]] = None) -> bool:
        """
        Preprocess all subjects in the dataset.
        
        Args:
            subject_ids: List of subject IDs to process (1-18). If None, processes all.
        
        Returns:
            True if successful, False otherwise
        """
        self.stats['start_time'] = datetime.now()
        
        # Find all MATLAB files
        if subject_ids is None:
            mat_files = sorted(list(self.preproc_dir.glob("S*_data_preproc.mat")))
        else:
            mat_files = []
            for subj_id in subject_ids:
                mat_file = self.preproc_dir / f"S{subj_id}_data_preproc.mat"
                if mat_file.exists():
                    mat_files.append(mat_file)
        
        if not mat_files:
            print(f"ERROR: No MATLAB files found in {self.preproc_dir}")
            return False
        
        print(f"Found {len(mat_files)} MATLAB files to process")
        print(f"Output directory: {self.tfrecord_dir}")
        print("="*70)
        
        # Process each subject
        successful_subjects = []
        for mat_file in tqdm(mat_files, desc="Processing subjects"):
            subject_id = self._extract_subject_id(mat_file)
            print(f"\nProcessing {subject_id} ({mat_file.name})...")
            
            try:
                result = self.preprocess_subject(mat_file)
                if result:
                    successful_subjects.append(subject_id)
                    self.stats['subjects_processed'] += 1
                    self.stats['total_trials'] += result['n_trials']
                    self.stats['total_trials_filtered'] += result['n_trials_valid']
                    self.stats['total_samples'] += result['n_samples']
                    print(f"[OK] Successfully processed {subject_id} ({result['n_trials_valid']}/{result['n_trials']} trials valid)")
                else:
                    self.stats['subjects_failed'] += 1
                    print(f"[FAIL] Failed to process {subject_id}")
            except Exception as e:
                print(f"[ERROR] Error processing {subject_id}: {e}")
                import traceback
                traceback.print_exc()
                self.stats['subjects_failed'] += 1
        
        self.stats['end_time'] = datetime.now()
        
        # Print summary
        self._print_summary()
        
        return self.stats['subjects_processed'] > 0
    
    def preprocess_subject(self, mat_file: Path) -> Optional[Dict]:
        """
        Preprocess a single subject's data.
        
        Args:
            mat_file: Path to the MATLAB file
        
        Returns:
            Dictionary with processing results or None if failed
        """
        try:
            # Load MATLAB file
            print(f"  Loading {mat_file.name}...")
            mat_data = sio.loadmat(str(mat_file), squeeze_me=False, struct_as_record=False)
            
            if 'data' not in mat_data:
                raise RuntimeError(f"No 'data' field in {mat_file.name}")
            
            data_struct = mat_data['data']
            
            # Extract expinfo first (needed for labels and filtering)
            # Note: expinfo may be missing - will return minimal dict with None values
            expinfo = self._extract_expinfo(data_struct, mat_data)
            
            # Extract trial-level data (keep as list of trials, don't concatenate)
            eeg_trials = self._extract_eeg_trials(data_struct)
            if eeg_trials is None or len(eeg_trials) == 0:
                raise RuntimeError(f"Failed to extract EEG trials from {mat_file.name}")
            
            # Verify extracted trials have correct shape
            for i, trial in enumerate(eeg_trials):
                if trial.shape != (self.trial_length, self.n_channels):
                    raise RuntimeError(
                        f"Extracted trial {i} has incorrect shape {trial.shape}, "
                        f"expected ({self.trial_length}, {self.n_channels}). "
                        f"This will cause TFRecord corruption!"
                    )
            
            # Extract audio trials
            wavA_trials, wavB_trials = self._extract_audio_trials(data_struct)
            if self.require_audio:
                if wavA_trials is None or wavB_trials is None:
                    raise RuntimeError(f"Failed to extract audio envelopes from {mat_file.name} (require_audio=True)")
            else:
                if wavA_trials is None:
                    wavA_trials = [None] * len(eeg_trials)
                if wavB_trials is None:
                    wavB_trials = [None] * len(eeg_trials)
            
            # Extract trial-level attention labels (no expansion to sample level)
            attention_labels = self._extract_attention_labels(data_struct, expinfo, len(eeg_trials))
            if attention_labels is None:
                raise RuntimeError(f"Failed to extract attention labels from {mat_file.name}")
            
            # Validate per-trial consistency
            n_trials = len(eeg_trials)
            if len(attention_labels) != n_trials:
                raise RuntimeError(f"Mismatch: {n_trials} EEG trials but {len(attention_labels)} labels")
            
            # Validate audio trial counts match EEG
            if wavA_trials is not None and len(wavA_trials) != n_trials:
                raise RuntimeError(f"Mismatch: {n_trials} EEG trials but {len(wavA_trials)} wavA trials")
            if wavB_trials is not None and len(wavB_trials) != n_trials:
                raise RuntimeError(f"Mismatch: {n_trials} EEG trials but {len(wavB_trials)} wavB trials")
            
            # Note: n_speakers may be unavailable - skip filtering by n_speakers
            # If n_speakers is available, we could filter, but it's not required
            
            # Note: attend_mf validation happens in _extract_attention_labels()
            # which can fallback to event structure, so we don't require it here
            
            # Helper function to safely extract scalar from expinfo field
            def get_expinfo_scalar(field_name, trial_idx):
                """Extract value from expinfo field for given trial index."""
                value = expinfo.get(field_name)
                if value is None:
                    return None
                if isinstance(value, (list, np.ndarray)):
                    if trial_idx < len(value):
                        return value[trial_idx]
                    return None
                return value
            
            # Helper function to convert any value to scalar
            def to_scalar(val):
                """
                Canonical scalar conversion: unwraps arrays, lists, converts to Python scalar.
                Returns None if value cannot be converted to a scalar (multi-element arrays/lists).
                """
                if val is None:
                    return None
                # Handle lists/tuples
                if isinstance(val, (list, tuple)):
                    if len(val) == 0:
                        return None
                    if len(val) == 1:
                        val = val[0]
                    else:
                        # Multi-element list - cannot convert to scalar
                        return None
                # Handle numpy arrays
                if isinstance(val, np.ndarray):
                    val = np.array(val).squeeze()
                    if val.size == 0:
                        return None
                    if val.size == 1:
                        return val.item()
                    # Multi-element array - cannot convert to scalar
                    return None
                return val
            
            # Filter trials based on n_speakers and validate per-trial consistency
            valid_trials = []
            filtered_reasons = {
                'n_speakers_missing': 0, 'n_speakers_invalid': 0, 'n_speakers_mismatch': 0,
                'trial_length_mismatch': 0, 
                'channel_mismatch': 0, 
                'audio_mismatch': 0
            }
            
            for trial_idx in range(n_trials):
                # Note: n_speakers filtering is disabled since n_speakers is unavailable
                # If n_speakers becomes available in the future, filtering can be re-enabled
                
                # Trust attention_labels[trial_idx] - already validated in _extract_attention_labels()
                # Labels may come from expinfo.attend_mf or event structure (trigger codes)
                
                # Validate trial consistency
                eeg_trial = eeg_trials[trial_idx]
                if eeg_trial.shape[0] != self.trial_length:
                    filtered_reasons['trial_length_mismatch'] += 1
                    continue
                
                if eeg_trial.shape[1] != self.n_channels:
                    filtered_reasons['channel_mismatch'] += 1
                    continue
                
                # Check audio consistency if available
                if wavA_trials is not None and wavA_trials[trial_idx] is not None:
                    if wavA_trials[trial_idx].shape[0] != self.trial_length:
                        filtered_reasons['audio_mismatch'] += 1
                        continue
                if wavB_trials is not None and wavB_trials[trial_idx] is not None:
                    if wavB_trials[trial_idx].shape[0] != self.trial_length:
                        filtered_reasons['audio_mismatch'] += 1
                        continue
                
                valid_trials.append(trial_idx)
            
            # Deduplicate valid_trials (safety check, though shouldn't be needed after fixing duplicate)
            valid_trials = sorted(set(valid_trials))
            
            if len(valid_trials) == 0:
                raise RuntimeError(f"No valid trials after filtering. Filter reasons: {filtered_reasons}")
            
            print(f"  Extracted {n_trials} trials, {len(valid_trials)} valid after filtering")
            print(f"    Filter reasons: {filtered_reasons}")
            print(f"    Label distribution: {dict(enumerate(np.bincount([attention_labels[i] for i in valid_trials], minlength=2)))}")
            
            # Create TFRecord file
            subject_id = self._extract_subject_id(mat_file)
            tfrecord_file = self.tfrecord_dir / f"fulsang_{subject_id}.tfrecords"
            
            print(f"  Creating TFRecord file: {tfrecord_file.name}")
            n_trials_written = self._create_tfrecord_file(
                tfrecord_file, eeg_trials, wavA_trials, wavB_trials,
                attention_labels, expinfo, valid_trials, subject_id
            )
            
            # Write manifest JSON with filtering information
            manifest_file = self.tfrecord_dir / f"fulsang_{subject_id}_manifest.json"
            self._write_manifest(manifest_file, subject_id, n_trials, valid_trials, 
                               attention_labels, expinfo, filtered_reasons)
            
            return {
                'subject_id': subject_id,
                'n_trials': n_trials,
                'n_trials_valid': len(valid_trials),
                'n_trials_written': n_trials_written,
                'n_samples': n_trials_written * self.trial_length,
                'tfrecord_file': str(tfrecord_file),
                'manifest_file': str(manifest_file)
            }
            
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            return None
        except Exception as e:
            print(f"  ERROR: Exception during preprocessing: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_expinfo(self, data_struct, mat_data: Dict) -> Optional[Dict]:
        """
        Extract expinfo structure from data_struct or mat_data.
        
        Returns:
            Dictionary with expinfo fields: attend_mf, attend_lr, acoustic_condition, n_speakers, etc.
        """
        try:
            if not isinstance(data_struct, np.ndarray) or data_struct.size == 0:
                return None
            
            first_elem = data_struct.flat[0]
            expinfo = None
            
            # Method 1: Try data_struct.flat[0].expinfo (most common in COCOHA-style)
            if hasattr(first_elem, 'expinfo'):
                expinfo = first_elem.expinfo
            elif hasattr(first_elem, 'dtype') and hasattr(first_elem.dtype, 'names') and 'expinfo' in first_elem.dtype.names:
                expinfo = first_elem['expinfo']
            
            # Method 2: Try top-level mat_data['expinfo']
            if expinfo is None and 'expinfo' in mat_data:
                expinfo = mat_data['expinfo']
            
            # If expinfo is missing, return minimal dict with None values
            if expinfo is None:
                return {
                    "attend_mf": None,
                    "attend_lr": None,
                    "acoustic_condition": None,
                    "n_speakers": None,
                    "wavfile_male": None,
                    "wavfile_female": None,
                    "trigger": None
                }
            
            # Extract fields from expinfo
            result = {
                "attend_mf": None,
                "attend_lr": None,
                "acoustic_condition": None,
                "n_speakers": None,
                "wavfile_male": None,
                "wavfile_female": None,
                "trigger": None
            }
            fields_to_extract = ['attend_mf', 'attend_lr', 'acoustic_condition', 'n_speakers', 
                               'wavfile_male', 'wavfile_female', 'trigger']
            
            for field in fields_to_extract:
                try:
                    if hasattr(expinfo, field):
                        value = getattr(expinfo, field)
                    elif isinstance(expinfo, dict) and field in expinfo:
                        value = expinfo[field]
                    elif isinstance(expinfo, np.ndarray):
                        if expinfo.dtype.names and field in expinfo.dtype.names:
                            value = expinfo[field]
                        elif expinfo.size > 0:
                            first_expinfo = expinfo.flat[0]
                            if hasattr(first_expinfo, 'dtype') and first_expinfo.dtype.names:
                                if field in first_expinfo.dtype.names:
                                    value = first_expinfo[field]
                                else:
                                    value = None
                            else:
                                value = None
                        else:
                            value = None
                    else:
                        value = None
                    
                    if value is not None:
                        # Normalize arrays: flatten and handle nested structures
                        if isinstance(value, np.ndarray):
                            # Handle object arrays (cell arrays) - extract each element
                            if value.dtype == object:
                                # For cell arrays, extract each element
                                extracted = []
                                for i in range(value.size):
                                    elem = value.flat[i]
                                    if isinstance(elem, np.ndarray):
                                        elem = elem.flatten()
                                    extracted.append(elem)
                                value = extracted
                            else:
                                # For numeric arrays, flatten
                                value = value.flatten()
                        
                        # For string fields (wavfile_male, wavfile_female), handle bytes/char arrays
                        if field in ['wavfile_male', 'wavfile_female']:
                            if isinstance(value, (list, np.ndarray)):
                                # Convert each element to string
                                str_list = []
                                for v in value:
                                    if isinstance(v, bytes):
                                        str_list.append(v.decode('utf-8', errors='ignore'))
                                    elif isinstance(v, np.ndarray) and v.dtype.char == 'U':
                                        str_list.append(str(v))
                                    elif isinstance(v, str):
                                        str_list.append(v)
                                    else:
                                        str_list.append(str(v))
                                value = str_list
                        
                        result[field] = value
                except Exception as e:
                    # Log but don't fail - some fields may be missing
                    result[field] = None
            
            return result
            
        except Exception as e:
            print(f"    Error extracting expinfo: {e}")
            # Return minimal dict instead of None
            return {
                "attend_mf": None,
                "attend_lr": None,
                "acoustic_condition": None,
                "n_speakers": None,
                "wavfile_male": None,
                "wavfile_female": None,
                "trigger": None
            }
    
    def _extract_eeg_trials(self, data_struct) -> Optional[List[np.ndarray]]:
        """
        Extract EEG data from MATLAB structure as list of trials.
        
        The structure is: data.eeg (cell array of 60 trials, each 3200 x 66)
        
        Returns:
            List of trial arrays, each (trial_length, n_channels)
        """
        try:
            # Access the nested structure: data.eeg
            if not isinstance(data_struct, np.ndarray) or data_struct.size == 0:
                return None
            
            first_elem = data_struct.flat[0]
            
            # Handle mat_struct objects (struct_as_record=False) - use attribute access
            if hasattr(first_elem, 'eeg'):
                eeg_field = first_elem.eeg
            # Handle structured arrays (struct_as_record=True) - use dictionary access
            elif hasattr(first_elem, 'dtype') and hasattr(first_elem.dtype, 'names') and 'eeg' in first_elem.dtype.names:
                eeg_field = first_elem['eeg']
            else:
                return None
            
            # eeg_field is a cell array (object array) with 60 trials
            if not isinstance(eeg_field, np.ndarray) or eeg_field.dtype != object:
                return None
            
            if eeg_field.size == 0:
                return None
            
            # Extract trials as separate arrays (don't concatenate)
            all_trials = []
            for i in range(eeg_field.size):
                trial_data = eeg_field.flat[i]
                
                # Handle different array shapes
                if isinstance(trial_data, np.ndarray):
                    if trial_data.ndim == 1:
                        # 1D array - might be a single sample or transposed
                        # If it's 66 elements, it's probably one sample (transposed)
                        # If it's 3200 elements, it's probably one channel
                        if trial_data.size == self.n_channels:
                            # This is likely one sample (66 channels) - skip or pad
                            print(f"    WARNING: Trial {i} is 1D with {trial_data.size} elements (expected 2D {self.trial_length}x{self.n_channels})")
                            continue
                        elif trial_data.size == self.trial_length:
                            # This is one channel - reshape to (trial_length, 1) and pad
                            trial_data = trial_data.reshape(-1, 1)
                            padding = np.zeros((trial_data.shape[0], self.n_channels - 1), dtype=np.float32)
                            trial_data = np.concatenate([trial_data, padding], axis=1)
                            all_trials.append(trial_data.astype(np.float32))
                        else:
                            print(f"    WARNING: Trial {i} is 1D with unexpected size {trial_data.size}, skipping")
                            continue
                    elif trial_data.ndim == 2:
                        # 2D array - check orientation
                        # Expected: (trial_length, n_channels) = (3200, 66)
                        # If we get (66, 3200), transpose it
                        if trial_data.shape[0] == self.n_channels and trial_data.shape[1] == self.trial_length:
                            # Transposed: (66, 3200) -> (3200, 66)
                            trial_data = trial_data.T
                        elif trial_data.shape[0] == self.trial_length and trial_data.shape[1] == self.n_channels:
                            # Correct orientation: (3200, 66)
                            pass
                        elif trial_data.shape[1] == self.n_channels:
                            # Has correct number of channels in second dimension
                            # Assume first dimension is time (might be shorter than trial_length)
                            if trial_data.shape[0] < self.trial_length:
                                # Pad with zeros
                                padding = np.zeros((self.trial_length - trial_data.shape[0], self.n_channels), dtype=np.float32)
                                trial_data = np.concatenate([trial_data, padding], axis=0)
                            elif trial_data.shape[0] > self.trial_length:
                                # Truncate
                                trial_data = trial_data[:self.trial_length, :]
                        elif trial_data.shape[0] == self.n_channels:
                            # Has correct number of channels in first dimension - transpose
                            trial_data = trial_data.T
                            if trial_data.shape[0] < self.trial_length:
                                padding = np.zeros((self.trial_length - trial_data.shape[0], self.n_channels), dtype=np.float32)
                                trial_data = np.concatenate([trial_data, padding], axis=0)
                            elif trial_data.shape[0] > self.trial_length:
                                trial_data = trial_data[:self.trial_length, :]
                        else:
                            # Try to fix by ensuring correct number of channels
                            if trial_data.shape[1] < self.n_channels:
                                # Pad with zeros
                                padding = np.zeros((trial_data.shape[0], self.n_channels - trial_data.shape[1]), dtype=np.float32)
                                trial_data = np.concatenate([trial_data.astype(np.float32), padding], axis=1)
                            elif trial_data.shape[1] > self.n_channels:
                                # Truncate
                                trial_data = trial_data[:, :self.n_channels]
                        
                        # Final validation
                        if trial_data.shape == (self.trial_length, self.n_channels):
                            all_trials.append(trial_data.astype(np.float32))
                        else:
                            print(f"    WARNING: Trial {i} has shape {trial_data.shape}, expected ({self.trial_length}, {self.n_channels}), skipping")
                    else:
                        print(f"    WARNING: Trial {i} has {trial_data.ndim} dimensions, expected 2, skipping")
                else:
                    print(f"    WARNING: Trial {i} is not a numpy array, skipping")
            
            if not all_trials:
                print(f"    ERROR: No valid trials extracted from EEG field (size: {eeg_field.size})")
                # Debug: show what we got
                if eeg_field.size > 0:
                    first_trial = eeg_field.flat[0]
                    print(f"    First trial type: {type(first_trial)}, shape: {getattr(first_trial, 'shape', 'N/A')}, ndim: {getattr(first_trial, 'ndim', 'N/A')}")
                return None
            
            # Verify all trials have correct shape
            for i, trial in enumerate(all_trials):
                if trial.shape != (self.trial_length, self.n_channels):
                    print(f"    WARNING: Trial {i} has incorrect shape {trial.shape}, expected ({self.trial_length}, {self.n_channels})")
            
            print(f"    Extracted {len(all_trials)} valid EEG trials (all shape: {all_trials[0].shape})")
            return all_trials
            
        except Exception as e:
            print(f"    Error extracting EEG trials: {e}")
            return None
    
    def _extract_audio_trials(self, data_struct) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
        """
        Extract audio envelope data (wavA and wavB) from MATLAB structure as list of trials.
        
        Supports both:
        - Cell arrays (object dtype): list of trial arrays
        - Numeric arrays: (trial_length, n_trials) or (trial_length, 1, n_trials), etc.
        
        Returns:
            (wavA_trials, wavB_trials) - both are lists of (trial_length, 1) arrays
        """
        try:
            if not isinstance(data_struct, np.ndarray) or data_struct.size == 0:
                return None, None
            
            first_elem = data_struct.flat[0]
            
            wavA_trials = None
            wavB_trials = None
            
            # Extract wavA - handle both mat_struct and structured array
            wavA_field = None
            if hasattr(first_elem, 'wavA'):
                wavA_field = first_elem.wavA
            elif hasattr(first_elem, 'dtype') and hasattr(first_elem.dtype, 'names') and 'wavA' in first_elem.dtype.names:
                wavA_field = first_elem['wavA']
            
            if wavA_field is not None:
                if isinstance(wavA_field, np.ndarray):
                    if wavA_field.dtype == object:
                        # Cell array (object dtype) - extract each trial
                        if wavA_field.size > 0:
                            all_trials = []
                            for i in range(wavA_field.size):
                                trial_data = wavA_field.flat[i]
                                if isinstance(trial_data, np.ndarray):
                                    if trial_data.ndim == 1:
                                        trial_data = trial_data.reshape(-1, 1)
                                    elif trial_data.ndim > 2:
                                        trial_data = trial_data.reshape(trial_data.shape[0], -1)
                                        # Ensure single feature
                                        if trial_data.shape[1] > 1:
                                            trial_data = trial_data[:, 0:1]
                                    all_trials.append(trial_data.astype(np.float32))
                            
                            if all_trials:
                                wavA_trials = all_trials
                    else:
                        # Numeric array - reshape to extract trials
                        # Common shapes: (trial_length, n_trials) or (trial_length, 1, n_trials)
                        if wavA_field.ndim == 2:
                            # Shape: (trial_length, n_trials)
                            if wavA_field.shape[0] == self.trial_length:
                                n_trials = wavA_field.shape[1]
                                wavA_trials = [wavA_field[:, i:i+1].astype(np.float32) for i in range(n_trials)]
                            elif wavA_field.shape[1] == self.trial_length:
                                # Transposed: (n_trials, trial_length)
                                n_trials = wavA_field.shape[0]
                                wavA_trials = [wavA_field[i:i+1, :].T.astype(np.float32) for i in range(n_trials)]
                        elif wavA_field.ndim == 3:
                            # Shape: (trial_length, 1, n_trials) or similar
                            if wavA_field.shape[0] == self.trial_length:
                                n_trials = wavA_field.shape[2]
                                wavA_trials = [wavA_field[:, 0, i:i+1].astype(np.float32) for i in range(n_trials)]
                            elif wavA_field.shape[2] == self.trial_length:
                                n_trials = wavA_field.shape[0]
                                wavA_trials = [wavA_field[i, 0, :].reshape(-1, 1).astype(np.float32) for i in range(n_trials)]
            
            # Extract wavB - handle both mat_struct and structured array
            wavB_field = None
            if hasattr(first_elem, 'wavB'):
                wavB_field = first_elem.wavB
            elif hasattr(first_elem, 'dtype') and hasattr(first_elem.dtype, 'names') and 'wavB' in first_elem.dtype.names:
                wavB_field = first_elem['wavB']
            
            if wavB_field is not None:
                if isinstance(wavB_field, np.ndarray):
                    if wavB_field.dtype == object:
                        # Cell array (object dtype) - extract each trial
                        if wavB_field.size > 0:
                            all_trials = []
                            for i in range(wavB_field.size):
                                trial_data = wavB_field.flat[i]
                                if isinstance(trial_data, np.ndarray):
                                    if trial_data.ndim == 1:
                                        trial_data = trial_data.reshape(-1, 1)
                                    elif trial_data.ndim > 2:
                                        trial_data = trial_data.reshape(trial_data.shape[0], -1)
                                        # Ensure single feature
                                        if trial_data.shape[1] > 1:
                                            trial_data = trial_data[:, 0:1]
                                    all_trials.append(trial_data.astype(np.float32))
                            
                            if all_trials:
                                wavB_trials = all_trials
                    else:
                        # Numeric array - reshape to extract trials
                        if wavB_field.ndim == 2:
                            if wavB_field.shape[0] == self.trial_length:
                                n_trials = wavB_field.shape[1]
                                wavB_trials = [wavB_field[:, i:i+1].astype(np.float32) for i in range(n_trials)]
                            elif wavB_field.shape[1] == self.trial_length:
                                n_trials = wavB_field.shape[0]
                                wavB_trials = [wavB_field[i:i+1, :].T.astype(np.float32) for i in range(n_trials)]
                        elif wavB_field.ndim == 3:
                            if wavB_field.shape[0] == self.trial_length:
                                n_trials = wavB_field.shape[2]
                                wavB_trials = [wavB_field[:, 0, i:i+1].astype(np.float32) for i in range(n_trials)]
                            elif wavB_field.shape[2] == self.trial_length:
                                n_trials = wavB_field.shape[0]
                                wavB_trials = [wavB_field[i, 0, :].reshape(-1, 1).astype(np.float32) for i in range(n_trials)]
            
            return wavA_trials, wavB_trials
            
        except Exception as e:
            print(f"    Error extracting audio trials: {e}")
            return None, None
    
    def _extract_attention_labels(self, data_struct, expinfo: Dict, n_trials: int) -> Optional[np.ndarray]:
        """
        Extract attention labels from expinfo structure.
        
        Based on MATLAB script: events_of_interest = expinfo.attend_mf
        where 1 = male, 2 = female
        
        We convert to binary: 0 = male, 1 = female
        
        Returns:
            Trial-level labels array of length n_trials (NOT expanded to sample level)
        
        Raises:
            RuntimeError: If labels cannot be extracted (no fallback)
        """
        try:
            # Method 1: Extract from expinfo.attend_mf (primary method)
            attend_mf = expinfo.get('attend_mf')
            
            if attend_mf is not None:
                # Helper to coerce to scalar
                def to_scalar(val):
                    """
                    Canonical scalar conversion: unwraps arrays, lists, converts to Python scalar.
                    Returns None if value cannot be converted to a scalar (multi-element arrays/lists).
                    """
                    if val is None:
                        return None
                    # Handle lists/tuples
                    if isinstance(val, (list, tuple)):
                        if len(val) == 0:
                            return None
                        if len(val) == 1:
                            val = val[0]
                        else:
                            # Multi-element list - cannot convert to scalar
                            return None
                    # Handle numpy arrays
                    if isinstance(val, np.ndarray):
                        val = np.array(val).squeeze()
                        if val.size == 0:
                            return None
                        if val.size == 1:
                            return val.item()
                        # Multi-element array - cannot convert to scalar
                        return None
                    return val
                
                # Convert to list/array and coerce each entry to scalar
                if isinstance(attend_mf, np.ndarray):
                    attend_mf_list = attend_mf.flatten().tolist()
                elif isinstance(attend_mf, (list, tuple)):
                    attend_mf_list = list(attend_mf)
                else:
                    attend_mf_list = [attend_mf]
                
                # Coerce each entry to scalar int
                vals = []
                for v in attend_mf_list:
                    v_scalar = to_scalar(v)
                    if v_scalar is None:
                        raise RuntimeError("attend_mf contains None values")
                    try:
                        v_int = int(v_scalar)
                        if v_int not in [1, 2]:
                            raise RuntimeError(f"Invalid attend_mf value: {v_int}. Expected 1 or 2.")
                        vals.append(v_int)
                    except (ValueError, TypeError) as e:
                        raise RuntimeError(f"Cannot convert attend_mf value {v} to int: {e}")
                
                # Validate all values are 1 or 2
                unique_vals = set(vals)
                if not unique_vals.issubset({1, 2}):
                    raise RuntimeError(f"Invalid attend_mf values: {unique_vals}. Expected only 1 and 2.")
                
                # Convert 1=male->0, 2=female->1
                trial_labels = np.array([0 if v == 1 else 1 for v in vals], dtype=np.int64)
                
                # Validate length matches n_trials
                if len(trial_labels) == n_trials:
                    print(f"    [OK] Extracted attention labels from expinfo.attend_mf ({len(trial_labels)} trials)")
                    return trial_labels
                else:
                    raise RuntimeError(
                        f"Label length mismatch: expinfo.attend_mf has {len(trial_labels)} values, "
                        f"but expected {n_trials} trials"
                    )
            
            # Method 2: Extract from event structure (trigger codes: 1=male, 2=female)
            if not isinstance(data_struct, np.ndarray) or data_struct.size == 0:
                raise RuntimeError("Cannot extract labels: data_struct is invalid")
            
            first_elem = data_struct.flat[0]
            
            # Try to get event structure
            event_field = None
            if hasattr(first_elem, 'event'):
                event_field = first_elem.event
            elif hasattr(first_elem, 'dtype') and hasattr(first_elem.dtype, 'names') and 'event' in first_elem.dtype.names:
                event_field = first_elem['event']
            
            if event_field is None:
                # Debug: show what attributes are available
                available_attrs = [k for k in dir(first_elem) if not k.startswith('_')]
                raise RuntimeError(
                    f"Failed to extract attention labels: event structure not found. "
                    f"Available attributes in data.flat[0]: {available_attrs}. "
                    f"Expected 'event' attribute."
                )
            
            try:
                # Access event.eeg - it might be a 1×1 struct containing the actual trial data
                # From diagnostic: event is mat_struct with 'eeg' attribute
                if hasattr(event_field, 'eeg'):
                    event_eeg_raw = event_field.eeg
                elif isinstance(event_field, np.ndarray) and event_field.dtype == object:
                    event_eeg_raw = event_field
                else:
                    raise RuntimeError(
                        f"Event structure found but 'eeg' attribute not accessible. "
                        f"Event type: {type(event_field)}, attributes: {[k for k in dir(event_field) if not k.startswith('_')][:10]}"
                    )
                
                # Handle case where event_eeg is a 1×1 struct containing the actual data
                # MATLAB often wraps arrays in 1×1 structs - we need to unwrap it
                event_eeg = None
                
                # Case 1: event_eeg_raw is already a numpy array
                if isinstance(event_eeg_raw, np.ndarray):
                    if event_eeg_raw.dtype == object:
                        # Check if it's a 1×1 struct that contains the actual array
                        if event_eeg_raw.size == 1:
                            # It's a 1×1 wrapper - extract the actual content
                            wrapped = event_eeg_raw.flat[0]
                            # The wrapped content might be the actual array of 60 trials
                            if isinstance(wrapped, np.ndarray) and wrapped.dtype == object:
                                if wrapped.size >= n_trials:
                                    event_eeg = wrapped
                                    print(f"    Unwrapped 1×1 struct: found {wrapped.size} trials inside")
                                else:
                                    print(f"    WARNING: Wrapped array has only {wrapped.size} elements, expected {n_trials}")
                            elif hasattr(wrapped, '__dict__'):
                                # It's a struct - look for fields that might contain the trial array
                                wrapped_attrs = [k for k in dir(wrapped) if not k.startswith('_')]
                                # Try common field names that might contain the 60 trials
                                for field_name in ['eeg', 'value', 'trial', 'data', 'events']:
                                    if hasattr(wrapped, field_name):
                                        field_val = getattr(wrapped, field_name)
                                        if isinstance(field_val, np.ndarray) and field_val.dtype == object:
                                            if field_val.size >= n_trials:
                                                event_eeg = field_val
                                                break
                                if event_eeg is None:
                                    # If no standard field, try to find any object array with >= n_trials elements
                                    for attr in wrapped_attrs:
                                        try:
                                            val = getattr(wrapped, attr)
                                            if isinstance(val, np.ndarray) and val.dtype == object and val.size >= n_trials:
                                                event_eeg = val
                                                print(f"    Found trial array in wrapped.{attr} (size: {val.size})")
                                                break
                                        except:
                                            pass
                        else:
                            # It's already the array of trials (not wrapped)
                            event_eeg = event_eeg_raw
                    else:
                        raise RuntimeError(
                            f"event.eeg is a numpy array but not object dtype. "
                            f"Type: {type(event_eeg_raw)}, dtype: {event_eeg_raw.dtype}, shape: {event_eeg_raw.shape}"
                        )
                elif hasattr(event_eeg_raw, '__dict__'):
                    # It's a mat_struct - look for fields containing the trial array
                    struct_attrs = [k for k in dir(event_eeg_raw) if not k.startswith('_')]
                    for field_name in ['eeg', 'value', 'trial', 'data', 'events']:
                        if hasattr(event_eeg_raw, field_name):
                            field_val = getattr(event_eeg_raw, field_name)
                            if isinstance(field_val, np.ndarray) and field_val.dtype == object:
                                if field_val.size >= n_trials:
                                    event_eeg = field_val
                                    print(f"    Found trial array in event_eeg_raw.{field_name} (size: {field_val.size})")
                                    break
                
                if event_eeg is None:
                    raise RuntimeError(
                        f"Could not extract trial array from event.eeg. "
                        f"event_eeg_raw type: {type(event_eeg_raw)}, "
                        f"shape: {getattr(event_eeg_raw, 'shape', 'N/A') if isinstance(event_eeg_raw, np.ndarray) else 'N/A'}, "
                        f"size: {getattr(event_eeg_raw, 'size', 'N/A') if isinstance(event_eeg_raw, np.ndarray) else 'N/A'}, "
                        f"attributes: {[k for k in dir(event_eeg_raw) if not k.startswith('_')][:10] if hasattr(event_eeg_raw, '__dict__') else 'N/A'}. "
                        f"If event_eeg_raw is 1×1, the actual trial array is likely inside a nested field."
                    )
                
                if event_eeg.dtype != object:
                    raise RuntimeError(
                        f"event.eeg (after unwrapping) is not an object array. "
                        f"Type: {type(event_eeg)}, dtype: {event_eeg.dtype}, shape: {event_eeg.shape}"
                    )
                
                # Use .size to count total elements (handles 2D arrays like (1, 60) or (60, 1))
                total_trials_in_event = event_eeg.size
                if total_trials_in_event < n_trials:
                    raise RuntimeError(
                        f"event.eeg has {total_trials_in_event} elements (shape: {event_eeg.shape}), "
                        f"but expected {n_trials} trials. "
                        f"Note: Unwrapped from 1×1 struct, but still don't have enough trials."
                    )
                
                # Extract trigger value from each trial
                # Use .flat to iterate regardless of array shape (1, 60) or (60, 1) or (60,)
                trigger_values = []
                for i in range(n_trials):
                    if i >= total_trials_in_event:
                        raise RuntimeError(
                            f"Trial index {i} exceeds available events ({total_trials_in_event})"
                        )
                    
                    trial_event = event_eeg.flat[i]
                    if not hasattr(trial_event, 'value'):
                        raise RuntimeError(
                            f"Trial {i} event does not have 'value' attribute. "
                            f"Attributes: {[k for k in dir(trial_event) if not k.startswith('_')][:10]}"
                        )
                    
                    val = trial_event.value
                    # Handle scalar or array
                    if isinstance(val, np.ndarray):
                        val = val.flatten()
                        if val.size == 0:
                            raise RuntimeError(f"Trial {i} event.value is empty array")
                        trigger_values.append(int(val[0]))
                    else:
                        trigger_values.append(int(val))
                
                if len(trigger_values) != n_trials:
                    raise RuntimeError(
                        f"Extracted {len(trigger_values)} trigger values, but expected {n_trials} trials"
                    )
                
                # Validate trigger codes are 1 or 2
                unique_triggers = set(trigger_values)
                if not unique_triggers.issubset({1, 2}):
                    raise RuntimeError(
                        f"Invalid trigger codes: {unique_triggers}. Expected only 1 and 2. "
                        f"Found in trials: {[i for i, v in enumerate(trigger_values) if v not in [1, 2]]}"
                    )
                
                # Convert trigger to binary label: 1=male->0, 2=female->1
                trial_labels = np.array([0 if v == 1 else 1 for v in trigger_values], dtype=np.int64)
                
                # Also update expinfo with trigger values
                expinfo['trigger'] = trigger_values
                
                print(f"    [OK] Extracted attention labels from event structure (trigger codes) ({len(trial_labels)} trials)")
                print(f"    Trigger code distribution: {dict(enumerate(np.bincount(trigger_values, minlength=3)[1:3]))}")
                return trial_labels
                
            except RuntimeError:
                # Re-raise RuntimeError as-is (these are our validation errors)
                raise
            except Exception as e:
                # Wrap other exceptions with more context
                raise RuntimeError(
                    f"Error extracting labels from event structure: {e}\n"
                    f"Event field type: {type(event_field)}, "
                    f"Event field attributes: {[k for k in dir(event_field) if not k.startswith('_')][:10] if hasattr(event_field, '__dict__') else 'N/A'}"
                ) from e
            
        except RuntimeError:
            # Re-raise RuntimeError as-is
            raise
        except Exception as e:
            raise RuntimeError(f"Error extracting attention labels: {e}")
    
    def _create_tfrecord_file(self, 
                              tfrecord_file: Path,
                              eeg_trials: List[np.ndarray],
                              wavA_trials: List[Optional[np.ndarray]],
                              wavB_trials: List[Optional[np.ndarray]],
                              attention_labels: np.ndarray,
                              expinfo: Dict,
                              valid_trials: List[int],
                              subject_id: str) -> int:
        """
        Create a TFRecord file from the preprocessed data (trial-by-trial).
        
        Args:
            eeg_trials: List of trial EEG arrays
            wavA_trials: List of trial wavA arrays (or None)
            wavB_trials: List of trial wavB arrays (or None)
            attention_labels: Trial-level labels array
            expinfo: Dictionary with expinfo fields
            valid_trials: List of valid trial indices to write
            subject_id: Subject identifier
        
        Returns:
            Number of trials written
        """
        n_trials_written = 0
        
        with tf.io.TFRecordWriter(str(tfrecord_file)) as writer:
            # Helper to convert any value to scalar
            def to_scalar(val):
                """
                Canonical scalar conversion: unwraps arrays, lists, converts to Python scalar.
                Returns None if value cannot be converted to a scalar (multi-element arrays/lists).
                """
                if val is None:
                    return None
                # Handle lists/tuples
                if isinstance(val, (list, tuple)):
                    if len(val) == 0:
                        return None
                    if len(val) == 1:
                        val = val[0]
                    else:
                        # Multi-element list - cannot convert to scalar
                        return None
                # Handle numpy arrays
                if isinstance(val, np.ndarray):
                    val = np.array(val).squeeze()
                    if val.size == 0:
                        return None
                    if val.size == 1:
                        return val.item()
                    # Multi-element array - cannot convert to scalar
                    return None
                return val
            
            def to_scalar_string(val):
                """Convert value to scalar string, handling arrays and bytes."""
                scalar = to_scalar(val)
                if scalar is None:
                    return None
                if isinstance(scalar, bytes):
                    return scalar.decode('utf-8', errors='ignore')
                return str(scalar)
            
            for valid_idx, trial_idx in enumerate(valid_trials):
                # Get trial data
                trial_eeg = eeg_trials[trial_idx]
                trial_wavA = wavA_trials[trial_idx] if wavA_trials else None
                trial_wavB = wavB_trials[trial_idx] if wavB_trials else None
                
                # Validate shapes before writing (safety check with detailed error)
                if trial_eeg.shape != (self.trial_length, self.n_channels):
                    raise RuntimeError(
                        f"Trial {trial_idx}: EEG shape mismatch! "
                        f"Got {trial_eeg.shape}, expected ({self.trial_length}, {self.n_channels}). "
                        f"EEG data appears to be corrupted or incorrectly extracted."
                    )
                
                # Verify it's actually 2D and not accidentally 1D
                if trial_eeg.ndim != 2:
                    raise RuntimeError(
                        f"Trial {trial_idx}: EEG is not 2D! Got {trial_eeg.ndim} dimensions, shape {trial_eeg.shape}"
                    )
                
                if trial_wavA is not None:
                    if trial_wavA.shape != (self.trial_length, 1):
                        raise RuntimeError(
                            f"Trial {trial_idx}: wavA shape {trial_wavA.shape} != expected ({self.trial_length}, 1)"
                        )
                if trial_wavB is not None:
                    if trial_wavB.shape != (self.trial_length, 1):
                        raise RuntimeError(
                            f"Trial {trial_idx}: wavB shape {trial_wavB.shape} != expected ({self.trial_length}, 1)"
                        )
                
                # Use trial label directly (no majority vote)
                trial_label = int(attention_labels[trial_idx])
                
                # Flatten EEG data and verify size
                # Use C-order flattening (row-major) to ensure correct order: (3200, 66) -> [sample0_ch0, sample0_ch1, ..., sample0_ch65, sample1_ch0, ...]
                eeg_flat = trial_eeg.flatten(order='C')
                expected_eeg_size = self.trial_length * self.n_channels
                
                if len(eeg_flat) != expected_eeg_size:
                    raise RuntimeError(
                        f"Trial {trial_idx}: Flattened EEG size {len(eeg_flat)} != expected {expected_eeg_size}. "
                        f"Original shape: {trial_eeg.shape}, ndim: {trial_eeg.ndim}. "
                        f"This should never happen if shape validation passed!"
                    )
                
                # Double-check: verify first few values make sense
                if len(eeg_flat) < expected_eeg_size:
                    raise RuntimeError(
                        f"Trial {trial_idx}: CRITICAL - Flattened array has only {len(eeg_flat)} values, "
                        f"expected {expected_eeg_size}. This will cause TFRecord corruption!"
                    )
                
                # Build features dictionary
                # Final safety check: ensure we're writing the full trial, not a summary
                eeg_list = eeg_flat.tolist()
                if len(eeg_list) != expected_eeg_size:
                    raise RuntimeError(
                        f"Trial {trial_idx}: CRITICAL ERROR - About to write {len(eeg_list)} values "
                        f"instead of {expected_eeg_size}! This will corrupt the TFRecord. "
                        f"trial_eeg.shape={trial_eeg.shape}, eeg_flat.shape={eeg_flat.shape}"
                    )
                
                features = {
                    'eeg': tf.train.Feature(
                        float_list=tf.train.FloatList(value=eeg_list)
                    ),
                    'attention_label': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[trial_label])
                    ),
                    'subject_id': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[subject_id.encode('utf-8')])
                    ),
                    'trial_idx': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[trial_idx])
                    ),
                    'n_channels': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[self.n_channels])
                    ),
                    'n_samples': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[self.trial_length])
                    ),
                    # Metadata fields
                    'sampling_rate': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[self.sampling_rate])
                    ),
                    'preprocessing_version': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[self.PREPROCESSING_VERSION.encode('utf-8')])
                    ),
                }
                
                # Add audio with separate flags for wavA and wavB
                wavA_missing = 1 if trial_wavA is None else 0
                wavB_missing = 1 if trial_wavB is None else 0
                
                features['wavA_missing'] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[wavA_missing])
                )
                features['wavB_missing'] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[wavB_missing])
                )
                
                if trial_wavA is not None:
                    features['wavA'] = tf.train.Feature(
                        float_list=tf.train.FloatList(value=trial_wavA.flatten())
                    )
                
                if trial_wavB is not None:
                    features['wavB'] = tf.train.Feature(
                        float_list=tf.train.FloatList(value=trial_wavB.flatten())
                    )
                
                # Helper to safely extract expinfo scalar
                def get_expinfo_scalar(field_name):
                    value = expinfo.get(field_name)
                    if value is None:
                        return None
                    if isinstance(value, (list, np.ndarray)):
                        if trial_idx < len(value):
                            return value[trial_idx]
                        return None
                    return value
                
                # Add expinfo fields with proper normalization using to_scalar
                attend_lr_val = get_expinfo_scalar('attend_lr')
                if attend_lr_val is not None:
                    attend_lr_scalar = to_scalar(attend_lr_val)
                    if attend_lr_scalar is not None:
                        try:
                            features['attend_lr'] = tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[int(attend_lr_scalar)])
                            )
                        except (ValueError, TypeError):
                            pass
                
                acoustic_val = get_expinfo_scalar('acoustic_condition')
                if acoustic_val is not None:
                    acoustic_scalar = to_scalar(acoustic_val)
                    if acoustic_scalar is not None:
                        try:
                            features['acoustic_condition'] = tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[int(acoustic_scalar)])
                            )
                        except (ValueError, TypeError):
                            pass
                
                n_speakers_val = get_expinfo_scalar('n_speakers')
                if n_speakers_val is not None:
                    n_speakers_scalar = to_scalar(n_speakers_val)
                    if n_speakers_scalar is not None:
                        try:
                            features['n_speakers'] = tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[int(n_speakers_scalar)])
                            )
                        except (ValueError, TypeError):
                            pass
                
                # Add trigger value (from event structure or expinfo)
                # Trigger is always written when available (required field)
                trigger_val = get_expinfo_scalar('trigger')
                if trigger_val is not None:
                    trigger_scalar = to_scalar(trigger_val)
                    if trigger_scalar is not None:
                        try:
                            features['trigger'] = tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[int(trigger_scalar)])
                            )
                        except (ValueError, TypeError):
                            pass
                
                # Add attend_mf raw value (1=male, 2=female) before binary conversion
                # Note: attend_mf may be None if labels came from trigger codes
                attend_mf_raw = get_expinfo_scalar('attend_mf')
                if attend_mf_raw is not None:
                    attend_mf_scalar = to_scalar(attend_mf_raw)
                    if attend_mf_scalar is not None:
                        try:
                            features['attend_mf_raw'] = tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[int(attend_mf_scalar)])
                            )
                        except (ValueError, TypeError):
                            pass
                
                # Add wavfile names if available (normalized to scalar strings)
                wavfile_male = get_expinfo_scalar('wavfile_male')
                if wavfile_male is not None:
                    wavfile_str = to_scalar_string(wavfile_male)
                    if wavfile_str is not None:
                        features['wavfile_male'] = tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[wavfile_str.encode('utf-8')])
                        )
                
                wavfile_female = get_expinfo_scalar('wavfile_female')
                if wavfile_female is not None:
                    wavfile_str = to_scalar_string(wavfile_female)
                    if wavfile_str is not None:
                        features['wavfile_female'] = tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[wavfile_str.encode('utf-8')])
                        )
                
                # Create and write example
                example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(example.SerializeToString())
                n_trials_written += 1
        
        return n_trials_written
    
    def _write_manifest(self, manifest_file: Path, subject_id: str, n_trials: int,
                       valid_trials: List[int], attention_labels: np.ndarray,
                       expinfo: Dict, filtered_reasons: Dict) -> None:
        """
        Write manifest JSON with filtering information and condition counts.
        
        Args:
            manifest_file: Path to write manifest JSON
            subject_id: Subject identifier
            n_trials: Total number of trials extracted
            valid_trials: List of valid trial indices
            attention_labels: Trial-level labels
            expinfo: Dictionary with expinfo fields
            filtered_reasons: Dictionary with counts of why trials were filtered
        """
        def to_scalar(val):
            """
            Canonical scalar conversion: unwraps arrays, lists, converts to Python scalar.
            Returns None if value cannot be converted to a scalar (multi-element arrays/lists).
            """
            if val is None:
                return None
            # Handle lists/tuples
            if isinstance(val, (list, tuple)):
                if len(val) == 0:
                    return None
                if len(val) == 1:
                    val = val[0]
                else:
                    # Multi-element list - cannot convert to scalar
                    return None
            # Handle numpy arrays
            if isinstance(val, np.ndarray):
                val = np.array(val).squeeze()
                if val.size == 0:
                    return None
                if val.size == 1:
                    return val.item()
                # Multi-element array - cannot convert to scalar
                return None
            return val
        
        def get_expinfo_scalar(field_name, trial_idx):
            """Extract value from expinfo field for given trial index."""
            value = expinfo.get(field_name)
            if value is None:
                return None
            if isinstance(value, (list, np.ndarray)):
                if trial_idx < len(value):
                    return value[trial_idx]
                return None
            return value
        
        # Count conditions for valid trials
        condition_counts = {
            'acoustic_condition': {},
            'attend_lr': {},
            'n_speakers': {},
            'attend_mf': {}
        }
        
        for trial_idx in valid_trials:
            acoustic = get_expinfo_scalar('acoustic_condition', trial_idx)
            if acoustic is not None:
                acoustic_scalar = to_scalar(acoustic)
                if acoustic_scalar is not None:
                    try:
                        acoustic = int(acoustic_scalar)
                        condition_counts['acoustic_condition'][acoustic] = condition_counts['acoustic_condition'].get(acoustic, 0) + 1
                    except (ValueError, TypeError):
                        pass
            
            attend_lr = get_expinfo_scalar('attend_lr', trial_idx)
            if attend_lr is not None:
                attend_lr_scalar = to_scalar(attend_lr)
                if attend_lr_scalar is not None:
                    try:
                        attend_lr = int(attend_lr_scalar)
                        condition_counts['attend_lr'][attend_lr] = condition_counts['attend_lr'].get(attend_lr, 0) + 1
                    except (ValueError, TypeError):
                        pass
            
            n_speakers = get_expinfo_scalar('n_speakers', trial_idx)
            if n_speakers is not None:
                n_speakers_scalar = to_scalar(n_speakers)
                if n_speakers_scalar is not None:
                    try:
                        n_speakers = int(n_speakers_scalar)
                        condition_counts['n_speakers'][n_speakers] = condition_counts['n_speakers'].get(n_speakers, 0) + 1
                    except (ValueError, TypeError):
                        pass
            
            attend_mf = get_expinfo_scalar('attend_mf', trial_idx)
            if attend_mf is not None:
                attend_mf_scalar = to_scalar(attend_mf)
                if attend_mf_scalar is not None:
                    try:
                        attend_mf = int(attend_mf_scalar)
                        condition_counts['attend_mf'][attend_mf] = condition_counts['attend_mf'].get(attend_mf, 0) + 1
                    except (ValueError, TypeError):
                        pass
        
        # Label distribution
        valid_labels = [attention_labels[i] for i in valid_trials]
        label_dist = dict(enumerate(np.bincount(valid_labels, minlength=2)))
        
        # Helper to convert NumPy types to JSON-serializable Python types
        def to_json_serializable(obj):
            """Recursively convert NumPy types to native Python types for JSON serialization."""
            if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_, np.bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: to_json_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [to_json_serializable(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                # For other types, try to convert if possible
                try:
                    # Try common conversions
                    if hasattr(obj, 'item'):  # NumPy scalar
                        return obj.item()
                    return obj
                except:
                    return obj
        
        manifest = {
            'subject_id': subject_id,
            'preprocessing_version': self.PREPROCESSING_VERSION,
            'n_trials_total': n_trials,
            'n_trials_valid': len(valid_trials),
            'valid_trial_indices': valid_trials,
            'filtered_reasons': filtered_reasons,
            'label_distribution': label_dist,
            'condition_counts': condition_counts,
            'filter_n_speakers': self.filter_n_speakers,
            'sampling_rate': self.sampling_rate,
            'n_channels': self.n_channels,
            'trial_length': self.trial_length,
            'timestamp': datetime.now().isoformat()
        }
        
        # Convert all NumPy types to native Python types before JSON serialization
        manifest = to_json_serializable(manifest)
        
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def _extract_subject_id(self, mat_file: Path) -> str:
        """Extract subject ID from filename (e.g., 'S1' from 'S1_data_preproc.mat')."""
        stem = mat_file.stem
        if stem.startswith('S') and '_data_preproc' in stem:
            return stem.replace('_data_preproc', '')
        return stem
    
    def _print_summary(self):
        """Print processing summary."""
        print("\n" + "="*70)
        print("PREPROCESSING SUMMARY")
        print("="*70)
        print(f"Subjects processed: {self.stats['subjects_processed']}")
        print(f"Subjects failed: {self.stats['subjects_failed']}")
        print(f"Total trials extracted: {self.stats['total_trials']}")
        print(f"Total trials after filtering: {self.stats['total_trials_filtered']}")
        print(f"Total samples: {self.stats['total_samples']:,}")
        print(f"Preprocessing version: {self.PREPROCESSING_VERSION}")
        
        if self.stats['start_time'] and self.stats['end_time']:
            duration = self.stats['end_time'] - self.stats['start_time']
            print(f"Processing time: {duration}")
        
        print(f"\nTFRecord files saved to: {self.tfrecord_dir}")
        print("="*70)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Preprocess Fulsang dataset from MATLAB files to TFRecords'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='Data/Fulsang',
        help='Directory containing Fulsang data (default: Data/Fulsang)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='fulsang_preprocessed',
        help='Output directory for TFRecord files (default: fulsang_preprocessed)'
    )
    parser.add_argument(
        '--subjects',
        type=int,
        nargs='+',
        default=None,
        help='Specific subject IDs to process (1-18). If not specified, processes all.'
    )
    parser.add_argument(
        '--filter_n_speakers',
        type=int,
        default=2,
        help='Only include trials with this many speakers (default: 2 for AAD task)'
    )
    parser.add_argument(
        '--require_audio',
        action='store_true',
        default=False,
        help='If set, fail if audio extraction fails. Otherwise, skip audio fields when missing.'
    )
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = FulsangPreprocessor(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        filter_n_speakers=args.filter_n_speakers,
        require_audio=args.require_audio
    )
    
    # Process subjects
    success = preprocessor.preprocess_all_subjects(subject_ids=args.subjects)
    
    if success:
        print("\n[OK] Preprocessing completed successfully!")
        return 0
    else:
        print("\n[FAIL] Preprocessing failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

