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
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
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
                 das_data_dir: str = "das_16subjects_preprocessed",
                 das_preprocessing_type: str = "16SUBJECTS",
                 das_original_dir: str = "Data/Das/4004271",  # Original Das .mat files (with stimuli info for envelope mapping)
                 das_audio_dir: str = "Data/Das/4004271/stimuli/stimuli",  # Das audio files
                 fulsang_raw_dir: str = "/home/py9363/telluride_decoding/Data/Fulsang/EEG",
                 fulsang_audio_dir: str = "/home/py9363/telluride_decoding/Data/Fulsang/AUDIO",
                 fulsang_mwf_output_dir: str = "MWF_cleaned_Fuglsang",
                 combined_dataset_dir: str = "combined_dataset",  # Centralized output directory
                 window_size: int = 512,  # samples at 128 Hz = 4 seconds
                 overlap: float = 0.5,
                 target_channels: int = 64,
                 target_sampling_rate: int = 128):
        """
        Initialize combined dataset.
        
        Args:
            das_data_dir: Directory containing Das preprocessed data (MWF, DASPREPROCESS, or 16SUBJECTS)
            das_preprocessing_type: Type of preprocessing used ("MWF", "DASPREPROCESS", or "16SUBJECTS")
            das_original_dir: Directory containing original Das .mat files (for envelope extraction)
            das_audio_dir: Directory containing Das audio files (for envelope extraction)
            fulsang_raw_dir: Directory containing Fulsang raw EEG data
            fulsang_audio_dir: Directory containing Fulsang audio data
            fulsang_mwf_output_dir: Output directory for Fulsang MWF processing (legacy, for backward compatibility)
            combined_dataset_dir: Centralized directory for all processed files (default: "combined_dataset")
            window_size: Window size in samples
            overlap: Window overlap fraction
            target_channels: Target number of channels (64 for Das compatibility)
            target_sampling_rate: Target sampling rate (128 Hz)
        """
        self.das_data_dir = Path(das_data_dir)
        self.das_preprocessing_type = das_preprocessing_type.upper()
        self.das_original_dir = Path(das_original_dir)  # Original Das .mat files for envelope extraction
        self.das_audio_dir = Path(das_audio_dir) if das_audio_dir else None  # Das audio files
        self.fulsang_raw_dir = Path(fulsang_raw_dir)
        self.fulsang_audio_dir = Path(fulsang_audio_dir) if fulsang_audio_dir else None
        
        # Centralized combined dataset directory
        self.combined_dataset_dir = Path(combined_dataset_dir)
        self.combined_dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories for processed files
        self.das_mwf_dir = self.combined_dataset_dir / "das_mwf"
        self.fulsang_mwf_dir = self.combined_dataset_dir / "fulsang_mwf"
        self.das_mwf_dir.mkdir(parents=True, exist_ok=True)
        self.fulsang_mwf_dir.mkdir(parents=True, exist_ok=True)
        self.window_size = window_size
        self.overlap = overlap
        self.target_channels = target_channels
        self.target_sampling_rate = target_sampling_rate
        
        # Parameters
        self.sampling_rate = target_sampling_rate
        self.n_channels = target_channels
        
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
        
        # Load Das data (MWF, DASPREPROCESS, or 16SUBJECTS)
        print(f"\nLoading Das data ({self.das_preprocessing_type})...")
        if self.das_preprocessing_type == "MWF":
            das_eeg, das_labels, das_metadata, das_trial_lengths, das_left_envs, das_right_envs = self._load_das_mwf_data()
        elif self.das_preprocessing_type == "16SUBJECTS":
            das_eeg, das_labels, das_metadata, das_trial_lengths, das_left_envs, das_right_envs = self._load_das_16subjects_data()
        else:  # DASPREPROCESS
            das_eeg, das_labels, das_metadata, das_trial_lengths, das_left_envs, das_right_envs = self._load_das_preprocessed_data()
        
        # Load Fulsang raw data and apply MWF
        print("\nLoading Fulsang raw data and applying MWF filtering...")
        fulsang_eeg, fulsang_labels, fulsang_metadata, fulsang_trial_lengths, fulsang_left_envs, fulsang_right_envs = self._load_fulsang_and_apply_mwf()
        
        # Normalize channel count BEFORE combining
        max_channels = max(das_eeg.shape[1], fulsang_eeg.shape[1])
        if das_eeg.shape[1] != fulsang_eeg.shape[1]:
            print(f"\nWarning: Channel mismatch - Das: {das_eeg.shape[1]}, Fulsang: {fulsang_eeg.shape[1]}")
            print(f"Aligning to {self.target_channels} channels (keeping all Das channels)")
            
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
        
        # Combine datasets
        print("\nCombining datasets...")
        self.eeg_data = np.vstack([das_eeg, fulsang_eeg])
        self.labels = np.hstack([das_labels, fulsang_labels])
        self.metadata = das_metadata + fulsang_metadata
        
        # Combine envelope streams
        # Ensure all envelopes are 2D (samples x 1) for vstack
        def ensure_2d(env_list):
            """Ensure all envelopes are 2D (samples x 1) for proper vstack."""
            result = []
            for env in env_list:
                if env is None:
                    continue
                env = np.asarray(env)
                if len(env.shape) == 1:
                    env = env.reshape(-1, 1)
                elif len(env.shape) > 2:
                    env = env.reshape(-1, 1)
                result.append(env)
            return result
        
        das_left_envs_2d = ensure_2d(das_left_envs) if das_left_envs else []
        das_right_envs_2d = ensure_2d(das_right_envs) if das_right_envs else []
        fulsang_left_envs_2d = ensure_2d(fulsang_left_envs) if fulsang_left_envs else []
        fulsang_right_envs_2d = ensure_2d(fulsang_right_envs) if fulsang_right_envs else []
        
        left_env_stream = np.vstack(das_left_envs_2d + fulsang_left_envs_2d) if das_left_envs_2d or fulsang_left_envs_2d else None
        right_env_stream = np.vstack(das_right_envs_2d + fulsang_right_envs_2d) if das_right_envs_2d or fulsang_right_envs_2d else None
        
        if left_env_stream is not None and right_env_stream is not None:
            self.left_envelope_stream = left_env_stream.astype(np.float32)
            self.right_envelope_stream = right_env_stream.astype(np.float32)
            self._total_frames = self.left_envelope_stream.shape[0]
            
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
            self.left_envelope_stream = np.zeros((len(self.eeg_data), 1), dtype=np.float32)
            self.right_envelope_stream = np.zeros((len(self.eeg_data), 1), dtype=np.float32)
            self._total_frames = len(self.eeg_data)
            print(f"\n⚠️  WARNING: No envelope streams created - using zero envelopes!")
            print(f"   This will cause CCA to fail. Check envelope extraction in data files.")
        
        self.n_channels = self.target_channels
        
        # Track trial boundaries for label mapping
        self.trial_boundaries = []
        self.trial_labels = []
        current_idx = 0
        
        # Track Das trial boundaries
        for label, trial_length in zip(das_labels, das_trial_lengths):
            self.trial_boundaries.append((current_idx, current_idx + trial_length))
            self.trial_labels.append(label)
            current_idx += trial_length
        
        # Track Fulsang trial boundaries
        for label, trial_length in zip(fulsang_labels, fulsang_trial_lengths):
            self.trial_boundaries.append((current_idx, current_idx + trial_length))
            self.trial_labels.append(label)
            current_idx += trial_length
        
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
    
    def _fallback_envelopes(self, length: int, label: int) -> Tuple[np.ndarray, np.ndarray]:
        ramp = np.linspace(0.0, 1.0, num=length, dtype=np.float32).reshape(-1, 1)
        zeros = np.zeros_like(ramp)
        if label == 0:
            return ramp, zeros
        return zeros, ramp
    
    def _extract_das_envelopes_from_original(self, original_trials, trial_idx: int, target_length: int, subject_id: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract envelopes from original Das file or audio files.
        
        The original Das files in Data/Das/4004271 should contain stimuli information
        that maps trials to audio files in the stimuli folder.
        """
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
                        left_env = self._extract_envelope_from_audio(audio_file, target_length, self.target_sampling_rate)
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
                        right_env = self._extract_envelope_from_audio(audio_file, target_length, self.target_sampling_rate)
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
                                               subject_num: Optional[str] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract envelopes from Fulsang audio files.
        
        Fulsang audio files are named by story and trial:
        - aske_story1_trial_1.wav, aske_story2_trial_1.wav, etc. (male speaker)
        - marianne_story1_trial_1.wav, marianne_story2_trial_1.wav, etc. (female speaker)
        
        Args:
            subject_id: Subject ID (e.g., 'S1' or 'sub01')
            trial_idx: Trial index
            target_length: Target envelope length
            trial_data: Optional trial data object that may contain audio_file_male/female fields
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
                
                # Separate by speaker
                aske_files = sorted([f for f in all_audio_files if 'aske' in f.name.lower()])
                marianne_files = sorted([f for f in all_audio_files if 'marianne' in f.name.lower()])
                
                # Map trial_idx to audio files (sequential mapping)
                # Each trial has both speakers, so we need to find matching story/trial pairs
                # Group files by story and trial number
                aske_by_trial = {}
                marianne_by_trial = {}
                
                for f in aske_files:
                    # Extract story and trial number from filename: aske_story1_trial_1.wav
                    parts = f.stem.split('_')
                    if len(parts) >= 3:
                        try:
                            story_num = int(parts[1].replace('story', ''))
                            trial_num = int(parts[2].replace('trial', ''))
                            key = (story_num, trial_num)
                            if key not in aske_by_trial:
                                aske_by_trial[key] = []
                            aske_by_trial[key].append(f)
                        except:
                            continue
                
                for f in marianne_files:
                    parts = f.stem.split('_')
                    if len(parts) >= 3:
                        try:
                            story_num = int(parts[1].replace('story', ''))
                            trial_num = int(parts[2].replace('trial', ''))
                            key = (story_num, trial_num)
                            if key not in marianne_by_trial:
                                marianne_by_trial[key] = []
                            marianne_by_trial[key].append(f)
                        except:
                            continue
                
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
            
            # Extract envelopes from audio files
            left_env = None
            right_env = None
            
            # In Fulsang, we need to determine which speaker corresponds to "left" and "right"
            # Based on the attention labels: 192 = Left attention, 191 = Right attention
            # We need to check if attend_mf indicates which speaker is attended
            # For now, we'll use a consistent mapping: aske (male) -> left, marianne (female) -> right
            # This can be verified by checking if the attention labels match the speaker assignments
            
            # Try to get attention label from trial_data to verify mapping
            attention_label = None
            if trial_data is not None:
                if hasattr(trial_data, 'attention_label'):
                    attention_label = trial_data.attention_label
                elif isinstance(trial_data, dict) and 'attention_label' in trial_data:
                    attention_label = trial_data['attention_label']
            
            if audio_file_male:
                if isinstance(audio_file_male, (str, Path)):
                    audio_path = self.fulsang_audio_dir / audio_file_male if not Path(audio_file_male).is_absolute() else Path(audio_file_male)
                else:
                    audio_path = audio_file_male
                
                if audio_path.exists():
                    left_env = self._extract_envelope_from_audio(audio_path, target_length, self.target_sampling_rate)
                    # Diagnostic: Log successful extraction
                    if not hasattr(self, '_fulsang_left_extracted'):
                        self._fulsang_left_extracted = 0
                    if self._fulsang_left_extracted < 3:
                        print(f"  ✓ Fulsang {subject_id} trial {trial_idx}: Extracted left envelope from {audio_path.name}")
                        self._fulsang_left_extracted += 1
                else:
                    if not hasattr(self, '_fulsang_left_missing'):
                        self._fulsang_left_missing = 0
                    if self._fulsang_left_missing < 3:
                        print(f"  ⚠️  Fulsang {subject_id} trial {trial_idx}: Left audio file not found: {audio_path}")
                        self._fulsang_left_missing += 1
            
            if audio_file_female:
                if isinstance(audio_file_female, (str, Path)):
                    audio_path = self.fulsang_audio_dir / audio_file_female if not Path(audio_file_female).is_absolute() else Path(audio_file_female)
                else:
                    audio_path = audio_file_female
                
                if audio_path.exists():
                    right_env = self._extract_envelope_from_audio(audio_path, target_length, self.target_sampling_rate)
                    # Diagnostic: Log successful extraction
                    if not hasattr(self, '_fulsang_right_extracted'):
                        self._fulsang_right_extracted = 0
                    if self._fulsang_right_extracted < 3:
                        print(f"  ✓ Fulsang {subject_id} trial {trial_idx}: Extracted right envelope from {audio_path.name}")
                        self._fulsang_right_extracted += 1
                else:
                    if not hasattr(self, '_fulsang_right_missing'):
                        self._fulsang_right_missing = 0
                    if self._fulsang_right_missing < 3:
                        print(f"  ⚠️  Fulsang {subject_id} trial {trial_idx}: Right audio file not found: {audio_path}")
                        self._fulsang_right_missing += 1
            
            # If we still don't have files, try direct pattern matching as fallback
            if left_env is None or right_env is None:
                # Try to find files by trial number directly
                aske_files = sorted(list(self.fulsang_audio_dir.glob('aske_*.wav')))
                marianne_files = sorted(list(self.fulsang_audio_dir.glob('marianne_*.wav')))
                
                # Use trial_idx to index into sorted files (simple sequential mapping)
                if trial_idx < len(aske_files) and left_env is None:
                    left_env = self._extract_envelope_from_audio(aske_files[trial_idx], target_length, self.target_sampling_rate)
                if trial_idx < len(marianne_files) and right_env is None:
                    right_env = self._extract_envelope_from_audio(marianne_files[trial_idx], target_length, self.target_sampling_rate)
            
            return left_env, right_env
            
        except Exception as e:
            return None, None
    
    def _extract_envelope_from_audio(self, audio_file: Path, target_length: int, target_fs: int = 128) -> Optional[np.ndarray]:
        """Extract envelope from audio file using simple method."""
        try:
            from scipy.io import wavfile
            from scipy import signal
            
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
            
            # Extract envelope: absolute value + smoothing
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
        if left_raw is None or right_raw is None:
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
        
        # Load original Das files from Data/Das/4004271 for envelope extraction
        # These files should have stimuli information needed to map to audio files
        original_das_files = {}
        if self.das_original_dir.exists():
            print(f"Loading original Das files from {self.das_original_dir} for envelope extraction...")
            for orig_file in self.das_original_dir.glob("S*.mat"):
                subject_id = orig_file.stem
                if subject_id not in original_das_files:
                    try:
                        orig_data = sio.loadmat(str(orig_file), squeeze_me=True, struct_as_record=False)
                        original_das_files[subject_id] = orig_data
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
                    except Exception as e:
                        print(f"Warning: Could not load original Das file {orig_file}: {e}")
        
        for mwf_file in tqdm(mwf_files, desc="Loading Das MWF data"):
            try:
                data = sio.loadmat(str(mwf_file), squeeze_me=True, struct_as_record=False)
                subject_id = mwf_file.stem.replace('_MWF', '')
                
                # Get original trial data for envelope extraction
                original_trials = None
                if subject_id in original_das_files:
                    orig_data = original_das_files[subject_id]
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
                        
                        # Convert to label (L=0, R=1)
                        label = 0 if str(attended_ear).upper() == 'L' else 1
                        
                        # Extract envelopes from original file or audio files
                        left_env, right_env = self._extract_das_envelopes_from_original(
                            original_trials, trial_idx, eeg_data.shape[0], subject_id
                        )
                        
                        # Verify mapping: Check if attended_ear matches the stimuli assignment (only once per subject)
                        if not hasattr(self, '_das_mapping_verified'):
                            self._das_mapping_verified = set()
                        if subject_id not in self._das_mapping_verified and original_trials and trial_idx < len(original_trials):
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
                if subject_id in original_das_files:
                    orig_data = original_das_files[subject_id]
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
                        
                        # Convert to label (L=0, R=1)
                        label = 0 if str(attended_ear).upper() == 'L' else 1
                        
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
    
    def _load_das_16subjects_data(self) -> Tuple[np.ndarray, np.ndarray, List[Dict], List[int]]:
        """Load Das 16-subjects TFRecord data."""
        import tensorflow as tf
        
        tfrecord_dir = self.das_data_dir / "tfrecords"
        if not tfrecord_dir.exists():
            raise ValueError(f"Das 16-subjects TFRecord directory does not exist: {tfrecord_dir}\n"
                           f"Please run preprocessing first: python3 das_preprocessing_16subjects.py")
        
        # Find all TFRecord files (check subdirectories too)
        tfrecord_files = list(tfrecord_dir.glob("*.tfrecords"))
        if not tfrecord_files:
            # Check subdirectories
            tfrecord_files = list(tfrecord_dir.glob("*/*.tfrecords"))
        if not tfrecord_files:
            raise ValueError(f"No TFRecord files found in {tfrecord_dir}\n"
                           f"Please run preprocessing first: python3 das_preprocessing_16subjects.py")
        
        all_eeg = []
        all_labels = []
        all_metadata = []
        trial_lengths = []
        all_left_env = []
        all_right_env = []
        current_trial_samples = []
        current_trial_label = None
        current_trial_id = None
        current_subject_id = None
        
        for tfrecord_file in tqdm(tfrecord_files, desc="Loading Das 16-subjects data"):
            try:
                dataset = tf.data.TFRecordDataset(str(tfrecord_file))
                
                try:
                    for record in dataset:
                        try:
                            example = tf.train.Example.FromString(record.numpy())
                            features = example.features.feature
                            
                            # Check required features
                            if 'eeg' not in features or 'attended_ear' not in features:
                                continue
                            
                            # Extract EEG data
                            eeg_values = features['eeg'].float_list.value
                            if not eeg_values or len(eeg_values) != 64:
                                continue
                            
                            eeg_sample = np.array(eeg_values, dtype=np.float32).reshape(1, 64)
                            
                            # Extract attended ear
                            attended_ear = features['attended_ear'].bytes_list.value[0].decode('utf-8')
                            label = 0 if attended_ear.upper() == 'L' else 1
                            
                            # Extract subject and trial info
                            subject_id = features['subject_id'].bytes_list.value[0].decode('utf-8') if 'subject_id' in features else "unknown"
                            trial_id = features['trial_id'].int64_list.value[0] if 'trial_id' in features else 0
                            
                            # Check if we're starting a new trial
                            if current_trial_id != trial_id or current_subject_id != subject_id:
                                # Save previous trial if exists
                                if current_trial_samples:
                                    trial_eeg = np.vstack(current_trial_samples)
                                    all_eeg.append(trial_eeg)
                                    all_labels.append(current_trial_label)
                                    trial_lengths.append(len(current_trial_samples))
                                    all_metadata.append({
                                        'subject_id': current_subject_id,
                                        'trial_idx': current_trial_id,
                                        'dataset': 'Das',
                                        'attended_ear': 'L' if current_trial_label == 0 else 'R',
                                        'preprocessing': '16SUBJECTS'
                                    })
                                    left_env, right_env = self._fallback_envelopes(trial_eeg.shape[0], current_trial_label)
                                    all_left_env.append(left_env)
                                    all_right_env.append(right_env)
                                
                                # Start new trial
                                current_trial_samples = [eeg_sample]
                                current_trial_label = label
                                current_trial_id = trial_id
                                current_subject_id = subject_id
                            else:
                                # Continue current trial
                                current_trial_samples.append(eeg_sample)
                        
                        except Exception as e:
                            continue
                except (tf.errors.OutOfRangeError, StopIteration) as e:
                    # Expected when dataset ends - this is normal
                    pass
                except Exception as e:
                    # Check if it's the expected OUT_OF_RANGE error (TensorFlow logs this as INFO)
                    error_str = str(e)
                    if "OUT_OF_RANGE" in error_str or "End of sequence" in error_str:
                        # This is expected when dataset ends - ignore it
                        pass
                    else:
                        # Other exceptions should be logged
                        print(f"Warning: Error iterating dataset {tfrecord_file.name}: {e}")
                
                # Save last trial
                if current_trial_samples:
                    trial_eeg = np.vstack(current_trial_samples)
                    all_eeg.append(trial_eeg)
                    all_labels.append(current_trial_label)
                    trial_lengths.append(len(current_trial_samples))
                    all_metadata.append({
                        'subject_id': current_subject_id,
                        'trial_idx': current_trial_id,
                        'dataset': 'Das',
                        'attended_ear': 'L' if current_trial_label == 0 else 'R',
                        'preprocessing': '16SUBJECTS'
                    })
                    # Use fallback envelopes for TFRecord format (envelopes not stored per-sample)
                    left_env, right_env = self._fallback_envelopes(trial_eeg.shape[0], current_trial_label)
                    all_left_env.append(left_env)
                    all_right_env.append(right_env)
                    current_trial_samples = []
            
            except Exception as e:
                print(f"Error loading {tfrecord_file}: {e}")
                continue
        
        if not all_eeg:
            raise ValueError("No valid Das 16-subjects data loaded")
        
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
                                # Handle data_preproc event codes (1 and 2) - if present
                                elif value_int == 1:
                                    # Event code 1 might be Left or Right - need to check expinfo
                                    # For now, map 1 -> 0 (Left) as default, but this might need adjustment
                                    return 0
                                elif value_int == 2:
                                    # Event code 2 might be Right - map to 1
                                    return 1
                                else:
                                    # All other event codes are non-attention trials - skip them
                                    # This includes: 131, 132, 133, 135, 137, 141, 150, 160, 224, 240, 248, 252, 254, etc.
                                    return None
                        except Exception as e:
                            return None
                        
                        return None
                    
                    # First pass: extract all labels from trials
                    label_lookup = {}  # Maps trial_idx -> (label, field_used)
                    skipped_trials = []
                    skipped_values = {}  # Track what values were in skipped trials
                    
                    # Debug: show structure of first trial
                    if len(trials) > 0:
                        first_trial = trials[0]
                        print(f"  Inspecting {subject_id}: First trial type: {type(first_trial)}")
                        if hasattr(first_trial, '__dict__'):
                            print(f"    Attributes: {list(first_trial.__dict__.keys())}")
                        elif isinstance(first_trial, dict):
                            print(f"    Dict keys: {list(first_trial.keys())}")
                        elif isinstance(first_trial, np.ndarray) and first_trial.dtype.names:
                            print(f"    Structured array fields: {first_trial.dtype.names}")
                    
                    for trial_idx, trial in enumerate(trials):
                        label_value, found, field_used = extract_label_from_trial(trial, trial_idx, debug=(trial_idx < 3))
                        if found and label_value is not None:
                            label_lookup[trial_idx] = (label_value, field_used)
                        else:
                            skipped_trials.append(trial_idx)
                            # Try to get the raw value to see what it is
                            if hasattr(trial, 'attention_label'):
                                raw_val = trial.attention_label
                                if isinstance(raw_val, np.ndarray) and raw_val.size > 0:
                                    raw_val = raw_val.item() if raw_val.size == 1 else raw_val.flatten()[0]
                                skipped_values[trial_idx] = raw_val
                            elif isinstance(trial, dict) and 'attention_label' in trial:
                                skipped_values[trial_idx] = trial['attention_label']
                    
                    if skipped_trials:
                        # Show unique values from skipped trials
                        unique_skipped = {}
                        for idx in skipped_trials[:20]:  # Check first 20 skipped
                            if idx in skipped_values:
                                val = skipped_values[idx]
                                if val not in unique_skipped:
                                    unique_skipped[val] = 0
                                unique_skipped[val] += 1
                        
                        if unique_skipped:
                            print(f"  {subject_id}: Skipped {len(skipped_trials)} trials. Sample skipped values: {dict(list(unique_skipped.items())[:5])}")
                        else:
                            print(f"  {subject_id}: Skipped {len(skipped_trials)} trials without extractable labels")
                    
                    # Second pass: process trials and use label lookup with smart fallback
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
                        # Pass subject_id and trial_idx for per-subject mapping
                        left_env, right_env = self._extract_fulsang_envelopes_from_audio(
                            subject_id, trial_idx, eeg_data.shape[0], 
                            trial_data=trial, subject_num=subject_num
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
                        
                        if left_env is None or right_env is None:
                            # Fallback to trying from trial object
                            left_env, right_env, _ = self._prepare_trial_envelopes(trial, eeg_data.shape[0], label, dataset_name='Fulsang-MWF')
                        else:
                            # Count as real envelopes
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
                            'label_source': 'extracted'  # All labels are now extracted, none inferred
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
        
        # Check label source (extracted vs fallback)
        extracted_count = sum(1 for meta in all_metadata if meta.get('label_source') == 'extracted')
        fallback_count = sum(1 for meta in all_metadata if meta.get('label_source') == 'fallback')
        if fallback_count > 0:
            print(f"  ⚠ Warning: {fallback_count} trials used fallback labels (out of {len(all_metadata)} total)")
        else:
            print(f"  ✓ All labels extracted successfully from data")
        
        return eeg_data, labels, all_metadata, trial_lengths, all_left_env, all_right_env
    
    def get_window_indices(self) -> List[Tuple[int, int, int]]:
        """Create sliding windows with labels."""
        window_indices = []
        step_size = int(self.window_size * (1 - self.overlap))
        
        for start_idx in range(0, len(self.eeg_data) - self.window_size + 1, step_size):
            end_idx = start_idx + self.window_size
            
            # Find which trial this window belongs to (use middle of window)
            mid_idx = start_idx + self.window_size // 2
            window_label = 0  # Default
            
            for trial_idx, (trial_start, trial_end) in enumerate(self.trial_boundaries):
                if trial_start <= mid_idx < trial_end:
                    window_label = self.trial_labels[trial_idx]
                    break
            
            window_indices.append((start_idx, end_idx, window_label))
        
        return window_indices

    def get_envelope_window(self, start_idx: int, end_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return left/right stimulus envelopes aligned with the EEG window."""
        if self.left_envelope_stream is None or self.right_envelope_stream is None:
            length = end_idx - start_idx
            return self._fallback_envelopes(length, 0)
        
        # Validate indices
        if start_idx < 0 or end_idx > len(self.left_envelope_stream):
            length = end_idx - start_idx
            return self._fallback_envelopes(length, 0)
        
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

