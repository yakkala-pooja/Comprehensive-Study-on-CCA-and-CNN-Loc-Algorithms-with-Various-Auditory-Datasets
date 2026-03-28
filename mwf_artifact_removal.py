#!/usr/bin/env python3
"""
Multi-channel Wiener Filtering (MWF) Artifact Removal for EEG Datasets

This script applies MWF artifact removal to:
1. Das dataset - 16 subjects, 20 trials per subject
2. Fuglsang dataset - 18 subjects, COCOHA format

Features:
- MWF artifact removal using EOG reference channels
- Downsampling for Fuglsang (512 Hz -> 128 Hz)
- Visualization of before/after filtering
- Unified preprocessing function
"""

import os
import sys
import numpy as np
import scipy.io as sio
from scipy import signal
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional, Union
import logging
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiChannelWienerFilter:
    """
    Multi-channel Wiener Filter (MWF) for EEG artifact removal.
    
    MWF estimates the artifact subspace from reference channels (EOG) or
    artifact segments and removes it from the EEG data.
    """
    
    def __init__(self, rank: Optional[int] = None, reg: float = 1e-6):
        """
        Initialize MWF.
        
        Args:
            rank: Number of components to remove (None = auto-detect)
            reg: Regularization parameter for numerical stability
        """
        self.rank = rank
        self.reg = reg
        self.filter_matrix = None
        
    def fit(self, eeg_data: np.ndarray, artifact_data: np.ndarray):
        """
        Fit MWF filter using artifact reference data.
        
        Args:
            eeg_data: EEG data, shape (n_samples, n_channels)
            artifact_data: Artifact reference data (EOG), shape (n_samples, n_ref_channels)
        """
        if eeg_data.shape[0] != artifact_data.shape[0]:
            raise ValueError("EEG and artifact data must have same number of samples")
        
        # Center the data
        eeg_mean = np.mean(eeg_data, axis=0, keepdims=True)
        artifact_mean = np.mean(artifact_data, axis=0, keepdims=True)
        
        eeg_centered = eeg_data - eeg_mean
        artifact_centered = artifact_data - artifact_mean
        
        # Compute covariance matrices
        n_samples = eeg_data.shape[0]
        R_xx = (eeg_centered.T @ eeg_centered) / (n_samples - 1)
        R_xy = (eeg_centered.T @ artifact_centered) / (n_samples - 1)
        R_yy = (artifact_centered.T @ artifact_centered) / (n_samples - 1)
        
        # Add regularization for numerical stability
        R_yy += self.reg * np.eye(R_yy.shape[0])
        
        # Compute Wiener filter: W = R_xy * inv(R_yy)
        try:
            R_yy_inv = np.linalg.inv(R_yy)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            R_yy_inv = np.linalg.pinv(R_yy)
        
        self.filter_matrix = R_xy @ R_yy_inv
        
        # If rank is specified, use SVD to reduce dimensionality
        if self.rank is not None and self.rank < self.filter_matrix.shape[1]:
            U, s, Vt = np.linalg.svd(self.filter_matrix, full_matrices=False)
            U = U[:, :self.rank]
            s = s[:self.rank]
            Vt = Vt[:self.rank, :]
            self.filter_matrix = U @ np.diag(s) @ Vt
        
        logger.debug(f"MWF filter fitted: shape {self.filter_matrix.shape}")
        
    def transform(self, eeg_data: np.ndarray, artifact_data: np.ndarray) -> np.ndarray:
        """
        Apply MWF filter to remove artifacts.
        
        Args:
            eeg_data: EEG data, shape (n_samples, n_channels)
            artifact_data: Artifact reference data, shape (n_samples, n_ref_channels)
            
        Returns:
            Cleaned EEG data, shape (n_samples, n_channels)
        """
        if self.filter_matrix is None:
            raise ValueError("Filter must be fitted before transformation")
        
        if eeg_data.shape[0] != artifact_data.shape[0]:
            raise ValueError("EEG and artifact data must have same number of samples")
        
        # Center the data
        eeg_mean = np.mean(eeg_data, axis=0, keepdims=True)
        artifact_mean = np.mean(artifact_data, axis=0, keepdims=True)
        
        eeg_centered = eeg_data - eeg_mean
        artifact_centered = artifact_data - artifact_mean
        
        # Estimate artifacts: artifacts = W * artifact_reference
        estimated_artifacts = artifact_centered @ self.filter_matrix.T
        
        # Remove artifacts: cleaned = eeg - estimated_artifacts
        cleaned_eeg = eeg_centered - estimated_artifacts
        
        # Add back the mean
        cleaned_eeg = cleaned_eeg + eeg_mean
        
        return cleaned_eeg
    
    def fit_transform(self, eeg_data: np.ndarray, artifact_data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(eeg_data, artifact_data)
        return self.transform(eeg_data, artifact_data)


class DasDatasetMWF:
    """
    Das dataset loader and MWF processor.
    
    Das dataset:
    - 16 subjects
    - 20 trials per subject stored as 'Sx.mat' files
    - EEG already high-pass filtered (0.5 Hz) and downsampled to 128 Hz
    - Need to reapply MWF using raw EEG signals if available
    """
    
    def __init__(self, data_dir: str = "Data/Das/4004271", output_dir: str = "MWF_cleaned_DAS",
                 audio_dir: str = "Data/Das/4004271/stimuli/stimuli"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Audio directory for envelope extraction
        self.audio_dir = Path(audio_dir) if audio_dir else None
        
        self.target_sampling_rate = 128  # Hz (preserve original)
        
        logger.info(f"Das Dataset MWF Processor initialized")
        logger.info(f"  Data directory: {self.data_dir}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Audio directory: {self.audio_dir}")
    
    def load_subject_data(self, subject_file: str) -> List[Dict]:
        """Load all trials for a subject."""
        mat_file = self.data_dir / subject_file
        
        if not mat_file.exists():
            raise FileNotFoundError(f"Subject file not found: {mat_file}")
        
        logger.info(f"Loading {mat_file}")
        
        try:
            mat_data = sio.loadmat(str(mat_file), squeeze_me=True, struct_as_record=False)
            trials = mat_data['trials']
            
            if not isinstance(trials, np.ndarray):
                trials = [trials]
            else:
                trials = trials.flatten()
            
            trial_list = []
            for trial_idx, trial in enumerate(trials):
                try:
                    # Extract EEG data
                    eeg_data = trial.RawData.EegData  # Shape: (n_samples, n_channels)
                    sample_rate = trial.FileHeader.SampleRate
                    
                    # Extract attended ear
                    attended_ear = trial.attended_ear
                    if isinstance(attended_ear, np.ndarray):
                        attended_ear = attended_ear.item() if attended_ear.size == 1 else str(attended_ear[0])
                    
                    # Extract stimuli information
                    stimuli = []
                    if hasattr(trial, 'stimuli'):
                        stimuli = trial.stimuli
                    
                    trial_dict = {
                        'eeg_data': eeg_data,
                        'sample_rate': sample_rate,
                        'attended_ear': attended_ear,
                        'stimuli': stimuli,
                        'trial_idx': trial_idx,
                        'subject_id': Path(subject_file).stem
                    }
                    
                    trial_list.append(trial_dict)
                    
                except Exception as e:
                    logger.warning(f"Error loading trial {trial_idx} from {subject_file}: {e}")
                    continue
            
            logger.info(f"  Loaded {len(trial_list)} trials from {subject_file}")
            return trial_list
            
        except Exception as e:
            logger.error(f"Error loading {mat_file}: {e}")
            return []
    
    def extract_eog_channels(self, eeg_data: np.ndarray, channel_labels: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract EOG channels from EEG data.
        
        For Das dataset, we need to identify EOG channels.
        If not available, we'll use artifact detection on the data itself.
        """
        n_channels = eeg_data.shape[1]
        
        if channel_labels is not None:
            # Try to find EOG channels by name
            eog_indices = []
            for i, label in enumerate(channel_labels):
                if label and ('EOG' in str(label).upper() or 'EXG' in str(label).upper()):
                    eog_indices.append(i)
            
            if len(eog_indices) >= 2:
                eog_channels = eeg_data[:, eog_indices]
                eeg_indices = [i for i in range(n_channels) if i not in eog_indices]
                eeg_channels = eeg_data[:, eeg_indices]
                return eeg_channels, eog_channels
        
        # Fallback: use high-variance channels as artifact reference
        # This is a simplified approach - in practice, EOG channels should be identified
        channel_variance = np.var(eeg_data, axis=0)
        n_ref_channels = min(4, n_channels // 16)
        ref_indices = np.argsort(channel_variance)[-n_ref_channels:]
        
        eog_channels = eeg_data[:, ref_indices]
        eeg_channels = eeg_data  # Use all channels for EEG
        
        logger.warning("EOG channels not explicitly identified, using high-variance channels as reference")
        
        return eeg_channels, eog_channels
    
    def apply_basic_artifact_removal(self, eeg_data: np.ndarray, fsample: float) -> np.ndarray:
        """
        Apply basic artifact removal without EOG reference.
        
        This method:
        1. Applies high-pass filtering (0.5 Hz) to remove slow drifts
        2. Applies notch filter (50/60 Hz) to remove line noise
        3. Removes bad channels based on variance
        4. Applies robust normalization
        
        This ensures compatibility between Das and Fuglsang datasets.
        """
        cleaned_eeg = eeg_data.copy()
        
        # 1. High-pass filter (0.5 Hz) to remove slow drifts
        nyquist = fsample / 2
        highpass_freq = 0.5 / nyquist
        if highpass_freq < 1.0:
            b, a = signal.butter(4, highpass_freq, btype='high')
            for ch in range(cleaned_eeg.shape[1]):
                cleaned_eeg[:, ch] = signal.filtfilt(b, a, cleaned_eeg[:, ch])
        
        # 2. Notch filter for line noise (50 Hz or 60 Hz depending on region)
        line_noise_freq = 50.0  # Can be 60.0 for US data
        notch_freq = line_noise_freq / nyquist
        if notch_freq < 1.0:
            b, a = signal.iirnotch(notch_freq, Q=30)
            for ch in range(cleaned_eeg.shape[1]):
                cleaned_eeg[:, ch] = signal.filtfilt(b, a, cleaned_eeg[:, ch])
        
        # 3. Remove bad channels (channels with extremely high variance)
        channel_variance = np.var(cleaned_eeg, axis=0)
        median_var = np.median(channel_variance)
        mad_var = np.median(np.abs(channel_variance - median_var))
        threshold = median_var + 5 * mad_var  # 5 MAD above median
        
        good_channels = channel_variance < threshold
        if np.sum(good_channels) < cleaned_eeg.shape[1]:
            logger.debug(f"Removed {np.sum(~good_channels)} bad channels based on variance")
            cleaned_eeg = cleaned_eeg[:, good_channels]
        
        # 4. Robust normalization (z-score using median and MAD)
        for ch in range(cleaned_eeg.shape[1]):
            ch_data = cleaned_eeg[:, ch]
            median_val = np.median(ch_data)
            mad_val = np.median(np.abs(ch_data - median_val))
            if mad_val > 0:
                cleaned_eeg[:, ch] = (ch_data - median_val) / mad_val
        
        return cleaned_eeg
    
    def process_subject(self, subject_file: str) -> Dict:
        """Process all trials for a subject with MWF, or add audio mapping to existing MWF files."""
        subject_id = Path(subject_file).stem
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing subject: {subject_id}")
        logger.info(f"{'='*60}")
        
        # Check if MWF file already exists
        existing_mwf_file = self.output_dir / f"{subject_id}_MWF.mat"
        if existing_mwf_file.exists():
            logger.info(f"Found existing MWF file: {existing_mwf_file}")
            logger.info("Adding audio file mapping to existing MWF file...")
            return self._add_audio_mapping_to_existing_mwf(subject_file, existing_mwf_file)
        
        # Load trials for new processing
        trials = self.load_subject_data(subject_file)
        
        if len(trials) == 0:
            logger.warning(f"No trials found for {subject_id}")
            return {}
        
        processed_trials = []
        
        for trial in tqdm(trials, desc=f"Processing {subject_id}"):
            try:
                eeg_data = trial['eeg_data']
                sample_rate = trial['sample_rate']
                
                # Ensure data is 2D (samples x channels)
                if len(eeg_data.shape) != 2:
                    if len(eeg_data.shape) == 1:
                        eeg_data = eeg_data.reshape(-1, 1)
                    else:
                        eeg_data = eeg_data.reshape(eeg_data.shape[0], -1)
                
                # Extract only EEG channels (exclude EOG for dataset compatibility)
                # Das dataset typically doesn't have separate EOG channels, so use all channels
                # Apply basic artifact removal for consistency with Fuglsang processing
                cleaned_eeg = self.apply_basic_artifact_removal(eeg_data, sample_rate)
                
                # Extract stimuli information and map to audio files
                left_audio_file = None
                right_audio_file = None
                stimuli = trial.get('stimuli', [])
                
                # Check if stimuli is not empty (handle NumPy arrays properly)
                has_stimuli = False
                if isinstance(stimuli, np.ndarray):
                    has_stimuli = stimuli.size > 0
                elif isinstance(stimuli, (list, tuple)):
                    has_stimuli = len(stimuli) > 0
                else:
                    has_stimuli = stimuli is not None and stimuli != []
                
                if has_stimuli and self.audio_dir and self.audio_dir.exists():
                    # Convert stimuli to list
                    if isinstance(stimuli, np.ndarray):
                        stimuli = stimuli.flatten().tolist()
                    elif not isinstance(stimuli, (list, tuple)):
                        stimuli = [stimuli]
                    
                    if len(stimuli) >= 2:
                        left_stim = str(stimuli[0]) if len(stimuli) > 0 else None
                        right_stim = str(stimuli[1]) if len(stimuli) > 1 else None
                        
                        # Find audio files
                        if left_stim:
                            for ext in ['.wav', '.WAV', '.mp3', '.MP3']:
                                audio_file = self.audio_dir / f"{left_stim}{ext}"
                                if audio_file.exists():
                                    left_audio_file = str(audio_file)
                                    break
                            # Try pattern matching if direct match failed
                            if not left_audio_file:
                                for f in self.audio_dir.glob(f"*{left_stim}*"):
                                    if f.suffix.lower() in ['.wav', '.mp3']:
                                        left_audio_file = str(f)
                                        break
                        
                        if right_stim:
                            for ext in ['.wav', '.WAV', '.mp3', '.MP3']:
                                audio_file = self.audio_dir / f"{right_stim}{ext}"
                                if audio_file.exists():
                                    right_audio_file = str(audio_file)
                                    break
                            if not right_audio_file:
                                for f in self.audio_dir.glob(f"*{right_stim}*"):
                                    if f.suffix.lower() in ['.wav', '.mp3']:
                                        right_audio_file = str(f)
                                        break
                        
                        # Log mapping for first few trials
                        if trial['trial_idx'] < 3:
                            logger.info(f"  Trial {trial['trial_idx']} audio mapping:")
                            logger.info(f"    Left stimulus: {left_stim} -> {left_audio_file if left_audio_file else 'NOT FOUND'}")
                            logger.info(f"    Right stimulus: {right_stim} -> {right_audio_file if right_audio_file else 'NOT FOUND'}")
                
                # Store processed trial
                processed_trial = {
                    'eeg_data': cleaned_eeg,
                    'sample_rate': sample_rate,
                    'attended_ear': trial['attended_ear'],
                    'stimuli': trial['stimuli'],
                    'trial_idx': trial['trial_idx'],
                    'subject_id': subject_id,
                    'original_shape': eeg_data.shape,
                    'cleaned_shape': cleaned_eeg.shape,
                    'left_audio_file': left_audio_file,  # Save audio file paths
                    'right_audio_file': right_audio_file
                }
                
                processed_trials.append(processed_trial)
                
            except Exception as e:
                logger.error(f"Error processing trial {trial.get('trial_idx', 'unknown')}: {e}")
                continue
        
        logger.info(f"Processed {len(processed_trials)}/{len(trials)} trials for {subject_id}")
        
        return {
            'subject_id': subject_id,
            'trials': processed_trials,
            'n_trials': len(processed_trials)
        }
    
    def _add_audio_mapping_to_existing_mwf(self, subject_file: str, mwf_file: Path) -> Dict:
        """Add audio file mapping to existing MWF file without reprocessing."""
        subject_id = Path(subject_file).stem
        logger.info(f"Adding audio mapping to existing MWF file for {subject_id}")
        
        try:
            # Load existing MWF file
            mwf_data = sio.loadmat(str(mwf_file), squeeze_me=True, struct_as_record=False)
            
            # Load original data to get stimuli information
            trials = self.load_subject_data(subject_file)
            
            if len(trials) == 0:
                logger.warning(f"No trials found for {subject_id}")
                return {}
            
            # Get trials from MWF file
            mwf_trials = mwf_data.get('trials', [])
            if not isinstance(mwf_trials, np.ndarray):
                mwf_trials = [mwf_trials] if mwf_trials else []
            else:
                mwf_trials = mwf_trials.flatten()
            
            # Map audio files for each trial
            updated_trials = []
            for trial_idx in range(len(mwf_trials)):
                mwf_trial = mwf_trials[trial_idx]
                
                # Extract existing trial data
                if isinstance(mwf_trial, dict):
                    eeg_data = mwf_trial.get('eeg_data')
                    sample_rate = mwf_trial.get('sample_rate')
                    attended_ear = mwf_trial.get('attended_ear')
                    trial_idx_existing = mwf_trial.get('trial_idx', trial_idx)
                else:
                    # Handle structured array or object
                    eeg_data = getattr(mwf_trial, 'eeg_data', None) if hasattr(mwf_trial, 'eeg_data') else None
                    sample_rate = getattr(mwf_trial, 'sample_rate', None) if hasattr(mwf_trial, 'sample_rate') else None
                    attended_ear = getattr(mwf_trial, 'attended_ear', None) if hasattr(mwf_trial, 'attended_ear') else None
                    trial_idx_existing = getattr(mwf_trial, 'trial_idx', trial_idx) if hasattr(mwf_trial, 'trial_idx') else trial_idx
                
                # Get stimuli from original trial
                stimuli = []
                if trial_idx < len(trials):
                    original_trial = trials[trial_idx]
                    stimuli = original_trial.get('stimuli', [])
                
                # Extract audio file paths
                left_audio_file = None
                right_audio_file = None
                
                # Check if stimuli is not empty (handle NumPy arrays properly)
                has_stimuli = False
                if isinstance(stimuli, np.ndarray):
                    has_stimuli = stimuli.size > 0
                elif isinstance(stimuli, (list, tuple)):
                    has_stimuli = len(stimuli) > 0
                else:
                    has_stimuli = stimuli is not None and stimuli != []
                
                if has_stimuli and self.audio_dir and self.audio_dir.exists():
                    # Convert stimuli to list
                    if isinstance(stimuli, np.ndarray):
                        stimuli = stimuli.flatten().tolist()
                    elif not isinstance(stimuli, (list, tuple)):
                        stimuli = [stimuli]
                    
                    if len(stimuli) >= 2:
                        left_stim = str(stimuli[0]) if len(stimuli) > 0 else None
                        right_stim = str(stimuli[1]) if len(stimuli) > 1 else None
                        
                        # Find audio files
                        if left_stim:
                            for ext in ['.wav', '.WAV', '.mp3', '.MP3']:
                                audio_file = self.audio_dir / f"{left_stim}{ext}"
                                if audio_file.exists():
                                    left_audio_file = str(audio_file)
                                    break
                            if not left_audio_file:
                                for f in self.audio_dir.glob(f"*{left_stim}*"):
                                    if f.suffix.lower() in ['.wav', '.mp3']:
                                        left_audio_file = str(f)
                                        break
                        
                        if right_stim:
                            for ext in ['.wav', '.WAV', '.mp3', '.MP3']:
                                audio_file = self.audio_dir / f"{right_stim}{ext}"
                                if audio_file.exists():
                                    right_audio_file = str(audio_file)
                                    break
                            if not right_audio_file:
                                for f in self.audio_dir.glob(f"*{right_stim}*"):
                                    if f.suffix.lower() in ['.wav', '.mp3']:
                                        right_audio_file = str(f)
                                        break
                        
                        if trial_idx < 3:
                            logger.info(f"  Trial {trial_idx}: Left={left_audio_file}, Right={right_audio_file}")
                
                # Create updated trial dict
                trial_dict = {
                    'eeg_data': eeg_data,
                    'sample_rate': sample_rate,
                    'attended_ear': attended_ear,
                    'trial_idx': trial_idx_existing,
                    'left_audio_file': left_audio_file,
                    'right_audio_file': right_audio_file
                }
                updated_trials.append(trial_dict)
            
            # Save updated MWF file with audio mapping
            save_dict = {
                'subject_id': subject_id,
                'n_trials': len(updated_trials),
                'trials': updated_trials
            }
            
            sio.savemat(str(mwf_file), save_dict)
            logger.info(f"Updated MWF file with audio mapping: {mwf_file}")
            
            return {
                'subject_id': subject_id,
                'trials': updated_trials,
                'n_trials': len(updated_trials)
            }
            
        except Exception as e:
            logger.error(f"Error adding audio mapping to existing MWF file: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def save_cleaned_data(self, processed_data: Dict):
        """Save cleaned EEG data as 'Sx_MWF.mat'."""
        subject_id = processed_data['subject_id']
        output_file = self.output_dir / f"{subject_id}_MWF.mat"
        
        # Prepare data for MATLAB format
        save_dict = {
            'subject_id': subject_id,
            'n_trials': processed_data['n_trials'],
            'trials': []
        }
        
        for trial in processed_data['trials']:
            trial_dict = {
                'eeg_data': trial['eeg_data'],
                'sample_rate': trial['sample_rate'],
                'attended_ear': trial['attended_ear'],
                'trial_idx': trial['trial_idx']
            }
            # Save audio file paths if available
            if 'left_audio_file' in trial and trial['left_audio_file']:
                trial_dict['left_audio_file'] = trial['left_audio_file']
            if 'right_audio_file' in trial and trial['right_audio_file']:
                trial_dict['right_audio_file'] = trial['right_audio_file']
            save_dict['trials'].append(trial_dict)
        
        sio.savemat(str(output_file), save_dict)
        logger.info(f"Saved cleaned data to {output_file}")
    
    def process_all_subjects(self):
        """Process all subjects in the dataset, or add audio mapping to existing MWF files."""
        subject_files = list(self.data_dir.glob("S*.mat"))
        subject_files.sort()
        
        if not subject_files:
            raise ValueError(f"No subject files found in {self.data_dir}")
        
        logger.info(f"Found {len(subject_files)} subject files")
        
        all_results = {}
        subjects_with_existing_mwf = 0
        subjects_processed_new = 0
        
        for subject_file in subject_files:
            try:
                # Check if MWF file already exists in output directory
                subject_id = Path(subject_file).stem
                existing_mwf_file = self.output_dir / f"{subject_id}_MWF.mat"
                
                if existing_mwf_file.exists():
                    logger.info(f"Found existing MWF file for {subject_id}, adding audio mapping...")
                    processed_data = self._add_audio_mapping_to_existing_mwf(subject_file.name, existing_mwf_file)
                    if processed_data:
                        # Save updated file
                        self.save_cleaned_data(processed_data)
                        all_results[processed_data['subject_id']] = processed_data['n_trials']
                        subjects_with_existing_mwf += 1
                else:
                    logger.info(f"Processing new MWF for {subject_id}...")
                    processed_data = self.process_subject(subject_file.name)
                    if processed_data:
                        self.save_cleaned_data(processed_data)
                        all_results[processed_data['subject_id']] = processed_data['n_trials']
                        subjects_processed_new += 1
            except Exception as e:
                logger.error(f"Error processing {subject_file.name}: {e}")
                continue
        
        logger.info(f"\nProcessing summary:")
        logger.info(f"  Subjects with existing MWF (audio mapping added): {subjects_with_existing_mwf}")
        logger.info(f"  Subjects processed new: {subjects_processed_new}")
        
        # Save summary
        summary_file = self.output_dir / "processing_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Das Dataset MWF Processing Summary\n")
            f.write("=" * 60 + "\n\n")
            for subject_id, n_trials in all_results.items():
                f.write(f"{subject_id}: {n_trials} trials processed\n")
            f.write(f"\nTotal subjects: {len(all_results)}\n")
            f.write(f"Total trials: {sum(all_results.values())}\n")
        
        logger.info(f"\nProcessing complete! Summary saved to {summary_file}")
        return all_results


def _load_expinfo_sidecar_fuglsang(eeg_base_path: Path, subject_id: int) -> Optional[SimpleNamespace]:
    """Load attend_mf and attend_lr from S{n}_expinfo.mat (from save_expinfo_only.m). Returns SimpleNamespace or None.
    Looks in eeg_base_path, Exp_Info/, parent/Exp_Info, and standalone Exp_Info (repo root / cwd)."""
    repo_root = Path(__file__).resolve().parent
    cwd = Path.cwd()
    bases = [
        eeg_base_path, eeg_base_path / "Exp_Info", eeg_base_path / "exp_info",
        repo_root / "Exp_Info", repo_root / "exp_info", cwd / "Exp_Info", cwd / "exp_info",
    ]
    if hasattr(eeg_base_path, 'parent') and eeg_base_path.parent:
        bases.extend([eeg_base_path.parent, eeg_base_path.parent / "Exp_Info", eeg_base_path.parent / "exp_info"])
    for base in bases:
        if base is None or not base.exists():
            continue
        path = base / f"S{subject_id}_expinfo.mat"
        if not path.exists():
            continue
        try:
            mat = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
            attend_mf = mat.get('attend_mf')
            attend_lr = mat.get('attend_lr')
            if attend_mf is None or attend_lr is None:
                exp = mat.get('expinfo') or mat.get('exp_info')
                if exp is not None:
                    if attend_mf is None and (hasattr(exp, 'attend_mf') or (isinstance(exp, dict) and 'attend_mf' in exp)):
                        attend_mf = exp.attend_mf if hasattr(exp, 'attend_mf') else exp['attend_mf']
                    if attend_lr is None and (hasattr(exp, 'attend_lr') or (isinstance(exp, dict) and 'attend_lr' in exp)):
                        attend_lr = exp.attend_lr if hasattr(exp, 'attend_lr') else exp['attend_lr']
            if attend_mf is not None or attend_lr is not None:
                attend_mf = np.atleast_1d(np.asarray(attend_mf).flatten()) if attend_mf is not None else None
                attend_lr = np.atleast_1d(np.asarray(attend_lr).flatten()) if attend_lr is not None else None
                return SimpleNamespace(attend_mf=attend_mf, attend_lr=attend_lr)
        except Exception:
            continue
    return None


def _log_expinfo_structure(expinfo, subject_id: int, log):
    """Log expinfo field names and presence of attend_mf/attend_lr for debugging."""
    try:
        fields = []
        if hasattr(expinfo, '_fieldnames'):
            fields = list(expinfo._fieldnames)
        elif hasattr(expinfo, 'dtype') and getattr(expinfo.dtype, 'names', None):
            fields = list(expinfo.dtype.names)
        elif isinstance(expinfo, dict):
            fields = list(expinfo.keys())
        else:
            fields = [k for k in dir(expinfo) if not k.startswith('_')]
        log.info(f"  expinfo (subject {subject_id}) field names: {fields}")
        for name in ['attend_mf', 'attend_lr', 'attendMf', 'attendLr']:
            v = getattr(expinfo, name, None) if not isinstance(expinfo, dict) else expinfo.get(name)
            if v is not None:
                arr = np.asarray(v).flatten()
                log.info(f"    expinfo.{name}: length {len(arr)}, sample {arr[:5].tolist()}")
            else:
                log.info(f"    expinfo.{name}: not present")
    except Exception as e:
        log.warning(f"  Could not inspect expinfo: {e}")


class FuglsangDatasetMWF:
    """
    Fuglsang dataset loader and MWF processor.
    
    Fuglsang dataset:
    - 18 subjects
    - 64+8 EOG/mastoid channels, sampled at 512 Hz
    - Organized in COCOHA Matlab Toolbox format
    - Raw EEG stored in data.eeg, triggers in data.event.eeg.value
    """
    
    def __init__(self, 
                 eeg_base_path: str = "/home/py9363/telluride_decoding/Data/Fulsang/EEG",
                 audio_base_path: str = "Data/Fulsang/AUDIO",
                 output_dir: str = "MWF_cleaned_Fuglsang",
                 truncate_to_sidecar_labels: bool = True,
                 force_segment_from_events: bool = True):
        self.eeg_base_path = Path(eeg_base_path)
        self.audio_base_path = Path(audio_base_path) if audio_base_path else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.truncate_to_sidecar_labels = bool(truncate_to_sidecar_labels)
        self.force_segment_from_events = bool(force_segment_from_events)
        
        self.original_sampling_rate = 512  # Hz
        self.target_sampling_rate = 128  # Hz (downsample to match Das)
        
        logger.info(f"Fuglsang Dataset MWF Processor initialized")
        logger.info(f"  EEG directory: {self.eeg_base_path}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Downsampling: {self.original_sampling_rate} Hz -> {self.target_sampling_rate} Hz")
        logger.info(f"  Trial/label alignment: {'truncate to sidecar labels' if self.truncate_to_sidecar_labels else 'no truncation'}")
        logger.info(f"  Trialization: {'force segment from data.event.eeg' if self.force_segment_from_events else 'use pre-segmented if present'}")
    
    def load_raw_eeg_data(self, subject_id: int) -> Dict:
        """Load raw EEG data for a subject. Tries S{id}.mat then S{id}_data_preproc.mat."""
        eeg_file = self.eeg_base_path / f"S{subject_id}.mat"
        if not eeg_file.exists():
            eeg_file = self.eeg_base_path / f"S{subject_id}_data_preproc.mat"
        if not eeg_file.exists():
            raise FileNotFoundError(f"EEG file not found: {self.eeg_base_path / f'S{subject_id}.mat'} or _data_preproc.mat")
        
        logger.info(f"Loading raw EEG data for subject {subject_id} from {eeg_file.name}")
        
        try:
            mat_data = sio.loadmat(str(eeg_file), squeeze_me=True, struct_as_record=False)
            
            # Extract data structure
            data = None
            expinfo = {}
            
            # Try different possible data field names
            for field_name in ['data', f'S{subject_id}', 'eeg_data']:
                if field_name in mat_data:
                    data = mat_data[field_name]
                    break
            
            if data is None:
                # Try to find the main data structure
                for key, value in mat_data.items():
                    if not key.startswith('__') and hasattr(value, '__dict__'):
                        data = value
                        break
            
            if data is None:
                raise ValueError(f"Could not find data structure in {eeg_file}")
            
            # Extract experimental info (must be in .mat as 'expinfo' or 'exp_info'; key 'None' is not expinfo)
            expinfo = mat_data.get('expinfo', {})
            if not expinfo:
                for info_name in ['exp_info', 'experiment_info', 'info']:
                    if info_name in mat_data:
                        expinfo = mat_data[info_name]
                        break
                else:
                    expinfo = {}
            
            # If raw file has no usable attend_mf/attend_lr, load from sidecar S*_expinfo.mat (from save_expinfo_only.m)
            def _has_attend(exp):
                if exp is None or (isinstance(exp, dict) and not exp):
                    return False
                if hasattr(exp, 'attend_mf') or hasattr(exp, 'attend_lr'):
                    return True
                if isinstance(exp, dict) and ('attend_mf' in exp or 'attend_lr' in exp):
                    return True
                return False
            if not _has_attend(expinfo):
                sidecar_expinfo = _load_expinfo_sidecar_fuglsang(self.eeg_base_path, subject_id)
                if sidecar_expinfo is not None:
                    expinfo = sidecar_expinfo
                    logger.info(f"  Using expinfo from sidecar S{subject_id}_expinfo.mat")
            
            # Log expinfo structure once so we can see if attend_mf / attend_lr exist
            if not getattr(self, '_expinfo_structure_logged', False) and expinfo is not None:
                _log_expinfo_structure(expinfo, subject_id, logger)
                self._expinfo_structure_logged = True
            
            # Parse EEG data structure
            parsed_data = self._parse_eeg_structure(data, subject_id)
            
            return {
                'data': parsed_data,
                'expinfo': expinfo,
                'subject_id': subject_id,
                'file_path': str(eeg_file)
            }
            
        except Exception as e:
            logger.error(f"Error loading EEG data for subject {subject_id}: {e}")
            raise
    
    def _parse_eeg_structure(self, data: any, subject_id: int) -> Dict:
        """Parse the EEG data structure from MATLAB file."""
        parsed = {}
        
        try:
            # Extract EEG data
            if hasattr(data, 'eeg'):
                eeg_data = data.eeg
                if isinstance(eeg_data, (list, np.ndarray)):
                    if isinstance(eeg_data, list):
                        # Already a list of trials
                        parsed['eeg'] = [np.array(e) for e in eeg_data]
                        logger.info(f"Extracted EEG data: {len(parsed['eeg'])} trials (from list)")
                    else:
                        # Single numpy array - could be continuous or single trial
                        eeg_array = np.array(eeg_data)
                        # Check if it's 2D (samples x channels) - likely continuous
                        if len(eeg_array.shape) == 2:
                            # Store as single array - will be segmented later based on events
                            parsed['eeg'] = eeg_array
                            logger.info(f"Extracted continuous EEG data: shape {eeg_array.shape}")
                        else:
                            parsed['eeg'] = eeg_array
                            logger.info(f"Extracted EEG data: shape {eeg_array.shape}")
                else:
                    parsed['eeg'] = np.array([eeg_data])
                    logger.info(f"Extracted EEG data: single element")
            else:
                raise ValueError("No EEG data found in structure")
            
            # Extract sampling frequency
            if hasattr(data, 'fsample'):
                if hasattr(data.fsample, 'eeg'):
                    parsed['fsample'] = float(data.fsample.eeg)
                else:
                    parsed['fsample'] = float(data.fsample)
                logger.info(f"Sampling frequency: {parsed['fsample']} Hz")
            else:
                logger.warning("No sampling frequency found, using default 512 Hz")
                parsed['fsample'] = 512.0
            
            # Extract channel labels - handle nested structures (e.g., label.eeg)
            labels = None
            label_source = None
            
            # First, log available attributes for debugging
            if hasattr(data, '__dict__'):
                available_attrs = [a for a in data.__dict__.keys() if not a.startswith('_')]
                logger.debug(f"Available data attributes: {available_attrs}")
            
            # Try multiple possible field names for channel labels
            label_fields = ['label', 'channels', 'chanlocs', 'channel', 'chan']
            for field_name in label_fields:
                if hasattr(data, field_name):
                    label_data = getattr(data, field_name)
                    logger.debug(f"Found '{field_name}' field, type: {type(label_data)}")
                    
                    # Check if label has nested 'eeg' attribute (similar to fsample.eeg)
                    if hasattr(label_data, 'eeg'):
                        labels = label_data.eeg
                        label_source = f"{field_name}.eeg"
                        logger.debug(f"Found nested {field_name}.eeg structure")
                        break
                    elif isinstance(label_data, (list, np.ndarray)):
                        if len(label_data) > 0:
                            labels = label_data
                            label_source = field_name
                            logger.debug(f"Found {field_name} as array/list")
                            break
                    elif hasattr(label_data, '__dict__'):
                        # Check for nested attributes
                        label_attrs = [a for a in label_data.__dict__.keys() if not a.startswith('_')]
                        logger.debug(f"{field_name} attributes: {label_attrs}")
                        if 'eeg' in label_attrs:
                            labels = getattr(label_data, 'eeg')
                            label_source = f"{field_name}.eeg"
                            logger.debug(f"Found {field_name}.eeg in attributes")
                            break
                        elif 'label' in label_attrs:
                            labels = getattr(label_data, 'label')
                            label_source = f"{field_name}.label"
                            logger.debug(f"Found {field_name}.label in attributes")
                            break
                        else:
                            # Try to extract from the structure itself
                            labels = label_data
                            label_source = field_name
                            logger.debug(f"Using {field_name} structure directly")
                            break
                    else:
                        # Single value or mat_struct
                        labels = label_data
                        label_source = field_name
                        logger.debug(f"Using {field_name} as single value")
                        break
            
            if labels is not None:
                # Convert to list of strings
                try:
                    if isinstance(labels, np.ndarray):
                        if labels.dtype == object:
                            # Handle array of objects (e.g., mat_struct)
                            parsed['label'] = []
                            for l in labels.flatten():
                                if hasattr(l, 'label'):
                                    parsed['label'].append(str(l.label))
                                elif hasattr(l, 'value'):
                                    parsed['label'].append(str(l.value))
                                elif hasattr(l, '__dict__'):
                                    # Try to find label in nested structure
                                    if 'label' in l.__dict__:
                                        parsed['label'].append(str(l.label))
                                    else:
                                        parsed['label'].append(str(l))
                                else:
                                    parsed['label'].append(str(l))
                        else:
                            parsed['label'] = [str(l) for l in labels.flatten()]
                    elif isinstance(labels, list):
                        parsed['label'] = [str(l) if not hasattr(l, 'value') else str(l.value) for l in labels]
                    else:
                        # Single value or mat_struct
                        if hasattr(labels, 'value'):
                            parsed['label'] = [str(labels.value)]
                        elif hasattr(labels, 'label'):
                            parsed['label'] = [str(labels.label)]
                        else:
                            parsed['label'] = [str(labels)]
                    
                    # Validate label count matches channel count
                    if isinstance(parsed['eeg'], list):
                        n_channels = parsed['eeg'][0].shape[1] if len(parsed['eeg']) > 0 else 64
                    else:
                        n_channels = parsed['eeg'].shape[1] if len(parsed['eeg'].shape) > 1 else 64
                    
                    if len(parsed['label']) != n_channels:
                        logger.warning(f"Channel label count ({len(parsed['label'])}) doesn't match channel count ({n_channels})")
                        # Truncate or pad as needed
                        if len(parsed['label']) > n_channels:
                            parsed['label'] = parsed['label'][:n_channels]
                        else:
                            parsed['label'].extend([f'Ch{i+1}' for i in range(len(parsed['label']), n_channels)])
                    
                    logger.info(f"Channel labels extracted from {label_source}: {len(parsed['label'])} channels")
                    if len(parsed['label']) > 0:
                        logger.info(f"  First 10 labels: {parsed['label'][:10]}")
                        if len(parsed['label']) > 10:
                            logger.info(f"  Last 5 labels: {parsed['label'][-5:]}")
                except Exception as e:
                    logger.warning(f"Error parsing labels from {label_source}: {e}")
                    labels = None  # Fall through to fallback
            
            # Fallback: Try to load from channel_names.json
            if labels is None or 'label' not in parsed:
                try:
                    import json
                    channel_names_file = Path('channel_names.json')
                    if channel_names_file.exists():
                        with open(channel_names_file, 'r') as f:
                            channel_config = json.load(f)
                        
                        # Determine number of channels
                        if isinstance(parsed['eeg'], list):
                            n_channels = parsed['eeg'][0].shape[1] if len(parsed['eeg']) > 0 else 64
                        else:
                            n_channels = parsed['eeg'].shape[1] if len(parsed['eeg'].shape) > 1 else 64
                        
                        # Use Fuglsang channel names
                        if 'fulsang' in channel_config:
                            if n_channels <= 64:
                                # EEG channels only
                                parsed['label'] = channel_config['fulsang']['eeg_channels'][:n_channels]
                                label_source = "channel_names.json (EEG only)"
                            elif n_channels <= 71:
                                # All channels (EEG + EOG)
                                parsed['label'] = channel_config['fulsang']['all_channels'][:n_channels]
                                label_source = "channel_names.json (EEG+EOG)"
                            else:
                                # More channels than expected, use what we have and pad
                                base_labels = channel_config['fulsang']['all_channels']
                                parsed['label'] = base_labels + [f'Ch{i+1}' for i in range(len(base_labels), n_channels)]
                                label_source = "channel_names.json (padded)"
                            
                            logger.info(f"Channel labels loaded from {label_source}: {len(parsed['label'])} channels")
                            logger.info(f"  First 10 labels: {parsed['label'][:10]}")
                            if len(parsed['label']) > 10:
                                logger.info(f"  Last 5 labels: {parsed['label'][-5:]}")
                        else:
                            raise ValueError("Fulsang channel config not found in channel_names.json")
                    else:
                        raise FileNotFoundError("channel_names.json not found")
                except Exception as e:
                    logger.warning(f"Could not load channel names from channel_names.json: {e}")
                    # Final fallback: Create default channel labels
                    if isinstance(parsed['eeg'], list):
                        n_channels = parsed['eeg'][0].shape[1] if len(parsed['eeg']) > 0 else 64
                    else:
                        n_channels = parsed['eeg'].shape[1] if len(parsed['eeg'].shape) > 1 else 64
                    parsed['label'] = [f'Ch{i+1}' for i in range(n_channels)]
                    logger.warning(f"No channel labels found, created default labels for {n_channels} channels")
            
            # Extract events
            if hasattr(data, 'event'):
                parsed['event'] = self._parse_events(data.event)
                logger.info(f"Extracted {len(parsed['event']['samples'])} events")
            else:
                logger.warning("No event information found")
                parsed['event'] = {'samples': [], 'values': []}
            
            return parsed
            
        except Exception as e:
            logger.error(f"Error parsing EEG structure for subject {subject_id}: {e}")
            raise
    
    def _parse_events(self, event_data: any) -> Dict:
        """Parse event structure from MATLAB data."""
        events = {'samples': [], 'values': []}
        
        try:
            if hasattr(event_data, 'eeg'):
                eeg_events = event_data.eeg
                
                if hasattr(eeg_events, 'sample'):
                    samples = eeg_events.sample
                    if isinstance(samples, (list, np.ndarray)):
                        events['samples'] = np.array(samples).flatten()
                    else:
                        events['samples'] = np.array([samples])
                    logger.debug(f"Parsed {len(events['samples'])} event samples")
                
                if hasattr(eeg_events, 'value'):
                    values = eeg_events.value
                    parsed_values = []
                    
                    if isinstance(values, (list, np.ndarray)):
                        if len(values) > 0:
                            # Handle array of objects (dtype=object) - each element might be a scalar or object
                            for v in values:
                                try:
                                    # Check if it's an object with a 'value' attribute (mat_struct)
                                    if hasattr(v, 'value'):
                                        val = v.value
                                        # Handle nested structures
                                        if isinstance(val, (list, np.ndarray)):
                                            if len(val) > 0:
                                                parsed_values.append(val[0] if isinstance(val, np.ndarray) else val[0])
                                            else:
                                                continue
                                        else:
                                            parsed_values.append(val)
                                    # Handle numpy arrays (dtype=object with scalar values)
                                    elif isinstance(v, np.ndarray):
                                        if v.size == 1:
                                            parsed_values.append(v.item())
                                        else:
                                            parsed_values.append(v.flatten()[0])
                                    # Handle nested arrays
                                    elif isinstance(v, (list, np.ndarray)):
                                        if len(v) > 0:
                                            if isinstance(v, np.ndarray):
                                                if v.size == 1:
                                                    parsed_values.append(v.item())
                                                else:
                                                    parsed_values.append(v.flatten()[0])
                                            else:
                                                parsed_values.append(v[0] if len(v) > 0 else v)
                                        else:
                                            continue
                                    # Scalar value (int, float, etc.)
                                    else:
                                        parsed_values.append(int(v) if isinstance(v, (int, float, np.integer, np.floating)) else v)
                                except Exception as e:
                                    logger.debug(f"Error parsing event value: {e}")
                                    continue
                            
                            events['values'] = np.array(parsed_values, dtype=np.int64)
                        else:
                            events['values'] = []
                    else:
                        # Single value
                        events['values'] = np.array([int(values)] if isinstance(values, (int, float, np.integer, np.floating)) else [values], dtype=np.int64)
                    
                    # Log unique values to help debug
                    if len(events['values']) > 0:
                        unique_vals = np.unique(events['values'])
                        attention_codes = [v for v in unique_vals if v in [191, 192]]
                        logger.info(f"Parsed {len(events['values'])} event values")
                        if attention_codes:
                            logger.info(f"  Found attention codes: {attention_codes}")
                            count_191 = np.sum(events['values'] == 191)
                            count_192 = np.sum(events['values'] == 192)
                            logger.info(f"    Code 191 (Right): {count_191} occurrences")
                            logger.info(f"    Code 192 (Left): {count_192} occurrences")
                        else:
                            logger.warning(f"  No attention codes (191/192) found in {len(unique_vals)} unique event values: {unique_vals[:10]}")
                    else:
                        logger.debug(f"Parsed {len(events['values'])} event values")
            else:
                logger.warning("Event data does not have 'eeg' attribute")
            
            # Log summary
            if len(events['samples']) > 0:
                logger.info(f"Parsed events: {len(events['samples'])} samples, {len(events['values'])} values")
                if len(events['samples']) > 0:
                    logger.debug(f"  First sample: {events['samples'][0]}, Last sample: {events['samples'][-1]}")
            else:
                logger.warning("No event samples found")
            
            return events
            
        except Exception as e:
            logger.error(f"Error parsing events: {e}", exc_info=True)
            return events
    
    def extract_eeg_channels_only(self, eeg_data: np.ndarray, channel_labels: List[str]) -> np.ndarray:
        """
        Extract only EEG channels from Fuglsang dataset, excluding EOG channels.
        
        Fuglsang dataset has 64+8 channels including EOG (EXG3-EXG8).
        Extract only EEG channels 1-64, ignore EOG channels 67-72.
        This ensures compatibility with Das dataset for combining.
        """
        eeg_indices = []
        
        for i, label in enumerate(channel_labels):
            label_str = str(label).upper()
            # Exclude EOG channels (EXG3-EXG8) and STATUS channels
            # Only include standard EEG channels
            if 'EXG' not in label_str and 'EOG' not in label_str and 'STATUS' not in label_str:
                eeg_indices.append(i)
        
        # If no EEG indices found, use first 64 channels (assuming standard layout)
        if len(eeg_indices) == 0:
            logger.warning("No EEG channels identified by labels, using first 64 channels")
            eeg_indices = list(range(min(64, eeg_data.shape[1])))
        else:
            # Limit to first 64 EEG channels for consistency
            eeg_indices = eeg_indices[:64]
        
        # Extract only EEG channels
        eeg_channels = eeg_data[:, eeg_indices]
        
        logger.info(f"Extracted {len(eeg_indices)} EEG channels (EOG channels excluded)")
        
        return eeg_channels
    
    def apply_basic_artifact_removal(self, eeg_data: np.ndarray, fsample: float) -> np.ndarray:
        """
        Apply basic artifact removal without EOG reference.
        
        This method:
        1. Applies high-pass filtering (0.5 Hz) to remove slow drifts
        2. Applies notch filter (50/60 Hz) to remove line noise
        3. Removes bad channels based on variance
        4. Applies robust normalization
        
        This ensures compatibility with Das dataset which doesn't have EOG reference.
        """
        cleaned_eeg = eeg_data.copy()
        
        # 1. High-pass filter (0.5 Hz) to remove slow drifts
        nyquist = fsample / 2
        highpass_freq = 0.5 / nyquist
        if highpass_freq < 1.0:
            b, a = signal.butter(4, highpass_freq, btype='high')
            for ch in range(cleaned_eeg.shape[1]):
                cleaned_eeg[:, ch] = signal.filtfilt(b, a, cleaned_eeg[:, ch])
        
        # 2. Notch filter for line noise (50 Hz or 60 Hz depending on region)
        line_noise_freq = 50.0  # Can be 60.0 for US data
        notch_freq = line_noise_freq / nyquist
        if notch_freq < 1.0:
            b, a = signal.iirnotch(notch_freq, Q=30)
            for ch in range(cleaned_eeg.shape[1]):
                cleaned_eeg[:, ch] = signal.filtfilt(b, a, cleaned_eeg[:, ch])
        
        # 3. Remove bad channels (channels with extremely high variance)
        channel_variance = np.var(cleaned_eeg, axis=0)
        median_var = np.median(channel_variance)
        mad_var = np.median(np.abs(channel_variance - median_var))
        threshold = median_var + 5 * mad_var  # 5 MAD above median
        
        good_channels = channel_variance < threshold
        if np.sum(good_channels) < cleaned_eeg.shape[1]:
            logger.debug(f"Removed {np.sum(~good_channels)} bad channels based on variance")
            cleaned_eeg = cleaned_eeg[:, good_channels]
        
        # 4. Robust normalization (z-score using median and MAD)
        for ch in range(cleaned_eeg.shape[1]):
            ch_data = cleaned_eeg[:, ch]
            median_val = np.median(ch_data)
            mad_val = np.median(np.abs(ch_data - median_val))
            if mad_val > 0:
                cleaned_eeg[:, ch] = (ch_data - median_val) / mad_val
        
        return cleaned_eeg
    
    def downsample_eeg(self, eeg_data: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
        """Downsample EEG data from original_fs to target_fs."""
        if original_fs <= target_fs:
            return eeg_data
        
        downsample_factor = int(original_fs / target_fs)
        n_samples = eeg_data.shape[0]
        n_channels = eeg_data.shape[1]
        
        # Use decimation for downsampling
        # Note: decimate may return slightly different lengths, so we process channel by channel
        # and then trim/pad to ensure consistent length
        downsampled_channels = []
        for ch in range(n_channels):
            downsampled_ch = signal.decimate(eeg_data[:, ch], downsample_factor, ftype='iir')
            downsampled_channels.append(downsampled_ch)
        
        # Find the minimum length to ensure all channels have the same length
        min_length = min(len(ch) for ch in downsampled_channels)
        
        # Trim all channels to the same length
        downsampled_data = np.zeros((min_length, n_channels))
        for ch in range(n_channels):
            downsampled_data[:, ch] = downsampled_channels[ch][:min_length]
        
        logger.debug(f"Downsampled from {original_fs} Hz to {target_fs} Hz: {eeg_data.shape} -> {downsampled_data.shape}")
        
        return downsampled_data
    
    def _segment_continuous_eeg(self, eeg_data: np.ndarray, events: Dict, 
                                 trial_length_samples: int = None) -> List[np.ndarray]:
        """
        Segment continuous EEG data into trials based on event markers.
        
        Args:
            eeg_data: Continuous EEG data (samples x channels)
            events: Event dictionary with 'samples' and 'values' keys
            trial_length_samples: Length of each trial in samples. If None, auto-detects from events.
            
        Returns:
            List of trial arrays
        """
        eeg_trials, _ = self._segment_continuous_eeg_with_labels(eeg_data, events, trial_length_samples)
        return eeg_trials
    
    def _segment_continuous_eeg_with_labels(self, eeg_data: np.ndarray, events: Dict, 
                                            trial_length_samples: int = None) -> Tuple[List[np.ndarray], Dict[int, int]]:
        """
        Segment continuous EEG data into trials based on event markers and map to attention labels.
        
        This method filters events to only use attention-related codes (191, 192) for segmentation,
        ensuring each trial is properly labeled with its attention direction.
        
        Args:
            eeg_data: Continuous EEG data (samples x channels)
            events: Event dictionary with 'samples' and 'values' keys
            trial_length_samples: Length of each trial in samples. If None, auto-detects from events.
            
        Returns:
            Tuple of (List of trial arrays, Dict mapping trial_idx -> event_code)
        """
        event_samples = events.get('samples', [])
        event_values = events.get('values', [])
        
        if len(event_samples) == 0:
            logger.warning("No event samples found for segmentation. Using entire recording as single trial.")
            return [eeg_data], {}
        
        # Convert to numpy array and ensure it's 1D
        if isinstance(event_samples, (list, np.ndarray)):
            event_samples = np.array(event_samples).flatten()
        else:
            event_samples = np.array([event_samples])
        
        if isinstance(event_values, (list, np.ndarray)):
            event_values = np.array(event_values).flatten()
        else:
            event_values = np.array([event_values]) if event_values is not None else np.array([])
        
        # Create mapping of event_sample -> event_value for attention codes (191, 192)
        attention_event_map = {}
        if len(event_values) == len(event_samples):
            for sample, value in zip(event_samples, event_values):
                # Handle nested arrays in event values (common in MATLAB struct arrays)
                if isinstance(value, (list, np.ndarray)):
                    if len(value) > 0:
                        val = value[0] if isinstance(value, np.ndarray) else value[0]
                    else:
                        continue
                else:
                    val = value
                
                # Only keep attention codes (191 = Right, 192 = Left)
                if val in [191, 192]:
                    attention_event_map[int(sample)] = int(val)
        
        # If no attention events found, log warning but use all events
        if len(attention_event_map) == 0:
            logger.warning("No attention event codes (191/192) found in events, using all events for segmentation")
            if len(event_values) == len(event_samples):
                for sample, value in zip(event_samples, event_values):
                    if isinstance(value, (list, np.ndarray)):
                        if len(value) > 0:
                            val = value[0] if isinstance(value, np.ndarray) else value[0]
                        else:
                            continue
                    else:
                        val = value
                    attention_event_map[int(sample)] = int(val)
        
        # Use only attention events for segmentation
        attention_event_samples = np.array(sorted(attention_event_map.keys()))
        
        if len(attention_event_samples) == 0:
            logger.warning("No valid event samples found for segmentation. Using entire recording as single trial.")
            return [eeg_data], {}
        
        # Auto-detect trial length from event spacing if not provided
        if trial_length_samples is None:
            if len(attention_event_samples) > 1:
                # Use median spacing between consecutive attention events
                event_spacings = np.diff(attention_event_samples)
                median_spacing = np.median(event_spacings)
                trial_length_samples = int(median_spacing)
                logger.info(f"Auto-detected trial length: {trial_length_samples} samples ({trial_length_samples/512:.1f} seconds at 512 Hz)")
            else:
                # Default trial length: ~60 seconds at 512 Hz = 30720 samples
                trial_length_samples = int(60 * 512)
                logger.info(f"Using default trial length: {trial_length_samples} samples ({trial_length_samples/512:.1f} seconds)")
        
        # Sort event samples to ensure chronological order
        attention_event_samples = np.sort(attention_event_samples)
        
        trials = []
        trial_event_mapping = {}  # Maps trial_idx -> event_code
        
        for i, event_sample in enumerate(attention_event_samples):
            start_idx = int(event_sample)
            
            # Get the event code for this trial
            event_code = attention_event_map[int(event_sample)]
            
            # Determine end index
            if i < len(attention_event_samples) - 1:
                # Use next event as end (if it's not too far)
                next_event = int(attention_event_samples[i + 1])
                max_trial_length = trial_length_samples * 2  # Allow up to 2x default length
                if next_event - start_idx <= max_trial_length:
                    end_idx = next_event
                else:
                    end_idx = start_idx + trial_length_samples
            else:
                # Last event - use fixed trial length
                end_idx = start_idx + trial_length_samples
            
            if start_idx < 0:
                logger.warning(f"Event sample {event_sample} is negative, skipping")
                continue
            
            if start_idx >= eeg_data.shape[0]:
                logger.warning(f"Event sample {event_sample} exceeds data length ({eeg_data.shape[0]}), skipping")
                continue
            
            if end_idx > eeg_data.shape[0]:
                logger.debug(f"Trial starting at {start_idx} would exceed data length ({eeg_data.shape[0]}), truncating")
                end_idx = eeg_data.shape[0]
            
            if end_idx > start_idx:
                trial_data = eeg_data[start_idx:end_idx, :]
                if trial_data.shape[0] > 0:  # Ensure non-empty trial
                    trial_idx = len(trials)
                    trials.append(trial_data)
                    trial_event_mapping[trial_idx] = event_code
        
        logger.info(f"Segmented continuous EEG into {len(trials)} trials")
        right_count = sum(1 for v in trial_event_mapping.values() if v == 191)
        left_count = sum(1 for v in trial_event_mapping.values() if v == 192)
        logger.info(f"  Event codes (191=Right, 192=Left): {right_count} × 191, {left_count} × 192 (used for segmentation only; labels may come from expinfo)")
        
        # Warn about severe class imbalance only when event-based labels would be used (no expinfo fallback in this method)
        if right_count > 0 and left_count > 0:
            imbalance_ratio = max(right_count, left_count) / min(right_count, left_count)
            if imbalance_ratio > 10:
                logger.warning(f"  ⚠️  Event-code imbalance: {imbalance_ratio:.1f}:1 (191 vs 192). If expinfo is loaded, trial labels will use expinfo instead.")
        elif left_count == 0 or right_count == 0:
            logger.warning(f"  ⚠️  Event codes: only one code present ({right_count} × 191, {left_count} × 192). Trial labels will use expinfo if available.")
        if len(trials) > 0:
            logger.debug(f"  Trial shapes: {[t.shape for t in trials[:3]]}...")  # Show first 3 trial shapes
        
        return trials, trial_event_mapping
    
    def process_subject(self, subject_id: int) -> Dict:
        """Process subject data with MWF and downsampling."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Fuglsang subject: {subject_id}")
        logger.info(f"{'='*60}")
        
        # Load raw EEG data
        raw_data = self.load_raw_eeg_data(subject_id)
        parsed_data = raw_data['data']
        expinfo = raw_data['expinfo']
        
        eeg_data = parsed_data['eeg']
        fsample = parsed_data['fsample']
        channel_labels = parsed_data['label']
        events = parsed_data['event']
        
        # Check if eeg_data is continuous (2D array) or already segmented (list)
        trial_event_mapping = {}  # Maps trial_idx -> event_code (191 or 192)
        if isinstance(eeg_data, np.ndarray) and len(eeg_data.shape) == 2:
            # Continuous data - need to segment into trials
            logger.info("Detected continuous EEG data, segmenting into trials based on events...")
            eeg_trials, trial_event_mapping = self._segment_continuous_eeg_with_labels(eeg_data, events)
        elif isinstance(eeg_data, list):
            # Already segmented into trials
            if self.force_segment_from_events:
                raise ValueError(
                    "force_segment_from_events=True but EEG is already pre-segmented (list). "
                    "Provide a continuous EEG file (data.eeg as 2D array) or disable force mode."
                )
            eeg_trials = eeg_data
            logger.info(f"Using pre-segmented trials: {len(eeg_trials)} trials")
        else:
            # Single array - treat as single trial
            eeg_trials = [eeg_data] if not isinstance(eeg_data, list) else eeg_data
            logger.info(f"Using single trial or converted to list: {len(eeg_trials)} trials")
        
        processed_trials = []
        
        # Extract attention labels from expinfo (preferred source)
        attention_labels = []
        if hasattr(expinfo, 'attend_mf'):
            attention_labels = expinfo.attend_mf
            # Convert numpy array to list if needed
            if isinstance(attention_labels, np.ndarray):
                attention_labels = attention_labels.flatten().tolist()
            logger.info(f"Found attend_mf in expinfo: {len(attention_labels)} labels")
        elif hasattr(expinfo, 'attend_lr'):
            attention_labels = expinfo.attend_lr
            # Convert numpy array to list if needed
            if isinstance(attention_labels, np.ndarray):
                attention_labels = attention_labels.flatten().tolist()
            logger.info(f"Found attend_lr in expinfo: {len(attention_labels)} labels")
        # Log expinfo left/right distribution (attend_lr: 1=left, 2=right) so it's clear we use this, not event 191/192 counts
        if hasattr(expinfo, 'attend_lr'):
            lr = np.atleast_1d(np.asarray(expinfo.attend_lr).flatten())
            left_exp = int(np.sum(lr == 1))
            right_exp = int(np.sum(lr == 2))
            logger.info(f"  expinfo attend_lr: {left_exp} Left, {right_exp} Right (used for trial labels)")
            if left_exp > 0 and right_exp > 0:
                ratio = max(left_exp, right_exp) / min(left_exp, right_exp)
                if ratio > 10:
                    logger.warning(f"  ⚠️  SEVERE CLASS IMBALANCE in expinfo: {ratio:.1f}:1")
                else:
                    logger.info(f"  (Labels from expinfo; event codes 191/192 used only for segmentation.)")
        elif attention_labels and hasattr(expinfo, 'attend_mf'):
            v1 = sum(1 for x in attention_labels if x == 1)
            v2 = sum(1 for x in attention_labels if x == 2)
            logger.info(f"  expinfo attend_mf: {v1} value 1, {v2} value 2")
        
        # Enforce 1:1 alignment between EEG trials and expinfo labels (sidecar/raw).
        # Some files contain extra EEG segments with no label (practice/aborted/partial).
        trial_indices_to_use = list(range(len(eeg_trials)))
        if self.truncate_to_sidecar_labels and attention_labels is not None and len(attention_labels) > 0:
            if len(eeg_trials) != len(attention_labels):
                n_use = min(len(eeg_trials), len(attention_labels))
                logger.warning(
                    f"Trial/label count mismatch for S{subject_id}: EEG trials={len(eeg_trials)} vs labels={len(attention_labels)}. "
                    f"Using first {n_use} trials for 1:1 alignment."
                )
                trial_indices_to_use = list(range(n_use))
                attention_labels = attention_labels[:n_use]

        for used_trial_idx, orig_trial_idx in enumerate(tqdm(trial_indices_to_use, desc=f"Processing S{subject_id}")):
            try:
                eeg_data = eeg_trials[orig_trial_idx]
                # Ensure eeg_data is 2D (samples x channels)
                if len(eeg_data.shape) == 1:
                    eeg_data = eeg_data.reshape(-1, 1)
                elif len(eeg_data.shape) > 2:
                    eeg_data = eeg_data.reshape(eeg_data.shape[0], -1)
                
                # Extract only EEG channels (exclude EOG for dataset compatibility)
                eeg_channels = self.extract_eeg_channels_only(eeg_data, channel_labels)
                
                # Apply basic artifact removal without EOG reference
                # Use simple high-pass filtering and artifact rejection instead of MWF
                # This ensures compatibility with Das dataset which doesn't have EOG reference
                cleaned_eeg = self.apply_basic_artifact_removal(eeg_channels, fsample)
                
                # Downsample from 512 Hz to 128 Hz
                cleaned_eeg = self.downsample_eeg(cleaned_eeg, fsample, self.target_sampling_rate)
                
                # Get attention label - prioritize expinfo, then event codes, then trial_event_mapping
                attention_label = None
                
                # Method 1: Use expinfo labels (most reliable)
                if attention_labels is not None and len(attention_labels) > 0 and used_trial_idx < len(attention_labels):
                    attention_label = attention_labels[used_trial_idx]
                    # Convert to event code format if needed (0/1 -> 192/191)
                    if attention_label in [0, 1]:
                        attention_label = 192 if attention_label == 0 else 191
                
                # Method 2: Use event code from trial_event_mapping (for segmented continuous data)
                elif orig_trial_idx in trial_event_mapping:
                    event_code = trial_event_mapping[orig_trial_idx]
                    # Only use if it's an attention code (191 or 192)
                    if event_code in [191, 192]:
                        attention_label = event_code
                
                # Method 3: Fallback to event values array (if available and matches trial count)
                elif 'values' in events and events['values'] is not None:
                    event_values = events['values']
                    if isinstance(event_values, np.ndarray):
                        event_values = event_values.flatten()
                    # Filter to only attention codes
                    attention_event_values = [v for v in event_values if v in [191, 192]]
                    if len(attention_event_values) > 0 and orig_trial_idx < len(attention_event_values):
                        attention_label = attention_event_values[orig_trial_idx]
                    elif len(event_values) > 0 and orig_trial_idx < len(event_values):
                        # Check if the event value at this index is an attention code
                        event_val = event_values[orig_trial_idx]
                        if event_val in [191, 192]:
                            attention_label = event_val
                
                # Extract audio file info from expinfo if available
                audio_file_male = None
                audio_file_female = None
                if expinfo:
                    try:
                        if hasattr(expinfo, 'wavfile_male'):
                            male_files = expinfo.wavfile_male
                            if isinstance(male_files, (list, np.ndarray)) and used_trial_idx < len(male_files):
                                audio_file_male = str(male_files[used_trial_idx]) if len(male_files) > used_trial_idx else None
                        if hasattr(expinfo, 'wavfile_female'):
                            female_files = expinfo.wavfile_female
                            if isinstance(female_files, (list, np.ndarray)) and used_trial_idx < len(female_files):
                                audio_file_female = str(female_files[used_trial_idx]) if len(female_files) > used_trial_idx else None
                    except Exception as e:
                        logger.debug(f"Could not extract audio file info for trial {used_trial_idx}: {e}")
                
                processed_trial = {
                    'eeg_data': cleaned_eeg,
                    'sample_rate': self.target_sampling_rate,  # After downsampling
                    'original_sample_rate': fsample,
                    'attention_label': attention_label,
                    'trial_idx': used_trial_idx,
                    'original_trial_idx': orig_trial_idx,
                    'subject_id': subject_id,
                    'original_shape': eeg_data.shape,
                    'cleaned_shape': cleaned_eeg.shape,
                    'audio_file_male': audio_file_male,  # Save audio file info
                    'audio_file_female': audio_file_female
                }
                
                processed_trials.append(processed_trial)
                
            except Exception as e:
                logger.error(f"Error processing trial {orig_trial_idx}: {e}")
                continue
        
        logger.info(f"Processed {len(processed_trials)}/{len(trial_indices_to_use)} trials for subject {subject_id}")
        
        return {
            'subject_id': subject_id,
            'trials': processed_trials,
            'n_trials': len(processed_trials),
            'expinfo': expinfo
        }
    
    def save_cleaned_data(self, processed_data: Dict):
        """Save cleaned EEG data as 'subXX_MWF.mat'."""
        subject_id = processed_data['subject_id']
        output_file = self.output_dir / f"sub{subject_id:02d}_MWF.mat"
        
        save_dict = {
            'subject_id': subject_id,
            'n_trials': processed_data['n_trials'],
            'trials': []
        }
        # Save expinfo (attend_mf, attend_lr) so CombinedDataset can use it for left/right envelope assignment
        expinfo = processed_data.get('expinfo')
        if expinfo is not None:
            expinfo_save = {}
            try:
                if hasattr(expinfo, 'attend_mf'):
                    v = expinfo.attend_mf
                    expinfo_save['attend_mf'] = np.atleast_1d(np.asarray(v).flatten())
                if hasattr(expinfo, 'attend_lr'):
                    v = expinfo.attend_lr
                    expinfo_save['attend_lr'] = np.atleast_1d(np.asarray(v).flatten())
                if isinstance(expinfo, dict):
                    if 'attend_mf' in expinfo:
                        expinfo_save['attend_mf'] = np.atleast_1d(np.asarray(expinfo['attend_mf']).flatten())
                    if 'attend_lr' in expinfo:
                        expinfo_save['attend_lr'] = np.atleast_1d(np.asarray(expinfo['attend_lr']).flatten())
                if expinfo_save:
                    save_dict['expinfo'] = expinfo_save
            except Exception as e:
                logger.warning(f"Could not save expinfo to MWF file: {e}")
        
        for trial in processed_data['trials']:
            trial_dict = {
                'eeg_data': trial['eeg_data'],
                'sample_rate': trial['sample_rate'],
                'original_sample_rate': trial['original_sample_rate'],
                'attention_label': trial['attention_label'],
                'trial_idx': trial['trial_idx']
            }
            # Save audio file info if available
            if 'audio_file_male' in trial and trial['audio_file_male']:
                trial_dict['audio_file_male'] = trial['audio_file_male']
            if 'audio_file_female' in trial and trial['audio_file_female']:
                trial_dict['audio_file_female'] = trial['audio_file_female']
            save_dict['trials'].append(trial_dict)
        
        sio.savemat(str(output_file), save_dict)
        logger.info(f"Saved cleaned data to {output_file}")
    
    def process_all_subjects(self):
        """Process all 18 subjects."""
        all_results = {}
        
        for subject_id in range(1, 19):
            try:
                processed_data = self.process_subject(subject_id)
                if processed_data:
                    self.save_cleaned_data(processed_data)
                    all_results[processed_data['subject_id']] = processed_data['n_trials']
            except FileNotFoundError:
                logger.warning(f"Subject {subject_id} file not found, skipping")
                continue
            except Exception as e:
                logger.error(f"Error processing subject {subject_id}: {e}")
                continue
        
        # Save summary
        summary_file = self.output_dir / "processing_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Fuglsang Dataset MWF Processing Summary\n")
            f.write("=" * 60 + "\n\n")
            for subject_id, n_trials in all_results.items():
                f.write(f"S{subject_id}: {n_trials} trials processed\n")
            f.write(f"\nTotal subjects: {len(all_results)}\n")
            f.write(f"Total trials: {sum(all_results.values())}\n")
        
        logger.info(f"\nProcessing complete! Summary saved to {summary_file}")
        return all_results


def visualize_mwf_results(eeg_before: np.ndarray, eeg_after: np.ndarray, 
                          sample_rate: float, subject_id: str, trial_idx: int,
                          output_dir: Path, dataset_name: str):
    """
    Visualize MWF results: before/after comparison, PSD, and variance reduction.
    
    Args:
        eeg_before: EEG data before MWF, shape (n_samples, n_channels)
        eeg_after: EEG data after MWF, shape (n_samples, n_channels)
        sample_rate: Sampling rate in Hz
        subject_id: Subject identifier
        trial_idx: Trial index
        output_dir: Output directory for figures
        dataset_name: Name of dataset ('Das' or 'Fuglsang')
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Time series comparison (first 5 seconds, first 5 channels)
    ax1 = plt.subplot(3, 2, 1)
    n_samples_plot = min(int(5 * sample_rate), eeg_before.shape[0])
    n_channels_plot = min(5, eeg_before.shape[1])
    
    time_axis = np.arange(n_samples_plot) / sample_rate
    
    for ch in range(n_channels_plot):
        ax1.plot(time_axis, eeg_before[:n_samples_plot, ch] + ch * 50, 
                'b-', alpha=0.6, linewidth=0.5, label='Before' if ch == 0 else '')
        ax1.plot(time_axis, eeg_after[:n_samples_plot, ch] + ch * 50, 
                'r-', alpha=0.8, linewidth=0.5, label='After' if ch == 0 else '')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Channel (offset)')
    ax1.set_title(f'EEG Time Series (First 5s, {n_channels_plot} channels)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Power Spectral Density
    ax2 = plt.subplot(3, 2, 2)
    freqs_before, psd_before = signal.welch(eeg_before[:, 0], fs=sample_rate, nperseg=min(2048, len(eeg_before)))
    freqs_after, psd_after = signal.welch(eeg_after[:, 0], fs=sample_rate, nperseg=min(2048, len(eeg_after)))
    
    ax2.semilogy(freqs_before, psd_before, 'b-', alpha=0.7, label='Before MWF', linewidth=2)
    ax2.semilogy(freqs_after, psd_after, 'r-', alpha=0.7, label='After MWF', linewidth=2)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power Spectral Density')
    ax2.set_title('Power Spectral Density (Channel 0)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, min(50, sample_rate / 2))
    
    # 3. Channel variance reduction
    ax3 = plt.subplot(3, 2, 3)
    var_before = np.var(eeg_before, axis=0)
    var_after = np.var(eeg_after, axis=0)
    variance_reduction = (var_before - var_after) / var_before * 100
    
    channels = np.arange(len(var_before))
    ax3.bar(channels, variance_reduction, alpha=0.7, color='green')
    ax3.set_xlabel('Channel')
    ax3.set_ylabel('Variance Reduction (%)')
    ax3.set_title('Variance Reduction per Channel')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Average variance across channels
    ax4 = plt.subplot(3, 2, 4)
    mean_var_before = np.mean(var_before)
    mean_var_after = np.mean(var_after)
    
    ax4.bar(['Before MWF', 'After MWF'], [mean_var_before, mean_var_after], 
            color=['blue', 'red'], alpha=0.7)
    ax4.set_ylabel('Mean Variance')
    ax4.set_title('Average Variance Across Channels')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Histogram of amplitudes
    ax5 = plt.subplot(3, 2, 5)
    ax5.hist(eeg_before.flatten(), bins=50, alpha=0.6, label='Before', color='blue', density=True)
    ax5.hist(eeg_after.flatten(), bins=50, alpha=0.6, label='After', color='red', density=True)
    ax5.set_xlabel('Amplitude')
    ax5.set_ylabel('Density')
    ax5.set_title('Amplitude Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary statistics
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
    
    stats_text = f"""
    MWF Processing Summary
    
    Dataset: {dataset_name}
    Subject: {subject_id}
    Trial: {trial_idx}
    Sampling Rate: {sample_rate} Hz
    
    Before MWF:
      Shape: {eeg_before.shape}
      Mean: {np.mean(eeg_before):.4f}
      Std: {np.std(eeg_before):.4f}
      Variance: {np.var(eeg_before):.4f}
    
    After MWF:
      Shape: {eeg_after.shape}
      Mean: {np.mean(eeg_after):.4f}
      Std: {np.std(eeg_after):.4f}
      Variance: {np.var(eeg_after):.4f}
    
    Variance Reduction: {np.mean(variance_reduction):.2f}%
    """
    
    ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', horizontalalignment='left')
    
    plt.suptitle(f'MWF Artifact Removal: {dataset_name} - {subject_id} - Trial {trial_idx}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / f"{dataset_name}_{subject_id}_trial{trial_idx}_MWF_verification.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visualization to {output_file}")


def preprocess_and_apply_mwf(dataset_type: str, input_path: str, output_path: str, 
                             visualize: bool = True, audio_dir: str = None) -> Dict:
    """
    Unified function to preprocess and apply MWF to Das or Fuglsang datasets.
    
    Args:
        dataset_type: 'Das' or 'Fuglsang'
        input_path: Path to input data directory
        output_path: Path to output directory
        visualize: Whether to create visualization plots
        
    Returns:
        Dictionary with processing summary
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Unified MWF Preprocessing: {dataset_type} Dataset")
    logger.info(f"{'='*60}")
    
    summary = {
        'dataset_type': dataset_type,
        'input_path': input_path,
        'output_path': output_path,
        'subjects_processed': {},
        'total_trials': 0,
        'sampling_rates': {}
    }
    
    if dataset_type.lower() == 'das':
        processor = DasDatasetMWF(
            data_dir=input_path, 
            output_dir=output_path,
            audio_dir=audio_dir if audio_dir else "Data/Das/4004271/stimuli/stimuli"
        )
        results = processor.process_all_subjects()
        
        # Create visualization for one trial if requested
        if visualize and results:
            first_subject = list(results.keys())[0]
            subject_file = processor.data_dir / f"{first_subject}.mat"
            processed_data = processor.process_subject(subject_file.name)
            
            if processed_data and len(processed_data['trials']) > 0:
                # Load original data for comparison
                trials = processor.load_subject_data(subject_file.name)
                if trials:
                    trial = processed_data['trials'][0]
                    original_trial = trials[0]
                    
                    # Get original and cleaned EEG
                    eeg_before = original_trial['eeg_data']
                    eeg_after = trial['eeg_data']
                    
                    # Ensure same shape for visualization
                    min_samples = min(eeg_before.shape[0], eeg_after.shape[0])
                    eeg_before = eeg_before[:min_samples, :]
                    eeg_after = eeg_after[:min_samples, :]
                    
                    vis_dir = Path(output_path) / "Results" / "MWF_verification"
                    visualize_mwf_results(
                        eeg_before, eeg_after, 
                        trial['sample_rate'],
                        first_subject, 0,
                        vis_dir, 'Das'
                    )
        
        summary['subjects_processed'] = results
        summary['total_trials'] = sum(results.values())
        summary['sampling_rates'] = {k: 128 for k in results.keys()}
        
    elif dataset_type.lower() == 'fuglsang':
        processor = FuglsangDatasetMWF(
            eeg_base_path=input_path,
            output_dir=output_path
        )
        results = processor.process_all_subjects()
        
        # Create visualization for one trial if requested
        if visualize and results:
            first_subject = list(results.keys())[0]
            processed_data = processor.process_subject(first_subject)
            
            if processed_data and len(processed_data['trials']) > 0:
                # Load original data for comparison
                raw_data = processor.load_raw_eeg_data(first_subject)
                parsed_data = raw_data['data']
                eeg_trials = parsed_data['eeg']
                fsample = parsed_data['fsample']
                
                if not isinstance(eeg_trials, list):
                    eeg_trials = [eeg_trials]
                
                if eeg_trials:
                    trial = processed_data['trials'][0]
                    eeg_before = eeg_trials[0]
                    
                    # Downsample original for comparison
                    eeg_before = processor.downsample_eeg(eeg_before, fsample, processor.target_sampling_rate)
                    eeg_after = trial['eeg_data']
                    
                    # Ensure same shape
                    min_samples = min(eeg_before.shape[0], eeg_after.shape[0])
                    eeg_before = eeg_before[:min_samples, :]
                    eeg_after = eeg_after[:min_samples, :]
                    
                    vis_dir = Path(output_path) / "Results" / "MWF_verification"
                    visualize_mwf_results(
                        eeg_before, eeg_after,
                        trial['sample_rate'],
                        f"S{first_subject}", 0,
                        vis_dir, 'Fuglsang'
                    )
        
        summary['subjects_processed'] = results
        summary['total_trials'] = sum(results.values())
        summary['sampling_rates'] = {f"S{k}": 128 for k in results.keys()}
        
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Use 'Das' or 'Fuglsang'")
    
    # Log summary
    logger.info(f"\n{'='*60}")
    logger.info("Processing Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Dataset: {dataset_type}")
    logger.info(f"Subjects processed: {len(summary['subjects_processed'])}")
    logger.info(f"Total trials: {summary['total_trials']}")
    logger.info(f"Output directory: {output_path}")
    
    return summary


def main():
    """Main function to process both datasets."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Apply MWF artifact removal to Das and Fuglsang datasets')
    parser.add_argument('--das_dir', type=str, default='Data/Das/4004271',
                       help='Das dataset directory')
    parser.add_argument('--das_audio_dir', type=str, default='Data/Das/4004271/stimuli/stimuli',
                       help='Das audio directory for envelope extraction')
    parser.add_argument('--fuglsang_eeg_dir', type=str, default='/home/py9363/telluride_decoding/Data/Fulsang/EEG',
                       help='Fuglsang EEG directory')
    parser.add_argument('--fuglsang_audio_dir', type=str, default='Data/Fulsang/AUDIO',
                       help='Fuglsang audio directory')
    parser.add_argument('--no_truncate_to_sidecar_labels', dest='truncate_to_sidecar_labels', action='store_false', default=True,
                       help='Disable truncation when EEG trials != expinfo label count (default: truncate to match labels).')
    parser.add_argument('--no_force_segment_from_events', dest='force_segment_from_events', action='store_false', default=True,
                       help='Allow using pre-segmented trials if present (default: always segment continuous EEG from data.event.eeg).')
    parser.add_argument('--dataset', type=str, choices=['das', 'fuglsang', 'both'], default='both',
                       help='Which dataset to process')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization plots')
    parser.add_argument('--unified', action='store_true',
                       help='Use unified preprocessing function')
    
    args = parser.parse_args()
    
    if args.unified:
        # Use unified function
        if args.dataset in ['das', 'both']:
            summary = preprocess_and_apply_mwf(
                'Das', args.das_dir, 'MWF_cleaned_DAS', 
                visualize=args.visualize,
                audio_dir=args.das_audio_dir
            )
        
        if args.dataset in ['fuglsang', 'both']:
            summary = preprocess_and_apply_mwf(
                'Fuglsang', args.fuglsang_eeg_dir, 'MWF_cleaned_Fuglsang',
                visualize=args.visualize
            )
    else:
        # Use individual processors
        if args.dataset in ['das', 'both']:
            logger.info("\n" + "="*60)
            logger.info("Processing Das Dataset")
            logger.info("="*60)
            das_processor = DasDatasetMWF(
                data_dir=args.das_dir, 
                output_dir='MWF_cleaned_DAS',
                audio_dir=args.das_audio_dir
            )
            das_results = das_processor.process_all_subjects()
            logger.info(f"Das dataset: {len(das_results)} subjects processed")
        
        if args.dataset in ['fuglsang', 'both']:
            logger.info("\n" + "="*60)
            logger.info("Processing Fuglsang Dataset")
            logger.info("="*60)
            fuglsang_processor = FuglsangDatasetMWF(
                eeg_base_path=args.fuglsang_eeg_dir,
                audio_base_path=args.fuglsang_audio_dir,
                output_dir='MWF_cleaned_Fuglsang',
                truncate_to_sidecar_labels=getattr(args, 'truncate_to_sidecar_labels', True),
                force_segment_from_events=getattr(args, 'force_segment_from_events', True)
            )
            fuglsang_results = fuglsang_processor.process_all_subjects()
            logger.info(f"Fuglsang dataset: {len(fuglsang_results)} subjects processed")
    
    logger.info("\n" + "="*60)
    logger.info("MWF Processing Complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
