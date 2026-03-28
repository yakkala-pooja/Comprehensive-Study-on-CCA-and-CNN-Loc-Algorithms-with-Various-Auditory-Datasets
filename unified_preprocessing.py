#!/usr/bin/env python3
"""
Unified Preprocessing for Das and Fuglsang Datasets

This script applies the same preprocessing pipeline to both datasets:
- Removes EOG channels (ignores them completely)
- Downsampling (Fuglsang: 512 Hz -> 128 Hz, Das: already 128 Hz)
- Bandpass filtering (1-40 Hz)
- Artifact removal (outlier detection and interpolation)
- MAD-based normalization
- Soft clipping

Features:
- Unified preprocessing pipeline for both datasets
- EOG channels are removed, not used
- Consistent processing across datasets
"""

import os
import sys
import numpy as np
import scipy.io as sio
from scipy import signal
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedPreprocessor:
    """
    Unified preprocessing for Das and Fuglsang datasets.
    Applies the same preprocessing pipeline to both, ignoring EOG channels.
    """
    
    def __init__(self, target_sampling_rate: int = 128):
        """
        Initialize unified preprocessor.
        
        Args:
            target_sampling_rate: Target sampling rate in Hz (default: 128 Hz)
        """
        self.target_sampling_rate = target_sampling_rate
        self.bandpass_low = 1.0  # Hz
        self.bandpass_high = 40.0  # Hz
        self.artifact_threshold = 5.0  # Standard deviations for artifact detection
        
        logger.info(f"UnifiedPreprocessor initialized")
        logger.info(f"  Target sampling rate: {self.target_sampling_rate} Hz")
        logger.info(f"  Bandpass filter: {self.bandpass_low}-{self.bandpass_high} Hz")
        logger.info(f"  Artifact threshold: {self.artifact_threshold} std")
    
    def remove_eog_channels(self, eeg_data: np.ndarray, channel_labels: Optional[List[str]] = None) -> Tuple[np.ndarray, List[int]]:
        """
        Remove EOG channels from EEG data.
        
        Args:
            eeg_data: EEG data, shape (n_samples, n_channels)
            channel_labels: Optional list of channel labels
            
        Returns:
            Tuple of (eeg_data_without_eog, eeg_channel_indices)
        """
        n_channels = eeg_data.shape[1]
        eog_indices = []
        eeg_indices = []
        
        if channel_labels is not None:
            # Identify EOG channels by label
            for i, label in enumerate(channel_labels):
                label_str = str(label).upper()
                # Check for EOG/EXG channels
                if 'EOG' in label_str or 'EXG' in label_str or 'STATUS' in label_str:
                    eog_indices.append(i)
                else:
                    eeg_indices.append(i)
        
        if len(eog_indices) == 0 and channel_labels is None:
            # If no labels provided and no EOG found, use all channels
            logger.warning("No channel labels provided, using all channels")
            eeg_indices = list(range(n_channels))
        elif len(eog_indices) == 0:
            # If labels provided but no EOG found, use all channels
            logger.info("No EOG channels found in labels, using all channels")
            eeg_indices = list(range(n_channels))
        else:
            logger.info(f"Removing {len(eog_indices)} EOG/EXG channels, keeping {len(eeg_indices)} EEG channels")
        
        # Extract EEG channels only
        if len(eeg_indices) > 0:
            eeg_data_clean = eeg_data[:, eeg_indices]
        else:
            # Fallback: use all channels if no EEG indices found
            logger.warning("No EEG channels identified, using all channels")
            eeg_data_clean = eeg_data
            eeg_indices = list(range(n_channels))
        
        return eeg_data_clean, eeg_indices
    
    def downsample(self, eeg_data: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
        """
        Downsample EEG data if needed.
        
        Args:
            eeg_data: EEG data, shape (n_samples, n_channels)
            original_fs: Original sampling rate in Hz
            target_fs: Target sampling rate in Hz
            
        Returns:
            Downsampled EEG data
        """
        if original_fs <= target_fs:
            logger.debug(f"No downsampling needed: {original_fs} Hz <= {target_fs} Hz")
            return eeg_data
        
        downsample_factor = int(original_fs / target_fs)
        n_samples = eeg_data.shape[0]
        n_channels = eeg_data.shape[1]
        
        # Use decimation for downsampling
        downsampled_data = np.zeros((n_samples // downsample_factor, n_channels))
        for ch in range(n_channels):
            downsampled_data[:, ch] = signal.decimate(eeg_data[:, ch], downsample_factor, ftype='iir')
        
        logger.debug(f"Downsampled from {original_fs} Hz to {target_fs} Hz: {eeg_data.shape} -> {downsampled_data.shape}")
        
        return downsampled_data
    
    def apply_bandpass_filter(self, eeg_data: np.ndarray, sample_rate: float) -> np.ndarray:
        """
        Apply bandpass filter (1-40 Hz).
        
        Args:
            eeg_data: EEG data, shape (n_samples, n_channels)
            sample_rate: Sampling rate in Hz
            
        Returns:
            Filtered EEG data
        """
        nyquist = sample_rate / 2
        low_freq = self.bandpass_low / nyquist
        high_freq = min(self.bandpass_high / nyquist, 0.99)
        
        # Design Butterworth filter
        b, a = signal.butter(4, [low_freq, high_freq], btype='band')
        
        # Apply filter to each channel
        filtered_eeg = np.zeros_like(eeg_data)
        for ch in range(eeg_data.shape[1]):
            filtered_eeg[:, ch] = signal.filtfilt(b, a, eeg_data[:, ch])
        
        logger.debug(f"Applied bandpass filter ({self.bandpass_low}-{self.bandpass_high} Hz)")
        
        return filtered_eeg
    
    def remove_artifacts(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Remove artifacts using outlier detection and interpolation.
        
        Args:
            eeg_data: EEG data, shape (n_samples, n_channels)
            
        Returns:
            Cleaned EEG data
        """
        cleaned_eeg = eeg_data.copy()
        
        for ch in range(eeg_data.shape[1]):
            channel_data = cleaned_eeg[:, ch]
            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)
            
            # Detect artifacts (>threshold standard deviations)
            artifacts = np.abs(channel_data - mean_val) > (self.artifact_threshold * std_val)
            
            if np.any(artifacts):
                # Interpolate over artifacts
                valid_indices = np.where(~artifacts)[0]
                if len(valid_indices) > 1:
                    from scipy.interpolate import interp1d
                    interp_func = interp1d(
                        valid_indices, 
                        channel_data[valid_indices], 
                        kind='linear', 
                        fill_value='extrapolate',
                        bounds_error=False
                    )
                    cleaned_eeg[:, ch] = interp_func(np.arange(len(channel_data)))
                    logger.debug(f"Removed {np.sum(artifacts)} artifacts from channel {ch}")
        
        return cleaned_eeg
    
    def normalize_mad(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Apply MAD-based normalization.
        
        Args:
            eeg_data: EEG data, shape (n_samples, n_channels)
            
        Returns:
            Normalized EEG data
        """
        # Compute MAD (Median Absolute Deviation) for each channel
        median_values = np.median(eeg_data, axis=0, keepdims=True)
        mad_values = np.median(np.abs(eeg_data - median_values), axis=0, keepdims=True)
        
        # Avoid division by zero
        mad_values = np.where(mad_values == 0, 1.0, mad_values)
        
        # Normalize
        normalized_eeg = (eeg_data - median_values) / mad_values
        
        logger.debug("Applied MAD-based normalization")
        
        return normalized_eeg
    
    def apply_soft_clipping(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Apply soft clipping to prevent extreme values.
        
        Args:
            eeg_data: EEG data, shape (n_samples, n_channels)
            
        Returns:
            Soft-clipped EEG data
        """
        clipped_eeg = np.tanh(eeg_data * 0.5)
        logger.debug("Applied soft clipping")
        return clipped_eeg
    
    def preprocess(self, eeg_data: np.ndarray, sample_rate: float, 
                  channel_labels: Optional[List[str]] = None) -> np.ndarray:
        """
        Apply complete preprocessing pipeline.
        
        Args:
            eeg_data: EEG data, shape (n_samples, n_channels)
            sample_rate: Original sampling rate in Hz
            channel_labels: Optional list of channel labels
            
        Returns:
            Preprocessed EEG data
        """
        # Step 1: Remove EOG channels
        eeg_data, eeg_indices = self.remove_eog_channels(eeg_data, channel_labels)
        
        # Step 2: Downsample if needed
        if sample_rate != self.target_sampling_rate:
            eeg_data = self.downsample(eeg_data, sample_rate, self.target_sampling_rate)
            sample_rate = self.target_sampling_rate
        
        # Step 3: Bandpass filtering
        eeg_data = self.apply_bandpass_filter(eeg_data, sample_rate)
        
        # Step 4: Artifact removal
        eeg_data = self.remove_artifacts(eeg_data)
        
        # Step 5: MAD-based normalization
        eeg_data = self.normalize_mad(eeg_data)
        
        # Step 6: Soft clipping
        eeg_data = self.apply_soft_clipping(eeg_data)
        
        # Step 7: Final quality check
        if np.any(np.isnan(eeg_data)) or np.any(np.isinf(eeg_data)):
            logger.warning("Invalid values detected after preprocessing, replacing with zeros")
            eeg_data = np.nan_to_num(eeg_data, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return eeg_data.astype(np.float32)


class DasDatasetPreprocessor:
    """
    Das dataset loader and preprocessor.
    
    Das dataset:
    - 16 subjects
    - 20 trials per subject stored as 'Sx.mat' files
    - EEG already high-pass filtered (0.5 Hz) and downsampled to 128 Hz
    """
    
    def __init__(self, data_dir: str = "Data/Das/4004271", output_dir: str = "preprocessed_Das"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.preprocessor = UnifiedPreprocessor(target_sampling_rate=128)
        
        logger.info(f"Das Dataset Preprocessor initialized")
        logger.info(f"  Data directory: {self.data_dir}")
        logger.info(f"  Output directory: {self.output_dir}")
    
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
                    
                    # Try to get channel labels if available
                    channel_labels = None
                    if hasattr(trial.RawData, 'ChannelLabels'):
                        channel_labels = trial.RawData.ChannelLabels
                    elif hasattr(trial.FileHeader, 'ChannelLabels'):
                        channel_labels = trial.FileHeader.ChannelLabels
                    
                    trial_dict = {
                        'eeg_data': eeg_data,
                        'sample_rate': sample_rate,
                        'attended_ear': attended_ear,
                        'stimuli': stimuli,
                        'trial_idx': trial_idx,
                        'subject_id': Path(subject_file).stem,
                        'channel_labels': channel_labels
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
    
    def process_subject(self, subject_file: str) -> Dict:
        """Process all trials for a subject."""
        subject_id = Path(subject_file).stem
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing subject: {subject_id}")
        logger.info(f"{'='*60}")
        
        # Load trials
        trials = self.load_subject_data(subject_file)
        
        if len(trials) == 0:
            logger.warning(f"No trials found for {subject_id}")
            return {}
        
        processed_trials = []
        
        for trial in tqdm(trials, desc=f"Processing {subject_id}"):
            try:
                eeg_data = trial['eeg_data']
                sample_rate = trial['sample_rate']
                channel_labels = trial.get('channel_labels', None)
                
                # Ensure data is 2D (samples x channels)
                if len(eeg_data.shape) != 2:
                    if len(eeg_data.shape) == 1:
                        eeg_data = eeg_data.reshape(-1, 1)
                    else:
                        eeg_data = eeg_data.reshape(eeg_data.shape[0], -1)
                
                # Apply unified preprocessing (removes EOG, applies all steps)
                preprocessed_eeg = self.preprocessor.preprocess(
                    eeg_data, sample_rate, channel_labels
                )
                
                # Store processed trial
                processed_trial = {
                    'eeg_data': preprocessed_eeg,
                    'sample_rate': self.preprocessor.target_sampling_rate,
                    'attended_ear': trial['attended_ear'],
                    'stimuli': trial['stimuli'],
                    'trial_idx': trial['trial_idx'],
                    'subject_id': subject_id,
                    'original_shape': eeg_data.shape,
                    'preprocessed_shape': preprocessed_eeg.shape
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
    
    def save_cleaned_data(self, processed_data: Dict):
        """Save preprocessed EEG data as 'Sx_preprocessed.mat'."""
        subject_id = processed_data['subject_id']
        output_file = self.output_dir / f"{subject_id}_preprocessed.mat"
        
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
            save_dict['trials'].append(trial_dict)
        
        sio.savemat(str(output_file), save_dict)
        logger.info(f"Saved preprocessed data to {output_file}")
    
    def process_all_subjects(self):
        """Process all subjects in the dataset."""
        subject_files = list(self.data_dir.glob("S*.mat"))
        subject_files.sort()
        
        if not subject_files:
            raise ValueError(f"No subject files found in {self.data_dir}")
        
        logger.info(f"Found {len(subject_files)} subject files")
        
        all_results = {}
        
        for subject_file in subject_files:
            try:
                processed_data = self.process_subject(subject_file.name)
                if processed_data:
                    self.save_cleaned_data(processed_data)
                    all_results[processed_data['subject_id']] = processed_data['n_trials']
            except Exception as e:
                logger.error(f"Error processing {subject_file.name}: {e}")
                continue
        
        # Save summary
        summary_file = self.output_dir / "processing_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Das Dataset Unified Preprocessing Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write("Preprocessing steps:\n")
            f.write("  1. Remove EOG channels\n")
            f.write("  2. Downsample to 128 Hz (if needed)\n")
            f.write("  3. Bandpass filter (1-40 Hz)\n")
            f.write("  4. Artifact removal (outlier detection + interpolation)\n")
            f.write("  5. MAD-based normalization\n")
            f.write("  6. Soft clipping\n\n")
            for subject_id, n_trials in all_results.items():
                f.write(f"{subject_id}: {n_trials} trials processed\n")
            f.write(f"\nTotal subjects: {len(all_results)}\n")
            f.write(f"Total trials: {sum(all_results.values())}\n")
        
        logger.info(f"\nProcessing complete! Summary saved to {summary_file}")
        return all_results


class FuglsangDatasetPreprocessor:
    """
    Fuglsang dataset loader and preprocessor.
    
    Fuglsang dataset:
    - 18 subjects
    - 64+8 EOG/mastoid channels, sampled at 512 Hz
    - Organized in COCOHA Matlab Toolbox format
    """
    
    def __init__(self, 
                 eeg_base_path: str = "Data/Fulsang/eeg",
                 output_dir: str = "preprocessed_Fuglsang"):
        self.eeg_base_path = Path(eeg_base_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.original_sampling_rate = 512  # Hz
        self.preprocessor = UnifiedPreprocessor(target_sampling_rate=128)
        
        logger.info(f"Fuglsang Dataset Preprocessor initialized")
        logger.info(f"  EEG directory: {self.eeg_base_path}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Downsampling: {self.original_sampling_rate} Hz -> {self.preprocessor.target_sampling_rate} Hz")
    
    def load_raw_eeg_data(self, subject_id: int) -> Dict:
        """Load raw EEG data for a subject."""
        eeg_file = self.eeg_base_path / f"S{subject_id}.mat"
        
        if not eeg_file.exists():
            raise FileNotFoundError(f"EEG file not found: {eeg_file}")
        
        logger.info(f"Loading raw EEG data for subject {subject_id}")
        
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
            
            # Extract experimental info
            expinfo = mat_data.get('expinfo', {})
            if not expinfo:
                for info_name in ['exp_info', 'experiment_info', 'info']:
                    if info_name in mat_data:
                        expinfo = mat_data[info_name]
                        break
            
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
                        parsed['eeg'] = [np.array(e) for e in eeg_data]
                    else:
                        parsed['eeg'] = np.array(eeg_data)
                else:
                    parsed['eeg'] = np.array([eeg_data])
                logger.info(f"Extracted EEG data: {len(parsed['eeg'])} trials")
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
            
            # Extract channel labels
            if hasattr(data, 'label'):
                labels = data.label
                if isinstance(labels, (list, np.ndarray)):
                    parsed['label'] = [str(l) for l in labels]
                else:
                    parsed['label'] = [str(labels)]
                logger.info(f"Channel labels: {len(parsed['label'])} channels")
            elif hasattr(data, 'channels'):
                parsed['label'] = data.channels
            else:
                # Create default channel labels
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
                
                if hasattr(eeg_events, 'value'):
                    values = eeg_events.value
                    if isinstance(values, (list, np.ndarray)):
                        if len(values) > 0 and isinstance(values[0], (list, np.ndarray)):
                            events['values'] = [v.item() if hasattr(v, 'item') else v for v in values]
                        else:
                            events['values'] = np.array(values).flatten()
                    else:
                        events['values'] = [values]
            
            return events
            
        except Exception as e:
            logger.warning(f"Error parsing events: {e}")
            return events
    
    def process_subject(self, subject_id: int) -> Dict:
        """Process subject data with unified preprocessing."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Fuglsang subject: {subject_id}")
        logger.info(f"{'='*60}")
        
        # Load raw EEG data
        raw_data = self.load_raw_eeg_data(subject_id)
        parsed_data = raw_data['data']
        expinfo = raw_data['expinfo']
        
        eeg_trials = parsed_data['eeg']
        fsample = parsed_data['fsample']
        channel_labels = parsed_data['label']
        events = parsed_data['event']
        
        # Convert to list if single trial
        if not isinstance(eeg_trials, list):
            eeg_trials = [eeg_trials]
        
        processed_trials = []
        
        # Extract attention labels from expinfo
        attention_labels = []
        if hasattr(expinfo, 'attend_mf'):
            attention_labels = expinfo.attend_mf
        elif hasattr(expinfo, 'attend_lr'):
            attention_labels = expinfo.attend_lr
        
        for trial_idx, eeg_data in enumerate(tqdm(eeg_trials, desc=f"Processing S{subject_id}")):
            try:
                # Ensure eeg_data is 2D (samples x channels)
                if len(eeg_data.shape) == 1:
                    eeg_data = eeg_data.reshape(-1, 1)
                elif len(eeg_data.shape) > 2:
                    eeg_data = eeg_data.reshape(eeg_data.shape[0], -1)
                
                # Apply unified preprocessing (removes EOG, applies all steps)
                preprocessed_eeg = self.preprocessor.preprocess(
                    eeg_data, fsample, channel_labels
                )
                
                # Get attention label
                attention_label = None
                if attention_labels and trial_idx < len(attention_labels):
                    attention_label = attention_labels[trial_idx]
                elif events['values'] and trial_idx < len(events['values']):
                    attention_label = events['values'][trial_idx]
                
                processed_trial = {
                    'eeg_data': preprocessed_eeg,
                    'sample_rate': self.preprocessor.target_sampling_rate,  # After downsampling
                    'original_sample_rate': fsample,
                    'attention_label': attention_label,
                    'trial_idx': trial_idx,
                    'subject_id': subject_id,
                    'original_shape': eeg_data.shape,
                    'preprocessed_shape': preprocessed_eeg.shape
                }
                
                processed_trials.append(processed_trial)
                
            except Exception as e:
                logger.error(f"Error processing trial {trial_idx}: {e}")
                continue
        
        logger.info(f"Processed {len(processed_trials)}/{len(eeg_trials)} trials for subject {subject_id}")
        
        return {
            'subject_id': subject_id,
            'trials': processed_trials,
            'n_trials': len(processed_trials),
            'expinfo': expinfo
        }
    
    def save_cleaned_data(self, processed_data: Dict):
        """Save preprocessed EEG data as 'subXX_preprocessed.mat'."""
        subject_id = processed_data['subject_id']
        output_file = self.output_dir / f"sub{subject_id:02d}_preprocessed.mat"
        
        save_dict = {
            'subject_id': subject_id,
            'n_trials': processed_data['n_trials'],
            'trials': []
        }
        
        for trial in processed_data['trials']:
            trial_dict = {
                'eeg_data': trial['eeg_data'],
                'sample_rate': trial['sample_rate'],
                'original_sample_rate': trial['original_sample_rate'],
                'attention_label': trial['attention_label'],
                'trial_idx': trial['trial_idx']
            }
            save_dict['trials'].append(trial_dict)
        
        sio.savemat(str(output_file), save_dict)
        logger.info(f"Saved preprocessed data to {output_file}")
    
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
            f.write("Fuglsang Dataset Unified Preprocessing Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write("Preprocessing steps:\n")
            f.write("  1. Remove EOG channels\n")
            f.write("  2. Downsample from 512 Hz to 128 Hz\n")
            f.write("  3. Bandpass filter (1-40 Hz)\n")
            f.write("  4. Artifact removal (outlier detection + interpolation)\n")
            f.write("  5. MAD-based normalization\n")
            f.write("  6. Soft clipping\n\n")
            for subject_id, n_trials in all_results.items():
                f.write(f"S{subject_id}: {n_trials} trials processed\n")
            f.write(f"\nTotal subjects: {len(all_results)}\n")
            f.write(f"Total trials: {sum(all_results.values())}\n")
        
        logger.info(f"\nProcessing complete! Summary saved to {summary_file}")
        return all_results


def preprocess_unified(dataset_type: str, input_path: str, output_path: str) -> Dict:
    """
    Unified function to preprocess Das or Fuglsang datasets.
    
    Args:
        dataset_type: 'Das' or 'Fuglsang'
        input_path: Path to input data directory
        output_path: Path to output directory
        
    Returns:
        Dictionary with processing summary
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Unified Preprocessing: {dataset_type} Dataset")
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
        processor = DasDatasetPreprocessor(data_dir=input_path, output_dir=output_path)
        results = processor.process_all_subjects()
        
        summary['subjects_processed'] = results
        summary['total_trials'] = sum(results.values())
        summary['sampling_rates'] = {k: 128 for k in results.keys()}
        
    elif dataset_type.lower() == 'fuglsang':
        processor = FuglsangDatasetPreprocessor(
            eeg_base_path=input_path,
            output_dir=output_path
        )
        results = processor.process_all_subjects()
        
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
    
    parser = argparse.ArgumentParser(description='Unified preprocessing for Das and Fuglsang datasets (ignores EOG)')
    parser.add_argument('--das_dir', type=str, default='Data/Das/4004271',
                       help='Das dataset directory')
    parser.add_argument('--fuglsang_eeg_dir', type=str, default='Data/Fulsang/eeg',
                       help='Fuglsang EEG directory')
    parser.add_argument('--dataset', type=str, choices=['das', 'fuglsang', 'both'], default='both',
                       help='Which dataset to process')
    
    args = parser.parse_args()
    
    if args.dataset in ['das', 'both']:
        summary = preprocess_unified(
            'Das', args.das_dir, 'preprocessed_Das'
        )
    
    if args.dataset in ['fuglsang', 'both']:
        summary = preprocess_unified(
            'Fuglsang', args.fuglsang_eeg_dir, 'preprocessed_Fuglsang'
        )
    
    logger.info("\n" + "="*60)
    logger.info("Unified Preprocessing Complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()

