#!/usr/bin/env python3
"""
DAS Preprocessing with 16 Subjects Support
Creates TFRecord files with proper subject information for DASCCA
"""

import os
import sys
import numpy as np
import scipy.io as sio
import tensorflow as tf
from pathlib import Path
import tempfile
import shutil
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# Add telluride_decoding to path
sys.path.append('telluride_decoding')

try:
    from telluride_decoding import decoding
    from telluride_decoding import brain_data
    from telluride_decoding import regression
    from telluride_decoding import ingest
    from telluride_decoding import attention_decoder
except ImportError as e:
    print(f"Warning: Could not import some telluride_decoding modules: {e}")
    print("Continuing with basic functionality...")

tf.compat.v1.enable_v2_behavior()


class DasPreprocessor16Subjects:
    """
    DAS Preprocessor that handles all 16 subjects with proper subject information.
    """
    
    def __init__(self, data_dir: str = "Data/Das/4004271", output_dir: str = "das_16subjects_preprocessed",
                 audio_dir: str = "Data/Das/4004271/stimuli/stimuli"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Audio directory for envelope extraction
        self.audio_dir = Path(audio_dir) if audio_dir else None
        
        # Create TFRecord directory
        self.tfrecord_dir = self.output_dir / "tfrecords"
        self.tfrecord_dir.mkdir(exist_ok=True)
        
        # DAS-specific parameters
        self.target_sampling_rate = 128  # Hz (matches KULeuven dataset standard)
        self.n_channels = 64  # EEG channels
        
        print(f"DAS 16-Subjects Preprocessor initialized:")
        print(f"  Data directory: {self.data_dir}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Audio directory: {self.audio_dir}")
        print(f"  TFRecord directory: {self.tfrecord_dir}")
        print(f"  Target sampling rate: {self.target_sampling_rate} Hz")
        print(f"  EEG channels: {self.n_channels}")
    
    def load_matlab_data(self, subject_file: str) -> List:
        """Load MATLAB data from subject file."""
        mat_file = self.data_dir / subject_file
        print(f"Loading {mat_file}")
        
        try:
            mat_data = sio.loadmat(str(mat_file), squeeze_me=True, struct_as_record=False)
            trials = mat_data['trials']
            
            if not isinstance(trials, np.ndarray):
                trials = [trials]
            
            # Convert to numpy array for consistent handling
            trials = np.array(trials)
            
            print(f"  Loaded {len(trials)} trials from {subject_file}")
            return trials
            
        except Exception as e:
            print(f"ERROR loading {mat_file}: {e}")
            return np.array([])
    
    def preprocess_eeg(self, eeg_data: np.ndarray, sample_rate: int, use_tanh: bool = False) -> np.ndarray:
        """Preprocess EEG data with downsampling and filtering.
        
        Args:
            eeg_data: EEG data (n_samples, n_channels)
            sample_rate: Original sampling rate in Hz
            use_tanh: Whether to apply tanh clipping (default: False, can distort signals)
        
        Returns:
            Preprocessed EEG data at target_sampling_rate
        """
        from scipy import signal
        
        # 1. Resample to target sampling rate using resample_poly (better than decimate)
        if sample_rate != self.target_sampling_rate:
            orig_fs = int(sample_rate)
            target_fs = self.target_sampling_rate
            
            # resample_poly: up/down are integers, stable and good anti-aliasing
            g = np.gcd(orig_fs, target_fs)
            up = target_fs // g
            down = orig_fs // g
            eeg_data = signal.resample_poly(eeg_data, up, down, axis=0)
            print(f"    Resampled from {sample_rate} Hz to {self.target_sampling_rate} Hz (up={up}, down={down})")
        
        # 2. Baseline correction (DC removal)
        eeg_data = eeg_data - np.mean(eeg_data, axis=0, keepdims=True)
        
        # 3. Bandpass filtering (0.5-40 Hz to match dataset's high-pass at 0.5 Hz)
        nyquist = self.target_sampling_rate / 2
        low_freq = 0.5 / nyquist  # Match dataset's 0.5 Hz high-pass
        high_freq = min(40.0 / nyquist, 0.99)
        
        b, a = signal.butter(4, [low_freq, high_freq], btype='band')
        
        filtered_eeg = np.zeros_like(eeg_data)
        for ch in range(eeg_data.shape[1]):
            filtered_eeg[:, ch] = signal.filtfilt(b, a, eeg_data[:, ch])
        
        # 4. Z-score normalization per channel (simpler than MAD)
        # Normalize per channel across the trial
        mean_per_ch = np.mean(filtered_eeg, axis=0, keepdims=True)
        std_per_ch = np.std(filtered_eeg, axis=0, keepdims=True)
        std_per_ch = np.where(std_per_ch == 0, 1.0, std_per_ch)  # Avoid division by zero
        filtered_eeg = (filtered_eeg - mean_per_ch) / std_per_ch
        
        # 5. Optional soft clipping (disabled by default to avoid signal distortion)
        if use_tanh:
            filtered_eeg = np.tanh(filtered_eeg * 0.5)
        
        return filtered_eeg.astype(np.float32)
    
    def create_tfrecord_data(self) -> int:
        """Create TFRecord files from all 16 DAS subjects."""
        print("Creating TFRecord files for all 16 DAS subjects...")
        
        # Find all subject files
        subject_files = list(self.data_dir.glob("S*.mat"))
        subject_files.sort()
        
        if not subject_files:
            raise ValueError(f"No subject files (S*.mat) found in {self.data_dir}")
        
        print(f"Found {len(subject_files)} subject files:")
        for subject_file in subject_files:
            print(f"  - {subject_file.name}")
        
        total_trials = 0
        subject_stats = {}
        observed_sample_rates = set()  # Track actual sample rates from files
        
        for subject_file in subject_files:
            subject_id = subject_file.stem  # e.g., "S1", "S2", etc.
            print(f"\nProcessing {subject_id}...")
            
            try:
                trials = self.load_matlab_data(subject_file.name)
                
                if len(trials) == 0:
                    print(f"  No trials found for {subject_id}")
                    continue
                
                subject_trials = 0
                subject_samples = 0
                
                for trial_idx, trial in enumerate(trials):
                    try:
                        # Debug: Print trial structure for first trial
                        if trial_idx == 0:
                            print(f"    Debug: Trial structure for {subject_id}:")
                            print(f"      Trial type: {type(trial)}")
                            print(f"      Trial attributes: {dir(trial)}")
                            if hasattr(trial, 'RawData'):
                                print(f"      RawData attributes: {dir(trial.RawData)}")
                            if hasattr(trial, 'FileHeader'):
                                print(f"      FileHeader attributes: {dir(trial.FileHeader)}")
                        
                        # Extract trial data with error handling
                        try:
                            eeg_data = trial.RawData.EegData
                        except AttributeError:
                            print(f"    Trial {trial_idx}: No RawData.EegData found")
                            continue
                        
                        try:
                            sample_rate = int(trial.FileHeader.SampleRate)
                            observed_sample_rates.add(sample_rate)
                        except AttributeError:
                            print(f"    Trial {trial_idx}: No FileHeader.SampleRate found")
                            continue
                        
                        # Extract TrialID from dataset (1-20) vs trial_index (0-based array index)
                        trial_index = trial_idx  # 0-based index in array
                        dataset_trial_id = None
                        if hasattr(trial, 'TrialID'):
                            try:
                                dataset_trial_id = int(trial.TrialID)
                            except (ValueError, TypeError):
                                pass
                        elif hasattr(trial, 'trialID'):
                            try:
                                dataset_trial_id = int(trial.trialID)
                            except (ValueError, TypeError):
                                pass
                        
                        try:
                            attended_ear = trial.attended_ear
                        except AttributeError:
                            print(f"    Trial {trial_idx}: No attended_ear found")
                            continue
                        
                        # Extract additional metadata fields
                        condition = None
                        experiment = None
                        part = None
                        attended_track = None
                        repetition = None
                        stim_left = None
                        stim_right = None
                        
                        # Try to extract condition (HRTF vs dry)
                        if hasattr(trial, 'condition'):
                            condition = str(trial.condition)
                        elif hasattr(trial, 'RawData') and hasattr(trial.RawData, 'condition'):
                            condition = str(trial.RawData.condition)
                        
                        # Try to extract experiment/part
                        if hasattr(trial, 'experiment'):
                            experiment = str(trial.experiment)
                        if hasattr(trial, 'part'):
                            part = str(trial.part)
                        
                        # Try to extract attended_track
                        if hasattr(trial, 'attended_track'):
                            attended_track = str(trial.attended_track)
                        elif hasattr(trial, 'attendedTrack'):
                            attended_track = str(trial.attendedTrack)
                        
                        # Try to extract repetition
                        if hasattr(trial, 'repetition'):
                            repetition = int(trial.repetition) if isinstance(trial.repetition, (int, np.integer)) else str(trial.repetition)
                        elif hasattr(trial, 'rep'):
                            repetition = int(trial.rep) if isinstance(trial.rep, (int, np.integer)) else str(trial.rep)
                        
                        # Handle attended_ear if it's an array
                        if hasattr(attended_ear, '__len__') and len(attended_ear) > 1:
                            attended_ear = attended_ear[0]  # Take first element
                        elif hasattr(attended_ear, '__len__') and len(attended_ear) == 1:
                            attended_ear = attended_ear[0]  # Take single element
                        # Convert to string if needed
                        attended_ear = str(attended_ear)
                        
                        # Validate data
                        if eeg_data is None or len(eeg_data) == 0:
                            print(f"    Trial {trial_idx}: No EEG data")
                            continue
                        
                        if eeg_data.shape[1] != self.n_channels:
                            print(f"    Trial {trial_idx}: Expected {self.n_channels} channels, got {eeg_data.shape[1]}")
                            continue
                        
                        # Validate attended_ear
                        if attended_ear not in ['L', 'R', 'Left', 'Right', 'left', 'right']:
                            print(f"    Trial {trial_idx}: Invalid attended_ear '{attended_ear}', skipping")
                            continue
                        
                        # Normalize attended_ear to L/R
                        if attended_ear in ['Left', 'left']:
                            attended_ear = 'L'
                        elif attended_ear in ['Right', 'right']:
                            attended_ear = 'R'
                        
                        # Extract stimuli information and map to audio files
                        left_audio_file = None
                        right_audio_file = None
                        stimuli = None
                        
                        # Try to get stimuli from trial
                        if hasattr(trial, 'stimuli'):
                            stimuli = trial.stimuli
                        elif hasattr(trial, 'RawData') and hasattr(trial.RawData, 'stimuli'):
                            stimuli = trial.RawData.stimuli
                        
                        # Extract individual stimulus identifiers
                        if stimuli is not None:
                            if isinstance(stimuli, np.ndarray):
                                stimuli_list = stimuli.flatten().tolist()
                            elif isinstance(stimuli, (list, tuple)):
                                stimuli_list = list(stimuli)
                            else:
                                stimuli_list = [stimuli]
                            
                            if len(stimuli_list) >= 2:
                                stim_left = str(stimuli_list[0])
                                stim_right = str(stimuli_list[1])
                            elif len(stimuli_list) == 1:
                                stim_left = str(stimuli_list[0])
                                stim_right = None
                            else:
                                stim_left = None
                                stim_right = None
                        else:
                            stim_left = None
                            stim_right = None
                        
                        if stimuli is not None and self.audio_dir and self.audio_dir.exists():
                            # Use already extracted stim_left/stim_right
                            left_stim = stim_left
                            right_stim = stim_right
                            
                            if left_stim or right_stim:
                                
                                # Find audio files
                                if left_stim:
                                    # Check if stimulus already has extension
                                    left_stim_clean = str(left_stim).strip()
                                    # Try direct match first (stimulus may already have extension)
                                    audio_file = self.audio_dir / left_stim_clean
                                    if audio_file.exists():
                                        left_audio_file = str(audio_file)
                                    else:
                                        # Try adding extensions if not found
                                        for ext in ['.wav', '.WAV', '.mp3', '.MP3']:
                                            audio_file = self.audio_dir / f"{left_stim_clean}{ext}"
                                            if audio_file.exists():
                                                left_audio_file = str(audio_file)
                                                break
                                        # Try pattern matching if direct match failed
                                        if not left_audio_file:
                                            for f in self.audio_dir.glob(f"*{left_stim_clean}*"):
                                                if f.suffix.lower() in ['.wav', '.mp3']:
                                                    left_audio_file = str(f)
                                                    break
                                
                                if right_stim:
                                    # Check if stimulus already has extension
                                    right_stim_clean = str(right_stim).strip()
                                    # Try direct match first (stimulus may already have extension)
                                    audio_file = self.audio_dir / right_stim_clean
                                    if audio_file.exists():
                                        right_audio_file = str(audio_file)
                                    else:
                                        # Try adding extensions if not found
                                        for ext in ['.wav', '.WAV', '.mp3', '.MP3']:
                                            audio_file = self.audio_dir / f"{right_stim_clean}{ext}"
                                            if audio_file.exists():
                                                right_audio_file = str(audio_file)
                                                break
                                        # Try pattern matching if direct match failed
                                        if not right_audio_file:
                                            for f in self.audio_dir.glob(f"*{right_stim_clean}*"):
                                                if f.suffix.lower() in ['.wav', '.mp3']:
                                                    right_audio_file = str(f)
                                                    break
                                
                                # Log mapping for first few trials
                                if trial_idx < 3:
                                    print(f"    Trial {trial_idx} audio mapping:")
                                    print(f"      Left stimulus: {left_stim} -> {left_audio_file if left_audio_file else 'NOT FOUND'}")
                                    print(f"      Right stimulus: {right_stim} -> {right_audio_file if right_audio_file else 'NOT FOUND'}")
                        
                        # Preprocess EEG (without tanh clipping by default)
                        eeg_data = self.preprocess_eeg(eeg_data, sample_rate, use_tanh=False)
                        
                        # Create TFRecord file for this trial
                        tfrecord_file = self.tfrecord_dir / f"{subject_id}_trial_{trial_idx:03d}.tfrecords"
                        
                        # Validate sample ordering (should start at 0 and be strictly increasing)
                        expected_samples = list(range(len(eeg_data)))
                        
                        with tf.io.TFRecordWriter(str(tfrecord_file)) as writer:
                            for i in range(len(eeg_data)):
                                # Validate sample_id
                                if i != expected_samples[i]:
                                    print(f"    WARNING: Sample ordering issue in trial {trial_idx}, sample {i}")
                                
                                # Create example with proper subject information
                                features = {
                                    'eeg': tf.train.Feature(float_list=tf.train.FloatList(value=eeg_data[i].flatten())),
                                    'attended_ear': tf.train.Feature(bytes_list=tf.train.BytesList(value=[attended_ear.encode('utf-8')])),
                                    'subject_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[subject_id.encode('utf-8')])),
                                    'trial_index': tf.train.Feature(int64_list=tf.train.Int64List(value=[trial_index])),  # 0-based array index
                                    'sample_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
                                    'file_source': tf.train.Feature(bytes_list=tf.train.BytesList(value=[subject_file.name.encode('utf-8')]))
                                }
                                
                                # Add dataset TrialID if available (1-20)
                                if dataset_trial_id is not None:
                                    features['trial_id'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[dataset_trial_id]))
                                
                                # Add additional metadata fields if available
                                if condition:
                                    features['condition'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[condition.encode('utf-8')]))
                                if experiment:
                                    features['experiment'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[experiment.encode('utf-8')]))
                                if part:
                                    features['part'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[part.encode('utf-8')]))
                                if attended_track:
                                    features['attended_track'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[attended_track.encode('utf-8')]))
                                if repetition is not None:
                                    if isinstance(repetition, int):
                                        features['repetition'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[repetition]))
                                    else:
                                        features['repetition'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(repetition).encode('utf-8')]))
                                if stim_left:
                                    features['stim_left'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[stim_left.encode('utf-8')]))
                                if stim_right:
                                    features['stim_right'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[stim_right.encode('utf-8')]))
                                
                                # Add audio file paths if available
                                if left_audio_file:
                                    features['left_audio_file'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[left_audio_file.encode('utf-8')]))
                                if right_audio_file:
                                    features['right_audio_file'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[right_audio_file.encode('utf-8')]))
                                
                                example = tf.train.Example(features=tf.train.Features(feature=features))
                                writer.write(example.SerializeToString())
                        
                        subject_trials += 1
                        subject_samples += len(eeg_data)
                        total_trials += 1
                        
                        print(f"    Trial {trial_idx}: {len(eeg_data)} samples, attended_ear={attended_ear}")
                        
                    except Exception as e:
                        print(f"    ERROR processing trial {trial_idx} for {subject_id}: {e}")
                        print(f"    Error type: {type(e).__name__}")
                        import traceback
                        print(f"    Traceback: {traceback.format_exc()}")
                        continue
                
                subject_stats[subject_id] = {
                    'trials': subject_trials,
                    'samples': subject_samples
                }
                
                print(f"  {subject_id}: {subject_trials} trials, {subject_samples} samples")
                
            except Exception as e:
                print(f"ERROR processing {subject_id}: {e}")
                print(f"Error type: {type(e).__name__}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                continue
        
        # Save preprocessing summary
        summary = {
            'total_subjects': len(subject_stats),
            'total_trials': total_trials,
            'subject_stats': subject_stats,
            'preprocessing_info': {
                'observed_sample_rates_hz': sorted(list(observed_sample_rates)),  # Actual rates from files
                'target_sampling_rate_hz': self.target_sampling_rate,
                'n_channels': self.n_channels,
                'preprocessing_method': 'DAS_16subjects_preprocessing',
                'note': 'Original sample rates extracted from FileHeader.SampleRate per trial'
            }
        }
        
        with open(self.output_dir / 'preprocessing_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Preprocessing completed!")
        print(f"  Total subjects: {len(subject_stats)}")
        print(f"  Total trials: {total_trials}")
        print(f"  TFRecord files: {len(list(self.tfrecord_dir.glob('*.tfrecords')))}")
        print(f"  Summary saved to: {self.output_dir / 'preprocessing_summary.json'}")
        
        return total_trials
    
    def create_train_test_split(self, train_ratio: float = 0.7, val_ratio: float = 0.15):
        """Create train/validation/test split directories."""
        print(f"\nCreating train/validation/test split...")
        
        # Create split directories
        train_dir = self.tfrecord_dir / "train"
        val_dir = self.tfrecord_dir / "val"
        test_dir = self.tfrecord_dir / "test"
        
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)
        
        # Get all TFRecord files
        tfrecord_files = list(self.tfrecord_dir.glob("S*_trial_*.tfrecords"))
        tfrecord_files.sort()
        
        print(f"Found {len(tfrecord_files)} TFRecord files")
        
        # Group by subject
        subject_files = {}
        for file in tfrecord_files:
            subject_id = file.name.split('_')[0]  # Extract S1, S2, etc.
            if subject_id not in subject_files:
                subject_files[subject_id] = []
            subject_files[subject_id].append(file)
        
        print(f"Found {len(subject_files)} subjects:")
        for subject_id, files in subject_files.items():
            print(f"  {subject_id}: {len(files)} trials")
        
        # Subject-wise splitting with shuffled order (fixed seed for reproducibility)
        subjects = sorted(subject_files.keys())  # Sort first for consistency
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        rng.shuffle(subjects)  # Shuffle with fixed seed
        
        n_subjects = len(subjects)
        n_train_subjects = int(train_ratio * n_subjects)
        n_val_subjects = int(val_ratio * n_subjects)
        
        train_subjects = subjects[:n_train_subjects]
        val_subjects = subjects[n_train_subjects:n_train_subjects + n_val_subjects]
        test_subjects = subjects[n_train_subjects + n_val_subjects:]
        
        print(f"\nSubject-wise split:")
        print(f"  Train subjects ({len(train_subjects)}): {train_subjects}")
        print(f"  Val subjects ({len(val_subjects)}): {val_subjects}")
        print(f"  Test subjects ({len(test_subjects)}): {test_subjects}")
        
        # Copy files to appropriate directories (preserve originals)
        def copy_files(subject_list, target_dir):
            copied_count = 0
            for subject_id in subject_list:
                for file in subject_files[subject_id]:
                    target_file = target_dir / file.name
                    shutil.copy2(str(file), str(target_file))  # copy2 preserves metadata
                    copied_count += 1
            return copied_count
        
        train_count = copy_files(train_subjects, train_dir)
        val_count = copy_files(val_subjects, val_dir)
        test_count = copy_files(test_subjects, test_dir)
        
        print(f"\nFiles copied:")
        print(f"  Train: {train_count} files")
        print(f"  Val: {val_count} files")
        print(f"  Test: {test_count} files")
        print(f"  Note: Original files preserved in {self.tfrecord_dir}")
        
        # Save split information
        split_info = {
            'train_subjects': train_subjects,
            'val_subjects': val_subjects,
            'test_subjects': test_subjects,
            'train_files': train_count,
            'val_files': val_count,
            'test_files': test_count,
            'split_method': 'subject_wise'
        }
        
        with open(self.output_dir / 'split_info.json', 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"✓ Split information saved to: {self.output_dir / 'split_info.json'}")


def main():
    """Main function to run DAS preprocessing with 16 subjects."""
    import argparse
    
    parser = argparse.ArgumentParser(description='DAS Preprocessing with 16 Subjects Support')
    parser.add_argument('--data_dir', type=str, default='Data/Das/4004271',
                       help='DAS data directory path')
    parser.add_argument('--output_dir', type=str, default='das_16subjects_preprocessed',
                       help='Output directory for preprocessed data')
    parser.add_argument('--audio_dir', type=str, default='Data/Das/4004271/stimuli/stimuli',
                       help='Directory containing Das audio files for envelope extraction')
    parser.add_argument('--create_split', action='store_true',
                       help='Create train/val/test split after preprocessing')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("DAS PREPROCESSING WITH 16 SUBJECTS SUPPORT")
    print("=" * 80)
    print("Features:")
    print("- Processes all 16 DAS subjects (S1-S16)")
    print("- Includes proper subject_id in TFRecord files")
    print("- Resampling to 128 Hz (matches KULeuven standard)")
    print("- Bandpass filtering (0.5-40 Hz, matches dataset's 0.5 Hz high-pass)")
    print("- Z-score normalization per channel")
    print("- Subject-wise train/val/test splitting (shuffled with fixed seed)")
    print("- Comprehensive metadata extraction (condition, experiment, stimuli, etc.)")
    print("- Stores both trial_index (0-based) and trial_id (dataset TrialID 1-20)")
    print("- Comprehensive preprocessing reports")
    print("=" * 80)
    
    # Create preprocessor
    preprocessor = DasPreprocessor16Subjects(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        audio_dir=args.audio_dir
    )
    
    # Run preprocessing
    print("\nStarting DAS preprocessing...")
    total_trials = preprocessor.create_tfrecord_data()
    
    if args.create_split:
        print("\nCreating train/validation/test split...")
        preprocessor.create_train_test_split()
    
    print("\n" + "=" * 80)
    print("DAS PREPROCESSING COMPLETE!")
    print("=" * 80)
    print(f"Total trials processed: {total_trials}")
    print(f"Output directory: {args.output_dir}")
    print(f"TFRecord directory: {args.output_dir}/tfrecords")
    print("=" * 80)


if __name__ == "__main__":
    main()
