#!/usr/bin/env python3
"""
DAS preprocessing for CombinedDataset only.
- Same EEG pipeline as 16-subjects up to 128 Hz, then 128→64 direct (no 32 Hz).
- Envelope in CombinedDataset: 1-band at 64 Hz (matches Fulsang; no 4-band gammatone).
Use this output with CombinedDataset(das_preprocessing_type="COMBINED_DAS", das_data_dir="das_combined_preprocessed", target_sampling_rate=64).
"""

import os
import sys
import numpy as np
import scipy.io as sio
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

tf.compat.v1.enable_v2_behavior()

# Intermediate rate for filtering; then resample to target (64 Hz)
INTERMEDIATE_FS = 128
TARGET_FS = 64


class DasPreprocessorCombined:
    """
    DAS preprocessor for combined dataset: 128→64 Hz direct, no 32 Hz.
    Envelope is extracted in CombinedDataset as 1-band at 64 Hz (matches Fulsang).
    """

    def __init__(self, data_dir: str = "Data/Das/4004271", output_dir: str = "das_combined_preprocessed",
                 audio_dir: str = "Data/Das/4004271/stimuli/stimuli"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.audio_dir = Path(audio_dir) if audio_dir else None
        self.tfrecord_dir = self.output_dir / "tfrecords"
        self.tfrecord_dir.mkdir(exist_ok=True)
        self.target_sampling_rate = TARGET_FS  # 64 Hz for Fulsang alignment
        self.n_channels = 64
        print(f"DAS Combined Preprocessor (for CombinedDataset):")
        print(f"  Data: {self.data_dir}, Output: {self.output_dir}")
        print(f"  Target rate: {self.target_sampling_rate} Hz (128→64 direct, no 32 Hz)")
        print(f"  Envelope: 1-band at 64 Hz in CombinedDataset (matches Fulsang)")

    def load_matlab_data(self, subject_file: str) -> List:
        mat_file = self.data_dir / subject_file
        try:
            mat_data = sio.loadmat(str(mat_file), squeeze_me=True, struct_as_record=False)
            trials = mat_data['trials']
            if not isinstance(trials, np.ndarray):
                trials = [trials]
            return np.array(trials)
        except Exception as e:
            print(f"ERROR loading {mat_file}: {e}")
            return np.array([])

    def preprocess_eeg(self, eeg_data: np.ndarray, sample_rate: int, use_tanh: bool = False) -> np.ndarray:
        """Preprocess: resample to 128 Hz, bandpass, z-score, then 128→64 direct."""
        from scipy import signal
        # 1) Resample to 128 Hz
        if sample_rate != INTERMEDIATE_FS:
            g = np.gcd(sample_rate, INTERMEDIATE_FS)
            up = INTERMEDIATE_FS // g
            down = sample_rate // g
            eeg_data = signal.resample_poly(eeg_data, up, down, axis=0)
        # 2) Baseline correction
        eeg_data = eeg_data - np.mean(eeg_data, axis=0, keepdims=True)
        # 3) Bandpass at 128 Hz (0.5–40 Hz)
        nyquist_128 = INTERMEDIATE_FS / 2
        low = max(0.001, 0.5 / nyquist_128)
        high = min(0.99, 40.0 / nyquist_128)
        b, a = signal.butter(4, [low, high], btype='band')
        for ch in range(eeg_data.shape[1]):
            eeg_data[:, ch] = signal.filtfilt(b, a, eeg_data[:, ch])
        # 4) Z-score per channel
        mean_per_ch = np.mean(eeg_data, axis=0, keepdims=True)
        std_per_ch = np.std(eeg_data, axis=0, keepdims=True)
        std_per_ch = np.where(std_per_ch == 0, 1.0, std_per_ch)
        eeg_data = (eeg_data - mean_per_ch) / std_per_ch
        # 5) 128→64 direct (no 32 Hz)
        if INTERMEDIATE_FS != self.target_sampling_rate:
            eeg_data = signal.resample_poly(eeg_data, self.target_sampling_rate, INTERMEDIATE_FS, axis=0)
        if use_tanh:
            eeg_data = np.tanh(eeg_data * 0.5)
        return eeg_data.astype(np.float32)

    def create_tfrecord_data(self) -> int:
        subject_files = sorted(self.data_dir.glob("S*.mat"))
        if not subject_files:
            raise ValueError(f"No S*.mat in {self.data_dir}")
        total_trials = 0
        subject_stats = {}
        for subject_file in subject_files:
            subject_id = subject_file.stem
            trials = self.load_matlab_data(subject_file.name)
            if len(trials) == 0:
                continue
            subject_trials = 0
            subject_samples = 0
            for trial_idx, trial in enumerate(trials):
                try:
                    eeg_data = np.asarray(trial.RawData.EegData, dtype=np.float64)
                    if eeg_data.ndim != 2:
                        continue
                    # Avoid "truth value of array is ambiguous": scalarize MATLAB scalars/arrays
                    sr = np.asarray(trial.FileHeader.SampleRate).flatten()
                    sample_rate = int(sr[0]) if sr.size > 0 else INTERMEDIATE_FS
                    attended_ear = getattr(trial, 'attended_ear', None)
                    if attended_ear is None:
                        continue
                    ae = np.asarray(attended_ear).flatten()
                    if ae.size == 0:
                        continue
                    attended_ear = str(ae.flat[0]).strip().upper()
                    if attended_ear not in ('L', 'R', 'LEFT', 'RIGHT'):
                        continue
                    if attended_ear in ('LEFT',):
                        attended_ear = 'L'
                    elif attended_ear in ('RIGHT',):
                        attended_ear = 'R'
                    if int(eeg_data.shape[1]) != self.n_channels:
                        continue
                    # Stimuli / audio paths (scalarize to avoid "truth value of array is ambiguous")
                    # Do not use "x or y" when x can be a numpy array: bool(array) raises.
                    stimuli = getattr(trial, 'stimuli', None)
                    if stimuli is None and hasattr(trial, 'RawData'):
                        stimuli = getattr(trial.RawData, 'stimuli', None)
                    stim_left = stim_right = None
                    if stimuli is not None and (not isinstance(stimuli, np.ndarray) or np.asarray(stimuli).size > 0):
                        sl = np.asarray(stimuli).flatten()
                        if sl.size >= 2:
                            stim_left = str(sl.flat[0]).strip()
                            stim_right = str(sl.flat[1]).strip()
                        elif sl.size == 1:
                            stim_left = str(sl.flat[0]).strip()
                    left_audio_file = right_audio_file = None
                    def resolve_audio(stim):
                        if stim is None or (isinstance(stim, (np.ndarray, np.generic)) and np.asarray(stim).size == 0):
                            return None
                        if not self.audio_dir or not self.audio_dir.exists():
                            return None
                        stem = str(np.asarray(stim).flat[0]).strip() if isinstance(stim, (np.ndarray, np.generic)) else str(stim).strip()
                        if not stem:
                            return None
                        p = self.audio_dir / stem
                        if p.exists():
                            return str(p)
                        for ext in ['.wav', '.WAV']:
                            p = self.audio_dir / (stem + ext)
                            if p.exists():
                                return str(p)
                        return None
                    left_audio_file = resolve_audio(stim_left)
                    right_audio_file = resolve_audio(stim_right)
                    eeg_data = self.preprocess_eeg(eeg_data, sample_rate, use_tanh=False)
                    tfrecord_file = self.tfrecord_dir / f"{subject_id}_trial_{trial_idx:03d}.tfrecords"
                    with tf.io.TFRecordWriter(str(tfrecord_file)) as writer:
                        for i in range(len(eeg_data)):
                            features = {
                                'eeg': tf.train.Feature(float_list=tf.train.FloatList(value=eeg_data[i].flatten())),
                                'attended_ear': tf.train.Feature(bytes_list=tf.train.BytesList(value=[attended_ear.encode('utf-8')])),
                                'subject_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[subject_id.encode('utf-8')])),
                                'trial_index': tf.train.Feature(int64_list=tf.train.Int64List(value=[trial_idx])),
                                'sample_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
                                'file_source': tf.train.Feature(bytes_list=tf.train.BytesList(value=[subject_file.name.encode('utf-8')])),
                            }
                            if hasattr(trial, 'TrialID'):
                                try:
                                    features['trial_id'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(trial.TrialID)]))
                                except (TypeError, ValueError):
                                    pass
                            if stim_left:
                                features['stim_left'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[stim_left.encode('utf-8')]))
                            if stim_right:
                                features['stim_right'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[stim_right.encode('utf-8')]))
                            if left_audio_file:
                                features['left_audio_file'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[left_audio_file.encode('utf-8')]))
                            if right_audio_file:
                                features['right_audio_file'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[right_audio_file.encode('utf-8')]))
                            example = tf.train.Example(features=tf.train.Features(feature=features))
                            writer.write(example.SerializeToString())
                    subject_trials += 1
                    subject_samples += len(eeg_data)
                    total_trials += 1
                except Exception as e:
                    print(f"  Skip {subject_id} trial {trial_idx}: {e}")
                    continue
            subject_stats[subject_id] = {'trials': subject_trials, 'samples': subject_samples}
        summary = {
            'total_subjects': len(subject_stats),
            'total_trials': total_trials,
            'subject_stats': subject_stats,
            'preprocessing_info': {
                'target_sampling_rate_hz': self.target_sampling_rate,
                'intermediate_fs_hz': INTERMEDIATE_FS,
                'n_channels': self.n_channels,
                'preprocessing_method': 'DAS_combined_for_CombinedDataset',
                'note': '128→64 direct, no 32 Hz. Envelope: 1-band at 64 Hz in CombinedDataset (matches Fulsang).',
            }
        }
        with open(self.output_dir / 'preprocessing_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Done. Total trials: {total_trials}, output: {self.tfrecord_dir}")
        return total_trials


def main():
    import argparse
    p = argparse.ArgumentParser(description='DAS preprocessing for CombinedDataset (128→64 Hz, 1-band envelope at 64 Hz)')
    p.add_argument('--data_dir', default='Data/Das/4004271')
    p.add_argument('--output_dir', default='das_combined_preprocessed')
    p.add_argument('--audio_dir', default='Data/Das/4004271/stimuli/stimuli')
    args = p.parse_args()
    preprocessor = DasPreprocessorCombined(data_dir=args.data_dir, output_dir=args.output_dir, audio_dir=args.audio_dir)
    preprocessor.create_tfrecord_data()


if __name__ == "__main__":
    main()
