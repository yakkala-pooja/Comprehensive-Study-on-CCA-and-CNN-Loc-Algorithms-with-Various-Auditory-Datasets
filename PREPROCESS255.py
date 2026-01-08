#!/usr/bin/env python3

import os
import sys
import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.append('.')
try:
    from mwf_artifact_removal import MultiChannelWienerFilter
except ImportError as e:
    print(f"Warning: Could not import MWF modules: {e}")

from scipy import signal
from scipy.io import wavfile

DAS_CHANNEL_ORDER = [
    'Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz', 'FC4', 'FT8',
    'T7', 'C3', 'Cz', 'C4', 'T8', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P7', 'P3', 'Pz', 'P4',
    'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'AF7', 'AF3', 'AFz', 'AF4',
    'AF8', 'F5', 'F1', 'F2', 'F6', 'FC5', 'FC1', 'FC2', 'FC6', 'C5', 'C1', 'C2', 'C6',
    'CP5', 'CP1', 'CP2', 'CP6', 'P5', 'P1', 'P2', 'P6', 'PO5', 'PO1', 'PO2', 'PO6', 'I1', 'Iz', 'I2'
]

CHANNEL_NAME_TO_SYNAMPS = {
    'Fp1': 249, 'Fpz': 251, 'Fp2': 253, 'F7': 208, 'F3': 213, 'Fz': 216, 'F4': 220, 'F8': 224,
    'FT7': 168, 'FC3': 171, 'FCz': None, 'FC4': 178, 'FT8': 182, 'T7': 131, 'C3': 135, 'Cz': 139,
    'C4': 143, 'T8': 147, 'TP7': 95, 'CP3': 99, 'CPz': 103, 'CP4': 107, 'TP8': 111, 'P7': 53,
    'P3': 57, 'Pz': 61, 'P4': 65, 'P8': 69, 'PO7': 29, 'PO3': 30, 'POz': 32, 'PO4': 34, 'PO8': 35,
    'O1': 9, 'Oz': 11, 'O2': 13, 'AF7': 236, 'AF3': 237, 'AFz': 238, 'AF4': 239, 'AF8': 240,
    'F5': 210, 'F1': 214, 'F2': 218, 'F6': 222, 'FC5': 169, 'FC1': 173, 'FC2': 176, 'FC6': 180,
    'C5': 133, 'C1': 137, 'C2': 141, 'C6': 145, 'CP5': 97, 'CP1': 101, 'CP2': 105, 'CP6': 109,
    'P5': 55, 'P1': 59, 'P2': 63, 'P6': 67, 'PO5': 28, 'PO1': 31, 'PO2': 33, 'PO6': 36,
    'I1': 0, 'Iz': 1, 'I2': 2
}

def get_das_channel_indices(channel_names: List[str], n_total_channels: int = 255) -> List[int]:
    indices = []
    for das_channel in DAS_CHANNEL_ORDER:
        if das_channel == 'FCz':
            continue
        synamps_num = CHANNEL_NAME_TO_SYNAMPS.get(das_channel)
        if synamps_num is None:
            continue
        
        found = False
        try:
            idx = channel_names.index(das_channel)
            indices.append(idx)
            found = True
        except ValueError:
            pass
        
        if not found:
            synamps_idx_0 = synamps_num - 1
            synamps_idx_1 = synamps_num
            if synamps_idx_0 >= 0 and synamps_idx_0 < n_total_channels:
                indices.append(synamps_idx_0)
                found = True
            elif synamps_idx_1 >= 0 and synamps_idx_1 < n_total_channels:
                indices.append(synamps_idx_1)
                found = True
        
        if not found:
            print(f"Warning: Channel {das_channel} (SynAmps {synamps_num}) could not be mapped")
    
    if len(indices) < 64:
        print(f"Warning: Only found {len(indices)} matching channels, expected 64")
        while len(indices) < 64:
            indices.append(0)
    
    return indices[:64]

def read_curry_dap_file(dap_file: Path) -> Dict:
    with open(dap_file, 'rt', encoding='latin-1', errors='ignore') as f:
        content = f.read()
    
    tokens = {
        'NumSamples': None, 'NUM_SAMPLES': None,
        'NumChannels': None, 'NUM_CHANNELS': None,
        'NumTrials': None, 'NUM_TRIALS': None,
        'SampleFreqHz': None, 'SAMPLE_FREQ_HZ': None,
        'TriggerOffsetUsec': None, 'TRIGGER_OFFSET_USEC': None,
        'DataFormat': None, 'DATA_FORMAT': None,
        'DataSampOrder': None, 'DATA_SAMP_ORDER': None
    }
    
    for token in tokens.keys():
        idx = content.find(token)
        if idx != -1:
            remaining = content[idx + len(token):]
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
    
    nSamples = tokens['NumSamples'] or tokens['NUM_SAMPLES'] or 0
    nChannels = tokens['NumChannels'] or tokens['NUM_CHANNELS'] or 0
    nTrials = tokens['NumTrials'] or tokens['NUM_TRIALS'] or 1
    fFrequency = tokens['SampleFreqHz'] or tokens['SAMPLE_FREQ_HZ'] or 1000
    nASCII = 1 if (tokens['DataFormat'] == 1 or tokens['DATA_FORMAT'] == 1) else 0
    nMultiplex = 1 if (tokens['DataSampOrder'] == 1 or tokens['DATA_SAMP_ORDER'] == 1) else 0
    
    return {
        'nSamples': int(nSamples),
        'nChannels': int(nChannels),
        'nTrials': int(nTrials),
        'fFrequency': float(fFrequency),
        'nASCII': nASCII,
        'nMultiplex': nMultiplex
    }

def read_curry_rs3_file(rs3_file: Path, nChannels: int) -> List[str]:
    try:
        with open(rs3_file, 'rt', encoding='latin-1', errors='ignore') as f:
            content = f.read()
        
        channel_names = [f'EEG{i+1}' for i in range(nChannels)]
        
        idx_positions = []
        idx = content.find('LABELS')
        while idx != -1:
            idx_positions.append(idx)
            idx = content.find('LABELS', idx + 1)
        
        if len(idx_positions) >= 4:
            for i in range(3, len(idx_positions), 4):
                start_idx = idx_positions[i-1]
                end_idx = idx_positions[i] if i < len(idx_positions) else len(content)
                section = content[start_idx:end_idx]
                
                lines = section.split('\n')
                nc = 0
                for line in lines[1:]:
                    line = line.strip()
                    if line and line != 'END_LIST' and nc < nChannels:
                        channel_names[nc] = line
                        nc += 1
                    if line == 'END_LIST' or nc >= nChannels:
                        break
        
        return channel_names
    except Exception as e:
        return [f'EEG{i+1}' for i in range(nChannels)]

def read_curry_dat_file(dat_file: Path, params: Dict) -> np.ndarray:
    nChannels = params['nChannels']
    nSamples = params['nSamples']
    nTrials = params['nTrials']
    nASCII = params['nASCII']
    nMultiplex = params['nMultiplex']
    
    if nASCII == 1:
        with open(dat_file, 'rt') as f:
            data = np.loadtxt(f)
        data = data.reshape(nChannels, nSamples * nTrials)
    else:
        with open(dat_file, 'rb') as f:
            data = np.fromfile(f, dtype=np.float32, count=nChannels * nSamples * nTrials)
        data = data.reshape(nChannels, nSamples * nTrials)
    
    if nMultiplex == 1:
        data = data.T
    
    return data

def load_curry_file(base_name: Path) -> Dict:
    dat_file = base_name.with_suffix('.dat')
    dap_file = base_name.with_suffix('.dap')
    rs3_file = base_name.with_suffix('.rs3')
    
    if not dat_file.exists() or not dap_file.exists() or not rs3_file.exists():
        raise FileNotFoundError(f"Missing CURRY files for {base_name}")
    
    params = read_curry_dap_file(dap_file)
    channel_names = read_curry_rs3_file(rs3_file, params['nChannels'])
    data = read_curry_dat_file(dat_file, params)
    
    if len(channel_names) != data.shape[0]:
        channel_names = [f'EEG{i+1}' for i in range(data.shape[0])]
    
    trigger_idx = None
    for i, name in enumerate(channel_names):
        if 'Trigger' in name or 'TRIGGER' in name.upper():
            trigger_idx = i
            break
    
    if trigger_idx is not None:
        trigger_data = data[trigger_idx, :].copy()
        data = np.delete(data, trigger_idx, axis=0)
        channel_names.pop(trigger_idx)
        params['nChannels'] -= 1
    else:
        trigger_data = None
    
    return {
        'eeg_data': data.T,
        'channel_names': channel_names,
        'sampling_rate': params['fFrequency'],
        'trigger': trigger_data,
        'params': params
    }

def extract_envelope_from_audio(audio_file: Path, target_length: int, target_fs: int = 128) -> Optional[np.ndarray]:
    try:
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
        
        envelope = np.abs(audio_data)
        
        if len(envelope) > 9:
            kernel = np.ones(9) / 9.0
            envelope = np.convolve(envelope, kernel, mode='same')
        
        if len(envelope) != target_length:
            src_idx = np.linspace(0.0, 1.0, num=len(envelope))
            dst_idx = np.linspace(0.0, 1.0, num=target_length)
            envelope = np.interp(dst_idx, src_idx, envelope)
        
        return envelope.reshape(-1, 1).astype(np.float32)
    except Exception as e:
        return None

def preprocess_eeg_window(eeg_window: np.ndarray, sampling_rate: float) -> np.ndarray:
    artifact_thresh = 5.0
    for ch in range(eeg_window.shape[1]):
        ch_data = eeg_window[:, ch]
        std_val = np.std(ch_data)
        mean_val = np.mean(ch_data)
        
        artifacts = np.abs(ch_data - mean_val) > (artifact_thresh * std_val)
        
        if np.any(artifacts):
            valid_indices = ~artifacts
            if np.sum(valid_indices) > 2:
                from scipy.interpolate import interp1d
                valid_data = ch_data[valid_indices]
                valid_time = np.where(valid_indices)[0]
                all_time = np.arange(len(ch_data))
                
                f_interp = interp1d(valid_time, valid_data, kind='linear', 
                                  bounds_error=False, fill_value='extrapolate')
                eeg_window[:, ch] = f_interp(all_time)
    
    eeg_window = eeg_window - np.mean(eeg_window, axis=0, keepdims=True)
    
    nyquist = sampling_rate / 2
    low_freq = 1.0 / nyquist
    high_freq = min(40.0 / nyquist, 0.99)
    
    b, a = signal.butter(4, [low_freq, high_freq], btype='band')
    
    filtered_eeg = np.zeros_like(eeg_window)
    for ch in range(eeg_window.shape[1]):
        filtered_eeg[:, ch] = signal.filtfilt(b, a, eeg_window[:, ch])
    
    mad = np.median(np.abs(filtered_eeg - np.median(filtered_eeg, axis=0)), axis=0)
    mad = np.where(mad == 0, 1.0, mad)
    filtered_eeg = filtered_eeg / mad
    
    filtered_eeg = np.tanh(filtered_eeg * 0.5)
    
    if np.any(np.isnan(filtered_eeg)) or np.any(np.isinf(filtered_eeg)):
        filtered_eeg = np.nan_to_num(filtered_eeg, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return filtered_eeg.astype(np.float32)

def downsample_eeg(eeg_data: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
    if original_fs == target_fs:
        return eeg_data
    
    num_samples = int(eeg_data.shape[0] * target_fs / original_fs)
    downsampled = np.zeros((num_samples, eeg_data.shape[1]), dtype=eeg_data.dtype)
    
    for ch in range(eeg_data.shape[1]):
        downsampled[:, ch] = signal.resample(eeg_data[:, ch], num_samples)
    
    return downsampled

def process_kuleuven_255_dataset(
    data_dir: str = "/home/py9363/telluride_decoding/Data/KULeuven 255",
    stimuli_dir: str = None,
    output_dir: str = "kuleuven_255_preprocessed",
    target_sampling_rate: int = 128,
    target_channels: int = 64,
    apply_mwf: bool = True
):
    target_channels = 64
    data_path = Path(data_dir)
    if stimuli_dir is None:
        stimuli_path = data_path / "stimuli" / "stimuli"
    else:
        stimuli_path = Path(stimuli_dir)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    stimulus_mapping = {
        '1L': ('part1_track1_dry', 'part1_track2_dry'),
        '1R': ('part1_track1_dry', 'part1_track2_dry'),
        '2L': ('part2_track2_dry', 'part2_track1_dry'),
        '2R': ('part2_track2_dry', 'part2_track1_dry')
    }
    
    all_subjects = []
    first_trial_processed = False
    for subject_idx, subject_dir in enumerate(sorted(data_path.glob("S*"))):
        if not subject_dir.is_dir():
            continue
        
        inner_dir = subject_dir / subject_dir.name
        if not inner_dir.exists():
            inner_dir = subject_dir
        
        subject_id = subject_dir.name
        print(f"\nProcessing subject {subject_id}...")
        
        subject_trials = []
        
        for trial_idx, trial_file in enumerate(sorted(inner_dir.glob("S*_AAD_*.dat"))):
            base_name = trial_file.with_suffix('')
            trial_code = base_name.name.split('_')[-1]
            
            if trial_code not in ['1L', '1R', '2L', '2R']:
                continue
            
            attended_ear = 'L' if trial_code.endswith('L') else 'R'
            label = 0 if attended_ear == 'L' else 1
            
            print(f"  Loading trial {trial_code} (attended: {attended_ear})...")
            
            try:
                curry_data = load_curry_file(base_name)
                eeg_data = curry_data['eeg_data']
                original_fs = curry_data['sampling_rate']
                
                print(f"    Original: {eeg_data.shape}, {original_fs} Hz, {curry_data['params']['nChannels']} channels")
                
                channel_names = curry_data.get('channel_names', [])
                n_total_channels = curry_data['params']['nChannels']
                
                if apply_mwf:
                    try:
                        mwf = MultiChannelWienerFilter()
                        eeg_data = mwf.filter(eeg_data)
                    except Exception as e:
                        print(f"    Warning: MWF failed, using raw data: {e}")
                
                das_channel_indices = get_das_channel_indices(channel_names, n_total_channels)
                
                if len(das_channel_indices) == 64 and max(das_channel_indices) < eeg_data.shape[1]:
                    eeg_data = eeg_data[:, das_channel_indices]
                    if not first_trial_processed:
                        print(f"    ✓ Successfully mapped {len(das_channel_indices)} Das-compatible channels")
                        print(f"    Channel indices range: {min(das_channel_indices)}-{max(das_channel_indices)}")
                        print(f"    Sample mapped channels: {[channel_names[i] if i < len(channel_names) else f'EEG{i+1}' for i in das_channel_indices[:5]]}")
                        first_trial_processed = True
                elif eeg_data.shape[1] > 64:
                    print(f"    ⚠ Warning: Could not map Das channels, using first 64 channels")
                    print(f"    Available channels: {len(channel_names)}, Expected: 64 Das channels")
                    print(f"    Found {len(das_channel_indices)} mapped channels")
                    eeg_data = eeg_data[:, :64]
                elif eeg_data.shape[1] < 64:
                    padding = np.zeros((eeg_data.shape[0], 64 - eeg_data.shape[1]), dtype=eeg_data.dtype)
                    eeg_data = np.hstack([eeg_data, padding])
                    print(f"    ⚠ Warning: Padded from {eeg_data.shape[1] - padding.shape[1]} to 64 channels")
                
                assert eeg_data.shape[1] == 64, f"Expected 64 channels to match Das dataset, got {eeg_data.shape[1]}"
                
                eeg_data = downsample_eeg(eeg_data, original_fs, target_sampling_rate)
                
                eeg_data = preprocess_eeg_window(eeg_data, target_sampling_rate)
                
                left_stim, right_stim = stimulus_mapping[trial_code]
                left_audio = stimuli_path / f"{left_stim}.wav"
                right_audio = stimuli_path / f"{right_stim}.wav"
                
                left_env = extract_envelope_from_audio(left_audio, eeg_data.shape[0], target_sampling_rate)
                right_env = extract_envelope_from_audio(right_audio, eeg_data.shape[0], target_sampling_rate)
                
                if left_env is None:
                    print(f"    ⚠ Warning: Could not extract left envelope from {left_audio.name}, using zeros")
                    left_env = np.zeros((eeg_data.shape[0], 1), dtype=np.float32)
                else:
                    if trial_idx == 0 and not first_trial_processed:
                        print(f"    ✓ Left envelope extracted: shape {left_env.shape}, non-zero samples: {np.count_nonzero(left_env)}/{len(left_env)}")
                
                if right_env is None:
                    print(f"    ⚠ Warning: Could not extract right envelope from {right_audio.name}, using zeros")
                    right_env = np.zeros((eeg_data.shape[0], 1), dtype=np.float32)
                else:
                    if trial_idx == 0 and not first_trial_processed:
                        print(f"    ✓ Right envelope extracted: shape {right_env.shape}, non-zero samples: {np.count_nonzero(right_env)}/{len(right_env)}")
                
                # Verify envelope length matches EEG length
                if left_env.shape[0] != eeg_data.shape[0] or right_env.shape[0] != eeg_data.shape[0]:
                    print(f"    ⚠ Warning: Envelope length mismatch! EEG: {eeg_data.shape[0]}, Left: {left_env.shape[0]}, Right: {right_env.shape[0]}")
                
                subject_trials.append({
                    'eeg_data': eeg_data,
                    'attended_ear': attended_ear,
                    'label': label,
                    'trial_code': trial_code,
                    'left_envelope': left_env,
                    'right_envelope': right_env,
                    'subject_id': subject_id
                })
                
                print(f"    Processed: {eeg_data.shape}, {target_sampling_rate} Hz, 64 channels (matching Das dataset)")
                print(f"    Envelopes: Left {left_env.shape}, Right {right_env.shape} (ready for CCA)")
                
            except Exception as e:
                print(f"    Error processing {trial_code}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if subject_trials:
            all_subjects.append({
                'subject_id': subject_id,
                'trials': subject_trials
            })
    
    print(f"\nSaving preprocessed data for {len(all_subjects)} subjects...")
    
    for subject_data in all_subjects:
        subject_id = subject_data['subject_id']
        output_file = output_path / f"{subject_id}_preprocessed.mat"
        
        trials_array = []
        for trial in subject_data['trials']:
            trial_struct = {
                'eeg_data': trial['eeg_data'],
                'attended_ear': trial['attended_ear'],
                'attention_label': trial['label'],
                'left_envelope': trial['left_envelope'],
                'right_envelope': trial['right_envelope'],
                'trial_code': trial['trial_code']
            }
            trials_array.append(trial_struct)
        
        save_dict = {
            'subject_id': subject_id,
            'trials': trials_array,
            'sampling_rate': target_sampling_rate,
            'n_channels': 64,
            'preprocessing': 'PREPROCESS255'
        }
        
        sio.savemat(str(output_file), save_dict)
        print(f"  Saved {output_file}")
    
    print(f"\n✓ Preprocessing complete!")
    print(f"  Subjects: {len(all_subjects)}")
    print(f"  Output directory: {output_path}")
    
    return output_path

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess KU Leuven 255-channel EEG dataset')
    parser.add_argument('--data_dir', type=str, 
                       default='/home/py9363/telluride_decoding/Data/KULeuven 255',
                       help='Directory containing KU Leuven 255 data')
    parser.add_argument('--stimuli_dir', type=str, default=None,
                       help='Directory containing WAV stimulus files')
    parser.add_argument('--output_dir', type=str, default='kuleuven_255_preprocessed',
                       help='Output directory for preprocessed data')
    parser.add_argument('--target_sampling_rate', type=int, default=128,
                       help='Target sampling rate (default: 128 Hz)')
    parser.add_argument('--target_channels', type=int, default=64,
                       help='Target number of channels (fixed at 64 to match Das dataset)')
    parser.add_argument('--no_mwf', action='store_true',
                       help='Skip MWF artifact removal')
    
    args = parser.parse_args()
    
    process_kuleuven_255_dataset(
        data_dir=args.data_dir,
        stimuli_dir=args.stimuli_dir,
        output_dir=args.output_dir,
        target_sampling_rate=args.target_sampling_rate,
        target_channels=args.target_channels,
        apply_mwf=not args.no_mwf
    )

