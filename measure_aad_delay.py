#!/usr/bin/env python3
"""
Measure AAD (Auditory Attention Decoding) delay using triggers.

The delay is the time lag between:
- Audio stimulus onset (marked by trigger)
- Neural response in EEG

This script:
1. Uses trigger to mark stimulus start
2. Computes cross-correlation between audio envelope and EEG
3. Finds the peak correlation lag (the delay)
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
from scipy.signal import correlate
import matplotlib.pyplot as plt

def measure_delay_from_crosscorrelation(audio_envelope, eeg_signal, fs, max_lag_ms=600):
    """
    Measure neural response delay using cross-correlation.
    
    Args:
        audio_envelope: Audio envelope signal (T,)
        eeg_signal: EEG signal (T, n_channels) - use mean or specific channel
        fs: Sampling rate in Hz
        max_lag_ms: Maximum lag to search (in milliseconds)
    
    Returns:
        delay_ms: Delay in milliseconds
        delay_samples: Delay in samples
        correlation: Peak correlation value
        all_corrs: All correlation values for plotting
        lags_ms: All lag values in milliseconds
    """
    # Use mean across channels for EEG (or can use specific channel)
    if eeg_signal.ndim > 1:
        eeg_mean = np.mean(eeg_signal, axis=1)
    else:
        eeg_mean = eeg_signal
    
    # Normalize signals
    audio_norm = (audio_envelope - np.mean(audio_envelope)) / (np.std(audio_envelope) + 1e-8)
    eeg_norm = (eeg_mean - np.mean(eeg_mean)) / (np.std(eeg_mean) + 1e-8)
    
    # Compute cross-correlation
    max_lag_samples = int(max_lag_ms * fs / 1000.0)
    correlation = correlate(eeg_norm, audio_norm, mode='full', method='auto')
    
    # Normalize correlation by signal lengths
    correlation = correlation / (len(eeg_norm) - np.abs(np.arange(len(correlation)) - len(correlation)//2))
    
    # Find center (zero lag)
    center = len(correlation) // 2
    
    # Extract lags around zero (positive lag = EEG lags audio, negative = audio lags EEG)
    # For neural response: we expect positive lag (EEG comes after audio)
    start_idx = max(0, center - max_lag_samples)
    end_idx = min(len(correlation), center + max_lag_samples + 1)
    correlation_window = correlation[start_idx:end_idx]
    lags_samples = np.arange(start_idx - center, end_idx - center)
    lags_ms = lags_samples * 1000.0 / fs
    
    # Find peak in positive lag region (150-400ms expected)
    positive_region = (lags_ms >= 0) & (lags_ms <= max_lag_ms)
    if np.any(positive_region):
        positive_corrs = correlation_window[positive_region]
        positive_lags_ms = lags_ms[positive_region]
        peak_idx = np.argmax(np.abs(positive_corrs))  # Use absolute value
        delay_ms = positive_lags_ms[peak_idx]
        delay_samples = int(delay_ms * fs / 1000.0)
        peak_correlation = positive_corrs[peak_idx]
    else:
        # Fallback: find overall peak
        peak_idx = np.argmax(np.abs(correlation_window))
        delay_ms = lags_ms[peak_idx]
        delay_samples = int(delay_ms * fs / 1000.0)
        peak_correlation = correlation_window[peak_idx]
    
    return delay_ms, delay_samples, peak_correlation, correlation_window, lags_ms

def analyze_fulsang_delay(subject_id='S1', trial_idx=0):
    """Analyze delay for Fulsang dataset."""
    print("="*80)
    print("MEASURING AAD DELAY - FULSANG DATASET")
    print("="*80)
    
    data_dir = Path("Data/Fulsang/DATA_preproc")
    mat_file = data_dir / f"{subject_id}_data_preproc.mat"
    
    if not mat_file.exists():
        print(f"File not found: {mat_file}")
        return None
    
    print(f"\nLoading {mat_file}...")
    mat_data = sio.loadmat(str(mat_file), squeeze_me=False, struct_as_record=False)
    
    if 'data' not in mat_data:
        print("No 'data' field found")
        return None
    
    data_struct = mat_data['data']
    if isinstance(data_struct, np.ndarray) and data_struct.size > 0:
        trial = data_struct.flat[trial_idx]
        
        # Extract EEG
        if hasattr(trial, 'eeg'):
            eeg_data = trial.eeg
            if isinstance(eeg_data, np.ndarray) and eeg_data.dtype == object:
                eeg_data = eeg_data.flat[0]
            if isinstance(eeg_data, np.ndarray):
                print(f"EEG shape: {eeg_data.shape}")
            else:
                print("EEG data not in expected format")
                return None
        else:
            print("No EEG data found")
            return None
        
        # Extract audio envelopes (wavA and wavB)
        wavA = None
        wavB = None
        if hasattr(trial, 'wavA'):
            wavA = trial.wavA
            if isinstance(wavA, np.ndarray) and wavA.dtype == object:
                wavA = wavA.flat[0]
        if hasattr(trial, 'wavB'):
            wavB = trial.wavB
            if isinstance(wavB, np.ndarray) and wavB.dtype == object:
                wavB = wavB.flat[0]
        
        # Get trigger to mark trial start
        trigger_sample = 0  # Trigger at trial start
        if hasattr(trial, 'event'):
            event = trial.event
            if isinstance(event, np.ndarray) and event.size > 0:
                first_event = event.flat[0]
                if hasattr(first_event, 'eeg'):
                    eeg_events = first_event.eeg
                    if isinstance(eeg_events, np.ndarray) and eeg_events.size > 0:
                        first_eeg_event = eeg_events.flat[0]
                        if hasattr(first_eeg_event, 'sample'):
                            sample_val = first_eeg_event.sample
                            if isinstance(sample_val, np.ndarray):
                                trigger_sample = int(sample_val.flatten()[0]) if sample_val.size > 0 else 0
                            else:
                                trigger_sample = int(sample_val)
        
        # Fulsang sampling rate: 64 Hz
        fs = 64.0
        
        print(f"\nTrial {trial_idx} Analysis:")
        print(f"  Trigger sample: {trigger_sample}")
        print(f"  Sampling rate: {fs} Hz")
        print(f"  EEG shape: {eeg_data.shape}")
        
        # Use wavA (attended speaker) for delay measurement
        if wavA is not None and isinstance(wavA, np.ndarray):
            # If multi-band, use first band or mean
            if wavA.ndim > 1:
                audio_envelope = np.mean(wavA, axis=1) if wavA.shape[1] > 1 else wavA[:, 0]
            else:
                audio_envelope = wavA
            
            # Align lengths
            min_len = min(len(audio_envelope), len(eeg_data))
            audio_envelope = audio_envelope[:min_len]
            eeg_signal = eeg_data[:min_len, :]
            
            print(f"  Audio envelope length: {len(audio_envelope)}")
            print(f"  EEG signal length: {len(eeg_signal)}")
            
            # Measure delay
            delay_ms, delay_samples, peak_corr, all_corrs, lags_ms = measure_delay_from_crosscorrelation(
                audio_envelope, eeg_signal, fs, max_lag_ms=600
            )
            
            print(f"\n" + "="*80)
            print("RESULTS:")
            print("="*80)
            print(f"  Neural Response Delay: {delay_ms:.2f} ms ({delay_samples} samples)")
            print(f"  Peak Correlation: {peak_corr:.4f}")
            print(f"\n  Expected range: 150-400ms (Ding & Simon, 2012)")
            print(f"  Your measurement: {delay_ms:.2f}ms")
            
            if 150 <= delay_ms <= 400:
                print(f"  YES: Delay is within expected range!")
            else:
                print(f"  WARNING: Delay is outside expected range (may need investigation)")
            
            # Plot
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(lags_ms, all_corrs, 'b-', linewidth=2)
            plt.axvline(x=delay_ms, color='r', linestyle='--', linewidth=2, label=f'Peak: {delay_ms:.1f}ms')
            plt.axvspan(150, 400, alpha=0.2, color='green', label='Expected range (150-400ms)')
            plt.xlabel('Lag (ms)')
            plt.ylabel('Cross-Correlation')
            plt.title('Cross-Correlation: EEG vs Audio')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            time_axis = np.arange(len(audio_envelope)) / fs
            plt.plot(time_axis[:int(5*fs)], audio_envelope[:int(5*fs)], 'b-', label='Audio Envelope', alpha=0.7)
            eeg_mean = np.mean(eeg_signal, axis=1)
            eeg_norm = (eeg_mean - np.mean(eeg_mean)) / (np.std(eeg_mean) + 1e-8)
            audio_norm = (audio_envelope - np.mean(audio_envelope)) / (np.std(audio_envelope) + 1e-8)
            plt.plot(time_axis[:int(5*fs)], eeg_norm[:int(5*fs)] * 0.5, 'r-', label='EEG (normalized, scaled)', alpha=0.7)
            plt.axvline(x=trigger_sample/fs, color='g', linestyle=':', linewidth=2, label='Trigger (trial start)')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Amplitude (normalized)')
            plt.title('Audio vs EEG (First 5 seconds)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'fulsang_delay_analysis_{subject_id}_trial{trial_idx}.png', dpi=150)
            print(f"\n  Plot saved to: fulsang_delay_analysis_{subject_id}_trial{trial_idx}.png")
            plt.show()
            
            return {
                'delay_ms': delay_ms,
                'delay_samples': delay_samples,
                'peak_correlation': peak_corr,
                'fs': fs
            }
    
    return None

if __name__ == "__main__":
    print("\n" + "="*80)
    print("AAD DELAY MEASUREMENT TOOL")
    print("="*80)
    print("\nThis tool measures the neural response delay in AAD:")
    print("  1. Uses trigger to mark stimulus onset")
    print("  2. Computes cross-correlation between audio and EEG")
    print("  3. Finds peak correlation lag (the delay)")
    print("\nExpected delay: 150-400ms (Ding & Simon, 2012)")
    print("="*80 + "\n")
    
    result = analyze_fulsang_delay(subject_id='S1', trial_idx=0)
    
    if result:
        print("\n" + "="*80)
        print("HOW TO USE THIS DELAY:")
        print("="*80)
        print("1. In your CCA model, use time-lagged audio features:")
        print(f"   - Lag range: {max(0, int(result['delay_ms']-50))}-{int(result['delay_ms']+50)}ms")
        print(f"   - Or use standard range: 150-400ms")
        print("\n2. Example code:")
        print("   from FULCCA import make_lagged_audio")
        print("   lag_samples = np.arange(10, 26)  # 150-400ms at 64Hz")
        print("   audio_lagged = make_lagged_audio(audio, lag_samples, fs=64.0)")
        print("\n3. The delay accounts for:")
        print("   - Auditory pathway processing: ~50-100ms")
        print("   - Cortical processing: ~100-200ms")
        print("   - Attention modulation: ~50-100ms")
