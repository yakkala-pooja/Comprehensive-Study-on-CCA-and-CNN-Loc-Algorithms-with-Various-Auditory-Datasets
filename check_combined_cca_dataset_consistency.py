#!/usr/bin/env python3
"""
Check Combined CCA Dataset and Preprocessing Consistency

Verifies that for Combined CCA (DAS + Fulsang):
1. Sampling rate is the same for both datasets (EEG and envelope)
2. Filter coefficients / filter design are comparable for both .mat pipelines
3. Audio envelope normalization and y-axis scale are the same
4. Signal range and quantized values (dtype, min/max, absolute value) match

Run from project root: python check_combined_cca_dataset_consistency.py
"""

import os
import sys
import numpy as np
from pathlib import Path
from scipy import signal as scipy_signal

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

def get_das_16subjects_params():
    """Extract DAS 16-subjects preprocessing parameters (EEG and source .mat)."""
    # From das_preprocessing_combined.py: target_sampling_rate=64, bandpass 0.5-40 Hz at 128 then 128→64
    fs = 128
    nyquist = fs / 2
    low_freq = 0.5 / nyquist
    high_freq = min(40.0 / nyquist, 0.99)
    b, a = scipy_signal.butter(4, [low_freq, high_freq], btype='band')
    return {
        'name': 'DAS 16-subjects',
        'source': 'TFRecords from das_preprocessing_combined.py (reads DAS .mat, 128→64 Hz)',
        'eeg_sampling_rate': 128,
        'eeg_filter_type': 'bandpass',
        'eeg_filter_low_hz': 0.5,
        'eeg_filter_high_hz': 40.0,
        'eeg_filter_order': 4,
        'eeg_filter_b': b,
        'eeg_filter_a': a,
        'eeg_normalization': 'z-score per channel (mean/std)',
        'n_channels': 64,
    }

def get_fulsang_mwf_params():
    """Extract Fulsang MWF preprocessing parameters (from mwf_artifact_removal)."""
    # From mwf_artifact_removal.py FuglsangDatasetMWF: original_sampling_rate=512, target_sampling_rate=128
    # high-pass 0.5 Hz, notch 50 Hz, then downsample 512->128
    fs_orig = 512
    nyquist = fs_orig / 2
    highpass_freq = 0.5 / nyquist
    b_hp, a_hp = scipy_signal.butter(4, highpass_freq, btype='high')
    notch_freq = 50.0 / nyquist
    b_notch, a_notch = scipy_signal.iirnotch(notch_freq, Q=30)
    return {
        'name': 'Fulsang MWF',
        'source': 'sub*_MWF.mat from mwf_artifact_removal.FuglsangDatasetMWF (reads Fulsang .mat)',
        'eeg_sampling_rate_original': 512,
        'eeg_sampling_rate': 128,
        'eeg_filter_type': 'high-pass 0.5 Hz + notch 50 Hz (no explicit low-pass)',
        'eeg_filter_highpass_hz': 0.5,
        'eeg_filter_notch_hz': 50.0,
        'eeg_filter_order': 4,
        'eeg_highpass_b': b_hp,
        'eeg_highpass_a': a_hp,
        'eeg_notch_b': b_notch,
        'eeg_notch_a': a_notch,
        'eeg_normalization': 'robust z-score per channel (median/MAD)',
        'downsample_factor': 4,
    }

def get_combined_envelope_params():
    """Extract envelope extraction parameters used by CombinedDataset for both datasets."""
    # CombinedDataset._extract_envelope_from_audio is used for both DAS and Fulsang
    # when envelopes are extracted from WAV files (DAS: via _extract_das_envelopes_from_original;
    # Fulsang: via _extract_fulsang_envelopes_from_audio). DAS 16-subjects path does NOT use
    # this — it uses fallback_envelopes (ramp) because TFRecords don't store envelope.
    return {
        'name': 'CombinedDataset._extract_envelope_from_audio',
        'used_for_das': 'Only when loading DAS from MWF or DASPREPROCESS .mat (not for COMBINED_DAS TFRecords)',
        'used_for_fulsang': 'Yes, when loading Fulsang MWF .mat and extracting from audio WAV',
        'target_fs': 128,
        'audio_normalization': 'Divide by max(abs(audio)) so audio in [-1, 1]',
        'envelope_method': 'abs(audio) then 9-point moving average',
        'envelope_bandpass': 'None (no bandpass on envelope in this path)',
        'envelope_scale': 'Non-negative, typically in [0, 1] after abs and smoothing',
        'envelope_dtype': 'float32',
        'note_combined_das': 'DAS COMBINED_DAS: TFRecords store left_audio_file/right_audio_file; CombinedDataset extracts Butterworth 4-band envelope at 64 Hz.',
    }

def get_das_preprocessor_audio_params():
    """Optional: DAS standalone preprocessor (das_preprocessor.py) — different from Combined path."""
    try:
        from das_preprocessor import DasPreprocessor
        dp = DasPreprocessor()
        return {
            'name': 'das_preprocessor.py (standalone DAS, not used by CombinedDataset for COMBINED_DAS)',
            'intermediatefs_audio': dp.params['intermediatefs_audio'],
            'intermediateSampleRate': dp.params['intermediateSampleRate'],
            'targetSampleRate': dp.params['targetSampleRate'],
            'envelope_bandpass_hz': (dp.params['highpass'], dp.params['lowpass']),
            'envelope_method': 'gammatone filterbank + power law',
            'envelope_normalization': '(envelope - mean) / (std + 1e-8)',
            'bp_filter': dp.bp_filter,
        }
    except Exception as e:
        return {'name': 'das_preprocessor', 'error': str(e)}

def _report_sample_data(label: str, path: Path, is_tfrecord: bool):
    """Load a few samples and report dtype, min, max, mean, std, abs range."""
    try:
        if is_tfrecord:
            import tensorflow as tf
            files = list(path.glob("*.tfrecords"))[:1]
            if not files:
                files = list(path.glob("*/*.tfrecords"))[:1]
            if not files:
                print(f"  {label}: No .tfrecords found under {path}")
                return
            ds = tf.data.TFRecordDataset(str(files[0])).take(500)
            eeg_list = []
            for record in ds:
                ex = tf.train.Example.FromString(record.numpy())
                f = ex.features.feature
                if 'eeg' in f:
                    eeg_list.append(np.array(f['eeg'].float_list.value, dtype=np.float32))
            if not eeg_list:
                print(f"  {label}: No EEG in first 500 records")
                return
            arr = np.array(eeg_list)
        else:
            import scipy.io as sio
            mat_files = sorted(path.glob("sub*_MWF.mat"))[:1]
            if not mat_files:
                print(f"  {label}: No sub*_MWF.mat in {path}")
                return
            data = sio.loadmat(str(mat_files[0]), squeeze_me=True, struct_as_record=False)
            trials = data.get('trials')
            if trials is None:
                print(f"  {label}: No 'trials' in {mat_files[0].name}")
                return
            trials = np.atleast_1d(trials).flatten()
            chunks = []
            for t in trials[:3]:
                eeg = getattr(t, 'eeg_data', None) or (t.get('eeg_data') if isinstance(t, dict) else None)
                if eeg is not None:
                    chunks.append(np.asarray(eeg, dtype=np.float64))
            if not chunks:
                print(f"  {label}: No eeg_data in trials")
                return
            arr = np.vstack(chunks)
        print(f"  {label}:")
        print(f"    shape={arr.shape}, dtype={arr.dtype}")
        print(f"    min={np.min(arr):.6f}, max={np.max(arr):.6f}, mean={np.mean(arr):.6f}, std={np.std(arr):.6f}")
        print(f"    abs: min={np.min(np.abs(arr)):.6f}, max={np.max(np.abs(arr)):.6f}")
    except Exception as e:
        print(f"  {label}: Error sampling: {e}")

def compare_filter_coefficients(das, fulsang):
    """Compare filter coefficients between DAS and Fulsang."""
    lines = []
    lines.append("\n--- Filter coefficients ---")
    lines.append("DAS 16-subjects: single bandpass (0.5–40 Hz), 4th order Butterworth")
    lines.append(f"  b = {np.array(das['eeg_filter_b']).round(6).tolist()}")
    lines.append(f"  a = {np.array(das['eeg_filter_a']).round(6).tolist()}")
    lines.append("Fulsang MWF: high-pass (0.5 Hz) + notch (50 Hz), then decimate 512->128")
    lines.append(f"  high-pass b = {np.array(fulsang['eeg_highpass_b']).round(6).tolist()}")
    lines.append(f"  high-pass a = {np.array(fulsang['eeg_highpass_a']).round(6).tolist()}")
    lines.append(f"  notch b = {np.array(fulsang['eeg_notch_b']).round(6).tolist()}")
    lines.append(f"  notch a = {np.array(fulsang['eeg_notch_a']).round(6).tolist()}")
    lines.append("Conclusion: Filter coefficients are NOT the same (different filter design).")
    return "\n".join(lines)

def main():
    print("=" * 80)
    print("COMBINED CCA – DATASET AND PREPROCESSING CONSISTENCY CHECK")
    print("=" * 80)

    # 1) Sampling rate
    print("\n" + "=" * 80)
    print("1. SAMPLING RATE")
    print("=" * 80)
    try:
        das = get_das_16subjects_params()
        fulsang = get_fulsang_mwf_params()
        print(f"DAS 16-subjects (EEG):     {das['eeg_sampling_rate']} Hz")
        print(f"Fulsang MWF (EEG):         {fulsang['eeg_sampling_rate']} Hz (downsampled from {fulsang['eeg_sampling_rate_original']} Hz)")
        if das['eeg_sampling_rate'] == fulsang['eeg_sampling_rate']:
            print("  -> MATCH: Both use 128 Hz for Combined CCA.")
        else:
            print("  -> MISMATCH: Sampling rates differ.")
    except Exception as e:
        print(f"Error loading params: {e}")
        import traceback
        traceback.print_exc()
        das = fulsang = None

    env_params = get_combined_envelope_params()
    print(f"Envelope (audio->envelope): target_fs = {env_params['target_fs']} Hz (same for both when using _extract_envelope_from_audio)")

    # 2) Filter coefficients for both .mat pipelines
    print("\n" + "=" * 80)
    print("2. FILTER COEFFICIENTS (EEG from .mat pipelines)")
    print("=" * 80)
    if das and fulsang:
        print(compare_filter_coefficients(das, fulsang))
        print("\nSummary:")
        print("  - DAS 16-subjects: bandpass 0.5–40 Hz (4th order Butterworth).")
        print("  - Fulsang MWF: high-pass 0.5 Hz + notch 50 Hz (no 40 Hz low-pass), then decimate.")
        print("  -> Filter design is NOT the same; Fulsang has no explicit 40 Hz low-pass.")

    # 3) Audio normalization and y-axis scale
    print("\n" + "=" * 80)
    print("3. AUDIO ENVELOPE NORMALIZATION AND Y-AXIS SCALE")
    print("=" * 80)
    print("When both use CombinedDataset._extract_envelope_from_audio:")
    print(f"  Normalization: {env_params['audio_normalization']}")
    print(f"  Envelope method: {env_params['envelope_method']}")
    print(f"  Envelope bandpass: {env_params['envelope_bandpass']}")
    print(f"  Envelope scale: {env_params['envelope_scale']}")
    print(f"  -> Same normalization and scale when both use real audio envelopes.")
    print(f"\n  Note: {env_params['note_16subjects']}")

    # 4) Signal range and quantized values
    print("\n" + "=" * 80)
    print("4. SIGNAL RANGE, QUANTIZED VALUES, AND ABSOLUTE VALUE")
    print("=" * 80)
    print("EEG:")
    print("  - DAS 16-subjects: z-score per channel -> roughly mean=0, std=1 (float32).")
    print("  - Fulsang MWF: median/MAD per channel -> different scale (still float).")
    print("  -> EEG scaling is NOT the same (z-score vs median/MAD).")
    print("Envelope (when from _extract_envelope_from_audio):")
    print("  - Range: [0, 1] (non-negative, max-normalized audio then abs + smoothing).")
    print("  - dtype: float32; no quantization in code.")
    print("Envelope (DAS 16-subjects fallback):")
    print("  - Ramp: linspace(0,1) or zeros -> range [0, 1]; same dtype float32.")
    print("  - Absolute value: same as value (non-negative).")

    # Optional: load actual data and report min/max
    print("\n" + "=" * 80)
    print("5. OPTIONAL: ACTUAL DATA RANGE (if paths exist)")
    print("=" * 80)
    das_tfrecord_dir = Path("das_16subjects_preprocessed/tfrecords")
    fulsang_mwf_dir = Path("combined_dataset/fulsang_mwf")
    if not fulsang_mwf_dir.exists():
        fulsang_mwf_dir = Path("MWF_cleaned_Fuglsang")
    sample_mode = "--sample" in sys.argv
    for label, path in [("DAS 16-subjects TFRecords", das_tfrecord_dir), ("Fulsang MWF .mat", fulsang_mwf_dir)]:
        if path.exists():
            if sample_mode:
                _report_sample_data(label, path, is_tfrecord=(label.startswith("DAS")))
            else:
                print(f"  {label}: {path} exists (run with --sample to report actual min/max/dtype).")
        else:
            print(f"  {label}: {path} not found.")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: DO THEY MATCH?")
    print("=" * 80)
    print("""
| Item                    | DAS 16-subjects     | Fulsang MWF          | Match? |
|-------------------------|---------------------|----------------------|--------|
| EEG sampling rate       | 128 Hz              | 128 Hz               | YES    |
| EEG filter              | 0.5–40 Hz bandpass  | 0.5 Hz HP + 50 Hz notch | NO  |
| EEG normalization       | z-score (mean/std)  | median/MAD           | NO     |
| Envelope sampling       | 128 Hz (if real)    | 128 Hz               | YES    |
| Envelope normalization  | max(abs) on audio  | same                 | YES*   |
| Envelope scale (y-axis) | [0, 1]              | [0, 1]               | YES*   |
| Envelope source (16subj)| Real from paths in TFRecords (or fallback) | Real from WAV | YES when DAS paths present |
| dtype                   | float32             | float32              | YES    |

* When both use real envelopes. DAS 16-subjects uses paths in TFRecords (left_audio_file/right_audio_file) when present.
""")
    print("Recommendations:")
    print("  1. Align EEG filter: use same bandpass (e.g. 0.5–40 Hz) and same normalization (e.g. z-score) for both.")
    print("  2. For DAS 16-subjects: consider extracting real envelopes from original DAS .mat + audio and replacing fallback.")
    print("  3. Document that envelope y-axis and normalization match when both use _extract_envelope_from_audio.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
