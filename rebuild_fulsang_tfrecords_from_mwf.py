#!/usr/bin/env python3
"""
Rebuild Fulsang TFRecords from the cleaned MWF .mat outputs with strict trial/label alignment.

Why this exists:
- Raw/MWF files sometimes contain extra EEG segments (practice/aborted/partial) not present in Exp_Info labels.
- This script writes a new TFRecord set where trial count/order matches available labels (1:1) and audio mapping.

Inputs:
- MWF .mat files produced by `mwf_artifact_removal.py` Fuglsang pipeline (e.g. MWF_cleaned_Fuglsang/subXX_MWF.mat)
- Audio WAV directory Data/Fulsang/AUDIO for envelopes (Aske/Marianne) OR expinfo wavfile_male/female if present

Outputs:
- TFRecords in --output_dir (default fulsang_preprocessed_rebuilt)

Note:
- This is intended for debugging/visualization and for building a consistent Fulsang-only dataset.
  CombinedDataset/CombinedCCA currently uses WAV envelopes directly and does not require TFRecords.
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import scipy.io as sio
import tensorflow as tf
from tqdm import tqdm


def _bytes_feature(v: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))


def _int64_feature(v: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v)]))


def _float_feature_list(arr: np.ndarray) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=[float(x) for x in arr.reshape(-1)]))


def _extract_env_from_wav(audio_path: Path, target_len: int, target_fs: int = 64) -> Optional[np.ndarray]:
    try:
        from scipy.io import wavfile
        from scipy import signal
        from scipy.signal import hilbert

        if not audio_path.exists():
            return None
        fs, audio = wavfile.read(str(audio_path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        if fs != target_fs:
            n = int(round(len(audio) * target_fs / fs))
            audio = signal.resample(audio, n)
        env = np.abs(hilbert(audio.astype(np.float64))).astype(np.float32)
        # smooth a bit
        if env.size > 9:
            kernel = np.ones(9, dtype=np.float32) / 9.0
            env = np.convolve(env, kernel, mode="same").astype(np.float32)
        # resize to match EEG window length (trial length in samples at 64 Hz)
        if env.size != target_len:
            src = np.linspace(0.0, 1.0, num=env.size)
            dst = np.linspace(0.0, 1.0, num=target_len)
            env = np.interp(dst, src, env).astype(np.float32)
        return env
    except Exception:
        return None


def _load_trials_from_mwf(mwf_file: Path) -> Tuple[str, List[dict]]:
    mat = sio.loadmat(str(mwf_file), squeeze_me=True, struct_as_record=False)
    subject_id = mat.get("subject_id", None)
    if isinstance(subject_id, np.ndarray):
        subject_id = str(subject_id.item())
    if subject_id is None:
        # sub01_MWF.mat -> 1
        subject_id = mwf_file.stem.replace("_MWF", "").replace("sub", "")
        subject_id = f"S{int(subject_id):02d}"
    trials = mat.get("trials", [])
    if not isinstance(trials, np.ndarray):
        trials = [trials] if trials else []
    else:
        trials = trials.flatten().tolist()

    out = []
    for t in trials:
        if isinstance(t, dict):
            d = t
        else:
            # mat_struct
            d = {}
            for k in ("eeg_data", "sample_rate", "attention_label", "trial_idx", "original_trial_idx", "audio_file_male", "audio_file_female"):
                if hasattr(t, k):
                    d[k] = getattr(t, k)
        if "eeg_data" not in d:
            continue
        out.append(d)
    return str(subject_id), out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mwf_dir", type=str, default="MWF_cleaned_Fuglsang")
    p.add_argument("--audio_dir", type=str, default="Data/Fulsang/AUDIO")
    p.add_argument("--output_dir", type=str, default="fulsang_preprocessed_rebuilt")
    p.add_argument("--sampling_rate", type=int, default=64)
    p.add_argument("--max_subjects", type=int, default=0, help="0 = all")
    args = p.parse_args()

    mwf_dir = Path(args.mwf_dir)
    out_dir = Path(args.output_dir)
    audio_dir = Path(args.audio_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mwf_files = sorted(mwf_dir.glob("sub*_MWF.mat"))
    if not mwf_files:
        raise SystemExit(f"No MWF files found in {mwf_dir}")

    manifest = {"tfrecords": [], "subjects": {}}

    for i, mwf_file in enumerate(mwf_files):
        if args.max_subjects and i >= args.max_subjects:
            break
        subject_id, trials = _load_trials_from_mwf(mwf_file)
        if not trials:
            continue

        tfrec_path = out_dir / f"{subject_id}.tfrecords"
        n_written = 0
        with tf.io.TFRecordWriter(str(tfrec_path)) as w:
            for trial in trials:
                eeg = np.asarray(trial["eeg_data"], dtype=np.float32)
                if eeg.ndim != 2:
                    continue
                n_samples = int(eeg.shape[0])
                n_channels = int(eeg.shape[1])
                label = int(trial.get("attention_label", 0))

                # Build wavA/wavB using filenames if present; otherwise skip audio fields
                wavA = None
                wavB = None
                male = trial.get("audio_file_male", None)
                female = trial.get("audio_file_female", None)
                if male:
                    wavA = _extract_env_from_wav(audio_dir / str(male), n_samples, target_fs=args.sampling_rate)
                if female:
                    wavB = _extract_env_from_wav(audio_dir / str(female), n_samples, target_fs=args.sampling_rate)

                feat = {
                    "subject_id": _bytes_feature(str(subject_id).encode("utf-8")),
                    "n_samples": _int64_feature(n_samples),
                    "n_channels": _int64_feature(n_channels),
                    "sampling_rate": _int64_feature(int(args.sampling_rate)),
                    "attention_label": _int64_feature(label),
                    "trial_idx": _int64_feature(int(trial.get("trial_idx", n_written))),
                    "original_trial_idx": _int64_feature(int(trial.get("original_trial_idx", trial.get("trial_idx", n_written)))),
                    "eeg": _float_feature_list(eeg),
                }
                if wavA is not None:
                    feat["wavA"] = _float_feature_list(wavA.astype(np.float32))
                if wavB is not None:
                    feat["wavB"] = _float_feature_list(wavB.astype(np.float32))

                ex = tf.train.Example(features=tf.train.Features(feature=feat))
                w.write(ex.SerializeToString())
                n_written += 1

        manifest["tfrecords"].append(str(tfrec_path))
        manifest["subjects"][subject_id] = {"n_trials": n_written, "source_mwf": str(mwf_file)}
        print(f"Wrote {n_written} trials to {tfrec_path}")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDone. Manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()

