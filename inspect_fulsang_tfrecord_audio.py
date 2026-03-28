#!/usr/bin/env python3
"""
Inspect Fulsang TFRecord files for wavA/wavB (audio envelope) content.
Use this to verify that TFRecords contain non-zero envelopes before running FULCCA.
"""

import sys
import numpy as np

def main():
    try:
        import tensorflow as tf
    except ImportError:
        print("Install tensorflow: pip install tensorflow")
        sys.exit(1)

    tfrecord_dir = "fulsang_preprocessed/tfrecords"
    if len(sys.argv) > 1:
        tfrecord_dir = sys.argv[1]

    from pathlib import Path
    path = Path(tfrecord_dir)
    if not path.exists():
        print(f"Directory not found: {path}")
        sys.exit(1)
    files = sorted(path.glob("*.tfrecords")) or sorted(path.glob("fulsang_*.tfrecords"))
    if not files:
        files = list(path.glob("*"))
    if not files:
        print(f"No TFRecord files in {path}")
        sys.exit(1)

    print(f"Inspecting Fulsang TFRecords in: {path}")
    print(f"Files: {len(files)}")
    n_samples_expected = 3200  # Fulsang trial length at 64 Hz

    for tfrecord_file in files[:3]:  # first 3 files
        print(f"\n--- {tfrecord_file.name} ---")
        try:
            dataset = tf.data.TFRecordDataset(str(tfrecord_file))
            for i, record in enumerate(dataset):
                if i >= 2:
                    break
                example = tf.train.Example()
                example.ParseFromString(record.numpy())
                features = example.features.feature
                keys = list(features.keys())
                print(f"  Record {i}: feature keys ({len(keys)}): {sorted(keys)[:15]}...")

                if 'n_samples' in features:
                    n_samples = int(features['n_samples'].int64_list.value[0])
                    print(f"    n_samples: {n_samples}")
                else:
                    n_samples = n_samples_expected

                has_wavA = 'wavA' in features
                has_wavB = 'wavB' in features
                wavA_missing = int(features['wavA_missing'].int64_list.value[0]) if 'wavA_missing' in features else None
                wavB_missing = int(features['wavB_missing'].int64_list.value[0]) if 'wavB_missing' in features else None

                print(f"    wavA: key present={has_wavA}, wavA_missing={wavA_missing}")
                print(f"    wavB: key present={has_wavB}, wavB_missing={wavB_missing}")

                if has_wavA:
                    vals = features['wavA'].float_list.value
                    print(f"    wavA length: {len(vals)}, expected: {n_samples}")
                    if vals:
                        arr = np.array(vals, dtype=np.float32)
                        print(f"    wavA stats: min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}, all_zero={np.allclose(arr, 0)}")
                if has_wavB:
                    vals = features['wavB'].float_list.value
                    print(f"    wavB length: {len(vals)}, expected: {n_samples}")
                    if vals:
                        arr = np.array(vals, dtype=np.float32)
                        print(f"    wavB stats: min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}, all_zero={np.allclose(arr, 0)}")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone. If wavA/wavB are missing or all_zero, re-run FULPRE.py and ensure Data/Fulsang/DATA_preproc/*.mat contain data.wavA and data.wavB.")

if __name__ == "__main__":
    main()
