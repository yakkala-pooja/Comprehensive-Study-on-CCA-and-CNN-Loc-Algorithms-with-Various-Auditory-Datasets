#!/usr/bin/env python3
"""
Test script for DAS combined preprocessing pipeline (for CombinedDataset).
Uses das_preprocessing_combined.py → das_combined_preprocessed (128→64 Hz, Butterworth envelope at 64 Hz).
"""

import os
import sys
from pathlib import Path

def test_preprocessing():
    """Test the DAS combined preprocessing pipeline."""
    print("=" * 80)
    print("TESTING DAS COMBINED PREPROCESSING PIPELINE")
    print("=" * 80)
    
    preprocessing_script = Path("das_preprocessing_combined.py")
    if not preprocessing_script.exists():
        print("✗ das_preprocessing_combined.py not found!")
        return False
    
    print("✓ Preprocessing script found")
    
    data_dir = Path("Data/Das/4004271")
    if not data_dir.exists():
        print(f"✗ DAS data directory not found: {data_dir}")
        return False
    
    print(f"✓ DAS data directory found: {data_dir}")
    
    subject_files = list(data_dir.glob("S*.mat"))
    if not subject_files:
        print("✗ No subject files (S*.mat) found in DAS data directory")
        return False
    
    print(f"✓ Found {len(subject_files)} subject files")
    
    output_dir = Path("das_combined_preprocessed")
    if output_dir.exists():
        print(f"✓ Preprocessing output directory exists: {output_dir}")
        tfrecord_dir = output_dir / "tfrecords"
        if tfrecord_dir.exists():
            tfrecord_files = list(tfrecord_dir.glob("*.tfrecords"))
            print(f"✓ Found {len(tfrecord_files)} TFRecord files")
        else:
            print("⚠ TFRecord directory not found")
    else:
        print("⚠ Preprocessing output directory not found")
        print("  Run: python das_preprocessing_combined.py")
    
    return True

def test_combined_dataset_refs():
    """Check that combined scripts reference COMBINED_DAS."""
    print("\n" + "=" * 80)
    print("CHECKING COMBINED SCRIPTS USE COMBINED_DAS")
    print("=" * 80)
    
    for name in ["CombinedDataset.py", "CombinedCCA.py"]:
        path = Path(name)
        if not path.exists():
            continue
        with open(path, 'r') as f:
            content = f.read()
        if 'COMBINED_DAS' in content and 'das_combined_preprocessed' in content:
            print(f"✓ {name} uses COMBINED_DAS / das_combined_preprocessed")
        else:
            print(f"⚠ {name} may not be configured for COMBINED_DAS")
    return True

def main():
    print("Testing DAS combined pipeline (for CombinedDataset)...")
    preprocessing_ok = test_preprocessing()
    refs_ok = test_combined_dataset_refs()
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    if preprocessing_ok and refs_ok:
        print("✓ Checks passed!")
        print("\nNext steps:")
        print("1. Run DAS preprocessing: python das_preprocessing_combined.py")
        print("2. Use CombinedDataset with default (COMBINED_DAS, das_combined_preprocessed, target_sampling_rate=64)")
    else:
        print("✗ Some checks failed. See above.")
    print("=" * 80)

if __name__ == "__main__":
    main()
