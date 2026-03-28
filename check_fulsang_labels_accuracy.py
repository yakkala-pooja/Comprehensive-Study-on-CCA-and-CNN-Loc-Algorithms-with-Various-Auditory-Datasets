#!/usr/bin/env python3
"""
Check the accuracy of Fulsang labels in TFRecord files.
Verifies that labels match the expected experimental design.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import Counter

def check_fulsang_labels_accuracy(tfrecord_dir: str = "fulsang_preprocessed/tfrecords"):
    """Check if labels in TFRecord files match expected Fulsang experimental design."""
    tfrecord_dir = Path(tfrecord_dir)
    tfrecord_files = list(tfrecord_dir.glob("*.tfrecords"))
    
    print("="*80)
    print("FULSANG LABEL ACCURACY CHECK")
    print("="*80)
    print(f"Checking {len(tfrecord_files)} TFRecord files\n")
    
    # Expected Fulsang experimental design:
    # - Each subject has 60 trials
    # - Each trial is 20 seconds = 1280 samples at 64 Hz (or 50 seconds = 3200 samples)
    # - Attention alternates between trials: trial 0 = class 0, trial 1 = class 1, etc.
    
    all_trial_labels = []
    all_subject_trials = {}
    
    for tfrecord_file in sorted(tfrecord_files):
        print(f"Analyzing {tfrecord_file.name}...")
        try:
            dataset = tf.data.TFRecordDataset(str(tfrecord_file))
            file_trials = []
            file_subject = None
            
            for record in dataset:
                example = tf.train.Example.FromString(record.numpy())
                features = example.features.feature
                
                # Get trial label
                if 'attention_label' in features:
                    label_list = features['attention_label'].int64_list.value
                    if label_list and len(label_list) > 0:
                        label = int(label_list[0])
                        file_trials.append(label)
                        all_trial_labels.append(label)
                
                # Get subject ID
                if 'subject_id' in features and file_subject is None:
                    subject_list = features['subject_id'].bytes_list.value
                    if subject_list and len(subject_list) > 0:
                        file_subject = subject_list[0].decode('utf-8')
                
                # Get trial index
                trial_idx = -1
                if 'trial_idx' in features:
                    trial_list = features['trial_idx'].int64_list.value
                    if trial_list and len(trial_list) > 0:
                        trial_idx = int(trial_list[0])
            
            if file_subject:
                all_subject_trials[file_subject] = file_trials
            
            label_counts = Counter(file_trials)
            print(f"  Subject: {file_subject}")
            print(f"  Number of trials: {len(file_trials)}")
            print(f"  Trial label distribution: {dict(label_counts)}")
            print(f"  Trial label sequence (first 20): {file_trials[:20]}")
            
            # Check if labels alternate as expected
            if len(file_trials) > 1:
                expected_pattern = [i % 2 for i in range(len(file_trials))]
                matches_expected = file_trials == expected_pattern
                print(f"  Matches expected alternating pattern (0,1,0,1...): {matches_expected}")
                
                if not matches_expected:
                    print(f"  Expected: {expected_pattern[:20]}")
                    print(f"  Actual:   {file_trials[:20]}")
                    # Check how many match
                    matches = sum(1 for i, (e, a) in enumerate(zip(expected_pattern, file_trials)) if e == a)
                    print(f"  Matches: {matches}/{len(file_trials)} ({100*matches/len(file_trials):.1f}%)")
            
            print()
            
        except Exception as e:
            print(f"  ERROR: {e}\n")
            import traceback
            traceback.print_exc()
            continue
    
    print("="*80)
    print("OVERALL ANALYSIS")
    print("="*80)
    
    overall_counts = Counter(all_trial_labels)
    print(f"Total trials across all subjects: {len(all_trial_labels)}")
    print(f"Overall trial label distribution: {dict(overall_counts)}")
    
    # Check if distribution is balanced
    if len(overall_counts) == 2:
        class_0_count = overall_counts.get(0, 0)
        class_1_count = overall_counts.get(1, 0)
        total = class_0_count + class_1_count
        balance_ratio = min(class_0_count, class_1_count) / max(class_0_count, class_1_count) if max(class_0_count, class_1_count) > 0 else 0
        print(f"Balance ratio: {balance_ratio:.3f} (1.0 = perfectly balanced)")
        
        if balance_ratio < 0.9:
            print("  ⚠ WARNING: Trial labels are imbalanced!")
    
    # Check per-subject patterns
    print("\nPer-subject pattern analysis:")
    for subject, trials in sorted(all_subject_trials.items()):
        if len(trials) >= 2:
            # Check if it alternates
            transitions = sum(1 for i in range(len(trials)-1) if trials[i] != trials[i+1])
            expected_transitions = len(trials) - 1
            is_alternating = transitions == expected_transitions
            
            print(f"  {subject}: {len(trials)} trials, {transitions} transitions")
            if is_alternating:
                print(f"    ✓ Alternating pattern (0,1,0,1...)")
            else:
                print(f"    ✗ NOT alternating - pattern: {trials[:10]}...")
    
    # Expected Fulsang design: 60 trials per subject, alternating
    expected_trials_per_subject = 60
    print(f"\nExpected: {expected_trials_per_subject} trials per subject, alternating (0,1,0,1...)")
    
    # Check if all subjects have correct number of trials
    incorrect_subjects = []
    for subject, trials in all_subject_trials.items():
        if len(trials) != expected_trials_per_subject:
            incorrect_subjects.append((subject, len(trials)))
    
    if incorrect_subjects:
        print(f"\n⚠ Subjects with incorrect trial counts:")
        for subject, count in incorrect_subjects:
            print(f"  {subject}: {count} trials (expected {expected_trials_per_subject})")
    else:
        print(f"\n✓ All subjects have {expected_trials_per_subject} trials")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    # Determine if labels are accurate
    all_alternating = all(
        sum(1 for i in range(len(trials)-1) if trials[i] != trials[i+1]) == len(trials) - 1
        for trials in all_subject_trials.values() if len(trials) > 1
    )
    
    all_correct_count = all(
        len(trials) == expected_trials_per_subject
        for trials in all_subject_trials.values()
    )
    
    if all_alternating and all_correct_count and len(overall_counts) == 2:
        print("✓ Labels appear ACCURATE:")
        print("  - All subjects have alternating trial labels (0,1,0,1...)")
        print("  - All subjects have correct number of trials")
        print("  - Overall distribution is balanced")
    else:
        print("✗ Labels may be INACCURATE:")
        if not all_alternating:
            print("  - Some subjects do not have alternating trial labels")
        if not all_correct_count:
            print("  - Some subjects have incorrect number of trials")
        if len(overall_counts) != 2:
            print(f"  - Unexpected number of classes: {list(overall_counts.keys())}")

if __name__ == "__main__":
    check_fulsang_labels_accuracy()

