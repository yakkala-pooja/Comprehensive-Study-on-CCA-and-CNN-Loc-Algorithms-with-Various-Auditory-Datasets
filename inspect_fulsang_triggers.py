#!/usr/bin/env python3
"""
Inspect Triggers in Fulsang Dataset

This script loads TFRecord files and reports on trigger information:
- Trigger values and their distribution
- Which trials have which triggers
- Trigger patterns across subjects
- Summary statistics

Usage:
    python inspect_fulsang_triggers.py [--tfrecord_dir DIR] [--subject_id SUBJECT] [--output_file FILE]
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import tensorflow as tf
from tqdm import tqdm
from collections import Counter, defaultdict
import os

# Optional pandas import
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Note: pandas not available. CSV export will be disabled.")

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class FulsangTriggerInspector:
    """Inspect trigger information in Fulsang dataset."""
    
    def __init__(self, tfrecord_dir: str):
        """
        Initialize the inspector.
        
        Args:
            tfrecord_dir: Directory containing TFRecord files
        """
        self.tfrecord_dir = Path(tfrecord_dir)
        
        if not self.tfrecord_dir.exists():
            raise ValueError(f"TFRecord directory not found: {self.tfrecord_dir}")
        
        # Find all TFRecord files
        self.tfrecord_files = list(self.tfrecord_dir.glob("*.tfrecords"))
        if not self.tfrecord_files:
            self.tfrecord_files = list(self.tfrecord_dir.glob("*/*.tfrecords"))
        if not self.tfrecord_files:
            self.tfrecord_files = list(self.tfrecord_dir.glob("*/*/*.tfrecords"))
        
        if not self.tfrecord_files:
            raise ValueError(f"No TFRecord files found in {self.tfrecord_dir}")
        
        print(f"Found {len(self.tfrecord_files)} TFRecord files")
    
    def inspect_all_triggers(self, subject_id: Optional[str] = None) -> List[Dict]:
        """
        Inspect all triggers in the dataset.
        
        Args:
            subject_id: Optional subject ID to filter
            
        Returns:
            List of dictionaries with trigger information
        """
        all_trigger_data = []
        
        for tfrecord_file in tqdm(self.tfrecord_files, desc="Inspecting triggers"):
            try:
                dataset = tf.data.TFRecordDataset(str(tfrecord_file))
                
                for record_idx, record in enumerate(dataset):
                    try:
                        example = tf.train.Example.FromString(record.numpy())
                        features = example.features.feature
                        
                        # Extract subject ID
                        file_subject_id = "unknown"
                        if 'subject_id' in features:
                            subject_values = features['subject_id'].bytes_list.value
                            if subject_values:
                                file_subject_id = subject_values[0].decode('utf-8')
                        
                        # Filter by subject if specified
                        if subject_id and file_subject_id != subject_id:
                            continue
                        
                        # Extract trigger
                        trigger = None
                        if 'trigger' in features:
                            trigger_values = features['trigger'].int64_list.value
                            if trigger_values:
                                trigger = int(trigger_values[0])
                        
                        # Extract other metadata
                        trial_idx = record_idx
                        if 'trial_idx' in features:
                            trial_idx_values = features['trial_idx'].int64_list.value
                            if trial_idx_values:
                                trial_idx = int(trial_idx_values[0])
                        
                        attention_label = None
                        if 'attention_label' in features:
                            label_values = features['attention_label'].int64_list.value
                            if label_values:
                                attention_label = int(label_values[0])
                        
                        # Extract experimental conditions
                        attend_mf = None
                        attend_lr = None
                        acoustic_condition = None
                        n_speakers = None
                        
                        if 'attend_mf_raw' in features:
                            attend_mf_values = features['attend_mf_raw'].int64_list.value
                            if attend_mf_values:
                                attend_mf = int(attend_mf_values[0])
                        
                        if 'attend_lr' in features:
                            attend_lr_values = features['attend_lr'].int64_list.value
                            if attend_lr_values:
                                attend_lr = int(attend_lr_values[0])
                        
                        if 'acoustic_condition' in features:
                            acoustic_values = features['acoustic_condition'].int64_list.value
                            if acoustic_values:
                                acoustic_condition = int(acoustic_values[0])
                        
                        if 'n_speakers' in features:
                            n_speakers_values = features['n_speakers'].int64_list.value
                            if n_speakers_values:
                                n_speakers = int(n_speakers_values[0])
                        
                        trigger_data = {
                            'subject_id': file_subject_id,
                            'trial_idx': trial_idx,
                            'record_idx': record_idx,
                            'file': tfrecord_file.name,
                            'trigger': trigger,
                            'attention_label': attention_label,
                            'attend_mf': attend_mf,  # 1=male, 2=female
                            'attend_lr': attend_lr,  # 1=left, 2=right
                            'acoustic_condition': acoustic_condition,  # 1=anechoic, 2=mild, 3=high reverb
                            'n_speakers': n_speakers
                        }
                        
                        all_trigger_data.append(trigger_data)
                        
                    except Exception as e:
                        print(f"Error processing record in {tfrecord_file.name}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error loading {tfrecord_file.name}: {e}")
                continue
        
        return all_trigger_data
    
    def print_summary(self, trigger_data: List[Dict]):
        """Print summary statistics about triggers."""
        if not trigger_data:
            print("No trigger data found!")
            return
        
        print("\n" + "="*80)
        print("FULSANG DATASET TRIGGER SUMMARY")
        print("="*80)
        
        # Basic statistics
        total_trials = len(trigger_data)
        trials_with_triggers = sum(1 for t in trigger_data if t['trigger'] is not None)
        trials_without_triggers = total_trials - trials_with_triggers
        
        print(f"\nTotal Trials: {total_trials}")
        print(f"Trials with triggers: {trials_with_triggers} ({100*trials_with_triggers/total_trials:.1f}%)")
        print(f"Trials without triggers: {trials_without_triggers} ({100*trials_without_triggers/total_trials:.1f}%)")
        
        # Trigger value distribution
        trigger_values = [t['trigger'] for t in trigger_data if t['trigger'] is not None]
        if trigger_values:
            trigger_counter = Counter(trigger_values)
            print(f"\nTrigger Value Distribution:")
            print(f"  Unique trigger values: {len(trigger_counter)}")
            print(f"  Trigger value range: {min(trigger_values)} to {max(trigger_values)}")
            print(f"\n  Most common triggers:")
            for trigger_val, count in trigger_counter.most_common(10):
                percentage = 100 * count / len(trigger_values)
                print(f"    Trigger {trigger_val}: {count} trials ({percentage:.1f}%)")
        
        # By subject
        subjects = set(t['subject_id'] for t in trigger_data)
        print(f"\nSubjects: {len(subjects)}")
        if len(subjects) <= 20:
            print(f"  Subject IDs: {sorted(subjects)}")
        
        # Trigger distribution by subject
        triggers_by_subject = defaultdict(list)
        for t in trigger_data:
            if t['trigger'] is not None:
                triggers_by_subject[t['subject_id']].append(t['trigger'])
        
        if triggers_by_subject:
            print(f"\nTrigger Distribution by Subject:")
            for subject in sorted(subjects)[:10]:  # Show first 10 subjects
                if subject in triggers_by_subject:
                    subject_triggers = triggers_by_subject[subject]
                    unique_triggers = set(subject_triggers)
                    print(f"  {subject}: {len(subject_triggers)} trials, {len(unique_triggers)} unique triggers: {sorted(unique_triggers)}")
        
        # Relationship with experimental conditions
        if any(t.get('attend_mf') is not None for t in trigger_data):
            print(f"\nTriggers by Attention Condition (attend_mf):")
            triggers_by_attend_mf = defaultdict(list)
            for t in trigger_data:
                if t['trigger'] is not None and t.get('attend_mf') is not None:
                    triggers_by_attend_mf[t['attend_mf']].append(t['trigger'])
            for attend_mf, triggers in sorted(triggers_by_attend_mf.items()):
                trigger_counter = Counter(triggers)
                attend_label = "Male" if attend_mf == 1 else "Female" if attend_mf == 2 else "Unknown"
                print(f"  Attend {attend_label} (attend_mf={attend_mf}): {len(triggers)} trials")
                print(f"    Unique triggers: {sorted(trigger_counter.keys())}")
        
        if any(t.get('acoustic_condition') is not None for t in trigger_data):
            print(f"\nTriggers by Acoustic Condition:")
            triggers_by_acoustic = defaultdict(list)
            for t in trigger_data:
                if t['trigger'] is not None and t.get('acoustic_condition') is not None:
                    triggers_by_acoustic[t['acoustic_condition']].append(t['trigger'])
            acoustic_labels = {1: "Anechoic", 2: "Mild Reverb", 3: "High Reverb"}
            for acoustic, triggers in sorted(triggers_by_acoustic.items()):
                trigger_counter = Counter(triggers)
                label = acoustic_labels.get(acoustic, f"Condition {acoustic}")
                print(f"  {label} (condition={acoustic}): {len(triggers)} trials")
                print(f"    Unique triggers: {sorted(trigger_counter.keys())}")
        
        print("\n" + "="*80)
    
    def create_trigger_table(self, trigger_data: List[Dict], max_rows: int = 50):
        """Create a table with trigger information (pandas DataFrame if available, else dict list)."""
        if HAS_PANDAS:
            df = pd.DataFrame(trigger_data)
            # Sort by subject and trial
            if 'subject_id' in df.columns and 'trial_idx' in df.columns:
                df = df.sort_values(['subject_id', 'trial_idx'])
            return df
        else:
            # Return sorted list of dicts
            sorted_data = sorted(trigger_data, key=lambda x: (x.get('subject_id', ''), x.get('trial_idx', 0)))
            return sorted_data
    
    def save_detailed_report(self, trigger_data: List[Dict], output_file: str):
        """Save detailed trigger report to CSV file."""
        if not HAS_PANDAS:
            print("Error: pandas is required to save CSV files. Install with: pip install pandas")
            return
        
        df = self.create_trigger_table(trigger_data)
        df.to_csv(output_file, index=False)
        print(f"\nDetailed trigger report saved to: {output_file}")
        print(f"  Total rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")


def main():
    parser = argparse.ArgumentParser(
        description='Inspect trigger information in Fulsang dataset'
    )
    parser.add_argument('--tfrecord_dir', type=str, default='fulsang_preprocessed',
                       help='Directory containing TFRecord files')
    parser.add_argument('--subject_id', type=str, default=None,
                       help='Subject ID to filter (e.g., S1, S2)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='CSV file to save detailed trigger report')
    parser.add_argument('--max_rows', type=int, default=50,
                       help='Maximum rows to display in table')
    
    args = parser.parse_args()
    
    # Create inspector
    print(f"Initializing inspector with TFRecord directory: {args.tfrecord_dir}")
    inspector = FulsangTriggerInspector(args.tfrecord_dir)
    
    # Inspect triggers
    print("\nLoading trigger data...")
    trigger_data = inspector.inspect_all_triggers(subject_id=args.subject_id)
    
    if not trigger_data:
        print("No trigger data found!")
        return
    
    # Print summary
    inspector.print_summary(trigger_data)
    
    # Display sample table
    print(f"\nSample Trigger Data (first {args.max_rows} rows):")
    print("-"*80)
    table_data = inspector.create_trigger_table(trigger_data, max_rows=args.max_rows)
    if HAS_PANDAS:
        print(table_data.head(args.max_rows).to_string())
    else:
        # Print as formatted table
        if table_data:
            # Print header
            if table_data:
                keys = list(table_data[0].keys())
                header = " | ".join(f"{k:15}" for k in keys)
                print(header)
                print("-" * len(header))
                # Print rows
                for row in table_data[:args.max_rows]:
                    values = [str(row.get(k, ''))[:15] for k in keys]
                    print(" | ".join(f"{v:15}" for v in values))
    
    # Save detailed report if requested
    if args.output_file:
        inspector.save_detailed_report(trigger_data, args.output_file)


if __name__ == '__main__':
    main()
