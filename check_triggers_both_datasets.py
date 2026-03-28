#!/usr/bin/env python3
"""
Comprehensive trigger inspection for both DAS and Fulsang datasets.
Checks if triggers are present and where they are positioned.
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TriggerInspector:
    """Inspect triggers in both DAS and Fulsang datasets."""
    
    def __init__(self):
        self.das_tfrecord_dir = Path("das_16subjects_preprocessed/tfrecords")
        self.fulsang_tfrecord_dir = Path("fulsang_preprocessed/tfrecords")
    
    def inspect_tfrecord_features(self, tfrecord_file: Path, max_records: int = 10) -> Dict:
        """Inspect what features are present in a TFRecord file."""
        features_found = set()
        sample_features = {}
        record_count = 0
        
        try:
            dataset = tf.data.TFRecordDataset(str(tfrecord_file))
            for record in dataset:
                if record_count >= max_records:
                    break
                    
                example = tf.train.Example.FromString(record.numpy())
                features = example.features.feature
                
                # Record all feature names
                for feature_name in features.keys():
                    features_found.add(feature_name)
                
                # For first record, get detailed info
                if record_count == 0:
                    for feature_name, feature in features.items():
                        feature_info = {}
                        
                        # Get feature type and values
                        if feature.HasField('int64_list'):
                            values = list(feature.int64_list.value)
                            feature_info['type'] = 'int64_list'
                            feature_info['values'] = values
                            feature_info['count'] = len(values)
                        elif feature.HasField('float_list'):
                            values = list(feature.float_list.value)
                            feature_info['type'] = 'float_list'
                            feature_info['values'] = values[:10] if len(values) > 10 else values  # First 10
                            feature_info['count'] = len(values)
                        elif feature.HasField('bytes_list'):
                            values = [v.decode('utf-8') if isinstance(v, bytes) else str(v) 
                                     for v in feature.bytes_list.value]
                            feature_info['type'] = 'bytes_list'
                            feature_info['values'] = values
                            feature_info['count'] = len(values)
                        
                        sample_features[feature_name] = feature_info
                
                record_count += 1
                
        except Exception as e:
            print(f"Error inspecting {tfrecord_file.name}: {e}")
            return {'error': str(e)}
        
        return {
            'all_features': sorted(features_found),
            'sample_features': sample_features,
            'records_checked': record_count
        }
    
    def check_trigger_presence(self, tfrecord_file: Path, max_records: int = 100) -> Dict:
        """Check if triggers are present and their positions."""
        trigger_info = {
            'file': tfrecord_file.name,
            'has_trigger_feature': False,
            'trigger_values': [],
            'trigger_positions': [],  # For per-sample triggers
            'trigger_type': None,  # 'trial_level' or 'sample_level'
            'total_records': 0,
            'records_with_trigger': 0,
            'sample_indices': []
        }
        
        try:
            dataset = tf.data.TFRecordDataset(str(tfrecord_file))
            record_idx = 0
            
            for record in dataset:
                if record_idx >= max_records:
                    break
                
                example = tf.train.Example.FromString(record.numpy())
                features = example.features.feature
                
                trigger_info['total_records'] += 1
                
                # Check for trigger feature
                if 'trigger' in features:
                    trigger_info['has_trigger_feature'] = True
                    trigger_values = features['trigger'].int64_list.value
                    
                    if trigger_values:
                        trigger_info['records_with_trigger'] += 1
                        trigger_val = int(trigger_values[0])
                        trigger_info['trigger_values'].append(trigger_val)
                        trigger_info['trigger_positions'].append(record_idx)
                
                # Also check for sample_idx to understand positioning
                if 'sample_idx' in features:
                    sample_idx_values = features['sample_idx'].int64_list.value
                    if sample_idx_values:
                        trigger_info['sample_indices'].append(int(sample_idx_values[0]))
                
                record_idx += 1
                
        except Exception as e:
            trigger_info['error'] = str(e)
        
        # Determine trigger type
        if trigger_info['has_trigger_feature']:
            unique_triggers = len(set(trigger_info['trigger_values']))
            if unique_triggers == 1 and len(trigger_info['trigger_values']) == trigger_info['total_records']:
                # All records have the same trigger - likely trial-level
                trigger_info['trigger_type'] = 'trial_level'
            elif len(trigger_info['trigger_values']) < trigger_info['total_records']:
                # Some records have triggers - likely sample-level events
                trigger_info['trigger_type'] = 'sample_level'
            else:
                trigger_info['trigger_type'] = 'mixed'
        
        return trigger_info
    
    def inspect_das_dataset(self) -> Dict:
        """Inspect DAS dataset for triggers."""
        print("\n" + "="*80)
        print("INSPECTING DAS DATASET")
        print("="*80)
        
        if not self.das_tfrecord_dir.exists():
            print(f"ERROR: DAS TFRecord directory not found: {self.das_tfrecord_dir}")
            return {}
        
        # Find TFRecord files
        tfrecord_files = list(self.das_tfrecord_dir.glob("**/*.tfrecords"))
        if not tfrecord_files:
            tfrecord_files = list(self.das_tfrecord_dir.glob("**/*.tfrecord"))
        
        if not tfrecord_files:
            print(f"No TFRecord files found in {self.das_tfrecord_dir}")
            return {}
        
        print(f"Found {len(tfrecord_files)} TFRecord files")
        
        # Inspect first few files
        results = {}
        for tfrecord_file in tfrecord_files[:5]:  # Check first 5 files
            print(f"\n--- Inspecting {tfrecord_file.name} ---")
            
            # Check features
            feature_info = self.inspect_tfrecord_features(tfrecord_file, max_records=5)
            print(f"Features found: {feature_info.get('all_features', [])}")
            
            # Check for trigger
            trigger_info = self.check_trigger_presence(tfrecord_file, max_records=100)
            
            print(f"Has 'trigger' feature: {trigger_info['has_trigger_feature']}")
            if trigger_info['has_trigger_feature']:
                print(f"Trigger type: {trigger_info['trigger_type']}")
                print(f"Records with trigger: {trigger_info['records_with_trigger']}/{trigger_info['total_records']}")
                if trigger_info['trigger_values']:
                    unique_triggers = set(trigger_info['trigger_values'])
                    print(f"Unique trigger values: {sorted(unique_triggers)}")
                    print(f"Trigger value distribution: {Counter(trigger_info['trigger_values'])}")
                    if trigger_info['trigger_positions']:
                        print(f"Trigger positions (first 10): {trigger_info['trigger_positions'][:10]}")
            
            # Show sample features for first file
            if tfrecord_file == tfrecord_files[0] and 'sample_features' in feature_info:
                print(f"\nDetailed feature information (first record):")
                for feat_name, feat_info in feature_info['sample_features'].items():
                    print(f"  {feat_name}: {feat_info['type']}, count={feat_info['count']}")
                    if feat_name == 'trigger' and 'values' in feat_info:
                        print(f"    Values: {feat_info['values']}")
            
            results[tfrecord_file.name] = {
                'features': feature_info,
                'trigger_info': trigger_info
            }
        
        return results
    
    def inspect_fulsang_dataset(self) -> Dict:
        """Inspect Fulsang dataset for triggers."""
        print("\n" + "="*80)
        print("INSPECTING FULSANG DATASET")
        print("="*80)
        
        if not self.fulsang_tfrecord_dir.exists():
            print(f"ERROR: Fulsang TFRecord directory not found: {self.fulsang_tfrecord_dir}")
            return {}
        
        # Find TFRecord files
        tfrecord_files = list(self.fulsang_tfrecord_dir.glob("*.tfrecords"))
        if not tfrecord_files:
            tfrecord_files = list(self.fulsang_tfrecord_dir.glob("*.tfrecord"))
        
        if not tfrecord_files:
            print(f"No TFRecord files found in {self.fulsang_tfrecord_dir}")
            return {}
        
        print(f"Found {len(tfrecord_files)} TFRecord files")
        
        # Inspect first few files
        results = {}
        for tfrecord_file in tfrecord_files[:5]:  # Check first 5 files
            print(f"\n--- Inspecting {tfrecord_file.name} ---")
            
            # Check features
            feature_info = self.inspect_tfrecord_features(tfrecord_file, max_records=5)
            print(f"Features found: {feature_info.get('all_features', [])}")
            
            # Check for trigger
            trigger_info = self.check_trigger_presence(tfrecord_file, max_records=100)
            
            print(f"Has 'trigger' feature: {trigger_info['has_trigger_feature']}")
            if trigger_info['has_trigger_feature']:
                print(f"Trigger type: {trigger_info['trigger_type']}")
                print(f"Records with trigger: {trigger_info['records_with_trigger']}/{trigger_info['total_records']}")
                if trigger_info['trigger_values']:
                    unique_triggers = set(trigger_info['trigger_values'])
                    print(f"Unique trigger values: {sorted(unique_triggers)}")
                    print(f"Trigger value distribution: {Counter(trigger_info['trigger_values'])}")
                    if trigger_info['trigger_positions']:
                        print(f"Trigger positions (first 10): {trigger_info['trigger_positions'][:10]}")
            
            # Show sample features for first file
            if tfrecord_file == tfrecord_files[0] and 'sample_features' in feature_info:
                print(f"\nDetailed feature information (first record):")
                for feat_name, feat_info in feature_info['sample_features'].items():
                    print(f"  {feat_name}: {feat_info['type']}, count={feat_info['count']}")
                    if feat_name == 'trigger' and 'values' in feat_info:
                        print(f"    Values: {feat_info['values']}")
            
            results[tfrecord_file.name] = {
                'features': feature_info,
                'trigger_info': trigger_info
            }
        
        return results
    
    def check_raw_mat_files(self):
        """Check raw MATLAB files for trigger information."""
        print("\n" + "="*80)
        print("CHECKING RAW MATLAB FILES")
        print("="*80)
        
        try:
            import scipy.io as sio
            
            # Check DAS
            das_mat_file = Path("Data/Das/4004271/S1.mat")
            if das_mat_file.exists():
                print(f"\n--- DAS: {das_mat_file} ---")
                mat_data = sio.loadmat(str(das_mat_file), squeeze_me=True, struct_as_record=False)
                
                if 'trials' in mat_data:
                    trials = mat_data['trials']
                    if not isinstance(trials, np.ndarray):
                        trials = [trials]
                    else:
                        trials = trials.flatten()
                    
                    print(f"Found {len(trials)} trials")
                    
                    # Check first trial
                    if len(trials) > 0:
                        first_trial = trials[0]
                        print(f"First trial attributes: {[k for k in dir(first_trial) if not k.startswith('_')]}")
                        
                        # Check for trigger
                        has_trigger = hasattr(first_trial, 'trigger') or hasattr(first_trial, 'Trigger')
                        print(f"Has trigger attribute: {has_trigger}")
                        
                        if hasattr(first_trial, 'attended_ear'):
                            print(f"Has attended_ear: {first_trial.attended_ear}")
                        
                        # Check RawData
                        if hasattr(first_trial, 'RawData'):
                            rawdata = first_trial.RawData
                            rawdata_attrs = [k for k in dir(rawdata) if not k.startswith('_')]
                            print(f"RawData attributes: {rawdata_attrs}")
                            
                            if hasattr(rawdata, 'trigger') or hasattr(rawdata, 'Trigger'):
                                print("RawData has trigger field")
            
            # Check Fulsang
            fulsang_mat_file = Path("Data/Fulsang/DATA_preproc/S1_data_preproc.mat")
            if fulsang_mat_file.exists():
                print(f"\n--- Fulsang: {fulsang_mat_file} ---")
                mat_data = sio.loadmat(str(fulsang_mat_file), squeeze_me=False, struct_as_record=False)
                
                if 'data' in mat_data:
                    data_struct = mat_data['data']
                    if isinstance(data_struct, np.ndarray) and data_struct.size > 0:
                        first_elem = data_struct.flat[0]
                        print(f"First element attributes: {[k for k in dir(first_elem) if not k.startswith('_')]}")
                        
                        # Check for event structure
                        if hasattr(first_elem, 'event'):
                            event = first_elem.event
                            print(f"Has event structure: {event is not None}")
                            if isinstance(event, np.ndarray) and event.size > 0:
                                first_event = event.flat[0]
                                if hasattr(first_event, 'eeg'):
                                    eeg_events = first_event.eeg
                                    if isinstance(eeg_events, np.ndarray) and eeg_events.size > 0:
                                        print(f"Found {eeg_events.size} EEG events")
                                        # Check first event
                                        first_eeg_event = eeg_events.flat[0]
                                        if hasattr(first_eeg_event, 'sample') and hasattr(first_eeg_event, 'value'):
                                            sample = first_eeg_event.sample
                                            value = first_eeg_event.value
                                            print(f"First event: sample={sample}, value={value}")
                
                # Check expinfo
                if 'expinfo' in mat_data:
                    expinfo = mat_data['expinfo']
                    if hasattr(expinfo, 'dtype') and expinfo.dtype.names:
                        print(f"expinfo fields: {expinfo.dtype.names}")
                        if 'trigger' in expinfo.dtype.names:
                            trigger_val = expinfo['trigger']
                            print(f"expinfo.trigger exists: {trigger_val}")
        
        except ImportError:
            print("scipy.io not available, skipping MATLAB file check")
        except Exception as e:
            print(f"Error checking MATLAB files: {e}")


def main():
    inspector = TriggerInspector()
    
    # Check raw MATLAB files first
    inspector.check_raw_mat_files()
    
    # Inspect TFRecord files
    das_results = inspector.inspect_das_dataset()
    fulsang_results = inspector.inspect_fulsang_dataset()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\nDAS Dataset:")
    das_has_triggers = any(r.get('trigger_info', {}).get('has_trigger_feature', False) 
                          for r in das_results.values())
    print(f"  Triggers present in TFRecords: {das_has_triggers}")
    if das_has_triggers:
        for file_name, result in das_results.items():
            trigger_info = result.get('trigger_info', {})
            if trigger_info.get('has_trigger_feature'):
                print(f"    {file_name}: {trigger_info.get('trigger_type', 'unknown')} triggers")
                if trigger_info.get('trigger_values'):
                    unique = set(trigger_info['trigger_values'])
                    print(f"      Unique values: {sorted(unique)}")
    
    print("\nFulsang Dataset:")
    fulsang_has_triggers = any(r.get('trigger_info', {}).get('has_trigger_feature', False) 
                               for r in fulsang_results.values())
    print(f"  Triggers present in TFRecords: {fulsang_has_triggers}")
    if fulsang_has_triggers:
        for file_name, result in fulsang_results.items():
            trigger_info = result.get('trigger_info', {})
            if trigger_info.get('has_trigger_feature'):
                print(f"    {file_name}: {trigger_info.get('trigger_type', 'unknown')} triggers")
                if trigger_info.get('trigger_values'):
                    unique = set(trigger_info['trigger_values'])
                    print(f"      Unique values: {sorted(unique)}")


if __name__ == '__main__':
    main()
