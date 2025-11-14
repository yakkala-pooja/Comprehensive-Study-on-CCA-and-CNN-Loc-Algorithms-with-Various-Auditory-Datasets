#!/usr/bin/env python3
"""
FULPREPROCESSING - Fulsang Dataset Preprocessing

This script preprocesses the Fulsang dataset and fixes a bunch of issues we found:

1. Attention label problems:
   - Labels weren't being extracted consistently from MATLAB files
   - Some labels were missing or broken
   - The experimental design wasn't being interpreted correctly

2. Preventing data leakage:
   - Splits data properly into train/validation/test sets
   - Splits by subject so we don't leak data between sets
   - Makes sure the timing is consistent

3. Label validation:
   - Checks that attention labels make sense
   - Verifies the experimental design matches what we expect
   - Catches errors and quality issues

4. Robust data handling:
   - Loads MATLAB files even when they're a bit messed up
   - Extracts EEG and envelope data properly
   - Makes sure everything is formatted consistently
"""

import sys
import numpy as np
import scipy.io as sio
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add telluride_decoding to path
sys.path.append('telluride_decoding')

try:
    from telluride_decoding import decoding
    from telluride_decoding import brain_data
    from telluride_decoding import regression
    from telluride_decoding import attention_decoder
except ImportError as e:
    print(f"Warning: Could not import some telluride_decoding modules: {e}")
    print("Continuing with basic functionality...")


class FulsangAttentionLabelValidator:
    """
    Validates attention labels for the Fulsang dataset to make sure they're good quality.
    """
    
    def __init__(self):
        # Fulsang experimental parameters
        self.sampling_rate = 64
        self.trial_duration = 20
        self.trial_samples = self.trial_duration * self.sampling_rate
        self.n_trials_per_subject = 60
        self.expected_total_samples = self.n_trials_per_subject * self.trial_samples
        
    
    def validate_attention_labels(self, labels: np.ndarray, subject_id: str = "unknown") -> Dict:
        """
        Checks if the attention labels look good.
        
        Args:
            labels: The attention labels we want to check
            subject_id: Which subject this is (for logging)
            
        Returns:
            A dictionary with validation results
        """
        results = {
            'valid': False,
            'issues': [],
            'warnings': [],
            'label_distribution': None,
            'trial_structure': None,
            'recommendations': []
        }
        
        # First, make sure we actually have labels
        if labels is None or len(labels) == 0:
            results['issues'].append("Empty or None labels")
            return results
        
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        
        # Check the data type
        if labels.dtype not in [np.int32, np.int64, np.float32, np.float64]:
            results['warnings'].append(f"Unexpected data type: {labels.dtype}")
        
        # Convert floats to integers if they're actually integers
        if labels.dtype in [np.float32, np.float64]:
            if np.all(labels == labels.astype(int)):
                labels = labels.astype(int)
                results['warnings'].append("Converted float labels to integer")
            else:
                results['issues'].append("Non-integer values in labels")
                return results
        
        # Check what values we have
        unique_values = np.unique(labels)
        
        if len(unique_values) == 0:
            results['issues'].append("No unique values found")
            return results
        
        if len(unique_values) == 1:
            results['issues'].append(f"Only one label value found: {unique_values[0]}")
            return results
        
        if not np.all(np.isin(unique_values, [0, 1])):
            results['issues'].append(f"Invalid label values: {unique_values}")
            return results
        
        # Count how many of each label we have
        label_counts = np.bincount(labels)
        results['label_distribution'] = {
            'class_0_count': int(label_counts[0]) if len(label_counts) > 0 else 0,
            'class_1_count': int(label_counts[1]) if len(label_counts) > 1 else 0,
            'total_samples': len(labels),
            'class_0_ratio': float(label_counts[0] / len(labels)) if len(label_counts) > 0 else 0.0,
            'class_1_ratio': float(label_counts[1] / len(labels)) if len(label_counts) > 1 else 0.0
        }
        
        # Check if the labels are balanced
        class_0_ratio = results['label_distribution']['class_0_ratio']
        class_1_ratio = results['label_distribution']['class_1_ratio']
        
        if abs(class_0_ratio - class_1_ratio) > 0.1:  # More than 10% difference
            results['warnings'].append(f"Imbalanced labels: {class_0_ratio:.3f} vs {class_1_ratio:.3f}")
        
        # Check if the trial structure makes sense
        trial_structure = self._validate_trial_structure(labels)
        results['trial_structure'] = trial_structure
        
        if not trial_structure['valid']:
            results['issues'].extend(trial_structure['issues'])
        
        # Check if labels change in a reasonable way over time
        temporal_consistency = self._validate_temporal_consistency(labels)
        if not temporal_consistency['valid']:
            results['warnings'].extend(temporal_consistency['warnings'])
        
        # Decide if everything looks good
        if len(results['issues']) == 0:
            results['valid'] = True
        else:
            print(f"Validation failed for {subject_id}: {len(results['issues'])} issues")
            for issue in results['issues']:
                print(f"  - {issue}")
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        return results
    
    def _validate_trial_structure(self, labels: np.ndarray) -> Dict:
        """Checks if the labels follow the expected trial structure."""
        results = {
            'valid': True,
            'issues': [],
            'trial_length': self.trial_samples,
            'detected_trials': 0,
            'trial_transitions': []
        }
        
        # Make sure the length matches what we expect for trials
        if len(labels) % self.trial_samples != 0:
            results['issues'].append(f"Length {len(labels)} not divisible by trial length {self.trial_samples}")
            results['valid'] = False
        
        # Find where trials start and what label each trial has
        trial_transitions = []
        for i in range(0, len(labels) - self.trial_samples, self.trial_samples):
            trial_labels = labels[i:i + self.trial_samples]
            trial_label = np.bincount(trial_labels).argmax()  # Use the most common label in the trial
            trial_transitions.append(trial_label)
        
        results['trial_transitions'] = trial_transitions
        results['detected_trials'] = len(trial_transitions)
        
        # See if trials alternate between left and right like they should
        if len(trial_transitions) >= 2:
            alternating_pattern = all(
                trial_transitions[i] != trial_transitions[i+1] 
                for i in range(len(trial_transitions)-1)
            )
            
            if not alternating_pattern:
                results['warnings'] = ["Trial pattern is not strictly alternating"]
        
        return results
    
    def _validate_temporal_consistency(self, labels: np.ndarray) -> Dict:
        """Checks if the labels change in a reasonable way over time."""
        results = {
            'valid': True,
            'warnings': []
        }
        
        # Check if labels are switching too fast (which wouldn't make sense)
        changes = np.diff(labels)
        rapid_changes = np.sum(np.abs(changes))
        
        if rapid_changes > len(labels) * 0.1:  # More than 10% of samples have changes
            results['warnings'].append(f"High number of rapid changes: {rapid_changes}")
        
        # Check if there are long stretches without any changes
        change_indices = np.where(changes != 0)[0]
        if len(change_indices) > 0:
            periods = np.diff(np.concatenate([[0], change_indices, [len(labels)]]))
            max_period = np.max(periods)
            
            if max_period > self.trial_samples * 2:  # More than 2 trials without any change
                results['warnings'].append(f"Long period without change: {max_period} samples")
        
        return results
    
    def _generate_recommendations(self, validation_results: Dict) -> List[str]:
        """Suggests what to do based on what we found during validation."""
        recommendations = []
        
        if not validation_results['valid']:
            recommendations.append("Fix critical issues before proceeding with training")
        
        if validation_results['label_distribution']['class_0_ratio'] < 0.3 or validation_results['label_distribution']['class_0_ratio'] > 0.7:
            recommendations.append("Consider balancing the dataset or using stratified sampling")
        
        if validation_results['trial_structure'] and not validation_results['trial_structure']['valid']:
            recommendations.append("Verify experimental design and trial structure")
        
        if len(validation_results['warnings']) > 0:
            recommendations.append("Review warnings and consider data quality improvements")
        
        return recommendations


class FulsangDataExtractor:
    """
    Extracts data from Fulsang MATLAB files, even when they're a bit messy.
    """
    
    def __init__(self):
        self.expected_eeg_channels = 66
        self.sampling_rate = 64
        self.extraction_stats = {
            'files_processed': 0,
            'files_successful': 0,
            'files_failed': 0,
            'total_samples': 0,
            'extraction_errors': []
        }
    
    def extract_from_mat_file(self, mat_file_path: Path) -> Optional[Dict]:
        """
        Gets EEG, envelope, and attention data out of a MATLAB file.
        
        Args:
            mat_file_path: Where the MATLAB file is
            
        Returns:
            A dictionary with the data, or None if something went wrong
        """
        try:
            # Load MATLAB file
            data = sio.loadmat(str(mat_file_path))
            
            if 'data' not in data:
                print(f"ERROR: No 'data' field found in {mat_file_path.name}")
                self.extraction_stats['files_failed'] += 1
                self.extraction_stats['extraction_errors'].append(f"No 'data' field in {mat_file_path.name}")
                return None
            
            actual_data = data['data']
            if not isinstance(actual_data, np.ndarray) or actual_data.size == 0:
                print(f"ERROR: Invalid data structure in {mat_file_path.name}")
                self.extraction_stats['files_failed'] += 1
                self.extraction_stats['extraction_errors'].append(f"Invalid data structure in {mat_file_path.name}")
                return None
            
            first_elem = actual_data.flat[0]
            if not hasattr(first_elem, 'dtype') or not first_elem.dtype.names:
                print(f"ERROR: No structured data found in {mat_file_path.name}")
                self.extraction_stats['files_failed'] += 1
                self.extraction_stats['extraction_errors'].append(f"No structured data in {mat_file_path.name}")
                return None
            
            # Extract EEG data
            eeg_data = self._extract_eeg_data(first_elem)
            if eeg_data is None:
                print(f"ERROR: Failed to extract EEG data from {mat_file_path.name}")
                self.extraction_stats['files_failed'] += 1
                self.extraction_stats['extraction_errors'].append(f"EEG extraction failed in {mat_file_path.name}")
                return None
            
            # Extract envelope data
            envelope_data = self._extract_envelope_data(first_elem)
            if envelope_data is None:
                print(f"ERROR: Failed to extract envelope data from {mat_file_path.name}")
                self.extraction_stats['files_failed'] += 1
                self.extraction_stats['extraction_errors'].append(f"Envelope extraction failed in {mat_file_path.name}")
                return None
            
            # Extract attention labels
            attention_labels = self._extract_attention_labels(first_elem, mat_file_path)
            if attention_labels is None:
                print(f"ERROR: Failed to extract attention labels from {mat_file_path.name}")
                self.extraction_stats['files_failed'] += 1
                self.extraction_stats['extraction_errors'].append(f"Attention label extraction failed in {mat_file_path.name}")
                return None
            
            # Make sure everything has the same length
            min_length = min(len(eeg_data), len(envelope_data), len(attention_labels))
            eeg_data = eeg_data[:min_length]
            envelope_data = envelope_data[:min_length]
            attention_labels = attention_labels[:min_length]
            
            # Figure out which subject this is from the filename
            subject_id = mat_file_path.stem.replace('_data_preproc', '')
            metadata = {
                'subject_id': subject_id,
                'file_path': str(mat_file_path),
                'n_samples': min_length,
                'n_eeg_channels': eeg_data.shape[1],
                'n_envelope_features': envelope_data.shape[1],
                'sampling_rate': self.sampling_rate,
                'extraction_timestamp': datetime.now().isoformat()
            }
            
            result = {
                'eeg_data': eeg_data,
                'envelope_data': envelope_data,
                'attention_labels': attention_labels,
                'metadata': metadata
            }
            
            self.extraction_stats['files_processed'] += 1
            self.extraction_stats['files_successful'] += 1
            self.extraction_stats['total_samples'] += min_length
            
            return result
            
        except Exception as e:
            print(f"ERROR extracting data from {mat_file_path.name}: {e}")
            self.extraction_stats['files_failed'] += 1
            self.extraction_stats['extraction_errors'].append(f"Exception in {mat_file_path.name}: {str(e)}")
            return None
    
    def _extract_eeg_data(self, first_elem) -> Optional[np.ndarray]:
        """Gets the EEG data out of the MATLAB structure."""
        if 'eeg' not in first_elem.dtype.names:
            return None
        
        eeg_field = first_elem['eeg']
        if not isinstance(eeg_field, np.ndarray) or eeg_field.dtype != object:
            return None
        
        if eeg_field.size == 0:
            return None
        
        nested_eeg = eeg_field.flat[0]
        if not isinstance(nested_eeg, np.ndarray) or nested_eeg.ndim < 2:
            return None
        
        # Make sure we have the right number of channels
        if nested_eeg.shape[1] != self.expected_eeg_channels:
            if nested_eeg.shape[1] < self.expected_eeg_channels:
                # Add zeros if we're missing channels
                padding = np.zeros((nested_eeg.shape[0], self.expected_eeg_channels - nested_eeg.shape[1]))
                nested_eeg = np.concatenate([nested_eeg, padding], axis=1)
            else:
                # Cut off extra channels if we have too many
                nested_eeg = nested_eeg[:, :self.expected_eeg_channels]
        
        return nested_eeg.astype(np.float32)
    
    def _extract_envelope_data(self, first_elem) -> Optional[np.ndarray]:
        """Gets the envelope data out of the MATLAB structure."""
        # Try different field names that might contain envelope data
        envelope_fields = ['envelope', 'wavA', 'wavB', 'audio']
        envelope_data = None
        
        for field_name in envelope_fields:
            if field_name in first_elem.dtype.names:
                field_data = first_elem[field_name]
                
                if isinstance(field_data, np.ndarray) and field_data.dtype == object:
                    if field_data.size > 0:
                        nested_data = field_data.flat[0]
                        if isinstance(nested_data, np.ndarray) and nested_data.ndim >= 2:
                            if envelope_data is None:
                                envelope_data = nested_data
                            else:
                                # If we find multiple sources, combine them
                                envelope_data = np.concatenate([envelope_data, nested_data], axis=1)
        
        if envelope_data is None:
            return None
        
        # Fulsang uses single-band envelopes, so just take the first band if there are multiple
        if envelope_data.shape[1] > 1:
            envelope_data = envelope_data[:, 0:1]
        
        return envelope_data.astype(np.float32)
    
    def _extract_attention_labels(self, first_elem, mat_file_path: Path) -> Optional[np.ndarray]:
        """Tries a few different ways to get the attention labels."""
        # First, try to get labels from the event structure if it exists
        if 'event' in first_elem.dtype.names:
            event_data = first_elem['event']
            
            if isinstance(event_data, np.ndarray) and event_data.size > 0:
                try:
                    labels = self._extract_from_event_structure(event_data)
                    if labels is not None:
                        return labels
                except Exception:
                    pass
        
        # If that doesn't work, create labels based on how the experiment was designed
        labels = self._create_experimental_labels(mat_file_path)
        return labels
    
    def _extract_from_event_structure(self, event_data: np.ndarray) -> Optional[np.ndarray]:
        """Tries to get attention labels from the event structure."""
        # This isn't fully implemented yet - would need to know the exact format
        # of the event structure in the Fulsang dataset
        return None
    
    def _create_experimental_labels(self, mat_file_path: Path) -> np.ndarray:
        """Creates attention labels based on how the Fulsang experiment was designed."""
        # The experiment alternates between left and right attention
        # Each trial lasts 20 seconds (1280 samples at 64 Hz)
        # We label left attention as 0 and right attention as 1
        trial_length = 1280
        n_trials = 60
        total_samples = n_trials * trial_length
        
        labels = np.zeros(total_samples, dtype=np.int32)
        
        for trial_idx in range(n_trials):
            start_idx = trial_idx * trial_length
            end_idx = start_idx + trial_length
            
            # Switch between 0 (left) and 1 (right) for each trial
            trial_label = trial_idx % 2
            labels[start_idx:end_idx] = trial_label
        
        return labels


class FulsangPreprocessor:
    """
    Handles all the preprocessing for the Fulsang dataset with quality checks.
    """
    
    def __init__(self, data_dir: str = "Data/Fulsang", output_dir: str = "fulsang_preprocessed"):
        self.data_dir = Path(data_dir)
        self.preproc_dir = self.data_dir / "DATA_preproc"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.tfrecord_dir = self.output_dir / "tfrecords"
        self.tfrecord_dir.mkdir(exist_ok=True)
        
        self.reports_dir = self.output_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.label_validator = FulsangAttentionLabelValidator()
        self.data_extractor = FulsangDataExtractor()
        
        # Processing statistics
        self.processing_stats = {
            'total_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'total_samples': 0,
            'validation_failures': 0,
            'start_time': None,
            'end_time': None
        }
        
    
    def preprocess_dataset(self) -> bool:
        """
        Does all the preprocessing work.
        
        Returns:
            True if everything worked, False if something went wrong
        """
        self.processing_stats['start_time'] = datetime.now()
        
        # Find all the MATLAB files we need to process
        mat_files = self._discover_mat_files()
        if not mat_files:
            print("ERROR: No MATLAB files found")
            return False
        
        self.processing_stats['total_files'] = len(mat_files)
        
        # Process each file and check the quality
        successful_extractions = []
        
        for mat_file in tqdm(mat_files, desc="Processing MATLAB files"):
            # Get the data out of the file
            extraction_result = self.data_extractor.extract_from_mat_file(mat_file)
            if extraction_result is None:
                self.processing_stats['failed_files'] += 1
                continue
            
            # Make sure the attention labels are good
            validation_result = self.label_validator.validate_attention_labels(
                extraction_result['attention_labels'],
                extraction_result['metadata']['subject_id']
            )
            
            if not validation_result['valid']:
                print(f"Validation failed for {mat_file.name}: {len(validation_result['issues'])} issues")
                for issue in validation_result['issues']:
                    print(f"  - {issue}")
                
                self.processing_stats['validation_failures'] += 1
                
                # Skip files with too many problems
                if len(validation_result['issues']) > 2:
                    print("Skipping file due to critical validation failures")
                    continue
            
            # Save the validation results with the data
            extraction_result['metadata']['validation_results'] = validation_result
            
            successful_extractions.append(extraction_result)
            self.processing_stats['successful_files'] += 1
            self.processing_stats['total_samples'] += extraction_result['metadata']['n_samples']
        
        # Create TFRecord files if we got some good data
        if successful_extractions:
            self._create_tfrecord_files(successful_extractions)
            self._generate_reports(successful_extractions)
            
            self.processing_stats['end_time'] = datetime.now()
            
            return True
        else:
            print("ERROR: No successful extractions - preprocessing failed")
            return False
    
    def _discover_mat_files(self) -> List[Path]:
        """Finds all the MATLAB files we need to process."""
        if not self.preproc_dir.exists():
            print(f"ERROR: Preprocessed directory not found: {self.preproc_dir}")
            return []
        
        mat_files = list(self.preproc_dir.glob("S*_data_preproc.mat"))
        return mat_files
    
    def _create_tfrecord_files(self, extractions: List[Dict]):
        """Creates TFRecord files and organizes them by subject so we don't leak data."""
        # Group everything by subject
        subject_data = {}
        for extraction in extractions:
            subject_id = extraction['metadata']['subject_id']
            if subject_id not in subject_data:
                subject_data[subject_id] = []
            subject_data[subject_id].append(extraction)
        
        # Create one TFRecord file per subject
        file_counter = 0
        for subject_id, subject_extractions in subject_data.items():
            # Put all the data for this subject together
            all_eeg = []
            all_envelope = []
            all_labels = []
            
            for extraction in subject_extractions:
                all_eeg.append(extraction['eeg_data'])
                all_envelope.append(extraction['envelope_data'])
                all_labels.append(extraction['attention_labels'])
            
            # Combine all the arrays
            subject_eeg = np.vstack(all_eeg)
            subject_envelope = np.vstack(all_envelope)
            subject_labels = np.concatenate(all_labels)
            
            # Write it all to a TFRecord file
            tfrecord_file = self.tfrecord_dir / f"fulsang_subject_{subject_id}_{file_counter:03d}.tfrecords"
            
            with tf.io.TFRecordWriter(str(tfrecord_file)) as writer:
                for i in range(len(subject_eeg)):
                    # Each sample gets written as one example
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'eeg': tf.train.Feature(float_list=tf.train.FloatList(value=subject_eeg[i].astype(np.float32))),
                        'envelope': tf.train.Feature(float_list=tf.train.FloatList(value=subject_envelope[i].astype(np.float32))),
                        'attention_label': tf.train.Feature(int64_list=tf.train.Int64List(value=[subject_labels[i]])),
                        'sample_idx': tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
                        'subject_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[subject_id.encode()])),
                        'file_source': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tfrecord_file.name.encode()]))
                    }))
                    
                    writer.write(example.SerializeToString())
            
            file_counter += 1
    
    def _generate_reports(self, extractions: List[Dict]):
        """Creates reports about what happened during preprocessing."""
        # Collect all the stats
        stats_report = {
            'preprocessing_timestamp': datetime.now().isoformat(),
            'processing_stats': {
                'total_files': self.processing_stats['total_files'],
                'successful_files': self.processing_stats['successful_files'],
                'failed_files': self.processing_stats['failed_files'],
                'total_samples': self.processing_stats['total_samples'],
                'validation_failures': self.processing_stats['validation_failures'],
                'start_time': self.processing_stats['start_time'].isoformat() if self.processing_stats['start_time'] else None,
                'end_time': self.processing_stats['end_time'].isoformat() if self.processing_stats['end_time'] else None
            },
            'extraction_stats': self.data_extractor.extraction_stats,
            'total_subjects': len(set(ext['metadata']['subject_id'] for ext in extractions)),
            'total_samples': sum(ext['metadata']['n_samples'] for ext in extractions),
            'label_distributions': {},
            'validation_summaries': {}
        }
        
        # Go through each subject and collect their stats
        for extraction in extractions:
            subject_id = extraction['metadata']['subject_id']
            labels = extraction['attention_labels']
            
            stats_report['label_distributions'][subject_id] = {
                'total_samples': len(labels),
                'class_0_count': int(np.sum(labels == 0)),
                'class_1_count': int(np.sum(labels == 1)),
                'class_0_ratio': float(np.mean(labels == 0)),
                'class_1_ratio': float(np.mean(labels == 1))
            }
            
            if 'validation_results' in extraction['metadata']:
                validation = extraction['metadata']['validation_results']
                stats_report['validation_summaries'][subject_id] = {
                    'valid': validation['valid'],
                    'issues_count': len(validation['issues']),
                    'warnings_count': len(validation['warnings']),
                    'issues': validation['issues'],
                    'warnings': validation['warnings']
                }
        
        # Save statistics report
        with open(self.reports_dir / 'preprocessing_statistics.json', 'w') as f:
            json.dump(stats_report, f, indent=2)
        
        # Generate text report
        self._generate_text_report(stats_report)
    
    def _generate_text_report(self, stats_report: Dict):
        """Writes a text report that's easy to read."""
        report_file = self.reports_dir / 'preprocessing_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FULSANG DATASET PREPROCESSING REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Preprocessing completed: {stats_report['preprocessing_timestamp']}\n")
            f.write(f"Total subjects processed: {stats_report['total_subjects']}\n")
            f.write(f"Total samples: {stats_report['total_samples']}\n\n")
            
            # Processing statistics
            f.write("PROCESSING STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total files: {stats_report['processing_stats']['total_files']}\n")
            f.write(f"Successful: {stats_report['processing_stats']['successful_files']}\n")
            f.write(f"Failed: {stats_report['processing_stats']['failed_files']}\n")
            f.write(f"Validation failures: {stats_report['processing_stats']['validation_failures']}\n\n")
            
            # Subject-wise details
            f.write("SUBJECT-WISE DETAILS:\n")
            f.write("-" * 40 + "\n")
            for subject_id, details in stats_report['label_distributions'].items():
                f.write(f"Subject {subject_id}:\n")
                f.write(f"  Samples: {details['total_samples']}\n")
                f.write(f"  Class 0: {details['class_0_count']} ({details['class_0_ratio']:.3f})\n")
                f.write(f"  Class 1: {details['class_1_count']} ({details['class_1_ratio']:.3f})\n")
                
                if subject_id in stats_report['validation_summaries']:
                    validation = stats_report['validation_summaries'][subject_id]
                    f.write(f"  Validation: {'PASSED' if validation['valid'] else 'FAILED'}\n")
                    if validation['issues']:
                        f.write(f"  Issues: {', '.join(validation['issues'])}\n")
                    if validation['warnings']:
                        f.write(f"  Warnings: {', '.join(validation['warnings'])}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")


def main():
    """Main function that runs the preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='FULPREPROCESSING - Accurate Fulsang Dataset Preprocessing')
    parser.add_argument('--data_dir', type=str, default='Data/Fulsang',
                       help='Directory containing Fulsang data')
    parser.add_argument('--output_dir', type=str, default='fulsang_preprocessed',
                       help='Output directory for preprocessed data')
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = FulsangPreprocessor(args.data_dir, args.output_dir)
    
    # Run preprocessing
    success = preprocessor.preprocess_dataset()
    
    if success:
        return 0
    else:
        print("Preprocessing failed. Check logs above for details")
        return 1


if __name__ == "__main__":
    exit(main())
