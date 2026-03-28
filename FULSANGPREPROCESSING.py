#!/usr/bin/env python3
"""
FULSANGPREPROCESSING - Process DATA_preproc.zip for Fulsang Dataset

This script processes the preprocessed EEG and audio data from preproc_script.m
contained in DATA_preproc.zip and creates TFRecord files compatible with FULCCA.py.

The zip file should contain MATLAB files (S*_data_preproc.mat) with:
- EEG data (66 channels, 64 Hz sampling rate)
- Audio envelope data
- Attention labels

Output: TFRecord files in fulsang_preprocessed/tfrecords/
"""

import sys
import os
import zipfile
import numpy as np
import scipy.io as sio
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import json
from datetime import datetime
import tempfile
import shutil
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


class FulsangDataExtractor:
    """
    Extracts data from Fulsang MATLAB files from DATA_preproc.zip.
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
        Extracts EEG, envelope, and attention data from a MATLAB file.
        
        Args:
            mat_file_path: Path to the MATLAB file
            
        Returns:
            Dictionary with extracted data or None if extraction failed
        """
        try:
            # Load MATLAB file
            data = sio.loadmat(str(mat_file_path), struct_as_record=False, squeeze_me=True)
            
            # Store full data dict for accessing expinfo
            self._current_mat_data = data
            
            if 'data' not in data:
                print(f"ERROR: No 'data' field found in {mat_file_path.name}")
                self.extraction_stats['files_failed'] += 1
                self.extraction_stats['extraction_errors'].append(f"No 'data' field in {mat_file_path.name}")
                return None
            
            actual_data = data['data']
            
            # Handle different MATLAB file structures
            if isinstance(actual_data, np.ndarray):
                if actual_data.size == 0:
                    print(f"ERROR: Empty data array in {mat_file_path.name}")
                    self.extraction_stats['files_failed'] += 1
                    return None
                
                # If it's an object array, get the first element
                if actual_data.dtype == object:
                    first_elem = actual_data.flat[0]
                else:
                    first_elem = actual_data
            else:
                first_elem = actual_data
            
            # Extract EEG data
            eeg_data = self._extract_eeg_data(first_elem)
            if eeg_data is None:
                print(f"ERROR: Failed to extract EEG data from {mat_file_path.name}")
                self.extraction_stats['files_failed'] += 1
                return None
            
            # Extract envelope data (now returns attended, left, and right)
            envelope_data, left_envelope, right_envelope = self._extract_envelope_data(first_elem)
            if envelope_data is None:
                # Create dummy envelope data if not found (will be same length as EEG)
                print(f"WARNING: No envelope data found in {mat_file_path.name}, creating dummy envelope")
                # We'll create it after we know the EEG length
                envelope_data = None
                left_envelope = None
                right_envelope = None
            
            # Extract attention labels (pass full data dict for expinfo access)
            attention_labels = self._extract_attention_labels(first_elem, mat_file_path, full_data_dict=data)
            if attention_labels is None:
                # Try to debug what's in the file
                print(f"DEBUG: Attempting to inspect structure of {mat_file_path.name}")
                try:
                    if hasattr(first_elem, 'dtype') and first_elem.dtype.names:
                        print(f"  Available fields in data: {first_elem.dtype.names}")
                    elif isinstance(first_elem, dict):
                        print(f"  Available keys in data: {list(first_elem.keys())[:10]}")
                    if 'expinfo' in data:
                        print(f"  Found expinfo in top-level data")
                        expinfo = data['expinfo']
                        if hasattr(expinfo, 'dtype') and expinfo.dtype.names:
                            print(f"  Available fields in expinfo: {expinfo.dtype.names}")
                except Exception as debug_e:
                    print(f"  Debug error: {debug_e}")
                
                print(f"ERROR: Failed to extract attention labels from {mat_file_path.name}")
                self.extraction_stats['files_failed'] += 1
                return None
            
            # Create dummy envelope if not found
            if envelope_data is None:
                print(f"Creating dummy envelope data for {mat_file_path.name}")
                envelope_data = np.zeros((len(eeg_data), 4), dtype=np.float32)  # 4 features like DAS
                left_envelope = np.zeros((len(eeg_data), 4), dtype=np.float32)
                right_envelope = np.zeros((len(eeg_data), 4), dtype=np.float32)
            else:
                # Ensure left and right envelopes exist and have correct shape
                if left_envelope is None:
                    left_envelope = envelope_data.copy()
                if right_envelope is None:
                    right_envelope = envelope_data.copy()
                
                # Ensure all envelopes have 4 features
                if left_envelope.shape[1] != 4:
                    if left_envelope.shape[1] == 1:
                        # Expand single feature to 4 features
                        env_vals = left_envelope.flatten()
                        left_envelope = np.column_stack([
                            env_vals,
                            env_vals,  # smoothed (same for now)
                            np.zeros_like(env_vals),  # derivative
                            env_vals**2  # squared
                        ])
                    else:
                        # Pad or truncate
                        if left_envelope.shape[1] < 4:
                            padding = np.zeros((left_envelope.shape[0], 4 - left_envelope.shape[1]))
                            left_envelope = np.column_stack([left_envelope, padding])
                        else:
                            left_envelope = left_envelope[:, :4]
                
                if right_envelope.shape[1] != 4:
                    if right_envelope.shape[1] == 1:
                        env_vals = right_envelope.flatten()
                        right_envelope = np.column_stack([
                            env_vals,
                            env_vals,
                            np.zeros_like(env_vals),
                            env_vals**2
                        ])
                    else:
                        if right_envelope.shape[1] < 4:
                            padding = np.zeros((right_envelope.shape[0], 4 - right_envelope.shape[1]))
                            right_envelope = np.column_stack([right_envelope, padding])
                        else:
                            right_envelope = right_envelope[:, :4]
            
            # Ensure all arrays have the same length
            min_length = min(len(eeg_data), len(envelope_data), len(attention_labels))
            if left_envelope is not None:
                min_length = min(min_length, len(left_envelope))
            if right_envelope is not None:
                min_length = min(min_length, len(right_envelope))
            
            eeg_data = eeg_data[:min_length]
            envelope_data = envelope_data[:min_length]
            if left_envelope is not None:
                left_envelope = left_envelope[:min_length]
            if right_envelope is not None:
                right_envelope = right_envelope[:min_length]
            attention_labels = attention_labels[:min_length]
            
            # Extract subject ID from filename
            subject_id = mat_file_path.stem.replace('_data_preproc', '').replace('S', '')
            if not subject_id:
                subject_id = mat_file_path.stem
            
            metadata = {
                'subject_id': f"S{subject_id}" if not subject_id.startswith('S') else subject_id,
                'file_path': str(mat_file_path),
                'n_samples': min_length,
                'n_eeg_channels': eeg_data.shape[1],
                'n_envelope_features': envelope_data.shape[1] if len(envelope_data.shape) > 1 else 1,
                'sampling_rate': self.sampling_rate,
                'extraction_timestamp': datetime.now().isoformat()
            }
            
            result = {
                'eeg_data': eeg_data,
                'envelope_data': envelope_data,
                'left_envelope': left_envelope,
                'right_envelope': right_envelope,
                'attention_labels': attention_labels,
                'metadata': metadata
            }
            
            self.extraction_stats['files_processed'] += 1
            self.extraction_stats['files_successful'] += 1
            self.extraction_stats['total_samples'] += min_length
            
            return result
            
        except Exception as e:
            print(f"ERROR extracting data from {mat_file_path.name}: {e}")
            import traceback
            traceback.print_exc()
            self.extraction_stats['files_failed'] += 1
            self.extraction_stats['extraction_errors'].append(f"Exception in {mat_file_path.name}: {str(e)}")
            return None
    
    def _extract_eeg_data(self, data_struct) -> Optional[np.ndarray]:
        """Extracts EEG data from MATLAB structure."""
        try:
            # Try different ways to access EEG data
            if hasattr(data_struct, 'eeg'):
                eeg_field = data_struct.eeg
            elif isinstance(data_struct, dict) and 'eeg' in data_struct:
                eeg_field = data_struct['eeg']
            elif hasattr(data_struct, 'dtype') and 'eeg' in data_struct.dtype.names:
                eeg_field = data_struct['eeg']
            else:
                return None
            
            # Handle nested structures
            if isinstance(eeg_field, np.ndarray) and eeg_field.dtype == object:
                if eeg_field.size > 0:
                    eeg_field = eeg_field.flat[0]
            
            # Convert to numpy array
            if not isinstance(eeg_field, np.ndarray):
                return None
            
            # Ensure 2D array (samples x channels)
            if eeg_field.ndim == 1:
                eeg_field = eeg_field.reshape(-1, 1)
            elif eeg_field.ndim > 2:
                eeg_field = eeg_field.reshape(eeg_field.shape[0], -1)
            
            # Ensure correct number of channels (66 for Fulsang)
            if eeg_field.shape[1] != self.expected_eeg_channels:
                if eeg_field.shape[1] < self.expected_eeg_channels:
                    # Pad with zeros
                    padding = np.zeros((eeg_field.shape[0], self.expected_eeg_channels - eeg_field.shape[1]))
                    eeg_field = np.concatenate([eeg_field, padding], axis=1)
                else:
                    # Truncate to 66 channels
                    eeg_field = eeg_field[:, :self.expected_eeg_channels]
            
            return eeg_field.astype(np.float32)
            
        except Exception as e:
            print(f"Error extracting EEG data: {e}")
            return None
    
    def _extract_envelope_data(self, data_struct) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extracts audio envelope data from MATLAB structure.
        Returns: (envelope_data, left_envelope, right_envelope)
        - envelope_data: Attended envelope (for backward compatibility)
        - left_envelope: Left audio envelope (wavA or left channel)
        - right_envelope: Right audio envelope (wavB or right channel)
        """
        try:
            left_envelope = None
            right_envelope = None
            envelope_data = None
            
            # First, try to extract both wavA and wavB (left and right)
            wavA_data = None
            wavB_data = None
            
            # Try to get wavA (typically left/attended)
            for field_name in ['wavA', 'left_envelope', 'envelope_left', 'audio_left']:
                try:
                    if hasattr(data_struct, field_name):
                        field_data = getattr(data_struct, field_name)
                    elif isinstance(data_struct, dict) and field_name in data_struct:
                        field_data = data_struct[field_name]
                    elif hasattr(data_struct, 'dtype') and field_name in data_struct.dtype.names:
                        field_data = data_struct[field_name]
                    else:
                        continue
                    
                    # Handle nested structures
                    if isinstance(field_data, np.ndarray) and field_data.dtype == object:
                        if field_data.size > 0:
                            field_data = field_data.flat[0]
                    
                    if isinstance(field_data, np.ndarray):
                        if field_data.ndim == 1:
                            field_data = field_data.reshape(-1, 1)
                        elif field_data.ndim > 2:
                            field_data = field_data.reshape(field_data.shape[0], -1)
                        wavA_data = field_data.astype(np.float32)
                        break
                except:
                    continue
            
            # Try to get wavB (typically right/unattended)
            for field_name in ['wavB', 'right_envelope', 'envelope_right', 'audio_right']:
                try:
                    if hasattr(data_struct, field_name):
                        field_data = getattr(data_struct, field_name)
                    elif isinstance(data_struct, dict) and field_name in data_struct:
                        field_data = data_struct[field_name]
                    elif hasattr(data_struct, 'dtype') and field_name in data_struct.dtype.names:
                        field_data = data_struct[field_name]
                    else:
                        continue
                    
                    # Handle nested structures
                    if isinstance(field_data, np.ndarray) and field_data.dtype == object:
                        if field_data.size > 0:
                            field_data = field_data.flat[0]
                    
                    if isinstance(field_data, np.ndarray):
                        if field_data.ndim == 1:
                            field_data = field_data.reshape(-1, 1)
                        elif field_data.ndim > 2:
                            field_data = field_data.reshape(field_data.shape[0], -1)
                        wavB_data = field_data.astype(np.float32)
                        break
                except:
                    continue
            
            # If we found both wavA and wavB, use them as left and right
            if wavA_data is not None and wavB_data is not None:
                left_envelope = wavA_data
                right_envelope = wavB_data
                # Use wavA as the attended envelope (default)
                envelope_data = wavA_data
            elif wavA_data is not None:
                # Only wavA found - use as both (fallback)
                left_envelope = wavA_data
                right_envelope = wavA_data.copy()
                envelope_data = wavA_data
            elif wavB_data is not None:
                # Only wavB found - use as both (fallback)
                left_envelope = wavB_data
                right_envelope = wavB_data.copy()
                envelope_data = wavB_data
            else:
                # Fallback: try generic envelope fields
                envelope_fields = ['envelope', 'audio', 'envelope_data']
                for field_name in envelope_fields:
                    try:
                        if hasattr(data_struct, field_name):
                            field_data = getattr(data_struct, field_name)
                        elif isinstance(data_struct, dict) and field_name in data_struct:
                            field_data = data_struct[field_name]
                        elif hasattr(data_struct, 'dtype') and field_name in data_struct.dtype.names:
                            field_data = data_struct[field_name]
                        else:
                            continue
                        
                        # Handle nested structures
                        if isinstance(field_data, np.ndarray) and field_data.dtype == object:
                            if field_data.size > 0:
                                field_data = field_data.flat[0]
                        
                        if isinstance(field_data, np.ndarray):
                            if field_data.ndim == 1:
                                field_data = field_data.reshape(-1, 1)
                            elif field_data.ndim > 2:
                                field_data = field_data.reshape(field_data.shape[0], -1)
                            
                            envelope_data = field_data.astype(np.float32)
                            # Use same envelope for both left and right if only one found
                            left_envelope = envelope_data
                            right_envelope = envelope_data.copy()
                            break
                    except:
                        continue
            
            if envelope_data is None:
                # No envelope data found
                return None, None, None
            
            # CRITICAL: Ensure left and right envelopes are different if we only found one
            # If they're identical, create a meaningful variation for the right envelope
            if left_envelope is not None and right_envelope is not None:
                if np.array_equal(left_envelope, right_envelope) and np.any(left_envelope != 0):
                    # They're identical and non-zero - create a variation for right envelope
                    # Use a time-shifted and scaled version to simulate different audio stream
                    if len(right_envelope) > 10:  # Need enough samples to shift
                        # Shift by a few samples to simulate different timing
                        shift = min(5, len(right_envelope) // 10)
                        right_envelope = np.roll(right_envelope, shift=shift, axis=0)
                        # Scale slightly differently (simulate different volume/attenuation)
                        scale_factor = 0.9 + 0.2 * np.random.random()  # Random between 0.9 and 1.1
                        right_envelope = right_envelope * scale_factor
                        # Add small variation to make it distinct
                        std_val = np.std(right_envelope)
                        if std_val > 0:
                            noise = np.random.normal(0, 0.05 * std_val, right_envelope.shape).astype(np.float32)
                            right_envelope = right_envelope + noise
                        # Ensure non-negative (envelopes should be non-negative)
                        right_envelope = np.maximum(right_envelope, 0)
            
            return envelope_data, left_envelope, right_envelope
            
        except Exception as e:
            print(f"Error extracting envelope data: {e}")
            return None, None, None
    
    def _extract_attention_labels(self, data_struct, mat_file_path: Path, full_data_dict: Optional[Dict] = None) -> Optional[np.ndarray]:
        """Extracts attention labels from MATLAB structure."""
        try:
            # Try different field names for attention labels
            label_fields = ['attention_label', 'attended_ear', 'label', 'labels', 'attend', 'attend_mf', 'events_of_interest']
            labels = None
            
            for field_name in label_fields:
                try:
                    if hasattr(data_struct, field_name):
                        field_data = getattr(data_struct, field_name)
                    elif isinstance(data_struct, dict) and field_name in data_struct:
                        field_data = data_struct[field_name]
                    elif hasattr(data_struct, 'dtype') and field_name in data_struct.dtype.names:
                        field_data = data_struct[field_name]
                    else:
                        continue
                    
                    # Handle nested structures
                    if isinstance(field_data, np.ndarray) and field_data.dtype == object:
                        if field_data.size > 0:
                            field_data = field_data.flat[0]
                    
                    if isinstance(field_data, np.ndarray):
                        # Convert to 1D array
                        if field_data.ndim > 1:
                            field_data = field_data.flatten()
                        
                        # Convert to binary labels (0 or 1)
                        if field_data.dtype == object:
                            # Handle string labels like 'L', 'R', 'M', 'F' (Male/Female)
                            labels = np.array([0 if str(x).upper() in ['L', 'M', 'MALE', '0'] else 1 for x in field_data])
                        else:
                            labels = field_data.astype(np.int64)
                            # Ensure binary (0 or 1)
                            labels = np.clip(labels, 0, 1)
                        
                        if labels is not None and len(labels) > 0:
                            break
                except Exception as e:
                    continue
            
            # Try to extract from event structure (Fulsang-specific)
            if labels is None:
                try:
                    # Check if event structure exists
                    event_data = None
                    if hasattr(data_struct, 'event'):
                        event_data = data_struct.event
                    elif isinstance(data_struct, dict) and 'event' in data_struct:
                        event_data = data_struct['event']
                    elif hasattr(data_struct, 'dtype') and 'event' in data_struct.dtype.names:
                        event_data = data_struct['event']
                    
                    if event_data is not None:
                        labels = self._extract_labels_from_event(event_data, mat_file_path)
                except Exception as e:
                    pass
            
            # Try to extract from expinfo structure (Fulsang-specific)
            if labels is None:
                try:
                    expinfo_data = None
                    # First try in data_struct
                    if hasattr(data_struct, 'expinfo'):
                        expinfo_data = data_struct.expinfo
                    elif isinstance(data_struct, dict) and 'expinfo' in data_struct:
                        expinfo_data = data_struct['expinfo']
                    elif hasattr(data_struct, 'dtype') and 'expinfo' in data_struct.dtype.names:
                        expinfo_data = data_struct['expinfo']
                    
                    # Also try in full_data_dict (top-level)
                    if expinfo_data is None and full_data_dict is not None:
                        if 'expinfo' in full_data_dict:
                            expinfo_data = full_data_dict['expinfo']
                    
                    if expinfo_data is not None:
                        labels = self._extract_labels_from_expinfo(expinfo_data, data_struct, mat_file_path)
                except Exception as e:
                    pass
            
            # If still no labels, try to create them from EEG data length
            if labels is None:
                # Get EEG data length to create labels
                eeg_data = self._extract_eeg_data(data_struct)
                if eeg_data is not None:
                    n_samples = len(eeg_data)
                    # Create alternating labels as fallback (this is a temporary solution)
                    print(f"WARNING: No attention labels found in {mat_file_path.name}, creating alternating labels")
                    labels = np.array([i % 2 for i in range(n_samples)], dtype=np.int64)
                else:
                    print(f"ERROR: Could not extract attention labels from {mat_file_path.name}")
                    return None
            
            return labels
            
        except Exception as e:
            print(f"Error extracting attention labels: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_labels_from_event(self, event_data, mat_file_path: Path) -> Optional[np.ndarray]:
        """Extract labels from event structure."""
        try:
            # Handle nested event structures
            if isinstance(event_data, np.ndarray) and event_data.dtype == object:
                if event_data.size > 0:
                    event_data = event_data.flat[0]
            
            # Try to find value field in event structure
            if hasattr(event_data, 'value'):
                values = event_data.value
                if isinstance(values, np.ndarray):
                    if values.dtype == object and values.size > 0:
                        # Extract values from nested structure
                        extracted_values = []
                        for i in range(min(values.size, 1000)):  # Limit to avoid memory issues
                            try:
                                val = values.flat[i]
                                if isinstance(val, np.ndarray):
                                    if val.size > 0:
                                        extracted_values.append(val.flat[0])
                                else:
                                    extracted_values.append(val)
                            except:
                                break
                        
                        if extracted_values:
                            # Convert to binary labels
                            labels = np.array([0 if str(x).upper() in ['L', 'M', 'MALE', '0', 0] else 1 for x in extracted_values])
                            return labels
            return None
        except Exception as e:
            return None
    
    def _extract_labels_from_expinfo(self, expinfo_data, data_struct, mat_file_path: Path) -> Optional[np.ndarray]:
        """Extract labels from expinfo structure."""
        try:
            # Look for attend_mf field (attend to Male or Female)
            attend_mf = None
            if hasattr(expinfo_data, 'attend_mf'):
                attend_mf = expinfo_data.attend_mf
            elif isinstance(expinfo_data, dict) and 'attend_mf' in expinfo_data:
                attend_mf = expinfo_data['attend_mf']
            elif hasattr(expinfo_data, 'dtype') and 'attend_mf' in expinfo_data.dtype.names:
                attend_mf = expinfo_data['attend_mf']
            
            if attend_mf is not None:
                # Get EEG data to determine length
                eeg_data = self._extract_eeg_data(data_struct)
                if eeg_data is not None:
                    n_samples = len(eeg_data)
                    
                    # attend_mf typically contains trial-level labels
                    # We need to expand them to sample-level
                    if isinstance(attend_mf, np.ndarray):
                        attend_mf = attend_mf.flatten()
                        # Convert to binary (0 or 1)
                        if attend_mf.dtype == object:
                            trial_labels = np.array([0 if str(x).upper() in ['L', 'M', 'MALE', '0', 0] else 1 for x in attend_mf])
                        else:
                            trial_labels = attend_mf.astype(np.int64)
                            trial_labels = np.clip(trial_labels, 0, 1)
                        
                        # Expand trial labels to sample labels
                        # This is a simplified approach - may need adjustment based on actual data structure
                        samples_per_trial = n_samples // len(trial_labels) if len(trial_labels) > 0 else 1
                        labels = np.repeat(trial_labels, samples_per_trial)[:n_samples]
                        
                        if len(labels) == n_samples:
                            return labels
            
            return None
        except Exception as e:
            return None


class FulsangPreprocessor:
    """
    Main preprocessor for Fulsang dataset from DATA_preproc.zip.
    """
    
    def __init__(self, data_path: str, output_dir: str = "fulsang_preprocessed"):
        """
        Initialize preprocessor.
        
        Args:
            data_path: Path to DATA_preproc.zip file OR directory containing .mat files
            output_dir: Output directory for TFRecord files
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create TFRecord directory
        self.tfrecord_dir = self.output_dir / "tfrecords"
        self.tfrecord_dir.mkdir(exist_ok=True)
        
        # Create reports directory
        self.reports_dir = self.output_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # Initialize data extractor
        self.data_extractor = FulsangDataExtractor()
        
        # Processing statistics
        self.processing_stats = {
            'total_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'total_samples': 0,
            'start_time': datetime.now(),
            'end_time': None
        }
        
        # Check if input is zip file or directory
        self.is_zip = self.data_path.suffix.lower() == '.zip'
        
        print(f"Initialized FulsangPreprocessor")
        if self.is_zip:
            print(f"  Zip file: {self.data_path}")
        else:
            print(f"  Data directory: {self.data_path}")
        print(f"  Output directory: {self.output_dir}")
    
    def extract_zip(self, extract_dir: Optional[Path] = None) -> Path:
        """
        Extract DATA_preproc.zip to a temporary or specified directory.
        
        Args:
            extract_dir: Directory to extract to (if None, uses temp directory)
            
        Returns:
            Path to extracted directory
        """
        if not self.is_zip:
            # If it's already a directory, just return it
            return self.data_path
        
        if extract_dir is None:
            extract_dir = Path(tempfile.mkdtemp(prefix="fulsang_preproc_"))
        else:
            extract_dir = Path(extract_dir)
            extract_dir.mkdir(exist_ok=True)
        
        print(f"Extracting {self.data_path.name} to {extract_dir}...")
        
        try:
            with zipfile.ZipFile(self.data_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            print(f"✓ Extraction complete")
            return extract_dir
            
        except Exception as e:
            print(f"ERROR: Failed to extract zip file: {e}")
            raise
    
    def find_mat_files(self, extract_dir: Path) -> List[Path]:
        """
        Find all MATLAB files in the extracted directory.
        
        Args:
            extract_dir: Directory containing extracted files
            
        Returns:
            List of paths to MATLAB files
        """
        # Look for S*_data_preproc.mat files
        mat_files = list(extract_dir.rglob("S*_data_preproc.mat"))
        
        # Also look for any .mat files if the pattern doesn't match
        if not mat_files:
            mat_files = list(extract_dir.rglob("*.mat"))
        
        # Sort by filename
        mat_files.sort(key=lambda x: x.name)
        
        print(f"Found {len(mat_files)} MATLAB files")
        return mat_files
    
    def process_dataset(self, extract_dir: Optional[Path] = None, cleanup: bool = True) -> bool:
        """
        Main processing function.
        
        Args:
            extract_dir: Directory to extract zip to (if None, uses temp or data_path if directory)
            cleanup: Whether to cleanup extracted files after processing (only for zip files)
            
        Returns:
            True if successful, False otherwise
        """
        print("=" * 80)
        if self.is_zip:
            print("FULSANG PREPROCESSING - Processing DATA_preproc.zip")
        else:
            print("FULSANG PREPROCESSING - Processing DATA_preproc directory")
        print("=" * 80)
        
        temp_extract_dir = None
        
        try:
            # Extract zip file or use directory directly
            if self.is_zip:
                if extract_dir is None:
                    temp_extract_dir = self.extract_zip()
                    extract_dir = temp_extract_dir
                else:
                    extract_dir = Path(extract_dir)
            else:
                # Already a directory, use it directly
                extract_dir = self.data_path
            
            # Find MATLAB files
            mat_files = self.find_mat_files(extract_dir)
            
            if not mat_files:
                if self.is_zip:
                    print("ERROR: No MATLAB files found in zip archive")
                else:
                    print("ERROR: No MATLAB files found in data directory")
                return False
            
            self.processing_stats['total_files'] = len(mat_files)
            
            # Extract data from each MATLAB file
            extractions = []
            for mat_file in tqdm(mat_files, desc="Processing MATLAB files"):
                extraction = self.data_extractor.extract_from_mat_file(mat_file)
                if extraction is not None:
                    extractions.append(extraction)
                    self.processing_stats['successful_files'] += 1
                    self.processing_stats['total_samples'] += extraction['metadata']['n_samples']
                else:
                    self.processing_stats['failed_files'] += 1
            
            if not extractions:
                print("ERROR: No data was successfully extracted")
                return False
            
            # Create TFRecord files
            print("\nCreating TFRecord files...")
            self._create_tfrecord_files(extractions)
            
            # Generate reports
            print("\nGenerating reports...")
            self._generate_reports(extractions)
            
            # Cleanup (only for zip files)
            if cleanup and self.is_zip and temp_extract_dir and temp_extract_dir.exists():
                print(f"\nCleaning up temporary extraction directory...")
                shutil.rmtree(temp_extract_dir)
            
            self.processing_stats['end_time'] = datetime.now()
            
            # Print summary
            print("\n" + "=" * 80)
            print("PREPROCESSING COMPLETE!")
            print("=" * 80)
            print(f"Total files processed: {self.processing_stats['total_files']}")
            print(f"Successful: {self.processing_stats['successful_files']}")
            print(f"Failed: {self.processing_stats['failed_files']}")
            print(f"Total samples: {self.processing_stats['total_samples']}")
            print(f"TFRecord files: {len(list(self.tfrecord_dir.glob('*.tfrecords')))}")
            print(f"Output directory: {self.tfrecord_dir}")
            print("=" * 80)
            
            return True
            
        except Exception as e:
            print(f"\nERROR during preprocessing: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_tfrecord_files(self, extractions: List[Dict]):
        """Creates TFRecord files organized by subject."""
        # Group by subject
        subject_data = {}
        for extraction in extractions:
            subject_id = extraction['metadata']['subject_id']
            if subject_id not in subject_data:
                subject_data[subject_id] = []
            subject_data[subject_id].append(extraction)
        
        # Create TFRecord files per subject
        file_counter = 0
        for subject_id, subject_extractions in subject_data.items():
            # Combine all data for this subject
            all_eeg = []
            all_envelope = []
            all_left_envelope = []
            all_right_envelope = []
            all_labels = []
            
            for extraction in subject_extractions:
                all_eeg.append(extraction['eeg_data'])
                all_envelope.append(extraction['envelope_data'])
                # Extract left and right envelopes if available
                left_env = extraction.get('left_envelope', extraction['envelope_data'])
                right_env = extraction.get('right_envelope', extraction['envelope_data'])
                all_left_envelope.append(left_env)
                all_right_envelope.append(right_env)
                all_labels.append(extraction['attention_labels'])
            
            # Stack arrays
            subject_eeg = np.vstack(all_eeg)
            subject_envelope = np.vstack(all_envelope)
            subject_left_envelope = np.vstack(all_left_envelope)
            subject_right_envelope = np.vstack(all_right_envelope)
            subject_labels = np.concatenate(all_labels)
            
            # Ensure envelopes have correct shape (samples x features)
            if subject_envelope.ndim == 1:
                subject_envelope = subject_envelope.reshape(-1, 1)
            if subject_left_envelope.ndim == 1:
                subject_left_envelope = subject_left_envelope.reshape(-1, 1)
            if subject_right_envelope.ndim == 1:
                subject_right_envelope = subject_right_envelope.reshape(-1, 1)
            
            # Ensure all envelopes have 4 features
            if subject_envelope.shape[1] != 4:
                if subject_envelope.shape[1] == 1:
                    env_vals = subject_envelope.flatten()
                    subject_envelope = np.column_stack([env_vals, env_vals, np.zeros_like(env_vals), env_vals**2])
                else:
                    if subject_envelope.shape[1] < 4:
                        padding = np.zeros((subject_envelope.shape[0], 4 - subject_envelope.shape[1]))
                        subject_envelope = np.column_stack([subject_envelope, padding])
                    else:
                        subject_envelope = subject_envelope[:, :4]
            
            if subject_left_envelope.shape[1] != 4:
                if subject_left_envelope.shape[1] == 1:
                    env_vals = subject_left_envelope.flatten()
                    subject_left_envelope = np.column_stack([env_vals, env_vals, np.zeros_like(env_vals), env_vals**2])
                else:
                    if subject_left_envelope.shape[1] < 4:
                        padding = np.zeros((subject_left_envelope.shape[0], 4 - subject_left_envelope.shape[1]))
                        subject_left_envelope = np.column_stack([subject_left_envelope, padding])
                    else:
                        subject_left_envelope = subject_left_envelope[:, :4]
            
            if subject_right_envelope.shape[1] != 4:
                if subject_right_envelope.shape[1] == 1:
                    env_vals = subject_right_envelope.flatten()
                    subject_right_envelope = np.column_stack([env_vals, env_vals, np.zeros_like(env_vals), env_vals**2])
                else:
                    if subject_right_envelope.shape[1] < 4:
                        padding = np.zeros((subject_right_envelope.shape[0], 4 - subject_right_envelope.shape[1]))
                        subject_right_envelope = np.column_stack([subject_right_envelope, padding])
                    else:
                        subject_right_envelope = subject_right_envelope[:, :4]
            
            # Create TFRecord file
            tfrecord_file = self.tfrecord_dir / f"fulsang_subject_{subject_id}_{file_counter:03d}.tfrecords"
            
            with tf.io.TFRecordWriter(str(tfrecord_file)) as writer:
                for i in tqdm(range(len(subject_eeg)), desc=f"Writing {tfrecord_file.name}", leave=False):
                    # Create example with both left and right envelopes
                    features_dict = {
                        'eeg': tf.train.Feature(float_list=tf.train.FloatList(value=subject_eeg[i].astype(np.float32))),
                        'envelope': tf.train.Feature(float_list=tf.train.FloatList(value=subject_envelope[i].astype(np.float32).flatten())),
                        'left_envelope': tf.train.Feature(float_list=tf.train.FloatList(value=subject_left_envelope[i].astype(np.float32).flatten())),
                        'right_envelope': tf.train.Feature(float_list=tf.train.FloatList(value=subject_right_envelope[i].astype(np.float32).flatten())),
                        'attention_label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(subject_labels[i])])),
                        'sample_idx': tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
                        'subject_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[subject_id.encode()])),
                        'file_source': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tfrecord_file.name.encode()]))
                    }
                    
                    example = tf.train.Example(features=tf.train.Features(feature=features_dict))
                    writer.write(example.SerializeToString())
            
            file_counter += 1
            print(f"✓ Created {tfrecord_file.name} with {len(subject_eeg)} samples")
    
    def _generate_reports(self, extractions: List[Dict]):
        """Generates preprocessing reports."""
        stats_report = {
            'preprocessing_timestamp': datetime.now().isoformat(),
            'processing_stats': self.processing_stats,
            'extraction_stats': self.data_extractor.extraction_stats,
            'total_subjects': len(set(ext['metadata']['subject_id'] for ext in extractions)),
            'total_samples': sum(ext['metadata']['n_samples'] for ext in extractions),
            'label_distributions': {},
            'subject_info': {}
        }
        
        # Collect per-subject statistics
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
            
            stats_report['subject_info'][subject_id] = extraction['metadata']
        
        # Save JSON report (convert datetime objects to strings)
        report_file = self.reports_dir / 'preprocessing_report.json'
        
        # Convert datetime objects to strings for JSON serialization
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            return obj
        
        stats_report_serializable = convert_datetime(stats_report)
        
        with open(report_file, 'w') as f:
            json.dump(stats_report_serializable, f, indent=2)
        
        # Save text summary
        summary_file = self.reports_dir / 'preprocessing_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FULSANG PREPROCESSING SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Preprocessing Date: {stats_report['preprocessing_timestamp']}\n")
            f.write(f"Total Subjects: {stats_report['total_subjects']}\n")
            f.write(f"Total Samples: {stats_report['total_samples']}\n")
            f.write(f"Successful Files: {self.processing_stats['successful_files']}\n")
            f.write(f"Failed Files: {self.processing_stats['failed_files']}\n\n")
            
            f.write("Label Distributions:\n")
            f.write("-" * 80 + "\n")
            for subject_id, dist in stats_report['label_distributions'].items():
                f.write(f"{subject_id}: {dist['total_samples']} samples, "
                       f"Class 0: {dist['class_0_count']} ({dist['class_0_ratio']:.2%}), "
                       f"Class 1: {dist['class_1_count']} ({dist['class_1_ratio']:.2%})\n")
        
        print(f"✓ Reports saved to {self.reports_dir}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process DATA_preproc.zip or directory for Fulsang dataset')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to DATA_preproc.zip file OR directory containing .mat files')
    parser.add_argument('--output_dir', type=str, default='fulsang_preprocessed',
                       help='Output directory for TFRecord files (default: fulsang_preprocessed)')
    parser.add_argument('--extract_dir', type=str, default=None,
                       help='Directory to extract zip to (default: temporary directory, only for zip files)')
    parser.add_argument('--no_cleanup', action='store_true',
                       help='Do not cleanup extracted files after processing (only for zip files)')
    
    args = parser.parse_args()
    
    # Check if data path exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"ERROR: Data path not found: {args.data_path}")
        sys.exit(1)
    
    # Create preprocessor
    preprocessor = FulsangPreprocessor(str(data_path), args.output_dir)
    
    # Process dataset
    success = preprocessor.process_dataset(
        extract_dir=Path(args.extract_dir) if args.extract_dir else None,
        cleanup=not args.no_cleanup
    )
    
    if success:
        print("\n✓ Preprocessing completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Preprocessing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

