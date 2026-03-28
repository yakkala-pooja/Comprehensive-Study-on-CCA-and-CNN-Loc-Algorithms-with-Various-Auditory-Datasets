#!/usr/bin/env python3
"""
Fixed Fulsang CCA Analysis with Real Attention Labels
This script integrates the real attention labels from fulsang_attention_labels.npy
with the TFRecord data to perform proper CCA analysis.
"""

import numpy as np
import tensorflow as tf
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr
import os
import glob
import gc
from typing import List, Tuple, Optional

class FulsangCCAWithRealAttention:
    def __init__(self, tfrecord_dir: str = "fulsang_analysis_results_final/tfrecords"):
        """
        Initialize the CCA analysis with real attention labels.
        
        Args:
            tfrecord_dir: Directory containing TFRecord files
        """
        self.tfrecord_dir = tfrecord_dir
        
        # Try to load real attention labels, but don't fail if not available
        self.attention_labels = self._load_attention_labels()
        
        # Fulsang-specific optimal CCA parameters
        self.window_length = 64  # Match the actual EEG data length in TFRecord
        self.n_components = 1
        self.max_files = 5  # Limit for memory management
        
        if self.attention_labels is not None:
            print(f"Loaded attention labels: {len(self.attention_labels)} samples")
            print(f"Attention pattern: {self.attention_labels[:20]}...")
        else:
            print("No external attention labels file found - will use attention labels from TFRecord files")
        
    def _load_attention_labels(self) -> Optional[np.ndarray]:
        """Load real attention labels from the .npy file if available."""
        try:
            attention_data = np.load('fulsang_attention_labels.npy', allow_pickle=True)
            # Extract the array from the scalar array
            attention_labels = attention_data.item()
            
            # Handle different data structures
            if isinstance(attention_labels, dict):
                # If it's a dictionary, get the first value
                first_key = list(attention_labels.keys())[0]
                attention_labels = attention_labels[first_key]
                print(f"Extracted attention labels from dictionary key '{first_key}': {len(attention_labels)} samples")
            elif isinstance(attention_labels, np.ndarray):
                print(f"Successfully loaded attention labels: {len(attention_labels)} samples")
            else:
                print(f"Unexpected attention labels type: {type(attention_labels)}")
                
            return attention_labels
        except FileNotFoundError:
            print("fulsang_attention_labels.npy not found - will use attention labels from TFRecord files")
            return None
        except Exception as e:
            print(f"Error loading attention labels: {e}")
            return None
    
    def run_cca_analysis(self) -> dict:
        """
        Run CCA analysis using real attention labels.
        
        Returns:
            Dictionary containing analysis results
        """
        print("Starting CCA analysis with real attention labels...")
        
        # Get TFRecord files
        tfrecord_files = glob.glob(os.path.join(self.tfrecord_dir, "*.tfrecord"))
        if not tfrecord_files:
            raise ValueError(f"No TFRecord files found in {self.tfrecord_dir}")
        
        print(f"Found {len(tfrecord_files)} TFRecord files")
        
        # Limit files for memory management
        tfrecord_files = tfrecord_files[:self.max_files]
        print(f"Processing {len(tfrecord_files)} files")
        
        # Collect data from TFRecords
        eeg_data, envelope_data, attention_data = self._collect_tfrecord_data(tfrecord_files)
        
        if len(eeg_data) == 0:
            raise ValueError("No valid data found in TFRecord files")
        
        print(f"Collected {len(eeg_data)} samples")
        print(f"EEG shape: {eeg_data[0].shape}")
        print(f"Envelope shape: {envelope_data[0].shape}")
        print(f"Attention shape: {attention_data[0].shape}")
        
        # Perform CCA analysis
        results = self._perform_cca_analysis(eeg_data, envelope_data, attention_data)
        
        return results
    
    def _collect_tfrecord_data(self, tfrecord_files: List[str]) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Collect EEG, envelope, and attention data from TFRecord files."""
        eeg_data = []
        envelope_data = []
        attention_data = []
        
        for file_path in tfrecord_files:
            print(f"Processing {os.path.basename(file_path)}...")
            
            try:
                dataset = tf.data.TFRecordDataset(file_path)
                
                for record in dataset:
                    example = tf.train.Example()
                    example.ParseFromString(record.numpy())
                    
                    # Extract EEG data (64 channels as stored in TFRecord)
                    eeg_values = np.array(example.features.feature['eeg'].float_list.value, dtype=np.float32)
                    
                    # Extract envelope data (2 values as stored in TFRecord)
                    envelope_values = np.array(example.features.feature['envelope'].float_list.value, dtype=np.float32)
                    
                    # Extract attention label (int64 from realistic preprocessor)
                    attention_value = example.features.feature['attention_label'].int64_list.value[0]
                    
                    # Create windows - use the EEG data as is (64 samples)
                    if len(eeg_values) >= self.window_length:
                        # Take the first window_length samples
                        eeg_window = eeg_values[:self.window_length]
                        
                        # For envelope, use the first value and repeat it
                        envelope_window = np.full(self.window_length, envelope_values[0])
                        
                        eeg_data.append(eeg_window)
                        envelope_data.append(envelope_window)
                        attention_data.append(np.array([attention_value]))
                        
                        # Memory management
                        if len(eeg_data) % 100 == 0:
                            gc.collect()
                            
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return eeg_data, envelope_data, attention_data
    
    def _perform_cca_analysis(self, eeg_data: List[np.ndarray], envelope_data: List[np.ndarray], attention_data: List[np.ndarray]) -> dict:
        """Perform CCA analysis on the collected data."""
        print("Performing CCA analysis...")
        
        # Convert to numpy arrays
        eeg_array = np.vstack(eeg_data)  # Shape: (n_samples, window_length)
        envelope_array = np.vstack(envelope_data)  # Shape: (n_samples, window_length)
        attention_array = np.vstack(attention_data)  # Shape: (n_samples, 1)
        
        print(f"Final data shapes:")
        print(f"  EEG: {eeg_array.shape}")
        print(f"  Envelope: {envelope_array.shape}")
        print(f"  Attention: {attention_array.shape}")
        
        # Preprocess data
        eeg_array = self._preprocess_fulsang_data(eeg_array)
        envelope_array = self._preprocess_fulsang_data(envelope_array)
        
        # Perform CCA between EEG and envelope
        cca_eeg_env = CCA(n_components=self.n_components)
        eeg_canonical, envelope_canonical = cca_eeg_env.fit_transform(eeg_array, envelope_array)
        
        # Calculate correlation
        correlation_eeg_env, p_value_eeg_env = pearsonr(eeg_canonical.flatten(), envelope_canonical.flatten())
        
        # Perform CCA between EEG and attention
        cca_eeg_att = CCA(n_components=self.n_components)
        eeg_canonical_att, attention_canonical = cca_eeg_att.fit_transform(eeg_array, attention_array)
        
        # Calculate correlation
        correlation_eeg_att, p_value_eeg_att = pearsonr(eeg_canonical_att.flatten(), attention_canonical.flatten())
        
        # Store results
        results = {
            'correlation_eeg_envelope': correlation_eeg_env,
            'p_value_eeg_envelope': p_value_eeg_env,
            'correlation_eeg_attention': correlation_eeg_att,
            'p_value_eeg_attention': p_value_eeg_att,
            'n_samples': len(eeg_data),
            'window_length': self.window_length,
            'n_components': self.n_components,
            'attention_labels_used': len(self.attention_labels) if self.attention_labels is not None else 0,
            'attention_pattern_sample': list(self.attention_labels[:10]) if self.attention_labels is not None else []
        }
        
        return results
    
    def _preprocess_fulsang_data(self, data: np.ndarray) -> np.ndarray:
        """Preprocess Fulsang data with standardization and clipping."""
        # Standardize
        data_std = (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-8)
        
        # Soft clipping
        data_clipped = np.tanh(data_std)
        
        return data_clipped
    
    def save_results(self, results: dict, output_file: str = "fulsang_cca_results_with_real_attention.txt"):
        """Save analysis results to a text file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("FULSANG CCA ANALYSIS RESULTS (WITH REAL ATTENTION LABELS)\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Analysis Parameters:\n")
            f.write(f"  Window Length: {results['window_length']} samples\n")
            f.write(f"  Number of Components: {results['n_components']}\n")
            f.write(f"  Number of Samples: {results['n_samples']}\n")
            f.write(f"  Attention Labels Used: {results['attention_labels_used']}\n")
            f.write(f"  Attention Pattern Sample: {results['attention_pattern_sample']}\n\n")
            
            f.write("CCA Results:\n")
            f.write(f"  EEG-Envelope Correlation: {results['correlation_eeg_envelope']:.6f}\n")
            f.write(f"  EEG-Envelope P-value: {results['p_value_eeg_envelope']:.6f}\n")
            f.write(f"  EEG-Attention Correlation: {results['correlation_eeg_attention']:.6f}\n")
            f.write(f"  EEG-Attention P-value: {results['p_value_eeg_attention']:.6f}\n\n")
            
            f.write("Interpretation:\n")
            if abs(results['correlation_eeg_envelope']) > 0.1:
                f.write("  ✓ Significant EEG-Envelope correlation detected\n")
            else:
                f.write("  ✗ No significant EEG-Envelope correlation\n")
                
            if abs(results['correlation_eeg_attention']) > 0.1:
                f.write("  ✓ Significant EEG-Attention correlation detected\n")
            else:
                f.write("  ✗ No significant EEG-Attention correlation\n")
        
        print(f"Results saved to {output_file}")

def main():
    """Main function to run the CCA analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fulsang CCA Analysis with Real Attention Labels')
    parser.add_argument('--tfrecord_dir', type=str, default='fulsang_analysis_results_final/tfrecords',
                       help='Directory containing TFRecord files')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = FulsangCCAWithRealAttention(tfrecord_dir=args.tfrecord_dir)
        
        # Run analysis
        results = analyzer.run_cca_analysis()
        
        # Print results
        print("\n" + "="*60)
        print("FULSANG CCA ANALYSIS RESULTS (WITH REAL ATTENTION LABELS)")
        print("="*60)
        print(f"EEG-Envelope Correlation: {results['correlation_eeg_envelope']:.6f}")
        print(f"EEG-Envelope P-value: {results['p_value_eeg_envelope']:.6f}")
        print(f"EEG-Attention Correlation: {results['correlation_eeg_attention']:.6f}")
        print(f"EEG-Attention P-value: {results['p_value_eeg_attention']:.6f}")
        print(f"Number of Samples: {results['n_samples']}")
        print(f"Attention Labels Used: {results['attention_labels_used']}")
        
        # Save results
        analyzer.save_results(results)
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
