#!/usr/bin/env python3
"""
Visualize Audio Signals with Triggers from Fulsang Dataset

This script loads audio signals (wavA and wavB) and trigger information from
Fulsang TFRecord files and creates visualizations showing the audio waveforms
with trigger markers overlaid.

WHAT ARE wavA AND wavB?
-----------------------
- wavA: Audio envelope of the ATTENDED speaker (the speaker the participant 
        is instructed to pay attention to)
- wavB: Audio envelope of the UNATTENDED speaker (the other speaker in the 
        cocktail party scenario)
- These are processed audio envelopes (not raw audio), resampled to match 
  the EEG sampling rate (64 Hz)
- Each can be either a male or female speaker, indicated by wavA_speaker 
  and wavB_speaker (1=male, 2=female)

WHERE ARE THE TRIGGERS?
----------------------
- Triggers are displayed as RED DASHED VERTICAL LINES at the start of each 
  trial (x=0 seconds)
- The trigger value is shown as a text annotation near the top of the plot
- Triggers are trial-level identifiers stored in the TFRecord files
- If event-based triggers are available, they appear as ORANGE DOTTED LINES 
  at their respective time points

Usage:
    python visualize_fulsang_audio_triggers.py [--tfrecord_dir DIR] [--subject_id SUBJECT] 
                                                [--trial_idx TRIAL] [--output_dir DIR]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import tensorflow as tf
from tqdm import tqdm
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class FulsangAudioVisualizer:
    """Visualize audio signals with triggers from Fulsang dataset."""
    
    def __init__(self, tfrecord_dir: str, sampling_rate: int = 64, 
                 raw_matlab_dir: Optional[str] = None):
        """
        Initialize the visualizer.
        
        Args:
            tfrecord_dir: Directory containing TFRecord files
            sampling_rate: Sampling rate of the audio data (Hz)
            raw_matlab_dir: Optional directory containing raw MATLAB files for trigger extraction
        """
        self.tfrecord_dir = Path(tfrecord_dir)
        self.sampling_rate = sampling_rate
        
        # Set up raw MATLAB directory for trigger extraction
        if raw_matlab_dir:
            self.raw_matlab_dir = Path(raw_matlab_dir)
        else:
            # Try to auto-detect
            possible_dirs = [
                Path("Data/Fulsang/DATA_preproc"),
                Path("Data/Fulsang"),
                self.tfrecord_dir.parent.parent / "Data" / "Fulsang" / "DATA_preproc"
            ]
            self.raw_matlab_dir = None
            for dir_path in possible_dirs:
                if dir_path.exists():
                    self.raw_matlab_dir = dir_path
                    break
        
        if not self.tfrecord_dir.exists():
            raise ValueError(f"TFRecord directory not found: {self.tfrecord_dir}")
        
        # Find all TFRecord files
        self.tfrecord_files = list(self.tfrecord_dir.glob("*.tfrecords"))
        if not self.tfrecord_files:
            # Try subdirectories
            self.tfrecord_files = list(self.tfrecord_dir.glob("*/*.tfrecords"))
        if not self.tfrecord_files:
            self.tfrecord_files = list(self.tfrecord_dir.glob("*/*/*.tfrecords"))
        
        if not self.tfrecord_files:
            raise ValueError(f"No TFRecord files found in {self.tfrecord_dir}")
        
        print(f"Found {len(self.tfrecord_files)} TFRecord files")
        if self.raw_matlab_dir:
            print(f"Raw MATLAB directory for triggers: {self.raw_matlab_dir}")
    
    def load_trial_data(self, subject_id: Optional[str] = None, 
                       trial_idx: Optional[int] = None) -> List[Dict]:
        """
        Load trial data from TFRecord files.
        
        Args:
            subject_id: Optional subject ID to filter (e.g., "S1", "S2")
            trial_idx: Optional trial index to filter
            
        Returns:
            List of trial dictionaries containing audio and trigger data
        """
        all_trials = []
        
        for tfrecord_file in tqdm(self.tfrecord_files, desc="Loading trials"):
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
                        
                        # Extract metadata
                        n_samples = int(features['n_samples'].int64_list.value[0]) if 'n_samples' in features else 3200
                        n_channels = int(features['n_channels'].int64_list.value[0]) if 'n_channels' in features else 66
                        sampling_rate = int(features['sampling_rate'].int64_list.value[0]) if 'sampling_rate' in features else self.sampling_rate
                        
                        # Extract audio signals
                        wavA = None
                        wavB = None
                        
                        if 'wavA' in features:
                            wavA_values = features['wavA'].float_list.value
                            if wavA_values and len(wavA_values) == n_samples:
                                wavA = np.array(wavA_values, dtype=np.float32)
                        
                        if 'wavB' in features:
                            wavB_values = features['wavB'].float_list.value
                            if wavB_values and len(wavB_values) == n_samples:
                                wavB = np.array(wavB_values, dtype=np.float32)
                        
                        # Extract trigger information
                        trigger = None
                        trigger_times = []  # For potential event-based triggers
                        if 'trigger' in features:
                            trigger_values = features['trigger'].int64_list.value
                            if trigger_values:
                                # If single value, it's a trial-level trigger
                                if len(trigger_values) == 1:
                                    trigger = int(trigger_values[0])
                                else:
                                    # Multiple values might be event timestamps
                                    trigger_times = [int(v) for v in trigger_values]
                                    trigger = trigger_values[0] if trigger_values else None
                        
                        # Also check for event-based triggers (if stored as separate feature)
                        if 'trigger_times' in features:
                            trigger_time_values = features['trigger_times'].float_list.value
                            if trigger_time_values:
                                trigger_times = [float(v) for v in trigger_time_values]
                        
                        # If trigger not found in TFRecord, try to load from raw MATLAB file
                        if trigger is None:
                            trigger = self._load_trigger_from_raw_matlab(file_subject_id, record_idx)
                        
                        # Extract attention label
                        attention_label = None
                        if 'attention_label' in features:
                            label_values = features['attention_label'].int64_list.value
                            if label_values:
                                attention_label = int(label_values[0])
                        
                        # Extract speaker information
                        wavA_speaker = None
                        wavB_speaker = None
                        if 'wavA_speaker' in features:
                            wavA_speaker = int(features['wavA_speaker'].int64_list.value[0])
                        if 'wavB_speaker' in features:
                            wavB_speaker = int(features['wavB_speaker'].int64_list.value[0])
                        
                        # Create trial data
                        trial_data = {
                            'subject_id': file_subject_id,
                            'trial_idx': len(all_trials),
                            'record_idx': record_idx,
                            'file': tfrecord_file.name,
                            'wavA': wavA,
                            'wavB': wavB,
                            'trigger': trigger,
                            'trigger_times': trigger_times,  # Event timestamps if available
                            'attention_label': attention_label,
                            'wavA_speaker': wavA_speaker,  # 1=male, 2=female
                            'wavB_speaker': wavB_speaker,  # 1=male, 2=female
                            'n_samples': n_samples,
                            'sampling_rate': sampling_rate,
                            'duration': n_samples / sampling_rate
                        }
                        
                        # Filter by trial index if specified
                        if trial_idx is not None and trial_data['trial_idx'] != trial_idx:
                            continue
                        
                        all_trials.append(trial_data)
                        
                    except Exception as e:
                        print(f"Error processing record in {tfrecord_file.name}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error loading {tfrecord_file.name}: {e}")
                continue
        
        return all_trials
    
    def _load_trigger_from_raw_matlab(self, subject_id: str, trial_idx: int) -> Optional[int]:
        """Load trigger value from raw MATLAB file if available."""
        if not self.raw_matlab_dir or not self.raw_matlab_dir.exists():
            return None
        
        try:
            import scipy.io as sio
            
            # Find MATLAB file for this subject
            mat_file = self.raw_matlab_dir / f"{subject_id}_data_preproc.mat"
            if not mat_file.exists():
                return None
            
            # Load MATLAB file
            mat_data = sio.loadmat(str(mat_file), squeeze_me=False, struct_as_record=False)
            
            if 'data' not in mat_data:
                return None
            
            data_struct = mat_data['data']
            if not isinstance(data_struct, np.ndarray) or data_struct.size == 0:
                return None
            
            if trial_idx >= data_struct.size:
                return None
            
            trial_elem = data_struct.flat[trial_idx]
            
            # Extract trigger from event structure
            if hasattr(trial_elem, 'event'):
                event = trial_elem.event
                if isinstance(event, np.ndarray) and event.size > 0:
                    first_event = event.flat[0]
                    if hasattr(first_event, 'eeg'):
                        eeg_events = first_event.eeg
                        if isinstance(eeg_events, np.ndarray) and eeg_events.size > 0:
                            first_eeg_event = eeg_events.flat[0]
                            if hasattr(first_eeg_event, 'value'):
                                value = first_eeg_event.value
                                if isinstance(value, np.ndarray):
                                    # Unwrap nested arrays
                                    while isinstance(value, np.ndarray) and value.dtype == object and value.size > 0:
                                        value = value.flat[0]
                                    if isinstance(value, np.ndarray):
                                        value_flat = value.flatten()
                                        if value_flat.size > 0:
                                            trigger_val = value_flat[0]
                                            if isinstance(trigger_val, (int, np.integer, float, np.floating)):
                                                return int(trigger_val)
            
            return None
            
        except Exception as e:
            # Silently fail - triggers from MATLAB are optional
            return None
    
    def visualize_trial(self, trial_data: Dict, output_path: Optional[str] = None,
                       show_plot: bool = True):
        """
        Visualize a single trial's audio signals with triggers.
        
        Args:
            trial_data: Dictionary containing trial data
            output_path: Optional path to save the figure
            show_plot: Whether to display the plot
        """
        wavA = trial_data.get('wavA')
        wavB = trial_data.get('wavB')
        trigger = trial_data.get('trigger')
        attention_label = trial_data.get('attention_label')
        sampling_rate = trial_data.get('sampling_rate', self.sampling_rate)
        n_samples = trial_data.get('n_samples', len(wavA) if wavA is not None else 3200)
        
        # Create time axis
        duration = n_samples / sampling_rate
        time_axis = np.linspace(0, duration, n_samples)
        
        # Create figure with subplots
        n_signals = sum([wavA is not None, wavB is not None])
        if n_signals == 0:
            print("No audio signals available for visualization")
            return
        
        fig, axes = plt.subplots(n_signals, 1, figsize=(14, 4 * n_signals), sharex=True)
        if n_signals == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Plot wavA
        if wavA is not None:
            ax = axes[plot_idx]
            ax.plot(time_axis, wavA, 'b-', linewidth=0.5, alpha=0.7, label='wavA')
            
            # Add speaker label
            wavA_speaker = trial_data.get('wavA_speaker')
            speaker_label = "Unknown"
            if wavA_speaker == 1:
                speaker_label = "Male"
            elif wavA_speaker == 2:
                speaker_label = "Female"
            
            # Add attention indicator
            attended = ""
            if attention_label is not None:
                if (attention_label == 0 and wavA_speaker == 1) or (attention_label == 1 and wavA_speaker == 2):
                    attended = " (ATTENDED)"
            
            ax.set_title(f'wavA - {speaker_label}{attended}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Amplitude', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Always show trigger marker at trial start (x=0 seconds = trial start time)
            # Use dotted line style as requested
            # NOTE: x=0 is the TIME POSITION (trial start), not the trigger value
            trigger_times = trial_data.get('trigger_times', [])
            if trigger is not None:
                trigger_label = f'Trial Start | Trigger Value: {trigger}'
                trigger_text = f'Trigger Value: {trigger}\n(Time: 0s = Trial Start)'
            else:
                trigger_label = 'Trial Start (Trigger value not stored)'
                trigger_text = 'Trial Start\n(Trigger value: N/A)'
            
            ax.axvline(x=0, color='red', linestyle=':', linewidth=2.5, 
                      label=trigger_label, alpha=0.9, zorder=10)
            
            # Add text annotation for trigger value
            ax.text(0.02, ax.get_ylim()[1] * 0.95, trigger_text, 
                   color='red', fontsize=9, fontweight='bold', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='red'),
                   verticalalignment='top', zorder=11)
            ax.legend(loc='upper right')
            
            # Add event-based trigger markers if available
            if trigger_times:
                for idx, t_time in enumerate(trigger_times):
                    # Convert to seconds if needed (assuming trigger_times are in samples)
                    if t_time > duration:
                        # If trigger time is larger than duration, it might be in samples
                        t_time_sec = t_time / sampling_rate
                    else:
                        t_time_sec = t_time
                    
                    if 0 <= t_time_sec <= duration:
                        ax.axvline(x=t_time_sec, color='orange', linestyle=':', 
                                  linewidth=2, alpha=0.7, label=f'Event {idx+1}' if idx == 0 else '')
                        ax.text(t_time_sec + 0.02, ax.get_ylim()[1] * 0.85, 
                               f'Event@{t_time_sec:.2f}s', color='orange', fontsize=9,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                               verticalalignment='top')
            
            plot_idx += 1
        
        # Plot wavB
        if wavB is not None:
            ax = axes[plot_idx]
            ax.plot(time_axis, wavB, 'g-', linewidth=0.5, alpha=0.7, label='wavB')
            
            # Add speaker label
            wavB_speaker = trial_data.get('wavB_speaker')
            speaker_label = "Unknown"
            if wavB_speaker == 1:
                speaker_label = "Male"
            elif wavB_speaker == 2:
                speaker_label = "Female"
            
            # Add attention indicator
            attended = ""
            if attention_label is not None:
                if (attention_label == 0 and wavB_speaker == 1) or (attention_label == 1 and wavB_speaker == 2):
                    attended = " (ATTENDED)"
            
            ax.set_title(f'wavB - {speaker_label}{attended}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Amplitude', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Always show trigger marker at trial start (x=0 seconds = trial start time)
            # Use dotted line style as requested
            # NOTE: x=0 is the TIME POSITION (trial start), not the trigger value
            trigger_times = trial_data.get('trigger_times', [])
            if trigger is not None:
                trigger_label = f'Trial Start | Trigger Value: {trigger}'
                trigger_text = f'Trigger Value: {trigger}\n(Time: 0s = Trial Start)'
            else:
                trigger_label = 'Trial Start (Trigger value not stored)'
                trigger_text = 'Trial Start\n(Trigger value: N/A)'
            
            ax.axvline(x=0, color='red', linestyle=':', linewidth=2.5, 
                      label=trigger_label, alpha=0.9, zorder=10)
            
            # Add text annotation for trigger value
            ax.text(0.02, ax.get_ylim()[1] * 0.95, trigger_text, 
                   color='red', fontsize=9, fontweight='bold', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='red'),
                   verticalalignment='top', zorder=11)
            ax.legend(loc='upper right')
            
            # Add event-based trigger markers if available
            if trigger_times:
                for idx, t_time in enumerate(trigger_times):
                    # Convert to seconds if needed (assuming trigger_times are in samples)
                    if t_time > duration:
                        # If trigger time is larger than duration, it might be in samples
                        t_time_sec = t_time / sampling_rate
                    else:
                        t_time_sec = t_time
                    
                    if 0 <= t_time_sec <= duration:
                        ax.axvline(x=t_time_sec, color='orange', linestyle=':', 
                                  linewidth=2, alpha=0.7, label=f'Event {idx+1}' if idx == 0 else '')
                        ax.text(t_time_sec + 0.02, ax.get_ylim()[1] * 0.85, 
                               f'Event@{t_time_sec:.2f}s', color='orange', fontsize=9,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                               verticalalignment='top')
        
        # Set common x-axis label
        axes[-1].set_xlabel('Time (seconds)', fontsize=10)
        
        # Add overall title with trigger information
        subject_id = trial_data.get('subject_id', 'Unknown')
        trial_idx = trial_data.get('trial_idx', 'Unknown')
        trigger = trial_data.get('trigger', 'N/A')
        title = f'Fulsang Audio Signals - Subject: {subject_id}, Trial: {trial_idx}'
        if trigger is not None:
            title += f' | Trigger: {trigger}'
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {output_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def visualize_multiple_trials(self, trials: List[Dict], max_trials: int = 10,
                                  output_dir: Optional[str] = None):
        """
        Visualize multiple trials.
        
        Args:
            trials: List of trial dictionaries
            max_trials: Maximum number of trials to visualize
            output_dir: Optional directory to save figures
        """
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        n_trials = min(len(trials), max_trials)
        print(f"Visualizing {n_trials} trials...")
        
        for i, trial in enumerate(trials[:n_trials]):
            subject_id = trial.get('subject_id', 'Unknown')
            trial_idx = trial.get('trial_idx', i)
            
            output_path = None
            if output_dir:
                output_path = Path(output_dir) / f"subject_{subject_id}_trial_{trial_idx}.png"
            
            print(f"\nTrial {i+1}/{n_trials}: Subject {subject_id}, Trial {trial_idx}")
            self.visualize_trial(trial, output_path=output_path, show_plot=False)
    
    def create_summary_plot(self, trials: List[Dict], output_path: Optional[str] = None):
        """
        Create a summary plot showing multiple trials in a grid.
        
        Args:
            trials: List of trial dictionaries
            output_path: Optional path to save the figure
        """
        n_trials = min(len(trials), 9)  # 3x3 grid
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i in range(n_trials):
            trial = trials[i]
            ax = axes[i]
            
            wavA = trial.get('wavA')
            wavB = trial.get('wavB')
            sampling_rate = trial.get('sampling_rate', self.sampling_rate)
            n_samples = trial.get('n_samples', len(wavA) if wavA is not None else 3200)
            
            duration = n_samples / sampling_rate
            time_axis = np.linspace(0, duration, n_samples)
            
            if wavA is not None:
                ax.plot(time_axis, wavA, 'b-', linewidth=0.3, alpha=0.6, label='wavA')
            if wavB is not None:
                ax.plot(time_axis, wavB, 'g-', linewidth=0.3, alpha=0.6, label='wavB')
            
            subject_id = trial.get('subject_id', 'Unknown')
            trial_idx = trial.get('trial_idx', i)
            trigger = trial.get('trigger', 'N/A')
            
            ax.set_title(f'S{subject_id} T{trial_idx} (T:{trigger})', fontsize=9)
            ax.set_xlabel('Time (s)', fontsize=8)
            ax.set_ylabel('Amplitude', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
        
        # Hide unused subplots
        for i in range(n_trials, 9):
            axes[i].axis('off')
        
        fig.suptitle('Fulsang Audio Signals Summary (wavA in blue, wavB in green)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved summary plot to {output_path}")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize audio signals with triggers from Fulsang dataset'
    )
    parser.add_argument('--tfrecord_dir', type=str, default='fulsang_preprocessed',
                       help='Directory containing TFRecord files')
    parser.add_argument('--subject_id', type=str, default=None,
                       help='Subject ID to filter (e.g., S1, S2)')
    parser.add_argument('--trial_idx', type=int, default=None,
                       help='Specific trial index to visualize')
    parser.add_argument('--output_dir', type=str, default='visualization_output',
                       help='Directory to save output figures')
    parser.add_argument('--max_trials', type=int, default=10,
                       help='Maximum number of trials to visualize')
    parser.add_argument('--summary', action='store_true',
                       help='Create summary plot instead of individual plots')
    parser.add_argument('--sampling_rate', type=int, default=64,
                       help='Sampling rate of audio data (Hz)')
    
    args = parser.parse_args()
    
    # Create visualizer
    print(f"Initializing visualizer with TFRecord directory: {args.tfrecord_dir}")
    visualizer = FulsangAudioVisualizer(args.tfrecord_dir, sampling_rate=args.sampling_rate)
    
    # Load trial data
    print("\nLoading trial data...")
    trials = visualizer.load_trial_data(subject_id=args.subject_id, trial_idx=args.trial_idx)
    
    if not trials:
        print("No trials found matching the criteria")
        return
    
    print(f"\nLoaded {len(trials)} trials")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"  Subjects: {len(set(t.get('subject_id') for t in trials))}")
    print(f"  Trials with wavA: {sum(1 for t in trials if t.get('wavA') is not None)}")
    print(f"  Trials with wavB: {sum(1 for t in trials if t.get('wavB') is not None)}")
    print(f"  Trials with triggers: {sum(1 for t in trials if t.get('trigger') is not None)}")
    
    # Create output directory
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Visualize
    if args.summary:
        output_path = Path(args.output_dir) / "summary_plot.png" if args.output_dir else None
        visualizer.create_summary_plot(trials, output_path=output_path)
    else:
        if args.trial_idx is not None and len(trials) == 1:
            # Single trial visualization
            output_path = Path(args.output_dir) / f"trial_{args.trial_idx}.png" if args.output_dir else None
            visualizer.visualize_trial(trials[0], output_path=output_path, show_plot=True)
        else:
            # Multiple trials
            visualizer.visualize_multiple_trials(trials, max_trials=args.max_trials, 
                                                output_dir=args.output_dir)


if __name__ == '__main__':
    main()
