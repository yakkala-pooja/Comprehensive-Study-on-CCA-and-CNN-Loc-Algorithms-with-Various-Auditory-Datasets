#!/usr/bin/env python3
"""
Visualize Audio Signals with Triggers from Raw Fulsang MATLAB Files

This script loads audio signals (wavA and wavB) and trigger information directly
from the raw/preprocessed MATLAB files in Data/Fulsang/DATA_preproc/ instead of
from processed TFRecord files.

Usage:
    python visualize_fulsang_raw_audio_triggers.py [--data_dir DIR] [--subject_id SUBJECT] 
                                                   [--trial_idx TRIAL] [--output_dir DIR]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Dict
import scipy.io as sio
from tqdm import tqdm


class FulsangRawAudioVisualizer:
    """Visualize audio signals with triggers from raw Fulsang MATLAB files."""
    
    def __init__(self, data_dir: str = "Data/Fulsang", sampling_rate: int = 64):
        """
        Initialize the visualizer.
        
        Args:
            data_dir: Directory containing Fulsang data (should have DATA_preproc subdirectory)
            sampling_rate: Sampling rate of the audio data (Hz)
        """
        self.data_dir = Path(data_dir)
        self.preproc_dir = self.data_dir / "DATA_preproc"
        self.sampling_rate = sampling_rate
        
        if not self.preproc_dir.exists():
            raise ValueError(f"DATA_preproc directory not found: {self.preproc_dir}")
        
        # Find all MATLAB files
        self.mat_files = list(self.preproc_dir.glob("S*_data_preproc.mat"))
        
        if not self.mat_files:
            raise ValueError(f"No MATLAB files found in {self.preproc_dir}")
        
        print(f"Found {len(self.mat_files)} MATLAB files")
    
    def _extract_expinfo(self, data_struct, mat_data) -> Dict:
        """Extract experimental info from MATLAB structure."""
        expinfo = {}
        
        # Try to get expinfo from mat_data
        if 'expinfo' in mat_data:
            expinfo_raw = mat_data['expinfo']
            # Convert MATLAB struct to dict
            if hasattr(expinfo_raw, 'dtype') and expinfo_raw.dtype.names:
                for field in expinfo_raw.dtype.names:
                    value = expinfo_raw[field]
                    # Unwrap arrays
                    if isinstance(value, np.ndarray):
                        if value.size == 1:
                            expinfo[field] = value.item()
                        else:
                            expinfo[field] = value
                    else:
                        expinfo[field] = value
        
        return expinfo
    
    def _extract_audio_trials(self, data_struct) -> tuple:
        """Extract wavA and wavB trials from MATLAB structure."""
        try:
            if not isinstance(data_struct, np.ndarray) or data_struct.size == 0:
                return None, None
            
            first_elem = data_struct.flat[0]
            
            wavA_trials = None
            wavB_trials = None
            
            # Extract wavA
            wavA_field = None
            if hasattr(first_elem, 'wavA'):
                wavA_field = first_elem.wavA
            elif hasattr(first_elem, 'dtype') and hasattr(first_elem.dtype, 'names') and 'wavA' in first_elem.dtype.names:
                wavA_field = first_elem['wavA']
            
            if wavA_field is not None:
                if isinstance(wavA_field, np.ndarray):
                    if wavA_field.dtype == object:
                        # Cell array of trials
                        wavA_trials = []
                        for i in range(wavA_field.size):
                            trial_data = wavA_field.flat[i]
                            if isinstance(trial_data, np.ndarray):
                                # Ensure it's 1D or 2D with single channel
                                if trial_data.ndim == 1:
                                    wavA_trials.append(trial_data)
                                elif trial_data.ndim == 2 and trial_data.shape[1] == 1:
                                    wavA_trials.append(trial_data[:, 0])
                                else:
                                    wavA_trials.append(trial_data.flatten())
                    else:
                        # Numeric array - might be (n_samples, n_trials) or (n_samples, 1, n_trials)
                        if wavA_field.ndim == 2:
                            # (n_samples, n_trials)
                            wavA_trials = [wavA_field[:, i] for i in range(wavA_field.shape[1])]
                        elif wavA_field.ndim == 3:
                            # (n_samples, 1, n_trials)
                            wavA_trials = [wavA_field[:, 0, i] for i in range(wavA_field.shape[2])]
                        else:
                            # Single trial
                            wavA_trials = [wavA_field.flatten()]
            
            # Extract wavB
            wavB_field = None
            if hasattr(first_elem, 'wavB'):
                wavB_field = first_elem.wavB
            elif hasattr(first_elem, 'dtype') and hasattr(first_elem.dtype, 'names') and 'wavB' in first_elem.dtype.names:
                wavB_field = first_elem['wavB']
            
            if wavB_field is not None:
                if isinstance(wavB_field, np.ndarray):
                    if wavB_field.dtype == object:
                        # Cell array of trials
                        wavB_trials = []
                        for i in range(wavB_field.size):
                            trial_data = wavB_field.flat[i]
                            if isinstance(trial_data, np.ndarray):
                                if trial_data.ndim == 1:
                                    wavB_trials.append(trial_data)
                                elif trial_data.ndim == 2 and trial_data.shape[1] == 1:
                                    wavB_trials.append(trial_data[:, 0])
                                else:
                                    wavB_trials.append(trial_data.flatten())
                    else:
                        # Numeric array
                        if wavB_field.ndim == 2:
                            wavB_trials = [wavB_field[:, i] for i in range(wavB_field.shape[1])]
                        elif wavB_field.ndim == 3:
                            wavB_trials = [wavB_field[:, 0, i] for i in range(wavB_field.shape[2])]
                        else:
                            wavB_trials = [wavB_field.flatten()]
            
            return wavA_trials, wavB_trials
            
        except Exception as e:
            print(f"Error extracting audio trials: {e}")
            return None, None
    
    def _extract_trigger_from_event(self, data_struct, trial_idx: int) -> Optional[int]:
        """Extract trigger value from event structure for a specific trial.
        
        Structure: data[i].event.eeg is array of events, each has 'value' field
        The first event's value typically contains the trigger code.
        """
        try:
            if not isinstance(data_struct, np.ndarray) or data_struct.size == 0:
                return None
            
            if trial_idx >= data_struct.size:
                return None
            
            trial_elem = data_struct.flat[trial_idx]
            
            # Try to get event structure
            event = None
            if hasattr(trial_elem, 'event'):
                event = trial_elem.event
            elif hasattr(trial_elem, 'dtype') and hasattr(trial_elem.dtype, 'names') and 'event' in trial_elem.dtype.names:
                event = trial_elem['event']
            
            if event is not None:
                # event.eeg is an array of events
                eeg_events = None
                if isinstance(event, np.ndarray) and event.size > 0:
                    first_event = event.flat[0]
                    if hasattr(first_event, 'eeg'):
                        eeg_events = first_event.eeg
                elif hasattr(event, 'eeg'):
                    eeg_events = event.eeg
                
                if eeg_events is not None and isinstance(eeg_events, np.ndarray) and eeg_events.size > 0:
                    # Get the first eeg event (usually contains the trigger)
                    first_eeg_event = eeg_events.flat[0]
                    
                    # Try to get value field
                    value = None
                    if hasattr(first_eeg_event, 'value'):
                        value = first_eeg_event.value
                    
                    if value is not None:
                        # value is typically a nested array like [[2]]
                        if isinstance(value, np.ndarray):
                            # Unwrap nested arrays
                            while isinstance(value, np.ndarray) and value.dtype == object and value.size > 0:
                                value = value.flat[0]
                            
                            # Now value should be a numeric array or scalar
                            if isinstance(value, np.ndarray):
                                # Flatten and get first element
                                value_flat = value.flatten()
                                if value_flat.size > 0:
                                    trigger_val = value_flat[0]
                                    if isinstance(trigger_val, (int, np.integer)):
                                        return int(trigger_val)
                                    elif isinstance(trigger_val, (float, np.floating)):
                                        return int(trigger_val)
                            elif isinstance(value, (int, np.integer, float, np.floating)):
                                return int(value)
            
            return None
            
        except Exception as e:
            print(f"Error extracting trigger from event: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_trial_data(self, subject_id: Optional[str] = None, 
                       trial_idx: Optional[int] = None) -> List[Dict]:
        """
        Load trial data from MATLAB files.
        
        Args:
            subject_id: Optional subject ID to filter (e.g., "S1", "S2")
            trial_idx: Optional trial index to filter
            
        Returns:
            List of trial dictionaries containing audio and trigger data
        """
        all_trials = []
        
        for mat_file in tqdm(self.mat_files, desc="Loading trials"):
            try:
                # Extract subject ID from filename
                file_subject_id = mat_file.stem.replace('_data_preproc', '').replace('_preproc', '')
                
                # Filter by subject if specified
                if subject_id and file_subject_id != subject_id:
                    continue
                
                # Load MATLAB file
                mat_data = sio.loadmat(str(mat_file), squeeze_me=False, struct_as_record=False)
                
                if 'data' not in mat_data:
                    print(f"Warning: No 'data' field in {mat_file.name}")
                    continue
                
                data_struct = mat_data['data']
                
                # Extract expinfo
                expinfo = self._extract_expinfo(data_struct, mat_data)
                
                # Extract audio trials
                wavA_trials, wavB_trials = self._extract_audio_trials(data_struct)
                
                if wavA_trials is None and wavB_trials is None:
                    print(f"Warning: No audio data found in {mat_file.name}")
                    continue
                
                # Get number of trials
                n_trials = max(
                    len(wavA_trials) if wavA_trials else 0,
                    len(wavB_trials) if wavB_trials else 0
                )
                
                # Extract trigger information
                trigger_values = expinfo.get('trigger', None)
                if trigger_values is None:
                    # Try to get from event structure
                    trigger_values = []
                    for i in range(n_trials):
                        trigger = self._extract_trigger_from_event(data_struct, i)
                        trigger_values.append(trigger)
                
                # Convert trigger to list if it's a single value
                if not isinstance(trigger_values, (list, np.ndarray)):
                    trigger_values = [trigger_values] * n_trials
                elif isinstance(trigger_values, np.ndarray) and trigger_values.size == 1:
                    trigger_values = [trigger_values.item()] * n_trials
                elif isinstance(trigger_values, np.ndarray):
                    trigger_values = trigger_values.flatten().tolist()
                
                # Extract attention labels
                attend_mf = expinfo.get('attend_mf', None)
                if attend_mf is not None:
                    if not isinstance(attend_mf, (list, np.ndarray)):
                        attend_mf = [attend_mf] * n_trials
                    elif isinstance(attend_mf, np.ndarray) and attend_mf.size == 1:
                        attend_mf = [attend_mf.item()] * n_trials
                    elif isinstance(attend_mf, np.ndarray):
                        attend_mf = attend_mf.flatten().tolist()
                
                # Create trial data
                for i in range(n_trials):
                    # Filter by trial index if specified
                    if trial_idx is not None and i != trial_idx:
                        continue
                    
                    wavA = wavA_trials[i] if wavA_trials and i < len(wavA_trials) else None
                    wavB = wavB_trials[i] if wavB_trials and i < len(wavB_trials) else None
                    
                    trigger = trigger_values[i] if i < len(trigger_values) else None
                    attention_label = None
                    if attend_mf and i < len(attend_mf):
                        # Convert attend_mf (1=male, 2=female) to label (0=male, 1=female)
                        attention_label = 0 if attend_mf[i] == 1 else 1 if attend_mf[i] == 2 else None
                    
                    trial_data = {
                        'subject_id': file_subject_id,
                        'trial_idx': i,
                        'file': mat_file.name,
                        'wavA': wavA,
                        'wavB': wavB,
                        'trigger': trigger,
                        'attention_label': attention_label,
                        'attend_mf': attend_mf[i] if attend_mf and i < len(attend_mf) else None,
                        'n_samples': len(wavA) if wavA is not None else (len(wavB) if wavB is not None else 3200),
                        'sampling_rate': self.sampling_rate,
                        'duration': (len(wavA) if wavA is not None else (len(wavB) if wavB is not None else 3200)) / self.sampling_rate
                    }
                    
                    all_trials.append(trial_data)
                    
            except Exception as e:
                print(f"Error loading {mat_file.name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return all_trials
    
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
        attend_mf = trial_data.get('attend_mf')
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
            
            # Add speaker label based on attend_mf
            speaker_label = "Unknown"
            if attend_mf == 1:
                speaker_label = "Male"
            elif attend_mf == 2:
                speaker_label = "Female"
            
            # Add attention indicator
            attended = ""
            if attention_label is not None or attend_mf is not None:
                # wavA is always the attended speaker
                attended = " (ATTENDED)"
            
            ax.set_title(f'wavA - {speaker_label}{attended}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Amplitude', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add trigger markers if available
            if trigger is not None:
                ax.axvline(x=0, color='red', linestyle='--', linewidth=3, 
                          label=f'Trial Trigger: {trigger}', alpha=0.8, zorder=10)
                ax.text(0.02, ax.get_ylim()[1] * 0.95, f'Trigger: {trigger}', 
                       color='red', fontsize=10, fontweight='bold', 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       verticalalignment='top', zorder=11)
                ax.legend(loc='upper right')
            
            plot_idx += 1
        
        # Plot wavB
        if wavB is not None:
            ax = axes[plot_idx]
            ax.plot(time_axis, wavB, 'g-', linewidth=0.5, alpha=0.7, label='wavB')
            
            # Add speaker label (complement of wavA)
            speaker_label = "Unknown"
            if attend_mf == 1:
                speaker_label = "Female"  # wavB is complement
            elif attend_mf == 2:
                speaker_label = "Male"  # wavB is complement
            
            ax.set_title(f'wavB - {speaker_label} (UNATTENDED)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Amplitude', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add trigger markers if available
            if trigger is not None:
                ax.axvline(x=0, color='red', linestyle='--', linewidth=3, 
                          label=f'Trial Trigger: {trigger}', alpha=0.8, zorder=10)
                ax.text(0.02, ax.get_ylim()[1] * 0.95, f'Trigger: {trigger}', 
                       color='red', fontsize=10, fontweight='bold', 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       verticalalignment='top', zorder=11)
                ax.legend(loc='upper right')
        
        # Set common x-axis label
        axes[-1].set_xlabel('Time (seconds)', fontsize=10)
        
        # Add overall title with trigger information
        subject_id = trial_data.get('subject_id', 'Unknown')
        trial_idx = trial_data.get('trial_idx', 'Unknown')
        title = f'Fulsang Audio Signals (RAW) - Subject: {subject_id}, Trial: {trial_idx}'
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
        """Visualize multiple trials."""
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
                output_path = Path(output_dir) / f"raw_subject_{subject_id}_trial_{trial_idx}.png"
            
            print(f"\nTrial {i+1}/{n_trials}: Subject {subject_id}, Trial {trial_idx}")
            self.visualize_trial(trial, output_path=output_path, show_plot=False)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize audio signals with triggers from raw Fulsang MATLAB files'
    )
    parser.add_argument('--data_dir', type=str, default='Data/Fulsang',
                       help='Directory containing Fulsang data (should have DATA_preproc subdirectory)')
    parser.add_argument('--subject_id', type=str, default=None,
                       help='Subject ID to filter (e.g., S1, S2)')
    parser.add_argument('--trial_idx', type=int, default=None,
                       help='Specific trial index to visualize')
    parser.add_argument('--output_dir', type=str, default='visualization_output_raw',
                       help='Directory to save output figures')
    parser.add_argument('--max_trials', type=int, default=10,
                       help='Maximum number of trials to visualize')
    parser.add_argument('--sampling_rate', type=int, default=64,
                       help='Sampling rate of audio data (Hz)')
    
    args = parser.parse_args()
    
    # Create visualizer
    print(f"Initializing visualizer with data directory: {args.data_dir}")
    visualizer = FulsangRawAudioVisualizer(args.data_dir, sampling_rate=args.sampling_rate)
    
    # Load trial data
    print("\nLoading trial data from MATLAB files...")
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
    if args.trial_idx is not None and len(trials) == 1:
        # Single trial visualization
        output_path = Path(args.output_dir) / f"raw_trial_{args.trial_idx}.png" if args.output_dir else None
        visualizer.visualize_trial(trials[0], output_path=output_path, show_plot=True)
    else:
        # Multiple trials
        visualizer.visualize_multiple_trials(trials, max_trials=args.max_trials, 
                                            output_dir=args.output_dir)


if __name__ == '__main__':
    main()
