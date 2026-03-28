#!/usr/bin/env python3
"""
Visualization script for MWF artifact removal results.

This script creates comprehensive visualizations comparing EEG signals
before and after MWF filtering for both Das and Fuglsang datasets.
"""

import numpy as np
import scipy.io as sio
from scipy import signal
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_mwf_data(dataset_type: str, subject_id: str, data_dir: Path):
    """Load MWF-cleaned data and original data for comparison."""
    if dataset_type.lower() == 'das':
        # Load MWF cleaned data
        mwf_file = data_dir / f"{subject_id}_MWF.mat"
        if not mwf_file.exists():
            raise FileNotFoundError(f"MWF file not found: {mwf_file}")
        
        mwf_data = sio.loadmat(str(mwf_file), squeeze_me=True, struct_as_record=False)
        
        # Load original data for comparison
        original_file = Path("Data/Das/4004271") / f"{subject_id}.mat"
        if not original_file.exists():
            logger.warning(f"Original file not found: {original_file}, using MWF data only")
            original_data = None
        else:
            original_data = sio.loadmat(str(original_file), squeeze_me=True, struct_as_record=False)
        
        return mwf_data, original_data
        
    elif dataset_type.lower() == 'fuglsang':
        # Load MWF cleaned data
        subject_num = int(subject_id.replace('S', '').replace('sub', ''))
        mwf_file = data_dir / f"sub{subject_num:02d}_MWF.mat"
        if not mwf_file.exists():
            raise FileNotFoundError(f"MWF file not found: {mwf_file}")
        
        mwf_data = sio.loadmat(str(mwf_file), squeeze_me=True, struct_as_record=False)
        
        # Load original data for comparison
        original_file = Path("/home/py9363/telluride_decoding/Data/Fulsang/EEG") / f"S{subject_num}.mat"
        if not original_file.exists():
            logger.warning(f"Original file not found: {original_file}, using MWF data only")
            original_data = None
        else:
            original_data = sio.loadmat(str(original_file), squeeze_me=True, struct_as_record=False)
        
        return mwf_data, original_data
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def compute_psd(eeg_data: np.ndarray, sample_rate: float, channel_idx: int = 0):
    """Compute power spectral density for a channel."""
    if len(eeg_data.shape) == 2:
        channel_data = eeg_data[:, channel_idx]
    else:
        channel_data = eeg_data
    
    freqs, psd = signal.welch(channel_data, fs=sample_rate, nperseg=min(2048, len(channel_data)))
    return freqs, psd


def plot_mwf_comparison(mwf_data: dict, original_data: dict, 
                       dataset_type: str, subject_id: str, 
                       trial_idx: int = 0, output_dir: Path = None):
    """
    Create comprehensive visualization comparing before/after MWF filtering.
    
    Args:
        mwf_data: MWF-cleaned data dictionary
        original_data: Original data dictionary (can be None)
        dataset_type: 'Das' or 'Fuglsang'
        subject_id: Subject identifier
        trial_idx: Trial index to visualize
        output_dir: Output directory for figures
    """
    if output_dir is None:
        output_dir = Path("Results/MWF_verification")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract trial data
    if 'trials' in mwf_data:
        trials = mwf_data['trials']
        if not isinstance(trials, np.ndarray):
            trials = [trials]
        else:
            trials = trials.flatten()
        
        if trial_idx >= len(trials):
            trial_idx = 0
        
        trial = trials[trial_idx]
        eeg_after = trial['eeg_data'] if hasattr(trial, 'eeg_data') else trial[0]['eeg_data']
        sample_rate = trial['sample_rate'] if hasattr(trial, 'sample_rate') else trial[0]['sample_rate']
        
        if isinstance(eeg_after, np.ndarray) and len(eeg_after.shape) == 2:
            pass  # Already 2D
        else:
            eeg_after = np.array(eeg_after)
            if len(eeg_after.shape) == 1:
                eeg_after = eeg_after.reshape(-1, 1)
    else:
        logger.error("Could not extract trial data from MWF file")
        return
    
    # Get original data for comparison
    eeg_before = None
    if original_data is not None:
        try:
            if dataset_type.lower() == 'das':
                if 'trials' in original_data:
                    orig_trials = original_data['trials']
                    if not isinstance(orig_trials, np.ndarray):
                        orig_trials = [orig_trials]
                    else:
                        orig_trials = orig_trials.flatten()
                    
                    if trial_idx < len(orig_trials):
                        orig_trial = orig_trials[trial_idx]
                        eeg_before = orig_trial.RawData.EegData
                        # Ensure same shape
                        min_samples = min(eeg_before.shape[0], eeg_after.shape[0])
                        eeg_before = eeg_before[:min_samples, :]
                        eeg_after = eeg_after[:min_samples, :]
            
            elif dataset_type.lower() == 'fuglsang':
                if 'data' in original_data:
                    data = original_data['data']
                    if hasattr(data, 'eeg'):
                        eeg_trials = data.eeg
                        if not isinstance(eeg_trials, list):
                            eeg_trials = [eeg_trials]
                        
                        if trial_idx < len(eeg_trials):
                            eeg_before = np.array(eeg_trials[trial_idx])
                            # Downsample to match MWF data (128 Hz)
                            if eeg_before.shape[0] > eeg_after.shape[0] * 4:
                                # Need to downsample
                                downsample_factor = 4
                                n_samples = eeg_before.shape[0] // downsample_factor
                                eeg_before_downsampled = np.zeros((n_samples, eeg_before.shape[1]))
                                for ch in range(eeg_before.shape[1]):
                                    eeg_before_downsampled[:, ch] = signal.decimate(
                                        eeg_before[:, ch], downsample_factor, ftype='iir'
                                    )
                                eeg_before = eeg_before_downsampled
                            
                            # Ensure same shape
                            min_samples = min(eeg_before.shape[0], eeg_after.shape[0])
                            eeg_before = eeg_before[:min_samples, :]
                            eeg_after = eeg_after[:min_samples, :]
        except Exception as e:
            logger.warning(f"Could not load original data for comparison: {e}")
            eeg_before = None
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    
    if eeg_before is not None:
        # 1. Time series comparison
        ax1 = plt.subplot(3, 2, 1)
        n_samples_plot = min(int(5 * sample_rate), eeg_before.shape[0])
        n_channels_plot = min(5, eeg_before.shape[1])
        
        time_axis = np.arange(n_samples_plot) / sample_rate
        
        for ch in range(n_channels_plot):
            offset = ch * np.std(eeg_before[:, ch]) * 3
            ax1.plot(time_axis, eeg_before[:n_samples_plot, ch] + offset, 
                    'b-', alpha=0.6, linewidth=0.5, label='Before' if ch == 0 else '')
            ax1.plot(time_axis, eeg_after[:n_samples_plot, ch] + offset, 
                    'r-', alpha=0.8, linewidth=0.5, label='After' if ch == 0 else '')
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Channel (offset)')
        ax1.set_title(f'EEG Time Series (First 5s, {n_channels_plot} channels)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Power Spectral Density
        ax2 = plt.subplot(3, 2, 2)
        freqs_before, psd_before = compute_psd(eeg_before, sample_rate, 0)
        freqs_after, psd_after = compute_psd(eeg_after, sample_rate, 0)
        
        ax2.semilogy(freqs_before, psd_before, 'b-', alpha=0.7, label='Before MWF', linewidth=2)
        ax2.semilogy(freqs_after, psd_after, 'r-', alpha=0.7, label='After MWF', linewidth=2)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Power Spectral Density')
        ax2.set_title('Power Spectral Density (Channel 0)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, min(50, sample_rate / 2))
        
        # 3. Channel variance reduction
        ax3 = plt.subplot(3, 2, 3)
        var_before = np.var(eeg_before, axis=0)
        var_after = np.var(eeg_after, axis=0)
        variance_reduction = (var_before - var_after) / var_before * 100
        
        channels = np.arange(len(var_before))
        ax3.bar(channels, variance_reduction, alpha=0.7, color='green')
        ax3.set_xlabel('Channel')
        ax3.set_ylabel('Variance Reduction (%)')
        ax3.set_title('Variance Reduction per Channel')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Average variance
        ax4 = plt.subplot(3, 2, 4)
        mean_var_before = np.mean(var_before)
        mean_var_after = np.mean(var_after)
        
        ax4.bar(['Before MWF', 'After MWF'], [mean_var_before, mean_var_after], 
                color=['blue', 'red'], alpha=0.7)
        ax4.set_ylabel('Mean Variance')
        ax4.set_title('Average Variance Across Channels')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Amplitude distribution
        ax5 = plt.subplot(3, 2, 5)
        ax5.hist(eeg_before.flatten(), bins=50, alpha=0.6, label='Before', color='blue', density=True)
        ax5.hist(eeg_after.flatten(), bins=50, alpha=0.6, label='After', color='red', density=True)
        ax5.set_xlabel('Amplitude')
        ax5.set_ylabel('Density')
        ax5.set_title('Amplitude Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Summary statistics
        ax6 = plt.subplot(3, 2, 6)
        ax6.axis('off')
        
        stats_text = f"""
        MWF Processing Summary
        
        Dataset: {dataset_type}
        Subject: {subject_id}
        Trial: {trial_idx}
        Sampling Rate: {sample_rate} Hz
        
        Before MWF:
          Shape: {eeg_before.shape}
          Mean: {np.mean(eeg_before):.4f}
          Std: {np.std(eeg_before):.4f}
          Variance: {np.var(eeg_before):.4f}
        
        After MWF:
          Shape: {eeg_after.shape}
          Mean: {np.mean(eeg_after):.4f}
          Std: {np.std(eeg_after):.4f}
          Variance: {np.var(eeg_after):.4f}
        
        Variance Reduction: {np.mean(variance_reduction):.2f}%
        """
    else:
        # Only after data available
        ax1 = plt.subplot(2, 2, 1)
        n_samples_plot = min(int(5 * sample_rate), eeg_after.shape[0])
        n_channels_plot = min(5, eeg_after.shape[1])
        
        time_axis = np.arange(n_samples_plot) / sample_rate
        
        for ch in range(n_channels_plot):
            offset = ch * np.std(eeg_after[:, ch]) * 3
            ax1.plot(time_axis, eeg_after[:n_samples_plot, ch] + offset, 
                    'r-', alpha=0.8, linewidth=0.5)
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Channel (offset)')
        ax1.set_title(f'EEG Time Series After MWF (First 5s, {n_channels_plot} channels)')
        ax1.grid(True, alpha=0.3)
        
        # PSD
        ax2 = plt.subplot(2, 2, 2)
        freqs_after, psd_after = compute_psd(eeg_after, sample_rate, 0)
        ax2.semilogy(freqs_after, psd_after, 'r-', alpha=0.7, label='After MWF', linewidth=2)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Power Spectral Density')
        ax2.set_title('Power Spectral Density (Channel 0)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, min(50, sample_rate / 2))
        
        # Variance
        ax3 = plt.subplot(2, 2, 3)
        var_after = np.var(eeg_after, axis=0)
        channels = np.arange(len(var_after))
        ax3.bar(channels, var_after, alpha=0.7, color='red')
        ax3.set_xlabel('Channel')
        ax3.set_ylabel('Variance')
        ax3.set_title('Variance per Channel (After MWF)')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Summary
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')
        stats_text = f"""
        MWF Processing Summary
        
        Dataset: {dataset_type}
        Subject: {subject_id}
        Trial: {trial_idx}
        Sampling Rate: {sample_rate} Hz
        
        After MWF:
          Shape: {eeg_after.shape}
          Mean: {np.mean(eeg_after):.4f}
          Std: {np.std(eeg_after):.4f}
          Variance: {np.var(eeg_after):.4f}
        """
    
    ax6 = plt.subplot(3, 2, 6) if eeg_before is not None else ax4
    ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', horizontalalignment='left')
    
    plt.suptitle(f'MWF Artifact Removal: {dataset_type} - {subject_id} - Trial {trial_idx}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / f"{dataset_type}_{subject_id}_trial{trial_idx}_MWF_verification.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visualization to {output_file}")


def main():
    """Main function to create visualizations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize MWF results')
    parser.add_argument('--dataset', type=str, choices=['das', 'fuglsang'], required=True,
                       help='Dataset type')
    parser.add_argument('--subject', type=str, required=True,
                       help='Subject ID (e.g., S1 or sub01)')
    parser.add_argument('--trial', type=int, default=0,
                       help='Trial index to visualize')
    parser.add_argument('--mwf_dir', type=str, 
                       default=None,
                       help='Directory containing MWF-cleaned files')
    parser.add_argument('--output_dir', type=str, default='Results/MWF_verification',
                       help='Output directory for figures')
    
    args = parser.parse_args()
    
    # Determine MWF directory
    if args.mwf_dir is None:
        if args.dataset.lower() == 'das':
            args.mwf_dir = 'MWF_cleaned_DAS'
        else:
            args.mwf_dir = 'MWF_cleaned_Fuglsang'
    
    mwf_dir = Path(args.mwf_dir)
    output_dir = Path(args.output_dir)
    
    try:
        mwf_data, original_data = load_mwf_data(args.dataset, args.subject, mwf_dir)
        plot_mwf_comparison(mwf_data, original_data, args.dataset, args.subject, 
                          args.trial, output_dir)
        logger.info("Visualization complete!")
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        raise


if __name__ == '__main__':
    main()

