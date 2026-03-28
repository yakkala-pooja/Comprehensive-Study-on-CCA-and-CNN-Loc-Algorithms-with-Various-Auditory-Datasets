#!/usr/bin/env python3
"""
Quick script to show where triggers are in the Fulsang dataset.

This script:
1. Inspects trigger values across the dataset
2. Shows a sample visualization with triggers clearly marked
3. Provides a summary of trigger locations

Usage:
    python show_fulsang_triggers.py [--tfrecord_dir DIR] [--subject_id SUBJECT]
"""

import argparse
from pathlib import Path
from inspect_fulsang_triggers import FulsangTriggerInspector
from visualize_fulsang_audio_triggers import FulsangAudioVisualizer

def main():
    parser = argparse.ArgumentParser(
        description='Show where triggers are in Fulsang dataset'
    )
    parser.add_argument('--tfrecord_dir', type=str, default='fulsang_preprocessed',
                       help='Directory containing TFRecord files')
    parser.add_argument('--subject_id', type=str, default=None,
                       help='Subject ID to filter (e.g., S1, S2)')
    parser.add_argument('--num_trials', type=int, default=5,
                       help='Number of trials to visualize')
    
    args = parser.parse_args()
    
    print("="*80)
    print("FULSANG DATASET TRIGGER INSPECTION")
    print("="*80)
    
    # Step 1: Inspect triggers
    print("\n[Step 1] Inspecting trigger information...")
    inspector = FulsangTriggerInspector(args.tfrecord_dir)
    trigger_data = inspector.inspect_all_triggers(subject_id=args.subject_id)
    
    if not trigger_data:
        print("No trigger data found!")
        return
    
    inspector.print_summary(trigger_data)
    
    # Step 2: Visualize sample trials with triggers
    print("\n[Step 2] Visualizing sample trials with triggers...")
    visualizer = FulsangAudioVisualizer(args.tfrecord_dir)
    
    # Get trials with triggers
    trials_with_triggers = [t for t in trigger_data if t['trigger'] is not None]
    
    if not trials_with_triggers:
        print("No trials with triggers found for visualization!")
        return
    
    print(f"\nVisualizing {min(args.num_trials, len(trials_with_triggers))} trials with triggers...")
    
    # Load trial data for visualization
    sample_trials = trials_with_triggers[:args.num_trials]
    
    # Create visualizations
    for i, trigger_info in enumerate(sample_trials):
        subject_id = trigger_info['subject_id']
        trial_idx = trigger_info['trial_idx']
        trigger_val = trigger_info['trigger']
        
        print(f"\n  Trial {i+1}: Subject {subject_id}, Trial {trial_idx}, Trigger: {trigger_val}")
        
        # Load and visualize this specific trial
        trials = visualizer.load_trial_data(subject_id=subject_id, trial_idx=trial_idx)
        
        if trials:
            output_path = f"trigger_visualization_S{subject_id}_T{trial_idx}_Trigger{trigger_val}.png"
            visualizer.visualize_trial(trials[0], output_path=output_path, show_plot=True)
            print(f"    Saved to: {output_path}")
        else:
            print(f"    Could not load trial data")
    
    print("\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    print(f"Triggers are stored as trial-level identifiers in the TFRecord files.")
    print(f"Each trial has a trigger value that marks the start of the trial.")
    print(f"In the visualizations, triggers appear as:")
    print(f"  - Red dashed vertical line at x=0 (trial start)")
    print(f"  - Text annotation showing 'Trigger: [value]'")
    print(f"  - Trigger value in the figure title")
    print("\nTrigger values may encode experimental conditions such as:")
    print("  - Attention direction (male/female)")
    print("  - Spatial position (left/right)")
    print("  - Acoustic condition (anechoic/reverb)")
    print("  - Number of speakers")
    print("="*80)


if __name__ == '__main__':
    main()
