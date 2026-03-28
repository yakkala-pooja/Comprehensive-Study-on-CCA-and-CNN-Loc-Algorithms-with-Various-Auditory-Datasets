#!/usr/bin/env python3
"""
Simple example script to visualize Fulsang audio signals with triggers.

This is a simplified example showing how to use the visualization script.
"""

from visualize_fulsang_audio_triggers import FulsangAudioVisualizer
from pathlib import Path

# Example 1: Visualize a specific subject and trial
print("Example 1: Visualizing specific subject and trial")
print("=" * 60)

# Initialize visualizer
tfrecord_dir = "fulsang_preprocessed"  # Adjust this path to your TFRecord directory
visualizer = FulsangAudioVisualizer(tfrecord_dir)

# Load trials for a specific subject
trials = visualizer.load_trial_data(subject_id="S1", trial_idx=0)

if trials:
    # Visualize the first trial
    visualizer.visualize_trial(trials[0], output_path="example_trial_visualization.png")
    print(f"Visualized trial: Subject {trials[0]['subject_id']}, Trial {trials[0]['trial_idx']}")
else:
    print("No trials found. Make sure the TFRecord directory path is correct.")

# Example 2: Create a summary plot of multiple trials
print("\nExample 2: Creating summary plot")
print("=" * 60)

# Load multiple trials
all_trials = visualizer.load_trial_data()

if all_trials:
    # Create summary plot
    visualizer.create_summary_plot(all_trials[:9], output_path="example_summary_plot.png")
    print(f"Created summary plot with {min(len(all_trials), 9)} trials")
else:
    print("No trials found.")

# Example 3: Visualize multiple trials and save to directory
print("\nExample 3: Visualizing multiple trials")
print("=" * 60)

if all_trials:
    output_dir = "visualization_output"
    visualizer.visualize_multiple_trials(all_trials[:5], max_trials=5, output_dir=output_dir)
    print(f"Saved visualizations to {output_dir}/")
