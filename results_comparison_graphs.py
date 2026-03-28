#!/usr/bin/env python3
"""
Results Comparison Graphs Generator

This script creates comprehensive comparison graphs between CNN-LOC and CCA results
for both DAS and Fulsang datasets. It handles different result formats and creates
visualizations for accuracy, correlation, and other performance metrics.

Author: AI Assistant
Date: 2025
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ResultsComparisonGraphs:
    """
    Generate comparison graphs for CNN-LOC and CCA results across datasets.
    """
    
    def __init__(self, output_dir: str = "comparison_graphs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize data storage
        self.cnn_results = {}
        self.cca_results = {}
        self.datasets = ['DAS', 'Fulsang']
        self.methods = ['CNN-LOC', 'CCA']
        
    def load_cnn_results(self, results_dir: str, dataset: str) -> Optional[Dict]:
        """Load CNN-LOC results from JSON file."""
        results_path = Path(results_dir) / "results.json"
        
        if not results_path.exists():
            print(f"Warning: CNN-LOC results not found for {dataset}: {results_path}")
            return None
            
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            # Extract key metrics
            cnn_data = {
                'accuracy': results.get('accuracy', 0.0),
                'loss': results.get('loss', 0.0),
                'best_val_acc': results.get('best_val_acc', 0.0),
                'timestamp': results.get('timestamp', 'Unknown'),
                'dataset': dataset,
                'method': 'CNN-LOC'
            }
            
            # Add comprehensive metrics if available
            if 'comprehensive_metrics' in results:
                comp_metrics = results['comprehensive_metrics']
                cnn_data.update({
                    'roc_auc': comp_metrics.get('roc_auc_metrics', {}).get('roc_auc_score', 0.0),
                    'precision': comp_metrics.get('macro_averages', {}).get('precision', 0.0),
                    'recall': comp_metrics.get('macro_averages', {}).get('recall', 0.0),
                    'f1_score': comp_metrics.get('macro_averages', {}).get('f1_score', 0.0)
                })
            
            # Add ROC-AUC metrics if available
            if 'roc_auc_metrics' in results:
                roc_metrics = results['roc_auc_metrics']
                cnn_data.update({
                    'roc_auc': roc_metrics.get('roc_auc_score', 0.0),
                    'average_precision': roc_metrics.get('average_precision', 0.0)
                })
            
            print(f"✓ Loaded CNN-LOC results for {dataset}: Accuracy = {cnn_data['accuracy']:.4f}")
            return cnn_data
            
        except Exception as e:
            print(f"Error loading CNN-LOC results for {dataset}: {e}")
            return None
    
    def load_cca_results(self, results_file: str, dataset: str) -> Optional[Dict]:
        """Load CCA results from text file."""
        results_path = Path(results_file)
        
        if not results_path.exists():
            print(f"Warning: CCA results not found for {dataset}: {results_path}")
            return None
            
        try:
            with open(results_path, 'r') as f:
                content = f.read()
            
            # Parse CCA results - handle different formats
            cca_data = {
                'dataset': dataset,
                'method': 'CCA',
                'correlation': 0.0,
                'p_value': 1.0,
                'jackknife_mean': 0.0,
                'jackknife_std': 0.0
            }
            
            # Extract correlation values using regex
            correlation_patterns = [
                r'EEG-Envelope Correlation:\s*([\d.-]+)',
                r'correlation:\s*([\d.-]+)',
                r'Correlation:\s*([\d.-]+)',
                r'Jackknife Results:\s*Regularization [\d.]+:\s*([\d.-]+)'
            ]
            
            for pattern in correlation_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    try:
                        cca_data['correlation'] = float(matches[0])
                        break
                    except ValueError:
                        continue
            
            # Extract p-values
            p_value_patterns = [
                r'EEG-Envelope P-value:\s*([\d.-]+)',
                r'p-value:\s*([\d.-]+)',
                r'P-value:\s*([\d.-]+)'
            ]
            
            for pattern in p_value_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    try:
                        cca_data['p_value'] = float(matches[0])
                        break
                    except ValueError:
                        continue
            
            # Extract jackknife results
            jackknife_pattern = r'Regularization [\d.]+:\s*([\d.-]+)\s*±\s*([\d.-]+)'
            jackknife_matches = re.findall(jackknife_pattern, content)
            if jackknife_matches:
                try:
                    cca_data['jackknife_mean'] = float(jackknife_matches[0][0])
                    cca_data['jackknife_std'] = float(jackknife_matches[0][1])
                except ValueError:
                    pass
            
            # Use correlation as accuracy equivalent for comparison
            cca_data['accuracy'] = abs(cca_data['correlation'])
            
            print(f"✓ Loaded CCA results for {dataset}: Correlation = {cca_data['correlation']:.4f}")
            return cca_data
            
        except Exception as e:
            print(f"Error loading CCA results for {dataset}: {e}")
            return None
    
    def load_all_results(self):
        """Load all available results from different directories."""
        print("Loading all available results...")
        
        # Define potential result directories
        result_dirs = {
            'DAS': {
                'cnn': ['cnn_loc_results_das', 'das_analysis_results_final'],
                'cca': ['das_analysis_results_final/cca_results_final.txt']
            },
            'Fulsang': {
                'cnn': ['fulsang_optimized_results', 'test_results'],
                'cca': ['fulsang_cca_results_with_real_attention.txt']
            }
        }
        
        # Load CNN-LOC results
        for dataset, dirs in result_dirs.items():
            for cnn_dir in dirs['cnn']:
                if Path(cnn_dir).exists():
                    cnn_data = self.load_cnn_results(cnn_dir, dataset)
                    if cnn_data:
                        self.cnn_results[dataset] = cnn_data
                        break
        
        # Load CCA results
        for dataset, dirs in result_dirs.items():
            for cca_file in dirs['cca']:
                if Path(cca_file).exists():
                    cca_data = self.load_cca_results(cca_file, dataset)
                    if cca_data:
                        self.cca_results[dataset] = cca_data
                        break
        
        print(f"Loaded CNN-LOC results for: {list(self.cnn_results.keys())}")
        print(f"Loaded CCA results for: {list(self.cca_results.keys())}")
    
    def create_accuracy_comparison(self):
        """Create accuracy comparison bar chart."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data
        datasets = []
        cnn_accuracies = []
        cca_accuracies = []
        
        for dataset in self.datasets:
            if dataset in self.cnn_results and dataset in self.cca_results:
                datasets.append(dataset)
                cnn_accuracies.append(self.cnn_results[dataset]['accuracy'])
                cca_accuracies.append(self.cca_results[dataset]['accuracy'])
        
        if not datasets:
            print("No data available for accuracy comparison")
            return
        
        # Create bar chart
        x = np.arange(len(datasets))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, cnn_accuracies, width, label='CNN-LOC', 
                      color='skyblue', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, cca_accuracies, width, label='CCA', 
                      color='lightcoral', alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy / Correlation', fontsize=14, fontweight='bold')
        ax.set_title('CNN-LOC vs CCA Performance Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'accuracy_comparison.pdf', bbox_inches='tight')
        plt.show()
        
        print("✓ Created accuracy comparison chart")
    
    def create_metrics_radar_chart(self):
        """Create radar chart comparing multiple metrics."""
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Define metrics to compare
        metrics = ['Accuracy', 'ROC-AUC', 'Precision', 'Recall', 'F1-Score']
        
        # Prepare data for each dataset and method
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
        
        for i, dataset in enumerate(self.datasets):
            for j, method in enumerate(self.methods):
                if dataset in self.cnn_results and method == 'CNN-LOC':
                    data = self.cnn_results[dataset]
                elif dataset in self.cca_results and method == 'CCA':
                    data = self.cca_results[dataset]
                else:
                    continue
                
                values = []
                for metric in metrics:
                    if metric == 'Accuracy':
                        values.append(data.get('accuracy', 0.0))
                    elif metric == 'ROC-AUC':
                        values.append(data.get('roc_auc', 0.0))
                    elif metric == 'Precision':
                        values.append(data.get('precision', 0.0))
                    elif metric == 'Recall':
                        values.append(data.get('recall', 0.0))
                    elif metric == 'F1-Score':
                        values.append(data.get('f1_score', 0.0))
                
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, 
                       label=f'{dataset} {method}', color=colors[i*2 + j])
                ax.fill(angles, values, alpha=0.25, color=colors[i*2 + j])
        
        # Customize the chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Performance Metrics Comparison\n(CNN-LOC vs CCA)', 
                    size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_radar_chart.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'metrics_radar_chart.pdf', bbox_inches='tight')
        plt.show()
        
        print("✓ Created metrics radar chart")
    
    def create_correlation_heatmap(self):
        """Create correlation heatmap between methods and datasets."""
        # Prepare data matrix
        datasets = []
        methods = []
        values = []
        
        for dataset in self.datasets:
            if dataset in self.cnn_results:
                datasets.append(f"{dataset} CNN-LOC")
                methods.append("CNN-LOC")
                values.append(self.cnn_results[dataset]['accuracy'])
            
            if dataset in self.cca_results:
                datasets.append(f"{dataset} CCA")
                methods.append("CCA")
                values.append(self.cca_results[dataset]['accuracy'])
        
        if not values:
            print("No data available for correlation heatmap")
            return
        
        # Create DataFrame
        df = pd.DataFrame({
            'Dataset': datasets,
            'Method': methods,
            'Performance': values
        })
        
        # Pivot for heatmap
        pivot_df = df.pivot_table(values='Performance', index='Dataset', columns='Method', fill_value=0)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot_df, annot=True, cmap='YlOrRd', fmt='.3f', 
                   cbar_kws={'label': 'Performance Score'}, ax=ax)
        
        ax.set_title('Performance Heatmap: CNN-LOC vs CCA', fontsize=16, fontweight='bold')
        ax.set_xlabel('Method', fontsize=14, fontweight='bold')
        ax.set_ylabel('Dataset', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'performance_heatmap.pdf', bbox_inches='tight')
        plt.show()
        
        print("✓ Created performance heatmap")
    
    def create_summary_table(self):
        """Create a summary table of all results."""
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        headers = ['Dataset', 'Method', 'Accuracy', 'Loss', 'ROC-AUC', 'Timestamp']
        
        for dataset in self.datasets:
            # CNN-LOC data
            if dataset in self.cnn_results:
                cnn_data = self.cnn_results[dataset]
                table_data.append([
                    dataset,
                    'CNN-LOC',
                    f"{cnn_data.get('accuracy', 0.0):.4f}",
                    f"{cnn_data.get('loss', 0.0):.4f}",
                    f"{cnn_data.get('roc_auc', 0.0):.4f}",
                    cnn_data.get('timestamp', 'Unknown')[:19] if cnn_data.get('timestamp') else 'Unknown'
                ])
            
            # CCA data
            if dataset in self.cca_results:
                cca_data = self.cca_results[dataset]
                table_data.append([
                    dataset,
                    'CCA',
                    f"{cca_data.get('accuracy', 0.0):.4f}",
                    f"{cca_data.get('loss', 0.0):.4f}",
                    f"{cca_data.get('roc_auc', 0.0):.4f}",
                    'N/A'
                ])
        
        if not table_data:
            print("No data available for summary table")
            return
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers, 
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.title('Results Summary Table', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'summary_table.pdf', bbox_inches='tight')
        plt.show()
        
        print("✓ Created summary table")
    
    def create_statistical_comparison(self):
        """Create statistical comparison plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Box plot comparison
        cnn_values = []
        cca_values = []
        
        for dataset in self.datasets:
            if dataset in self.cnn_results:
                cnn_values.append(self.cnn_results[dataset]['accuracy'])
            if dataset in self.cca_results:
                cca_values.append(self.cca_results[dataset]['accuracy'])
        
        if cnn_values and cca_values:
            ax1.boxplot([cnn_values, cca_values], labels=['CNN-LOC', 'CCA'])
            ax1.set_title('Performance Distribution Comparison', fontweight='bold')
            ax1.set_ylabel('Accuracy')
            ax1.grid(True, alpha=0.3)
        
        # 2. Scatter plot
        datasets_scatter = []
        cnn_scatter = []
        cca_scatter = []
        
        for dataset in self.datasets:
            if dataset in self.cnn_results and dataset in self.cca_results:
                datasets_scatter.append(dataset)
                cnn_scatter.append(self.cnn_results[dataset]['accuracy'])
                cca_scatter.append(self.cca_results[dataset]['accuracy'])
        
        if datasets_scatter:
            ax2.scatter(cnn_scatter, cca_scatter, s=100, alpha=0.7)
            for i, dataset in enumerate(datasets_scatter):
                ax2.annotate(dataset, (cnn_scatter[i], cca_scatter[i]), 
                           xytext=(5, 5), textcoords='offset points')
            
            # Add diagonal line
            min_val = min(min(cnn_scatter), min(cca_scatter))
            max_val = max(max(cnn_scatter), max(cca_scatter))
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
            
            ax2.set_xlabel('CNN-LOC Accuracy')
            ax2.set_ylabel('CCA Accuracy')
            ax2.set_title('Method Correlation', fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        # 3. Performance improvement
        improvements = []
        dataset_names = []
        
        for dataset in self.datasets:
            if dataset in self.cnn_results and dataset in self.cca_results:
                cnn_acc = self.cnn_results[dataset]['accuracy']
                cca_acc = self.cca_results[dataset]['accuracy']
                improvement = ((cnn_acc - cca_acc) / cca_acc) * 100 if cca_acc > 0 else 0
                improvements.append(improvement)
                dataset_names.append(dataset)
        
        if improvements:
            colors = ['green' if x > 0 else 'red' for x in improvements]
            bars = ax3.bar(dataset_names, improvements, color=colors, alpha=0.7)
            ax3.set_title('CNN-LOC vs CCA Improvement (%)', fontweight='bold')
            ax3.set_ylabel('Improvement (%)')
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax3.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, improvement in zip(bars, improvements):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                        f'{improvement:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        # 4. Method comparison pie chart
        if cnn_values and cca_values:
            avg_cnn = np.mean(cnn_values)
            avg_cca = np.mean(cca_values)
            
            sizes = [avg_cnn, avg_cca]
            labels = ['CNN-LOC', 'CCA']
            colors = ['skyblue', 'lightcoral']
            
            ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Average Performance Distribution', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'statistical_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'statistical_comparison.pdf', bbox_inches='tight')
        plt.show()
        
        print("✓ Created statistical comparison plots")
    
    def generate_all_graphs(self):
        """Generate all comparison graphs."""
        print("Generating comprehensive comparison graphs...")
        print(f"Output directory: {self.output_dir}")
        
        # Load all results
        self.load_all_results()
        
        if not self.cnn_results and not self.cca_results:
            print("No results found to generate graphs!")
            return
        
        # Generate all graphs
        self.create_accuracy_comparison()
        self.create_metrics_radar_chart()
        self.create_correlation_heatmap()
        self.create_summary_table()
        self.create_statistical_comparison()
        
        # Generate summary report
        self.generate_summary_report()
        
        print(f"\n✓ All graphs generated successfully!")
        print(f"✓ Output saved to: {self.output_dir}")
    
    def generate_summary_report(self):
        """Generate a text summary report."""
        report_path = self.output_dir / 'comparison_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("CNN-LOC vs CCA Results Comparison Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # CNN-LOC Results
            f.write("CNN-LOC Results:\n")
            f.write("-" * 20 + "\n")
            for dataset, data in self.cnn_results.items():
                f.write(f"{dataset} Dataset:\n")
                f.write(f"  Accuracy: {data.get('accuracy', 0.0):.4f}\n")
                f.write(f"  Loss: {data.get('loss', 0.0):.4f}\n")
                f.write(f"  ROC-AUC: {data.get('roc_auc', 0.0):.4f}\n")
                f.write(f"  Timestamp: {data.get('timestamp', 'Unknown')}\n\n")
            
            # CCA Results
            f.write("CCA Results:\n")
            f.write("-" * 15 + "\n")
            for dataset, data in self.cca_results.items():
                f.write(f"{dataset} Dataset:\n")
                f.write(f"  Correlation: {data.get('correlation', 0.0):.4f}\n")
                f.write(f"  P-value: {data.get('p_value', 1.0):.4f}\n")
                f.write(f"  Jackknife Mean: {data.get('jackknife_mean', 0.0):.4f}\n")
                f.write(f"  Jackknife Std: {data.get('jackknife_std', 0.0):.4f}\n\n")
            
            # Comparison
            f.write("Comparison Summary:\n")
            f.write("-" * 20 + "\n")
            for dataset in self.datasets:
                if dataset in self.cnn_results and dataset in self.cca_results:
                    cnn_acc = self.cnn_results[dataset]['accuracy']
                    cca_acc = self.cca_results[dataset]['accuracy']
                    improvement = ((cnn_acc - cca_acc) / cca_acc) * 100 if cca_acc > 0 else 0
                    
                    f.write(f"{dataset} Dataset:\n")
                    f.write(f"  CNN-LOC Accuracy: {cnn_acc:.4f}\n")
                    f.write(f"  CCA Accuracy: {cca_acc:.4f}\n")
                    f.write(f"  Improvement: {improvement:.2f}%\n\n")
        
        print("✓ Generated summary report")


def main():
    """Main function to run the comparison graph generator."""
    print("CNN-LOC vs CCA Results Comparison Graph Generator")
    print("=" * 55)
    
    # Create the graph generator
    generator = ResultsComparisonGraphs()
    
    # Generate all graphs
    generator.generate_all_graphs()
    
    print("\n" + "=" * 55)
    print("Graph generation completed successfully!")
    print("Check the 'comparison_graphs' directory for all output files.")


if __name__ == "__main__":
    main()
