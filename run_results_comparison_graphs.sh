#!/bin/bash
#SBATCH --job-name=results_comparison_graphs
#SBATCH --output=results_comparison_graphs_%j.out
#SBATCH --error=results_comparison_graphs_%j.err
#SBATCH --time=01:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=tier3
#SBATCH --account=neurosteer

# Results Comparison Graphs Runner
# This script runs the results comparison graph generator

echo "=================================================================================="
echo "RUNNING CNN-LOC vs CCA RESULTS COMPARISON GRAPH GENERATOR"
echo "=================================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"

# Load required modules (adjust based on your cluster)
# module load python/3.8
# module load matplotlib
# module load numpy

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MPLBACKEND=Agg  # Use non-interactive backend for matplotlib

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Check if required Python packages are available
echo "Checking Python dependencies..."
python -c "
import sys
required_packages = ['numpy', 'matplotlib', 'seaborn', 'pandas', 'json', 'pathlib']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f'Missing packages: {missing_packages}')
    print('Please install missing packages using: pip install ' + ' '.join(missing_packages))
    sys.exit(1)
else:
    print('All required packages are available')
"

if [ $? -ne 0 ]; then
    echo "Error: Missing required Python packages"
    exit 1
fi

# Run the graph generator
echo ""
echo "Running results comparison graph generator..."
python results_comparison_graphs.py

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "=================================================================================="
    echo "GRAPH GENERATION COMPLETED SUCCESSFULLY!"
    echo "Finished at: $(date)"
    echo "=================================================================================="
    
    # Check for generated files
    if [ -d "comparison_graphs" ]; then
        echo ""
        echo "Generated files in 'comparison_graphs' directory:"
        ls -la comparison_graphs/
        
        echo ""
        echo "Generated graphs:"
        echo "  - accuracy_comparison.png/pdf: Bar chart comparing CNN-LOC vs CCA accuracy"
        echo "  - metrics_radar_chart.png/pdf: Radar chart of multiple performance metrics"
        echo "  - performance_heatmap.png/pdf: Heatmap showing performance across datasets"
        echo "  - summary_table.png/pdf: Summary table of all results"
        echo "  - statistical_comparison.png/pdf: Statistical analysis plots"
        echo "  - comparison_summary.txt: Text summary of all results"
    else
        echo "Warning: No 'comparison_graphs' directory found"
    fi
else
    echo ""
    echo "=================================================================================="
    echo "GRAPH GENERATION FAILED!"
    echo "Exit code: $exit_code"
    echo "Finished at: $(date)"
    echo "=================================================================================="
fi

echo ""
echo "To view the generated graphs, open the PNG files in the 'comparison_graphs' directory"
echo "or check the PDF files for publication-quality figures."
echo ""
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
