"""
Example usage of the feature validation plotting script.

This script demonstrates how to run the validation analysis with your CSV files.
"""

import subprocess
import sys
from pathlib import Path

def run_validation_analysis():
    """Run the feature validation analysis."""
    
    # Define paths to your CSV files
    original_csv = "../../Preprocessing/analysis_results.csv"  # Original dataset analysis
    normalized_csv = "../../Datasets/UnifiedPreprocessed/analysis_results_data.csv"  # Normalized dataset analysis
    
    # Check if files exist
    if not Path(original_csv).exists():
        print(f"Error: Original CSV not found at {original_csv}")
        print("Please update the path to your original analysis results CSV")
        return
    
    if not Path(normalized_csv).exists():
        print(f"Error: Normalized CSV not found at {normalized_csv}")
        print("Please update the path to your normalized analysis results CSV")
        return
    
    # Run the validation script
    cmd = [
        sys.executable, 
        "feature_validation_plots.py",
        original_csv,
        normalized_csv,
        "--output-dir", "validation_results",
        "--show"  # Remove this if you don't want to show plots interactively
    ]
    
    print("Running feature validation analysis...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Analysis completed successfully!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running analysis: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")

if __name__ == "__main__":
    run_validation_analysis()