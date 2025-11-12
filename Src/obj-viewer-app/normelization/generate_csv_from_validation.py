"""
Generate normalization_statistics.csv from existing validation_detailed.json

This script reads the already-saved validation data and generates the CSV
without needing to reprocess all the meshes.

Usage:
    python generate_csv_from_validation.py
"""

import json
import csv
from pathlib import Path

def generate_normalization_csv(validation_json_path, output_csv_path):
    """
    Generate normalization_statistics.csv from validation_detailed.json
    
    Args:
        validation_json_path: Path to validation_detailed.json
        output_csv_path: Path where to save normalization_statistics.csv
    """
    print(f"Loading validation data from: {validation_json_path}")
    
    # Load the validation JSON
    with open(validation_json_path, 'r') as f:
        validation_data = json.load(f)
    
    # Extract the detailed validations list
    all_validations = validation_data.get('detailed_validations', [])
    
    if not all_validations:
        print("No detailed validations found in the JSON file!")
        return False
    
    print(f"Found {len(all_validations)} validations")
    
    # Prepare CSV data in the same format as normalization.py
    csv_data = []
    for i, validation in enumerate(all_validations, 1):
        # Extract the same metrics as normalization.py
        centering_validation = validation.get('centering_validation', {})
        scaling_validation = validation.get('scaling_validation', {})
        
        # Get barycenter distances before/after translation
        bary_before = centering_validation.get('original', {}).get('distance_from_origin', 0)
        bary_after = centering_validation.get('translated', {}).get('distance_from_origin', 0)
        
        # Get bbox dimensions before/after scaling
        bbox_before = validation.get('bbox_before_scaling', 0.0)
        bbox_after = scaling_validation.get('max_dimension', 1.0)
        
        row = {
            'mesh_index': i,
            'bary_before_translation': float(bary_before),
            'bary_after_translation': float(bary_after),
            'bbox_before_scaling': float(bbox_before),
            'bbox_after_scaling': float(bbox_after)
        }
        csv_data.append(row)
    
    # Write to CSV file
    print(f"Writing CSV to: {output_csv_path}")
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['mesh_index', 'bary_before_translation', 'bary_after_translation', 
                     'bbox_before_scaling', 'bbox_after_scaling']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"Successfully generated normalization_statistics.csv with {len(csv_data)} rows!")
    
    # Print sample statistics
    print(f"\n Sample Statistics:")
    print(f"   First row: {csv_data[0]}")
    if len(csv_data) > 1:
        print(f"   Last row:  {csv_data[-1]}")
    
    return True


if __name__ == "__main__":
    # Define paths
    base_dir = Path(__file__).parent.parent.parent.parent / "Datasets" / "UnifiedPreprocessed"
    
    # Process each dataset that has validation_detailed.json
    datasets = ['Data_sampled']  # Add more datasets as needed
    
    for dataset_name in datasets:
        dataset_dir = base_dir / dataset_name
        validation_json = dataset_dir / "validation_detailed.json"
        output_csv = dataset_dir / "normalization_statistics.csv"
        
        if not validation_json.exists():
            print(f"️ Skipping {dataset_name}: validation_detailed.json not found")
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*70}")
        
        success = generate_normalization_csv(validation_json, output_csv)
        
        if success:
            print(f"CSV saved to: {output_csv}")
        else:
            print(f"Failed to generate CSV for {dataset_name}")
    
    print(f"\n{'='*70}")
    print("All done!")
    print(f"{'='*70}")
