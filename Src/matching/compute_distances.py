"""
Script to compute pairwise EMD distances between shapes for all descriptors.

Creates distance matrices for each descriptor (A3, D1, D2, D3, D4) where:
- Rows and columns represent shapes
- Cell (i,j) contains the EMD distance between shape i and shape j
- Only lower triangle is computed (since EMD is symmetric)
- Diagonal contains zeros (distance from shape to itself)
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from shapeFeatures import Shape
from distance import ShapeDistance


def compute_pairwise_distances(csv_file_path, num_shapes=None, output_dir="distance_matrices"):
    """
    Compute pairwise EMD distances between shapes for all descriptors.
    
    Args:
        csv_file_path (str): Path to the CSV file containing shape features
        num_shapes (int, optional): Number of shapes to compare (default: all shapes)
        output_dir (str): Directory to save distance matrices
    """
    # Read the CSV to get the list of shapes
    print(f"Reading shape data from: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    
    # Get shape names
    shape_names = df['shape'].tolist()
    
    # Limit to specified number of shapes if requested
    if num_shapes is not None:
        shape_names = shape_names[:num_shapes]
        print(f"Computing distances for first {num_shapes} shapes")
    else:
        print(f"Computing distances for all {len(shape_names)} shapes")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # List of descriptors to compute distances for
    descriptors = ['A3', 'D1', 'D2', 'D3', 'D4']
    
    # Load all Shape objects first
    print("\nLoading shape objects...")
    shapes = []
    for shape_name in tqdm(shape_names, desc="Loading shapes"):
        try:
            shape = Shape(shape_name, csv_file_path)
            shapes.append(shape)
        except Exception as e:
            print(f"\nWarning: Could not load shape {shape_name}: {e}")
            shapes.append(None)
    
    # Filter out any shapes that failed to load
    valid_indices = [i for i, s in enumerate(shapes) if s is not None]
    shapes = [shapes[i] for i in valid_indices]
    shape_names = [shape_names[i] for i in valid_indices]
    
    print(f"\nSuccessfully loaded {len(shapes)} shapes")
    
    # Compute distances for each descriptor
    for descriptor in descriptors:
        print(f"\n{'='*60}")
        print(f"Computing {descriptor} distances...")
        print(f"{'='*60}")
        
        # Initialize distance matrix with NaN
        n = len(shapes)
        distance_matrix = np.full((n, n), np.nan)
        
        # Fill diagonal with zeros (sanity check)
        np.fill_diagonal(distance_matrix, 0.0)
        
        # Compute pairwise distances (only lower triangle)
        # Total number of comparisons (excluding diagonal)
        total_comparisons = n * (n - 1) // 2
        
        with tqdm(total=total_comparisons, desc=f"{descriptor} comparisons") as pbar:
            for i in range(n):
                for j in range(i):  # Only compute lower triangle (j < i)
                    try:
                        # Create distance calculator
                        dist_calc = ShapeDistance(shapes[i], shapes[j])
                        
                        # Compute EMD for this descriptor
                        distance = dist_calc.histogram_distance(descriptor)
                        
                        # Store in matrix (only lower triangle)
                        distance_matrix[i, j] = distance
                        
                    except Exception as e:
                        print(f"\nError computing distance between {shape_names[i]} and {shape_names[j]}: {e}")
                        distance_matrix[i, j] = np.nan
                    
                    pbar.update(1)
        
        # Create DataFrame with shape names as index and columns
        df_distances = pd.DataFrame(
            distance_matrix,
            index=shape_names,
            columns=shape_names
        )
        
        # Save to CSV
        output_file = os.path.join(output_dir, f"distances_{descriptor}.csv")
        df_distances.to_csv(output_file)
        print(f"Saved {descriptor} distance matrix to: {output_file}")
        
        # Print statistics
        valid_distances = distance_matrix[~np.isnan(distance_matrix) & (distance_matrix > 0)]
        if len(valid_distances) > 0:
            print(f"  Min distance: {np.min(valid_distances):.6f}")
            print(f"  Max distance: {np.max(valid_distances):.6f}")
            print(f"  Mean distance: {np.mean(valid_distances):.6f}")
            print(f"  Median distance: {np.median(valid_distances):.6f}")
    
    print(f"\n{'='*60}")
    print("All distance matrices computed successfully!")
    print(f"Output saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Compute pairwise EMD distances between shapes for all descriptors"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="final_006_cleaned.csv",
        help="Path to CSV file containing shape features (default: final_006_cleaned.csv)"
    )
    parser.add_argument(
        "--num-shapes",
        type=int,
        default=None,
        help="Number of shapes to compare (default: all shapes)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="distance_matrices",
        help="Directory to save distance matrices (default: distance_matrices)"
    )
    
    args = parser.parse_args()
    
    # Get script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Resolve CSV file path
    if not os.path.isabs(args.csv):
        csv_path = os.path.join(script_dir, args.csv)
    else:
        csv_path = args.csv
    
    # Resolve output directory path
    if not os.path.isabs(args.output_dir):
        output_path = os.path.join(script_dir, args.output_dir)
    else:
        output_path = args.output_dir
    
    # Run the computation
    compute_pairwise_distances(csv_path, args.num_shapes, output_path)
