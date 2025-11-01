#!/usr/bin/env python3

import sys
import os

# Add the Src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from matching.corrected_zscore_shape_query import CorrectedZScoreShapeQuery

def test_save_combined():
    """Test that the combined matrix is saved correctly."""
    print("Testing combined matrix saving...")
    
    # Load optimized weights
    optimized_weights = {
        'compactness': 0.0747,
        'convexity': 0.0914, 
        'diameter': 0.0993,
        'eccentricity': 0.1253,
        'rectangularity': 0.0980,
        'surface_area': 0.1134,
        'A3_hist': 0.0275,
        'D1_hist': 0.0876,
        'D2_hist': 0.0935,
        'D3_hist': 0.1131,
        'D4_hist': 0.0762
    }
    
    # Initialize with optimized weights
    query_system = CorrectedZScoreShapeQuery(
        csv_file_path="final_006_cleaned.csv",
        cache_dir="distance_matrices_zscore_corrected_full",
        weights=optimized_weights,
        combination_method="weighted_sum",
        debug=True
    )
    
    # Perform a single query to trigger matrix computation and saving
    try:
        results = query_system.query("m0", k=5)
        print("\\nQuery completed successfully!")
        print(f"Combined matrix shape: {query_system.combined_distance_matrix.shape}")
        
        # Check if file was saved
        cache_dir = "distance_matrices_zscore_corrected_full"
        combined_file = os.path.join(cache_dir, "combined_distance_matrix_weighted_sum.csv")
        if os.path.exists(combined_file):
            print(f"✅ Combined matrix successfully saved to: {combined_file}")
        else:
            print(f"❌ Combined matrix file not found at: {combined_file}")
            
    except Exception as e:
        print(f"Error during query: {e}")

if __name__ == "__main__":
    test_save_combined()