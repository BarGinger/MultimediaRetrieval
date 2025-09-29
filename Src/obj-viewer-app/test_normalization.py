"""
Test script for Step 3.1: Full Normalization with Caching
"""

import numpy as np
import pandas as pd
from core.file_index import get_file_tree
from core.analysis_cache import merge_analysis_data
from core.shapeMesh import ShapeMesh

def test_normalization_methods():
    """Test the normalization pipeline on a few sample shapes"""
    
    print("Testing Step 3.1: Full Normalization Pipeline")
    print("=" * 50)
    
    # Get some test files
    file_df = get_file_tree()
    file_df = merge_analysis_data(file_df, "LabeledPSB")
    
    # Test on first few shapes
    test_shapes = file_df.head(3)
    
    for idx, (_, row) in enumerate(test_shapes.iterrows()):
        print(f"\nTesting shape {idx+1}: {row['filename']}")
        print("-" * 30)
        
        try:
            # Create mesh
            mesh = ShapeMesh.from_file_row(row)
            
            # Get normalization info
            norm_info = mesh.get_normalization_info()
            
            print(f"Original center: {norm_info['original']['center']}")
            print(f"Original dimensions: {norm_info['original']['bounding_box']['dimensions']}")
            print(f"Max original dimension: {np.max(norm_info['original']['bounding_box']['dimensions']):.3f}")
            
            print(f"\nAfter centering: {norm_info['after_centering']['center']}")
            
            print(f"\nPCA eigenvalues: {norm_info['pca']['eigenvalues']}")
            print(f"Explained variance ratio: {norm_info['pca']['explained_variance_ratio']}")
            
            if 'flipping' in norm_info:
                print(f"\nFlipping test values: {norm_info['flipping']['moment_test_values']}")
                print(f"Flip factors: {norm_info['flipping']['flip_factors']}")
            
            print(f"\nFinal center: {norm_info['final']['center']}")
            print(f"Final max dimension: {norm_info['final']['max_dimension']:.6f}")
            print(f"Final dimensions: {norm_info['final']['bounding_box']['dimensions']}")
            
            # Verify normalization properties
            success = True
            
            # Check if centered (should be very close to origin)
            center_error = np.linalg.norm(norm_info['final']['center'])
            if center_error > 1e-10:
                print(f"⚠️  WARNING: Shape not properly centered (error: {center_error:.2e})")
                success = False
            else:
                print("✅ Shape properly centered")
            
            # Check if scaled to unit size
            if abs(norm_info['final']['max_dimension'] - 1.0) > 1e-6:
                print(f"⚠️  WARNING: Shape not properly scaled (max dim: {norm_info['final']['max_dimension']:.6f})")
                success = False
            else:
                print("✅ Shape properly scaled to unit size")
            
            if success:
                print("✅ Normalization PASSED")
            else:
                print("❌ Normalization FAILED")
                
        except Exception as e:
            print(f"❌ ERROR processing {row['filename']}: {str(e)}")
            import traceback
            traceback.print_exc()

def test_cached_normalization():
    """Test the cached normalization system"""
    
    print("\n" + "=" * 50)
    print("Testing cached normalization system")
    print("=" * 50)
    
    from core.normalized_cache import normalized_cache
    
    # Get some test files
    file_df = get_file_tree()
    file_df = merge_analysis_data(file_df, "LabeledPSB")
    
    test_shape = file_df.head(1).iloc[0]
    
    print(f"Testing with: {test_shape['filename']}")
    
    # Check if normalized version is available
    is_available = normalized_cache.is_normalized_available(test_shape['filename'], "LabeledPSB")
    print(f"Cached normalized version available: {is_available}")
    
    if is_available:
        # Test loading cached version
        cached_mesh = normalized_cache.load_normalized_shape(test_shape['filename'], "LabeledPSB")
        if cached_mesh:
            print(f"Successfully loaded cached mesh: {len(cached_mesh.vertices)} vertices")
            
            # Verify it's normalized
            center = np.mean(cached_mesh.vertices, axis=0)
            dims = np.ptp(cached_mesh.vertices, axis=0)
            max_dim = np.max(dims)
            
            print(f"Cached mesh center: {center}")
            print(f"Cached mesh max dimension: {max_dim:.6f}")
            
            if np.linalg.norm(center) < 1e-10 and abs(max_dim - 1.0) < 1e-6:
                print("✅ Cached mesh is properly normalized")
            else:
                print("❌ Cached mesh normalization is incorrect")
        else:
            print("❌ Failed to load cached mesh")
    else:
        print("💡 Run preprocessing to generate cached normalized shapes:")
        print("   cd preprocessing")
        print("   python normalize_database.py")

def test_normalization_statistics():
    """Test normalization using statistics and histograms as recommended in technical tips"""
    
    print("\n" + "=" * 50)
    print("Testing normalization with statistics (as per technical tips)")
    print("=" * 50)
    
    import matplotlib.pyplot as plt
    
    # Get file data
    file_df = get_file_tree()
    file_df = merge_analysis_data(file_df, "LabeledPSB")
    
    # Test on larger sample
    test_shapes = file_df.head(20)  # Use more shapes for statistics
    
    # Collect normalization statistics
    original_centers = []
    normalized_centers = []
    original_max_dims = []
    normalized_max_dims = []
    original_bboxes = []
    normalized_bboxes = []
    
    print("Collecting normalization statistics...")
    
    for idx, (_, row) in enumerate(test_shapes.iterrows()):
        try:
            mesh = ShapeMesh.from_file_row(row)
            
            # Original stats
            orig_center = np.mean(mesh.vertices, axis=0)
            orig_dims = np.ptp(mesh.vertices, axis=0)
            orig_max_dim = np.max(orig_dims)
            
            # Normalized stats
            norm_vertices = mesh.apply_full_normalization()
            norm_center = np.mean(norm_vertices, axis=0)
            norm_dims = np.ptp(norm_vertices, axis=0)
            norm_max_dim = np.max(norm_dims)
            
            # Store for statistics
            original_centers.append(np.linalg.norm(orig_center))
            normalized_centers.append(np.linalg.norm(norm_center))
            original_max_dims.append(orig_max_dim)
            normalized_max_dims.append(norm_max_dim)
            original_bboxes.append(orig_dims)
            normalized_bboxes.append(norm_dims)
            
        except Exception as e:
            print(f"Error with {row['filename']}: {e}")
    
    # Print statistics as recommended in technical tips
    print(f"\nNormalization Statistics (from {len(original_centers)} shapes):")
    print("-" * 40)
    
    print(f"Original center distances from origin:")
    print(f"  Mean: {np.mean(original_centers):.3f}")
    print(f"  Max:  {np.max(original_centers):.3f}")
    print(f"  Min:  {np.min(original_centers):.3f}")
    
    print(f"\nNormalized center distances from origin:")
    print(f"  Mean: {np.mean(normalized_centers):.2e}")
    print(f"  Max:  {np.max(normalized_centers):.2e}")
    print(f"  Min:  {np.min(normalized_centers):.2e}")
    
    print(f"\nOriginal max dimensions:")
    print(f"  Mean: {np.mean(original_max_dims):.3f}")
    print(f"  Max:  {np.max(original_max_dims):.3f}")
    print(f"  Min:  {np.min(original_max_dims):.3f}")
    
    print(f"\nNormalized max dimensions:")
    print(f"  Mean: {np.mean(normalized_max_dims):.6f}")
    print(f"  Max:  {np.max(normalized_max_dims):.6f}")
    print(f"  Min:  {np.min(normalized_max_dims):.6f}")
    
    # Verification checks
    print(f"\n✅ Verification Checks:")
    center_errors = np.array(normalized_centers)
    scale_errors = np.abs(np.array(normalized_max_dims) - 1.0)
    
    centered_correctly = np.sum(center_errors < 1e-10)
    scaled_correctly = np.sum(scale_errors < 1e-6)
    
    print(f"  Shapes properly centered: {centered_correctly}/{len(center_errors)} ({centered_correctly/len(center_errors)*100:.1f}%)")
    print(f"  Shapes properly scaled:   {scaled_correctly}/{len(scale_errors)} ({scaled_correctly/len(scale_errors)*100:.1f}%)")
    
    if centered_correctly == len(center_errors) and scaled_correctly == len(scale_errors):
        print("  🎉 ALL SHAPES PASS NORMALIZATION TESTS!")
    else:
        print("  ⚠️  Some shapes failed normalization tests")

if __name__ == "__main__":
    test_normalization_methods()
    test_cached_normalization()
    test_normalization_statistics()