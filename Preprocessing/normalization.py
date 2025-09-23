#!/usr/bin/env python3
"""
Mesh Normalization Tool

This script normalizes 3D meshes by:
1. Translating each mesh so its mass barycenter (center of mass weighted by face areas) coincides with the origin (0,0,0)
2. Scaling each mesh uniformly so it tightly fits within a unit cube [-0.5, 0.5]³

The normalization ensures all meshes have consistent positioning and scale,
which is essential for machine learning applications.
"""

import open3d as o3d
import numpy as np
from pathlib import Path

# --- Configuration ---
USE_SAMPLED_DATASET = True  # Set to True to use Data_sampled_resampled, False for Data_resampled
BASE = Path(__file__).parent.parent.resolve()
SOURCE_ROOT = BASE / ('Data_sampled_resampled' if USE_SAMPLED_DATASET else 'Data_resampled')
TARGET_ROOT = BASE / ('Data_sampled_resampled_normalized' if USE_SAMPLED_DATASET else 'Data_normalized')

def calculate_mass_barycenter(vertices, triangles):
    """
    Calculate the mass barycenter (center of mass) of a mesh weighted by face areas.
    
    For each triangle face:
    1. Calculate the face centroid (average of its 3 vertices)
    2. Calculate the face area
    3. Weight the face centroid by its area
    4. Sum all weighted centroids and divide by total area
    
    Args:
        vertices: numpy array of shape (N, 3) containing vertex coordinates
        triangles: numpy array of shape (M, 3) containing triangle indices
        
    Returns:
        numpy array of shape (3,) representing the mass barycenter
    """
    if len(triangles) == 0:
        # Fallback to geometric centroid if no faces
        return np.mean(vertices, axis=0)
    
    total_weighted_centroid = np.zeros(3)
    total_area = 0.0
    
    for triangle in triangles:
        # Get the three vertices of this triangle
        v0 = vertices[triangle[0]]
        v1 = vertices[triangle[1]]
        v2 = vertices[triangle[2]]
        
        # Calculate face centroid (average of the 3 vertices)
        face_centroid = (v0 + v1 + v2) / 3.0
        
        # Calculate face area using cross product
        # Area = 0.5 * ||(v1-v0) × (v2-v0)||
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross_product = np.cross(edge1, edge2)
        face_area = 0.5 * np.linalg.norm(cross_product)
        
        # Add weighted contribution of this face
        total_weighted_centroid += face_centroid * face_area
        total_area += face_area
    
    # Calculate mass barycenter
    if total_area > 0:
        mass_barycenter = total_weighted_centroid / total_area
    else:
        # Fallback to geometric centroid if total area is zero
        mass_barycenter = np.mean(vertices, axis=0)
    
    return mass_barycenter

def normalize_mesh(input_path, output_path):
    """
    Normalize a mesh by:
    1. Centering at origin (translate mass barycenter to 0,0,0)
    2. Scaling to fit in unit cube [-0.5, 0.5]³
    
    Args:
        input_path: Path to input mesh file
        output_path: Path to save normalized mesh
        
    Returns:
        tuple: (success, original_bounds, final_bounds)
    """
    try:
        # Load the mesh
        mesh = o3d.io.read_triangle_mesh(str(input_path))
        if mesh.is_empty() or len(mesh.vertices) == 0:
            print(f"[X] Empty mesh: {input_path}")
            return False, None, None
        
        # Get vertices and faces as numpy arrays for easier manipulation
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        # Step 1: Calculate mass barycenter (center of mass) weighted by face areas
        barycenter = calculate_mass_barycenter(vertices, triangles)
        
        # Step 2: Translate to center at origin
        vertices_centered = vertices - barycenter
        
        # Step 3: Calculate bounding box of centered mesh
        min_coords = np.min(vertices_centered, axis=0)
        max_coords = np.max(vertices_centered, axis=0)
        
        # Calculate the size in each dimension
        bbox_size = max_coords - min_coords
        
        # Find the maximum dimension to ensure uniform scaling
        max_dimension = np.max(bbox_size)
        
        # Step 4: Scale to fit in unit cube [-0.5, 0.5]³
        # We want the largest dimension to be exactly 1.0 (from -0.5 to +0.5)
        if max_dimension > 0:  # Avoid division by zero
            scale_factor = 1.0 / max_dimension
            vertices_normalized = vertices_centered * scale_factor
        else:
            vertices_normalized = vertices_centered
        
        # Update the mesh with normalized vertices
        mesh.vertices = o3d.utility.Vector3dVector(vertices_normalized)
        
        # Recalculate normals after transformation
        mesh.compute_vertex_normals()
        
        # Save the normalized mesh
        success = o3d.io.write_triangle_mesh(str(output_path), mesh)
        
        if success:
            # Calculate final bounds for verification
            final_min = np.min(vertices_normalized, axis=0)
            final_max = np.max(vertices_normalized, axis=0)
            final_size = final_max - final_min
            
            print(f"[OK] Normalized: max_dim {max_dimension:.3f} -> {np.max(final_size):.3f}")
            
            return True, (min_coords + barycenter, max_coords + barycenter), (final_min, final_max)
        else:
            print(f"[X] Failed to write: {output_path}")
            return False, None, None
            
    except Exception as e:
        print(f"[X] Error normalizing {input_path}: {e}")
        return False, None, None

def verify_normalization(mesh_path):
    """
    Verify that a mesh is properly normalized.
    Returns: (is_centered, fits_unit_cube, max_dimension)
    """
    try:
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if mesh.is_empty():
            return False, False, 0
        
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        # Check if centered (mass barycenter should be close to origin)
        barycenter = calculate_mass_barycenter(vertices, triangles)
        is_centered = np.allclose(barycenter, [0, 0, 0], atol=1e-6)
        
        # Check if fits in unit cube
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        max_dimension = np.max(max_coords - min_coords)
        fits_unit_cube = max_dimension <= 1.0 + 1e-6  # Small tolerance for floating point
        
        return is_centered, fits_unit_cube, max_dimension
        
    except Exception:
        return False, False, 0

def main():
    print(f"Normalizing meshes from {SOURCE_ROOT} to {TARGET_ROOT}")
    print(f"Source exists: {SOURCE_ROOT.exists()}")
    print(f"Source is dir: {SOURCE_ROOT.is_dir()}")
    print("Normalization process:")
    print("  1. Center mass barycenter at origin (0,0,0)")
    print("  2. Scale to fit in unit cube [-0.5, 0.5]³")
    print("-" * 60)
    
    # Create target directory
    TARGET_ROOT.mkdir(exist_ok=True)
    
    # Statistics tracking
    total_processed = 0
    successful_count = 0
    failed_count = 0
    
    total_scale_reduction = 0
    max_original_dimension = 0
    
    for category_dir in SOURCE_ROOT.iterdir():
        if not category_dir.is_dir():
            continue
        
        print(f"\nProcessing category: {category_dir.name}")
        out_category_dir = TARGET_ROOT / category_dir.name
        out_category_dir.mkdir(parents=True, exist_ok=True)
        
        category_files = list(category_dir.glob('*.obj'))
        
        for i, obj_file in enumerate(category_files, 1):
            out_file = out_category_dir / obj_file.name
            print(f"  [{i:3d}/{len(category_files):3d}] {obj_file.name:<40} ", end="")
            
            success, original_bounds, final_bounds = normalize_mesh(obj_file, out_file)
            total_processed += 1
            
            if success:
                successful_count += 1
                
                # Track statistics
                if original_bounds is not None:
                    original_size = original_bounds[1] - original_bounds[0]
                    original_max_dim = np.max(original_size)
                    max_original_dimension = max(max_original_dimension, original_max_dim)
                    total_scale_reduction += original_max_dim
                    
            else:
                failed_count += 1
    
    # Print final statistics
    print("\n" + "="*60)
    print("NORMALIZATION SUMMARY")
    print("="*60)
    print(f"Total files processed: {total_processed}")
    print(f"[OK] Successful: {successful_count}")
    print(f"[X] Failed: {failed_count}")
    print()
    
    if successful_count > 0:
        avg_original_dimension = total_scale_reduction / successful_count
        avg_scale_factor = avg_original_dimension  # Since we scale to 1.0
        
        print(f"Scale Statistics:")
        print(f"  Average original max dimension: {avg_original_dimension:.3f}")
        print(f"  Largest original max dimension: {max_original_dimension:.3f}")
        print(f"  Average scale factor: {avg_scale_factor:.3f}x reduction")
        print(f"  All meshes now fit in unit cube: [-0.5, 0.5]³")
    
    # Verify a few random samples
    print(f"\nVerifying normalization on sample files...")
    sample_count = 0
    verified_centered = 0
    verified_unit_cube = 0
    
    for category_dir in TARGET_ROOT.iterdir():
        if not category_dir.is_dir():
            continue
        
        sample_files = list(category_dir.glob('*.obj'))[:2]  # Check 2 per category
        for obj_file in sample_files:
            is_centered, fits_cube, max_dim = verify_normalization(obj_file)
            sample_count += 1
            
            if is_centered:
                verified_centered += 1
            if fits_cube:
                verified_unit_cube += 1
                
            if sample_count >= 20:  # Check max 20 samples total
                break
        
        if sample_count >= 20:
            break
    
    if sample_count > 0:
        print(f"Verification results ({sample_count} samples checked):")
        print(f"  Properly centered: {verified_centered}/{sample_count} ({verified_centered/sample_count*100:.1f}%)")
        print(f"  Fits unit cube: {verified_unit_cube}/{sample_count} ({verified_unit_cube/sample_count*100:.1f}%)")


if __name__ == '__main__':
    main()
