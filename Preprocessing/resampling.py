import os
import open3d as o3d
import numpy as np
from pathlib import Path

# --- Toggle for sampled dataset ---
USE_SAMPLED_DATASET = True  # Set to True to use Data_sampled, False for full Data
BASE = Path(__file__).parent.parent.resolve()
SOURCE_ROOT = BASE / ('Datasets/Data_sampled' if USE_SAMPLED_DATASET else 'Datasets/Data')
TARGET_ROOT = BASE / ('Datasets/Data_sampled_resampled' if USE_SAMPLED_DATASET else 'Datasets/Data_resampled')
TARGET_VERTEX_COUNT = 5000
TOLERANCE = 0.25  # 25% tolerance for resampling


def edge_split_subdivision(mesh, target_vertices):
    """
    Feature-preserving subdivision using edge splitting instead of smoothing.
    This method preserves sharp edges and geometric features better than loop subdivision.
    """
    import copy
    
    # Make a copy to avoid modifying the original
    result_mesh = copy.deepcopy(mesh)
    current_count = len(result_mesh.vertices)
    
    # Calculate how many iterations we might need
    # Edge splitting roughly doubles vertex count per iteration
    multiplier_needed = target_vertices / current_count
    iterations = max(1, int(np.log2(multiplier_needed)))
    iterations = min(iterations, 3)  # Limit to prevent excessive subdivision
    
    print(f"      Edge split subdivision: {iterations} iterations planned")
    
    for i in range(iterations):
        try:
            # Use midpoint subdivision as it's more feature-preserving
            result_mesh = result_mesh.subdivide_midpoint(number_of_iterations=1)
            new_count = len(result_mesh.vertices)
            print(f"        Iteration {i+1}: {current_count} -> {new_count} vertices")
            current_count = new_count
            
            # Stop if we've reached our target
            if current_count >= target_vertices * 0.8:
                break
                
        except Exception as e:
            print(f"      Edge split iteration {i+1} failed: {e}")
            break
    
    return result_mesh


def controlled_remeshing(mesh, target_vertices):
    """
    Alternative feature-preserving remeshing approach.
    Uses mesh repair and careful subdivision.
    """
    try:
        # First ensure mesh is manifold and clean
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        # If mesh is very sparse, use multiple careful subdivisions
        current_vertices = len(mesh.vertices)
        result_mesh = mesh
        
        # Use iterative approach with small steps
        while len(result_mesh.vertices) < target_vertices * 0.8:
            # Check if we can safely do midpoint subdivision
            test_mesh = result_mesh.subdivide_midpoint(number_of_iterations=1)
            
            if len(test_mesh.vertices) <= target_vertices * 1.4:
                result_mesh = test_mesh
                print(f"      Controlled remesh step: {len(result_mesh.vertices)} vertices")
            else:
                # Would overshoot, stop here
                break
        
        return result_mesh
        
    except Exception as e:
        print(f"      Controlled remeshing failed: {e}")
        return mesh


def iterative_decimation(mesh, target_vertices, max_iterations=5):
    """
    Iteratively decimate mesh to precisely reach target vertex count.
    Uses binary search approach to find the right face count.
    """
    current_vertices = len(mesh.vertices)
    current_faces = len(mesh.triangles)
    
    if current_vertices <= target_vertices:
        return mesh
    
    print(f"      Starting iterative decimation: {current_vertices} -> {target_vertices} vertices")
    
    # Initial bounds for binary search
    min_faces = max(100, int(target_vertices * 0.5))  # Conservative minimum
    max_faces = current_faces
    
    best_mesh = mesh
    best_vertex_count = current_vertices
    best_error = abs(current_vertices - target_vertices)
    
    for iteration in range(max_iterations):
        # Binary search for optimal face count
        target_faces = (min_faces + max_faces) // 2
        
        try:
            # Try decimation with current target_faces
            test_mesh = mesh.simplify_quadric_decimation(target_faces)
            test_vertices = len(test_mesh.vertices)
            
            # Calculate error
            error = abs(test_vertices - target_vertices)
            
            print(f"        Iteration {iteration + 1}: {target_faces} faces -> {test_vertices} vertices (error: {error})")
            
            # Update best result if this is better
            if error < best_error:
                best_mesh = test_mesh
                best_vertex_count = test_vertices
                best_error = error
            
            # Check if we're close enough
            tolerance_range = target_vertices * 0.05  # 5% tolerance for convergence
            if error <= tolerance_range:
                print(f"        ✅ Converged with {test_vertices} vertices (error: {error})")
                return test_mesh
            
            # Adjust search bounds
            if test_vertices > target_vertices:
                # Too many vertices, need fewer faces
                max_faces = target_faces - 1
            else:
                # Too few vertices, need more faces
                min_faces = target_faces + 1
            
            # Check if search space is exhausted
            if min_faces >= max_faces:
                break
                
        except Exception as e:
            print(f"        Iteration {iteration + 1} failed: {e}")
            # Adjust bounds to avoid this face count
            if target_faces == min_faces:
                min_faces += 1
            else:
                max_faces = target_faces - 1
            
            if min_faces >= max_faces:
                break
    
    print(f"        Final result: {best_vertex_count} vertices (error: {best_error})")
    return best_mesh


def resample_mesh(input_path, output_path, target_vertices=TARGET_VERTEX_COUNT):
    """
    Resample mesh to target vertex count if it's outside the tolerance range.
    - Too small shapes (< 75% of target): subdivide/refine
    - Too large shapes (> 125% of target): simplify/decimate
    - Shapes within tolerance: copy as-is
    """
    mesh = o3d.io.read_triangle_mesh(str(input_path))
    if mesh.is_empty():
        print(f"[X] Empty mesh: {input_path}")
        return False, 0, 0
    
    original_vertices = len(mesh.vertices)
    original_faces = len(mesh.triangles)
    
    # Calculate tolerance bounds
    min_vertices = int(target_vertices * (1 - TOLERANCE))  # 3750
    max_vertices = int(target_vertices * (1 + TOLERANCE))  # 6250
    
    # Check if resampling is needed
    if min_vertices <= original_vertices <= max_vertices:
        # Within tolerance, copy as-is
        o3d.io.write_triangle_mesh(str(output_path), mesh)
        print(f"[OK] No resampling needed: {original_vertices} vertices (within tolerance)")
        return True, original_vertices, original_vertices
    
    try:
        if original_vertices < min_vertices:            # Too small - need to subdivide/refine
            mesh_resampled = subdivide_mesh(mesh, target_vertices)
            action = "subdivided"
        else:
            # Too large - need to simplify/decimate
            # Use iterative decimation to precisely target vertex count
            mesh_resampled = iterative_decimation(mesh, target_vertices)
            action = "simplified"
        
        # Ensure we have a valid mesh
        if mesh_resampled.is_empty() or len(mesh_resampled.vertices) == 0:
            print(f"[X] Resampling resulted in empty mesh: {input_path}")
            return False, original_vertices, 0
        
        # Write the resampled mesh
        o3d.io.write_triangle_mesh(str(output_path), mesh_resampled)
        final_vertices = len(mesh_resampled.vertices)
        print(f"[OK] {action.capitalize()}: {original_vertices} -> {final_vertices} vertices")
        return True, original_vertices, final_vertices
        
    except Exception as e:
        print(f"[X] Error resampling {input_path}: {e}")
        return False, original_vertices, 0


def subdivide_mesh(mesh, target_vertices):
    """
    Intelligently subdivide mesh to approach target vertex count.
    Uses multiple strategies to avoid overshooting.
    """
    current_vertices = len(mesh.vertices)
    
    # Strategy 1: For very small meshes (< 500 vertices), use conservative subdivision
    if current_vertices < 500:
        return subdivide_small_mesh(mesh, target_vertices)
    
    # Strategy 2: For medium meshes (500-2000), use controlled loop subdivision
    elif current_vertices < 2000:
        return subdivide_medium_mesh(mesh, target_vertices)
    
    # Strategy 3: For larger meshes that still need subdivision, use minimal subdivision
    else:
        return subdivide_large_mesh(mesh, target_vertices)


def subdivide_small_mesh(mesh, target_vertices):
    """Handle very small meshes with feature-preserving subdivision."""
    current_vertices = len(mesh.vertices)
    subdivided_mesh = mesh
    
    # For very small meshes, we need to be extremely careful to preserve shape
    # Use edge splitting and midpoint subdivision to avoid smoothing
    max_iterations = 3
    
    for i in range(max_iterations):
        current_count = len(subdivided_mesh.vertices)
        
        # Calculate how much more density we need
        multiplier_needed = target_vertices / current_count
        
        # If we're close enough, stop
        if current_count >= target_vertices * 0.75:
            break
            
        # Choose subdivision method based on how much we need to grow
        if multiplier_needed > 3.5:
            # Need significant growth - use midpoint subdivision (preserves features)
            try:
                subdivided_mesh = subdivided_mesh.subdivide_midpoint(number_of_iterations=1)
                print(f"      Applied midpoint subdivision: {current_count} -> {len(subdivided_mesh.vertices)} vertices")
            except Exception as e:
                # If midpoint fails, try controlled edge split
                print(f"      Midpoint failed, trying edge split subdivision")
                subdivided_mesh = edge_split_subdivision(subdivided_mesh, target_vertices)
                break
        else:
            # Moderate growth needed - try one careful iteration
            try:
                # First try midpoint (most conservative)
                test_mesh = subdivided_mesh.subdivide_midpoint(number_of_iterations=1)
                if len(test_mesh.vertices) <= target_vertices * 1.3:
                    subdivided_mesh = test_mesh
                    print(f"      Applied controlled midpoint: {current_count} -> {len(subdivided_mesh.vertices)} vertices")
                else:
                    # Midpoint would overshoot, try edge split
                    subdivided_mesh = edge_split_subdivision(subdivided_mesh, target_vertices)
                    break
            except Exception as e:
                print(f"⚠️ Small mesh subdivision failed: {e}")
                break
    
    # Final adjustment if we overshot
    final_count = len(subdivided_mesh.vertices)
    if final_count > target_vertices * 1.3:
        try:
            subdivided_mesh = iterative_decimation(subdivided_mesh, target_vertices)
            print(f"      Final decimation: {final_count} -> {len(subdivided_mesh.vertices)} vertices")
        except Exception as e:
            print(f"⚠️ Small mesh final decimation failed: {e}")
    
    return subdivided_mesh


def subdivide_medium_mesh(mesh, target_vertices):
    """Handle medium-sized meshes with feature-preserving subdivision."""
    current_vertices = len(mesh.vertices)
    subdivided_mesh = mesh
    
    # For medium meshes, prioritize feature preservation
    multiplier_needed = target_vertices / current_vertices
    
    if multiplier_needed <= 3:
        # Moderate growth - use midpoint subdivision (most conservative)
        try:
            subdivided_mesh = subdivided_mesh.subdivide_midpoint(number_of_iterations=1)
            print(f"      Applied midpoint subdivision: {current_vertices} -> {len(subdivided_mesh.vertices)} vertices")
            
            # Fine-tune if needed
            final_count = len(subdivided_mesh.vertices)
            if final_count > target_vertices * 1.2:
                subdivided_mesh = iterative_decimation(subdivided_mesh, target_vertices)
                print(f"      Post-subdivision decimation: {final_count} -> {len(subdivided_mesh.vertices)} vertices")
                
        except Exception as e:
            print(f"⚠️ Medium mesh midpoint subdivision failed, trying alternative: {e}")
            # Fallback to controlled remeshing
            subdivided_mesh = controlled_remeshing(mesh, target_vertices)
    else:
        # Need more significant growth - use multiple conservative steps
        try:
            # First midpoint subdivision
            subdivided_mesh = subdivided_mesh.subdivide_midpoint(number_of_iterations=1)
            intermediate_count = len(subdivided_mesh.vertices)
            print(f"      First midpoint: {current_vertices} -> {intermediate_count} vertices")
            
            # Check if we need another iteration
            if intermediate_count < target_vertices * 0.7:
                # Apply second midpoint subdivision carefully
                test_mesh = subdivided_mesh.subdivide_midpoint(number_of_iterations=1)
                if len(test_mesh.vertices) <= target_vertices * 1.4:
                    subdivided_mesh = test_mesh
                    print(f"      Second midpoint: {intermediate_count} -> {len(subdivided_mesh.vertices)} vertices")
                else:
                    # Would overshoot, try edge split instead
                    subdivided_mesh = edge_split_subdivision(subdivided_mesh, target_vertices)
            
            # Final adjustment if needed
            final_count = len(subdivided_mesh.vertices)
            if final_count > target_vertices * 1.2:
                subdivided_mesh = iterative_decimation(subdivided_mesh, target_vertices)
                print(f"      Final decimation: {final_count} -> {len(subdivided_mesh.vertices)} vertices")
                
        except Exception as e:
            print(f"⚠️ Medium mesh complex subdivision failed: {e}")
            # Fallback to simple approach
            try:
                subdivided_mesh = mesh.subdivide_midpoint(number_of_iterations=1)
            except:
                subdivided_mesh = mesh  # Last resort - return original
    
    return subdivided_mesh


def subdivide_large_mesh(mesh, target_vertices):
    """Handle larger meshes that still need some subdivision with feature preservation."""
    # For meshes that are already reasonably large but still below target,
    # use only the most conservative subdivision methods
    try:
        # Always try midpoint subdivision first (most feature-preserving)
        subdivided_mesh = mesh.subdivide_midpoint(number_of_iterations=1)
        print(f"      Applied midpoint subdivision: {len(mesh.vertices)} -> {len(subdivided_mesh.vertices)} vertices")
        
        # If still not enough and we can safely do another iteration
        if len(subdivided_mesh.vertices) < target_vertices * 0.8:
            test_mesh = subdivided_mesh.subdivide_midpoint(number_of_iterations=1)
            if len(test_mesh.vertices) <= target_vertices * 1.3:
                subdivided_mesh = test_mesh
                print(f"      Second midpoint subdivision: -> {len(subdivided_mesh.vertices)} vertices")
        
        # Adjust if overshot
        final_count = len(subdivided_mesh.vertices)
        if final_count > target_vertices * 1.2:
            subdivided_mesh = iterative_decimation(subdivided_mesh, target_vertices)
            print(f"      Post-subdivision decimation: {final_count} -> {len(subdivided_mesh.vertices)} vertices")
            
    except Exception as e:
        print(f"⚠️ Large mesh subdivision failed: {e}")
        subdivided_mesh = mesh  # Return original if all fails
    
    return subdivided_mesh

def main():
    print(f"Resampling meshes from {SOURCE_ROOT} to {TARGET_ROOT}")
    print(f"Target: {TARGET_VERTEX_COUNT} vertices (±{TOLERANCE*100}% tolerance)")
    print(f"Range: {int(TARGET_VERTEX_COUNT * (1-TOLERANCE))} - {int(TARGET_VERTEX_COUNT * (1+TOLERANCE))} vertices")
    print("-" * 60)
    
    # Statistics tracking
    total_processed = 0
    subdivided_count = 0
    simplified_count = 0
    unchanged_count = 0
    failed_count = 0
    
    original_vertices_total = 0
    final_vertices_total = 0
    
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
            
            success, original_verts, final_verts = resample_mesh(obj_file, out_file)
            total_processed += 1
            original_vertices_total += original_verts
            
            if success:
                final_vertices_total += final_verts
                min_vertices = int(TARGET_VERTEX_COUNT * (1 - TOLERANCE))
                max_vertices = int(TARGET_VERTEX_COUNT * (1 + TOLERANCE))
                
                if original_verts < min_vertices:
                    subdivided_count += 1
                elif original_verts > max_vertices:
                    simplified_count += 1
                else:
                    unchanged_count += 1
            else:
                failed_count += 1
    
    # Print final statistics
    print("\n" + "="*60)
    print("RESAMPLING SUMMARY")
    print("="*60)
    print(f"Total files processed: {total_processed}")
    print(f"[OK] Successful: {total_processed - failed_count}")
    print(f"[X] Failed: {failed_count}")
    print()
    print(f"Actions taken:")
    print(f"  Subdivided (too small): {subdivided_count}")
    print(f"  Simplified (too large): {simplified_count}")
    print(f"  Unchanged (within tolerance): {unchanged_count}")
    print()
    if total_processed - failed_count > 0:
        avg_original = original_vertices_total / total_processed
        avg_final = final_vertices_total / (total_processed - failed_count)
        print(f"Average vertex count:")
        print(f"  Before: {avg_original:.0f} vertices")
        print(f"  After:  {avg_final:.0f} vertices")
        print(f"  Target: {TARGET_VERTEX_COUNT} vertices")


if __name__ == '__main__':
    main()
