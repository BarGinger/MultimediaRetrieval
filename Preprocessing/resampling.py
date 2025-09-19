import os
import open3d as o3d
import numpy as np
from pathlib import Path

# --- Toggle for sampled dataset ---
USE_SAMPLED_DATASET = True  # Set to True to use Data_sampled, False for full Data
BASE = Path(__file__).parent.parent.resolve()
SOURCE_ROOT = BASE / ('Data_sampled' if USE_SAMPLED_DATASET else 'Data')
TARGET_ROOT = BASE / ('Data_sampled_resampled' if USE_SAMPLED_DATASET else 'Data_resampled')
TARGET_VERTEX_COUNT = 5000
TOLERANCE = 0.25  # 25% tolerance for resampling


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
            # NOTE: simplify_quadric_decimation targets NUMBER OF FACES, not vertices!
            # We need to estimate target faces from target vertices
            # Typical ratio: faces ≈ 2 × vertices for well-formed meshes
            target_faces = int(target_vertices * 1.8)  # Conservative estimate
            mesh_resampled = mesh.simplify_quadric_decimation(target_faces)
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
    """Handle very small meshes with careful subdivision."""
    current_vertices = len(mesh.vertices)
    subdivided_mesh = mesh
    
    # For very small meshes, we might need multiple subdivisions
    # But we need to be very careful not to overshoot
    max_iterations = 3
    
    for i in range(max_iterations):
        current_count = len(subdivided_mesh.vertices)
        
        # Estimate what the next subdivision will produce
        # Loop subdivision typically increases vertices by factor of 3.5-4.5
        estimated_next = current_count * 4
        
        # If the next subdivision would overshoot significantly, try different approach
        if estimated_next > target_vertices * 1.3:
            # Use edge split instead of full loop subdivision for more control
            try:
                subdivided_mesh = subdivided_mesh.subdivide_midpoint(number_of_iterations=1)
            except:
                # If midpoint fails, try one loop subdivision and then decimate
                try:
                    subdivided_mesh = subdivided_mesh.subdivide_loop(number_of_iterations=1)                    # Immediately decimate if we overshot
                    if len(subdivided_mesh.vertices) > target_vertices * 1.2:
                        target_faces = int(target_vertices * 1.8)
                        subdivided_mesh = subdivided_mesh.simplify_quadric_decimation(target_faces)
                except Exception as e:
                    print(f"⚠️ Small mesh subdivision failed: {e}")
                    break
            break
        else:
            # Safe to do loop subdivision
            try:
                subdivided_mesh = subdivided_mesh.subdivide_loop(number_of_iterations=1)
                current_count = len(subdivided_mesh.vertices)
                
                # Stop if we're close enough to target
                if current_count >= target_vertices * 0.8:
                    break
                    
            except Exception as e:
                print(f"⚠️ Small mesh loop subdivision failed: {e}")
                break
      # Final adjustment if needed
    final_count = len(subdivided_mesh.vertices)
    if final_count > target_vertices * 1.3:
        try:
            target_faces = int(target_vertices * 1.8)
            subdivided_mesh = subdivided_mesh.simplify_quadric_decimation(target_faces)
        except Exception as e:
            print(f"⚠️ Small mesh final decimation failed: {e}")
    
    return subdivided_mesh


def subdivide_medium_mesh(mesh, target_vertices):
    """Handle medium-sized meshes with controlled subdivision."""
    current_vertices = len(mesh.vertices)
    subdivided_mesh = mesh
    
    # For medium meshes, usually 1-2 subdivisions should be enough
    multiplier_needed = target_vertices / current_vertices
    
    if multiplier_needed <= 4:
        # One subdivision should be sufficient
        try:
            subdivided_mesh = subdivided_mesh.subdivide_loop(number_of_iterations=1)
              # Fine-tune if needed
            final_count = len(subdivided_mesh.vertices)
            if final_count > target_vertices * 1.2:
                target_faces = int(target_vertices * 1.8)
                subdivided_mesh = subdivided_mesh.simplify_quadric_decimation(target_faces)
                
        except Exception as e:
            print(f"⚠️ Medium mesh subdivision failed: {e}")
    else:
        # Need more than one subdivision, but be careful
        try:
            # First subdivision
            subdivided_mesh = subdivided_mesh.subdivide_loop(number_of_iterations=1)
            intermediate_count = len(subdivided_mesh.vertices)
            
            # Check if we need another
            if intermediate_count < target_vertices * 0.7:
                # Try midpoint subdivision for more controlled growth
                subdivided_mesh = subdivided_mesh.subdivide_midpoint(number_of_iterations=1)
              # Final adjustment
            final_count = len(subdivided_mesh.vertices)
            if final_count > target_vertices * 1.2:
                target_faces = int(target_vertices * 1.8)
                subdivided_mesh = subdivided_mesh.simplify_quadric_decimation(target_faces)
                
        except Exception as e:
            print(f"⚠️ Medium mesh complex subdivision failed: {e}")
    
    return subdivided_mesh


def subdivide_large_mesh(mesh, target_vertices):
    """Handle larger meshes that still need some subdivision."""
    # For meshes that are already reasonably large but still below target,
    # use minimal subdivision
    try:
        # Try midpoint subdivision first (more conservative)
        subdivided_mesh = mesh.subdivide_midpoint(number_of_iterations=1)
        
        # If still not enough, try one loop subdivision
        if len(subdivided_mesh.vertices) < target_vertices * 0.8:
            subdivided_mesh = mesh.subdivide_loop(number_of_iterations=1)
              # Adjust if overshot
        final_count = len(subdivided_mesh.vertices)
        if final_count > target_vertices * 1.2:
            target_faces = int(target_vertices * 1.8)
            subdivided_mesh = subdivided_mesh.simplify_quadric_decimation(target_faces)
            
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
