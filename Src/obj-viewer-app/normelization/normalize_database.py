"""
Unified Preprocessing & Normalization Script
Combines remeshing (Step 2) with your existing complete 4-step normalization (Step 3.1)
Order: Remeshing → Translation → PCA → Flipping → Scaling
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import time
import open3d as o3d
import csv
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist
from scipy import stats
import matplotlib.pyplot as plt
import csv as csv_module
import traceback
from scipy.spatial import ConvexHull


# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.file_index import get_file_tree
# from core.analysis_cache import merge_analysis_data
from core.shapeMesh import ShapeMesh


# Numerical tolerances - following normalization.py improvements
AREA_EPS = 1e-12          # Minimum total surface area before falling back to mean
RECENTER_EPS = 1e-9       # Threshold to apply second recentering pass (pre-scaling)


# Import the enhanced functions from ShapeMesh
from core.shapeMesh import calculate_mass_barycenter


class UnifiedPreprocessingProcessor:
    def __init__(self, target_vertices=7500, output_base_dir=None):
        """
        Unified processor combining remeshing + your existing complete normalization
        
        Parameters:
            target_vertices (int): Target number of vertices for remeshing
            output_base_dir (str or Path): Output directory for processed shapes
        """
        self.target_vertices = target_vertices
        self.min_acceptable_vertices = 5000
        self.max_acceptable_vertices = 10000
        self.gentle_decimation_ratio = 0.70
        self.fine_decimation_ratio = 0.85
        self.max_decimation_passes = 8
        
        # Use same path resolution as your normalize_database.py
        if output_base_dir is None:
            cwd = Path.cwd()
            dataset_path = "Datasets/UnifiedPreprocessed"
            candidates = [cwd / dataset_path, cwd.parent / dataset_path, cwd.parent.parent / dataset_path]
            self.output_base_dir = next((p for p in candidates if p.parent.exists()), candidates[-1])
        else:
            self.output_base_dir = Path(output_base_dir)
        
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'start_time': None,
            'errors': [],
            'remeshing_stats': [],
            'normalization_stats': {
                'centering_errors': [],
                'scaling_errors': [],
                'by_category': {}
            },
            'recentering_triggered': [],  # Track two-pass recentering
            'eigenvalue_ratios': {
                'lambda1_over_lambda2': [],
                'lambda2_over_lambda3': [],
                'condition_numbers': []
            },
            'transformation_magnitudes': {
                'translations': [],
                'rotations': [],
                'scalings': []
            },
            'aspect_ratio_errors': [],
            'moment_values': {
                'x_axis': [],
                'y_axis': [],
                'z_axis': []
            }
        }
    
    def setup_output_directories(self, datasets):
        """Create output directories for each dataset"""
        print(f"📁 Setting up output directories in: {self.output_base_dir.absolute()}")
        for dataset in datasets:
            dataset_dir = self.output_base_dir / dataset
            dataset_dir.mkdir(parents=True, exist_ok=True)
            print(f"   Created: {dataset_dir}")

    
    def apply_remeshing_if_needed(self, mesh_path, target_vertices, tolerance=0.2):
        """
        Apply remeshing if vertex count significantly differs from target
        
        Parameters:
            mesh_path (Path): Path to the OBJ file
            target_vertices (int): Target number of vertices
            tolerance (float): Tolerance for vertex count difference (0.2 = 20%)
            
        Returns:
            tuple: (vertices, faces, remeshed_flag)
        """
        try:
            # Load with Open3D for remeshing
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))

            if len(mesh.vertices) == 0:
                print(f"❌ Empty mesh: {mesh_path}")
                return None, None, False

            # Aggressive cleaning to ensure mesh quality
            try:
                mesh.remove_degenerate_triangles()
                mesh.remove_duplicated_vertices()
                mesh.remove_duplicated_triangles()
                mesh.remove_non_manifold_edges()
                mesh.remove_unreferenced_vertices()
            except Exception as e:
                print(f"⚠️ Cleaning error: {e}")

            current_vertices = len(mesh.vertices)

            # Always try to reach target vertex count (±10%)
            lower_bound = int(target_vertices * 0.9)
            upper_bound = int(target_vertices * 1.1)
            was_remeshed = False

            if lower_bound <= current_vertices <= upper_bound:
                print(f"  ✅ No remeshing needed ({current_vertices} vertices within ±10% of target)")
                return np.asarray(mesh.vertices), np.asarray(mesh.triangles), False

            # Decimate if too many vertices
            if current_vertices > upper_bound:
                print(f"  🔄 Aggressive decimation from {current_vertices} to target {target_vertices}")
                mesh = self._decimate_to_range(mesh)
                was_remeshed = True
            # Upsample if too few vertices
            elif current_vertices < lower_bound:
                print(f"  🔄 Aggressive upsampling from {current_vertices} to target {target_vertices}")
                # Temporarily allow more subdivision passes for compliance
                orig_max_subdiv = getattr(self, 'max_decimation_passes', 8)
                self.max_decimation_passes = max(8, orig_max_subdiv)
                mesh = self._upsample_to_range(mesh)
                # If still below, force one more subdivision if possible
                if len(mesh.vertices) < lower_bound and hasattr(mesh, 'subdivide_midpoint'):
                    try:
                        mesh = mesh.subdivide_midpoint(number_of_iterations=1)
                        print(f"    🔄 Final forced subdivision, now {len(mesh.vertices)} vertices")
                    except Exception as e:
                        print(f"    ⚠️ Final subdivision failed: {e}")
                was_remeshed = True

            # Final compacting and trimming
            try:
                mesh.remove_unreferenced_vertices()
            except Exception:
                pass

            # If still above upper bound, decimate again
            if len(mesh.vertices) > upper_bound:
                print(f"    🔧 Final trimming: {len(mesh.vertices)} → target {target_vertices}")
                mesh = self._decimate_to_range(mesh)

            final_vertices = len(mesh.vertices)
            print(f"  ✅ Remeshing result: {final_vertices} effective vertices, {len(mesh.triangles)} faces")

            # Collect remeshing stats with enhanced upsampling log if available
            remeshing_entry = {
                'original_vertices': current_vertices,
                'target_vertices': target_vertices,
                'final_vertices': final_vertices,
                'reduction_ratio': final_vertices / current_vertices if current_vertices > 0 else 1.0
            }

            # Add upsampling log if available (from _upsample_to_range)
            if hasattr(self, '_current_upsampling_log'):
                remeshing_entry['upsampling_log'] = self._current_upsampling_log
                delattr(self, '_current_upsampling_log')

            self.stats['remeshing_stats'].append(remeshing_entry)

            return np.asarray(mesh.vertices), np.asarray(mesh.triangles), was_remeshed

        except Exception as e:
            print(f"❌ Remeshing failed for {mesh_path}: {e}")
            return None, None, False

    def _compact_mesh(self, mesh):
        """Remove vertices not referenced by any triangle to keep effective counts consistent."""
        try:
            mesh.remove_unreferenced_vertices()
        except Exception:
            pass
        return mesh

    def _decimate_to_range(self, mesh):
        """Decimation logic adapted from resampling_simple.py"""
        # Start from effective counts
        mesh = self._compact_mesh(mesh)
        passes = 0
        while passes < self.max_decimation_passes and len(mesh.vertices) > self.max_acceptable_vertices:
            passes += 1
            # Compact each pass to get effective counts
            mesh = self._compact_mesh(mesh)
            current_v = len(mesh.vertices)
            faces = len(mesh.triangles)
            ratio = self.target_vertices / max(current_v, 1)

            if ratio < 0.55:
                retain_ratio = self.gentle_decimation_ratio
            elif ratio < 0.85:
                retain_ratio = 0.78
            else:
                retain_ratio = self.fine_decimation_ratio

            target_faces = max(200, int(faces * retain_ratio))

            try:
                new_mesh = mesh.simplify_quadric_decimation(target_faces)
                new_mesh = self._compact_mesh(new_mesh)
                new_v = len(new_mesh.vertices)
                if new_v >= current_v - 20:
                    # Stagnation -> force more aggressive step once
                    target_faces = max(100, int(faces * 0.5))
                    new_mesh = mesh.simplify_quadric_decimation(target_faces)
                    mesh = self._compact_mesh(mesh)
                    new_v = len(new_mesh.vertices)
                print(f"    Pass {passes}: {current_v} -> {new_v} effective verts (faces {faces}->{len(new_mesh.triangles)})")
                mesh = new_mesh
            except Exception as e:
                print(f"    Decimation failed pass {passes}: {e}")
                break
        return mesh
    
    def _upsample_to_range(self, mesh):
        """
        Enhanced upsampling with Loop subdivision, quality checks, and connectivity validation
        
        Improvements:
        1. Uses Loop subdivision (smoother, better quality)
        2. Checks connectivity after each subdivision pass
        3. Validates no floating/detached geometry created
        4. Conservative limits on upsampling factor
        5. Falls back gracefully on failures
        6. Logs all validation results for export
        """
        max_subdiv_passes = 4
        target_min_fill = self.min_acceptable_vertices
        target_soft_cap = 9000
        allow_overshoot_factor = 1.25
        max_upsampling_factor = 10  # Don't allow more than 10x vertex increase
        
        original_vertex_count = len(mesh.vertices)
        max_allowed_vertices = original_vertex_count * max_upsampling_factor
        
        # Initialize upsampling log for validation export
        upsampling_log = {
            'original_vertices': original_vertex_count,
            'target_range': [self.min_acceptable_vertices, self.max_acceptable_vertices],
            'max_upsampling_factor': max_upsampling_factor,
            'subdivision_passes': [],
            'quality_checks': [],
            'warnings': [],
            'final_result': {}
        }
        
        # For very low-poly meshes (< 500 vertices), be more aggressive
        if original_vertex_count < 500:
            msg = f"Very low-poly mesh ({original_vertex_count} vertices), using aggressive upsampling"
            print(f"    ⚡ {msg}")
            upsampling_log['warnings'].append(msg)
            max_subdiv_passes = 6  # Allow up to 6 passes for small meshes
            max_upsampling_factor = 20  # Allow up to 20x increase
            max_allowed_vertices = original_vertex_count * max_upsampling_factor
            upsampling_log['max_upsampling_factor'] = max_upsampling_factor
            upsampling_log['max_subdiv_passes_adjusted'] = max_subdiv_passes
        
        passes = 0
        subdivision_method = 'unknown'
        
        while passes < max_subdiv_passes and len(mesh.vertices) < target_min_fill:
            current_v = len(mesh.vertices)
            
            # Check 1: Prevent excessive upsampling
            if current_v >= max_allowed_vertices:
                msg = f"Reached max upsampling factor ({max_upsampling_factor}x), stopping at {current_v} vertices"
                print(f"    🛑 {msg}")
                upsampling_log['warnings'].append(msg)
                break
            
            # Check 2: Predict overshoot before subdivision
            predicted_vertices = current_v * 2  # Rough estimate
            if predicted_vertices > self.target_vertices * allow_overshoot_factor:
                msg = f"Predicted overshoot ({predicted_vertices} > {self.target_vertices * allow_overshoot_factor:.0f}), stopping"
                print(f"    🛑 {msg}")
                upsampling_log['warnings'].append(msg)
                break
            
            # Check 3: Validate mesh connectivity BEFORE subdivision (skip on first pass)
            # On first pass, the mesh might have minor topology issues after cleaning,
            # but subdivision might actually improve it
            # SKIP THIS CHECK - it's too restrictive and prevents valid subdivision
            # We'll check connectivity AFTER subdivision instead to catch actual problems
            # if passes > 0:
            #     connectivity_before = self._validate_mesh_connectivity(mesh)
            #     if not connectivity_before['is_valid']:
            #         msg = f"Mesh has connectivity issues before subdivision pass {passes + 1}"
            #         print(f"    ❌ {msg}")
            #         upsampling_log['warnings'].append(msg)
            #         upsampling_log['quality_checks'].append({
            #             'pass': passes + 1,
            #             'stage': 'pre_subdivision',
            #             'check': 'connectivity',
            #             'result': connectivity_before
            #         })
            #         break
            
            try:
                # Always use Midpoint subdivision - it's more robust for meshes with complex topology
                # Loop subdivision can fragment meshes at non-manifold edges, causing parts to detach
                # Midpoint is slightly less smooth but preserves mesh connectivity much better
                subdivision_method = 'midpoint'
                print(f"    🔄 Applying Midpoint subdivision (pass {passes + 1})...")
                new_mesh = mesh.subdivide_midpoint(number_of_iterations=1)

                
                new_vertex_count = len(new_mesh.vertices)
                increase_factor = new_vertex_count / current_v if current_v > 0 else 1.0
                
                # Check 4: Validate reasonable vertex increase
                # For low-poly meshes (<1000 vertices), allow higher increase factors
                # since subdivision naturally creates more vertices
                # For higher-poly meshes, be more conservative
                max_increase = 5.0 if current_v < 1000 else 3.0
                
                if increase_factor > max_increase:
                    msg = f"Suspicious vertex increase: {current_v} → {new_vertex_count} ({increase_factor:.1f}x > {max_increase}x), reverting"
                    print(f"    ⚠️  {msg}")
                    upsampling_log['warnings'].append(msg)
                    upsampling_log['subdivision_passes'].append({
                        'pass': passes + 1,
                        'method': subdivision_method,
                        'before_vertices': current_v,
                        'after_vertices': new_vertex_count,
                        'increase_factor': increase_factor,
                        'status': 'rejected_excessive_increase'
                    })
                    break
                
                # Check 5: Validate mesh connectivity AFTER subdivision
                connectivity_after = self._validate_mesh_connectivity(new_mesh)
                
                # Allow multiple components - complex shapes like airplanes can have 6+ legitimate parts
                # Only reject if subdivision INCREASES the component count (indicates artifacts)
                # Get original component count
                if passes == 0:
                    # First pass - check original mesh component count
                    connectivity_original = self._validate_mesh_connectivity(mesh)
                    original_components = connectivity_original['connected_components']
                else:
                    # Use the component count from previous iteration
                    original_components = getattr(self, '_last_component_count', 1)
                
                # Store current component count for next iteration
                self._last_component_count = connectivity_after['connected_components']
                
                # Reject only if subdivision INCREASED components (breaking the mesh apart)
                components_increased = connectivity_after['connected_components'] > original_components * 1.5
                
                if not connectivity_after['is_valid'] and components_increased:
                    # If Loop subdivision increased components, try midpoint as fallback
                    if subdivision_method == 'loop' and hasattr(mesh, 'subdivide_midpoint'):
                        print(f"    ⚠️  Loop subdivision increased components {original_components}→{connectivity_after['connected_components']}, trying midpoint fallback...")
                        try:
                            new_mesh_fallback = mesh.subdivide_midpoint(number_of_iterations=1)
                            connectivity_fallback = self._validate_mesh_connectivity(new_mesh_fallback)
                            
                            fallback_increased = connectivity_fallback['connected_components'] > original_components * 1.5
                            if not fallback_increased:
                                # Midpoint worked better, use it
                                print(f"    ✅ Midpoint subdivision successful ({connectivity_fallback['connected_components']} components)")
                                new_mesh = new_mesh_fallback
                                new_vertex_count = len(new_mesh.vertices)
                                increase_factor = new_vertex_count / current_v if current_v > 0 else 1.0
                                subdivision_method = 'midpoint_fallback'
                                connectivity_after = connectivity_fallback
                                self._last_component_count = connectivity_fallback['connected_components']
                            else:
                                # Midpoint also failed, reject
                                msg = f"Both Loop and midpoint subdivision increased components ({original_components}→{connectivity_after['connected_components']} and {connectivity_fallback['connected_components']}), stopping"
                                print(f"    ❌ {msg}")
                                upsampling_log['warnings'].append(msg)
                                upsampling_log['quality_checks'].append({
                                    'pass': passes + 1,
                                    'stage': 'post_subdivision',
                                    'check': 'connectivity',
                                    'result': {'loop': connectivity_after, 'midpoint': connectivity_fallback}
                                })
                                upsampling_log['subdivision_passes'].append({
                                    'pass': passes + 1,
                                    'method': 'loop_and_midpoint_both_failed',
                                    'before_vertices': current_v,
                                    'after_vertices': new_vertex_count,
                                    'increase_factor': increase_factor,
                                    'status': 'rejected_connectivity'
                                })
                                break
                        except Exception as e:
                            msg = f"Midpoint fallback failed: {str(e)}"
                            print(f"    ❌ {msg}")
                            upsampling_log['warnings'].append(msg)
                            upsampling_log['subdivision_passes'].append({
                                'pass': passes + 1,
                                'method': subdivision_method,
                                'before_vertices': current_v,
                                'after_vertices': new_vertex_count,
                                'increase_factor': increase_factor,
                                'status': 'rejected_connectivity'
                            })
                            break
                    else:
                        # No fallback available or already using midpoint
                        msg = f"Subdivision increased components {original_components}→{connectivity_after['connected_components']}, stopping"
                        print(f"    ❌ {msg}")
                        upsampling_log['warnings'].append(msg)
                        upsampling_log['quality_checks'].append({
                            'pass': passes + 1,
                            'stage': 'post_subdivision',
                            'check': 'connectivity',
                            'result': connectivity_after
                        })
                        upsampling_log['subdivision_passes'].append({
                            'pass': passes + 1,
                            'method': subdivision_method,
                            'before_vertices': current_v,
                            'after_vertices': new_vertex_count,
                            'increase_factor': increase_factor,
                            'status': 'rejected_connectivity'
                        })
                        break
                
                # Check 6: Validate no floating geometry created
                floating_check = self._validate_no_floating_geometry(new_mesh)
                if not floating_check['is_valid']:
                    msg = f"Subdivision created floating geometry, reverting to {current_v} vertices"
                    print(f"    ❌ {msg}")
                    upsampling_log['warnings'].append(msg)
                    upsampling_log['quality_checks'].append({
                        'pass': passes + 1,
                        'stage': 'post_subdivision',
                        'check': 'floating_geometry',
                        'result': floating_check
                    })
                    upsampling_log['subdivision_passes'].append({
                        'pass': passes + 1,
                        'method': subdivision_method,
                        'before_vertices': current_v,
                        'after_vertices': new_vertex_count,
                        'increase_factor': increase_factor,
                        'status': 'rejected_floating_geometry'
                    })
                    break
                
                # Check 7: Detect subdivision stagnation
                if new_vertex_count <= current_v + 10:
                    msg = f"Subdivision stagnated ({current_v} → {new_vertex_count}), stopping"
                    print(f"    ⚠️  {msg}")
                    upsampling_log['warnings'].append(msg)
                    upsampling_log['subdivision_passes'].append({
                        'pass': passes + 1,
                        'method': subdivision_method,
                        'before_vertices': current_v,
                        'after_vertices': new_vertex_count,
                        'increase_factor': increase_factor,
                        'status': 'rejected_stagnation'
                    })
                    break
                
                # All checks passed, accept the subdivided mesh
                mesh = new_mesh
                passes += 1
                print(f"    ✅ Pass {passes}: {current_v} → {new_vertex_count} vertices (quality checks passed)")
                
                upsampling_log['subdivision_passes'].append({
                    'pass': passes,
                    'method': subdivision_method,
                    'before_vertices': current_v,
                    'after_vertices': new_vertex_count,
                    'increase_factor': increase_factor,
                    'status': 'accepted',
                    'connectivity_check': connectivity_after,
                    'floating_geometry_check': floating_check
                })
                
                # Check 8: Stop if we've reached a reasonable target
                if new_vertex_count >= target_soft_cap:
                    msg = f"Reached soft cap ({target_soft_cap}), stopping"
                    print(f"    ✅ {msg}")
                    upsampling_log['warnings'].append(msg)
                    break
                    
            except Exception as e:
                msg = f"Subdivision failed at pass {passes + 1}: {str(e)}"
                print(f"    ❌ {msg}")
                upsampling_log['warnings'].append(msg)
                upsampling_log['subdivision_passes'].append({
                    'pass': passes + 1,
                    'method': subdivision_method,
                    'before_vertices': current_v,
                    'status': 'error',
                    'error': str(e)
                })
                break
        
        # Final validation
        final_vertex_count = len(mesh.vertices)
        upsampling_factor = final_vertex_count / original_vertex_count if original_vertex_count > 0 else 1.0
        
        print(f"    📊 Upsampling complete: {original_vertex_count} → {final_vertex_count} vertices ({upsampling_factor:.1f}x)")
        
        upsampling_log['final_result'] = {
            'final_vertices': final_vertex_count,
            'upsampling_factor': upsampling_factor,
            'passes_completed': passes,
            'subdivision_method_used': subdivision_method
        }
        
        # If still significantly below minimum, issue warning but don't force further subdivision
        if final_vertex_count < self.min_acceptable_vertices:
            deficit = self.min_acceptable_vertices - final_vertex_count
            msg = f"Final vertex count ({final_vertex_count}) below target minimum ({self.min_acceptable_vertices}), deficit: {deficit} vertices"
            print(f"    ⚠️  {msg}")
            print(f"    ⚡ Forcing final subdivision to reach minimum if possible")
            upsampling_log['warnings'].append(msg)
            upsampling_log['warnings'].append("Forcing final subdivision to reach minimum")
            upsampling_log['final_result']['below_minimum'] = True
            # Try to force subdivision
            if hasattr(mesh, 'subdivide_midpoint'):
                try:
                    mesh = mesh.subdivide_midpoint(number_of_iterations=1)
                    print(f"    ⚡ Final forced subdivision, now {len(mesh.vertices)} vertices")
                except Exception as e:
                    print(f"    ⚠️ Final subdivision failed: {e}")
        
        # If we overshot the absolute max, trim gently
        if final_vertex_count > self.max_acceptable_vertices:
            print(f"    🔧 Final trimming: {final_vertex_count} → target ~{self.target_vertices}")
            upsampling_log['warnings'].append(f"Overshot maximum, applying decimation from {final_vertex_count}")
            mesh = self._decimate_to_range(mesh)
            upsampling_log['final_result']['decimation_applied'] = True
            upsampling_log['final_result']['final_vertices_after_decimation'] = len(mesh.vertices)
        
        # Store upsampling log for validation export
        if not hasattr(self, '_current_upsampling_log'):
            self._current_upsampling_log = upsampling_log
        else:
            self._current_upsampling_log = upsampling_log
        
        return mesh
    
    def _validate_mesh_connectivity(self, mesh):
        """
        Validate mesh connectivity and topology
        
        Returns dictionary with validation results:
        - is_valid: bool (overall connectivity status)
        - isolated_vertices: int (vertices not referenced by any triangle)
        - degenerate_triangles: int (triangles with duplicate vertices)
        - connected_components: int (number of separate mesh parts)
        - details: dict with additional information
        """
        result = {
            'is_valid': True,
            'isolated_vertices': 0,
            'degenerate_triangles': 0,
            'connected_components': 0,
            'details': {}
        }
        
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        if len(vertices) == 0 or len(triangles) == 0:
            result['is_valid'] = False
            result['details']['error'] = 'Empty mesh'
            return result
        
        # Check 1: Validate all triangle indices are within bounds
        max_vertex_index = len(vertices) - 1
        invalid_indices = np.any((triangles < 0) | (triangles > max_vertex_index))
        if invalid_indices:
            result['is_valid'] = False
            result['details']['invalid_triangle_indices'] = True
            return result
        
        # Check 2: Find degenerate triangles (triangles where vertices are duplicated)
        degenerate_count = 0
        for tri in triangles:
            if tri[0] == tri[1] or tri[1] == tri[2] or tri[0] == tri[2]:
                degenerate_count += 1
        
        result['degenerate_triangles'] = degenerate_count
        if degenerate_count > len(triangles) * 0.01:  # More than 1% degenerate
            result['is_valid'] = False
            result['details']['excessive_degenerate_triangles'] = degenerate_count
        
        # Check 3: Find isolated vertices (not referenced by any triangle)
        referenced_vertices = np.unique(triangles.flatten())
        all_vertex_indices = set(range(len(vertices)))
        isolated = all_vertex_indices - set(referenced_vertices)
        result['isolated_vertices'] = len(isolated)
        
        if len(isolated) > len(vertices) * 0.05:  # More than 5% isolated
            result['is_valid'] = False
            result['details']['excessive_isolated_vertices'] = len(isolated)
        
        # Check 4: Count connected components (mesh parts)
        # Build adjacency graph from triangles
        try:
            # Create adjacency list for vertices
            adjacency = {i: set() for i in range(len(vertices))}
            for tri in triangles:
                adjacency[tri[0]].update([tri[1], tri[2]])
                adjacency[tri[1]].update([tri[0], tri[2]])
                adjacency[tri[2]].update([tri[0], tri[1]])
            
            # Find connected components using BFS
            visited = set()
            components = 0
            
            for start_vertex in range(len(vertices)):
                if start_vertex in visited:
                    continue
                
                # BFS from this vertex
                components += 1
                queue = [start_vertex]
                visited.add(start_vertex)
                
                while queue:
                    current = queue.pop(0)
                    for neighbor in adjacency[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
            
            result['connected_components'] = components
            
            # Multiple connected components are acceptable for complex shapes (airplanes, furniture, etc.)
            # Only flag as invalid if there are an excessive number of components (>50 indicates fragmentation)
            if components > 50:
                result['is_valid'] = False
                result['details']['excessive_components'] = components
            elif components > 1:
                # Note multiple components but don't mark as invalid
                result['details']['multiple_components'] = components
                
        except Exception as e:
            result['details']['connectivity_check_error'] = str(e)
        
        return result
    
    def _validate_no_floating_geometry(self, mesh):
        """
        Validate that subdivision didn't create floating/detached geometry
        
        Checks for:
        1. Vertices significantly far from the main mesh body
        2. Suspicious spatial outliers that indicate subdivision artifacts
        
        Returns dictionary with validation results:
        - is_valid: bool (True if no floating geometry detected)
        - outlier_count: int (number of suspicious vertices)
        - max_distance_ratio: float (ratio of max to median distance from centroid)
        - details: dict with additional information
        """
        result = {
            'is_valid': True,
            'outlier_count': 0,
            'max_distance_ratio': 0.0,
            'details': {}
        }
        
        vertices = np.asarray(mesh.vertices)
        
        if len(vertices) < 10:
            result['details']['too_few_vertices'] = len(vertices)
            return result
        
        # Calculate distances from centroid
        centroid = vertices.mean(axis=0)
        distances = np.linalg.norm(vertices - centroid, axis=1)
        
        # Statistical analysis
        median_distance = np.median(distances)
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        max_distance = np.max(distances)
        
        result['details']['median_distance'] = float(median_distance)
        result['details']['mean_distance'] = float(mean_distance)
        result['details']['std_distance'] = float(std_distance)
        result['details']['max_distance'] = float(max_distance)
        
        if median_distance == 0:
            result['details']['warning'] = 'All vertices at same location'
            return result
        
        # Check 1: Max distance ratio
        # If max distance is more than 10x the median, likely has floating geometry
        max_distance_ratio = max_distance / median_distance if median_distance > 0 else 0
        result['max_distance_ratio'] = float(max_distance_ratio)
        
        if max_distance_ratio > 10.0:
            result['is_valid'] = False
            result['details']['excessive_max_distance_ratio'] = max_distance_ratio
        
        # Check 2: Outlier detection using IQR method
        q75 = np.percentile(distances, 75)
        q25 = np.percentile(distances, 25)
        iqr = q75 - q25
        
        if iqr > 0:
            outlier_threshold = q75 + 3 * iqr  # 3x IQR beyond Q3
            outliers = distances > outlier_threshold
            outlier_count = np.sum(outliers)
            result['outlier_count'] = int(outlier_count)
            result['details']['outlier_threshold'] = float(outlier_threshold)
            result['details']['iqr'] = float(iqr)
            
            # If more than 1% of vertices are outliers, flag as suspicious
            if outlier_count > len(vertices) * 0.01:
                result['is_valid'] = False
                result['details']['excessive_outliers'] = outlier_count
        
        # Check 3: Distance spread
        # If std deviation is more than 2x the mean, spatial distribution is suspicious
        if mean_distance > 0 and std_distance / mean_distance > 2.0:
            result['is_valid'] = False
            result['details']['excessive_distance_spread'] = std_distance / mean_distance
        
        return result
    
    def validate_mesh_quality(self, vertices, faces):

        """Validate mesh quality and topology"""
        quality_data = {
            'vertex_count': len(vertices),
            'face_count': len(faces),
            'degenerate_faces': 0,
            'edge_lengths': {'mean': 0, 'std': 0, 'min': 0, 'max': 0},
            'face_areas': {'mean': 0, 'std': 0, 'min': 0, 'max': 0},
            'aspect_ratios': {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        }
        
        if len(faces) == 0:
            return quality_data
        
        # Check for degenerate faces and compute face properties
        face_areas = []
        edge_lengths = []
        aspect_ratios = []
        
        for face in faces:
            if len(face) >= 3:
                v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                
                # Edge lengths
                e1 = np.linalg.norm(v1 - v0)
                e2 = np.linalg.norm(v2 - v1)
                e3 = np.linalg.norm(v0 - v2)
                edge_lengths.extend([e1, e2, e3])
                
                # Face area
                area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
                face_areas.append(area)
                
                # Aspect ratio (longest edge / shortest edge)
                edges = [e1, e2, e3]
                if min(edges) > 1e-12:
                    aspect_ratio = max(edges) / min(edges)
                    aspect_ratios.append(aspect_ratio)
                else:
                    quality_data['degenerate_faces'] += 1
        
        # Compute statistics
        if edge_lengths:
            quality_data['edge_lengths'] = {
                'mean': float(np.mean(edge_lengths)),
                'std': float(np.std(edge_lengths)),
                'min': float(np.min(edge_lengths)),
                'max': float(np.max(edge_lengths))
            }
        
        if face_areas:
            quality_data['face_areas'] = {
                'mean': float(np.mean(face_areas)),
                'std': float(np.std(face_areas)),
                'min': float(np.min(face_areas)),
                'max': float(np.max(face_areas))
            }
        
        if aspect_ratios:
            quality_data['aspect_ratios'] = {
                'mean': float(np.mean(aspect_ratios)),
                'std': float(np.std(aspect_ratios)),
                'min': float(np.min(aspect_ratios)),
                'max': float(np.max(aspect_ratios))
            }
        
        return quality_data
    
    def analyze_transformations(self, step_vertices):
        """Analyze transformation magnitudes between steps"""
        transformations = {}
        
        # Translation analysis
        if 'original' in step_vertices and 'translated' in step_vertices:
            orig_center = np.mean(step_vertices['original'], axis=0)
            trans_center = np.mean(step_vertices['translated'], axis=0)
            translation_magnitude = float(np.linalg.norm(orig_center - trans_center))
            transformations['translation_magnitude'] = translation_magnitude
        
        # Rotation analysis (PCA alignment)
        if 'translated' in step_vertices and 'aligned' in step_vertices:
            try:
                # Compute rotation angle between principal axes
                pca_before = PCA(n_components=3)
                pca_after = PCA(n_components=3)
                pca_before.fit(step_vertices['translated'])
                pca_after.fit(step_vertices['aligned'])
                
                # Calculate rotation angle between major eigenvectors
                dot_product = np.abs(np.dot(pca_before.components_[0], pca_after.components_[0]))
                rotation_angle = np.arccos(np.clip(dot_product, 0, 1)) * 180 / np.pi
                transformations['rotation_angle_degrees'] = float(rotation_angle)
                
                # Condition number analysis
                eigenvals_before = pca_before.explained_variance_
                condition_number = float(eigenvals_before[0] / eigenvals_before[-1]) if eigenvals_before[-1] > 1e-12 else float('inf')
                transformations['condition_number'] = condition_number
                
            except Exception as e:
                transformations['rotation_analysis_error'] = str(e)
        
        # Scaling analysis
        if 'flipped' in step_vertices and 'scaled' in step_vertices:
            bbox_before = np.ptp(step_vertices['flipped'], axis=0)
            bbox_after = np.ptp(step_vertices['scaled'], axis=0)
            max_dim_before = np.max(bbox_before)
            max_dim_after = np.max(bbox_after)
            scale_factor = float(max_dim_after / max_dim_before) if max_dim_before > 1e-12 else 1.0
            transformations['scale_factor'] = scale_factor
        
        return transformations
    
    def category_specific_validation(self, category, validation_data):
        """Apply category-specific validation criteria"""
        category_analysis = {
            'category': category,
            'complexity_class': 'unknown',
            'expected_properties': {},
            'validation_adjustments': {}
        }
        
        # Define category complexity classes
        simple_shapes = ['Sphere', 'Cube', 'Cylinder', 'Cone']
        mechanical_objects = ['Car', 'Airplane', 'Helicopter', 'Motorcycle', 'Bicycle', 'Train', 'Truck', 'Bus']
        organic_shapes = ['Tree', 'Plant', 'Fish', 'Bird', 'Hand', 'HumanHead', 'Humanoid']
        furniture = ['Chair', 'Table', 'Bed', 'Shelf', 'Door']
        
        if any(simple in category for simple in simple_shapes):
            category_analysis['complexity_class'] = 'simple'
            category_analysis['expected_properties'] = {
                'high_symmetry': True,
                'regular_geometry': True,
                'expected_alignment_quality': 0.9
            }
        elif any(mech in category for mech in mechanical_objects):
            category_analysis['complexity_class'] = 'mechanical'
            category_analysis['expected_properties'] = {
                'moderate_symmetry': True,
                'engineered_geometry': True,
                'expected_alignment_quality': 0.7
            }
        elif any(org in category for org in organic_shapes):
            category_analysis['complexity_class'] = 'organic'
            category_analysis['expected_properties'] = {
                'low_symmetry': True,
                'irregular_geometry': True,
                'expected_alignment_quality': 0.5
            }
        elif any(furn in category for furn in furniture):
            category_analysis['complexity_class'] = 'furniture'
            category_analysis['expected_properties'] = {
                'functional_symmetry': True,
                'manufactured_geometry': True,
                'expected_alignment_quality': 0.8
            }
        else:
            category_analysis['complexity_class'] = 'other'
            category_analysis['expected_properties'] = {
                'expected_alignment_quality': 0.6
            }
        
        # Evaluate performance against expectations
        actual_alignment = validation_data.get('alignment_validation', {}).get('alignment_quality', 0)
        expected_alignment = category_analysis['expected_properties'].get('expected_alignment_quality', 0.6)
        
        category_analysis['performance_evaluation'] = {
            'alignment_meets_expectation': actual_alignment >= expected_alignment * 0.8,
            'alignment_performance_ratio': float(actual_alignment / expected_alignment) if expected_alignment > 0 else 1.0
        }
        
        return category_analysis
    
    def create_validation_plots(self, output_dir, all_validation_data):
        """Create comprehensive validation plots""" 
        
        plots_dir = os.path.join(output_dir, 'validation_plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Success Rate Overview
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        success_rates = {}
        error_counts = {}
        
        for data in all_validation_data:
            category = data.get('category', 'Unknown')
            validation = data.get('validation_data', {})
            
            # Count successes
            centering_ok = validation.get('centering_validation', {}).get('passes_tolerance', False)
            scaling_ok = validation.get('scaling_validation', {}).get('passes_tolerance', False)
            alignment_ok = validation.get('alignment_validation', {}).get('alignment_quality', 0) > 0.5
            
            if category not in success_rates:
                success_rates[category] = {'centering': 0, 'scaling': 0, 'alignment': 0, 'total': 0}
                error_counts[category] = 0
            
            success_rates[category]['total'] += 1
            if centering_ok:
                success_rates[category]['centering'] += 1
            if scaling_ok:
                success_rates[category]['scaling'] += 1
            if alignment_ok:
                success_rates[category]['alignment'] += 1
            
            # Count errors
            if data.get('status') == 'error':
                error_counts[category] += 1
        
        # Plot success rates
        categories = list(success_rates.keys())
        centering_rates = [success_rates[cat]['centering'] / success_rates[cat]['total'] * 100 for cat in categories]
        scaling_rates = [success_rates[cat]['scaling'] / success_rates[cat]['total'] * 100 for cat in categories]
        alignment_rates = [success_rates[cat]['alignment'] / success_rates[cat]['total'] * 100 for cat in categories]
        
        x = np.arange(len(categories))
        width = 0.25
        
        ax1.bar(x - width, centering_rates, width, label='Centering', alpha=0.8)
        ax1.bar(x, scaling_rates, width, label='Scaling', alpha=0.8)
        ax1.bar(x + width, alignment_rates, width, label='Alignment', alpha=0.8)
        
        ax1.set_xlabel('Category')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Normalization Success Rates by Category')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Error Distribution
        error_categories = list(error_counts.keys())
        error_values = list(error_counts.values())
        error_values_array = np.array(error_values)
        if error_values and np.sum(error_values_array) > 0 and not np.isnan(error_values_array).any():
            ax2.pie(error_values, labels=error_categories, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Error Distribution by Category')
        else:
            ax2.text(0.5, 0.5, 'No Errors Found', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Error Distribution (No Errors)')
        
        # 3. Alignment Quality Distribution
        alignment_qualities = []
        alignment_categories = []
        
        for data in all_validation_data:
            if data.get('status') == 'success':
                quality = data.get('validation_data', {}).get('alignment_validation', {}).get('alignment_quality', 0)
                category = data.get('category', 'Unknown')
                alignment_qualities.append(quality)
                alignment_categories.append(category)
        
        if alignment_qualities:
            ax3.hist(alignment_qualities, bins=20, alpha=0.7, edgecolor='black')
            ax3.axvline(np.mean(alignment_qualities), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(alignment_qualities):.3f}')
            ax3.set_xlabel('Alignment Quality')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Distribution of Alignment Quality Scores')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Vertex Count Distribution
        vertex_counts = []
        for data in all_validation_data:
            if data.get('status') == 'success':
                count = data.get('mesh_analysis', {}).get('final_vertex_count', 0)
                if count > 0:
                    vertex_counts.append(count)
        
        if vertex_counts:
            ax4.hist(vertex_counts, bins=20, alpha=0.7, edgecolor='black')
            ax4.axvline(np.mean(vertex_counts), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(vertex_counts):.0f}')
            ax4.axvline(7500, color='green', linestyle='--', label='Target: 7500')
            ax4.set_xlabel('Final Vertex Count')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Distribution of Final Vertex Counts')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'validation_overview.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Step-by-step Analysis Plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        step_names = ['Original', 'Resampled', 'Translated', 'Aligned', 'Flipped', 'Scaled']
        
        for i, step in enumerate(step_names):
            vertex_counts_step = []
            for data in all_validation_data:
                if data.get('status') == 'success':
                    step_data = data.get('step_analysis', {})
                    count = step_data.get(f'{step.lower()}_vertex_count', 0)
                    if count > 0:
                        vertex_counts_step.append(count)
            
            if vertex_counts_step:
                axes[i].hist(vertex_counts_step, bins=15, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{step} - Vertex Count Distribution')
                axes[i].set_xlabel('Vertex Count')
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
                
                # Add statistics
                mean_count = np.mean(vertex_counts_step)
                axes[i].axvline(mean_count, color='red', linestyle='--', 
                               label=f'Mean: {mean_count:.0f}')
                axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'step_by_step_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Category Performance Heatmap
        if len(categories) > 1:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            performance_matrix = []
            metrics = ['Centering', 'Scaling', 'Alignment', 'Overall']
            
            for category in categories:
                row = []
                total = success_rates[category]['total']
                if total > 0:
                    row.append(success_rates[category]['centering'] / total * 100)
                    row.append(success_rates[category]['scaling'] / total * 100)
                    row.append(success_rates[category]['alignment'] / total * 100)
                    overall = (success_rates[category]['centering'] + 
                              success_rates[category]['scaling'] + 
                              success_rates[category]['alignment']) / (3 * total) * 100
                    row.append(overall)
                else:
                    row = [0, 0, 0, 0]
                performance_matrix.append(row)
            
            performance_matrix = np.array(performance_matrix)
            
            sns.heatmap(performance_matrix, annot=True, fmt='.1f', 
                       xticklabels=metrics, yticklabels=categories,
                       cmap='RdYlGn', vmin=0, vmax=100, ax=ax)
            ax.set_title('Category Performance Heatmap (%)')
            ax.set_xlabel('Validation Metric')
            ax.set_ylabel('Category')
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'category_performance_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Validation plots saved to: {plots_dir}")
        return plots_dir
    
    def apply_enhanced_normalization(self, mesh, debug=False):
        """Apply enhanced 4-step normalization using ShapeMesh enhanced methods
        
        This method now uses the enhanced implementations in ShapeMesh that include:
        - Area-weighted barycenter with two-pass recentering
        - Post-scaling recenter safety pass
        - Numerical robustness with epsilon tolerances
        
        Args:
            mesh: ShapeMesh object
            debug: If True, print intermediate results
            
        Returns:
            normalized_vertices: np.array of normalized vertices
            normalization_stats: dict with detailed statistics
        """
        # Use the enhanced apply_full_normalization from ShapeMesh
        normalized_vertices = mesh.apply_full_normalization(debug=debug)
        
        # Calculate statistics for tracking
        vertices = mesh.vertices.copy()
        
        # Convert faces to triangles for barycenter calculations
        triangles = []
        for face in mesh.faces:
            if len(face) >= 3:
                triangles.append(face[:3])
        triangles = np.array(triangles) if triangles else np.array([])
        
        # Calculate before/after statistics
        bary_before_norm = np.linalg.norm(calculate_mass_barycenter(vertices, triangles))
        bary_after_norm = np.linalg.norm(calculate_mass_barycenter(normalized_vertices, triangles))
        bbox_before_scaling = np.max(np.ptp(vertices, axis=0))
        bbox_after_scaling = np.max(np.ptp(normalized_vertices, axis=0))
        
        normalization_stats = {
            'bary_before_translation': bary_before_norm,
            'bary_after_translation': bary_after_norm,
            'bbox_before_scaling': bbox_before_scaling,
            'bbox_after_scaling': bbox_after_scaling
        }
        
        return normalized_vertices, normalization_stats
    
    def process_shape(self, row, dataset_name):
        """
        Process a single shape through the complete unified pipeline:
        1. Load original mesh
        2. Apply remeshing if needed
        3. Apply your existing complete 4-step normalization
        4. Save results with metadata
        """
        try:
            original_filepath = Path(row['filepath'])
            
            # Smart skip logic: check if shape is already processed
            category = row.get('category', 'Unknown')
            category_dir = self.output_base_dir / dataset_name / category
            base_name = Path(row.get('filename', original_filepath.name)).stem
            normalized_obj_path = category_dir / f"{base_name}_unified.obj"
            metadata_path = category_dir / f"{base_name}_metadata.json"
            
            if normalized_obj_path.exists() and metadata_path.exists():
                # Check if output is newer than input
                output_mtime = min(normalized_obj_path.stat().st_mtime, metadata_path.stat().st_mtime)
                input_mtime = original_filepath.stat().st_mtime
                
                if output_mtime > input_mtime:
                    print(f"  ✅ Already processed: {original_filepath.name}")
                    self.stats['successful'] += 1
                    self.stats['total_processed'] += 1
                    return True
                else:
                    print(f"  🔄 Input file newer than output, reprocessing: {original_filepath.name}")
            
            # Step 1 & 2: Load and remesh if needed
            vertices, faces, was_remeshed = self.apply_remeshing_if_needed(
                original_filepath, self.target_vertices
            )
            
            if vertices is None:
                print(f"❌ Failed to load/remesh: {original_filepath.name}")
                return False
            
            # Store original vertices for step-by-step validation
            # ALWAYS load from original file to ensure we have truly unmodified vertices
            print(f"  📥 Loading original unmodified mesh for validation...")
            original_mesh = o3d.io.read_triangle_mesh(str(original_filepath))
            original_vertices = np.asarray(original_mesh.vertices)
            original_faces = np.asarray(original_mesh.triangles)
            
            print(f"  📊 Original: {len(original_vertices)} vertices, {len(original_faces)} faces")
            if was_remeshed:
                print(f"  📊 After processing: {len(vertices)} vertices, {len(faces)} faces")
            
            # Step 3: Create ShapeMesh with potentially remeshed data
            mesh = ShapeMesh(
                vertices=vertices,
                faces=faces,
                category=row.get('category'),
                filename=row.get('filename'),
                filepath=str(original_filepath),
                size=row.get('size')
            )
            
            # Step 4: Save step-by-step files for validation
            print(f"  📁 Saving step-by-step validation files...")
            step_files, step_vertices = self.save_step_by_step_files(
                mesh, original_vertices, original_faces, vertices, was_remeshed, category_dir, base_name
            )
            
            # Step 5: Perform comprehensive validation
            print(f"  🔍 Performing comprehensive validation...")
            validation_data = self.perform_comprehensive_validation(
                mesh, step_vertices, category_dir, base_name
            )
            
            # Enhanced validation with new methods
            print(f"  🔍 Performing advanced validation analysis...")
            
            # Mesh quality analysis
            final_vertices = step_vertices['scaled']
            final_faces = mesh.faces
            validation_data['mesh_quality'] = self.validate_mesh_quality(final_vertices, final_faces)
            
            # Transformation analysis
            validation_data['transformations'] = self.analyze_transformations(step_vertices)
            
            # Category-specific validation
            validation_data['category_analysis'] = self.category_specific_validation(
                mesh.category, validation_data
            )
            
            # Step 6: Apply enhanced 4-step normalization for final output
            print(f"  🔧 Applying enhanced 4-step normalization...")
            normalized_vertices, normalization_stats = self.apply_enhanced_normalization(mesh, debug=False)
            
            # Step 7: Save final normalized OBJ
            self.save_normalized_obj(mesh, normalized_vertices, normalized_obj_path, was_remeshed)
            
            # Step 8: Save metadata (enhanced with validation info)
            enhanced_metadata = {
                'step_files': {k: str(v) for k, v in step_files.items()},
                'validation_summary': {
                    'overall_success': validation_data['cross_step_validation']['overall_normalization_success'],
                    'centering_error': validation_data['cross_step_validation']['final_centering_error'],
                    'scaling_error': validation_data['cross_step_validation']['final_scaling_error'],
                    'alignment_quality': validation_data['alignment_validation'].get('alignment_quality', 0),
                    'flipping_success': validation_data['flipping_validation'].get('flipping_successful', False)
                }
            }
            self.save_enhanced_metadata(mesh, metadata_path, was_remeshed, len(vertices), len(faces), 
                                      normalization_stats, enhanced_metadata)
            
            # Store validation data for dataset summary
            if not hasattr(self, 'all_validations'):
                self.all_validations = []
            self.all_validations.append(validation_data)
            
            # Update statistics using enhanced normalization stats
            center_error = normalization_stats['bary_after_translation']
            scale_error = abs(normalization_stats['bbox_after_scaling'] - 1.0)
            
            self.stats['normalization_stats']['centering_errors'].append(center_error)
            self.stats['normalization_stats']['scaling_errors'].append(scale_error)
            
            # Also store the enhanced stats for analysis
            if 'enhanced_stats' not in self.stats:
                self.stats['enhanced_stats'] = {
                    'bary_before_translation': [],
                    'bary_after_translation': [],
                    'bbox_before_scaling': [],
                    'bbox_after_scaling': []
                }
            
            self.stats['enhanced_stats']['bary_before_translation'].append(normalization_stats['bary_before_translation'])
            self.stats['enhanced_stats']['bary_after_translation'].append(normalization_stats['bary_after_translation'])
            self.stats['enhanced_stats']['bbox_before_scaling'].append(normalization_stats['bbox_before_scaling'])
            self.stats['enhanced_stats']['bbox_after_scaling'].append(normalization_stats['bbox_after_scaling'])
            
            # Collect additional statistics for comprehensive validation
            if 'eigenvalue_analysis' in validation_data:
                eigen_data = validation_data['eigenvalue_analysis']
                if 'lambda1_over_lambda2' in eigen_data and not np.isinf(eigen_data['lambda1_over_lambda2']):
                    self.stats['eigenvalue_ratios']['lambda1_over_lambda2'].append(eigen_data['lambda1_over_lambda2'])
                if 'lambda2_over_lambda3' in eigen_data and not np.isinf(eigen_data['lambda2_over_lambda3']):
                    self.stats['eigenvalue_ratios']['lambda2_over_lambda3'].append(eigen_data['lambda2_over_lambda3'])
                if 'condition_number' in eigen_data and not np.isinf(eigen_data['condition_number']):
                    self.stats['eigenvalue_ratios']['condition_numbers'].append(eigen_data['condition_number'])
            
            if 'recentering_analysis' in validation_data:
                recenter_data = validation_data['recentering_analysis']
                if recenter_data.get('second_pass_triggered', False):
                    self.stats['recentering_triggered'].append({
                        'filename': mesh.filename,
                        'residual_norm': recenter_data['residual_barycenter_norm']
                    })
            
            if 'transformations' in validation_data:
                trans_data = validation_data['transformations']
                if 'translation_magnitude' in trans_data:
                    self.stats['transformation_magnitudes']['translations'].append(trans_data['translation_magnitude'])
                if 'rotation_angle_degrees' in trans_data:
                    self.stats['transformation_magnitudes']['rotations'].append(trans_data['rotation_angle_degrees'])
                if 'scale_factor' in trans_data:
                    self.stats['transformation_magnitudes']['scalings'].append(trans_data['scale_factor'])
            
            if 'aspect_ratio_analysis' in validation_data:
                aspect_data = validation_data['aspect_ratio_analysis']
                if 'preservation_error' in aspect_data:
                    self.stats['aspect_ratio_errors'].append(aspect_data['preservation_error'])
            
            if 'flipping_validation' in validation_data:
                flip_data = validation_data['flipping_validation']
                if 'moment_test_values' in flip_data:
                    moments = flip_data['moment_test_values']
                    if len(moments) >= 3:
                        self.stats['moment_values']['x_axis'].append(moments[0])
                        self.stats['moment_values']['y_axis'].append(moments[1])
                        self.stats['moment_values']['z_axis'].append(moments[2])
            
            # Track by category (your existing logic)
            if category not in self.stats['normalization_stats']['by_category']:
                self.stats['normalization_stats']['by_category'][category] = {
                    'count': 0, 'successful': 0
                }
            
            self.stats['normalization_stats']['by_category'][category]['count'] += 1
            
            if center_error < 1e-10 and scale_error < 1e-6:
                self.stats['normalization_stats']['by_category'][category]['successful'] += 1
                self.stats['successful'] += 1
            else:
                self.stats['failed'] += 1
            
            self.stats['total_processed'] += 1
            return True
            
        except Exception as e:
            error_msg = f"Error processing {row.get('filename', 'unknown')}: {str(e)}"
            print(f"\n❌ {error_msg}")
            traceback.print_exc()
            self.stats['errors'].append(error_msg)
            self.stats['failed'] += 1
            self.stats['total_processed'] += 1
            return False
    
    def save_obj_file(self, vertices, faces, output_path, header_info):
        """Save vertices and faces as OBJ file with custom header"""
        # Ensure directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header - replace problematic Unicode characters
            for line in header_info:
                # Replace arrow symbols with ASCII equivalents
                safe_line = str(line).replace('→', '->')
                f.write(f"# {safe_line}\n")
            f.write(f"# Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write vertices
            for vertex in vertices:
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            
            # Write faces (convert to 1-indexed for OBJ format)
            for face in faces:
                if len(face) >= 3:
                    face_indices = [str(idx + 1) for idx in face]
                    f.write(f"f {' '.join(face_indices)}\n")

    def save_normalized_obj(self, mesh, normalized_vertices, output_path, was_remeshed):
        """Save normalized mesh as OBJ file (enhanced version of your method)"""
        header_info = [
            f"Unified Preprocessed OBJ file from {mesh.filename}",
            f"Applied remeshing: {'Yes' if was_remeshed else 'No'}",
            f"Applied 4-step normalization: centering, PCA alignment, flipping, scaling"
        ]
        self.save_obj_file(normalized_vertices, mesh.faces, output_path, header_info)

    def save_step_by_step_files(self, mesh, original_vertices, original_faces, remeshed_vertices, was_remeshed, category_dir, base_name):
        """Save OBJ files for each processing step for validation"""
        step_files = {}
        
        # Step 0: Original shape - use TRULY ORIGINAL vertices and faces
        step_files['00_original'] = category_dir / f"{base_name}_00_original.obj"
        header_info = [f"Original shape: {mesh.filename}", "No processing applied - exact copy of source file"]
        self.save_obj_file(original_vertices, original_faces, step_files['00_original'], header_info)
        
        # Step 1: After remeshing (if applied)
        if was_remeshed:
            step_files['01_remeshed'] = category_dir / f"{base_name}_01_remeshed.obj"
            header_info = [
                f"After remeshing: {mesh.filename}",
                f"Vertex count: {len(original_vertices)} -> {len(remeshed_vertices)}",
                f"Target vertices: {self.target_vertices}"
            ]
            self.save_obj_file(remeshed_vertices, mesh.faces, step_files['01_remeshed'], header_info)
            working_vertices = remeshed_vertices
        else:
            working_vertices = original_vertices
        
        # Create temp mesh for step-by-step normalization
        temp_mesh = ShapeMesh(
            vertices=working_vertices,
            faces=mesh.faces,
            category=mesh.category,
            filename=mesh.filename,
            filepath=mesh.filepath,
            size=mesh.size
        )
        
        # Step 2: After translation (centering)
        step1_vertices = temp_mesh._apply_centering(working_vertices)
        step_files['02_translated'] = category_dir / f"{base_name}_02_translated.obj"
        header_info = [
            f"After translation (centering): {mesh.filename}",
            "Applied area-weighted barycenter centering",
            "Shape moved to origin"
        ]
        self.save_obj_file(step1_vertices, mesh.faces, step_files['02_translated'], header_info)
        
        # Step 3: After PCA alignment
        step2_vertices = temp_mesh._apply_pca_alignment(step1_vertices)
        step_files['03_aligned'] = category_dir / f"{base_name}_03_aligned.obj"
        header_info = [
            f"After PCA alignment: {mesh.filename}",
            "Applied principal component analysis alignment",
            "Major eigenvector -> X-axis, Medium eigenvector -> Y-axis"
        ]
        self.save_obj_file(step2_vertices, mesh.faces, step_files['03_aligned'], header_info)
        
        # Step 4: After flipping
        step3_vertices = temp_mesh._apply_flipping(step2_vertices)
        step_files['04_flipped'] = category_dir / f"{base_name}_04_flipped.obj"
        header_info = [
            f"After flipping: {mesh.filename}",
            "Applied moment test for consistent orientation",
            "All moment test values forced positive"
        ]
        self.save_obj_file(step3_vertices, mesh.faces, step_files['04_flipped'], header_info)
        
        # Step 5: After scaling (final)
        step4_vertices = temp_mesh._apply_scaling(step3_vertices)
        step_files['05_scaled'] = category_dir / f"{base_name}_05_scaled.obj"
        header_info = [
            f"Final normalized shape: {mesh.filename}",
            "Applied unit bounding box scaling",
            "Maximum dimension = 1.0, centered at origin"
        ]
        self.save_obj_file(step4_vertices, mesh.faces, step_files['05_scaled'], header_info)
        
        # Collect all step vertices for validation
        step_vertices = {
            'original': original_vertices,
            'resampled': working_vertices,
            'translated': step1_vertices,
            'aligned': step2_vertices,
            'flipped': step3_vertices,
            'scaled': step4_vertices
        }
        
        return step_files, step_vertices
        
    def perform_comprehensive_validation(self, mesh, step_vertices, category_dir, base_name):
        """Perform comprehensive validation of all normalization steps"""
        validation_data = {
            'filename': mesh.filename,
            'category': mesh.category,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'vertex_counts': {},
            'centering_validation': {},
            'alignment_validation': {},
            'flipping_validation': {},
            'scaling_validation': {},
            'cross_step_validation': {}
        }
        
        # Convert faces to triangles for calculations
        triangles = []
        for face in mesh.faces:
            if len(face) >= 3:
                triangles.append(face[:3])
        triangles = np.array(triangles) if triangles else np.array([])
        
        # Vertex counts validation
        for step_name, vertices in step_vertices.items():
            validation_data['vertex_counts'][step_name] = len(vertices)
        
        # A. Centering Validation
        for step_name, vertices in step_vertices.items():
            # Only use triangles if they're valid for the current vertex set
            if len(triangles) > 0 and len(vertices) > 0:
                # Check if triangle indices are valid for this vertex set
                max_vertex_index = np.max(triangles) if len(triangles) > 0 else 0
                if max_vertex_index < len(vertices):
                    try:
                        barycenter = calculate_mass_barycenter(vertices, triangles)
                        barycenter_distance = float(np.linalg.norm(barycenter))
                    except (IndexError, ValueError):
                        # Fall back to simple mean if mass barycenter fails
                        barycenter = np.mean(vertices, axis=0)
                        barycenter_distance = float(np.linalg.norm(barycenter))
                else:
                    # Triangles are invalid for this vertex set, use simple mean
                    barycenter = np.mean(vertices, axis=0)
                    barycenter_distance = float(np.linalg.norm(barycenter))
            else:
                barycenter = np.mean(vertices, axis=0)
                barycenter_distance = float(np.linalg.norm(barycenter))
            
            validation_data['centering_validation'][step_name] = {
                'barycenter': barycenter.tolist(),
                'distance_from_origin': barycenter_distance,
                'properly_centered': barycenter_distance < 1e-10
            }
        
        # B. Alignment Validation (PCA verification)
        aligned_vertices = step_vertices['aligned']
        try:            
            pca = PCA(n_components=3)
            pca.fit(aligned_vertices)
            
            eigenvalues = pca.explained_variance_
            eigenvectors = pca.components_
            
            # Check eigenvalue ordering (should be descending)
            eigenvalue_ordering_correct = np.all(eigenvalues[:-1] >= eigenvalues[1:])
            
            # Check if principal axes align with coordinate axes
            # After PCA alignment, major eigenvector should align with X, medium with Y
            x_alignment = np.abs(np.dot(eigenvectors[0], [1, 0, 0]))  # Major → X
            y_alignment = np.abs(np.dot(eigenvectors[1], [0, 1, 0]))  # Medium → Y
            z_alignment = np.abs(np.dot(eigenvectors[2], [0, 0, 1]))  # Minor → Z
            
            validation_data['alignment_validation'] = {
                'eigenvalues': eigenvalues.tolist(),
                'eigenvalue_ordering_correct': bool(eigenvalue_ordering_correct),
                'eigenvector_alignment': {
                    'major_to_x': float(x_alignment),
                    'medium_to_y': float(y_alignment),
                    'minor_to_z': float(z_alignment)
                },
                'alignment_quality': float((x_alignment + y_alignment + z_alignment) / 3.0)
            }
        except Exception as e:
            validation_data['alignment_validation'] = {'error': str(e)}
        
        # C. Flipping Validation (Moment test verification)
        flipped_vertices = step_vertices['flipped']
        if len(triangles) > 0 and len(flipped_vertices) > 0:
            # Check if triangle indices are valid for flipped vertices
            max_vertex_index = np.max(triangles) if len(triangles) > 0 else 0
            if max_vertex_index < len(flipped_vertices):
                try:
                    triangle_centers = []
                    for tri in triangles:
                        face_vertices = flipped_vertices[tri]
                        center = np.mean(face_vertices, axis=0)
                        triangle_centers.append(center)
                    
                    triangle_centers = np.array(triangle_centers)
                    
                    # Compute moment test values
                    moment_values = np.zeros(3)
                    for i in range(3):
                        coords = triangle_centers[:, i]
                        moment_values[i] = np.sum(np.sign(coords) * (coords ** 2))
                    
                    all_moments_positive = np.all(moment_values >= -1e-10)  # Allow small numerical errors
                    
                    validation_data['flipping_validation'] = {
                        'moment_test_values': moment_values.tolist(),
                        'all_moments_positive': bool(all_moments_positive),
                        'flipping_successful': bool(all_moments_positive)
                    }
                except (IndexError, ValueError):
                    validation_data['flipping_validation'] = {'error': 'Triangle indexing failed for flipped vertices'}
            else:
                validation_data['flipping_validation'] = {'error': 'Triangle indices exceed flipped vertex count'}
        else:
            validation_data['flipping_validation'] = {'error': 'No triangles available for moment test'}
        
        # Store bbox dimensions BEFORE scaling (from the flipped step)
        # This is needed for normalization_statistics.csv
        flipped_vertices = step_vertices['flipped']
        bbox_before_min = np.min(flipped_vertices, axis=0)
        bbox_before_max = np.max(flipped_vertices, axis=0)
        bbox_before_dimensions = bbox_before_max - bbox_before_min
        max_dimension_before_scaling = np.max(bbox_before_dimensions)
        
        validation_data['bbox_before_scaling'] = float(max_dimension_before_scaling)
        
        # D. Scaling Validation
        scaled_vertices = step_vertices['scaled']
        bbox_min = np.min(scaled_vertices, axis=0)
        bbox_max = np.max(scaled_vertices, axis=0)
        bbox_dimensions = bbox_max - bbox_min
        max_dimension = np.max(bbox_dimensions)
        
        validation_data['scaling_validation'] = {
            'bounding_box': {
                'min': bbox_min.tolist(),
                'max': bbox_max.tolist(),
                'dimensions': bbox_dimensions.tolist()
            },
            'max_dimension': float(max_dimension),
            'unit_scaling_achieved': abs(max_dimension - 1.0) < 1e-6,
            'scaling_error': abs(max_dimension - 1.0)
        }
        
        # E. Cross-Step Validation
        # Check centering preservation through steps
        centering_preserved = {}
        for step_name in ['aligned', 'flipped', 'scaled']:
            if step_name in step_vertices:
                center_dist = validation_data['centering_validation'][step_name]['distance_from_origin']
                centering_preserved[step_name] = center_dist < 1e-8
        
        # Check scaling preservation after all steps
        final_center_dist = validation_data['centering_validation']['scaled']['distance_from_origin']
        final_scaling_error = validation_data['scaling_validation']['scaling_error']
        
        validation_data['cross_step_validation'] = {
            'centering_preserved_after_alignment': centering_preserved.get('aligned', False),
            'centering_preserved_after_flipping': centering_preserved.get('flipped', False),
            'centering_preserved_after_scaling': centering_preserved.get('scaled', False),
            'final_centering_error': float(final_center_dist),
            'final_scaling_error': float(final_scaling_error),
            'overall_normalization_success': (
                final_center_dist < 1e-10 and 
                final_scaling_error < 1e-6 and
                validation_data['flipping_validation'].get('flipping_successful', False)
            )
        }
        
        # F. Eigenvalue Ratio Analysis (for PCA quality assessment)
        try:
            eigenvalues = validation_data['alignment_validation'].get('eigenvalues', [0, 0, 0])
            if len(eigenvalues) == 3 and eigenvalues[0] > 0:
                validation_data['eigenvalue_analysis'] = {
                    'eigenvalues': eigenvalues,
                    'lambda1_over_lambda2': float(eigenvalues[0] / eigenvalues[1]) if eigenvalues[1] > 1e-12 else float('inf'),
                    'lambda2_over_lambda3': float(eigenvalues[1] / eigenvalues[2]) if eigenvalues[2] > 1e-12 else float('inf'),
                    'condition_number': float(eigenvalues[0] / eigenvalues[2]) if eigenvalues[2] > 1e-12 else float('inf'),
                    'anisotropy_score': float((eigenvalues[0] - eigenvalues[2]) / eigenvalues[0]) if eigenvalues[0] > 1e-12 else 0.0
                }
        except Exception as e:
            validation_data['eigenvalue_analysis'] = {'error': str(e)}
        
        # G. Two-Pass Recentering Tracking
        if 'translated' in step_vertices:
            translated_vertices = step_vertices['translated']
            # Check if triangle indices are valid
            max_vertex_index = np.max(triangles) if len(triangles) > 0 else 0
            if len(triangles) > 0 and max_vertex_index < len(translated_vertices):
                try:
                    residual_barycenter = calculate_mass_barycenter(translated_vertices, triangles)
                except:
                    residual_barycenter = np.mean(translated_vertices, axis=0)
            else:
                residual_barycenter = np.mean(translated_vertices, axis=0)
            
            residual_distance = float(np.linalg.norm(residual_barycenter))
            
            validation_data['recentering_analysis'] = {
                'residual_barycenter_norm': residual_distance,
                'second_pass_triggered': residual_distance > RECENTER_EPS,
                'recenter_threshold': RECENTER_EPS
            }
        
        # H. Per-Vertex Displacement Analysis (track transformation magnitudes)
        vertex_displacements = {}
        step_sequence = ['original', 'resampled', 'translated', 'aligned', 'flipped', 'scaled']
        
        for i in range(1, len(step_sequence)):
            prev_step = step_sequence[i-1]
            curr_step = step_sequence[i]
            
            if prev_step in step_vertices and curr_step in step_vertices:
                prev_verts = step_vertices[prev_step]
                curr_verts = step_vertices[curr_step]
                
                # Only compute if vertex counts match
                if len(prev_verts) == len(curr_verts):
                    displacements = np.linalg.norm(curr_verts - prev_verts, axis=1)
                    vertex_displacements[f'{prev_step}_to_{curr_step}'] = {
                        'mean': float(np.mean(displacements)),
                        'max': float(np.max(displacements)),
                        'std': float(np.std(displacements)),
                        'median': float(np.median(displacements))
                    }
        
        validation_data['vertex_displacement_analysis'] = vertex_displacements
        
        # I. Aspect Ratio Preservation
        if 'original' in step_vertices and 'scaled' in step_vertices:
            original_bbox = np.ptp(step_vertices['original'], axis=0)
            scaled_bbox = np.ptp(step_vertices['scaled'], axis=0)
            
            if np.min(original_bbox) > 1e-12 and np.min(scaled_bbox) > 1e-12:
                # Normalize to get aspect ratios
                original_aspect = original_bbox / np.max(original_bbox)
                scaled_aspect = scaled_bbox / np.max(scaled_bbox)
                
                aspect_preservation_error = float(np.linalg.norm(original_aspect - scaled_aspect))
                
                validation_data['aspect_ratio_analysis'] = {
                    'original_aspect_ratio': original_aspect.tolist(),
                    'scaled_aspect_ratio': scaled_aspect.tolist(),
                    'preservation_error': aspect_preservation_error,
                    'aspect_preserved': aspect_preservation_error < 1e-6
                }
        
        # J. Compactness Metric (for geometric property preservation)
        try:
            # Compute for scaled mesh
            if len(step_vertices['scaled']) >= 4:
                hull = ConvexHull(step_vertices['scaled'])
                volume = hull.volume
                surface_area = hull.area
                
                # Compactness: C = 36π * V^2 / A^3 (sphere = 1.0)
                compactness = (36 * np.pi * volume**2) / (surface_area**3) if surface_area > 0 else 0
                
                validation_data['compactness_analysis'] = {
                    'convex_hull_volume': float(volume),
                    'convex_hull_surface_area': float(surface_area),
                    'compactness': float(compactness),
                    'sphericity': float(compactness)  # Same as compactness
                }
        except Exception as e:
            validation_data['compactness_analysis'] = {'error': str(e)}
        
        # K. Symmetry Detection (for moment test interpretation)
        bbox_dims = np.ptp(step_vertices['scaled'], axis=0)
        sorted_dims = np.sort(bbox_dims)[::-1]  # Descending order
        
        symmetry_classes = {
            'spherical': False,  # All dimensions equal
            'cylindrical': False,  # Two dimensions equal
            'asymmetric': False   # All dimensions different
        }
        
        dim_tolerance = 0.1  # 10% tolerance
        if np.allclose(sorted_dims, sorted_dims[0], rtol=dim_tolerance):
            symmetry_classes['spherical'] = True
        elif np.allclose(sorted_dims[1:], sorted_dims[1], rtol=dim_tolerance):
            symmetry_classes['cylindrical'] = True
        else:
            symmetry_classes['asymmetric'] = True
        
        validation_data['symmetry_analysis'] = {
            'bounding_box_dimensions': bbox_dims.tolist(),
            'sorted_dimensions': sorted_dims.tolist(),
            'symmetry_classification': symmetry_classes,
            'dimension_ratios': {
                'max_to_medium': float(sorted_dims[0] / sorted_dims[1]) if sorted_dims[1] > 1e-12 else float('inf'),
                'medium_to_min': float(sorted_dims[1] / sorted_dims[2]) if sorted_dims[2] > 1e-12 else float('inf')
            }
        }
        
        # Save validation data to JSON
        validation_file = category_dir / f"{base_name}_validation.json"
        # Ensure directory exists
        validation_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.complexfloating)):
                return float(obj)
            elif isinstance(obj, (np.integer)):
                return int(obj)
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert the validation data
        serializable_validation_data = convert_numpy_types(validation_data)
        
        with open(validation_file, 'w') as f:
            json.dump(serializable_validation_data, f, indent=2)
        
        return validation_data
    
    def save_normalization_statistics_csv(self, all_validations, dataset_output_dir):
        """
        Generate normalization_statistics.csv in the same format as normalization.py
        
        This creates a simple per-mesh CSV with the same columns as the original
        normalization.py script for easy comparison and analysis.
        
        Args:
            all_validations: List of validation dictionaries
            dataset_output_dir: Path to dataset-specific output directory
        """
        stats_file = dataset_output_dir / "normalization_statistics.csv"
        
        # Prepare data in the same format as normalization.py
        csv_data = []
        for i, validation in enumerate(all_validations, 1):
            # Extract the same metrics as normalization.py
            centering_validation = validation.get('centering_validation', {})
            scaling_validation = validation.get('scaling_validation', {})
            
            # Get barycenter distances before/after translation
            bary_before = centering_validation.get('original', {}).get('distance_from_origin', 0)
            bary_after = centering_validation.get('translated', {}).get('distance_from_origin', 0)
            
            # Get bbox dimensions before/after scaling
            # bbox_before_scaling: stored from the 'flipped' step (before scaling was applied)
            bbox_before = validation.get('bbox_before_scaling', 0.0)
            
            # After scaling
            bbox_after = scaling_validation.get('max_dimension', 1.0)
            
            row = {
                'mesh_index': i,
                'bary_before_translation': float(bary_before),
                'bary_after_translation': float(bary_after),
                'bbox_before_scaling': float(bbox_before),
                'bbox_after_scaling': float(bbox_after)
            }
            csv_data.append(row)
        
        # Write to CSV file in the exact same format as normalization.py        
        with open(stats_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['mesh_index', 'bary_before_translation', 'bary_after_translation', 'bbox_before_scaling', 'bbox_after_scaling']
            writer = csv_module.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(csv_data)
        
        print(f"📊 Normalization statistics CSV saved: {stats_file}")
        return stats_file
    
    def save_dataset_validation_summary(self, all_validations, dataset_output_dir):
        """Save comprehensive validation summary for entire dataset
        
        Args:
            all_validations: List of validation dictionaries
            dataset_output_dir: Path to dataset-specific output directory
        """
        summary_file = dataset_output_dir / "validation_summary.csv"
        detailed_file = dataset_output_dir / "validation_detailed.json"
        
        # Prepare CSV data
        csv_data = []
        successful_normalizations = 0
        total_shapes = len(all_validations)
        
        category_stats = {}
        
        for validation in all_validations:
            row = {
                'filename': validation['filename'],
                'category': validation['category'],
                'original_vertices': validation['vertex_counts'].get('original', 0),
                'final_vertices': validation['vertex_counts'].get('scaled', 0),
                'centering_error': validation['cross_step_validation']['final_centering_error'],
                'scaling_error': validation['cross_step_validation']['final_scaling_error'],
                'alignment_quality': validation['alignment_validation'].get('alignment_quality', 0),
                'flipping_success': validation['flipping_validation'].get('flipping_successful', False),
                'overall_success': validation['cross_step_validation']['overall_normalization_success']
            }
            csv_data.append(row)
            
            if row['overall_success']:
                successful_normalizations += 1
            
            # Category statistics
            category = validation['category']
            if category not in category_stats:
                category_stats[category] = {'total': 0, 'successful': 0}
            category_stats[category]['total'] += 1
            if row['overall_success']:
                category_stats[category]['successful'] += 1
        
        # Save CSV
        with open(summary_file, 'w', newline='', encoding='utf-8') as f:
            if csv_data:
                writer = csv_module.DictWriter(f, fieldnames=csv_data[0].keys())
                writer.writeheader()
                writer.writerows(csv_data)
        
        # Prepare detailed summary
        summary_stats = {
            'processing_summary': {
                'total_shapes': total_shapes,
                'successful_normalizations': successful_normalizations,
                'success_rate': successful_normalizations / max(total_shapes, 1) * 100,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'validation_statistics': {
                'centering_errors': {
                    'mean': np.mean([v['cross_step_validation']['final_centering_error'] for v in all_validations]),
                    'max': np.max([v['cross_step_validation']['final_centering_error'] for v in all_validations]),
                    'std': np.std([v['cross_step_validation']['final_centering_error'] for v in all_validations])
                },
                'scaling_errors': {
                    'mean': np.mean([v['cross_step_validation']['final_scaling_error'] for v in all_validations]),
                    'max': np.max([v['cross_step_validation']['final_scaling_error'] for v in all_validations]),
                    'std': np.std([v['cross_step_validation']['final_scaling_error'] for v in all_validations])
                },
                'alignment_quality': {
                    'mean': np.mean([v['alignment_validation'].get('alignment_quality', 0) for v in all_validations]),
                    'min': np.min([v['alignment_validation'].get('alignment_quality', 0) for v in all_validations]),
                    'std': np.std([v['alignment_validation'].get('alignment_quality', 0) for v in all_validations])
                },
                'flipping_success_rate': np.mean([v['flipping_validation'].get('flipping_successful', False) for v in all_validations]) * 100
            },
            'category_performance': category_stats,
            'detailed_validations': all_validations
        }
        
        # Save detailed JSON with error handling
        try:
            def convert_np(obj):
                if isinstance(obj, dict):
                    return {k: convert_np(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_np(v) for v in obj]
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                else:
                    return obj
            with open(detailed_file, 'w') as f:
                json.dump(convert_np(summary_stats), f, indent=2)
        except Exception as e:
            print(f"❌ Error saving detailed validation JSON to {detailed_file}: {e}")            
            traceback.print_exc()
        
        print(f"\n📊 Validation Summary:")
        print(f"   Total shapes validated: {total_shapes}")
        print(f"   Successful normalizations: {successful_normalizations}")
        print(f"   Success rate: {successful_normalizations / max(total_shapes, 1) * 100:.1f}%")
        print(f"   Mean centering error: {summary_stats['validation_statistics']['centering_errors']['mean']:.2e}")
        print(f"   Mean scaling error: {summary_stats['validation_statistics']['scaling_errors']['mean']:.2e}")
        print(f"   Mean alignment quality: {summary_stats['validation_statistics']['alignment_quality']['mean']:.3f}")
        print(f"   Flipping success rate: {summary_stats['validation_statistics']['flipping_success_rate']:.1f}%")
        print(f"\n📄 Validation files saved:")
        print(f"   Summary CSV: {summary_file}")
        print(f"   Detailed JSON: {detailed_file}")
        
        return summary_stats
    
    def save_enhanced_metadata(self, mesh, output_path, was_remeshed, final_vertices, final_faces, normalization_stats=None, enhanced_metadata=None):
        """Save enhanced metadata including remeshing info and enhanced normalization stats"""
        norm_info = mesh.get_normalization_info()
        
        # Convert numpy arrays to lists (your existing logic)
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        metadata = {
            'original_filename': mesh.filename,
            'category': mesh.category,
            'processing_info': {
                'remeshing_applied': was_remeshed,
                'target_vertices': self.target_vertices,
                'final_vertices_count': final_vertices,
                'final_faces_count': final_faces
            },
            'normalization_info': convert_numpy(norm_info),
            'enhanced_normalization_stats': convert_numpy(normalization_stats) if normalization_stats else None,
            'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'pipeline_version': 'unified_enhanced_v1.2_with_validation'
        }
        
        # Add enhanced metadata if provided
        if enhanced_metadata:
            metadata.update(convert_numpy(enhanced_metadata))
        
        # Ensure directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def regenerate_validation_from_existing_steps(self, dataset_name):
        """
        Regenerate validation data from existing step-by-step OBJ files
        
        This is MUCH faster than reprocessing - just reads existing files
        and recomputes validation metrics with new bbox_before_scaling data.
        
        Args:
            dataset_name: Name of the dataset to regenerate validations for
        
        Returns:
            Number of validations successfully regenerated
        """
        print(f"\n🔄 Regenerating validation data from existing step files...")
        print(f"   Dataset: {dataset_name}")
        
        dataset_output_dir = self.output_base_dir / dataset_name
        
        if not dataset_output_dir.exists():
            print(f"❌ Dataset output directory not found: {dataset_output_dir}")
            return 0
        
        # Find all categories with processed shapes
        category_dirs = [d for d in dataset_output_dir.iterdir() if d.is_dir() and d.name != 'validation_plots']
        
        regenerated_count = 0
        all_validations = []
        
        for category_dir in tqdm(category_dirs, desc="Regenerating validations"):
            category = category_dir.name
            
            # Find all step file sets (look for *_05_scaled.obj files as markers)
            scaled_files = list(category_dir.glob("*_05_scaled.obj"))
            
            for scaled_file in scaled_files:
                base_name = scaled_file.stem.replace("_05_scaled", "")
                
                # Check if all step files exist
                step_files = {
                    'original': category_dir / f"{base_name}_00_original.obj",
                    'translated': category_dir / f"{base_name}_02_translated.obj",
                    'aligned': category_dir / f"{base_name}_03_aligned.obj",
                    'flipped': category_dir / f"{base_name}_04_flipped.obj",
                    'scaled': category_dir / f"{base_name}_05_scaled.obj"
                }
                
                # Check for optional remeshed file
                remeshed_file = category_dir / f"{base_name}_01_remeshed.obj"
                if remeshed_file.exists():
                    step_files['resampled'] = remeshed_file
                else:
                    # Use original as resampled if no remeshing was done
                    step_files['resampled'] = step_files['original']
                
                # Verify all required files exist
                if not all(f.exists() for f in step_files.values()):
                    print(f"⚠️  Skipping {base_name}: Missing step files")
                    continue
                
                try:
                    # Load vertices and faces from each step file
                    # Important: Each step may have different face connectivity after remeshing
                    step_vertices = {}
                    step_faces = {}
                    
                    for step_name, step_file in step_files.items():
                        mesh = o3d.io.read_triangle_mesh(str(step_file))
                        vertices = np.asarray(mesh.vertices)
                        faces = np.asarray(mesh.triangles)
                        
                        # Validate that face indices are within bounds
                        if len(faces) > 0 and len(vertices) > 0:
                            max_index = np.max(faces)
                            if max_index >= len(vertices):
                                print(f"⚠️  Warning: {base_name} {step_name} has invalid face indices (max: {max_index}, vertices: {len(vertices)})")
                                # Try to filter out invalid faces
                                valid_faces = faces[np.all(faces < len(vertices), axis=1)]
                                if len(valid_faces) > 0:
                                    faces = valid_faces
                                    print(f"   Filtered to {len(valid_faces)} valid faces")
                                else:
                                    print(f"   No valid faces found, skipping this shape")
                                    continue
                        
                        step_vertices[step_name] = vertices
                        step_faces[step_name] = faces
                    
                    # Use faces from the scaled step for ShapeMesh (most reliable)
                    final_faces = step_faces.get('scaled', step_faces.get('original'))
                    
                    # Create a minimal ShapeMesh object for validation
                    temp_mesh = ShapeMesh(
                        vertices=step_vertices['scaled'],
                        faces=final_faces,
                        category=category,
                        filename=f"{base_name}.obj",
                        filepath=str(step_files['original']),
                        size=None
                    )
                    
                    # Perform comprehensive validation with the loaded step vertices
                    validation_data = self.perform_comprehensive_validation(
                        temp_mesh, step_vertices, category_dir, base_name
                    )
                    
                    # Add additional validation analyses
                    validation_data['mesh_quality'] = self.validate_mesh_quality(
                        step_vertices['scaled'], final_faces
                    )
                    validation_data['transformations'] = self.analyze_transformations(step_vertices)
                    validation_data['category_analysis'] = self.category_specific_validation(
                        category, validation_data
                    )
                    
                    all_validations.append(validation_data)
                    regenerated_count += 1
                    
                except Exception as e:
                    print(f"❌ Error regenerating validation for {base_name}: {str(e)}")
                    continue
        
        # Store validations for summary generation
        if not hasattr(self, 'all_validations'):
            self.all_validations = []
        self.all_validations = all_validations
        
        print(f"\n✅ Successfully regenerated {regenerated_count} validation files")
        return regenerated_count
    
    def process_dataset(self, dataset_name):
        """Process all shapes in a dataset (enhanced version of your method)"""
        print(f"\nProcessing dataset: {dataset_name}")
        print("=" * 60)
        print("Enhanced Pipeline: Remeshing → Enhanced Translation → PCA → Flipping → Enhanced Scaling")
        print("Improvements: Two-pass recentering, area-weighted barycenter, post-scaling recenter")
        print("=" * 60)
        
        # Get dataset-specific output directory
        dataset_output_dir = self.output_base_dir / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get file list for dataset (your existing logic)
        file_df = get_file_tree(data_dir=dataset_name)
        
        if len(file_df) == 0:
            print(f"No files found for dataset {dataset_name}")
            return
        
        print(f"Found {len(file_df)} shapes across {file_df['category'].nunique()} categories")
        
        # Check if shapes are already processed - if so, offer to just regenerate validation
        sample_row = file_df.iloc[0]
        category = sample_row.get('category', 'Unknown')
        category_dir = dataset_output_dir / category
        base_name = Path(sample_row.get('filename', '')).stem
        normalized_obj_path = category_dir / f"{base_name}_unified.obj"
        
        if normalized_obj_path.exists():
            print(f"\n💡 Detected existing processed shapes!")
            print(f"   To save time, regenerating validation data from existing step files...")
            print(f"   (This is ~100x faster than reprocessing all meshes)")
            
            # Regenerate validation data from existing step files
            regenerated_count = self.regenerate_validation_from_existing_steps(dataset_name)
            
            if regenerated_count == 0:
                print(f"\n⚠️  No validations regenerated. Falling back to full processing...")
                # Fall through to normal processing below
                process_shapes = True
            else:
                # Skip to validation summary generation
                process_shapes = False
        else:
            # Normal processing path
            print(f"\n🔄 Processing shapes...")
            process_shapes = True
        
        # Process shapes if needed
        if process_shapes:
            # Process each shape (your existing loop structure)
            for idx, row in tqdm(file_df.iterrows(), total=len(file_df), desc=f"Processing {dataset_name}"):
                success = self.process_shape(row, dataset_name)
                
                # Progress logging every 50 shapes
                if (idx + 1) % 50 == 0:
                    success_rate = self.stats['successful'] / max(self.stats['total_processed'], 1) * 100
                    print(f"\nProgress: {idx+1}/{len(file_df)} ({success_rate:.1f}% success rate)")
        
        # Generate validation summary for this dataset
        if hasattr(self, 'all_validations') and self.all_validations:
            print(f"\n🔍 Generating comprehensive validation summary for {dataset_name}...")
            
            # Save comprehensive validation summary (enhanced CSV + detailed JSON)
            try:
                validation_stats = self.save_dataset_validation_summary(self.all_validations, dataset_output_dir)
            except Exception as e:
                print(f"Error occurred while saving validation summary: {e}")
                print(traceback.format_exc())
                validation_stats = {}

            # Save normalization_statistics.csv in normalization.py format for easy comparison
            try:                    
                print(f"\n📊 Generating normalization_statistics.csv (normalization.py format)...")
                self.save_normalization_statistics_csv(self.all_validations, dataset_output_dir)
            except Exception as e:
                print(f"Error occurred while saving normalization statistics: {e}")
                print(traceback.format_exc())

            # Create comprehensive validation plots
            try:
                print(f"\n📊 Creating validation visualization plots...")
                plots_dir = self.create_validation_plots(str(dataset_output_dir), self.all_validations)
                print(f"✅ Validation plots saved to: {plots_dir}")
            except Exception as e:
                print(f"Error occurred while creating validation plots: {e}")
                print(traceback.format_exc())

            # Save dataset-specific processing report
            try:
                print(f"\n📄 Saving dataset-specific processing report...")
                self.save_processing_report(dataset_name)

            except Exception as e:
                print(f"Error occurred while saving processing report: {e}")
                print(traceback.format_exc())

            return validation_stats
    
    def _convert_numpy_types(self, obj):
        """Recursively convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        else:
            return obj
    
    def save_processing_report(self, dataset_name=None):
        """Save comprehensive processing report (enhanced version)
        
        Args:
            dataset_name: If provided, saves dataset-specific report. Otherwise saves global summary.
        """
        if dataset_name:
            # Save dataset-specific report
            dataset_output_dir = self.output_base_dir / dataset_name
            report_path = dataset_output_dir / f"{dataset_name}_processing_report.json"
        else:
            # Save global summary report
            report_path = self.output_base_dir / "unified_processing_report.json"
        
        processing_time = time.time() - self.stats['start_time']
        
        # Enhanced report with remeshing stats
        report = {
            'dataset': dataset_name if dataset_name else 'all_datasets',
            'processing_summary': {
                'total_shapes': self.stats['total_processed'],
                'successful': self.stats['successful'],
                'failed': self.stats['failed'],
                'success_rate': self.stats['successful'] / max(self.stats['total_processed'], 1) * 100,
                'processing_time_seconds': processing_time,
                'shapes_per_second': self.stats['total_processed'] / max(processing_time, 1)
            },
            'remeshing_summary': {
                'shapes_remeshed': len(self.stats['remeshing_stats']),
                'target_vertices': self.target_vertices,
                'avg_reduction_ratio': float(np.mean([s['reduction_ratio'] for s in self.stats['remeshing_stats']])) if self.stats['remeshing_stats'] else 1.0
            },
            'normalization_quality': {
                'mean_centering_error': float(np.mean(self.stats['normalization_stats']['centering_errors'])) if self.stats['normalization_stats']['centering_errors'] else 0.0,
                'max_centering_error': float(np.max(self.stats['normalization_stats']['centering_errors'])) if self.stats['normalization_stats']['centering_errors'] else 0.0,
                'mean_scaling_error': float(np.mean(self.stats['normalization_stats']['scaling_errors'])) if self.stats['normalization_stats']['scaling_errors'] else 0.0,
                'max_scaling_error': float(np.max(self.stats['normalization_stats']['scaling_errors'])) if self.stats['normalization_stats']['scaling_errors'] else 0.0
            },
            'by_category': self.stats['normalization_stats']['by_category'],
            'errors': self.stats['errors'],
            'remeshing_details': self.stats['remeshing_stats']
        }
        
        # Convert all numpy types to native Python types
        report = self._convert_numpy_types(report)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        if dataset_name:
            print(f"📄 Dataset-specific processing report saved: {report_path}")
        else:
            print(f"📄 Global processing report saved: {report_path}")
        
        return report
    
    def print_summary(self, report):
        """Print comprehensive processing summary"""
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("UNIFIED PREPROCESSING & NORMALIZATION COMPLETE")
        summary_lines.append("=" * 80)
        
        summary = report['processing_summary']
        remesh_summary = report['remeshing_summary']
        quality = report['normalization_quality']
        
        summary_lines.append(f"📊 Processing Summary:")
        summary_lines.append(f"   Total shapes processed: {summary['total_shapes']}")
        summary_lines.append(f"   Successful: {summary['successful']}")
        summary_lines.append(f"   Failed: {summary['failed']}")
        summary_lines.append(f"   Success rate: {summary['success_rate']:.1f}%")
        summary_lines.append(f"   Processing time: {summary['processing_time_seconds']:.1f}s")
        summary_lines.append(f"   Speed: {summary['shapes_per_second']:.1f} shapes/second")
        
        summary_lines.append(f"")
        summary_lines.append(f"🔄 Remeshing Summary:")
        summary_lines.append(f"   Shapes remeshed: {remesh_summary['shapes_remeshed']}")
        summary_lines.append(f"   Target vertices: {remesh_summary['target_vertices']}")
        summary_lines.append(f"   Avg reduction ratio: {remesh_summary['avg_reduction_ratio']:.3f}")
        
        summary_lines.append(f"")
        summary_lines.append(f"🎯 Normalization Quality (using your existing verification):")
        summary_lines.append(f"   Mean centering error: {quality['mean_centering_error']:.2e}")
        summary_lines.append(f"   Max centering error: {quality['max_centering_error']:.2e}")
        summary_lines.append(f"   Mean scaling error: {quality['mean_scaling_error']:.2e}")
        summary_lines.append(f"   Max scaling error: {quality['max_scaling_error']:.2e}")
        
        # Check compliance with technical tips
        centered_ok = quality['max_centering_error'] < 1e-10
        scaled_ok = quality['max_scaling_error'] < 1e-6
        
        summary_lines.append(f"")
        if centered_ok and scaled_ok:
            summary_lines.append(f"🎉 FULL TECHNICAL TIPS COMPLIANCE ACHIEVED!")
        else:
            summary_lines.append(f"⚠️  Some shapes may not meet technical tips precision requirements")
        
        summary_lines.append(f"")
        summary_lines.append(f"📁 Output Directory: {self.output_base_dir}")
        summary_lines.append(f"📄 Detailed Report: {self.output_base_dir / 'unified_processing_report.json'}")
        
        if len(self.stats['errors']) > 0:
            summary_lines.append(f"")
            summary_lines.append(f"⚠️  {len(self.stats['errors'])} errors occurred (see report for details)")
        
        # Print to console
        print("\n" + "\n".join(summary_lines))
        
        # Save to text file
        summary_file = self.output_base_dir / "processing_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Processing Summary Report\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Pipeline Version: unified_enhanced_v1.2_with_validation\n")
            f.write("\n")
            f.write("\n".join(summary_lines))
            f.write("\n")
        
        print(f"📄 Processing summary saved to: {summary_file}")
        
        return summary_file

    def analyze_shape(self, shape_path):
        """
        Analyze a single OBJ shape file for vertex/face counts and properties
        Based on analyzer_tool.py pattern
        """
        try:
            with open(shape_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading {shape_path}: {e}")
            return None

        vertices = []
        faces = []
        face_types = set()

        for line in lines:
            line = line.strip()
            if line.startswith('v '):
                vertices.append(line)
            elif line.startswith('f '):
                face_parts = line.split()[1:]  # Skip 'f'
                faces.append(face_parts)
                face_types.add(len(face_parts))

        num_vertices = len(vertices)
        num_faces = len(faces)

        # Calculate bounding box if vertices exist
        if vertices:
            coords = []
            for v_line in vertices:
                parts = v_line.split()[1:]  # Skip 'v'
                coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
            coords = np.array(coords)
            
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)
            bbox = {
                'min': min_coords.tolist(),
                'max': max_coords.tolist(),
                'size': (max_coords - min_coords).tolist()
            }
        else:
            bbox = None

        return {
            'num_vertices': num_vertices,
            'num_faces': num_faces,
            'face_types': list(face_types),
            'bounding_box': bbox
        }

    def analyze_processed_dataset(self, dataset_name, output_dir=None):
        """
        Generate analysis CSV for a processed dataset
        Based on analyzer_tool.py pattern with smart skip logic
        
        Args:
            dataset_name: Name of the dataset (e.g., "Data", "UnifiedPreprocessed/Data")
            output_dir: Custom output directory for CSV file (defaults to dataset folder)
        """
        if output_dir is None:
            if dataset_name.startswith("UnifiedPreprocessed"):
                # For unified preprocessed, save in the UnifiedPreprocessed folder
                dataset_path = Path("../../Datasets/UnifiedPreprocessed/Data")
                csv_file = Path("../../Datasets/UnifiedPreprocessed") / f"analysis_results_unified_data.csv"
            else:
                # For original datasets, save in preprocessing folder
                dataset_path = self.output_base_dir / dataset_name
                csv_file = Path("../../Preprocessing") / f"analysis_results_{dataset_name.lower()}.csv"
        else:
            dataset_path = Path(output_dir) / dataset_name
            csv_file = Path(output_dir) / f"analysis_results_{dataset_name.lower()}.csv"
        
        # Make paths absolute
        dataset_path = dataset_path.resolve()
        csv_file = csv_file.resolve()
        
        print(f"📊 Analyzing dataset: {dataset_name}")
        print(f"   Dataset path: {dataset_path}")
        print(f"   CSV output: {csv_file}")
        
        # Ensure output directory exists
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Smart skip logic: check if CSV exists and is recent
        if csv_file.exists():
            csv_mtime = csv_file.stat().st_mtime
            try:
                dataset_mtime = max(
                    (p.stat().st_mtime for p in dataset_path.rglob("*.obj")),
                    default=0
                )
                
                if csv_mtime > dataset_mtime:
                    print(f"📊 Analysis CSV for {dataset_name} is up to date, skipping generation")
                    return csv_file
                else:
                    print(f"📊 Dataset {dataset_name} has newer files, regenerating analysis CSV")
            except Exception as e:
                print(f"⚠️  Could not check file timestamps: {e}, regenerating CSV")
        else:
            print(f"📊 Generating new analysis CSV for {dataset_name}")
        
        results = []
        total_shapes = 0
        
        # Count total shapes first for progress bar
        if not dataset_path.exists():
            print(f"❌ Dataset path does not exist: {dataset_path}")
            return None
            
        for class_folder in dataset_path.iterdir():
            if class_folder.is_dir():
                # Count specific files based on dataset type
                if "UnifiedPreprocessed" in dataset_name:
                    # For unified preprocessed, count _05_scaled.obj files (final processed files)
                    obj_files = list(class_folder.glob("*_05_scaled.obj"))
                else:
                    # For original datasets, count all .obj files
                    obj_files = list(class_folder.glob("*.obj"))
                total_shapes += len(obj_files)
        
        if total_shapes == 0:
            print(f"❌ No OBJ files found in {dataset_path}")
            return None
        
        # Analyze each shape
        with tqdm(total=total_shapes, desc=f"Analyzing {dataset_name}") as pbar:
            for class_folder in dataset_path.iterdir():
                if not class_folder.is_dir():
                    continue
                
                class_name = class_folder.name
                
                if "UnifiedPreprocessed" in dataset_name:
                    # For unified preprocessed, analyze the final processed files
                    obj_files = list(class_folder.glob("*_05_scaled.obj"))
                else:
                    # For original datasets, analyze all OBJ files
                    obj_files = list(class_folder.glob("*.obj"))
                
                for shape_file in obj_files:
                    analysis = self.analyze_shape(shape_file)
                    if analysis:
                        results.append({
                            'class': class_name,
                            'shape_file': shape_file.name,
                            **analysis
                        })
                    pbar.update(1)
        
        # Save results to CSV
        fieldnames = ['class', 'shape_file', 'num_vertices', 'num_faces', 'face_types', 'bounding_box']
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                # Convert face_types and bounding_box to strings for CSV
                row = result.copy()
                row['face_types'] = ','.join(map(str, row['face_types']))
                row['bounding_box'] = json.dumps(row['bounding_box'])
                writer.writerow(row)
        
        print(f"✅ Analysis CSV saved: {csv_file}")
        print(f"   Analyzed {len(results)} shapes across {len(set(r['class'] for r in results))} classes")
        
        return csv_file

    def generate_analysis_for_all_datasets(self):
        """
        Generate analysis CSV files for all original and processed datasets
        """
        print("\n" + "=" * 60)
        print("GENERATING ANALYSIS CSV FILES FOR ALL DATASETS")
        print("=" * 60)
        
        generated_csvs = []
        
        # 1. Generate analysis for original datasets in Preprocessing folder
        print("\n📊 ANALYZING ORIGINAL DATASETS")
        print("-" * 40)

        original_datasets = [] # ["Data", "Data_sampled", "Data_resampled", "Jet"]

        for dataset_name in original_datasets:
            try:
                # Check if original dataset exists
                original_dataset_path = Path(f"../../Datasets/{dataset_name}")
                if original_dataset_path.exists():
                    print(f"\n🔍 Analyzing original dataset: {dataset_name}")
                    csv_file = self.analyze_processed_dataset(dataset_name, output_dir=str(self.output_base_dir / dataset_name))
                    if csv_file:
                        generated_csvs.append(csv_file)
                else:
                    print(f"⚠️  Original dataset not found: {original_dataset_path}")
            except Exception as e:
                print(f"❌ Error analyzing original dataset {dataset_name}: {e}")
        
        # 2. Generate analysis for processed datasets in their respective folders
        print("\n📊 ANALYZING PROCESSED DATASETS")
        print("-" * 40)
        
        # Check if unified preprocessed dataset exists
        unified_path = Path("../../Datasets/UnifiedPreprocessed/Data")
        if unified_path.exists():
            try:
                print(f"\n🔍 Analyzing UnifiedPreprocessed dataset")
                csv_file = self.analyze_processed_dataset("UnifiedPreprocessed/Data")
                if csv_file:
                    generated_csvs.append(csv_file)
            except Exception as e:
                print(f"❌ Error analyzing UnifiedPreprocessed dataset: {e}")
        else:
            print(f"⚠️  UnifiedPreprocessed dataset not found: {unified_path}")
        
        # 3. Check for any other processed datasets in the output directory
        if self.output_base_dir.exists():
            print(f"\n🔍 Checking for additional processed datasets in: {self.output_base_dir}")
            for dataset_dir in self.output_base_dir.iterdir():
                if dataset_dir.is_dir() and dataset_dir.name not in ["validation_plots", "__pycache__"]:
                    # Check if it has OBJ files
                    has_obj_files = any(dataset_dir.rglob("*.obj"))
                    if has_obj_files:
                        try:
                            print(f"\n🔍 Analyzing additional processed dataset: {dataset_dir.name}")
                            csv_file = self.analyze_processed_dataset(dataset_dir.name)
                            if csv_file:
                                generated_csvs.append(csv_file)
                        except Exception as e:
                            print(f"❌ Error analyzing dataset {dataset_dir.name}: {e}")
        
        # 4. Summary
        print("\n" + "=" * 60)
        print("ANALYSIS CSV GENERATION COMPLETE")
        print("=" * 60)
        
        if generated_csvs:
            print(f"✅ Generated {len(generated_csvs)} analysis CSV files:")
            for csv_file in generated_csvs:
                print(f"   📄 {csv_file}")
        else:
            print("⚠️  No analysis CSV files were generated")
        
        return generated_csvs

def main():
    """Main enhanced unified preprocessing function"""
    print("🚀 Starting Enhanced Unified Preprocessing & Normalization with Comprehensive Validation")
    print("Pipeline: Vertex-based Remeshing → Enhanced 4-Step Normalization → Step-by-Step Validation")
    print("Remeshing approach: Target vertices=7500 (range: 5000-10000)")
    print("Validation features:")
    print("  • Step-by-step OBJ file generation (00_original → 05_scaled)")
    print("  • Alignment verification (PCA eigenvalue ordering)")
    print("  • Flipping verification (moment test validation)")
    print("  • Cross-step validation (centering/scaling preservation)")
    print("  • Comprehensive CSV/JSON reporting for analysis")
    print("Enhanced features from normalization.py:")
    print("  • Two-pass recentering with numerical tolerance")
    print("  • Area-weighted barycenter calculation")
    print("  • Robust handling of degenerate meshes")
    print("  • Post-scaling recenter safety pass")
    print("Leveraging robust ShapeMesh implementation with normalization.py improvements")
    
    # Use your preferred dataset
    datasets = ["Data"]  # Adjust as needed
    # datasets = ["Data_sampled"]  # Adjust as needed
    # datasets = ["Jet"]  # Adjust as needed
    
    # Initialize processor with remeshing target
    processor = UnifiedPreprocessingProcessor(target_vertices=7500)
    processor.stats['start_time'] = time.time()
    
    # # Setup directories
    # processor.setup_output_directories(datasets)
    
    # # Process each dataset (process_dataset has its own smart skip/regeneration logic)
    # for dataset in datasets:
    #     try:
    #         processor.process_dataset(dataset)
    #     except Exception as e:
    #         print(f"❌ Failed to process dataset {dataset}: {str(e)}")
    
    # # Generate and save processing report if any processing occurred
    # if processor.stats['total_processed'] > 0 or (hasattr(processor, 'all_validations') and processor.all_validations):
    #     report = processor.save_processing_report()
    #     processor.print_summary(report)
    
    # Always generate analysis CSV files (with their own smart skip logic)
    processor.generate_analysis_for_all_datasets()

if __name__ == "__main__":
    main()