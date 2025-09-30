#!/usr/bin/env python3
"""
Mesh Normalization Tool with Automatic Outlier Handling

Enhancements in this revision:
1. Two‑pass recentring: after initial area‑weighted barycenter translation a residual
    barycenter is recomputed and, if above a small epsilon, removed again.
2. Area robustness: if the total surface area is extremely small (degenerate / thin
    meshes) we fall back to a simple vertex mean to avoid amplifying numerical noise.
3. Post‑scaling recenter: scaling should not move the barycenter, but a final safety
    pass guarantees the saved mesh has a barycenter at ~0 (within RECENTER_EPS).
4. Clear epsilon constants to make numerical tolerances explicit.

Result: After translation almost all meshes should report a barycenter distance that
falls into a single ~0 bin (floating point jitter only).
"""

import open3d as o3d
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import sys
import os
from contextlib import redirect_stderr
import datetime

# --- Configuration ---
USE_SAMPLED_DATASET = False
CREATE_ONLY_BEFORE_PLOTS = False  # True → only before stats
BASE = Path(__file__).parent.parent.resolve()
SOURCE_ROOT = BASE / ('Datasets/Data_sampled_resampled_simple' if USE_SAMPLED_DATASET else 'Datasets/Data_resampled')
TARGET_ROOT = BASE / ('Datasets/Data_sampled_resampled_normalized' if USE_SAMPLED_DATASET else 'Datasets/Data_resampled_normalized')

NUM_BINS = 50

# Numerical tolerances
AREA_EPS = 1e-12          # Minimum total surface area before falling back to mean
RECENTER_EPS = 1e-9       # Threshold to apply second recentering pass (pre-scaling)

# --- Functions ---

class LogToFile:
    """Context manager to redirect stdout and stderr output to a log file."""
    def __init__(self, log_file_path, redirect_stderr=True, redirect_stdout=False):
        self.log_file_path = log_file_path
        self.redirect_stderr = redirect_stderr
        self.redirect_stdout = redirect_stdout
        self.log_file = None
        self.old_stderr = None
        self.old_stdout = None
    
    def __enter__(self):
        self.log_file = open(self.log_file_path, 'a', encoding='utf-8')
        if self.redirect_stderr:
            self.old_stderr = sys.stderr
            sys.stderr = self.log_file
        if self.redirect_stdout:
            self.old_stdout = sys.stdout
            sys.stdout = self.log_file
        return self
    
    def __exit__(self, *args):
        if self.old_stderr:
            sys.stderr = self.old_stderr
        if self.old_stdout:
            sys.stdout = self.old_stdout
        if self.log_file:
            self.log_file.close()

def calculate_mass_barycenter(vertices, triangles):
    """Area‑weighted (mass) barycenter with degeneracy fallback.

    Falls back to simple vertex mean if:
      * Mesh has no triangles, or
      * Total accumulated triangle area < AREA_EPS
    """
    if len(triangles) == 0:
        return np.mean(vertices, axis=0)
    total_weighted_centroid = np.zeros(3, dtype=np.float64)
    total_area = 0.0
    for tri in triangles:
        v0, v1, v2 = vertices[tri]
        face_centroid = (v0 + v1 + v2) / 3.0
        edge1, edge2 = v1 - v0, v2 - v0
        face_area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
        if face_area <= 0.0:
            continue  # Skip degenerate faces
        total_weighted_centroid += face_centroid * face_area
        total_area += face_area
    if total_area < AREA_EPS:
        return np.mean(vertices, axis=0)
    return total_weighted_centroid / total_area

def normalize_mesh(input_path, output_path, only_before=False, log_file=None):
    try:
        if log_file:
            with LogToFile(log_file, redirect_stderr=True):
                mesh = o3d.io.read_triangle_mesh(str(input_path))
        else:
            mesh = o3d.io.read_triangle_mesh(str(input_path))
            
        if mesh.is_empty() or len(mesh.vertices) == 0:
            return False, None

        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        # --- Translation stats ---
        # Initial area‑weighted barycenter
        bary_before = calculate_mass_barycenter(vertices, triangles)
        bary_offset_before = float(np.linalg.norm(bary_before))

        # First translation pass
        vertices_translated = vertices - bary_before

        # Recompute barycenter after translation (should be near zero)
        residual_bary = calculate_mass_barycenter(vertices_translated, triangles)
        residual_norm = float(np.linalg.norm(residual_bary))

        # Optional second pass for numerical drift or degeneracy effects
        if residual_norm > RECENTER_EPS:
            if log_file:
                with LogToFile(log_file, redirect_stderr=True):
                    pass  # Message will be logged automatically
            else:
                tqdm.write(f"    [!] Applying second recentering pass (residual {residual_norm:.2e})")
            vertices_translated -= residual_bary
            residual_bary = calculate_mass_barycenter(vertices_translated, triangles)
            residual_norm = float(np.linalg.norm(residual_bary))

        bary_offset_after_translation = residual_norm
        bbox_before_scaling = np.max(np.max(vertices_translated, axis=0) - np.min(vertices_translated, axis=0))

        if only_before:
            return True, {
                "bary_before_translation": bary_offset_before,
                "bary_after_translation": bary_offset_after_translation,
                "bbox_before_scaling": bbox_before_scaling,
                "bbox_after_scaling": None
            }

        # --- Scaling step ---
        min_c, max_c = np.min(vertices_translated, axis=0), np.max(vertices_translated, axis=0)
        max_dim = np.max(max_c - min_c)
        vertices_scaled = vertices_translated * (1.0 / max_dim) if max_dim > 0 else vertices_translated

        mesh.vertices = o3d.utility.Vector3dVector(vertices_scaled)
        mesh.compute_vertex_normals()
        
        if log_file:
            with LogToFile(log_file, redirect_stderr=True):
                if not o3d.io.write_triangle_mesh(str(output_path), mesh):
                    return False, None
        else:
            if not o3d.io.write_triangle_mesh(str(output_path), mesh):
                return False, None

        bbox_after_scaling = np.max(np.max(vertices_scaled, axis=0) - np.min(vertices_scaled, axis=0))

        return True, {
            "bary_before_translation": bary_offset_before,
            "bary_after_translation": bary_offset_after_translation,
            "bbox_before_scaling": bbox_before_scaling,
            "bbox_after_scaling": bbox_after_scaling
        }

    except Exception as e:
        if log_file:
            with LogToFile(log_file, redirect_stderr=True):
                pass  # Error will be logged automatically
        else:
            tqdm.write(f"[X] Error {input_path}: {e}")
        return False, None

def plot_histogram_auto_outliers(
    data,
    title,
    xlabel,
    ylabel,
    out_path,
    num_bins=NUM_BINS,
    show_outlier_bar=True,
    std_factor=0.5,
    quantile_q=0.98,
    use_quantile=True,
):
    """Plot histogram with optional aggregated outlier bar.

    Parameters
    ----------
    data : sequence of float
    show_outlier_bar : bool, default True
        When False, outliers are silently clipped but not rendered as a separate bar.
    std_factor : float, default 0.5
        Multiplier for standard deviation added to mean to define clipping threshold.
    """
    if len(data) == 0:
        return
    mean_val = np.mean(data)
    std_val = np.std(data)
    threshold_std = mean_val + std_factor * std_val
    # Guard against zero std so threshold > 0 when data not all zero
    if threshold_std <= 0:
        threshold_std = max(1e-12, float(np.max(data)))
    # Quantile-based cap to avoid huge empty spans caused by extreme outliers
    if use_quantile:
        try:
            quantile_thr = float(np.quantile(data, quantile_q))
        except Exception:
            quantile_thr = threshold_std
        # Use the smaller threshold to focus on dense region
        threshold = min(threshold_std, quantile_thr)
        # Ensure threshold not absurdly tiny (at least median)
        median_val = np.median(data)
        if threshold < median_val:
            threshold = median_val
    else:
        threshold = threshold_std
    data = np.asarray(data, dtype=float)

    # Split into in-range and outliers (strict > threshold) without clipping so histogram isn't inflated
    in_range_mask = data <= threshold
    in_data = data[in_range_mask]
    outliers = np.count_nonzero(~in_range_mask)

    # Build bin edges up to threshold (inclusive at right edge)
    bins = np.linspace(0.0, threshold, num_bins + 1)
    bin_width = bins[1] - bins[0]

    plt.figure(figsize=(6,4))
    plt.hist(in_data, bins=bins, alpha=0.75, edgecolor='black', label='Data')

    if show_outlier_bar and outliers > 0:
        # Place outlier bar just beyond the last bin to avoid overlapping / double counting
        outlier_bar_x = threshold  # start at threshold
        plt.bar(outlier_bar_x, outliers, width=bin_width, align='edge', color='red', alpha=0.55, label=f'Outliers (> {threshold:.2g})')
        # Extend x-limits to show the extra bar fully
        plt.xlim(0, threshold + bin_width)
    # Compute simple occupancy metric for debug (optional future tuning)
    # non_empty_bins = np.count_nonzero(np.histogram(in_data, bins=bins)[0])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if show_outlier_bar and outliers > 0:
        plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def export_stats_to_csv(stats, output_path):
    """Export normalization statistics to CSV file.
    
    Parameters
    ----------
    stats : dict
        Dictionary containing lists of statistics for each metric
    output_path : Path
        Path where the CSV file will be saved
    """
    # Find the maximum length to handle cases where some lists might be shorter
    max_length = max(len(stats[key]) for key in stats.keys() if len(stats[key]) > 0)
    
    # Prepare data for CSV export
    csv_data = []
    for i in range(max_length):
        row = {
            'mesh_index': i + 1,
            'bary_before_translation': stats["bary_before_translation"][i] if i < len(stats["bary_before_translation"]) else None,
            'bary_after_translation': stats["bary_after_translation"][i] if i < len(stats["bary_after_translation"]) else None,
            'bbox_before_scaling': stats["bbox_before_scaling"][i] if i < len(stats["bbox_before_scaling"]) else None,
            'bbox_after_scaling': stats["bbox_after_scaling"][i] if i < len(stats["bbox_after_scaling"]) else None
        }
        csv_data.append(row)
    
    # Write to CSV file
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['mesh_index', 'bary_before_translation', 'bary_after_translation', 'bbox_before_scaling', 'bbox_after_scaling']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"Statistics exported to CSV: {output_path}")

def main():
    print(f"Processing meshes from {SOURCE_ROOT} to {TARGET_ROOT}")
    TARGET_ROOT.mkdir(exist_ok=True, parents=True)

    # Create log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = BASE / "Preprocessing/figures"
    log_dir.mkdir(exist_ok=True, parents=True)
    log_file_path = log_dir / f"normalization_log_{timestamp}.txt"
    
    # Initialize log file with header
    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write(f"Normalization Log - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source: {SOURCE_ROOT}\n")
        f.write(f"Target: {TARGET_ROOT}\n")
        f.write("="*80 + "\n\n")
    
    print(f"Logging all output to: {log_file_path}")

    stats = {
        "bary_before_translation": [],
        "bary_after_translation": [],
        "bbox_before_scaling": [],
        "bbox_after_scaling": []
    }

    total_processed, successful_count, failed_count = 0, 0, 0

    # Count total files for progress bar
    total_files = 0
    for category_dir in SOURCE_ROOT.iterdir():
        if category_dir.is_dir():
            total_files += len(list(category_dir.glob('*.obj')))

    # Initialize progress bar
    with tqdm(total=total_files, desc="Processing meshes", unit="mesh", leave=True) as pbar:
        for category_dir in SOURCE_ROOT.iterdir():
            if not category_dir.is_dir():
                continue
            out_category_dir = TARGET_ROOT / category_dir.name
            out_category_dir.mkdir(parents=True, exist_ok=True)

            category_files = list(category_dir.glob('*.obj'))
            for i, obj_file in enumerate(category_files, 1):
                out_file = out_category_dir / obj_file.name
                pbar.set_description(f"Processing {category_dir.name}: {obj_file.name[:30]}")
                
                # Log the current file being processed
                with LogToFile(log_file_path, redirect_stderr=True):
                    print(f"Processing {category_dir.name}: {obj_file.name}")
                
                success, mesh_stats = normalize_mesh(obj_file, out_file, only_before=CREATE_ONLY_BEFORE_PLOTS, log_file=log_file_path)
                total_processed += 1
                if success:
                    successful_count += 1
                    stats["bary_before_translation"].append(mesh_stats["bary_before_translation"])
                    stats["bary_after_translation"].append(mesh_stats["bary_after_translation"])
                    stats["bbox_before_scaling"].append(mesh_stats["bbox_before_scaling"])
                    if mesh_stats["bbox_after_scaling"] is not None:
                        stats["bbox_after_scaling"].append(mesh_stats["bbox_after_scaling"])
                    pbar.set_postfix({"✓": successful_count, "✗": failed_count})
                else:
                    failed_count += 1
                    pbar.set_postfix({"✓": successful_count, "✗": failed_count})
                
                pbar.update(1)

    # Summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total files processed: {total_processed}")
    print(f"[OK] Successful: {successful_count}")
    print(f"[X] Failed: {failed_count}")

    # Plot histograms
    fig_dir = BASE / "Preprocessing/figures"
    fig_dir.mkdir(exist_ok=True, parents=True)

    plot_histogram_auto_outliers(stats["bary_before_translation"], "Barycenter Offset Before Translation", "Distance to Origin", "Count", fig_dir / "barycenter-histogram-before-translation.png")
    plot_histogram_auto_outliers(stats["bary_after_translation"], "Barycenter Offset After Translation", "Distance to Origin", "Count", fig_dir / "barycenter-histogram-after-translation.png")
    plot_histogram_auto_outliers(stats["bbox_before_scaling"], "Bounding Box Longest Side Before Scaling", "Length", "Count", fig_dir / "bbox-histogram-before-scaling.png")
    if len(stats["bbox_after_scaling"]) > 0:
        # Disable separate outlier bar for 'after' plot
        plot_histogram_auto_outliers(
            stats["bbox_after_scaling"],
            "Bounding Box Longest Side After Scaling",
            "Length",
            "Count",
            fig_dir / "bbox-histogram-after-scaling.png",
            show_outlier_bar=False,
        )

    print(f"\nHistograms saved to {fig_dir}")
    
    # Export statistics to CSV
    csv_output_path = fig_dir / "normalization_statistics.csv"
    export_stats_to_csv(stats, csv_output_path)
    
    print(f"All processing logs saved to: {log_file_path}")

if __name__ == '__main__':
    main()
