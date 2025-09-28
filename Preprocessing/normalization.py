#!/usr/bin/env python3
"""
Mesh Normalization Tool with Automatic Outlier Handling

This version:
1. Separates translation and scaling steps for before/after stats.
2. Automatically determines histogram outliers using mean + 3*std.
3. Toggle to only calculate 'before' stats without normalization.
"""

import open3d as o3d
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# --- Configuration ---
USE_SAMPLED_DATASET = True
CREATE_ONLY_BEFORE_PLOTS = False  # True → only before stats
BASE = Path(__file__).parent.parent.resolve()
SOURCE_ROOT = BASE / ('Datasets/Data_sampled_resampled_simple' if USE_SAMPLED_DATASET else 'Datasets/Data_resampled')
TARGET_ROOT = BASE / ('Datasets/Data_sampled_resampled_normalized' if USE_SAMPLED_DATASET else 'Datasets/Data_normalized')

NUM_BINS = 50

# --- Functions ---

def calculate_mass_barycenter(vertices, triangles):
    if len(triangles) == 0:
        return np.mean(vertices, axis=0)
    total_weighted_centroid = np.zeros(3)
    total_area = 0.0
    for triangle in triangles:
        v0, v1, v2 = vertices[triangle]
        face_centroid = (v0 + v1 + v2) / 3.0
        edge1, edge2 = v1 - v0, v2 - v0
        face_area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
        total_weighted_centroid += face_centroid * face_area
        total_area += face_area
    return total_weighted_centroid / total_area if total_area > 0 else np.mean(vertices, axis=0)

def normalize_mesh(input_path, output_path, only_before=False):
    try:
        mesh = o3d.io.read_triangle_mesh(str(input_path))
        if mesh.is_empty() or len(mesh.vertices) == 0:
            return False, None

        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        # --- Translation stats ---
        bary_before = calculate_mass_barycenter(vertices, triangles)
        bary_offset_before = np.linalg.norm(bary_before)
        vertices_translated = vertices - bary_before
        bary_offset_after_translation = np.linalg.norm(calculate_mass_barycenter(vertices_translated, triangles))
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
        print(f"[X] Error {input_path}: {e}")
        return False, None

def plot_histogram_auto_outliers(data, title, xlabel, ylabel, out_path, num_bins=NUM_BINS):
    if len(data) == 0:
        return
    mean_val = np.mean(data)
    std_val = np.std(data)
    threshold = mean_val + 1 * std_val
    data_clipped = np.clip(data, None, threshold)
    bins = np.linspace(0, threshold, num_bins)
    outliers = np.sum(np.array(data) > threshold)

    plt.figure(figsize=(6,4))
    plt.hist(data_clipped, bins=bins, alpha=0.75, edgecolor='black', label='Data')
    if outliers > 0:
        plt.bar(threshold, outliers, width=(bins[1]-bins[0]), color='red', alpha=0.5, label='Outliers')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    print(f"Processing meshes from {SOURCE_ROOT} to {TARGET_ROOT}")
    TARGET_ROOT.mkdir(exist_ok=True, parents=True)

    stats = {
        "bary_before_translation": [],
        "bary_after_translation": [],
        "bbox_before_scaling": [],
        "bbox_after_scaling": []
    }

    total_processed, successful_count, failed_count = 0, 0, 0

    for category_dir in SOURCE_ROOT.iterdir():
        if not category_dir.is_dir():
            continue
        print(f"\nProcessing category: {category_dir.name}")
        out_category_dir = TARGET_ROOT / category_dir.name
        out_category_dir.mkdir(parents=True, exist_ok=True)

        category_files = list(category_dir.glob('*.obj'))
        for i, obj_file in enumerate(category_files, 1):
            out_file = out_category_dir / obj_file.name
            print(f"  [{i:3d}/{len(category_files):3d}] {obj_file.name:<40}", end="")
            success, mesh_stats = normalize_mesh(obj_file, out_file, only_before=CREATE_ONLY_BEFORE_PLOTS)
            total_processed += 1
            if success:
                successful_count += 1
                stats["bary_before_translation"].append(mesh_stats["bary_before_translation"])
                stats["bary_after_translation"].append(mesh_stats["bary_after_translation"])
                stats["bbox_before_scaling"].append(mesh_stats["bbox_before_scaling"])
                if mesh_stats["bbox_after_scaling"] is not None:
                    stats["bbox_after_scaling"].append(mesh_stats["bbox_after_scaling"])
                print("[OK]")
            else:
                failed_count += 1
                print("[X]")

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
        plot_histogram_auto_outliers(stats["bbox_after_scaling"], "Bounding Box Longest Side After Scaling", "Length", "Count", fig_dir / "bbox-histogram-after-scaling.png")

    print(f"\nHistograms saved to {fig_dir}")

if __name__ == '__main__':
    main()
