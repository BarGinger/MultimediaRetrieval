"""Simplified resampling pipeline.

Goal:
  - Reduce complexity from the strategy-heavy `resampling.py`.
  - Avoid explosive midpoint subdivision cascades (4x growth each step).
  - Provide gentler up-sampling using point sampling + surface reconstruction.
  - Provide steady, multi-pass simplification for over-dense meshes.

Approach:
  * If mesh within acceptable range -> copy.
  * If below range -> upsample via Poisson disk point sampling + Ball Pivoting reconstruction (BPA).
      - If BPA fails or still too small, fallback to single midpoint subdivision then light decimation.
  * If above range -> iterative gentle decimation (ratio-based) until inside or just above target.

This is intentionally self-contained so you can A/B compare with the original pipeline.

Requirements: open3d>=0.18
"""
from __future__ import annotations
import open3d as o3d
from pathlib import Path
import numpy as np
import math

# Configuration (define first so later tuning constants can reference them)
TARGET_VERTEX_COUNT = 7500
MIN_ACCEPTABLE_VERTICES = 5000
MAX_ACCEPTABLE_VERTICES = 10000
GENTLE_DECIMATION_RATIO = 0.70   # retain 70% faces each pass when far
FINE_DECIMATION_RATIO = 0.85     # when close (>9000)
MAX_DECIMATION_PASSES = 8
LOG_PREFIX = "[SIMPLE]"

# --- Feature preservation tuning ---
SHARP_EDGE_ANGLE_DEG = 35.0          # Edge dihedral threshold to consider an edge sharp
MAX_SUBDIV_PASSES = 4                # Hard limit on midpoint subdivision passes
TARGET_MIN_FILL = MIN_ACCEPTABLE_VERTICES  # Subdivision stops once we pass this
TARGET_SOFT_CAP = 9000               # Upper soft stop for subdivision even if below min acceptable
ALLOW_OVERSHOOT_FACTOR = 1.25        # Stop subdividing if vertex count would exceed target*factor

# Planar / smooth preservation tuning
PLANE_K_NEIGHBORS = 12               # k-NN for plane fitting during reprojection
PLANE_RMS_TOL_FACTOR = 0.001         # RMS plane fit tolerance relative to bbox diagonal
PLANE_PROJECTION_BLEND = 0.7         # Blend weight toward planar projection when confident (higher preserves flatness)
GENERIC_BLEND = 0.25                 # Default blend toward nearest vertex cluster for non-planar regions
SKIP_REPROJECT_DIST = 1e-9           # Threshold to treat a vertex as identical to an original

# NOTE: We intentionally removed Poisson/BPA reconstruction (caused holes / smoothing) in favor
# of controlled, global midpoint subdivision with early stopping and curvature awareness.

BASE = Path(__file__).parent.parent.resolve()
SOURCE_ROOT = BASE / 'Datasets' / 'Data'
TARGET_ROOT = BASE / 'Datasets' / 'Data_resampled'


def load_mesh(path: Path) -> o3d.geometry.TriangleMesh | None:
    m = o3d.io.read_triangle_mesh(str(path))
    if m.is_empty():
        return None
    if not m.has_vertex_normals():
        m.compute_vertex_normals()
    return m


def clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    return mesh


def decimate_to_range(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    v = len(mesh.vertices)
    if v <= MAX_ACCEPTABLE_VERTICES:
        return mesh
    print(f"{LOG_PREFIX} Simplify {v} verts → target window {TARGET_VERTEX_COUNT} ± ({MIN_ACCEPTABLE_VERTICES}-{MAX_ACCEPTABLE_VERTICES})")
    passes = 0
    while passes < MAX_DECIMATION_PASSES and len(mesh.vertices) > MAX_ACCEPTABLE_VERTICES:
        passes += 1
        current_v = len(mesh.vertices)
        faces = len(mesh.triangles)
        ratio = TARGET_VERTEX_COUNT / current_v
        if ratio < 0.55:
            retain_ratio = GENTLE_DECIMATION_RATIO
        elif ratio < 0.85:
            retain_ratio = 0.78
        else:
            retain_ratio = FINE_DECIMATION_RATIO
        target_faces = max(200, int(faces * retain_ratio))
        try:
            new_mesh = mesh.simplify_quadric_decimation(target_faces)
            new_v = len(new_mesh.vertices)
            if new_v >= current_v - 20:
                # Stagnation -> force more aggressive step once
                target_faces = max(100, int(faces * 0.5))
                new_mesh = mesh.simplify_quadric_decimation(target_faces)
                new_v = len(new_mesh.vertices)
            print(f"{LOG_PREFIX}  Pass {passes}: {current_v} -> {new_v} verts (faces {faces}->{len(new_mesh.triangles)})")
            mesh = new_mesh
        except Exception as e:
            print(f"{LOG_PREFIX}  Decimation failed pass {passes}: {e}")
            break
    return mesh


def _compute_sharp_edge_mask(mesh: o3d.geometry.TriangleMesh, angle_thresh_deg: float) -> set[tuple[int,int]]:
    """Identify sharp edges (unordered vertex index pairs) based on face normal dihedral angle."""
    mesh.compute_triangle_normals()
    tris = np.asarray(mesh.triangles)
    norms = np.asarray(mesh.triangle_normals)
    edge_to_faces: dict[tuple[int,int], list[int]] = {}
    for fi, (a,b,c) in enumerate(tris):
        edges = [(a,b),(b,c),(c,a)]
        for u,v in edges:
            key = (u,v) if u < v else (v,u)
            edge_to_faces.setdefault(key, []).append(fi)
    sharp_edges = set()
    cos_thresh = math.cos(math.radians(angle_thresh_deg))
    for e, flist in edge_to_faces.items():
        if len(flist) != 2:  # boundary edge: treat as sharp to preserve silhouette
            sharp_edges.add(e)
            continue
        n1, n2 = norms[flist[0]], norms[flist[1]]
        dp = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
        if dp < cos_thresh:  # large angle => sharp
            sharp_edges.add(e)
    return sharp_edges


def feature_preserving_subdivide(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Apply a single conservative midpoint subdivision pass with light vertex re-projection
    toward original mesh surface (using nearest triangle barycentric). This helps reduce
    excessive smoothing while still increasing density.
    """
    original = mesh
    try:
        subdiv = mesh.subdivide_midpoint(number_of_iterations=1)
    except Exception:
        return mesh
    # Light feature-aware reprojection to reduce warping & ripples.
    try:
        orig_pts = np.asarray(original.vertices)
        new_pts = np.asarray(subdiv.vertices)
        if not (len(orig_pts) and len(new_pts)):
            return subdiv
        # Bounding box diagonal for scale-aware tolerances
        bb_min = orig_pts.min(axis=0)
        bb_max = orig_pts.max(axis=0)
        diag = np.linalg.norm(bb_max - bb_min) + 1e-12
        plane_rms_tol = PLANE_RMS_TOL_FACTOR * diag

        # For meshes above threshold size skip (cost) – rely on midpoint only
        if len(orig_pts) > 20000:
            return subdiv

        # Precompute for brute-force small k-NN (simple and sufficient for these sizes)
        updated = new_pts.copy()
        for i, p in enumerate(new_pts):
            # Skip if coincides with an original vertex (avoid perturbing original anchors)
            # (Using min distance quick check)
            dists_all = np.linalg.norm(orig_pts - p, axis=1)
            min_d = dists_all.min()
            if min_d < SKIP_REPROJECT_DIST:
                continue
            # Get k nearest original vertices
            if len(orig_pts) <= PLANE_K_NEIGHBORS:
                nn_idx = np.argsort(dists_all)
            else:
                nn_idx = np.argpartition(dists_all, PLANE_K_NEIGHBORS)[:PLANE_K_NEIGHBORS]
            neigh = orig_pts[nn_idx]
            c = neigh.mean(axis=0)
            # Fit plane via SVD
            M = neigh - c
            try:
                _, s, vh = np.linalg.svd(M, full_matrices=False)
            except Exception:
                continue
            normal = vh[-1]
            # Plane RMS error
            proj_dists = np.abs((M @ normal))
            rms = math.sqrt((proj_dists**2).mean())
            if rms < plane_rms_tol:
                # Project point onto plane
                off = p - c
                dist = np.dot(off, normal)
                projected = p - dist * normal
                # Blend toward projection strongly for planar region
                updated[i] = p * (1.0 - PLANE_PROJECTION_BLEND) + projected * PLANE_PROJECTION_BLEND
            else:
                # Generic gentle pull toward local neighborhood centroid to reduce drift only
                centroid = neigh.mean(axis=0)
                updated[i] = p * (1.0 - GENERIC_BLEND) + centroid * GENERIC_BLEND
        subdiv.vertices = o3d.utility.Vector3dVector(updated)
    except Exception:
        # Fail-safe: return unmodified subdivision
        return subdiv
    return subdiv


def upsample_feature_preserving(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    start_v = len(mesh.vertices)
    if start_v >= MIN_ACCEPTABLE_VERTICES:
        return mesh
    print(f"{LOG_PREFIX} Upsample (feature-preserving) {start_v} verts")
    passes = 0
    while passes < MAX_SUBDIV_PASSES and len(mesh.vertices) < TARGET_MIN_FILL:
        current_v = len(mesh.vertices)
        # Predict overshoot: midpoint roughly ~4x faces, ~2x vertices (empirical). Stop if too big.
        if current_v * 2 > TARGET_VERTEX_COUNT * ALLOW_OVERSHOOT_FACTOR:
            break
        mesh = feature_preserving_subdivide(mesh)
        passes += 1
        print(f"{LOG_PREFIX}  Subdiv pass {passes}: {current_v} -> {len(mesh.vertices)} verts")
        if len(mesh.vertices) >= TARGET_SOFT_CAP:
            break
    # If still below minimum acceptable, allow one final pass even if above soft cap threshold
    if len(mesh.vertices) < MIN_ACCEPTABLE_VERTICES and passes < MAX_SUBDIV_PASSES:
        prev = len(mesh.vertices)
        mesh = feature_preserving_subdivide(mesh)
        print(f"{LOG_PREFIX}  Final assist pass: {prev}->{len(mesh.vertices)}")
    # If we overshot absolute max, trim gently
    if len(mesh.vertices) > MAX_ACCEPTABLE_VERTICES:
        mesh = decimate_to_range(mesh)
    return mesh


def process_mesh(path: Path, out_path: Path):
    mesh = load_mesh(path)
    if mesh is None:
        print(f"{LOG_PREFIX} [X] Empty mesh: {path.name}")
        return False, 0, 0
    mesh = clean_mesh(mesh)
    original_v = len(mesh.vertices)

    # Already inside window
    if MIN_ACCEPTABLE_VERTICES <= original_v <= MAX_ACCEPTABLE_VERTICES:
        o3d.io.write_triangle_mesh(str(out_path), mesh)
        print(f"{LOG_PREFIX} OK (unchanged): {original_v} verts")
        return True, original_v, original_v

    if original_v > MAX_ACCEPTABLE_VERTICES:
        mesh = decimate_to_range(mesh)
    else:
        mesh = upsample_feature_preserving(mesh)

    final_v = len(mesh.vertices)
    o3d.io.write_triangle_mesh(str(out_path), mesh)
    status = "upsampled" if original_v < MIN_ACCEPTABLE_VERTICES else "simplified" if original_v > MAX_ACCEPTABLE_VERTICES else "unchanged"
    print(f"{LOG_PREFIX} {status}: {original_v} -> {final_v} verts")
    return True, original_v, final_v


def main():
    print(f"{LOG_PREFIX} Simple resampling from {SOURCE_ROOT} -> {TARGET_ROOT}")
    print(f"{LOG_PREFIX} Target {TARGET_VERTEX_COUNT} (acceptable {MIN_ACCEPTABLE_VERTICES}-{MAX_ACCEPTABLE_VERTICES})")
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)

    stats = {"processed":0, "upsampled":0, "simplified":0, "unchanged":0,
             "original_total":0, "final_total":0}

    for cat_dir in sorted(SOURCE_ROOT.iterdir()):
        if not cat_dir.is_dir():
            continue
        out_dir = TARGET_ROOT / cat_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        cat_files = list(cat_dir.glob('*.obj'))
        if not cat_files:
            continue
        print(f"\n{LOG_PREFIX} Category: {cat_dir.name} ({len(cat_files)} files)")
        for i, f in enumerate(cat_files, 1):
            print(f"  [{i:3d}/{len(cat_files):3d}] {f.name:<32} ", end="")
            ok, ov, fv = process_mesh(f, out_dir / f.name)
            if not ok:
                continue
            stats["processed"] += 1
            stats["original_total"] += ov
            stats["final_total"] += fv
            if ov < MIN_ACCEPTABLE_VERTICES:
                stats["upsampled"] += 1
            elif ov > MAX_ACCEPTABLE_VERTICES:
                stats["simplified"] += 1
            else:
                stats["unchanged"] += 1

    if stats["processed"]:
        print("\n" + "="*60)
        print(f"{LOG_PREFIX} SUMMARY")
        print("="*60)
        print(f"Processed: {stats['processed']}")
        print(f"  Upsampled:  {stats['upsampled']}")
        print(f"  Simplified: {stats['simplified']}")
        print(f"  Unchanged:  {stats['unchanged']}")
        print(f"Avg original verts: {stats['original_total']/stats['processed']:.0f}")
        print(f"Avg final verts:    {stats['final_total']/stats['processed']:.0f}")
    else:
        print(f"{LOG_PREFIX} No meshes processed.")

if __name__ == "__main__":
    main()
