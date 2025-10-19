from scipy.spatial.distance import pdist
import numpy as np
from core.shapeMesh import ShapeMesh, calculate_mass_barycenter
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from core.transformations import MeshTransformations
from typing import Tuple, List
import csv
from pathlib import Path

# tqdm for progress bars (optional)
try:
    from tqdm import tqdm
except Exception:
    # Fallback - identity function
    def tqdm(iterable, **kwargs):
        return iterable

class MeshExtractions:
    """
    Note that all extractions assume that the mesh is normalized, and centered around the barecenter.
    """
    def test():
        # create shapeMesh object and compute properties
        # shape1 = ShapeMesh.from_file("Datasets\\UnifiedPreprocessed\\Data\\Door\\D01005_unified.obj")
        # shape2 = ShapeMesh.from_file("Datasets\\UnifiedPreprocessed\\Data\\AircraftBuoyant\\m1338_unified.obj")
        # shape3 = ShapeMesh.from_file("Datasets\\UnifiedPreprocessed\\Data\\PlantWildNonTree\\m963_unified.obj")

        # area1 = MeshExtractions.eccentricity(shape1)
        # area2 = MeshExtractions.eccentricity(shape2)
        # area3 = MeshExtractions.eccentricity(shape3)

        # shapetest = ShapeMesh.from_file("Datasets\\UnifiedPreprocessed\\Data\\PlantWildNonTree\\m963_unified.obj")
        
        # shapetest.save_as_obj('before.obj')
        # vol1 = MeshExtractions.volume(shapetest)
        # shapetest = MeshTransformations.fill_holes(shapetest)
        # vol2 = MeshExtractions.volume(shapetest)
        # shapetest = MeshTransformations.orient_faces_consistently(shapetest)
        # vol3 = MeshExtractions.volume(shapetest)
        # print(f"Volume before: {vol1}, after: {vol2}, vol {vol3}")
        # shapetest.save_as_obj('after.obj')

        # shapetest = ShapeMesh.from_file("Datasets\\UnifiedPreprocessed\\Data\\Bed\\D00031_unified.obj")
        # # A3 descriptor and histogram
        # A3_hist, A3_bins = MeshExtractions.A3(shapetest)
        # plt.bar(A3_bins[:-1], A3_hist, width=(A3_bins[1] - A3_bins[0]), align='edge', edgecolor='black')
        # plt.xlabel('Angle (radians)')
        # plt.ylabel('Frequency')
        # plt.title('A3 Angle Histogram')
        # plt.show()

        # # D1 descriptor and histogram
        # D1_hist, D1_bins = MeshExtractions.D1(shapetest)
        # plt.bar(D1_bins[:-1], D1_hist, width=(D1_bins[1] - D1_bins[0]), align='edge', edgecolor='black')
        # plt.xlabel('Distance from barycenter')
        # plt.ylabel('Frequency')
        # plt.title('D1 Distance Histogram')
        # plt.show()

        # # D2 descriptor and histogram
        # D2_hist, D2_bins = MeshExtractions.D2(shapetest)
        # plt.bar(D2_bins[:-1], D2_hist, width=(D2_bins[1] - D2_bins[0]), align='edge', edgecolor='black')
        # plt.xlabel('Distance between vertices')
        # plt.ylabel('Frequency')
        # plt.title('D2 Distance Histogram')
        # plt.show()
        # # D3 descriptor and histogram
        # D3_hist, D3_bins = MeshExtractions.D3(shapetest)
        # plt.bar(D3_bins[:-1], D3_hist, width=(D3_bins[1] - D3_bins[0]), align='edge', edgecolor='black')
        # plt.plot(D3_bins[:-1], D3_hist, color='red', marker='o')
        # plt.xlabel('Triangle Area')
        # plt.ylabel('Frequency')
        # plt.title('D3 Triangle Area Histogram')
        # plt.show()

        # # D4 descriptor and histogram
        # D4_hist, D4_bins = MeshExtractions.D4(shapetest)
        # plt.bar(D4_bins[:-1], D4_hist, width=(D4_bins[1] - D4_bins[0]), align='edge', edgecolor='black')
        # plt.plot(D4_bins[:-1], D4_hist, color='red', marker='o')
        # plt.xlabel('Tetrahedron Volume')
        # plt.ylabel('Frequency')
        # plt.title('D4 Tetrahedron Volume Histogram')
        # plt.show()

        # Run full dataset extraction and save histograms to CSV
        MeshExtractions.compute_and_save_all_descriptors()

        pass

    @staticmethod
    def compute_and_save_all_descriptors(output_csv: str = None, data_root: str = None):
        """Compute descriptors (A3, D1, D2, D3, D4) for all shapes in the dataset and save histograms.

        The output CSV will have columns:
          name, A3_hist, A3_bins, D1_hist, D1_bins, D2_hist, D2_bins, D3_hist, D3_bins, D4_hist, D4_bins

        Histograms and bins are stored as semicolon-separated floats.
        """
        repo_root = Path(__file__).resolve().parents[3]

        # Prefer the unified prepared dataset folder
        preferred = repo_root / 'Datasets' / 'UnifiedPreprocessed' / 'Data'
        if data_root:
            data_root_path = Path(data_root)
        elif preferred.exists() and preferred.is_dir():
            data_root_path = preferred
        else:
            # Fallbacks
            for p in [repo_root / 'Data_sampled', repo_root / 'Data', repo_root / 'Data_resampled', repo_root / 'Data_sampled_resampled']:
                if p.exists() and p.is_dir():
                    data_root_path = p
                    break
            else:
                raise FileNotFoundError('Could not find dataset root. Provide data_root or ensure Datasets/UnifiedPreprocessed/Data exists.')

        # Gather all *_unified_prepared.obj files recursively
        files = sorted(data_root_path.rglob('*_unified_prepared.obj'))
        if not files:
            raise FileNotFoundError(f'No *_unified_prepared.obj files found under {data_root_path}')

        if output_csv is None:
            output_csv_path = repo_root / 'output' / 'descriptors_all_histograms.csv'
        else:
            output_csv_path = Path(output_csv)

        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            'name',
            'A3_hist', 'A3_bins',
            'D1_hist', 'D1_bins',
            'D2_hist', 'D2_bins',
            'D3_hist', 'D3_bins',
            'D4_hist', 'D4_bins'
        ]

        total = 0
        failed = 0

        with output_csv_path.open('w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # Iterate files with progress bar
            for obj_file in tqdm(files, desc='Computing descriptors', unit='file'):
                total += 1
                rel_name = str(obj_file.relative_to(repo_root))
                try:
                    mesh = ShapeMesh.from_file(str(obj_file))

                    # Compute descriptors
                    A3_hist, A3_bins = MeshExtractions.A3(mesh)
                    D1_hist, D1_bins = MeshExtractions.D1(mesh)
                    D2_hist, D2_bins = MeshExtractions.D2(mesh)
                    D3_hist, D3_bins = MeshExtractions.D3(mesh)
                    D4_hist, D4_bins = MeshExtractions.D4(mesh)

                    # Convert arrays to semicolon-separated strings
                    def arr_to_str(arr):
                        return ';'.join([f"{float(x):.6f}" for x in np.asarray(arr).tolist()])

                    row = {
                        'name': rel_name,
                        'A3_hist': arr_to_str(A3_hist),
                        'A3_bins': arr_to_str(A3_bins),
                        'D1_hist': arr_to_str(D1_hist),
                        'D1_bins': arr_to_str(D1_bins),
                        'D2_hist': arr_to_str(D2_hist),
                        'D2_bins': arr_to_str(D2_bins),
                        'D3_hist': arr_to_str(D3_hist),
                        'D3_bins': arr_to_str(D3_bins),
                        'D4_hist': arr_to_str(D4_hist),
                        'D4_bins': arr_to_str(D4_bins),
                    }

                    writer.writerow(row)
                except Exception as e:
                    print(f"Failed to process {rel_name}: {e}")
                    failed += 1

        print(f"Finished. Processed: {total - failed}, Failed: {failed}, Output: {output_csv_path}")

    @staticmethod
    def sample_vertices(mesh: ShapeMesh, n: int) -> np.ndarray:
        """Randomly sample n vertices from the mesh."""
        if n > len(mesh.vertices):
            raise ValueError("Cannot sample more vertices than exist in the mesh.")
        indices = np.random.choice(len(mesh.vertices), size=n, replace=False)
        return mesh.vertices[indices]

    @staticmethod
    def surface_area(mesh: ShapeMesh) -> float:
        """
        Compute surface area of a shapeMesh object.
        Uses: A = 0.5 * Σ || (v2 - v1) x (v3 - v1) ||
        """
        area = 0.0
        for f in mesh.faces:
            if len(f) != 3:
                raise ValueError("Non-triangular face detected.")
            v1, v2, v3 = mesh.vertices[f[:3]]
            area += 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
        return area
    
    @staticmethod
    def surface_area_obb(mesh: ShapeMesh) -> float:
        """
        Compute surface area of the oriented bounding box of a shapeMesh object.
        Uses: A = 2(w*h + h*d + w*d)
        """
        w, h, d = mesh.dimensions
        return 2 * (w * h + h * d + w * d)

    
    @staticmethod
    def volume(mesh: ShapeMesh) -> float:
        """
        Enclosed volume of a mesh.
        Uses: V = (1/6) * Σ ((v0 x v1) · v2)
        """

        vol = 0.0
        for f in mesh.faces:
            if len(f) != 3:
                raise ValueError("Non-triangular face detected.")
            v1, v2, v3 = mesh.vertices[f[:3]]
            vol += np.dot(np.cross(v1, v2), v3)
        return abs(vol) / 6.0
    
    @staticmethod
    def compactness(mesh: ShapeMesh, S: float = None, V: float = None) -> float:
        """
        Sphere-compactness measure: S^3 / (36π V^2) (from the slides).
        = 1 for a perfect sphere, >1 for less compact shapes.
        """
        if S is None:
            S = MeshExtractions.surface_area(mesh)
        if V is None:
            V = MeshExtractions.volume(mesh)
        if S <= 0 or V <= 0:
            return 0.0
        return float((S ** 3) / (36.0 * math.pi * (V ** 2)))

    @staticmethod
    def rectangularity(mesh: ShapeMesh, surface_area_mesh: float = None, surface_area_obb: float = None) -> float:
        """Returns the 3D rectangularity (shape volume divided by OBB volume)."""
        if surface_area_obb is None:
            surface_area_obb = MeshExtractions.surface_area_obb(mesh)
        if surface_area_mesh is None:
            surface_area_mesh = MeshExtractions.surface_area(mesh)
        if surface_area_obb == 0:
            return 0
        
        return surface_area_mesh / surface_area_obb

    @staticmethod
    def diameter(mesh: ShapeMesh) -> float:
        """Returns the diameter (largest distance between any two vertices)"""
        return float(pdist(mesh.vertices, metric="euclidean").max())

    @staticmethod
    def convexity(mesh, V_mesh: float = None, V_hull: float = None) -> float:
        """Returns the convexity (shape volume divided by convex hull volume)."""
        if V_mesh is None:
            V_mesh = MeshExtractions.volume(mesh)
        if V_hull is None:
            V_hull = MeshExtractions.volume(MeshTransformations.create_convex_hull(mesh))
        
        if V_hull == 0:
            return 0
        return V_mesh / V_hull
    
    @staticmethod
    def get_eigen_values_vectors(mesh: ShapeMesh) -> Tuple[List[float], List[List[float]]]:
        """Compute eigenvalues (λ) and eigenvectors of the mesh vertices."""
        vertices = mesh.vertices

        cov = np.cov(vertices, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort descending
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        return eigenvalues, eigenvectors.T
    

    @staticmethod
    def eccentricity(mesh: ShapeMesh) -> float:
        """Returns the eccentricity (ratio of largest to smallest eigenvalues of covariance matrix)."""
        eigenvalues, _ = MeshExtractions.get_eigen_values_vectors(mesh)
        
        if min(eigenvalues) == 0.0:
            return 0.0
        return max(eigenvalues) / min(eigenvalues)
    
    @staticmethod
    def A3(mesh: ShapeMesh) -> float:
        """Sample angles between triples of vertices and return their distribution as a histogram.

        For each sample we pick three vertex indices (with replacement), compute the angle at the
        first vertex between the vectors to the other two vertices, and build a histogram of those
        angles. The number of samples is chosen based on mesh size (vertices) using a simple
        heuristic to balance cost and resolution.

        Returns:
            (hist_freqs, bin_edges) where hist_freqs is an array of length 10 containing
            normalized frequencies that sum to 1.0, and bin_edges length 11.
        """
        num_vertices = len(mesh.vertices)

        # Number of histogram bins (change here to adjust resolution)
        bins = 20

        # Heuristic: sample roughly up to half the vertices but clamp between 100 and 5000
        n_samples = 100000

        verts = mesh.vertices

        # Draw indices with replacement so we can generate many samples even for small meshes
        idx = np.random.randint(0, num_vertices, size=(n_samples, 3))

        angles = []
        for trip in idx:
            v1 = verts[trip[0]]
            v2 = verts[trip[1]]
            v3 = verts[trip[2]]

            a = v2 - v1
            b = v3 - v1
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na <= 0 or nb <= 0:
                # degenerate — skip
                continue
            cos_theta = np.dot(a, b) / (na * nb)
            # numerical safety
            cos_theta = min(1.0, max(-1.0, float(cos_theta)))
            theta = math.acos(cos_theta)
            angles.append(theta)

        if len(angles) == 0:
            max_a3 = _GLOBAL_MAXES.get('A3', math.pi) * 1.10
            return np.zeros(bins, dtype=float), np.linspace(0.0, max_a3, bins + 1)

        # Build a bins-bin histogram over [0, pi] but fold any angles > upper bound into the last bin
        max_a3 = _GLOBAL_MAXES.get('A3', math.pi)  # already includes buffer when loaded
        # compute histogram up to max_a3
        hist_counts, bin_edges = np.histogram(angles, bins=bins, range=(0.0, max_a3))
        # count outliers > max_a3 and add to last bin
        outliers = np.sum(np.asarray(angles) > max_a3)
        if outliers > 0:
            hist_counts[-1] += outliers
        total = float(hist_counts.sum())
        if total > 0.0:
            hist = hist_counts.astype(float) / total
        else:
            hist = np.zeros_like(hist_counts, dtype=float)

        return hist, bin_edges

    @staticmethod
    def D1(mesh: ShapeMesh) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the D1 descriptor: distribution of distances from mass barycenter to sampled vertices.

        Uses vertex sampling with replacement (like A3). Number of samples follows the same
        heuristic as A3. The histogram uses a fixed set of bin edges for all shapes so descriptors
        are directly comparable. Shapes are assumed to be normalized to fit in a unit box, so
        the maximum possible distance from barycenter to a vertex is sqrt(3)/2.

        Returns:
            (hist_freqs, bin_edges) where hist_freqs sums to 1.0 and bin_edges has length bins+1.
        """
        num_vertices = len(mesh.vertices)

        # Number of histogram bins (fixed across shapes)
        bins = 20

        max_radius = math.sqrt(3.0) / 2.0

        # Heuristic: sample roughly up to half the vertices but clamp between 100 and 5000
        n_samples = num_vertices

        # Compute mass barycenter (area-weighted) using helper from shapeMesh
        bary = calculate_mass_barycenter(mesh.vertices, mesh.faces)

        # Sample vertex indices with replacement
        idx = np.random.randint(0, num_vertices, size=n_samples)
        sampled = mesh.vertices[idx]

        # Distances from barycenter
        dists = np.linalg.norm(sampled - bary, axis=1)

        # Histogram with fixed common edges and normalize to frequencies
        max_d1 = _GLOBAL_MAXES.get('D1', max_radius)
        hist_counts, bin_edges = np.histogram(dists, bins=bins, range=(0.0, max_d1))
        outliers = np.sum(dists > max_d1)
        if outliers > 0:
            hist_counts[-1] += outliers
        total = float(hist_counts.sum())
        if total > 0.0:
            hist = hist_counts.astype(float) / total
        else:
            hist = np.zeros_like(hist_counts, dtype=float)

        return hist, bin_edges

    @staticmethod
    def D2(mesh: ShapeMesh) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the D2 descriptor: distribution of distances between two random vertices.

        Preferences:
        - Vertex sampling with replacement (like A3/D1).
        - Sample count heuristic identical to A3/D1.
        - 10 fixed bins across all shapes, normalized frequencies returned.

        Returns:
            (hist_freqs, bin_edges) where hist_freqs sums to 1.0 and bin_edges has length bins+1.
        """
        num_vertices = len(mesh.vertices)

        bins = 20

        # Maximum possible distance inside a unit box is between opposite corners
        max_dist = math.sqrt(3.0)

        # Heuristic: sample roughly up to half the vertices but clamp between 100 and 5000
        n_samples = 100000

        verts = mesh.vertices

        # Sample pairs of indices with replacement
        idx = np.random.randint(0, num_vertices, size=(n_samples, 2))
        p1 = verts[idx[:, 0]]
        p2 = verts[idx[:, 1]]

        dists = np.linalg.norm(p1 - p2, axis=1)

        max_d2 = _GLOBAL_MAXES.get('D2', max_dist)
        hist_counts, bin_edges = np.histogram(dists, bins=bins, range=(0.0, max_d2))
        outliers = np.sum(dists > max_d2)
        if outliers > 0:
            hist_counts[-1] += outliers
        total = float(hist_counts.sum())
        if total > 0.0:
            hist = hist_counts.astype(float) / total
        else:
            hist = np.zeros_like(hist_counts, dtype=float)

        return hist, bin_edges

    @staticmethod
    def D3(mesh: ShapeMesh) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the D3 descriptor: distribution of triangle areas from 3 random vertices.

        - Vertex sampling with replacement.
        - Sample count heuristic identical to A3/D1/D2.
        - 10 fixed bins across all shapes, normalized frequencies returned.

        Notes:
        - Because shapes are normalized into a unit box, a conservative fixed upper bound for
          triangle area is 1.0 (area of the unit square). This guarantees consistent bin edges.
        """
        num_vertices = len(mesh.vertices)
        bins = 20

        max_area = 1.0

        n_samples = 100000
        verts = mesh.vertices
        idx = np.random.randint(0, num_vertices, size=(n_samples, 3))

        areas = []
        for trip in idx:
            v1 = verts[trip[0]]
            v2 = verts[trip[1]]
            v3 = verts[trip[2]]
            area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
            areas.append(area)

        max_d3 = _GLOBAL_MAXES.get('D3', max_area)
        hist_counts, bin_edges = np.histogram(areas, bins=bins, range=(0.0, max_d3))
        outliers = np.sum(np.asarray(areas) > max_d3)
        if outliers > 0:
            hist_counts[-1] += outliers
        total = float(hist_counts.sum())
        hist = hist_counts.astype(float) / total if total > 0 else np.zeros_like(hist_counts, dtype=float)
        return hist, bin_edges

    @staticmethod
    def D4(mesh: ShapeMesh) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the D4 descriptor: distribution of tetrahedron volumes from 4 random vertices.

        - Vertex sampling with replacement.
        - Sample count heuristic identical to other descriptors.
        - 10 fixed bins across all shapes, normalized frequencies returned.

        Notes:
        - Max tetrahedron volume in a unit cube is 1/6 (one canonical simplex that partitions the cube).
        """
        num_vertices = len(mesh.vertices)
        bins = 20

        max_vol = 1.0 / 6.0

        n_samples = 100000
        verts = mesh.vertices
        idx = np.random.randint(0, num_vertices, size=(n_samples, 4))

        vols = []
        for quad in idx:
            v0 = verts[quad[0]]
            v1 = verts[quad[1]]
            v2 = verts[quad[2]]
            v3 = verts[quad[3]]
            # Volume of tetrahedron = |dot((v1-v0), cross(v2-v0, v3-v0))| / 6
            vol = abs(np.dot(v1 - v0, np.cross(v2 - v0, v3 - v0))) / 6.0
            vols.append(vol)

        max_d4 = _GLOBAL_MAXES.get('D4', max_vol)
        hist_counts, bin_edges = np.histogram(vols, bins=bins, range=(0.0, max_d4))
        outliers = np.sum(np.asarray(vols) > max_d4)
        if outliers > 0:
            hist_counts[-1] += outliers
        total = float(hist_counts.sum())
        hist = hist_counts.astype(float) / total if total > 0 else np.zeros_like(hist_counts, dtype=float)
        return hist, bin_edges

# Load global maxima for descriptors (with fallback defaults)
def _load_global_maxes():
    # Determine repository root (4 levels up from core folder -> MMR)
    repo_root = Path(__file__).resolve().parents[3]
    # Prefer percentile-based cutoffs (99th) if available
    percentile_csv = repo_root / "output" / "descriptors_global_percentiles_99.csv"
    global_csv = repo_root / "output" / "descriptors_global_minmax.csv"

    # Defaults (previous hard-coded assumptions)
    defaults = {
        'A3': math.pi,
        'D1': math.sqrt(3.0) / 2.0,
        'D2': math.sqrt(3.0),
        'D3': 1.0,
        'D4': 1.0 / 6.0
    }

    # If a percentile CSV for p99 exists, prefer those values and add 10% buffer
    try:
        if percentile_csv.exists():
            with percentile_csv.open('r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                row = next(reader, None)
                if row is not None:
                    out = {}
                    for key in ['A3', 'D1', 'D2', 'D3', 'D4']:
                        field_name = f"percentile_{key}"
                        v = row.get(field_name, '')
                        try:
                            fv = float(v)
                            # apply 10% buffer
                            fv_buf = fv * 1.10 if fv > 0 else defaults[key]
                            out[key] = fv_buf
                        except Exception:
                            out[key] = defaults[key]
                    return out
    except Exception:
        # fallback to global minmax if percentile read fails
        pass

    # Fallback: read the original global min/max CSV (older behavior)
    if not global_csv.exists():
        return defaults

    try:
        with global_csv.open('r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            row = next(reader, None)
            if row is None:
                return defaults

            out = {}
            for key in ['A3', 'D1', 'D2', 'D3', 'D4']:
                field_name = f"max_{key}"
                v = row.get(field_name, '')
                try:
                    fv = float(v)
                    if fv <= 0:
                        out[key] = defaults[key]
                    else:
                        out[key] = fv
                except Exception:
                    out[key] = defaults[key]
            return out
    except Exception:
        return defaults

# Load once at import time
_GLOBAL_MAXES = _load_global_maxes()
