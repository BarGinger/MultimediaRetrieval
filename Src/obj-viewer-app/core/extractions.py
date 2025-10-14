from scipy.spatial.distance import pdist
import numpy as np
from core.shapeMesh import ShapeMesh, calculate_mass_barycenter
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from core.transformations import MeshTransformations
from typing import Tuple, List

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

        shapetest = ShapeMesh.from_file("Datasets\\UnifiedPreprocessed\\Data\\PlantWildNonTree\\m963_unified.obj")
        
        shapetest.save_as_obj('before.obj')
        vol1 = MeshExtractions.volume(shapetest)
        shapetest = MeshTransformations.fill_holes(shapetest)
        vol2 = MeshExtractions.volume(shapetest)
        shapetest = MeshTransformations.orient_faces_consistently(shapetest)
        vol3 = MeshExtractions.volume(shapetest)
        print(f"Volume before: {vol1}, after: {vol2}, vol {vol3}")
        shapetest.save_as_obj('after.obj')

        shapetest = ShapeMesh.from_file("Datasets\\UnifiedPreprocessed\\Data\\Bed\\D00031_unified.obj")
        # A3 descriptor and histogram
        A3_hist, A3_bins = MeshExtractions.A3(shapetest)
        plt.bar(A3_bins[:-1], A3_hist, width=(A3_bins[1] - A3_bins[0]), align='edge', edgecolor='black')
        plt.xlabel('Angle (radians)')
        plt.ylabel('Frequency')
        plt.title('A3 Angle Histogram')
        plt.show()

        # D1 descriptor and histogram
        D1_hist, D1_bins = MeshExtractions.D1(shapetest)
        plt.bar(D1_bins[:-1], D1_hist, width=(D1_bins[1] - D1_bins[0]), align='edge', edgecolor='black')
        plt.xlabel('Distance from barycenter')
        plt.ylabel('Frequency')
        plt.title('D1 Distance Histogram')
        plt.show()

        # D2 descriptor and histogram
        D2_hist, D2_bins = MeshExtractions.D2(shapetest)
        plt.bar(D2_bins[:-1], D2_hist, width=(D2_bins[1] - D2_bins[0]), align='edge', edgecolor='black')
        plt.xlabel('Distance between vertices')
        plt.ylabel('Frequency')
        plt.title('D2 Distance Histogram')
        plt.show()

        pass

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
        bins = 10

        # Heuristic: sample roughly up to half the vertices but clamp between 100 and 5000
        n_samples = max(100, min(5000, num_vertices // 2))

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
            return np.zeros(bins, dtype=float), np.linspace(0.0, math.pi, bins + 1)

        # Build a bins-bin histogram over [0, pi] and normalize to frequencies (sum == 1)
        hist_counts, bin_edges = np.histogram(angles, bins=bins, range=(0.0, math.pi))
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
        bins = 10

        max_radius = math.sqrt(3.0) / 2.0

        # Heuristic: sample roughly up to half the vertices but clamp between 100 and 5000
        n_samples = max(100, min(5000, num_vertices // 2))

        # Compute mass barycenter (area-weighted) using helper from shapeMesh
        bary = calculate_mass_barycenter(mesh.vertices, mesh.faces)

        # Sample vertex indices with replacement
        idx = np.random.randint(0, num_vertices, size=n_samples)
        sampled = mesh.vertices[idx]

        # Distances from barycenter
        dists = np.linalg.norm(sampled - bary, axis=1)

        # Histogram with fixed common edges and normalize to frequencies
        hist_counts, bin_edges = np.histogram(dists, bins=bins, range=(0.0, max_radius))
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

        bins = 10

        # Maximum possible distance inside a unit box is between opposite corners
        max_dist = math.sqrt(3.0)

        # Heuristic: sample roughly up to half the vertices but clamp between 100 and 5000
        n_samples = max(100, min(5000, num_vertices // 2))

        verts = mesh.vertices

        # Sample pairs of indices with replacement
        idx = np.random.randint(0, num_vertices, size=(n_samples, 2))
        p1 = verts[idx[:, 0]]
        p2 = verts[idx[:, 1]]

        dists = np.linalg.norm(p1 - p2, axis=1)

        hist_counts, bin_edges = np.histogram(dists, bins=bins, range=(0.0, max_dist))
        total = float(hist_counts.sum())
        if total > 0.0:
            hist = hist_counts.astype(float) / total
        else:
            hist = np.zeros_like(hist_counts, dtype=float)

        return hist, bin_edges
