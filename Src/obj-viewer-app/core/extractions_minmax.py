from scipy.spatial.distance import pdist
import numpy as np
from core.shapeMesh import ShapeMesh, calculate_mass_barycenter
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from core.transformations import MeshTransformations
from typing import Tuple, List

class MeshExtractionsMinMax:
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
        # vol1 = MeshExtractionsMinMax.volume(shapetest)
        # shapetest = MeshTransformations.fill_holes(shapetest)
        # vol2 = MeshExtractionsMinMax.volume(shapetest)
        # shapetest = MeshTransformations.orient_faces_consistently(shapetest)
        # vol3 = MeshExtractionsMinMax.volume(shapetest)
        # print(f"Volume before: {vol1}, after: {vol2}, vol {vol3}")
        # shapetest.save_as_obj('after.obj')

        shapetest = ShapeMesh.from_file("Datasets\\UnifiedPreprocessed\\Data\\Bed\\D00031_unified.obj")
        # A3 descriptor and histogram
        A3_min, A3_max = MeshExtractionsMinMax.A3(shapetest)
        print(f"A3 min: {A3_min}, A3 max: {A3_max}")
        D1_min, D1_max = MeshExtractionsMinMax.D1(shapetest)
        print(f"D1 min: {D1_min}, D1 max: {D1_max}")

        pass

    @staticmethod
    def A3(mesh: ShapeMesh) -> float:
        num_vertices = len(mesh.vertices)

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

        return float(min(angles)), float(max(angles))

    @staticmethod
    def D1(mesh: ShapeMesh) -> Tuple[np.ndarray, np.ndarray]:
        num_vertices = len(mesh.vertices)


        # Heuristic: sample roughly up to half the vertices but clamp between 100 and 5000
        n_samples = num_vertices

        # Compute mass barycenter (area-weighted) using helper from shapeMesh
        bary = calculate_mass_barycenter(mesh.vertices, mesh.faces)

        # Sample vertex indices with replacement
        idx = np.random.randint(0, num_vertices, size=n_samples)
        sampled = mesh.vertices[idx]

        # Distances from barycenter
        dists = np.linalg.norm(sampled - bary, axis=1)

        return min(dists), max(dists)

    @staticmethod
    def D2(mesh: ShapeMesh) -> Tuple[np.ndarray, np.ndarray]:
        num_vertices = len(mesh.vertices)

        bins = 30

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

        hist_counts, bin_edges = np.histogram(dists, bins=bins, range=(0.0, max_dist))
        total = float(hist_counts.sum())
        if total > 0.0:
            hist = hist_counts.astype(float) / total
        else:
            hist = np.zeros_like(hist_counts, dtype=float)

        return hist, bin_edges

    @staticmethod
    def D3(mesh: ShapeMesh) -> Tuple[np.ndarray, np.ndarray]:
        num_vertices = len(mesh.vertices)
        bins = 30

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

        hist_counts, bin_edges = np.histogram(areas, bins=bins, range=(0.0, max_area))
        total = float(hist_counts.sum())
        hist = hist_counts.astype(float) / total if total > 0 else np.zeros_like(hist_counts, dtype=float)
        return hist, bin_edges

    @staticmethod
    def D4(mesh: ShapeMesh) -> Tuple[np.ndarray, np.ndarray]:
        num_vertices = len(mesh.vertices)
        bins = 30

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

        hist_counts, bin_edges = np.histogram(vols, bins=bins, range=(0.0, max_vol))
        total = float(hist_counts.sum())
        hist = hist_counts.astype(float) / total if total > 0 else np.zeros_like(hist_counts, dtype=float)
        return hist, bin_edges
