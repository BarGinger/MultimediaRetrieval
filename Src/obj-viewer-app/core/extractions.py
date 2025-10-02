from scipy.spatial.distance import pdist
import numpy as np
from core.shapeMesh import ShapeMesh
import math
from scipy.spatial.distance import pdist

class MeshExtractions:
    def test():
        # create shapeMesh object and compute properties
        shape1 = ShapeMesh.from_file("Datasets\\UnifiedPreprocessed\\Data\\Door\\D01005_unified.obj")
        shape2 = ShapeMesh.from_file("Datasets\\UnifiedPreprocessed\\Data\\AircraftBuoyant\\m1338_unified.obj")
        shape3 = ShapeMesh.from_file("Datasets\\UnifiedPreprocessed\\Data\\PlantWildNonTree\\m963_unified.obj")

        area1 = MeshExtractions.diameter(shape1)
        area2 = MeshExtractions.diameter(shape2)
        area3 = MeshExtractions.diameter(shape3)

        pass

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
    def convexity(mesh):
        """Returns the convexity (shape volume divided by convex hull volume)."""
        hull = mesh.convex_hull
        hull_volume = hull.volume
        if hull_volume == 0:
            return 0
        return mesh.volume / hull_volume

    @staticmethod
    def eccentricity(mesh):
        """Returns the eccentricity (ratio of largest to smallest eigenvalues of covariance matrix)."""
        cov = np.cov(mesh.vertices.T)
        eigvals = np.linalg.eigvalsh(cov)
        if np.min(eigvals) == 0:
            return 0
        return np.max(eigvals) / np.min(eigvals)