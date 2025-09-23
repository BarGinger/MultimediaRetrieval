from scipy.spatial.distance import pdist
import numpy as np

class MeshExtractions:
    @staticmethod
    def surface_area(mesh):
        """Returns the surface area of the mesh."""
        return mesh.area

    @staticmethod
    def compactness(mesh):
        """Returns the compactness of the mesh with respect to a sphere."""
        # Compactness = Surface Area^1.5 / Volume
        if mesh.volume == 0:
            return 0
        return (mesh.area ** 1.5) / mesh.volume

    @staticmethod
    def rectangularity(mesh):
        """Returns the 3D rectangularity (shape volume divided by OBB volume)."""
        obb = mesh.bounding_box_oriented
        obb_volume = obb.volume
        if obb_volume == 0:
            return 0
        return mesh.volume / obb_volume

    @staticmethod
    def diameter(mesh):
        """Returns the diameter (largest distance between any two vertices)."""
        if len(mesh.vertices) < 2:
            return 0
        return pdist(mesh.vertices).max()

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