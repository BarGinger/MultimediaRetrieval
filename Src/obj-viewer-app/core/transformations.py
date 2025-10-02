from scipy.spatial.distance import pdist
import numpy as np
from core.shapeMesh import ShapeMesh
import math
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull, QhullError

class MeshTransformations:

    @staticmethod
    def create_convex_hull(mesh: ShapeMesh) -> ShapeMesh:
        """
        Create the convex hull of `mesh_obj` and return it as a new ShapeMesh.
        """
        vertices = mesh.vertices
        # create convex hull
        try:
            hull = ConvexHull(vertices)
        except QhullError as e:
            raise ValueError(f"Convex hull computation failed: {e}") from e

        faces = np.asarray(hull.simplices, dtype=np.int32)

        # remove non used vertices
        used = np.unique(faces.ravel()) # list of used vertex indices
        remap = np.full(vertices.shape[0], -1, dtype=np.int32) # -1 everywhere, otherwise index of vertex in used
        remap[used] = np.arange(used.size, dtype=np.int32) # new vertex indices

        vertices_out = vertices[used]
        faces_out = remap[faces]

        return ShapeMesh(
            vertices=vertices_out,
            faces=faces_out,
            category=mesh.category,
            filename=(mesh.filename or "mesh") + "_hull",
            face_types=None,
            bounding_box=None,
            size=None,
            filepath=None,
            base_mesh=None,  # keep None; wrap in trimesh if you want downstream features
        )
    
