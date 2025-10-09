from scipy.spatial.distance import pdist
import numpy as np
from core.shapeMesh import ShapeMesh, calculate_edge_barycenter
import math
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull, QhullError
from collections import Counter

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
    
    def fill_holes(mesh: ShapeMesh) -> ShapeMesh:
        faces = mesh.faces

        edges = []
        for face in faces:
            i, j, k = face
            edges.append(tuple(sorted((i, j))))
            edges.append(tuple(sorted((j, k))))
            edges.append(tuple(sorted((k, i))))

        edge_counts = Counter(edges)
        boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]

        boundary_edges_grouped = []
        # per hole:
        while (len(boundary_edges) > 0):
            # create a group
            group = []

            # put the first edge in
            group.append(boundary_edges[0])
            del boundary_edges[0]
            target_vertex = group[0][1]
            finish_vertex = group[0][0]

            # find all edges until you reach the starting edge
            while target_vertex != finish_vertex:
                for i in range(0, len(boundary_edges)):
                    if boundary_edges[i][0] == target_vertex:
                        group.append(boundary_edges[i])
                        target_vertex = boundary_edges[i][1]
                        del boundary_edges[i]
                        break
                    if boundary_edges[i][1] == target_vertex:
                        group.append(boundary_edges[i])
                        target_vertex = boundary_edges[i][0]
                        del boundary_edges[i]
                        break

            # add hole to result
            boundary_edges_grouped.append(group)

        # for eacht hole calculate the barycenter and add new faces to the mesh
        for group in boundary_edges_grouped:
            barycenter = calculate_edge_barycenter(mesh.vertices, group)
            mesh.vertices = np.append(mesh.vertices, [barycenter], axis=0)
            for edge in group:
                mesh.faces = np.append(mesh.faces, [[edge[0], edge[1], len(mesh.vertices) - 1]], axis=0)


        return mesh
    
