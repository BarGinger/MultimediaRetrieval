from scipy.spatial.distance import pdist
import numpy as np
from core.shapeMesh import ShapeMesh, calculate_edge_barycenter
import math
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull, QhullError
from collections import Counter, deque, defaultdict

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
    
    def __fill_holes(mesh: ShapeMesh) -> ShapeMesh:
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
                found = False
                for i in range(0, len(boundary_edges)):
                    if boundary_edges[i][0] == target_vertex:
                        group.append(boundary_edges[i])
                        target_vertex = boundary_edges[i][1]
                        del boundary_edges[i]
                        found = True
                        break
                    if boundary_edges[i][1] == target_vertex:
                        group.append(boundary_edges[i])
                        target_vertex = boundary_edges[i][0]
                        del boundary_edges[i]
                        found = True
                        break
                if not found:
                    print(f"Error: no connecting edge found, stopping hole filling with shape {mesh.filename}")
                    group = []
                    break
                    

            # add hole to result
            if group != []:
                boundary_edges_grouped.append(group)

        # for eacht hole calculate the barycenter and add new faces to the mesh
        for group in boundary_edges_grouped:
            barycenter = calculate_edge_barycenter(mesh.vertices, group)
            mesh.vertices = np.append(mesh.vertices, [barycenter], axis=0)
            for edge in group:
                mesh.faces = np.append(mesh.faces, [[edge[0], edge[1], len(mesh.vertices) - 1]], axis=0)


        return mesh
    
    @staticmethod
    def __orient_faces_consistently(mesh: ShapeMesh) -> ShapeMesh:
        """
        Ensure all face windings are consistent so that adjacent faces traverse a shared edge in opposite directions"""

        faces = mesh.faces.copy()                           # the faces of the original mesh
        number_of_faces = faces.shape[0]                    # the number of faces in the original mesh
        edge_to_faces = defaultdict(list)                   # dictionary of any edge to its face and the direction
        visited = np.zeros(number_of_faces, dtype=bool)     # bool array which indicates weither the BFS already visited the face

        def face_edges(face):
            a, b, c = int(face[0]), int(face[1]), int(face[2])
            return [(a, b), (b, c), (c, a)]
        
        def face_flip(fi: int): # no it is not a dessert
           # (ABC -> ACB)
            faces[fi] = faces[fi][[0, 2, 1]]

        # make an adjacency matrix for each edge
        for f in range(number_of_faces):
            for u, v in face_edges(faces[f]):
                key = (min(u, v), max(u, v))
                edge_to_faces[key].append((f, (u, v)))

        # execute a BFS
        for face in range(number_of_faces):
            if visited[face]:
                continue

            queue = deque([face])
            visited[face] = True

            while queue:
                current_face = queue.popleft()
                current_edges = face_edges(faces[current_face])

                for u, v in current_edges:
                    key = (min(u,v), max(u,v))

                    neighbours = edge_to_faces.get(key, [])

                    current_direction = (u, v)

                    for (neighbour, neighbour_direction) in neighbours:
                        if neighbour == current_face:
                            continue
                        if visited[neighbour]:
                            continue

                        # flip all neighbours to be opposite to the current face
                        if neighbour_direction == current_direction:
                            face_flip(neighbour)

                            for e2 in face_edges(faces[neighbour]):
                                key2 = (min(e2[0], e2[1]), max(e2[0], e2[1]))
                                lst = edge_to_faces[key2]
                                for i, (fidx, _) in enumerate(lst):
                                    if fidx == neighbour:
                                        lst[i] = (fidx, e2)
                                        break

                        visited[neighbour] = True
                        queue.append(neighbour)

        return ShapeMesh(
            vertices=mesh.vertices,
            faces=faces,
            category=mesh.category,
            filename=(mesh.filename or "mesh") + "_oriented",
            face_types=mesh.face_types,
            bounding_box=mesh.bounding_box,
            size=mesh.size,
            filepath=mesh.filepath,
            base_mesh=mesh.base_mesh,
        )

    @staticmethod
    def __cleanup(mesh: ShapeMesh, area_epsilon: float = 1e-12) -> ShapeMesh:
        """
        Clean a mesh by:
          1) removing degenerate faces (repeated indices or near-zero area),
          2) removing duplicate faces (ignoring orientation),
          3) dropping unreferenced vertices and reindexing faces.
        """
        vertices = mesh.vertices
        faces = mesh.faces.astype(np.int32, copy=True)

        keep = np.ones(len(faces), dtype=bool)

        # 1) removing degenerate faces (repeated indices or near-zero area)
        non_repeated = np.array([len({int(a), int(b), int(c)}) == 3 for a, b, c in faces])
        keep &= non_repeated

        if area_epsilon is not None and area_epsilon > 0:
            face_id = faces[keep]
            if len(face_id) > 0:
                v1 = vertices[face_id[:, 0]]
                v2 = vertices[face_id[:, 1]]
                v3 = vertices[face_id[:, 2]]
                areas = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)
                nz = np.ones(len(faces), dtype=bool)
                nz_indices = np.where(keep)[0]
                nz[nz_indices] = areas > area_epsilon
                keep &= nz

        faces = faces[keep]

        # 2) removing duplicate faces (ignoring orientation)
        if len(faces) > 0:
            seen = set()
            unique_rows = []
            for face in faces:
                key = tuple(sorted((int(face[0]), int(face[1]), int(face[2]))))
                if key in seen:
                    continue
                seen.add(key)
                unique_rows.append(face)
            faces = np.array(unique_rows, dtype=np.int32)

        # 3) Drop unreferenced vertices and reindex faces        
        used = np.unique(faces.ravel())
        remap = -np.ones(vertices.shape[0], dtype=np.int32)
        remap[used] = np.arange(used.size, dtype=np.int32)
        vertices_new = vertices[used]
        faces_new = remap[faces]

        return ShapeMesh(
            vertices=vertices_new,
            faces=faces_new,
            category=mesh.category,
            filename=(mesh.filename or "mesh") + "_cleaned",
            face_types=mesh.face_types,
            bounding_box=mesh.bounding_box,
            size=mesh.size,
            filepath=mesh.filepath,
            base_mesh=mesh.base_mesh,
        )


    @staticmethod
    def prepare_for_extraction(mesh: ShapeMesh) -> ShapeMesh:
        """
        how we are going to prepare meshes for feature extraction:
            1) Fill holes in the mesh,
            2) remove duplicate/degenerate faces and drop unused vertices,
            3) orient faces consistently.
        """
        # fill_holes mutates; make a lightweight copy first to avoid side-effects
        tmp = ShapeMesh(
            vertices=mesh.vertices.copy(),
            faces=mesh.faces.copy(),
            category=mesh.category,
            filename=mesh.filename,
            face_types=mesh.face_types,
            bounding_box=mesh.bounding_box,
            size=mesh.size,
            filepath=mesh.filepath,
            base_mesh=mesh.base_mesh,
        )

        res = MeshTransformations.__fill_holes(tmp)
        res = MeshTransformations.__cleanup(res)
        res = MeshTransformations.__orient_faces_consistently(res)
        return res
