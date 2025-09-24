import pandas as pd
import json
from core.obj_parser import OBJParser  
import numpy as np
# import trimesh  # Uncomment if you use trimesh


def load_meshes_from_csv(csv_path, obj_dir):
    df = pd.read_csv(csv_path)
    meshes = [ShapeMesh.from_row(row, obj_dir) for _, row in df.iterrows()]
    return meshes


@classmethod
def from_row(cls, row, obj_dir):
    filepath = f"{obj_dir}/{row['filename']}"
    vertices, faces = OBJParser.parse_obj_file(filepath)
    bounding_box = None
    if 'bounding_box' in row and isinstance(row['bounding_box'], str):
        try:
            bounding_box = json.loads(row['bounding_box'])
        except Exception:
            bounding_box = None
    return cls(
        vertices=vertices,
        faces=faces,
        category=row.get('category') or row.get('class'),
        filename=row['filename'],
        face_types=row.get('face_types'),
        bounding_box=bounding_box
    )


class ShapeMesh:
    def __init__(self, vertices, faces, category=None, filename=None, face_types=None, bounding_box=None, base_mesh=None):
        self.vertices = np.array(vertices)
        self.faces = np.array(faces)
        self.category = category
        self.filename = filename
        self.face_types = face_types
        self.bounding_box = bounding_box  # Should be a dict with 'min' and 'max'
        # If using trimesh, wrap it
        # self.base_mesh = base_mesh or trimesh.Trimesh(vertices=self.vertices, faces=self.faces)
        self.base_mesh = base_mesh  # For now, can be None or your own mesh class

    @property
    def num_vertices(self):
        return len(self.vertices)

    @property
    def num_faces(self):
        return len(self.faces)
    @property
    def area(self):
        if self.base_mesh and hasattr(self.base_mesh, 'area'):
            return self.base_mesh.area
        # Custom area calculation fallback
        return None

    @property
    def volume(self):
        if self.base_mesh and hasattr(self.base_mesh, 'volume'):
            return self.base_mesh.volume
        # Custom volume calculation fallback
        return None

    @property
    def diameter(self):
        if len(self.vertices) < 2:
            return 0
        from scipy.spatial.distance import pdist
        return pdist(self.vertices).max()

    @property
    def compactness(self):
        if self.volume == 0 or self.area is None:
            return 0
        return (self.area ** 1.5) / self.volume

    @property
    def rectangularity(self):
        if self.base_mesh and hasattr(self.base_mesh, 'bounding_box_oriented'):
            obb = self.base_mesh.bounding_box_oriented
            obb_volume = obb.volume
            if obb_volume == 0:
                return 0
            return self.volume / obb_volume
        return None

    @property
    def convexity(self):
        if self.base_mesh and hasattr(self.base_mesh, 'convex_hull'):
            hull = self.base_mesh.convex_hull
            hull_volume = hull.volume
            if hull_volume == 0:
                return 0
            return self.volume / hull_volume
        return None

    @property
    def eccentricity(self):
        cov = np.cov(self.vertices.T)
        eigvals = np.linalg.eigvalsh(cov)
        if np.min(eigvals) == 0:
            return 0
        return np.max(eigvals) / np.min(eigvals)

    # Add more custom properties/methods as needed

    def to_dict(self):
        return {
            "category": self.category,
            "filename": self.filename,
            "num_vertices": self.num_vertices,
            "num_faces": self.num_faces,
            "face_types": self.face_types,
            "bounding_box": self.bounding_box,
            "vertices": self.vertices,
            "faces": self.faces,
            "area": self.area,
            "volume": self.volume,
            "diameter": self.diameter,
            "compactness": self.compactness,
            "rectangularity": self.rectangularity,
            "convexity": self.convexity,
            "eccentricity": self.eccentricity,
        }
    
    
