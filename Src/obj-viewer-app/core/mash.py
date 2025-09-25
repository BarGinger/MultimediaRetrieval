import pandas as pd
import json
from core.obj_parser import OBJParser  
import numpy as np
from dash import html
import os
# import trimesh  # Uncomment if you use trimesh


def _num(n):
    """Format number with commas for thousands separator."""
    return f"{int(n):,}"


def _bytes(b):
    """Format bytes with appropriate units."""
    try:
        b = int(b)
    except Exception:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    x = float(b)
    while x >= 1024 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    return f"{x:.1f} {units[i]}"


def load_meshes_from_csv(csv_path, obj_dir):
    df = pd.read_csv(csv_path)
    meshes = [ShapeMesh.from_row(row, obj_dir) for _, row in df.iterrows()]
    return meshes


class ShapeMesh:
    def __init__(self, vertices, faces, category=None, filename=None, face_types=None, bounding_box=None, base_mesh=None, size=None, filepath=None):
        self.vertices = np.array(vertices)
        self.faces = np.array(faces)
        self.category = category
        self.filename = filename
        self.face_types = face_types
        self.bounding_box = bounding_box  # Should be a dict with 'min' and 'max'
        self.size = size  # File size in bytes
        self.filepath = filepath  # Full path to the file
        # If using trimesh, wrap it
        # self.base_mesh = base_mesh or trimesh.Trimesh(vertices=self.vertices, faces=self.faces)
        self.base_mesh = base_mesh  # For now, can be None or your own mesh class
    
    @classmethod
    def from_row(cls, row, obj_dir):
        """Create ShapeMesh from a DataFrame row."""
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
            bounding_box=bounding_box,
            size=row.get('size'),
            filepath=filepath
        )
    
    @classmethod
    def from_file(cls, filepath, category=None, filename=None):
        """Create ShapeMesh directly from a file path."""
        vertices, faces = OBJParser.parse_obj_file(filepath)
        if filename is None:
            filename = os.path.basename(filepath)
        return cls(
            vertices=vertices,
            faces=faces,
            category=category,
            filename=filename,
            filepath=filepath
        )
    
    @classmethod
    def from_file_row(cls, row):
        """
        Create ShapeMesh from a file DataFrame row (from file_index).
        This is the main method that should be used in callbacks.
        """
        vertices, faces = OBJParser.parse_obj_file(row['filepath'])
        return cls(
            vertices=vertices,
            faces=faces,
            category=row.get('category'),
            filename=row.get('filename'),
            size=row.get('size'),
            filepath=row['filepath']
        )

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

    @property
    def dimensions(self):
        """Get the dimensions (width, height, depth) of the mesh."""
        if len(self.vertices) == 0:
            return [0, 0, 0]
        minc = self.vertices.min(axis=0)
        maxc = self.vertices.max(axis=0)
        return maxc - minc
    
    @property
    def quality(self):
        """Assess the quality of the mesh based on vertex and face count."""
        return "Good" if (len(self.vertices) > 100 and len(self.faces) > 50) else "Low Resolution"
    
    def get_card_header_html(self):
        """
        Generate the header part of the shape card with metadata for Dash HTML.
        This replaces the get_card_header function from callbacks.py
        
        Returns:
        html.Div - Dash HTML Div component with formatted metadata
        """        
        dims = self.dimensions
        
        return html.Div([
            html.Div([
                html.Span("📁 ", className="shape-info-icon"), html.Strong("Category: "),
                html.Span(self.category or "Unknown")
            ], className="shape-info-prop"),
            html.Div([
                html.Span("💾 ", className="shape-info-icon"), html.Strong("Size: "),
                html.Span(_bytes(self.size or 0))
            ], className="shape-info-prop"),
            html.Div([
                html.Span("🔺 ", className="shape-info-icon"), html.Strong("Vertices: "),
                html.Span(_num(len(self.vertices)))
            ], className="shape-info-prop"),
            html.Div([
                html.Span("🔷 ", className="shape-info-icon"), html.Strong("Faces: "),
                html.Span(_num(len(self.faces)))
            ], className="shape-info-prop"),
            html.Div([
                html.Span("📐 ", className="shape-info-icon"), html.Strong("Dims: "),
                html.Span(f"X {dims[0]:.2f} · Y {dims[1]:.2f} · Z {dims[2]:.2f}")
            ], className="shape-info-prop"),
            html.Div([
                html.Span("🎯 ", className="shape-info-icon"), html.Strong("Quality: "),
                html.Span(self.quality)
            ], className="shape-info-prop"),
        ], className="shape-info-header")
    
    def get_formatted_info(self):
        """
        Get formatted mesh information as a dictionary.
        Useful for non-HTML contexts.
        """
        dims = self.dimensions
        return {
            'category': self.category or "Unknown",
            'filename': self.filename or "Unknown",
            'size_formatted': _bytes(self.size or 0),
            'vertices_formatted': _num(len(self.vertices)),
            'faces_formatted': _num(len(self.faces)),
            'dimensions_formatted': f"X {dims[0]:.2f} · Y {dims[1]:.2f} · Z {dims[2]:.2f}",
            'quality': self.quality,
            'num_vertices': len(self.vertices),
            'num_faces': len(self.faces),
            'dimensions': dims
        }

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
    
    
