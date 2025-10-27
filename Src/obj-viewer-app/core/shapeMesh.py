import pandas as pd
import json
from core.obj_parser import OBJParser  
import numpy as np
from dash import html
import os
from sklearn.decomposition import PCA
# import trimesh  # Uncomment if you use trimesh

# Numerical tolerances for enhanced normalization (matching normalize_database.py)
AREA_EPS = 1e-12          # Minimum total surface area before falling back to mean
RECENTER_EPS = 1e-9       # Threshold to apply second recentering pass (pre-scaling)


def calculate_mass_barycenter(vertices, faces):
    """Area‑weighted (mass) barycenter with degeneracy fallback.
    
    Enhanced version that falls back to simple vertex mean if:
      * Mesh has no triangles, or
      * Total accumulated triangle area < AREA_EPS
    """
    if len(faces) == 0:
        return np.mean(vertices, axis=0)
    
    total_weighted_centroid = np.zeros(3, dtype=np.float64)
    total_area = 0.0
    
    for face in faces:
        v0, v1, v2 = vertices[face]
        face_centroid = (v0 + v1 + v2) / 3.0
        edge1, edge2 = v1 - v0, v2 - v0
        face_area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
        if face_area <= 0.0:
            continue  # Skip degenerate faces
        total_weighted_centroid += face_centroid * face_area
        total_area += face_area
    
    if total_area < AREA_EPS:
        return np.mean(vertices, axis=0)
    
    return total_weighted_centroid / total_area

def calculate_edge_barycenter(vertices, edges):
    """same as above but uses edges instead of faces"""
    if len(edges) == 0:
        raise ValueError("Cannot compute barycenter for empty edge list")
    
    # Calculate midpoint for each edge
    edge_midpoints = []
    for v1_idx, v2_idx in edges:
        v1 = vertices[v1_idx]
        v2 = vertices[v2_idx]
        midpoint = (v1 + v2) / 2.0
        edge_midpoints.append(midpoint)
    
    # Barycenter is the average of all edge midpoints
    edge_midpoints = np.array(edge_midpoints)
    barycenter = np.mean(edge_midpoints, axis=0)
    
    return barycenter

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
    def __init__(
        self,
        vertices,
        faces,
        category=None,
        filename=None,
        face_types=None,
        bounding_box=None,
        base_mesh=None,
        size=None,
        filepath=None,
        # analysis / derived properties (optional)
        surface_area=None,
        compactness=None,
        rectangularity=None,
        diameter=None,
        convexity=None,
        eccentricity=None,
        A3_hist=None,
        A3_bins=None,
        D1_hist=None,
        D1_bins=None,
        D2_hist=None,
        D2_bins=None,
        D3_hist=None,
        D3_bins=None,
        D4_hist=None,
        D4_bins=None,
        shape_file=None,
        name=None,
        class_b=None,
    ):
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

        # Optional analysis-derived properties (store under _* to avoid clobbering @property descriptors)
        # Also expose analysis_* aliases for direct access if needed elsewhere.
        self._surface_area = surface_area
        self.analysis_surface_area = surface_area

        self._compactness = compactness
        self.analysis_compactness = compactness

        self._rectangularity = rectangularity
        self.analysis_rectangularity = rectangularity

        self._diameter = diameter
        self.analysis_diameter = diameter

        self._convexity = convexity
        self.analysis_convexity = convexity

        self._eccentricity = eccentricity
        self.analysis_eccentricity = eccentricity

        self._A3_hist = A3_hist
        self.analysis_A3_hist = A3_hist
        self._A3_bins = A3_bins
        self.analysis_A3_bins = A3_bins

        self._D1_hist = D1_hist
        self.analysis_D1_hist = D1_hist
        self._D1_bins = D1_bins
        self.analysis_D1_bins = D1_bins

        self._D2_hist = D2_hist
        self.analysis_D2_hist = D2_hist
        self._D2_bins = D2_bins
        self.analysis_D2_bins = D2_bins

        self._D3_hist = D3_hist
        self.analysis_D3_hist = D3_hist
        self._D3_bins = D3_bins
        self.analysis_D3_bins = D3_bins

        self._D4_hist = D4_hist
        self.analysis_D4_hist = D4_hist
        self._D4_bins = D4_bins
        self.analysis_D4_bins = D4_bins

        self.analysis_shape_file = shape_file
        self.analysis_name = name
        self.analysis_class_b = class_b
    
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
            filepath=filepath,
            surface_area=row.get('surface_area'),
            compactness=row.get('compactness'),
            rectangularity=row.get('rectangularity'),
            diameter=row.get('diameter'),
            convexity=row.get('convexity'),
            eccentricity=row.get('eccentricity'),
            A3_hist=row.get('A3_hist'),
            A3_bins=row.get('A3_bins'),
            D1_hist=row.get('D1_hist'),
            D1_bins=row.get('D1_bins'),
            D2_hist=row.get('D2_hist'),
            D2_bins=row.get('D2_bins'),
            D3_hist=row.get('D3_hist'),
            D3_bins=row.get('D3_bins'),
            D4_hist=row.get('D4_hist'),
            D4_bins=row.get('D4_bins'),
            shape_file=row.get('shape_file'),
            name=row.get('name'),
            class_b=row.get('class_b')
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
    def from_file_row(cls, row, obj_dir=None, use_normalized=False, dataset=None):
        """Create ShapeMesh from DataFrame row, optionally using pre-normalized version"""
        
        # If requesting normalized version and it's available, use that
        if use_normalized and dataset:
            from core.normalized_cache import normalized_cache
            if normalized_cache.is_normalized_available(row['filename'], dataset):
                normalized_mesh = normalized_cache.load_normalized_shape(row['filename'], dataset)
                if normalized_mesh:
                    # Update metadata from original row
                    normalized_mesh.category = row.get('category', normalized_mesh.category)
                    return normalized_mesh
        
        # Fall back to original implementation
        if obj_dir is None:
            filepath = row['filepath']
        else:
            filepath = os.path.join(obj_dir, row['filename'])
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"OBJ file not found: {filepath}")
        
        vertices, faces = OBJParser.parse_obj_file(filepath)
        
        return cls(
            vertices=vertices,
            faces=faces,
            category=row.get('category'),
            filename=row.get('filename'),
            filepath=filepath,
            size=row.get('size'),
            surface_area=row.get('surface_area'),
            compactness=row.get('compactness'),
            rectangularity=row.get('rectangularity'),
            diameter=row.get('diameter'),
            convexity=row.get('convexity'),
            eccentricity=row.get('eccentricity'),
            A3_hist=row.get('A3_hist'),
            A3_bins=row.get('A3_bins'),
            D1_hist=row.get('D1_hist'),
            D1_bins=row.get('D1_bins'),
            D2_hist=row.get('D2_hist'),
            D2_bins=row.get('D2_bins'),
            D3_hist=row.get('D3_hist'),
            D3_bins=row.get('D3_bins'),
            D4_hist=row.get('D4_hist'),
            D4_bins=row.get('D4_bins'),
            shape_file=row.get('shape_file'),
            name=row.get('name'),
            class_b=row.get('class_b')
        )

    @property
    def num_vertices(self):
        return len(self.vertices)

    @property
    def num_faces(self):
        return len(self.faces)
    @property
    def area(self):
        # Prefer analysis-provided value when present
        if getattr(self, '_surface_area', None) is not None:
            try:
                return float(self._surface_area)
            except Exception:
                return self._surface_area

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
        # Use analysis-provided diameter if available
        if getattr(self, '_diameter', None) is not None:
            try:
                return float(self._diameter)
            except Exception:
                return self._diameter

        if len(self.vertices) < 2:
            return 0
        from scipy.spatial.distance import pdist
        return pdist(self.vertices).max()

    @property
    def compactness(self):
        # Prefer analysis-provided compactness
        if getattr(self, '_compactness', None) is not None:
            try:
                return float(self._compactness)
            except Exception:
                return self._compactness

        if self.volume == 0 or self.area is None:
            return 0
        return (self.area ** 1.5) / self.volume

    @property
    def rectangularity(self):
        if getattr(self, '_rectangularity', None) is not None:
            try:
                return float(self._rectangularity)
            except Exception:
                return self._rectangularity

        if self.base_mesh and hasattr(self.base_mesh, 'bounding_box_oriented'):
            obb = self.base_mesh.bounding_box_oriented
            obb_volume = obb.volume
            if obb_volume == 0:
                return 0
            return self.volume / obb_volume
        return None

    @property
    def convexity(self):
        if getattr(self, '_convexity', None) is not None:
            try:
                return float(self._convexity)
            except Exception:
                return self._convexity

        if self.base_mesh and hasattr(self.base_mesh, 'convex_hull'):
            hull = self.base_mesh.convex_hull
            hull_volume = hull.volume
            if hull_volume == 0:
                return 0
            return self.volume / hull_volume
        return None

    @property
    def eccentricity(self):
        if getattr(self, '_eccentricity', None) is not None:
            try:
                return float(self._eccentricity)
            except Exception:
                return self._eccentricity

        cov = np.cov(self.vertices.T)
        eigvals = np.linalg.eigvalsh(cov)
        if np.min(eigvals) == 0:
            return 0
        return np.max(eigvals) / np.min(eigvals)

    # Exactly the properties of the OBB_aligned bounding box: width, height, depth
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
    
    def analyze_object_orientation(self):
        """
        Analyze the object's natural orientation based on its geometry.
        Returns information about which axis should be "up" for proper viewing.
        """
        if len(self.vertices) == 0:
            return {"up_axis": "z", "confidence": 0.0, "reasoning": "No vertices"}
        
        # Get bounding box dimensions
        dims = self.dimensions
        if np.sum(dims) == 0:
            return {"up_axis": "z", "confidence": 0.0, "reasoning": "Zero dimensions"}
        
        # Find the dominant axis (usually height for most objects)
        dominant_axis_idx = np.argmax(dims)
        axis_names = ["x", "y", "z"]
        dominant_axis = axis_names[dominant_axis_idx]
        
        # Calculate confidence based on how much taller the dominant axis is
        sorted_dims = np.sort(dims)
        if sorted_dims[1] > 0:
            aspect_ratio = sorted_dims[2] / sorted_dims[1]  # tallest / second tallest
        else:
            aspect_ratio = 1.0
        
        confidence = min(1.0, (aspect_ratio - 1.0) / 2.0)  # Scale 0-1 based on aspect ratio
        
        # Category-specific heuristics
        reasoning = f"Dominant axis ({dominant_axis}) is {aspect_ratio:.2f}x larger"
        
        if self.category:
            category_lower = self.category.lower()
            
            # Objects that should be tall (Z-up)
            if any(keyword in category_lower for keyword in 
                   ['tree', 'building', 'skyscraper', 'tower', 'lamp', 'bottle', 'vase', 'rocket', 'humanoid', 'human']):
                if dominant_axis != 'z':
                    # Suggest rotation if object isn't already Z-up
                    reasoning += f" (Category '{self.category}' suggests Z-up orientation)"
                    return {"up_axis": "z", "confidence": 0.8, "reasoning": reasoning, "needs_rotation": True}
                else:
                    reasoning += f" (Category '{self.category}' confirms Z-up)"
                    confidence = max(confidence, 0.8)
            
            # Objects that are typically horizontal
            elif any(keyword in category_lower for keyword in 
                     ['car', 'vehicle', 'truck', 'bus', 'plane', 'aircraft', 'ship', 'boat', 'table', 'bed']):
                if dominant_axis == 'z':
                    # These objects might be lying on their side
                    reasoning += f" (Category '{self.category}' suggests Y-up orientation)"
                    return {"up_axis": "y", "confidence": 0.7, "reasoning": reasoning, "needs_rotation": True}
            
            # Animals/creatures (usually Y-up or Z-up depending on pose)
            elif any(keyword in category_lower for keyword in 
                     ['animal', 'bird', 'fish', 'quadruped', 'insect']):
                reasoning += f" (Category '{self.category}' - natural pose)"
                confidence = max(confidence, 0.6)
        
        return {
            "up_axis": dominant_axis, 
            "confidence": confidence, 
            "reasoning": reasoning,
            "needs_rotation": False
        }
    
    def get_optimal_orientation(self):
        """
        Get the optimal orientation by rotating the object if needed.
        Returns rotated vertices and the rotation matrix applied.
        """
        if len(self.vertices) == 0:
            return self.vertices.copy(), np.eye(3), {"needs_rotation": False}
        
        orientation = self.analyze_object_orientation()
        
        # TEMPORARILY DISABLE VERTEX ROTATION - TEST CAMERA APPROACH ONLY
        return self.vertices.copy(), np.eye(3), orientation
        
        if not orientation.get("needs_rotation", False):
            # Object is already properly oriented
            return self.vertices.copy(), np.eye(3), orientation
        
        # Get current dimensions to determine rotation needed
        dims = self.dimensions
        # Find which axis is currently dominant (tallest)
        dominant_axis_idx = np.argmax(dims)
        axis_names = ["x", "y", "z"]
        current_up = axis_names[dominant_axis_idx]
        desired_up = orientation["up_axis"]  # This is what we want (usually "z")
        
        # Create rotation matrix
        rotation_matrix = np.eye(3)
        
        if current_up == "y" and desired_up == "z":
            # Rotate around X-axis by 90 degrees to make Y->Z
            angle = np.pi / 2
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)]
            ])
        elif current_up == "x" and desired_up == "z":
            # Rotate around Y-axis by 90 degrees to make X->Z
            angle = np.pi / 2
            rotation_matrix = np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ])
        
        # Apply rotation to vertices
        center = self.vertices.mean(axis=0)
        centered_vertices = self.vertices - center
        rotated_vertices = centered_vertices @ rotation_matrix.T
        final_vertices = rotated_vertices + center
        
        # Debug logging
        if not np.allclose(rotation_matrix, np.eye(3)):
            print(f"[DEBUG] Rotating {current_up}-up to {desired_up}-up")
            print(f"[DEBUG] Original dims: x={dims[0]:.3f}, y={dims[1]:.3f}, z={dims[2]:.3f}")
            new_dims = np.ptp(final_vertices, axis=0)
            print(f"[DEBUG] Rotated dims: x={new_dims[0]:.3f}, y={new_dims[1]:.3f}, z={new_dims[2]:.3f}")
            
            # Check vertex ranges before and after
            orig_min, orig_max = self.vertices.min(axis=0), self.vertices.max(axis=0)
            new_min, new_max = final_vertices.min(axis=0), final_vertices.max(axis=0)
            print(f"[DEBUG] Original Z range: {orig_min[2]:.3f} to {orig_max[2]:.3f}")
            print(f"[DEBUG] Rotated Z range: {new_min[2]:.3f} to {new_max[2]:.3f}")
        
        return final_vertices, rotation_matrix, orientation

    def get_optimal_camera_position(self, distance_factor=2.0, rotated_vertices=None):
        """
        Calculate optimal camera position for viewing this object.
        
        Args:
            distance_factor: Multiplier for camera distance from object
            rotated_vertices: Optional pre-rotated vertices to use for positioning
            
        Returns:
            dict: Camera configuration for Plotly
        """
        vertices_to_use = rotated_vertices if rotated_vertices is not None else self.vertices
        
        if len(vertices_to_use) == 0:
            return {
                "eye": {"x": 1.5, "y": 1.5, "z": 1.5},
                "center": {"x": 0, "y": 0, "z": 0},
                "up": {"x": 0, "y": 0, "z": 1}
            }
        
        # Get object center and dimensions from potentially rotated vertices
        center = vertices_to_use.mean(axis=0)
        min_coords = vertices_to_use.min(axis=0)
        max_coords = vertices_to_use.max(axis=0)
        dims = max_coords - min_coords
        max_dim = np.max(dims)
        
        # Set camera distance based on object size
        distance = max_dim * distance_factor
        
        # Determine the dominant axis for camera up vector
        dominant_axis_idx = np.argmax(dims)
        
        # Set camera up vector to match object's natural orientation
        if dominant_axis_idx == 0:  # X is dominant
            up_vector = {"x": 1, "y": 0, "z": 0}
        elif dominant_axis_idx == 1:  # Y is dominant  
            up_vector = {"x": 0, "y": 1, "z": 0}
        else:  # Z is dominant
            up_vector = {"x": 0, "y": 0, "z": 1}
        
        # Adjust camera position based on the up vector
        if dominant_axis_idx == 1:  # Y-up objects
            eye = {
                "x": center[0] + distance * 0.8,
                "y": center[1],  # Keep Y center
                "z": center[2] + distance * 0.8
            }
        elif dominant_axis_idx == 0:  # X-up objects
            eye = {
                "x": center[0],  # Keep X center
                "y": center[1] + distance * 0.8,
                "z": center[2] + distance * 0.8
            }
        else:  # Z-up objects (standard)
            eye = {
                "x": center[0] + distance * 0.8,
                "y": center[1] + distance * 0.8, 
                "z": center[2] + distance * 0.6
            }
        
        camera_config = {
            "eye": eye,
            "center": {"x": center[0], "y": center[1], "z": center[2]},
            "up": up_vector
        }
        
        # Debug output
        axis_names = ["X", "Y", "Z"]
        print(f"[DEBUG] Object dominant axis: {axis_names[dominant_axis_idx]}, camera up: {up_vector}")
        
        return camera_config
    
    def get_card_header_html(self):
        """
        Generate the header part of the shape card with metadata for Dash HTML.
        This replaces the get_card_header function from callbacks.py
        
        Returns:
        html.Div - Dash HTML Div component with formatted metadata
        """
        dims = self.dimensions
        return html.Div([
            # Filename section with prominent display
            html.Div([
                html.Span("📄 ", className="shape-info-icon"),
                html.Strong("File: ", style={'color': '#27ae60'}),
                html.Span(self.filename or "Unknown", style={'color': '#27ae60', 'fontWeight': 'bold'})
            ], style={'width': '100%', 'marginBottom': '6px', 'borderBottom': '1px solid #27ae60', 'paddingBottom': '3px'}),

            # Other metadata in flexible layout
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
                html.Span(_num(len(self.vertices)), id='shape-vertices')
            ], className="shape-info-prop"),
            html.Div([
                html.Span("🔷 ", className="shape-info-icon"), html.Strong("Faces: "),
                html.Span(_num(len(self.faces)), id='shape-faces')
            ], className="shape-info-prop"),
            html.Div([
                html.Span("📐 ", className="shape-info-icon"), html.Strong("Dims: "),
                html.Span(f"X {dims[0]:.2f} · Y {dims[1]:.2f} · Z {dims[2]:.2f}")
            ], className="shape-info-prop"),


            # --- NEW ROW: Elementary Descriptors (all in one row) ---
            html.Div([
                html.Div([
                    html.Span("🌐 ", className="shape-info-icon"), html.Strong("Surface Area: "),
                    html.Span(f"{self.area:.4f}" if self.area is not None else "-")
                ], className="shape-info-prop shape-info-descriptor"),
                html.Div([
                    html.Span("⚪ ", className="shape-info-icon"), html.Strong("Compactness: "),
                    html.Span(f"{self.compactness:.4f}" if self.compactness is not None else "-")
                ], className="shape-info-prop shape-info-descriptor"),
                html.Div([
                    html.Span("📦 ", className="shape-info-icon"), html.Strong("Rectangularity: "),
                    html.Span(f"{self.rectangularity:.4f}" if self.rectangularity is not None else "-")
                ], className="shape-info-prop shape-info-descriptor"),
                html.Div([
                    html.Span("📏 ", className="shape-info-icon"), html.Strong("Diameter: "),
                    html.Span(f"{self.diameter:.4f}" if self.diameter is not None else "-")
                ], className="shape-info-prop shape-info-descriptor"),
                html.Div([
                    html.Span("⛰️ ", className="shape-info-icon"), html.Strong("Convexity: "),
                    html.Span(f"{self.convexity:.4f}" if self.convexity is not None else "-")
                ], className="shape-info-prop shape-info-descriptor"),
                html.Div([
                    html.Span("🧮 ", className="shape-info-icon"), html.Strong("Eccentricity: "),
                    html.Span(f"{self.eccentricity:.4f}" if self.eccentricity is not None else "-")
                ], className="shape-info-prop shape-info-descriptor"),
            ], style={'display': 'flex', 'flexDirection': 'row', 'gap': '3px', 'margin': '8px 0'}),
            # --- END NEW ROW ---

            html.Div([
                html.Span("🎯 ", className="shape-info-icon"), html.Strong("Quality: "),
                html.Span(self.quality)
            ], className="shape-info-prop"),
            html.Div([
                html.Span("📹 ", className="shape-info-icon"), html.Strong("View: "),
                html.Span(self._get_orientation_display())
            ], className="shape-info-prop"),
        ], className="shape-info-header")
    
    def _get_orientation_display(self):
        """Get a short display string for the object's orientation."""
        try:
            orientation = self.analyze_object_orientation()
            up_axis = orientation["up_axis"].upper()
            confidence = orientation["confidence"]
            
            if confidence > 0.7:
                return f"{up_axis}-up (Auto)"
            elif confidence > 0.4:
                return f"{up_axis}-up (Est.)"
            else:
                return "Standard view"
        except Exception:
            return "Standard view"
    
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
    
    def apply_full_normalization(self, debug=False):
        """Apply the complete enhanced 4-step normalization pipeline:
        
        Following the exact order from technical tips with enhanced implementations:
        1. Translation (enhanced centering) - area-weighted barycenter with two-pass recentering
        2. Alignment (pose) - align principal axes with coordinate frame using PCA
        3. Flipping - orient shape consistently using moment test
        4. Scaling (enhanced) - fit in unit bounding box with post-scaling recenter
        
        Enhanced features:
        - Area-weighted barycenter calculation with degeneracy fallback
        - Two-pass recentering for numerical robustness
        - Post-scaling recenter safety pass
        - Numerical tolerances (AREA_EPS, RECENTER_EPS)
        
        Args:
            debug: If True, print intermediate results for verification
        
        Returns:
        - normalized_vertices: np.array, the fully normalized vertices
        """
        vertices = self.vertices.copy()
        
        if debug:
            print(f"Original: center={np.mean(vertices, axis=0):.3f}, max_dim={np.max(np.ptp(vertices, axis=0)):.3f}")
        
        # Step 1: Translation (centering) - Following technical tips order
        vertices = self._apply_centering(vertices)
        if debug:
            print(f"After centering: center={np.mean(vertices, axis=0):.6f}")
        
        # Step 2: Alignment (pose) using PCA - Following technical tips order
        vertices = self._apply_pca_alignment(vertices)
        if debug:
            center = np.mean(vertices, axis=0)
            dims = np.ptp(vertices, axis=0)
            print(f"After alignment: center={center:.6f}, dims={dims:.3f}")
        
        # Step 3: Flipping using moment test - Following technical tips order
        vertices = self._apply_flipping(vertices)
        if debug:
            center = np.mean(vertices, axis=0)
            dims = np.ptp(vertices, axis=0)
            print(f"After flipping: center={center:.6f}, dims={dims:.3f}")
        
        # Step 4: Scaling to unit bounding box - Following technical tips order
        vertices = self._apply_scaling(vertices)
        if debug:
            center = np.mean(vertices, axis=0)
            max_dim = np.max(np.ptp(vertices, axis=0))
            print(f"After scaling: center={center:.6f}, max_dim={max_dim:.6f}")
        
        return vertices
    
    def _apply_centering(self, vertices):
        """Enhanced centering with two-pass recentering from normalization.py
        
        Center the shape at the origin using area-weighted barycenter with fallback to vertex mean.
        Includes two-pass recentering for numerical robustness.
        """
        # Convert faces to triangles for area calculations
        triangles = []
        for face in self.faces:
            if len(face) >= 3:
                triangles.append(face[:3])  # Use first 3 vertices for triangulation
        triangles = np.array(triangles) if triangles else np.array([])
        
        # First translation pass using area-weighted barycenter
        bary_before = calculate_mass_barycenter(vertices, triangles)
        vertices_translated = vertices - bary_before
        
        # Recompute barycenter after translation (should be near zero)
        residual_bary = calculate_mass_barycenter(vertices_translated, triangles)
        residual_norm = float(np.linalg.norm(residual_bary))
        
        # Optional second pass for numerical drift or degeneracy effects
        if residual_norm > RECENTER_EPS:
            vertices_translated -= residual_bary
        
        return vertices_translated
    
    def _apply_pca_alignment(self, vertices):
        """Align shape using PCA eigenvectors as described in technical tips.
        
        The update formula from the tips:
        x_updated = (p_i - c) · e1
        y_updated = (p_i - c) · e2  
        z_updated = (p_i - c) · (e1 × e2)
        
        Where e1, e2 are the major and medium eigenvectors.
        Note: vertices should already be centered when this method is called.
        """
        # Compute covariance matrix and eigenvectors from centered vertices
        pca = PCA(n_components=3)
        pca.fit(vertices)
        
        # Get eigenvectors (already sorted by eigenvalue magnitude)
        e1 = pca.components_[0]  # Major eigenvector (largest eigenvalue)
        e2 = pca.components_[1]  # Medium eigenvector 
        e3 = np.cross(e1, e2)   # Minor eigenvector (computed as cross product)
        
        # Normalize eigenvectors to unit length
        e1 = e1 / np.linalg.norm(e1)
        e2 = e2 / np.linalg.norm(e2) 
        e3 = e3 / np.linalg.norm(e3)
        
        # Apply alignment transformation using dot products
        # Note: since vertices are already centered, we use them directly
        aligned_vertices = np.zeros_like(vertices)
        aligned_vertices[:, 0] = np.dot(vertices, e1)  # x = (p_i - c) · e1
        aligned_vertices[:, 1] = np.dot(vertices, e2)  # y = (p_i - c) · e2  
        aligned_vertices[:, 2] = np.dot(vertices, e3)  # z = (p_i - c) · (e1 × e2)
        
        return aligned_vertices
    
    def _apply_flipping(self, vertices):
        """Apply flipping test using moment test as described in technical tips.
        
        For each axis, compute f_i = Σ sign(C_t,i) * (C_t,i)^2
        where C_t,i is the i-th coordinate of triangle center t.
        
        Then flip along axis i if f_i < 0.
        """
        if len(self.faces) == 0:
            return vertices
            
        # Compute triangle centers
        triangle_centers = []
        for face in self.faces:
            if len(face) >= 3:  # Valid triangle/polygon
                face_vertices = vertices[face[:3]]  # Use first 3 vertices for triangulation
                center = np.mean(face_vertices, axis=0)
                triangle_centers.append(center)
        
        if len(triangle_centers) == 0:
            return vertices
            
        triangle_centers = np.array(triangle_centers)
        
        # Compute flipping test values for each axis
        f = np.zeros(3)
        for i in range(3):  # For x, y, z axes
            coords = triangle_centers[:, i]
            f[i] = np.sum(np.sign(coords) * (coords ** 2))
        
        # Apply flipping using the exact formula from technical tips:
        # xiupdated = xi * sign(f0), yiupdated = yi * sign(f1), ziupdated = zi * sign(f2)
        flip_factors = np.sign(f)
        flip_factors[flip_factors == 0] = 1  # Avoid flipping if f[i] = 0
        
        # Apply scaling factors (mirroring = scaling by -1 when sign is negative)
        flipped_vertices = vertices * flip_factors
        
        return flipped_vertices
    
    def _apply_scaling(self, vertices):
        """Enhanced scaling with post-scaling recenter safety pass from normalization.py
        
        Scale shape to fit in unit bounding box with post-scaling recenter to ensure
        the final mesh barycenter is at the origin.
        """
        # Standard scaling
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        dimensions = max_coords - min_coords
        
        # Find maximum dimension
        max_dimension = np.max(dimensions)
        
        if max_dimension > 0:
            # Scale to unit size
            scale_factor = 1.0 / max_dimension
            vertices_scaled = vertices * scale_factor
        else:
            vertices_scaled = vertices
        
        # Post-scaling recenter safety pass
        # Convert faces to triangles for area calculations
        triangles = []
        for face in self.faces:
            if len(face) >= 3:
                triangles.append(face[:3])  # Use first 3 vertices for triangulation
        triangles = np.array(triangles) if triangles else np.array([])
        
        final_bary = calculate_mass_barycenter(vertices_scaled, triangles)
        final_bary_norm = np.linalg.norm(final_bary)
        
        if final_bary_norm > RECENTER_EPS:
            vertices_scaled -= final_bary
            
        return vertices_scaled
    
    def get_normalization_info(self):
        """Get detailed information about the normalization process"""
        vertices = self.vertices.copy()
        info = {}
        
        # Original shape info
        original_center = np.mean(vertices, axis=0)
        original_bbox = {
            'min': np.min(vertices, axis=0),
            'max': np.max(vertices, axis=0),
            'dimensions': np.max(vertices, axis=0) - np.min(vertices, axis=0)
        }
        info['original'] = {
            'center': original_center,
            'bounding_box': original_bbox
        }
        
        # Step 1: Centering
        centered_vertices = self._apply_centering(vertices)
        info['after_centering'] = {
            'center': np.mean(centered_vertices, axis=0)
        }
        
        # Step 2: PCA Alignment
        aligned_vertices = self._apply_pca_alignment(centered_vertices)
        
        # Compute PCA info
        pca = PCA(n_components=3)
        pca.fit(centered_vertices)
        info['pca'] = {
            'eigenvalues': pca.explained_variance_,
            'eigenvectors': pca.components_,
            'explained_variance_ratio': pca.explained_variance_ratio_
        }
        
        # Step 3: Flipping
        flipped_vertices = self._apply_flipping(aligned_vertices)
        
        # Compute flipping test values
        if len(self.faces) > 0:
            triangle_centers = []
            for face in self.faces:
                if len(face) >= 3:
                    face_vertices = aligned_vertices[face[:3]]
                    center = np.mean(face_vertices, axis=0)
                    triangle_centers.append(center)
            
            if len(triangle_centers) > 0:
                triangle_centers = np.array(triangle_centers)
                f = np.zeros(3)
                for i in range(3):
                    coords = triangle_centers[:, i]
                    f[i] = np.sum(np.sign(coords) * (coords ** 2))
                
                info['flipping'] = {
                    'moment_test_values': f,
                    'flip_factors': np.sign(f)
                }
        
        # Step 4: Final scaling
        final_vertices = self._apply_scaling(flipped_vertices)
        final_bbox = {
            'min': np.min(final_vertices, axis=0),
            'max': np.max(final_vertices, axis=0), 
            'dimensions': np.max(final_vertices, axis=0) - np.min(final_vertices, axis=0)
        }
        info['final'] = {
            'center': np.mean(final_vertices, axis=0),
            'bounding_box': final_bbox,
            'max_dimension': np.max(final_bbox['dimensions'])
        }
        
        return info

    def get_normalized_mesh(self):
        """Create a new ShapeMesh with fully normalized vertices"""
        normalized_vertices = self.apply_full_normalization()
        
        return ShapeMesh(
            vertices=normalized_vertices,
            faces=self.faces,
            category=self.category,
            filename=self.filename,
            face_types=self.face_types,
            bounding_box=None,  # Will be recomputed
            size=self.size,
            filepath=self.filepath
        )

    def get_normalized_vertices_cached(self, dataset=None):
        """Get normalized vertices efficiently - use cache if available, compute otherwise"""
        
        # Try to load from cache first
        if dataset:
            from core.normalized_cache import normalized_cache
            if normalized_cache.is_normalized_available(self.filename, dataset):
                normalized_mesh = normalized_cache.load_normalized_shape(self.filename, dataset)
                if normalized_mesh:
                    return normalized_mesh.vertices
        
        # Fall back to computing normalization
        return self.apply_full_normalization()

    def save_as_obj(self, filepath: str):
        """
        Save this ShapeMesh to a Wavefront OBJ file.

        Args:
            filepath (str): Destination file path (should end with .obj)
        """
        with open(filepath, "w") as f:
            # Write vertices
            for v in self.vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")

            # Write faces (OBJ is 1-based indexing)
            for face in self.faces:
                f.write("f " + " ".join(str(idx + 1) for idx in face) + "\n")

    
