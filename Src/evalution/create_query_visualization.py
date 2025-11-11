"""Create visual query result grids for high, medium, and low F1-score classes.

This script generates a figure showing retrieval results for representative queries
from three performance tiers (high, medium, low F1-score classes) across all three
descriptor aggregation approaches.

Usage (from project root):
    python -m Src.evalution.create_query_visualization
"""
import os
import re
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List, Dict, Tuple, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from evaluation script
try:
    from Src.evalution.evalution import normalize_id, load_analysis_labels
except ImportError:
    print("Warning: Could not import from evalution.py, defining functions locally")
    
    def normalize_id(name: str) -> str:
        """Normalize a filename to a short id."""
        if pd.isna(name):
            return ""
        s = os.path.basename(str(name)).lower()
        if s.endswith('.obj'):
            s = s[:-4]
        m = re.match(r"^([a-z0-9]+)", s)
        if m:
            return m.group(1)
        return s
    
    def load_analysis_labels(path: str) -> pd.DataFrame:
        """Load class labels from analysis file."""
        df = pd.read_csv(path)
        filename_col = None
        for c in ['shape_file', 'filename', 'name', 'shape', 'file']:
            if c in df.columns:
                filename_col = c
                break
        if filename_col is None:
            raise ValueError("No filename column found")
        df = df.rename(columns={filename_col: 'filename'})
        if 'class' not in df.columns:
            raise ValueError("No class column found")
        df['id'] = df['filename'].apply(normalize_id)
        return df[['filename', 'id', 'class']]


# Import category colors from viewer
try:
    # Go up one level from evalution to project root, then import
    viewer_path = Path(__file__).parent.parent / "obj-viewer-app" / "viewer"
    sys.path.insert(0, str(viewer_path))
    from category_colors import CATEGORY_COLOR_MAP
except ImportError:
    print("Warning: Could not import category colors, using default colors")
    CATEGORY_COLOR_MAP = {}

# Import rotation configuration
try:
    from Src.evalution.shape_rotation_config import SHAPE_ROTATIONS, CLASS_ROTATIONS, DEFAULT_ROTATION
except ImportError:
    print("Warning: Could not import rotation config, using defaults")
    # SHAPE_ROTATIONS = {
    #     'd00131': {'rot_x': 90, 'rot_y': 30, 'rot_z': -10, 'elev': 15, 'azim': 60},
    #     'm355': {'rot_x': 85, 'rot_y': 15, 'rot_z': 5, 'elev': 20, 'azim': 45},
    #     'm365': {'rot_x': 85, 'rot_y': 15, 'rot_z': 5, 'elev': 20, 'azim': 45},
    #     'D00072': {'rot_x': 85, 'rot_y': 15, 'rot_z': 5, 'elev': 20, 'azim': 45},
    #     'm526': {'rot_x': 85, 'rot_y': 15, 'rot_z': 5, 'elev': 20, 'azim': 45},
    #     'd00340': {'rot_x': 85, 'rot_y': 15, 'rot_z': 5, 'elev': 20, 'azim': 45},
    #     'd00358': {'rot_x': 85, 'rot_y': 15, 'rot_z': 5, 'elev': 20, 'azim': 45},
    #     'm472': {'rot_x': 85, 'rot_y': 15, 'rot_z': 5, 'elev': 20, 'azim': 45},
    #     'D00960': {'rot_x': 85, 'rot_y': 15, 'rot_z': 5, 'elev': 20, 'azim': 45},
    #     'm168': {'rot_x': 85, 'rot_y': 15, 'rot_z': 5, 'elev': 20, 'azim': 45},
    #     'm189': {'rot_x': 85, 'rot_y': 15, 'rot_z': 5, 'elev': 20, 'azim': 45},
    #     'm00400': {'rot_x': 85, 'rot_y': 15, 'rot_z': 5, 'elev': 20, 'azim': 45},
    #     'D00054': {'rot_x': 85, 'rot_y': 15, 'rot_z': 5, 'elev': 20, 'azim': 45},
    #     'm168': {'rot_x': 85, 'rot_y': 15, 'rot_z': 5, 'elev': 20, 'azim': 45},
    #     'm189': {'rot_x': 85, 'rot_y': 15, 'rot_z': 5, 'elev': 20, 'azim': 45},
    #     'D00616': {'rot_x': 85, 'rot_y': 15, 'rot_z': 5, 'elev': 20, 'azim': 45},
    # }
    SHAPE_ROTATIONS = {
        # Human heads - rotate to face camera
        'd00131': {'rot_x': 90, 'rot_y': 210, 'rot_z': 0, 'elev': 20, 'azim': 45},
        'm355': {'rot_x': 90, 'rot_y': 195, 'rot_z': 0, 'elev': 20, 'azim': 45},
        'm365': {'rot_x': 90, 'rot_y': 200, 'rot_z': 0, 'elev': 20, 'azim': 45},
        'D00072': {'rot_x': 90, 'rot_y': 195, 'rot_z': 0, 'elev': 20, 'azim': 45},
        'D00313': {'rot_x': 90, 'rot_y': 200, 'rot_z': 0, 'elev': 20, 'azim': 45},
        'D00487': {'rot_x': 90, 'rot_y': 200, 'rot_z': 0, 'elev': 20, 'azim': 45},
        'D00072': {'rot_x': 90, 'rot_y': 195, 'rot_z': 0, 'elev': 20, 'azim': 45},
        
        # Cup - needs to stand upright
        'm526': {'rot_x': 90, 'rot_y': 15, 'rot_z': 0, 'elev': 20, 'azim': 45},
        
        # Vase (m527) - similar to cup
        'm527': {'rot_x': 90, 'rot_y': 15, 'rot_z': 0, 'elev': 25, 'azim': 45},
    }
    CLASS_ROTATIONS = {}
    DEFAULT_ROTATION = {'rot_x': 90, 'rot_y': 15, 'rot_z': 0, 'elev': 20, 'azim': 45}


# Default paths
DEFAULT_MATCHING = [
    "Src/matching/matrix_minmax_optimized.csv",
    "Src/matching/matrix_rank_based_optimized.csv",
    "Src/matching/matrix_weighted_sum.csv",
]

APPROACH_NAMES = {
    "matrix_minmax_optimized.csv": "MinMax Normalization",
    "matrix_rank_based_optimized.csv": "Rank-Based Transformation",
    "matrix_weighted_sum.csv": "Standardization (Z-score)",
}

# Map between CSV filename and approach key in F1 scores
APPROACH_KEY_MAP = {
    "matrix_minmax_optimized.csv": "matrix_minmax_optimized",
    "matrix_rank_based_optimized.csv": "matrix_rank_based_optimized",
    "matrix_weighted_sum.csv": "matrix_weighted_sum",
}

DEFAULT_ANALYSIS = "Datasets/UnifiedPreprocessed/Data/analysis_results_unifiedPreprocessed_data.csv"
DEFAULT_F1_SCORES = "Src/evalution/figures/combined_per_class_summary.csv"
DEFAULT_MESH_DIR = "Datasets/UnifiedPreprocessed/Data"
DEFAULT_OUTPUT_DIR = "Reports/Step6"

NUMBER_OF_SIMILAR_SHAPES = 5


def load_distance_matrix(path: str) -> pd.DataFrame:
    """Load a distance matrix CSV into a DataFrame."""
    df = pd.read_csv(path, index_col=0)
    return df


def get_class_f1_scores(f1_path: str) -> Dict[str, Dict[str, float]]:
    """Load F1 scores per class and approach.
    
    Returns dict: {approach_name: {class_name: f1_score}}
    """
    df = pd.read_csv(f1_path)
    
    # Filter to only F1 scores
    df_f1 = df[['approach', 'class', 'F1']].copy()
    
    # Group by approach
    result = {}
    for approach in df_f1['approach'].unique():
        approach_data = df_f1[df_f1['approach'] == approach]
        result[approach] = dict(zip(approach_data['class'], approach_data['F1']))
    
    return result


def select_representative_classes(f1_scores: Dict[str, float], 
                                   exclude_macro: bool = True) -> Tuple[List[str], List[str], List[str]]:
    """Select high, medium, and low F1-score classes.
    
    Returns (high_classes, medium_classes, low_classes) where each is a list of 3 class names
    """
    # Filter out macro averages if present
    classes = [(cls, score) for cls, score in f1_scores.items() 
               if not (exclude_macro and ('Macro' in cls or 'Overall' in cls))]
    
    # Sort by F1 score
    classes_sorted = sorted(classes, key=lambda x: x[1], reverse=True)
    
    if len(classes_sorted) < 9:
        raise ValueError("Not enough classes to select representatives")
    
    # Select 3 classes from high tier (top ranked)
    high_classes = [classes_sorted[0][0], classes_sorted[1][0], classes_sorted[2][0]]
    
    # Select 3 classes from medium tier (around middle)
    mid_idx = len(classes_sorted) // 2
    medium_classes = [
        classes_sorted[mid_idx - 1][0],
        classes_sorted[mid_idx][0],
        classes_sorted[mid_idx + 1][0]
    ]
    
    # Select 3 classes from low tier (bottom ranked)
    low_classes = [classes_sorted[-3][0], classes_sorted[-2][0], classes_sorted[-1][0]]
    
    return high_classes, medium_classes, low_classes


def retrieve_closest_shapes(query_id: str, 
                            distance_matrix: pd.DataFrame,
                            analysis_df: pd.DataFrame,
                            n: int = 10) -> List[Dict]:
    """Retrieve n closest shapes for a query ID.
    
    Returns list of dicts with keys: 'id', 'class', 'distance'
    """
    # Find matching row in distance matrix - try multiple patterns
    matching_rows = []
    
    # Try exact match with underscore suffix
    matching_rows = [idx for idx in distance_matrix.index if idx.startswith(query_id + "_")]
    
    # If not found, try without case sensitivity
    if not matching_rows:
        query_id_lower = query_id.lower()
        matching_rows = [idx for idx in distance_matrix.index 
                        if idx.lower().startswith(query_id_lower + "_")]
    
    # If still not found, try just the prefix without underscore requirement
    if not matching_rows:
        matching_rows = [idx for idx in distance_matrix.index 
                        if normalize_id(idx) == query_id.lower()]
    
    if not matching_rows:
        print(f"Warning: No matching row for ID {query_id}")
        print(f"   Available IDs sample: {list(distance_matrix.index[:5])}")
        return []
    
    row_name = matching_rows[0]
    
    # Get distance vector
    distances_series = distance_matrix.loc[row_name]
    
    # Remove self-match
    distances_series = distances_series[distances_series.index != row_name]
    
    # Sort and get top-k
    sorted_distances = distances_series.sort_values()
    top_k = sorted_distances.head(n)
    
    # Map to IDs and classes
    results = []
    for dist_idx, dist_val in top_k.items():
        # Extract ID from distance matrix column name
        m = re.match(r"([A-Za-z0-9]+)_", dist_idx)
        if not m:
            continue
        shape_id = m.group(1)
        
        # Look up class in analysis (ensure IDs are lowercase for matching)
        row = analysis_df[analysis_df['id'].str.lower() == shape_id.lower()]
        if row.empty:
            shape_class = "Unknown"
        else:
            shape_class = row.iloc[0]['class']
        
        results.append({
            'id': shape_id,
            'class': shape_class,
            'distance': float(dist_val)
        })
    
    return results


def get_best_query_shape_from_class(class_name: str, 
                                     analysis_df: pd.DataFrame,
                                     distance_matrix: pd.DataFrame,
                                     n: int = 10) -> Optional[str]:
    """Select the query shape with best retrieval accuracy from the specified class.
    
    Args:
        class_name: The class to select from
        analysis_df: DataFrame with shape IDs and classes
        distance_matrix: Distance matrix for retrievals
        n: Number of neighbors to consider for accuracy calculation
    
    Returns:
        ID of the shape with best retrieval accuracy in this class
    """
    class_shapes = analysis_df[analysis_df['class'] == class_name]
    if class_shapes.empty:
        return None
    
    best_shape_id = None
    best_accuracy = -1
    
    # Test each shape in the class
    for _, row in class_shapes.iterrows():
        shape_id = row['id']
        
        # Get retrieval results for this shape
        results = retrieve_closest_shapes(shape_id, distance_matrix, analysis_df, n=n)
        
        if not results:
            continue
        
        # Calculate accuracy (percentage of correct class in top-n)
        correct = sum(1 for r in results if r['class'] == class_name)
        accuracy = correct / len(results)
        
        # Track best
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_shape_id = shape_id
    
    return best_shape_id


def load_obj_file(obj_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load vertices and faces from an OBJ file.
    
    Returns (vertices, faces) where vertices is Nx3 and faces is Mx3 (triangle indices).
    Returns (None, None) if file not found or error.
    """
    if not os.path.exists(obj_path):
        return None, None
    
    try:
        vertices = []
        faces = []
        
        with open(obj_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('v '):
                    # Vertex: v x y z
                    parts = line.split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('f '):
                    # Face: f v1 v2 v3 (may have v/vt/vn format)
                    parts = line.split()[1:]
                    # Extract just vertex indices (before any /)
                    face_indices = []
                    for part in parts:
                        idx = int(part.split('/')[0]) - 1  # OBJ is 1-indexed
                        face_indices.append(idx)
                    
                    # Triangulate if needed (simple fan triangulation)
                    if len(face_indices) == 3:
                        faces.append(face_indices)
                    elif len(face_indices) > 3:
                        # Fan triangulation from first vertex
                        for i in range(1, len(face_indices) - 1):
                            faces.append([face_indices[0], face_indices[i], face_indices[i + 1]])
        
        if not vertices or not faces:
            return None, None
        
        return np.array(vertices), np.array(faces)
    
    except Exception as e:
        print(f"Error loading OBJ file {obj_path}: {e}")
        return None, None


def find_shape_obj_file(shape_id: str, class_name: str, data_dir: str) -> Optional[str]:
    """Find the OBJ file path for a given shape ID and class.
    
    Args:
        shape_id: The shape identifier (e.g., 'd00487')
        class_name: The class folder name (e.g., 'HumanHead')
        data_dir: Root directory containing class folders
    
    Returns:
        Full path to OBJ file if found, None otherwise
    """
    # Construct path: data_dir/class_name/shape_id*.obj
    class_dir = os.path.join(data_dir, class_name)
    
    if not os.path.exists(class_dir):
        print(f"  Warning: Class directory not found: {class_dir}")
        return None
    
    # Look for files matching the shape_id - prioritize certain suffixes
    shape_id_upper = shape_id.upper()
    shape_id_lower = shape_id.lower()
    
    # Priority order for file versions
    preferred_suffixes = ['_06_fill_holes_and_orientation.obj']
    
    # Try to find with preferred suffixes
    for suffix in preferred_suffixes:
        # Try with uppercase D
        for filename in os.listdir(class_dir):
            if filename.upper() == (shape_id_upper + suffix).upper():
                return os.path.join(class_dir, filename)
    
    # Fallback: find any file that starts with the shape_id
    for filename in os.listdir(class_dir):
        if filename.lower().endswith('.obj'):
            # Check if filename starts with the shape_id (case-insensitive)
            if filename.upper().startswith(shape_id_upper):
                return os.path.join(class_dir, filename)
            # Try with D prefix
            if not shape_id_upper.startswith('D') and filename.upper().startswith('D' + shape_id_upper):
                return os.path.join(class_dir, filename)
    
    print(f"  Warning: OBJ file not found for {shape_id} in {class_name}")
    available = [f for f in os.listdir(class_dir) if f.endswith('.obj')][:3]
    print(f"    Available files: {available}")
    return None


def get_class_color(class_name: str) -> Tuple[float, float, float]:
    """Get RGB color for a class name.
    
    Uses CATEGORY_COLOR_MAP if available, otherwise generates a consistent color.
    """
    if CATEGORY_COLOR_MAP and class_name in CATEGORY_COLOR_MAP:
        hex_color = CATEGORY_COLOR_MAP[class_name]
        # Convert hex to RGB (0-1 range)
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
        return (r, g, b)
    
    # Fallback: generate color from class name hash
    hash_val = hash(class_name)
    np.random.seed(hash_val % 2**31)
    return tuple(np.random.rand(3))


def get_rotation_config(shape_id: str, class_name: str) -> Dict[str, float]:
    """Get rotation configuration for a specific shape.
    
    Priority order:
    1. SHAPE_ROTATIONS (per-shape ID)
    2. CLASS_ROTATIONS (per-class)
    3. DEFAULT_ROTATION (fallback)
    
    Args:
        shape_id: The shape identifier (e.g., 'd00131', 'm355')
        class_name: The class name (e.g., 'HumanHead')
    
    Returns:
        Dictionary with rotation parameters
    """
    # Check for shape-specific configuration first
    if shape_id in SHAPE_ROTATIONS:
        return SHAPE_ROTATIONS[shape_id]
    
    # Fall back to class-level configuration
    if class_name in CLASS_ROTATIONS:
        return CLASS_ROTATIONS[class_name]
    
    # Use default
    return DEFAULT_ROTATION


def render_mesh_to_axis(ax: Axes3D, vertices: np.ndarray, faces: np.ndarray, 
                       color: Tuple[float, float, float],
                       shape_id: str,
                       class_name: str,
                       edge_color: Optional[str] = None,
                       edge_linewidth: float = 0.5):
    """Render a 3D mesh to a matplotlib 3D axis with improved appearance.
    
    Args:
        ax: Matplotlib 3D axis
        vertices: Nx3 array of vertex coordinates
        faces: Mx3 array of triangle indices
        color: RGB tuple (0-1 range) for face color
        shape_id: Shape identifier for rotation configuration
        class_name: Class name for rotation configuration
        edge_color: Optional edge color (e.g., 'green', 'red')
        edge_linewidth: Width of edges
    """
    # Get rotation configuration for this specific shape
    config = get_rotation_config(shape_id, class_name)
    
    # Apply rotations based on class-specific configuration
    # Rotation around X axis
    theta_x = np.radians(config['rot_x'])
    rot_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])
    
    # Rotation around Y axis
    theta_y = np.radians(config['rot_y'])
    rot_y = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])
    
    # Rotation around Z axis
    theta_z = np.radians(config['rot_z'])
    rot_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])
    
    # Apply all rotations
    vertices_rotated = vertices @ rot_x.T @ rot_y.T @ rot_z.T
    
    # Create mesh collection
    mesh = []
    for face in faces:
        triangle = vertices_rotated[face]
        mesh.append(triangle)
    
    # Lighten the color to make shapes brighter
    # Mix with white to brighten: new_color = color * 0.6 + white * 0.4
    brightened_color = tuple(c * 0.6 + 0.4 for c in color)
    
    # Create 3D polygon collection with improved appearance
    # Adjust opacity based on whether we have colored borders
    if edge_color is None:
        # For query shapes without special border
        poly = Poly3DCollection(mesh, 
                               alpha=0.95,
                               facecolors=brightened_color,
                               edgecolors='gray',
                               linewidths=0.1,
                               shade=True,
                               lightsource=None)  # Use default lighting
    else:
        # For result shapes with colored borders - more transparent to see borders
        poly = Poly3DCollection(mesh, 
                               alpha=0.9,
                               facecolors=brightened_color,
                               edgecolors=edge_color,
                               linewidths=edge_linewidth,
                               shade=True,
                               lightsource=None)  # Use default lighting
    
    ax.add_collection3d(poly)
    
    # Set axis limits based on mesh bounds with padding
    all_verts = vertices_rotated
    x_min, x_max = all_verts[:, 0].min(), all_verts[:, 0].max()
    y_min, y_max = all_verts[:, 1].min(), all_verts[:, 1].max()
    z_min, z_max = all_verts[:, 2].min(), all_verts[:, 2].max()
    
    # Calculate range and center
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    max_range = max(x_range, y_range, z_range) * 0.55  # Slightly larger for better view
    
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    z_mid = (z_max + z_min) / 2
    
    # Set equal aspect ratio limits
    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)
    
    # Clean up axis appearance - remove all ticks, labels, grid
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    
    # Make background white and clean
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_facecolor('white')
    ax.yaxis.pane.set_facecolor('white')
    ax.zaxis.pane.set_facecolor('white')
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    # Use class-specific camera angle
    ax.view_init(elev=config['elev'], azim=config['azim'])
    
    # Set distance to make shape more prominent
    ax.dist = 7
def create_query_grid(approach_results: Dict[str, Dict[str, List]], 
                      query_classes: Dict[str, str],
                      query_ids: Dict[str, str],
                      f1_scores: Dict[str, Dict[str, float]],
                      output_path: str,
                      rank_number: int,
                      tier_name: str,
                      data_dir: str = DEFAULT_MESH_DIR):
    """Create a grid visualization showing query results with 3D meshes for one rank and one tier.
    
    approach_results: {approach_name: {tier_label: [result_dicts]}}
    query_classes: {tier_label: class_name}
    query_ids: {tier_label: query_id}
    f1_scores: {approach_name: {class_name: f1_score}}
    output_path: path to save the figure
    rank_number: which rank (1, 2, or 3) this figure represents
    tier_name: which tier ('high', 'medium', or 'low')
    data_dir: path to directory containing class folders with OBJ files
    """
    # Setup: 3 approaches (rows) x 1 tier (column group)
    # Each row has 6 columns: 1 query + 5 results
    
    fig = plt.figure(figsize=(14, 12))
    
    # Create grid: 3 rows (approaches), 1 tier group, 6 cols per group
    n_rows = 3
    n_cols = 1 + NUMBER_OF_SIMILAR_SHAPES  # query + results
    
    # Use GridSpec for flexible layout
    gs = GridSpec(n_rows, n_cols,
                  figure=fig, hspace=0.25, wspace=0.05,
                  left=0.08, right=0.95, top=0.88, bottom=0.10)
    
    approach_names = list(approach_results.keys())
    # Get the specific tier label for this rank and tier
    tier_label = f'{tier_name}_{rank_number}'
    
    print(f"\n   Rendering 3D meshes for {tier_name.upper()} tier, Rank #{rank_number}...")
    
    for row_idx, approach in enumerate(approach_names):
        # No tier loop - just one tier per figure
        query_class = query_classes[tier_label]
        query_id = query_ids[tier_label]
        results = approach_results[approach].get(tier_label, [])
        
        # Query shape (first column)
        ax_query = fig.add_subplot(gs[row_idx, 0], projection='3d')
        
        # Load and render query mesh
        query_obj_path = find_shape_obj_file(query_id, query_class, data_dir)
        if query_obj_path:
            vertices, faces = load_obj_file(query_obj_path)
            if vertices is not None and faces is not None:
                query_color = get_class_color(query_class)
                render_mesh_to_axis(ax_query, vertices, faces, query_color, query_id, query_class)
            else:
                ax_query.text(0.5, 0.5, 0.5, 'Load\nError', ha='center', va='center', fontsize=8)
                ax_query.set_xlim(0, 1)
                ax_query.set_ylim(0, 1)
                ax_query.set_zlim(0, 1)
                ax_query.axis('off')
        else:
            ax_query.text(0.5, 0.5, 0.5, 'Not\nFound', ha='center', va='center', fontsize=8)
            ax_query.set_xlim(0, 1)
            ax_query.set_ylim(0, 1)
            ax_query.set_zlim(0, 1)
            ax_query.axis('off')
        
        # Calculate accuracy for this query
        correct = sum(1 for r in results if r['class'] == query_class)
        accuracy = correct / len(results) if results else 0.0
        
        # Add label below with ID, class, and accuracy
        ax_query.text2D(0.5, -0.08, f"{query_id}\nClass: {query_class}\nAcc: {accuracy:.1%}", 
                       ha='center', va='top', fontsize=10, fontweight='bold', transform=ax_query.transAxes)
        
        if row_idx == 0:
            ax_query.set_title("Query\nShape", fontsize=10, pad=5)
        
        # Result shapes (next NUMBER_OF_SIMILAR_SHAPES columns)
        for res_idx in range(NUMBER_OF_SIMILAR_SHAPES):
            ax_result = fig.add_subplot(gs[row_idx, 1 + res_idx], projection='3d')
            
            if res_idx < len(results):
                result = results[res_idx]
                result_class = result['class']
                result_id = result['id']
                result_dist = result['distance']
                
                # Load and render result mesh
                result_obj_path = find_shape_obj_file(result_id, result_class, data_dir)
                
                if result_obj_path:
                    vertices, faces = load_obj_file(result_obj_path)
                    if vertices is not None and faces is not None:
                        result_color = get_class_color(result_class)
                        
                        # Determine border color - use thicker edges for emphasis
                        if result_class == query_class:
                            border_color = 'darkgreen'
                            border_width = 2.0
                        else:
                            border_color = 'darkred'
                            border_width = 2.0
                        
                        render_mesh_to_axis(ax_result, vertices, faces, result_color,
                                          result_id, result_class, edge_color=border_color, 
                                          edge_linewidth=border_width)
                    else:
                        ax_result.text(0.5, 0.5, 0.5, 'Error', ha='center', va='center', fontsize=8)
                        ax_result.set_xlim(0, 1)
                        ax_result.set_ylim(0, 1)
                        ax_result.set_zlim(0, 1)
                        ax_result.axis('off')
                else:
                    ax_result.text(0.5, 0.5, 0.5, '?', ha='center', va='center', fontsize=16)
                    ax_result.set_xlim(0, 1)
                    ax_result.set_ylim(0, 1)
                    ax_result.set_zlim(0, 1)
                    ax_result.axis('off')
                
                # Add label with ID and distance and category
                ax_result.text2D(0.5, -0.08, f"{result_id}\nClass: {result_class}\nDist: {result_dist:.3f}", 
                                ha='center', va='top', fontsize=10,
                                transform=ax_result.transAxes)
            else:
                ax_result.set_axis_off()
            
            # Column header (only in first row)
            if row_idx == 0:
                ax_result.set_title(f"Rank {res_idx + 1}", fontsize=10, pad=5)
    
    print("   ✓ Meshes rendered")
    
    # Add row labels (approach names)
    for row_idx, approach in enumerate(approach_names):
        fig.text(0.01, 0.75 - row_idx * 0.30, 
                APPROACH_NAMES.get(approach, approach),
                ha='left', va='center', fontsize=12, fontweight='bold',
                rotation=90)
    
    # Get class name and F1 scores for title
    class_name = query_classes[tier_label]
    
    # Get F1 score (average across approaches for display)
    f1_values = []
    for app_filename in approach_names:
        app_key = APPROACH_KEY_MAP.get(app_filename, app_filename.replace('.csv', ''))
        if app_key in f1_scores and class_name in f1_scores[app_key]:
            f1_values.append(f1_scores[app_key][class_name])
    
    avg_f1 = np.mean(f1_values) if f1_values else 0.0
    
    # Overall title
    tier_display = tier_name.upper()
    fig.suptitle(f"{tier_display} F1-Score Class: {class_name} (Query Rank #{rank_number})\nAverage F1: {avg_f1:.3f}",
                fontsize=14, fontweight='bold', y=0.96)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='lightgreen', edgecolor='darkgreen', linewidth=2, 
                      label='Correct (same class)'),
        mpatches.Patch(facecolor='lightcoral', edgecolor='darkred', linewidth=2, 
                      label='Incorrect (different class)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, 
              fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.02))
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', pad_inches=0.1)
    print(f"   ✓ Saved {tier_name} tier, rank #{rank_number} visualization to: {output_path}")
    plt.close()


def main():
    """Main execution function."""
    print("="*70)
    print("Creating Query Visualization Grid")
    print("="*70)
    
    # Load F1 scores
    print(f"\n1. Loading F1 scores from: {DEFAULT_F1_SCORES}")
    f1_scores = get_class_f1_scores(DEFAULT_F1_SCORES)
    
    # Use rank-based approach for class selection (best performing)
    rank_based_key = 'matrix_rank_based_optimized'
    if rank_based_key not in f1_scores:
        rank_based_key = list(f1_scores.keys())[0]
    
    # Select representative classes
    print(f"\n2. Selecting representative classes...")
    high_classes, medium_classes, low_classes = select_representative_classes(
        f1_scores[rank_based_key])
    
    print(f"   High F1 classes:")
    for i, cls in enumerate(high_classes, 1):
        print(f"      {i}. {cls} (F1={f1_scores[rank_based_key][cls]:.3f})")
    print(f"   Medium F1 classes:")
    for i, cls in enumerate(medium_classes, 1):
        print(f"      {i}. {cls} (F1={f1_scores[rank_based_key][cls]:.3f})")
    print(f"   Low F1 classes:")
    for i, cls in enumerate(low_classes, 1):
        print(f"      {i}. {cls} (F1={f1_scores[rank_based_key][cls]:.3f})")
    
    # Load analysis labels
    print(f"\n3. Loading class labels from: {DEFAULT_ANALYSIS}")
    analysis_df = load_analysis_labels(DEFAULT_ANALYSIS)
    print(f"   Loaded {len(analysis_df)} shapes")
    
    # Load distance matrix for query selection (ONLY use rank-based for fair comparison)
    print(f"\n4. Loading distance matrix for query selection...")
    print(f"   Using ONLY matrix_rank_based_optimized to select query shapes")
    print(f"   (This ensures all approaches are evaluated on the same queries)")
    best_matrix_path = os.path.join(project_root, DEFAULT_MATCHING[1])  # rank-based
    selection_distance_matrix = load_distance_matrix(best_matrix_path)
    
    # Select query shapes based on best accuracy using rank-based approach
    print(f"\n5. Selecting query shapes with best accuracy (using rank-based approach)...")
    query_ids = {}
    query_classes = {}
    
    # Combine all classes with tier labels
    all_tier_classes = [
        ('high_1', high_classes[0]),
        ('high_2', high_classes[1]),
        ('high_3', high_classes[2]),
        ('medium_1', medium_classes[0]),
        ('medium_2', medium_classes[1]),
        ('medium_3', medium_classes[2]),
        ('low_1', low_classes[0]),
        ('low_2', low_classes[1]),
        ('low_3', low_classes[2])
    ]
    
    for tier_label, class_name in all_tier_classes:
        print(f"   Finding best query for {tier_label} ({class_name})...")
        query_id = get_best_query_shape_from_class(class_name, analysis_df, 
                                                    selection_distance_matrix,
                                                    n=NUMBER_OF_SIMILAR_SHAPES)
        if query_id:
            query_ids[tier_label] = query_id
            query_classes[tier_label] = class_name
            print(f"      Selected: {query_id}")
        else:
            print(f"   Warning: No shapes found for {class_name}")
            return
    
    # Process each approach
    print(f"\n6. Retrieving similar shapes for each approach...")
    approach_results = {}
    
    for matrix_path in DEFAULT_MATCHING:
        full_path = os.path.join(project_root, matrix_path)
        approach_name = os.path.basename(matrix_path)
        
        print(f"\n   Processing: {APPROACH_NAMES.get(approach_name, approach_name)}")
        
        # Load distance matrix
        distance_matrix = load_distance_matrix(full_path)
        
        # Retrieve results for each tier
        tier_results = {}
        for tier_label, query_id in query_ids.items():
            results = retrieve_closest_shapes(query_id, distance_matrix, 
                                             analysis_df, n=NUMBER_OF_SIMILAR_SHAPES)
            tier_results[tier_label] = results
            
            # Calculate accuracy
            query_class = query_classes[tier_label]
            correct = sum(1 for r in results if r['class'] == query_class)
            accuracy = correct / len(results) if results else 0
            print(f"      {tier_label:10s}: {correct}/{len(results)} correct (Acc: {accuracy:.2%})")
        
        approach_results[approach_name] = tier_results
    
    # Create visualization
    print(f"\n7. Creating visualizations...")
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    
    # Create 9 separate figures - one for each combination of rank and tier
    for rank in [1, 2, 3]:
        for tier in ['high', 'medium', 'low']:
            output_path = os.path.join(DEFAULT_OUTPUT_DIR, f"query_examples_{tier}_rank_{rank}.png")
            create_query_grid(approach_results, query_classes, query_ids, 
                             f1_scores, output_path, rank, tier)
    
    print(f"\n{'='*70}")
    print("✓ Query visualization complete!")
    print(f"   Generated 9 figures (3 ranks × 3 tiers) in: {DEFAULT_OUTPUT_DIR}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
