"""Interactive tool to adjust 3D shape rotations and save configurations.

This tool allows you to:
1. Load all shapes that will be used in visualizations
2. Interactively adjust rotation parameters using sliders
3. Save the rotation configuration for each shape
4. Export the final configuration to shape_rotation_config.py

Usage (from project root):
    python -m Src.evalution.interactive_rotation_adjuster
"""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from the main visualization script
from Src.evalution.create_query_visualization import (
    load_obj_file, find_shape_obj_file, get_class_color,
    load_analysis_labels, get_class_f1_scores, select_representative_classes,
    load_distance_matrix, get_best_query_shape_from_class,
    retrieve_closest_shapes, DEFAULT_ANALYSIS, DEFAULT_F1_SCORES,
    DEFAULT_MATCHING, NUMBER_OF_SIMILAR_SHAPES, DEFAULT_MESH_DIR
)


class ShapeRotationAdjuster:
    """Interactive tool for adjusting shape rotations."""
    
    def __init__(self, shapes_to_adjust: List[Tuple[str, str, str]]):
        """Initialize the adjuster.
        
        Args:
            shapes_to_adjust: List of (shape_id, class_name, label) tuples
        """
        self.shapes_to_adjust = shapes_to_adjust
        self.current_index = 0
        self.rotation_configs = {}
        
        # Default rotation values
        self.rot_x = 90
        self.rot_y = 15
        self.rot_z = 0
        self.elev = 20
        self.azim = 45
        
        # Initialize shape data attributes
        self.vertices = None
        self.faces = None
        self.class_name = ""
        self.shape_id = ""
        
        # Flag to prevent updates during loading
        self.loading = False
        
        # Setup the figure and controls
        self.setup_figure()
        
    def setup_figure(self):
        """Setup the matplotlib figure with 3D view and sliders."""
        self.fig = plt.figure(figsize=(12, 10))
        
        # 3D axes for shape display
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_position([0.1, 0.35, 0.8, 0.6])
        
        # Title
        self.title_text = self.fig.text(0.5, 0.97, '', ha='center', fontsize=14, fontweight='bold')
        
        # Sliders
        slider_color = 'lightgoldenrodyellow'
        
        # rot_x slider (0-360)
        ax_rot_x = self.fig.add_axes([0.15, 0.25, 0.7, 0.02])
        self.slider_rot_x = Slider(ax_rot_x, 'Rot X', 0, 360, valinit=self.rot_x, valstep=5, color=slider_color)
        
        # rot_y slider (0-360)
        ax_rot_y = self.fig.add_axes([0.15, 0.20, 0.7, 0.02])
        self.slider_rot_y = Slider(ax_rot_y, 'Rot Y', 0, 360, valinit=self.rot_y, valstep=5, color=slider_color)
        
        # rot_z slider (0-360)
        ax_rot_z = self.fig.add_axes([0.15, 0.15, 0.7, 0.02])
        self.slider_rot_z = Slider(ax_rot_z, 'Rot Z', 0, 360, valinit=self.rot_z, valstep=5, color=slider_color)
        
        # elev slider (0-90)
        ax_elev = self.fig.add_axes([0.15, 0.10, 0.7, 0.02])
        self.slider_elev = Slider(ax_elev, 'Elevation', 0, 90, valinit=self.elev, valstep=5, color=slider_color)
        
        # azim slider (0-360)
        ax_azim = self.fig.add_axes([0.15, 0.05, 0.7, 0.02])
        self.slider_azim = Slider(ax_azim, 'Azimuth', 0, 360, valinit=self.azim, valstep=5, color=slider_color)
        
        # Connect sliders to update function
        self.slider_rot_x.on_changed(self.update)
        self.slider_rot_y.on_changed(self.update)
        self.slider_rot_z.on_changed(self.update)
        self.slider_elev.on_changed(self.update)
        self.slider_azim.on_changed(self.update)
        
        # Buttons
        ax_prev = self.fig.add_axes([0.15, 0.01, 0.1, 0.03])
        self.btn_prev = Button(ax_prev, 'Previous', color='lightblue')
        self.btn_prev.on_clicked(self.previous_shape)
        
        ax_next = self.fig.add_axes([0.3, 0.01, 0.1, 0.03])
        self.btn_next = Button(ax_next, 'Next', color='lightblue')
        self.btn_next.on_clicked(self.next_shape)
        
        ax_save = self.fig.add_axes([0.45, 0.01, 0.15, 0.03])
        self.btn_save = Button(ax_save, 'Save & Next', color='lightgreen')
        self.btn_save.on_clicked(self.save_and_next)
        
        ax_export = self.fig.add_axes([0.65, 0.01, 0.2, 0.03])
        self.btn_export = Button(ax_export, 'Export All & Exit', color='orange')
        self.btn_export.on_clicked(self.export_and_exit)
        
        # Info text
        self.info_text = self.fig.text(0.5, 0.30, '', ha='center', fontsize=10, style='italic')
        
        # Load first shape
        self.load_current_shape()
        
    def load_current_shape(self):
        """Load and display the current shape."""
        # Set loading flag to prevent slider updates
        self.loading = True
        
        if self.current_index >= len(self.shapes_to_adjust):
            self.info_text.set_text("All shapes processed! Click 'Export All & Exit' to save.")
            self.ax.clear()
            self.ax.text(0.5, 0.5, 0.5, 'All Done!', ha='center', va='center', fontsize=20, fontweight='bold')
            plt.draw()
            self.loading = False
            return
        
        shape_id, class_name, label = self.shapes_to_adjust[self.current_index]
        
        # Update title
        self.title_text.set_text(f"Shape {self.current_index + 1}/{len(self.shapes_to_adjust)}: {label}\n"
                                 f"ID: {shape_id} | Class: {class_name}")
        
        # Load shape if not already in configs (use saved config if available)
        if shape_id in self.rotation_configs:
            config = self.rotation_configs[shape_id]
            self.rot_x = config['rot_x']
            self.rot_y = config['rot_y']
            self.rot_z = config['rot_z']
            self.elev = config['elev']
            self.azim = config['azim']
            
            # Update sliders
            self.slider_rot_x.set_val(self.rot_x)
            self.slider_rot_y.set_val(self.rot_y)
            self.slider_rot_z.set_val(self.rot_z)
            self.slider_elev.set_val(self.elev)
            self.slider_azim.set_val(self.azim)
        else:
            # Reset to defaults
            self.rot_x = 90
            self.rot_y = 15
            self.rot_z = 0
            self.elev = 20
            self.azim = 45
            
            # Update sliders
            self.slider_rot_x.set_val(self.rot_x)
            self.slider_rot_y.set_val(self.rot_y)
            self.slider_rot_z.set_val(self.rot_z)
            self.slider_elev.set_val(self.elev)
            self.slider_azim.set_val(self.azim)
        
        # Load OBJ file
        obj_path = find_shape_obj_file(shape_id, class_name, DEFAULT_MESH_DIR)
        if not obj_path:
            self.info_text.set_text(f"❌ OBJ file not found for {shape_id}")
            self.vertices = None
            self.faces = None
        else:
            self.vertices, self.faces = load_obj_file(obj_path)
            if self.vertices is None:
                self.info_text.set_text(f"❌ Failed to load {shape_id}")
            else:
                saved_msg = " (Previously saved)" if shape_id in self.rotation_configs else ""
                self.info_text.set_text(f"✓ Loaded {shape_id}{saved_msg} - Adjust sliders and click 'Save & Next'")
        
        self.class_name = class_name
        self.shape_id = shape_id
        
        # Clear loading flag
        self.loading = False
        
        # Render
        self.render_shape()
        
    def render_shape(self):
        """Render the shape with current rotation settings."""
        self.ax.clear()
        
        if self.vertices is None or self.faces is None:
            self.ax.text(0.5, 0.5, 0.5, 'Shape not loaded', ha='center', va='center', fontsize=12)
            plt.draw()
            return
        
        # Get current values from sliders
        self.rot_x = self.slider_rot_x.val
        self.rot_y = self.slider_rot_y.val
        self.rot_z = self.slider_rot_z.val
        self.elev = self.slider_elev.val
        self.azim = self.slider_azim.val
        
        # Apply rotations
        theta_x = np.radians(self.rot_x)
        rot_x = np.array([
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)]
        ])
        
        theta_y = np.radians(self.rot_y)
        rot_y = np.array([
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)]
        ])
        
        theta_z = np.radians(self.rot_z)
        rot_z = np.array([
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z), np.cos(theta_z), 0],
            [0, 0, 1]
        ])
        
        vertices_rotated = self.vertices @ rot_x.T @ rot_y.T @ rot_z.T
        
        # Create mesh
        mesh = []
        for face in self.faces:
            triangle = vertices_rotated[face]
            mesh.append(triangle)
        
        # Get color
        color = get_class_color(self.class_name)
        
        # Render
        poly = Poly3DCollection(mesh, 
                               alpha=0.85,
                               facecolors=color,
                               edgecolors='black',
                               linewidths=0.05,
                               shade=True)
        self.ax.add_collection3d(poly)
        
        # Set limits
        all_verts = vertices_rotated
        x_min, x_max = all_verts[:, 0].min(), all_verts[:, 0].max()
        y_min, y_max = all_verts[:, 1].min(), all_verts[:, 1].max()
        z_min, z_max = all_verts[:, 2].min(), all_verts[:, 2].max()
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        max_range = max(x_range, y_range, z_range) * 0.55
        
        x_mid = (x_max + x_min) / 2
        y_mid = (y_max + y_min) / 2
        z_mid = (z_max + z_min) / 2
        
        self.ax.set_xlim(x_mid - max_range, x_mid + max_range)
        self.ax.set_ylim(y_mid - max_range, y_mid + max_range)
        self.ax.set_zlim(z_mid - max_range, z_mid + max_range)
        
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        self.ax.grid(False)
        
        # Set camera angle
        self.ax.view_init(elev=self.elev, azim=self.azim)
        self.ax.dist = 7
        
        plt.draw()
        
    def update(self, val):
        """Update the visualization when sliders change."""
        # Don't update while loading a new shape
        if not self.loading:
            self.render_shape()
        
    def save_current(self):
        """Save current rotation configuration."""
        if self.vertices is not None and self.faces is not None:
            self.rotation_configs[self.shape_id] = {
                'rot_x': float(self.rot_x),
                'rot_y': float(self.rot_y),
                'rot_z': float(self.rot_z),
                'elev': float(self.elev),
                'azim': float(self.azim)
            }
            print(f"✓ Saved rotation config for {self.shape_id}")
            return True
        return False
        
    def previous_shape(self, event):
        """Go to previous shape."""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_shape()
            
    def next_shape(self, event):
        """Go to next shape without saving."""
        if self.current_index < len(self.shapes_to_adjust):
            self.current_index += 1
            self.load_current_shape()
            
    def save_and_next(self, event):
        """Save current configuration and move to next shape."""
        if self.save_current():
            self.current_index += 1
            self.load_current_shape()
            
    def export_and_exit(self, event):
        """Export all configurations to file and close."""
        self.export_configs()
        plt.close(self.fig)
        
    def export_configs(self):
        """Export rotation configurations to shape_rotation_config.py."""
        output_path = project_root / "Src" / "evalution" / "shape_rotation_config.py"
        
        with open(output_path, 'w') as f:
            f.write('"""Shape rotation configuration for visualization.\n\n')
            f.write('This file contains rotation parameters for individual shapes and classes.\n')
            f.write('Generated by interactive_rotation_adjuster.py\n')
            f.write('"""\n\n')
            
            f.write('# Per-shape rotation configurations (highest priority)\n')
            f.write('SHAPE_ROTATIONS = {\n')
            for shape_id, config in sorted(self.rotation_configs.items()):
                f.write(f"    '{shape_id}': {{'rot_x': {config['rot_x']}, 'rot_y': {config['rot_y']}, ")
                f.write(f"'rot_z': {config['rot_z']}, 'elev': {config['elev']}, 'azim': {config['azim']}}},\n")
            f.write('}\n\n')
            
            f.write('# Per-class rotation configurations (medium priority)\n')
            f.write('CLASS_ROTATIONS = {\n')
            f.write('}\n\n')
            
            f.write('# Default rotation configuration (fallback)\n')
            f.write("DEFAULT_ROTATION = {'rot_x': 90, 'rot_y': 15, 'rot_z': 0, 'elev': 20, 'azim': 45}\n")
        
        print(f"\n{'='*70}")
        print(f"✓ Exported {len(self.rotation_configs)} rotation configurations to:")
        print(f"  {output_path}")
        print(f"{'='*70}\n")
        
    def run(self):
        """Run the interactive adjuster."""
        plt.show()


def collect_all_shapes_for_visualization():
    """Collect all shapes that will be used in the final visualization.
    
    Returns list of (shape_id, class_name, label) tuples.
    """
    print("="*70)
    print("Collecting shapes for rotation adjustment...")
    print("="*70)
    
    # Load F1 scores
    print(f"\n1. Loading F1 scores from: {DEFAULT_F1_SCORES}")
    f1_scores = get_class_f1_scores(DEFAULT_F1_SCORES)
    
    # Use rank-based approach for class selection
    rank_based_key = 'matrix_rank_based_optimized'
    if rank_based_key not in f1_scores:
        rank_based_key = list(f1_scores.keys())[0]
    
    # Select representative classes
    print(f"\n2. Selecting representative classes...")
    high_classes, medium_classes, low_classes = select_representative_classes(
        f1_scores[rank_based_key])
    
    # Load analysis labels
    print(f"\n3. Loading class labels from: {DEFAULT_ANALYSIS}")
    analysis_df = load_analysis_labels(DEFAULT_ANALYSIS)
    
    # Load distance matrix for query selection
    print(f"\n4. Loading distance matrix...")
    best_matrix_path = os.path.join(project_root, DEFAULT_MATCHING[1])
    selection_distance_matrix = load_distance_matrix(best_matrix_path)
    
    # Load all distance matrices to get all shapes that will appear
    print(f"\n5. Loading all distance matrices to collect shapes...")
    all_distance_matrices = {}
    for matrix_path in DEFAULT_MATCHING:
        full_path = os.path.join(project_root, matrix_path)
        approach_name = os.path.basename(matrix_path)
        all_distance_matrices[approach_name] = load_distance_matrix(full_path)
    
    # Collect all unique shapes
    shapes_set = set()
    
    # Get all tier classes
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
    
    print(f"\n6. Collecting all shapes from queries and results...")
    
    for tier_label, class_name in all_tier_classes:
        # Get query shape
        query_id = get_best_query_shape_from_class(class_name, analysis_df, 
                                                    selection_distance_matrix,
                                                    n=NUMBER_OF_SIMILAR_SHAPES)
        if query_id:
            shapes_set.add((query_id, class_name, f"Query: {tier_label}"))
            
            # Get results from each approach
            for approach_name, distance_matrix in all_distance_matrices.items():
                results = retrieve_closest_shapes(query_id, distance_matrix, 
                                                 analysis_df, n=NUMBER_OF_SIMILAR_SHAPES)
                for idx, result in enumerate(results, 1):
                    shapes_set.add((result['id'], result['class'], 
                                  f"Result for {tier_label} (rank {idx})"))
    
    # Convert to sorted list
    shapes_list = sorted(list(shapes_set), key=lambda x: (x[2], x[0]))
    
    print(f"\n✓ Collected {len(shapes_list)} unique shapes to adjust")
    print(f"  This includes queries and all retrieval results across all approaches")
    
    return shapes_list


def main():
    """Main execution function."""
    # Collect all shapes
    shapes_to_adjust = collect_all_shapes_for_visualization()
    
    if not shapes_to_adjust:
        print("No shapes to adjust!")
        return
    
    print(f"\n{'='*70}")
    print("Starting Interactive Rotation Adjuster")
    print(f"{'='*70}")
    print("\nInstructions:")
    print("  - Use sliders to adjust rotation and camera angles")
    print("  - rot_x, rot_y, rot_z: Rotate the shape around X, Y, Z axes")
    print("  - elev: Camera elevation (height)")
    print("  - azim: Camera azimuth (horizontal position)")
    print("  - Click 'Save & Next' to save current settings and move to next shape")
    print("  - Click 'Next' to skip without saving")
    print("  - Click 'Previous' to go back")
    print("  - Click 'Export All & Exit' when done to save all configurations")
    print(f"{'='*70}\n")
    
    # Create and run adjuster
    adjuster = ShapeRotationAdjuster(shapes_to_adjust)
    adjuster.run()


if __name__ == '__main__':
    main()
