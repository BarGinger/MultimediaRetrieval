import pandas as pd
import numpy as np
import os
import json


class Shape:
    """
    A class to represent a 3D shape and its features.
    
    Features are loaded from a CSV file containing precomputed shape descriptors,
    including global features and histogram-based descriptors.
    """
    
    def __init__(self, obj_file_path, csv_file_path="final_006_cleaned.csv", df: pd.DataFrame | None = None):
        """
        Initialize a Shape object with features from the CSV file.
        
        Args:
            obj_file_path (str): Path to the .obj file for this shape
            csv_file_path (str): Path to the CSV file containing shape features
                                (default: "final_006_cleaned.csv" in the same directory)
            df (pd.DataFrame, optional): Pre-loaded DataFrame to avoid repeated CSV reads.
                                         If provided, csv_file_path is only used for reference.
        """
        self.obj_file_path = obj_file_path
        self.shape_name = os.path.basename(obj_file_path)
        
        # If csv_file_path is relative, assume it's in the same directory as this script
        if not os.path.isabs(csv_file_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            csv_file_path = os.path.join(script_dir, csv_file_path)
        
        self.csv_file_path = csv_file_path
        
        # Load features from CSV or DataFrame
        self._load_features(df)
    
    def _load_features(self, df: pd.DataFrame | None = None):
        """Load all features from the CSV file and set them as properties."""
        # Read the CSV file only if DataFrame not provided
        if df is None:
            df = pd.read_csv(self.csv_file_path)
        
        # Find the row corresponding to this shape
        row = df[df['shape'] == self.shape_name]
        
        if row.empty:
            raise ValueError(f"Shape '{self.shape_name}' not found in CSV file '{self.csv_file_path}'")
        
        # Get the first (and should be only) matching row
        row = row.iloc[0]
        
        # Store basic identification
        self.shape = row['shape']
        self.shape_class = row['class']
        
        # Store mesh properties
        self.num_vertices = int(row['num_vertices']) if pd.notna(row['num_vertices']) else None
        self.num_faces = int(row['num_faces']) if pd.notna(row['num_faces']) else None
        self.face_types = row['face_types']
        
        # Parse and store bounding box as a dictionary
        if pd.notna(row['bounding_box']):
            self.bounding_box = json.loads(row['bounding_box'].replace("'", '"'))
        else:
            self.bounding_box = None
        
        # Store global shape descriptors
        self.surface_area = float(row['surface_area']) if pd.notna(row['surface_area']) else None
        self.compactness = float(row['compactness']) if pd.notna(row['compactness']) else None
        self.rectangularity = float(row['rectangularity']) if pd.notna(row['rectangularity']) else None
        self.diameter = float(row['diameter']) if pd.notna(row['diameter']) else None
        self.convexity = float(row['convexity']) if pd.notna(row['convexity']) else None
        self.eccentricity = float(row['eccentricity']) if pd.notna(row['eccentricity']) else None
        
        # Store histogram descriptors as numpy arrays
        # A3 descriptor (angle between 3 random points)
        self.A3_hist = self._parse_array(row['A3_hist'])
        self.A3_bins = self._parse_array(row['A3_bins'])
        
        # D1 descriptor (distance from centroid to surface)
        self.D1_hist = self._parse_array(row['D1_hist'])
        self.D1_bins = self._parse_array(row['D1_bins'])
        
        # D2 descriptor (distance between 2 random points)
        self.D2_hist = self._parse_array(row['D2_hist'])
        self.D2_bins = self._parse_array(row['D2_bins'])
        
        # D3 descriptor (square root of area of triangle formed by 3 random points)
        self.D3_hist = self._parse_array(row['D3_hist'])
        self.D3_bins = self._parse_array(row['D3_bins'])
        
        # D4 descriptor (cube root of volume of tetrahedron formed by 4 random points)
        self.D4_hist = self._parse_array(row['D4_hist'])
        self.D4_bins = self._parse_array(row['D4_bins'])
    
    def _parse_array(self, value):
        """
        Parse a semicolon-separated string into a numpy array.
        
        Args:
            value: String value with semicolon-separated numbers
            
        Returns:
            numpy.ndarray: Array of float values
        """
        if pd.isna(value):
            return None
        
        # Split by semicolon and convert to float array
        return np.array([float(x) for x in str(value).split(';')])
    
    def get_histogram_features(self):
        """
        Get all histogram features as a dictionary.
        
        Returns:
            dict: Dictionary with histogram names as keys and (hist, bins) tuples as values
        """
        return {
            'A3': (self.A3_hist, self.A3_bins),
            'D1': (self.D1_hist, self.D1_bins),
            'D2': (self.D2_hist, self.D2_bins),
            'D3': (self.D3_hist, self.D3_bins),
            'D4': (self.D4_hist, self.D4_bins)
        }
    
    def get_global_features(self):
        """
        Get all global shape features as a dictionary.
        
        Returns:
            dict: Dictionary with feature names as keys and values
        """
        return {
            'surface_area': self.surface_area,
            'compactness': self.compactness,
            'rectangularity': self.rectangularity,
            'diameter': self.diameter,
            'convexity': self.convexity,
            'eccentricity': self.eccentricity
        }
    
    def get_all_features_vector(self):
        """
        Get all features as a single concatenated numpy array.
        Useful for distance calculations and machine learning.
        
        Returns:
            numpy.ndarray: Concatenated array of all features
        """
        features = []
        
        # Add global features
        global_features = self.get_global_features()
        for key in sorted(global_features.keys()):
            value = global_features[key]
            if value is not None:
                features.append(value)
        
        # Add histogram features
        hist_features = self.get_histogram_features()
        for key in sorted(hist_features.keys()):
            hist, bins = hist_features[key]
            if hist is not None:
                features.extend(hist)
        
        return np.array(features)
    
    def __repr__(self):
        """String representation of the Shape object."""
        return (f"Shape(name='{self.shape}', class='{self.shape_class}', "
                f"vertices={self.num_vertices}, faces={self.num_faces})")
    
    def __str__(self):
        """Human-readable string representation."""
        return (f"Shape: {self.shape}\n"
                f"  Class: {self.shape_class}\n"
                f"  Vertices: {self.num_vertices}\n"
                f"  Faces: {self.num_faces}")


# Example usage
if __name__ == "__main__":
    # Example: Load a shape
    shape = Shape("m1337_06_fill_holes_and_orientation.obj")
    
    print(shape)
    print("\nGlobal Features:")
    for key, value in shape.get_global_features().items():
        print(f"  {key}: {value}")
    
    print("\nHistogram Features:")
    for key, (hist, bins) in shape.get_histogram_features().items():
        print(f"  {key}: hist shape={hist.shape if hist is not None else None}, "
              f"bins shape={bins.shape if bins is not None else None}")
    
    print(f"\nFull feature vector shape: {shape.get_all_features_vector().shape}")
