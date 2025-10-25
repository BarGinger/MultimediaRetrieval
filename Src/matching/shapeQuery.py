"""
Shape querying system with weighted distance matrices and k-nearest neighbor retrieval.

This module provides:
- Automatic computation or loading of total-distance matrices based on descriptor weights
- Deterministic filename generation from weight configurations (for caching)
- k-NN query functionality for shape retrieval
"""

import os
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from shapeFeatures import Shape
from distance import ShapeDistance


class ShapeQuery:
    """
    A class for querying shapes using weighted descriptor distances.
    
    Manages total distance matrix computation/caching and provides k-NN retrieval.
    """
    
    def __init__(self, 
                 csv_file_path: str = "final_006_cleaned.csv",
                 weights_csv: str = "distance_weights.csv",
                 precomputed_dir: str = "distance_matrices_normalized_98",
                 total_distance_dir: str = "total_distances",
                 compute_matrix: bool = True,
                 num_shapes: int | None = None,
                 debug: bool = False):
        """
        Initialize the ShapeQuery system.
        
        Args:
            csv_file_path: Path to CSV containing shape features
            weights_csv: Path to CSV containing descriptor weights
            precomputed_dir: Directory with precomputed per-descriptor distance matrices
            total_distance_dir: Directory to save/load total distance matrices
            compute_matrix: If True, compute or load the total distance matrix on init
            num_shapes: Limit to first N shapes (default: all shapes)
            debug: If True, print timing and debugging information
        """
        self.debug = debug
        
        # Resolve paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        if not os.path.isabs(csv_file_path):
            csv_file_path = os.path.join(script_dir, csv_file_path)
        if not os.path.isabs(weights_csv):
            weights_csv = os.path.join(script_dir, weights_csv)
        if not os.path.isabs(precomputed_dir):
            precomputed_dir = os.path.join(script_dir, precomputed_dir)
        if not os.path.isabs(total_distance_dir):
            total_distance_dir = os.path.join(script_dir, total_distance_dir)
        
        self.csv_file_path = csv_file_path
        self.weights_csv = weights_csv
        self.precomputed_dir = precomputed_dir
        self.total_distance_dir = total_distance_dir
        self.num_shapes = num_shapes
        
        # Load shapes
        print(f"Loading shapes from: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        self.shape_names = df['shape'].tolist()
        
        if num_shapes is not None:
            self.shape_names = self.shape_names[:num_shapes]
            print(f"Limited to first {num_shapes} shapes")
        
        print(f"Loading {len(self.shape_names)} shape objects...")
        self.shapes = []
        for shape_name in tqdm(self.shape_names, desc="Loading shapes"):
            try:
                shape = Shape(shape_name, csv_file_path)
                self.shapes.append(shape)
            except Exception as e:
                print(f"\nWarning: Could not load shape {shape_name}: {e}")
                self.shapes.append(None)
        
        # Filter out failed loads
        valid_indices = [i for i, s in enumerate(self.shapes) if s is not None]
        self.shapes = [self.shapes[i] for i in valid_indices]
        self.shape_names = [self.shape_names[i] for i in valid_indices]
        print(f"Successfully loaded {len(self.shapes)} shapes\n")
        
        # Distance matrix (to be computed or loaded)
        self.distance_matrix = None
        self.distance_matrix_path = None
        
        if compute_matrix:
            self._load_or_compute_distance_matrix()
    
    def _generate_matrix_filename(self) -> str:
        """
        Generate a deterministic filename from the weight configuration.
        
        Uses a hash of the sorted (descriptor, weight) pairs to create a unique
        but reproducible filename for each weight configuration.
        
        Returns:
            str: Filename like "total_distances_abc123def.csv"
        """
        # Read weights
        wdf = pd.read_csv(self.weights_csv)
        if 'descriptor' not in wdf.columns or 'weight' not in wdf.columns:
            raise ValueError("Weights CSV must have 'descriptor' and 'weight' columns")
        
        # Create deterministic string representation
        # Sort by descriptor name for consistency
        wdf_sorted = wdf.sort_values('descriptor')
        
        # Build string: "descriptor1:weight1,descriptor2:weight2,..."
        weight_str = ','.join([
            f"{row['descriptor']}:{row['weight']:.10f}"
            for _, row in wdf_sorted.iterrows()
        ])
        
        # Hash it (MD5 is fine for filename generation)
        hash_obj = hashlib.md5(weight_str.encode('utf-8'))
        hash_hex = hash_obj.hexdigest()[:12]  # Use first 12 chars
        
        return f"total_distances_{hash_hex}.csv"
    
    def _load_or_compute_distance_matrix(self):
        """
        Load or compute the total distance matrix based on current weights.
        
        If a cached matrix exists for the current weight configuration, load it.
        Otherwise, compute it and save for future use.
        """
        os.makedirs(self.total_distance_dir, exist_ok=True)
        
        # Generate filename from weights
        filename = self._generate_matrix_filename()
        self.distance_matrix_path = os.path.join(self.total_distance_dir, filename)
        
        # Try to load existing matrix
        if os.path.exists(self.distance_matrix_path):
            print(f"Loading cached total distance matrix: {filename}")
            self.distance_matrix = pd.read_csv(self.distance_matrix_path, index_col=0)
            print("Matrix loaded successfully\n")
            return
        
        # Compute new matrix
        print(f"Computing total distance matrix: {filename}")
        print(f"Using weights from: {os.path.basename(self.weights_csv)}")
        print(f"Using precomputed distances from: {os.path.basename(self.precomputed_dir)}\n")
        
        n = len(self.shapes)
        distance_matrix = np.full((n, n), np.nan)
        np.fill_diagonal(distance_matrix, 0.0)
        
        total_comparisons = n * (n - 1) // 2
        
        if self.debug:
            import time
            print("\n[DEBUG] Starting distance computation loop...")
            print(f"[DEBUG] Total comparisons: {total_comparisons}")
            print(f"[DEBUG] Precomputed dir: {self.precomputed_dir}")
            print(f"[DEBUG] Weights CSV: {self.weights_csv}\n")
        
        with tqdm(total=total_comparisons, desc="Computing total distances", disable=self.debug) as pbar:
            for i in range(n):
                for j in range(i):  # Lower triangle only
                    if self.debug and (i * n + j) < 5:  # Debug first few iterations
                        print(f"\n[DEBUG] === Iteration {i},{j} ===")
                        print(f"[DEBUG] Shape A: {self.shape_names[i]}")
                        print(f"[DEBUG] Shape B: {self.shape_names[j]}")
                        t_start = time.time()
                    
                    try:
                        # Create distance calculator with precomputed matrices
                        if self.debug and (i * n + j) < 5:
                            t0 = time.time()
                        
                        dist_calc = ShapeDistance(
                            self.shapes[i], 
                            self.shapes[j],
                            precomputed_dir=self.precomputed_dir,
                            debug=self.debug if (i * n + j) < 5 else False
                        )
                        
                        if self.debug and (i * n + j) < 5:
                            t1 = time.time()
                            print(f"[DEBUG] ShapeDistance init took: {(t1-t0)*1000:.2f}ms")
                        
                        # Compute weighted total distance
                        if self.debug and (i * n + j) < 5:
                            t0 = time.time()
                        
                        distance = dist_calc.total_distance(
                            weights_csv=self.weights_csv,
                            normalize_missing=True
                        )
                        
                        if self.debug and (i * n + j) < 5:
                            t1 = time.time()
                            print(f"[DEBUG] total_distance() took: {(t1-t0)*1000:.2f}ms")
                            print(f"[DEBUG] Computed distance: {distance:.6f}")
                            t_end = time.time()
                            print(f"[DEBUG] Total iteration time: {(t_end-t_start)*1000:.2f}ms")
                        
                        distance_matrix[i, j] = distance
                        
                    except Exception as e:
                        print(f"\nError computing distance between {self.shape_names[i]} and {self.shape_names[j]}: {e}")
                        distance_matrix[i, j] = np.nan
                    
                    pbar.update(1)
        
        # Create DataFrame
        self.distance_matrix = pd.DataFrame(
            distance_matrix,
            index=self.shape_names,
            columns=self.shape_names
        )
        
        # Save to CSV
        self.distance_matrix.to_csv(self.distance_matrix_path)
        print(f"\nSaved total distance matrix to: {self.distance_matrix_path}")
        
        # Print statistics
        valid_distances = distance_matrix[~np.isnan(distance_matrix) & (distance_matrix > 0)]
        if len(valid_distances) > 0:
            print(f"  Min distance: {np.min(valid_distances):.6f}")
            print(f"  Max distance: {np.max(valid_distances):.6f}")
            print(f"  Mean distance: {np.mean(valid_distances):.6f}")
            print(f"  Median distance: {np.median(valid_distances):.6f}\n")
    
    def query(self, query_shape_name: str, k: int = 10, include_self: bool = False) -> pd.DataFrame:
        """
        Find the k nearest shapes to a query shape.
        
        Args:
            query_shape_name: Name of the query shape (must be in the dataset)
            k: Number of nearest neighbors to return (default: 10)
            include_self: If True, include the query shape itself in results (default: False)
        
        Returns:
            pd.DataFrame with columns ['shape', 'distance', 'rank'] sorted by distance
        
        Raises:
            ValueError: If query shape not found or distance matrix not computed
        """
        if self.distance_matrix is None:
            raise ValueError("Distance matrix not computed. Set compute_matrix=True or call _load_or_compute_distance_matrix()")
        
        if query_shape_name not in self.distance_matrix.index:
            raise ValueError(f"Query shape '{query_shape_name}' not found in distance matrix")
        
        # Get distances from query shape to all others
        distances = self.distance_matrix.loc[query_shape_name]
        
        # Handle symmetric matrix (check both row and column)
        if distances.isna().all():
            # Try column (in case it's in upper triangle)
            distances = self.distance_matrix[query_shape_name]
        
        # Combine with column if some values are NaN (for lower-triangle matrices)
        col_distances = self.distance_matrix[query_shape_name]
        distances = distances.combine_first(col_distances)
        
        # Remove NaN values
        distances = distances.dropna()
        
        # Optionally exclude the query shape itself
        if not include_self and query_shape_name in distances.index:
            distances = distances.drop(query_shape_name)
        
        # Sort by distance
        distances_sorted = distances.sort_values()
        
        # Take top k
        top_k = distances_sorted.head(k)
        
        # Build result DataFrame
        result = pd.DataFrame({
            'shape': top_k.index,
            'distance': top_k.values,
            'rank': range(1, len(top_k) + 1)
        })
        
        return result
    
    def batch_query(self, query_shape_names: list[str], k: int = 10, include_self: bool = False) -> dict:
        """
        Query multiple shapes at once.
        
        Args:
            query_shape_names: List of query shape names
            k: Number of nearest neighbors per query
            include_self: Include query shapes in their own results
        
        Returns:
            dict mapping query_shape_name -> result DataFrame
        """
        results = {}
        for query_name in tqdm(query_shape_names, desc="Batch querying"):
            try:
                results[query_name] = self.query(query_name, k, include_self)
            except Exception as e:
                print(f"\nError querying {query_name}: {e}")
                results[query_name] = None
        return results
    
    def __repr__(self):
        """String representation."""
        n = len(self.shapes) if self.shapes else 0
        return f"ShapeQuery(shapes={n}, weights={os.path.basename(self.weights_csv)})"


# Example usage
if __name__ == "__main__":
    # # Initialize query system with default weights
    # query_system = ShapeQuery(
    #     csv_file_path="final_006_cleaned.csv",
    #     weights_csv="distance_weights.csv"
    # )
    
    # # Query a single shape
    # query_shape = query_system.shape_names[0]
    # print(f"Querying for shape: {query_shape}\n")
    
    # results = query_system.query(query_shape, k=10)
    # print("Top 10 nearest shapes:")
    # print(results)
    
    # Optional: batch query multiple shapes
    # test_queries = query_system.shape_names[:5]
    # batch_results = query_system.batch_query(test_queries, k=5)
    # for query, result in batch_results.items():
    #     print(f"\nQuery: {query}")
    #     print(result)

    # Test with just 10 shapes to see debug output
  qs = ShapeQuery(
      num_shapes=10,
      debug=True
  )
