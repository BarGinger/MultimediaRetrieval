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
        
        # Load shapes - OPTIMIZATION: Load CSV once and reuse
        print(f"Loading shapes from: {csv_file_path}")
        import time
        t_csv_start = time.time()
        df = pd.read_csv(csv_file_path)
        t_csv_end = time.time()
        if debug:
            print(f"[DEBUG] CSV load took: {(t_csv_end-t_csv_start)*1000:.0f}ms")
        
        self.shape_names = df['shape'].tolist()
        
        if num_shapes is not None:
            self.shape_names = self.shape_names[:num_shapes]
            df = df[df['shape'].isin(self.shape_names)]  # Filter DataFrame too
            print(f"Limited to first {num_shapes} shapes")
        
        print(f"Loading {len(self.shape_names)} shape objects...")
        t_load_start = time.time()
        self.shapes = []
        for shape_name in tqdm(self.shape_names, desc="Loading shapes", disable=debug):
            try:
                # Pass DataFrame to avoid repeated CSV reads
                shape = Shape(shape_name, csv_file_path, df=df)
                self.shapes.append(shape)
            except Exception as e:
                print(f"\nWarning: Could not load shape {shape_name}: {e}")
                self.shapes.append(None)
        t_load_end = time.time()
        if debug:
            print(f"[DEBUG] Shape object creation took: {(t_load_end-t_load_start)*1000:.0f}ms")
        
        # Filter out failed loads
        valid_indices = [i for i, s in enumerate(self.shapes) if s is not None]
        self.shapes = [self.shapes[i] for i in valid_indices]
        self.shape_names = [self.shape_names[i] for i in valid_indices]
        print(f"Successfully loaded {len(self.shapes)} shapes\n")
        
        # Create mapping from shape name to class for query results
        self.shape_to_class = {s.shape: s.shape_class for s in self.shapes}
        
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
        
        OPTIMIZED: Uses vectorized operations on precomputed matrices instead of
        creating ShapeDistance objects for every pair.
        """
        import time
        t_total_start = time.time()
        
        os.makedirs(self.total_distance_dir, exist_ok=True)
        
        # Generate filename from weights
        filename = self._generate_matrix_filename()
        self.distance_matrix_path = os.path.join(self.total_distance_dir, filename)
        
        # Try to load existing matrix
        if os.path.exists(self.distance_matrix_path):
            print(f"Loading cached total distance matrix: {filename}")
            t_load = time.time()
            self.distance_matrix = pd.read_csv(self.distance_matrix_path, index_col=0)
            t_load_end = time.time()
            print(f"Matrix loaded successfully ({(t_load_end-t_load)*1000:.0f}ms)\n")
            return
        
        # Compute new matrix
        print(f"Computing total distance matrix: {filename}")
        print(f"Using weights from: {os.path.basename(self.weights_csv)}")
        print(f"Using precomputed distances from: {os.path.basename(self.precomputed_dir)}\n")
        
        # === OPTIMIZATION: Load weights ONCE ===
        t_weights = time.time()
        wdf = pd.read_csv(self.weights_csv)
        if 'descriptor' not in wdf.columns or 'weight' not in wdf.columns:
            raise ValueError("Weights CSV must have 'descriptor' and 'weight' columns")
        weights = {row['descriptor']: float(row['weight']) for _, row in wdf.iterrows()}
        t_weights_end = time.time()
        if self.debug:
            print(f"[DEBUG] Loaded weights ({(t_weights_end-t_weights)*1000:.0f}ms): {weights}\n")
        
        # === OPTIMIZATION: Load ALL precomputed matrices ONCE ===
        t_matrices_start = time.time()
        descriptors = ['A3', 'D1', 'D2', 'D3', 'D4']
        global_descriptors = ['surface_area', 'compactness', 'rectangularity', 
                              'diameter', 'convexity', 'eccentricity']
        
        precomputed_matrices = {}
        cache_hits = 0
        cache_misses = 0
        
        # Collect all descriptors to load
        all_descriptors = [(desc, False) for desc in descriptors] + \
                          [(desc, True) for desc in global_descriptors]
        
        print(f"Loading {len(all_descriptors)} precomputed distance matrices...")
        
        for desc, is_global in tqdm(all_descriptors, desc="Loading matrices", disable=self.debug):
            fname = f"distances_global_{desc}.csv" if is_global else f"distances_{desc}.csv"
            path = os.path.join(self.precomputed_dir, fname)
            
            if os.path.exists(path):
                df_dist = pd.read_csv(path, index_col=0)
                # Make symmetric: fill upper triangle from lower triangle
                df_dist = df_dist.combine_first(df_dist.T)
                
                # Take absolute value to ensure non-negative distances
                # Standardized distances can be negative, but distances must be >= 0
                df_dist = df_dist.abs()
                
                precomputed_matrices[desc] = df_dist
                cache_hits += 1
            else:
                cache_misses += 1
        
        t_matrices_end = time.time()
        print(f"Loaded {cache_hits}/{len(all_descriptors)} matrices in {(t_matrices_end-t_matrices_start):.1f}s")
        if cache_misses > 0:
            print(f"⚠ Warning: {cache_misses} descriptor matrices not found - those will be skipped")
        
        # === OPTIMIZATION: Vectorized weighted sum ===
        t_compute_start = time.time()
        n = len(self.shapes)
        
        # Initialize with zeros
        total_distance_matrix = np.zeros((n, n))
        weight_sum_used = 0.0
        descriptors_used = 0
        
        # Get shape names for lookups
        shape_names = [s.shape for s in self.shapes]
        
        print(f"Computing weighted distance matrix for {n} shapes...")
        
        for desc_name, weight in tqdm(weights.items(), desc="Applying weights", disable=self.debug):
            if desc_name not in precomputed_matrices:
                if self.debug:
                    print(f"[DEBUG] Skipping {desc_name} (no precomputed matrix)")
                continue
            
            desc_matrix = precomputed_matrices[desc_name]
            
            # Extract relevant submatrix for our shapes
            # Handle missing shapes gracefully
            try:
                # Reindex to match our shape order, fill missing with NaN
                desc_submatrix = desc_matrix.reindex(index=shape_names, columns=shape_names, fill_value=np.nan)
                desc_values = desc_submatrix.values
                
                # Add weighted contribution (skip NaN values)
                valid_mask = ~np.isnan(desc_values)
                total_distance_matrix[valid_mask] += weight * desc_values[valid_mask]
                
                weight_sum_used += weight
                descriptors_used += 1
                
                if self.debug:
                    valid_count = np.sum(valid_mask)
                    print(f"[DEBUG] {desc_name}: weight={weight:.4f}, valid_entries={valid_count}/{n*n}")
            
            except Exception as e:
                print(f"Warning: Could not process descriptor {desc_name}: {e}")
                continue
        
        t_compute_end = time.time()
        print(f"\nWeighted sum computed using {descriptors_used} descriptors")
        print(f"  Matrix loading: {(t_matrices_end-t_matrices_start):.1f}s")
        print(f"  Computation: {(t_compute_end-t_compute_start):.1f}s")
        
        # Normalize by sum of used weights (so distances are in same scale)
        if weight_sum_used > 0:
            total_distance_matrix /= weight_sum_used
        
        # Create DataFrame
        self.distance_matrix = pd.DataFrame(
            total_distance_matrix,
            index=shape_names,
            columns=shape_names
        )
        
        # Save to CSV
        t_save = time.time()
        self.distance_matrix.to_csv(self.distance_matrix_path)
        t_save_end = time.time()
        print(f"  Saving to CSV: {(t_save_end-t_save):.1f}s")
        
        # Print statistics
        valid_distances = total_distance_matrix[total_distance_matrix > 0]
        if len(valid_distances) > 0:
            print(f"\nDistance matrix statistics:")
            print(f"  Shape: {n}×{n} ({n*n:,} total entries)")
            print(f"  Non-zero: {len(valid_distances):,} ({100*len(valid_distances)/(n*n):.1f}%)")
            print(f"  Min: {np.min(valid_distances):.6f}")
            print(f"  Max: {np.max(valid_distances):.6f}")
            print(f"  Mean: {np.mean(valid_distances):.6f}")
            print(f"  Median: {np.median(valid_distances):.6f}")
        
        t_total_end = time.time()
        print(f"\n{'='*60}")
        print(f"Total time: {(t_total_end-t_total_start):.1f}s")
        print(f"Saved to: {os.path.basename(self.distance_matrix_path)}")
        print(f"{'='*60}\n")
    
    def query(self, query_shape_name: str, k: int = 10, include_self: bool = False) -> pd.DataFrame:
        """
        Find the k nearest shapes to a query shape.
        
        Args:
            query_shape_name: Name of the query shape (must be in the dataset)
            k: Number of nearest neighbors to return (default: 10)
            include_self: If True, include the query shape itself in results (default: False)
        
        Returns:
            pd.DataFrame with columns ['shape', 'class', 'distance', 'rank'] sorted by distance
        
        Raises:
            ValueError: If query shape not found or distance matrix not computed
        """
        if self.distance_matrix is None:
            raise ValueError("Distance matrix not computed. Set compute_matrix=True or call _load_or_compute_distance_matrix()")
        
        if query_shape_name not in self.distance_matrix.index:
            raise ValueError(f"Query shape '{query_shape_name}' not found in distance matrix")
        
        # Get query shape class
        query_class = self.shape_to_class.get(query_shape_name, 'Unknown')
        print(f"Query: {query_shape_name} (class: {query_class})")
        
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
        
        # Build result DataFrame with shape class
        result = pd.DataFrame({
            'shape': top_k.index,
            'class': [self.shape_to_class.get(shape, 'Unknown') for shape in top_k.index],
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
    # Test with a small subset first to see performance gains
    print("="*60)
    print("Testing optimized ShapeQuery with 20 shapes")
    print("="*60 + "\n")
    
    qs = ShapeQuery()

    
    print("="*60 + "\n")
    print("INITIALIZED ShapeQuery\n")
    print("="*60 + "\n")
    
    # Try a query
    if len(qs.shape_names) > 0:
        test_shape = qs.shape_names[0]
        print(f"\n{'='*60}")
        print("="*60 + "\n")
        results = qs.query(test_shape, k=10)
        print(results)
