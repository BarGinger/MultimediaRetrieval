"""
Final HybridShapeQuery: Raw distances with [0,1] normalization + individual descriptor weights.

This approach:
1. Uses raw EMD and Euclidean distances (no z-score standardization)
2. Normalizes each distance matrix to [0,1] range 
3. Allows individual weights for each descriptor
4. Lets optimization discover which descriptors are discriminative
5. Avoids negative distance cancellation issues
"""

import os
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import euclidean_distances


class FinalShapeQuery:
    """
    Final corrected ShapeQuery using raw distances with individual descriptor weights.
    
    Key improvements:
    - Raw distance computation (EMD + Euclidean)
    - Simple [0,1] normalization (no z-score standardization)
    - Individual weights per descriptor (optimizable)
    - No negative distance cancellation
    - Fast caching for optimization
    """
    
    def __init__(self, 
                 csv_file_path: str = "Src/matching/final_006_cleaned.csv",
                 cache_dir: str = "distance_matrices_raw_normalized",
                 weights: dict = None,
                 num_shapes: int = None,
                 debug: bool = False):
        """
        Initialize the final shape query system.
        
        Args:
            csv_file_path: Path to CSV containing shape features
            cache_dir: Directory to cache distance matrices
            weights: Dictionary with individual descriptor weights (must sum to 1.0)
            num_shapes: Limit to first N shapes (default: all)
            debug: If True, print detailed information
        """
        self.debug = debug
        self.cache_dir = cache_dir
        self.csv_file_path = csv_file_path
        self.num_shapes = num_shapes
        
        # Default equal weights for all 11 descriptors
        if weights is None:
            # Start with equal weights - optimization will find better ones
            weight_per_descriptor = 1.0 / 11
            self.weights = {
                'surface_area': weight_per_descriptor,
                'compactness': weight_per_descriptor, 
                'rectangularity': weight_per_descriptor,
                'diameter': weight_per_descriptor,
                'convexity': weight_per_descriptor,
                'eccentricity': weight_per_descriptor,
                'A3_hist': weight_per_descriptor,
                'D1_hist': weight_per_descriptor,
                'D2_hist': weight_per_descriptor,
                'D3_hist': weight_per_descriptor,
                'D4_hist': weight_per_descriptor
            }
        else:
            self.weights = weights.copy()
            
        # Validate weights
        if abs(sum(self.weights.values()) - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {sum(self.weights.values())}")
        
        # Feature definitions
        self.global_features = ['surface_area', 'compactness', 'rectangularity', 'diameter', 'convexity', 'eccentricity']
        self.histogram_features = ['A3_hist', 'D1_hist', 'D2_hist', 'D3_hist', 'D4_hist']
        
        # Load data
        self._load_data()
        
        # Pre-compute or load distance matrices
        self.distance_matrices = {}
        self._load_or_compute_matrices()
        
        # Current combined distance matrix
        self.combined_distance_matrix = None
        self._compute_combined_matrix()
    
    def _load_data(self):
        """Load and prepare feature data."""
        if self.debug:
            print(f"Loading feature data from: {os.path.basename(self.csv_file_path)}")
        
        self.features_df = pd.read_csv(self.csv_file_path)
        
        if self.num_shapes:
            self.features_df = self.features_df.head(self.num_shapes)
            if self.debug:
                print(f"Limited to first {self.num_shapes} shapes")
        
        if self.debug:
            print(f"Dataset: {len(self.features_df)} shapes, {len(self.features_df['class'].unique())} classes")
        
        # Check available features
        available_global = [f for f in self.global_features if f in self.features_df.columns]
        available_histograms = [f for f in self.histogram_features if f in self.features_df.columns]
        
        self.available_global = available_global
        self.available_histograms = available_histograms
        
        # Extract data
        self.shape_names = self.features_df['shape'].tolist()
        self.shape_classes = self.features_df['class'].tolist()
        
        if self.debug:
            print(f"Global features ({len(self.available_global)}): {self.available_global}")
            print(f"Histogram features ({len(self.available_histograms)}): {self.available_histograms}")
    
    def _parse_histogram(self, hist_str):
        """Parse histogram string into numpy array."""
        if pd.isna(hist_str) or hist_str == '':
            return np.array([])
        return np.array([float(x) for x in hist_str.split(';')])
    
    def _compute_emd_matrix(self, hist_feature):
        """Compute EMD distance matrix for a histogram feature."""
        n = len(self.features_df)
        emd_matrix = np.zeros((n, n))
        
        # Pre-parse all histograms
        histograms = []
        for i in range(n):
            hist_str = self.features_df.iloc[i][hist_feature]
            histograms.append(self._parse_histogram(hist_str))
        
        if self.debug:
            print(f"  Computing EMD for {hist_feature}...")
        
        for i in tqdm(range(n), desc=f"EMD {hist_feature}", disable=not self.debug):
            for j in range(i+1, n):
                hist1 = histograms[i]
                hist2 = histograms[j]
                
                if len(hist1) == 0 or len(hist2) == 0 or len(hist1) != len(hist2):
                    dist = 1.0  # Max distance for missing/incompatible data
                else:
                    try:
                        # Use Wasserstein distance (1D EMD)
                        dist = wasserstein_distance(hist1, hist2)
                        # Limit to reasonable range (histograms sum to ~1)
                        dist = min(dist, 1.0)
                    except:
                        dist = 1.0
                
                emd_matrix[i, j] = dist
                emd_matrix[j, i] = dist  # Symmetric
        
        return emd_matrix
    
    def _compute_euclidean_matrix(self, global_feature):
        """Compute Euclidean distance matrix for a global feature."""
        # Extract feature values (already normalized in preprocessing)
        feature_values = self.features_df[global_feature].values.reshape(-1, 1)
        
        # Compute pairwise distances
        distance_matrix = euclidean_distances(feature_values)
        
        return distance_matrix
    
    def _normalize_distance_matrix(self, distance_matrix, descriptor_name):
        """
        Normalize distance matrix to [0,1] range using min-max normalization.
        
        This preserves distance meaning while ensuring equal ranges:
        - 0 = identical (minimum distance)
        - 1 = maximally different (maximum distance in dataset)
        """
        min_dist = np.min(distance_matrix)
        max_dist = np.max(distance_matrix)
        
        if max_dist <= min_dist:
            # No variation - return zeros
            if self.debug:
                print(f"    {descriptor_name}: no variation (all distances = {min_dist:.6f})")
            return np.zeros_like(distance_matrix)
        
        # Min-max normalization to [0,1]
        normalized_matrix = (distance_matrix - min_dist) / (max_dist - min_dist)
        
        if self.debug:
            print(f"    {descriptor_name}: raw range [{min_dist:.6f}, {max_dist:.6f}] → normalized [0, 1]")
        
        return normalized_matrix
    
    def _load_or_compute_matrices(self):
        """Load or compute all distance matrices."""
        os.makedirs(self.cache_dir, exist_ok=True)
        
        if self.debug:
            print(f"\\nLoading/computing raw distance matrices...")
        
        # Process each descriptor
        all_descriptors = self.available_global + self.available_histograms
        
        for descriptor in all_descriptors:
            cache_file = os.path.join(self.cache_dir, f"{descriptor}_raw_normalized.csv")
            
            if os.path.exists(cache_file):
                if self.debug:
                    print(f"Loading cached: {descriptor}")
                distance_matrix = pd.read_csv(cache_file, index_col=0).values
            else:
                if self.debug:
                    print(f"Computing: {descriptor}")
                
                if descriptor in self.available_global:
                    # Euclidean distance for global features
                    raw_matrix = self._compute_euclidean_matrix(descriptor)
                else:
                    # EMD for histogram features
                    raw_matrix = self._compute_emd_matrix(descriptor)
                
                # Normalize to [0,1] range (no z-score standardization)
                distance_matrix = self._normalize_distance_matrix(raw_matrix, descriptor)
                
                # Save to cache
                df_cache = pd.DataFrame(distance_matrix, index=self.shape_names, columns=self.shape_names)
                df_cache.to_csv(cache_file)
                if self.debug:
                    print(f"  Cached to: {os.path.basename(cache_file)}")
            
            self.distance_matrices[descriptor] = distance_matrix
        
        if self.debug:
            print(f"All {len(all_descriptors)} raw distance matrices ready.\\n")
    
    def _compute_combined_matrix(self):
        """Compute weighted combination of distance matrices."""
        if self.debug:
            print("Computing weighted combination of raw distances...")
        
        n = len(self.shape_names)
        combined_matrix = np.zeros((n, n))
        weight_sum = 0.0
        
        for descriptor, weight in self.weights.items():
            if descriptor in self.distance_matrices:
                combined_matrix += weight * self.distance_matrices[descriptor]
                weight_sum += weight
                if self.debug:
                    print(f"  {descriptor}: weight={weight:.4f}")
        
        # Normalize by sum of weights used
        if weight_sum > 0:
            combined_matrix /= weight_sum
        
        self.combined_distance_matrix = pd.DataFrame(
            combined_matrix, 
            index=self.shape_names, 
            columns=self.shape_names
        )
        
        if self.debug:
            print(f"Combined matrix computed (weight_sum={weight_sum:.4f})\\n")
    
    def update_weights(self, new_weights: dict):
        """
        Update weights and recompute combined matrix.
        
        This is FAST because distance matrices are pre-computed.
        """
        # Validate weights
        if abs(sum(new_weights.values()) - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {sum(new_weights.values())}")
        
        # Only recompute if weights changed
        if new_weights != self.weights:
            self.weights = new_weights.copy()
            self._compute_combined_matrix()
    
    def query(self, query_shape_name: str, k: int = 10, include_self: bool = False) -> pd.DataFrame:
        """Find the k nearest shapes to a query shape."""
        if query_shape_name not in self.shape_names:
            raise ValueError(f"Query shape '{query_shape_name}' not found in dataset")
        
        # Get distances from query to all others
        query_distances = self.combined_distance_matrix.loc[query_shape_name].sort_values()
        
        if not include_self:
            query_distances = query_distances.drop(query_shape_name)
        
        # Get top k
        top_k = query_distances.head(k)
        
        # Build result DataFrame
        results = []
        for rank, (shape_name, distance) in enumerate(top_k.items(), 1):
            shape_idx = self.shape_names.index(shape_name)
            shape_class = self.shape_classes[shape_idx]
            
            results.append({
                'shape': shape_name,
                'class': shape_class,
                'distance': distance,
                'rank': rank
            })
        
        return pd.DataFrame(results)
    
    def evaluate_retrieval(self, n_queries: int = 100, k: int = 30) -> dict:
        """Evaluate retrieval performance using random queries."""
        import random
        
        # Select diverse query shapes
        unique_classes = list(set(self.shape_classes))
        query_indices = []
        
        # Handle None case: use comprehensive evaluation with multiple queries per class
        if n_queries is None:
            # Use 2-3 queries per class that has multiple shapes for robust evaluation
            for class_name in unique_classes:
                class_indices = [i for i, c in enumerate(self.shape_classes) if c == class_name]
                if len(class_indices) >= 2:
                    # Take 2 random samples from each class (or 1 if only 2 shapes total)
                    n_samples = min(2, len(class_indices) - 1)  # Leave at least 1 for retrieval
                    query_indices.extend(random.sample(class_indices, n_samples))
        else:
            # Take samples from each class
            for class_name in unique_classes:
                class_indices = [i for i, c in enumerate(self.shape_classes) if c == class_name]
                if len(class_indices) >= 2:
                    n_samples = min(2, len(class_indices), max(1, n_queries // len(unique_classes)))
                    query_indices.extend(random.sample(class_indices, n_samples))
            
            # Limit to requested number
            query_indices = query_indices[:n_queries]
        
        precisions = []
        
        for query_idx in tqdm(query_indices, desc="Evaluating queries", disable=not self.debug):
            query_shape = self.shape_names[query_idx]
            query_class = self.shape_classes[query_idx]
            
            # Get top k retrievals
            results = self.query(query_shape, k=k, include_self=False)
            retrieved_classes = results['class'].tolist()
            
            # Calculate precision@k
            correct = sum(1 for c in retrieved_classes if c == query_class)
            precision_at_k = correct / k
            precisions.append(precision_at_k)
        
        return {
            'mean_precision@k': np.mean(precisions),
            'std_precision@k': np.std(precisions),
            'n_queries': len(query_indices),
            'k': k
        }


# Test the final approach
if __name__ == "__main__":
    print("="*80)
    print("TESTING FINAL APPROACH: RAW DISTANCES + INDIVIDUAL WEIGHTS")
    print("="*80)
    
    print("This approach:")
    print("✅ Uses raw EMD + Euclidean distances")
    print("✅ Simple [0,1] min-max normalization") 
    print("✅ Individual weights per descriptor")
    print("✅ No negative distance cancellation")
    print("✅ Ready for weight optimization")
    print()
    
    # Test with 500 shapes first
    print("Testing with 500 shapes...")
    
    try:
        query_system = FinalShapeQuery(
            num_shapes=500,
            debug=True
        )
        
        print("\\nEvaluating with equal weights...")
        metrics = query_system.evaluate_retrieval(n_queries=50, k=30)
        
        print("\\n" + "="*60)
        print("FINAL APPROACH RESULTS (500 shapes):")
        print("="*60)
        print(f"Precision@30: {metrics['mean_precision@k']:.3f} ± {metrics['std_precision@k']:.3f}")
        print(f"Queries tested: {metrics['n_queries']}")
        
        # Show sample distances
        print(f"\\nSample distance matrix (surface_area, first 3x3):")
        sample = query_system.distance_matrices['surface_area'][:3, :3]
        print(sample.round(4))
        print(f"All values in [0,1]: {np.min(sample):.3f} to {np.max(sample):.3f}")
        
        if metrics['mean_precision@k'] >= 0.25:
            print("\\n🎉 EXCELLENT: Ready for optimization!")
        elif metrics['mean_precision@k'] >= 0.20:
            print("\\n📈 GOOD: Solid baseline for optimization")
        else:
            print("\\n📊 BASELINE: Shows approach viability")
        
        print(f"\\nNext step: Optimize individual descriptor weights with Bayesian optimization!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\\n" + "="*80)