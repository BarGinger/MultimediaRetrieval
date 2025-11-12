"""
Fixed Z-Score ShapeQuery: Correct statistical standardization without absolute values.

This approach:
1. Uses raw EMD and Euclidean distances (no broken abs() on z-scores)
2. Applies proper z-score standardization: (distance - mean) / std
3. Allows negative z-scores (indicating "closer than average" pairs)
4. Uses individual weights for each descriptor
5. Lets optimization discover discriminative descriptors
"""

import os
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import euclidean_distances


class CorrectedZScoreShapeQuery:
    
    def __init__(self, 
                 csv_file_path: str = "final_006_cleaned.csv",
                 cache_dir: str = "distance_matrices_zscore_corrected",
                 weights: dict = None,
                 num_shapes: int = None,
                 combination_method: str = "weighted_sum",
                 debug: bool = False):
        """
        Initialize the corrected z-score shape query system.
        
        Args:
            csv_file_path: Path to CSV containing shape features
            cache_dir: Directory to cache z-score standardized distance matrices
            weights: Dictionary with individual descriptor weights (must sum to 1.0)
            num_shapes: Limit to first N shapes (default: all)
            combination_method: "weighted_sum" or "feature_space" for final distance computation
            debug: If True, print detailed information
        """
        self.debug = debug
        self.cache_dir = cache_dir
        self.csv_file_path = csv_file_path
        self.num_shapes = num_shapes
        self.combination_method = combination_method
        
        # Default equal weights for all 11 descriptors
        if weights is None:
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
        
        self.shape_names = self.features_df['shape'].tolist()
        self.shape_classes = self.features_df['class'].tolist()
        
        # Determine available features
        self.available_global = [f for f in self.global_features if f in self.features_df.columns]
        self.available_histograms = [f for f in self.histogram_features if f in self.features_df.columns]
        
        if self.debug:
            print(f"Dataset: {len(self.shape_names)} shapes, {len(set(self.shape_classes))} classes")
            print(f"Global features ({len(self.available_global)}): {self.available_global}")
            print(f"Histogram features ({len(self.available_histograms)}): {self.available_histograms}")
    
    def _load_or_compute_matrices(self):
        """Load or compute all z-score standardized distance matrices."""
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Also create a raw distances subdirectory
        raw_cache_dir = os.path.join(self.cache_dir, "raw_distances")
        os.makedirs(raw_cache_dir, exist_ok=True)
        
        if self.debug:
            print(f"\\nLoading/computing z-score standardized distance matrices...")
        
        # Process each descriptor
        all_descriptors = self.available_global + self.available_histograms
        
        for descriptor in all_descriptors:
            cache_file = os.path.join(self.cache_dir, f"{descriptor}_zscore.csv")
            raw_cache_file = os.path.join(raw_cache_dir, f"{descriptor}_raw.csv")
            
            if os.path.exists(cache_file):
                if self.debug:
                    print(f"Loading cached: {descriptor}")
                distance_matrix = pd.read_csv(cache_file, index_col=0).values
            else:
                if self.debug:
                    print(f"Computing: {descriptor}")
                
                # Compute raw distances first
                if descriptor in self.available_global:
                    # Euclidean distance for global features
                    raw_matrix = self._compute_euclidean_matrix(descriptor)
                else:
                    # EMD for histogram features
                    raw_matrix = self._compute_emd_matrix(descriptor)
                
                # Save raw distances to CSV for future experimentation
                df_raw = pd.DataFrame(raw_matrix, index=self.shape_names, columns=self.shape_names)
                df_raw.to_csv(raw_cache_file)
                if self.debug:
                    print(f"  Raw distances saved to: {os.path.basename(raw_cache_file)}")
                
                # Apply z-score standardization (NO .abs()!)
                distance_matrix = self._zscore_standardize_matrix(raw_matrix, descriptor)
                
                # Save z-score normalized distances
                df_cache = pd.DataFrame(distance_matrix, index=self.shape_names, columns=self.shape_names)
                df_cache.to_csv(cache_file)
                if self.debug:
                    print(f"  Z-score normalized saved to: {os.path.basename(cache_file)}")
            
            self.distance_matrices[descriptor] = distance_matrix
        
        if self.debug:
            print(f"All {len(all_descriptors)} z-score distance matrices ready.\\n")
    
    def _compute_euclidean_matrix(self, feature_name):
        """Compute Euclidean distance matrix for a global feature."""
        values = self.features_df[feature_name].values.reshape(-1, 1)
        
        # Handle NaN values
        valid_mask = ~np.isnan(values.flatten())
        
        if not valid_mask.any():
            return np.full((len(self.shape_names), len(self.shape_names)), np.nan)
        
        # Compute pairwise Euclidean distances
        distance_matrix = euclidean_distances(values, values)
        
        # Set invalid entries to NaN
        for i, valid_i in enumerate(valid_mask):
            for j, valid_j in enumerate(valid_mask):
                if not (valid_i and valid_j):
                    distance_matrix[i, j] = np.nan
        
        return distance_matrix
    
    def _compute_emd_matrix(self, feature_name):
        """Compute EMD distance matrix for a histogram feature."""
        n = len(self.shape_names)
        distance_matrix = np.zeros((n, n))
        
        # Get bins column name
        bins_col = feature_name.replace('_hist', '_bins')
        
        # Get bin centers (same for all shapes)
        first_shape_row = self.features_df.iloc[0]
        bins_str = first_shape_row[bins_col]
        bin_edges = np.array([float(x) for x in bins_str.split(';')])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Extract histograms
        histograms = []
        for shape_name in self.shape_names:
            shape_row = self.features_df[self.features_df['shape'] == shape_name].iloc[0]
            hist_str = shape_row[feature_name]
            
            if pd.isna(hist_str):
                histograms.append(None)
            else:
                try:
                    # Parse histogram string to array (semicolon-separated)
                    hist_values = [float(x) for x in hist_str.split(';')]
                    hist_array = np.array(hist_values)
                    # Histograms are already frequency histograms - no need to renormalize
                    histograms.append(hist_array)
                except:
                    histograms.append(None)
        
        # Compute pairwise EMD
        for i in range(n):
            for j in range(n):
                if i == j:
                    distance_matrix[i, j] = 0.0
                elif histograms[i] is None or histograms[j] is None:
                    distance_matrix[i, j] = np.nan
                else:
                    try:
                        # Use Wasserstein distance (1D EMD) with proper bin centers
                        distance = wasserstein_distance(
                            bin_centers, bin_centers,
                            histograms[i], histograms[j]
                        )
                        distance_matrix[i, j] = distance
                    except:
                        distance_matrix[i, j] = np.nan
        
        return distance_matrix
    
    def _zscore_standardize_matrix(self, raw_matrix, descriptor):
        # Get valid (non-NaN, non-diagonal) distances
        n = raw_matrix.shape[0]
        valid_distances = []
        
        for i in range(n):
            for j in range(n):
                if i != j and not np.isnan(raw_matrix[i, j]):
                    valid_distances.append(raw_matrix[i, j])
        
        if len(valid_distances) == 0:
            if self.debug:
                print(f"  Warning: No valid distances for {descriptor}")
            return raw_matrix
        
        # Compute statistics
        mean = np.mean(valid_distances)
        std = np.std(valid_distances)
        
        if std < 1e-12:
            std = 1.0  # Avoid division by zero
        
        # Apply z-score: (x - mean) / std
        zscore_matrix = (raw_matrix - mean) / std
        
        if self.debug:
            # Count negative z-scores
            valid_z = zscore_matrix[~np.isnan(zscore_matrix)]
            negative_count = np.sum(valid_z < 0)
            print(f"  {descriptor}: mean={mean:.4f}, std={std:.4f}, {negative_count}/{len(valid_z)} negative z-scores")
        
        return zscore_matrix
    
    def _save_combined_matrix(self):
        """Save the combined distance matrix to CSV."""
        if hasattr(self, 'combined_distance_matrix') and self.combined_distance_matrix is not None:
            combined_path = os.path.join(self.cache_dir, f"combined_distance_matrix_{self.combination_method}.csv")
            df_combined = pd.DataFrame(
                self.combined_distance_matrix, 
                index=self.shape_names, 
                columns=self.shape_names
            )
            df_combined.to_csv(combined_path)
            if self.debug:
                print(f"Combined matrix saved to: {combined_path}")
    
    def _compute_combined_matrix(self):
        """Compute final distance matrix using specified combination method."""
        if self.combination_method == "weighted_sum":
            self._compute_weighted_sum_matrix()
        elif self.combination_method == "feature_space":
            self._compute_feature_space_matrix()
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
    
    def _compute_weighted_sum_matrix(self):
        """Compute weighted combination of z-score standardized distance matrices."""
        if self.debug:
            print("Computing weighted combination of z-score distances...")
        
        n = len(self.shape_names)
        combined_matrix = np.zeros((n, n))
        weight_sum = 0.0
        
        for descriptor, weight in self.weights.items():
            if descriptor in self.distance_matrices:
                combined_matrix += weight * self.distance_matrices[descriptor]
                weight_sum += weight
                if self.debug:
                    print(f"  {descriptor}: weight={weight:.4f}")
        
        if weight_sum > 0:
            combined_matrix /= weight_sum
        
        self.combined_distance_matrix = combined_matrix
        self._save_combined_matrix()
        
        if self.debug:
            print(f"Combined matrix computed (weight_sum={weight_sum:.4f})\\n")
    
    def _compute_feature_space_matrix(self):
        """Compute Euclidean distances in z-score feature space."""
        if self.debug:
            print("Computing distances in z-score feature space...")
        
        n = len(self.shape_names)
        combined_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    combined_matrix[i, j] = 0.0
                else:
                    # Build feature vector from all z-scores
                    feature_vector = []
                    for descriptor in self.weights.keys():  # Use same descriptors as weights
                        if descriptor in self.distance_matrices:
                            zscore = self.distance_matrices[descriptor][i, j]
                            feature_vector.append(zscore)
                    
                    # Compute Euclidean distance
                    combined_matrix[i, j] = np.linalg.norm(feature_vector)
        
        self.combined_distance_matrix = combined_matrix
        self._save_combined_matrix()
        
        if self.debug:
            print(f"Feature space distance matrix computed ({n}x{n})\\n")
    
    def update_weights(self, new_weights):
        """Update weights and recompute combined matrix (fast)."""
        self.weights = new_weights.copy()
        self._compute_combined_matrix()
    
    def query(self, query_shape: str, k: int = 30, include_self: bool = False):
        """Find k nearest shapes to query using z-score standardized distances."""
        if query_shape not in self.shape_names:
            raise ValueError(f"Query shape '{query_shape}' not found in dataset")
        
        query_idx = self.shape_names.index(query_shape)
        distances = self.combined_distance_matrix[query_idx, :]
        
        # Create list of (distance, index) pairs
        distance_pairs = [(distances[i], i) for i in range(len(distances))]
        
        # Remove self if not included
        if not include_self:
            distance_pairs = [(d, i) for d, i in distance_pairs if i != query_idx]
        
        # Sort by distance (ascending - smaller z-scores are closer)
        distance_pairs.sort(key=lambda x: x[0])
        
        # Take top k
        top_k = distance_pairs[:k]
        
        # Build result dictionary
        results = {}
        for rank, (distance, idx) in enumerate(top_k, 1):
            shape_name = self.shape_names[idx]
            results[shape_name] = distance
        
        return results
    
    def evaluate_retrieval(self, n_queries: int = 100, k: int = 30) -> dict:
        """Evaluate retrieval performance using random queries."""
        import random
        
        # Select diverse query shapes
        unique_classes = list(set(self.shape_classes))
        query_indices = []
        
        # Handle None case: use comprehensive evaluation with multiple queries per class
        if n_queries is None:
            # Use 2 queries per class that has multiple shapes for robust evaluation
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
            
            # Count correct retrievals
            correct = 0
            for shape_name in results.keys():
                shape_idx = self.shape_names.index(shape_name)
                if self.shape_classes[shape_idx] == query_class:
                    correct += 1
            
            precision = correct / k if k > 0 else 0.0
            precisions.append(precision)
        
        return {
            'mean_precision@k': np.mean(precisions) if precisions else 0.0,
            'std_precision@k': np.std(precisions) if precisions else 0.0,
            'n_queries': len(precisions)
        }


# Example usage and testing
if __name__ == "__main__":
    print("="*80)
    print("Z-SCORE COMBINATION METHODS COMPARISON")
    print("="*80)
    print("Testing: Weighted sum vs. Feature space Euclidean distance")
    print("="*80 + "\\n")
    
    # Test both approaches on 500 shapes for comparison
    test_shapes = 500
    test_queries = 20
    
    print("="*50)
    print("METHOD 1: WEIGHTED SUM (Original)")
    print("="*50)
    
    shape_query_sum = CorrectedZScoreShapeQuery(
        csv_file_path="final_006_cleaned.csv",
        cache_dir="distance_matrices_zscore_corrected",
        num_shapes=test_shapes,
        combination_method="weighted_sum",
        debug=True
    )
    
    print(f"\\nEvaluating weighted sum approach...")
    sum_metrics = shape_query_sum.evaluate_retrieval(n_queries=test_queries, k=30)
    
    print(f"Weighted Sum Results:")
    print(f"  Precision@30: {sum_metrics['mean_precision@k']:.3f} ± {sum_metrics['std_precision@k']:.3f}")
    print(f"  Queries: {sum_metrics['n_queries']}")
    
    print("\\n" + "="*50)
    print("METHOD 2: FEATURE SPACE EUCLIDEAN")
    print("="*50)
    
    shape_query_euclidean = CorrectedZScoreShapeQuery(
        csv_file_path="final_006_cleaned.csv",
        cache_dir="distance_matrices_zscore_corrected",  # Reuse cached z-score matrices
        num_shapes=test_shapes,
        combination_method="feature_space",
        debug=True
    )
    
    print(f"\\nEvaluating feature space approach...")
    euclidean_metrics = shape_query_euclidean.evaluate_retrieval(n_queries=test_queries, k=30)
    
    print(f"Feature Space Results:")
    print(f"  Precision@30: {euclidean_metrics['mean_precision@k']:.3f} ± {euclidean_metrics['std_precision@k']:.3f}")
    print(f"  Queries: {euclidean_metrics['n_queries']}")
    
    print("\\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    sum_precision = sum_metrics['mean_precision@k']
    euclidean_precision = euclidean_metrics['mean_precision@k']
    improvement = ((euclidean_precision - sum_precision) / sum_precision) * 100 if sum_precision > 0 else 0
    
    print(f"Weighted Sum:     {sum_precision:.3f} ± {sum_metrics['std_precision@k']:.3f}")
    print(f"Feature Space:    {euclidean_precision:.3f} ± {euclidean_metrics['std_precision@k']:.3f}")
    print(f"Improvement:      {improvement:+.1f}%")
    
    if improvement > 0:
        print("\\nFeature space approach performs better!")
        best_method = "feature_space"
        print(f"   Winner: Feature Space (+{improvement:.1f}% improvement)")
    elif improvement < -5:
        print("\\nWeighted sum approach performs better.")
        best_method = "weighted_sum"
        print(f"   Winner: Weighted Sum ({-improvement:.1f}% better)")
    else:
        print("\\nPerformance is similar between approaches.")
        best_method = "feature_space"  # Default to new approach if similar
        print("   Winner: Feature Space (default choice)")
    
    print("\\n" + "="*80)
    print("FULL DATASET EVALUATION WITH WINNING APPROACH")
    print("="*80)
    
    print(f"\\nRunning {best_method} approach on full dataset...")
    
    full_shape_query = CorrectedZScoreShapeQuery(
        csv_file_path="final_006_cleaned.csv",
        cache_dir="distance_matrices_zscore_corrected_full",  # Separate cache for full dataset
        num_shapes=None,  # Full dataset
        combination_method=best_method,
        debug=True
    )
    
    print(f"Full dataset: {len(full_shape_query.shape_names)} shapes, {len(set(full_shape_query.shape_classes))} classes")
    
    print(f"\\nEvaluating {best_method} on full dataset...")
    full_metrics = full_shape_query.evaluate_retrieval(n_queries=100, k=30)
    
    print(f"\\nFinal Results ({best_method} on full dataset):")
    print(f"  Precision@30: {full_metrics['mean_precision@k']:.3f} ± {full_metrics['std_precision@k']:.3f}")
    print(f"  Queries: {full_metrics['n_queries']}")
    print(f"  Dataset size: {len(full_shape_query.shape_names)} shapes")
    
    print("\\n" + "="*80)
    print("COMPARISON TEST COMPLETE")
    print("="*80)