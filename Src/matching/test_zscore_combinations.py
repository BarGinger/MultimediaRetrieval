#!/usr/bin/env python3
"""
Test different z-score combination methods.

This script experiments with various ways to combine z-score standardized distances
beyond the current weighted sum approach, while keeping the original optimized
approach intact.
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import random

# Add the Src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from matching.corrected_zscore_shape_query import CorrectedZScoreShapeQuery

class ZScoreCombinationTester:
    def __init__(self, csv_file_path: str = "final_006_cleaned.csv", debug: bool = True):
        """Initialize with the corrected z-score base."""
        self.csv_file_path = csv_file_path
        self.debug = debug
        
        # Use EQUAL weights for fair comparison across methods
        # (Optimized weights were tuned specifically for weighted sum)
        self.equal_weights = {
            'compactness': 1/11,
            'convexity': 1/11, 
            'diameter': 1/11,
            'eccentricity': 1/11,
            'rectangularity': 1/11,
            'surface_area': 1/11,
            'A3_hist': 1/11,
            'D1_hist': 1/11,
            'D2_hist': 1/11,
            'D3_hist': 1/11,
            'D4_hist': 1/11
        }
        
        # Initialize base system to load z-score matrices
        self.base_system = CorrectedZScoreShapeQuery(
            csv_file_path=csv_file_path,
            cache_dir="distance_matrices_zscore_corrected_full",
            weights=self.equal_weights,  # Use equal weights for fair comparison
            combination_method="weighted_sum",
            debug=False  # Suppress debug for base loading
        )
        
        self.shape_names = self.base_system.shape_names
        self.distance_matrices = self.base_system.distance_matrices
        
        if self.debug:
            print(f"Loaded {len(self.distance_matrices)} z-score distance matrices")
            print(f"Matrix shape: {list(self.distance_matrices.values())[0].shape}")
    
    def _combine_squared_zscore(self, weights: Dict[str, float]) -> np.ndarray:
        """
        Combination method 1: Squared z-scores
        Use z^2 instead of |z| - emphasizes extreme deviations more.
        """
        n = len(self.shape_names)
        combined_matrix = np.zeros((n, n))
        weight_sum = 0.0
        
        for descriptor, weight in weights.items():
            if descriptor in self.distance_matrices:
                zscore_matrix = self.distance_matrices[descriptor]
                # Square the z-scores to emphasize extreme values
                squared_zscore = zscore_matrix ** 2
                combined_matrix += weight * squared_zscore
                weight_sum += weight
        
        if weight_sum > 0:
            combined_matrix /= weight_sum
            
        return combined_matrix
    
    def _combine_exponential_zscore(self, weights: Dict[str, float], scale: float = 0.5) -> np.ndarray:
        """
        Combination method 2: Exponential transformation
        Use exp(scale * |z|) to create exponential emphasis on large deviations.
        """
        n = len(self.shape_names)
        combined_matrix = np.zeros((n, n))
        weight_sum = 0.0
        
        for descriptor, weight in weights.items():
            if descriptor in self.distance_matrices:
                zscore_matrix = self.distance_matrices[descriptor]
                # Apply exponential transformation
                exp_zscore = np.exp(scale * np.abs(zscore_matrix))
                combined_matrix += weight * exp_zscore
                weight_sum += weight
        
        if weight_sum > 0:
            combined_matrix /= weight_sum
            
        return combined_matrix
    
    def _combine_sigmoid_zscore(self, weights: Dict[str, float], steepness: float = 1.0) -> np.ndarray:
        """
        Combination method 3: Sigmoid transformation
        Use sigmoid(steepness * z) to create smooth non-linear emphasis.
        """
        n = len(self.shape_names)
        combined_matrix = np.zeros((n, n))
        weight_sum = 0.0
        
        for descriptor, weight in weights.items():
            if descriptor in self.distance_matrices:
                zscore_matrix = self.distance_matrices[descriptor]
                # Apply sigmoid transformation
                sigmoid_zscore = 1 / (1 + np.exp(-steepness * zscore_matrix))
                combined_matrix += weight * sigmoid_zscore
                weight_sum += weight
        
        if weight_sum > 0:
            combined_matrix /= weight_sum
            
        return combined_matrix
    
    def _combine_rank_based(self, weights: Dict[str, float]) -> np.ndarray:
        """
        Combination method 4: Rank-based combination
        Convert z-scores to ranks within each descriptor, then combine ranks.
        """
        n = len(self.shape_names)
        combined_matrix = np.zeros((n, n))
        weight_sum = 0.0
        
        for descriptor, weight in weights.items():
            if descriptor in self.distance_matrices:
                zscore_matrix = self.distance_matrices[descriptor]
                
                # Convert to ranks (lower z-score = lower rank = closer)
                # Flatten, rank, reshape back
                flat_matrix = zscore_matrix.flatten()
                ranks = np.argsort(np.argsort(flat_matrix)).reshape(zscore_matrix.shape)
                # Normalize ranks to [0, 1]
                rank_matrix = ranks / (n * n - 1)
                
                combined_matrix += weight * rank_matrix
                weight_sum += weight
        
        if weight_sum > 0:
            combined_matrix /= weight_sum
            
        return combined_matrix
    
    def _combine_percentile_based(self, weights: Dict[str, float], percentile_threshold: float = 95) -> np.ndarray:
        """
        Combination method 5: Percentile-based combination
        Cap extreme z-scores at percentile threshold to reduce outlier impact.
        """
        n = len(self.shape_names)
        combined_matrix = np.zeros((n, n))
        weight_sum = 0.0
        
        for descriptor, weight in weights.items():
            if descriptor in self.distance_matrices:
                zscore_matrix = self.distance_matrices[descriptor]
                
                # Cap at percentile threshold
                valid_zscores = zscore_matrix[~np.isnan(zscore_matrix)]
                upper_cap = np.percentile(valid_zscores, percentile_threshold)
                lower_cap = np.percentile(valid_zscores, 100 - percentile_threshold)
                
                capped_matrix = np.clip(zscore_matrix, lower_cap, upper_cap)
                combined_matrix += weight * capped_matrix
                weight_sum += weight
        
        if weight_sum > 0:
            combined_matrix /= weight_sum
            
        return combined_matrix
    
    def evaluate_combination_method(self, combined_matrix: np.ndarray, method_name: str, num_queries: int = None) -> Dict:
        """Evaluate a combination method using all shapes or random sample."""
        if self.debug:
            print(f"\\nEvaluating {method_name}...")
        
        # Use all shapes if num_queries is None or >= total shapes
        if num_queries is None or num_queries >= len(self.shape_names):
            query_shapes = self.shape_names
            use_all = True
        else:
            # Random sampling for evaluation
            random.seed(42)  # For reproducibility
            query_shapes = random.sample(self.shape_names, num_queries)
            use_all = False
        
        if self.debug and use_all:
            print(f"  Using ALL {len(query_shapes)} shapes as queries for definitive results")
        
        total_precision = 0.0
        valid_queries = 0
        
        # Load ground truth
        features_df = pd.read_csv(self.csv_file_path)
        
        for query_shape in query_shapes:
            try:
                query_idx = self.shape_names.index(query_shape)
                query_class = features_df[features_df['shape'] == query_shape]['class'].iloc[0]
                
                # Get distances for this query
                distances = combined_matrix[query_idx, :]
                
                # Sort by distance (ascending)
                sorted_indices = np.argsort(distances)
                
                # Exclude self and get top k=10
                k = 10
                results = []
                for idx in sorted_indices[1:]:  # Skip self (index 0)
                    if len(results) >= k:
                        break
                    if not np.isnan(distances[idx]):
                        results.append(self.shape_names[idx])
                
                # Calculate precision@10
                relevant_count = 0
                for result_shape in results:
                    result_class = features_df[features_df['shape'] == result_shape]['class'].iloc[0]
                    if result_class == query_class:
                        relevant_count += 1
                
                precision = relevant_count / len(results) if results else 0.0
                total_precision += precision
                valid_queries += 1
                
            except Exception as e:
                if self.debug:
                    print(f"Error with query {query_shape}: {e}")
                continue
        
        avg_precision = total_precision / valid_queries if valid_queries > 0 else 0.0
        
        return {
            'method': method_name,
            'precision_at_10': avg_precision,
            'valid_queries': valid_queries,
            'total_queries': len(query_shapes),
            'used_all_shapes': use_all
        }
    
    def test_all_combinations(self, use_all_queries: bool = True):
        """Test all combination methods with definitive evaluation."""
        if use_all_queries:
            print("Testing different z-score combination methods...")
            print(f"Using ALL {len(self.shape_names)} shapes as queries for definitive results...")
            print("(No sampling variance - pure performance comparison)")
            num_queries = None  # Signal to use all shapes
        else:
            print("Testing different z-score combination methods...")
            print("Using 500 query sample for faster testing...")
            num_queries = 500
            
        print("=" * 60)
        
        results = []
        
        # Method 1: Original weighted sum (baseline)
        baseline_matrix = self.base_system.combined_distance_matrix
        baseline_result = self.evaluate_combination_method(baseline_matrix, "Weighted Sum (Baseline)", num_queries)
        results.append(baseline_result)
        
        # Method 2: Squared z-scores
        squared_matrix = self._combine_squared_zscore(self.equal_weights)
        squared_result = self.evaluate_combination_method(squared_matrix, "Squared Z-Scores", num_queries)
        results.append(squared_result)
        
        # Method 3: Exponential transformation (scale=0.3)
        exp_matrix = self._combine_exponential_zscore(self.equal_weights, scale=0.3)
        exp_result = self.evaluate_combination_method(exp_matrix, "Exponential (scale=0.3)", num_queries)
        results.append(exp_result)
        
        # Method 4: Exponential transformation (scale=0.5)
        exp_matrix_2 = self._combine_exponential_zscore(self.equal_weights, scale=0.5)
        exp_result_2 = self.evaluate_combination_method(exp_matrix_2, "Exponential (scale=0.5)", num_queries)
        results.append(exp_result_2)
        
        # Method 5: Sigmoid transformation (steepness=1.0)
        sigmoid_matrix = self._combine_sigmoid_zscore(self.equal_weights, steepness=1.0)
        sigmoid_result = self.evaluate_combination_method(sigmoid_matrix, "Sigmoid (steepness=1.0)", num_queries)
        results.append(sigmoid_result)
        
        # Method 6: Sigmoid transformation (steepness=2.0)
        sigmoid_matrix_2 = self._combine_sigmoid_zscore(self.equal_weights, steepness=2.0)
        sigmoid_result_2 = self.evaluate_combination_method(sigmoid_matrix_2, "Sigmoid (steepness=2.0)", num_queries)
        results.append(sigmoid_result_2)
        
        # Method 7: Rank-based combination
        rank_matrix = self._combine_rank_based(self.equal_weights)
        rank_result = self.evaluate_combination_method(rank_matrix, "Rank-Based", num_queries)
        results.append(rank_result)
        
        # Method 8: Percentile-based (95th percentile cap)
        percentile_matrix = self._combine_percentile_based(self.equal_weights, percentile_threshold=95)
        percentile_result = self.evaluate_combination_method(percentile_matrix, "Percentile Cap (95%)", num_queries)
        results.append(percentile_result)
        
        # Method 9: Percentile-based (90th percentile cap)
        percentile_matrix_2 = self._combine_percentile_based(self.equal_weights, percentile_threshold=90)
        percentile_result_2 = self.evaluate_combination_method(percentile_matrix_2, "Percentile Cap (90%)", num_queries)
        results.append(percentile_result_2)
        
        # Display results
        print("\\nResults Summary:")
        print("=" * 60)
        print(f"{'Method':<25} {'Precision@10':<12} {'Valid Queries':<12}")
        print("-" * 60)
        
        # Sort by precision
        results.sort(key=lambda x: x['precision_at_10'], reverse=True)
        
        for result in results:
            precision_pct = result['precision_at_10'] * 100
            print(f"{result['method']:<25} {precision_pct:>8.1f}%     {result['valid_queries']:>8}/{result['total_queries']}")
        
        print("\\n" + "=" * 60)
        
        # Find best method
        best_method = results[0]
        baseline_precision = next(r['precision_at_10'] for r in results if r['method'] == "Weighted Sum (Baseline)")
        
        print("\\nStatistical Analysis:")
        print("-" * 40)
        
        if results[0]['used_all_shapes']:
            print(f"Sample size: ALL {results[0]['total_queries']} shapes (complete dataset)")
            print("No sampling variance - these are definitive results!")
        else:
            print(f"Sample size: {results[0]['total_queries']} queries")
            print(f"Expected std error: ~{np.sqrt(0.25 * 0.75 / results[0]['total_queries']):.3f} (assuming p≈0.25)")
        
        if best_method['method'] != "Weighted Sum (Baseline)":
            improvement = ((best_method['precision_at_10'] - baseline_precision) / baseline_precision) * 100
            diff = best_method['precision_at_10'] - baseline_precision
            
            print(f"🎯 Best method: {best_method['method']}")
            print(f"   Precision@10: {best_method['precision_at_10']*100:.1f}% (+{improvement:+.1f}% vs baseline)")
            print(f"   Absolute difference: {diff:.3f}")
            
            if results[0]['used_all_shapes']:
                print(f"   ✅ DEFINITIVE RESULT (using complete dataset)")
                if diff > 0.005:  # 0.5 percentage point threshold
                    print(f"   🎉 MEANINGFUL IMPROVEMENT (>{0.5}% absolute difference)")
                else:
                    print(f"   📊 MARGINAL IMPROVEMENT (<{0.5}% absolute difference)")
            else:
                # Rough significance test (assuming binomial distribution)
                margin_of_error = 1.96 * np.sqrt(0.25 * 0.75 / results[0]['total_queries'])  # 95% confidence
                print(f"   95% margin of error: ±{margin_of_error:.3f}")
                
                if diff > 2 * margin_of_error:
                    print(f"   ✅ STATISTICALLY SIGNIFICANT (difference > 2× margin of error)")
                elif diff > margin_of_error:
                    print(f"   ⚠️  POSSIBLY SIGNIFICANT (difference > margin of error)")
                else:
                    print(f"   ❌ NOT STATISTICALLY SIGNIFICANT (difference < margin of error)")
        else:
            print(f"✅ Baseline method remains best: {baseline_precision*100:.1f}%")
        
        return results

def main():
    """Main execution function."""
    print("Z-Score Combination Method Testing")
    print("=" * 50)
    
    # Initialize tester
    tester = ZScoreCombinationTester(debug=True)
    
    # Run all tests
    results = tester.test_all_combinations(use_all_queries=True)  # Use entire dataset for definitive results
    
    print("\\nTesting completed!")

if __name__ == "__main__":
    main()