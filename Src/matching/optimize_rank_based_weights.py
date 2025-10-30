#!/usr/bin/env python3
"""
Optimize weights for the rank-based z-score combination method.

Since rank-based showed superior performance with equal weights (27.8% vs 26.8%),
this script optimizes weights specifically for the rank-based approach to maximize
performance further.
"""

import sys
import os
import numpy as np
import pandas as pd
import optuna
from typing import Dict, List
import random
from datetime import datetime

# Add the Src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from matching.corrected_zscore_shape_query import CorrectedZScoreShapeQuery

class RankBasedOptimizer:
    def __init__(self, csv_file_path: str = "final_006_cleaned.csv", debug: bool = True):
        """Initialize rank-based weight optimizer."""
        self.csv_file_path = csv_file_path
        self.debug = debug
        
        # Load z-score distance matrices
        print("Loading z-score distance matrices...")
        equal_weights = {desc: 1/11 for desc in [
            'compactness', 'convexity', 'diameter', 'eccentricity', 'rectangularity',
            'surface_area', 'A3_hist', 'D1_hist', 'D2_hist', 'D3_hist', 'D4_hist'
        ]}
        
        self.base_system = CorrectedZScoreShapeQuery(
            csv_file_path=csv_file_path,
            cache_dir="distance_matrices_zscore_corrected_full",
            weights=equal_weights,
            combination_method="weighted_sum",
            debug=False
        )
        
        self.shape_names = self.base_system.shape_names
        self.distance_matrices = self.base_system.distance_matrices
        self.features_df = pd.read_csv(csv_file_path)
        
        # PRE-COMPUTE rank matrices for ALL descriptors (expensive operation)
        print("Pre-computing rank matrices for optimization speed...")
        self.rank_matrices = self._precompute_rank_matrices()
        
        # Pre-compute evaluation queries for consistency
        random.seed(42)
        self.evaluation_queries = random.sample(self.shape_names, 500)  # Fixed set for optimization
        
        print(f"Loaded {len(self.distance_matrices)} z-score matrices")
        print(f"Pre-computed {len(self.rank_matrices)} rank matrices")
        print(f"Using {len(self.evaluation_queries)} queries for optimization")
    
    def _precompute_rank_matrices(self) -> Dict[str, np.ndarray]:
        """Pre-compute rank matrices for all descriptors to speed up optimization."""
        rank_matrices = {}
        n = len(self.shape_names)
        
        for descriptor in self.distance_matrices.keys():
            if self.debug:
                print(f"  Computing ranks for {descriptor}...")
            
            zscore_matrix = self.distance_matrices[descriptor]
            
            # Convert to ranks (lower z-score = lower rank = closer)
            flat_matrix = zscore_matrix.flatten()
            ranks = np.argsort(np.argsort(flat_matrix)).reshape(zscore_matrix.shape)
            # Normalize ranks to [0, 1]
            rank_matrix = ranks / (n * n - 1)
            
            rank_matrices[descriptor] = rank_matrix
        
        return rank_matrices
    
    def _combine_rank_based_fast(self, weights: Dict[str, float]) -> np.ndarray:
        """
        Fast rank-based combination using pre-computed rank matrices.
        """
        n = len(self.shape_names)
        combined_matrix = np.zeros((n, n))
        weight_sum = 0.0
        
        for descriptor, weight in weights.items():
            if descriptor in self.rank_matrices:
                rank_matrix = self.rank_matrices[descriptor]
                combined_matrix += weight * rank_matrix
                weight_sum += weight
        
        if weight_sum > 0:
            combined_matrix /= weight_sum
            
        return combined_matrix
    
    def _combine_rank_based(self, weights: Dict[str, float]) -> np.ndarray:
        """
        Original rank-based combination (for final evaluation when not optimizing).
        """
        n = len(self.shape_names)
        combined_matrix = np.zeros((n, n))
        weight_sum = 0.0
        
        for descriptor, weight in weights.items():
            if descriptor in self.distance_matrices:
                zscore_matrix = self.distance_matrices[descriptor]
                
                # Convert to ranks (lower z-score = lower rank = closer)
                flat_matrix = zscore_matrix.flatten()
                ranks = np.argsort(np.argsort(flat_matrix)).reshape(zscore_matrix.shape)
                # Normalize ranks to [0, 1]
                rank_matrix = ranks / (n * n - 1)
                
                combined_matrix += weight * rank_matrix
                weight_sum += weight
        
        if weight_sum > 0:
            combined_matrix /= weight_sum
            
        return combined_matrix
    
    def evaluate_weights(self, weights: Dict[str, float]) -> float:
        """Evaluate rank-based method with given weights using fixed query set."""
        
        # Create combined matrix using pre-computed ranks (FAST)
        combined_matrix = self._combine_rank_based_fast(weights)
        
        total_precision = 0.0
        valid_queries = 0
        
        for query_shape in self.evaluation_queries:
            try:
                query_idx = self.shape_names.index(query_shape)
                query_class = self.features_df[self.features_df['shape'] == query_shape]['class'].iloc[0]
                
                # Get distances for this query
                distances = combined_matrix[query_idx, :]
                
                # Sort by distance (ascending)
                sorted_indices = np.argsort(distances)
                
                # Exclude self and get top k=10
                k = 10
                results = []
                for idx in sorted_indices[1:]:  # Skip self
                    if len(results) >= k:
                        break
                    if not np.isnan(distances[idx]):
                        results.append(self.shape_names[idx])
                
                # Calculate precision@10
                relevant_count = 0
                for result_shape in results:
                    result_class = self.features_df[self.features_df['shape'] == result_shape]['class'].iloc[0]
                    if result_class == query_class:
                        relevant_count += 1
                
                precision = relevant_count / len(results) if results else 0.0
                total_precision += precision
                valid_queries += 1
                
            except Exception:
                continue
        
        avg_precision = total_precision / valid_queries if valid_queries > 0 else 0.0
        return avg_precision
    
    def objective_rank_based(self, trial) -> float:
        """Optuna objective function for rank-based weight optimization."""
        
        # Sample individual weights for each descriptor
        weights = {
            'compactness': trial.suggest_float('compactness', 0.001, 0.3),
            'convexity': trial.suggest_float('convexity', 0.001, 0.3),
            'diameter': trial.suggest_float('diameter', 0.001, 0.3),
            'eccentricity': trial.suggest_float('eccentricity', 0.001, 0.3),
            'rectangularity': trial.suggest_float('rectangularity', 0.001, 0.3),
            'surface_area': trial.suggest_float('surface_area', 0.001, 0.3),
            'A3_hist': trial.suggest_float('A3_hist', 0.001, 0.3),
            'D1_hist': trial.suggest_float('D1_hist', 0.001, 0.3),
            'D2_hist': trial.suggest_float('D2_hist', 0.001, 0.3),
            'D3_hist': trial.suggest_float('D3_hist', 0.001, 0.3),
            'D4_hist': trial.suggest_float('D4_hist', 0.001, 0.3)
        }
        
        # Evaluate performance
        precision = self.evaluate_weights(weights)
        
        if trial.number % 10 == 0:
            print(f"Trial {trial.number}: Precision@10 = {precision:.3f}")
        
        return precision
    
    def optimize_weights(self, n_trials: int = 200):
        """Run Bayesian optimization to find best weights for rank-based method."""
        
        print("=" * 70)
        print("RANK-BASED WEIGHT OPTIMIZATION")
        print("=" * 70)
        print(f"Method: Rank-based z-score combination")
        print(f"Optimization trials: {n_trials}")
        print(f"Evaluation queries: {len(self.evaluation_queries)}")
        print(f"Target: Beat equal weights baseline (27.8% precision@10)")
        print()
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=10)
        )
        
        # Run optimization
        print("Starting optimization...")
        study.optimize(self.objective_rank_based, n_trials=n_trials)
        
        # Get best results
        best_trial = study.best_trial
        best_weights = best_trial.params
        best_precision = best_trial.value
        
        print("\\n" + "=" * 70)
        print("OPTIMIZATION COMPLETED")
        print("=" * 70)
        print(f"Best precision@10: {best_precision:.3f} ({best_precision*100:.1f}%)")
        print(f"Best trial number: {best_trial.number}")
        
        print("\\nOptimized weights:")
        for descriptor, weight in best_weights.items():
            print(f"  {descriptor:15s}: {weight:.4f}")
        
        # Final evaluation with larger sample
        print("\n" + "=" * 50)
        print("FINAL EVALUATION (entire dataset)")
        print("=" * 50)
        
        # Evaluate with entire dataset
        final_queries = self.shape_names  # Use ALL shapes as queries
        
        combined_matrix = self._combine_rank_based(best_weights)
        total_precision = 0.0
        valid_queries = 0
        
        for query_shape in final_queries:
            try:
                query_idx = self.shape_names.index(query_shape)
                query_class = self.features_df[self.features_df['shape'] == query_shape]['class'].iloc[0]
                
                distances = combined_matrix[query_idx, :]
                sorted_indices = np.argsort(distances)
                
                k = 10
                results = []
                for idx in sorted_indices[1:]:
                    if len(results) >= k:
                        break
                    if not np.isnan(distances[idx]):
                        results.append(self.shape_names[idx])
                
                relevant_count = 0
                for result_shape in results:
                    result_class = self.features_df[self.features_df['shape'] == result_shape]['class'].iloc[0]
                    if result_class == query_class:
                        relevant_count += 1
                
                precision = relevant_count / len(results) if results else 0.0
                total_precision += precision
                valid_queries += 1
                
            except Exception:
                continue
        
        final_precision = total_precision / valid_queries if valid_queries > 0 else 0.0
        
        print(f"Final precision@10: {final_precision:.3f} ({final_precision*100:.1f}%)")
        print(f"Valid queries: {valid_queries}/{len(final_queries)}")
        
        # Compare with baselines
        equal_weights_precision = 0.278  # From previous test
        weighted_sum_precision = 0.268   # From previous test
        
        print("\\nComparison with baselines:")
        print(f"  Rank-based (optimized):  {final_precision*100:.1f}%")
        print(f"  Rank-based (equal):      {equal_weights_precision*100:.1f}%")
        print(f"  Weighted sum (optimized): ??.?% (our previous best: 28.1%)")
        print(f"  Weighted sum (equal):     {weighted_sum_precision*100:.1f}%")
        
        improvement_vs_equal = ((final_precision - equal_weights_precision) / equal_weights_precision) * 100
        print(f"\\nImprovement vs rank-based equal weights: {improvement_vs_equal:+.1f}%")
        
        # Save results
        results_dir = "optimization_results_rank_based"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save optimized weights
        weights_df = pd.DataFrame([best_weights])
        weights_file = os.path.join(results_dir, "optimized_weights_rank_based.csv")
        weights_df.to_csv(weights_file, index=False)
        
        # Save the combined distance matrix from best weights
        print("Saving optimized rank-based combined distance matrix...")
        best_combined_matrix = self._combine_rank_based(best_weights)
        combined_df = pd.DataFrame(
            best_combined_matrix,
            index=self.shape_names,
            columns=self.shape_names
        )
        combined_file = os.path.join(results_dir, "combined_distance_matrix_rank_based_optimized.csv")
        combined_df.to_csv(combined_file)
        print(f"Combined matrix saved: {combined_file}")
        
        # Also save to main cache directory for easy access
        main_cache_combined = "distance_matrices_zscore_corrected_full/combined_distance_matrix_rank_based_optimized.csv"
        combined_df.to_csv(main_cache_combined)
        print(f"Combined matrix also saved: {main_cache_combined}")
        
        # Save evaluation results
        results_data = {
            'optimization_date': datetime.now().isoformat(),
            'n_trials': n_trials,
            'best_trial_number': best_trial.number,
            'optimization_precision': best_precision,
            'final_evaluation_precision': final_precision,
            'evaluation_queries': len(final_queries),
            'improvement_vs_equal_weights_pct': improvement_vs_equal,
            'optimized_weights': best_weights
        }
        
        results_file = os.path.join(results_dir, "optimization_results_rank_based.json")
        import json
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\\nResults saved to: {results_dir}/")
        print(f"  - {weights_file}")
        print(f"  - {combined_file}")
        print(f"  - {results_file}")
        print(f"\\nCombined matrix also available at:")
        print(f"  - {main_cache_combined}")
        
        return best_weights, final_precision

def main():
    """Main execution function."""
    print("Rank-Based Z-Score Combination Weight Optimization")
    print("=" * 60)
    
    # Initialize optimizer
    optimizer = RankBasedOptimizer(debug=True)
    
    # Run optimization
    best_weights, final_precision = optimizer.optimize_weights(n_trials=200)
    
    print("\\nOptimization completed!")

if __name__ == "__main__":
    main()