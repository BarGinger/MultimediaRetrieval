"""
Fair comparison: Both approaches with true 1000 random queries.
"""

import random
import numpy as np
from tqdm import tqdm

def evaluate_with_1000_queries(shape_query, k=10):
    """
    Evaluate with truly random 1000 queries, not class-stratified sampling.
    Handles both dict and DataFrame return types from query methods.
    """
    n_shapes = len(shape_query.shape_names)
    n_queries = min(1000, n_shapes - 1)  # Can't query more shapes than we have
    
    # Pure random sampling
    query_indices = random.sample(range(n_shapes), n_queries)
    
    precisions = []
    
    for query_idx in tqdm(query_indices, desc="Evaluating queries"):
        query_shape = shape_query.shape_names[query_idx]
        query_class = shape_query.shape_classes[query_idx]
        
        # Get top k retrievals
        results = shape_query.query(query_shape, k=k, include_self=False)
        
        # Handle different return types
        if isinstance(results, dict):
            # CorrectedZScoreShapeQuery returns dict {shape_name: distance}
            result_shapes = list(results.keys())
        else:
            # FinalShapeQuery returns DataFrame with 'shape' column
            result_shapes = results['shape'].tolist()
        
        # Count correct retrievals
        correct = 0
        for shape_name in result_shapes:
            shape_idx = shape_query.shape_names.index(shape_name)
            if shape_query.shape_classes[shape_idx] == query_class:
                correct += 1
        
        precision = correct / k if k > 0 else 0.0
        precisions.append(precision)
    
    return {
        'mean_precision@k': np.mean(precisions) if precisions else 0.0,
        'std_precision@k': np.std(precisions) if precisions else 0.0,
        'n_queries': len(precisions)
    }

def compare_both_approaches():
    """Compare both approaches with fair 1000 query evaluation."""
    
    print("="*80)
    print("FAIR COMPARISON: 1000 RANDOM QUERIES")
    print("="*80)
    
    # Test 1: Corrected Z-Score (Equal Weights)
    print("\\n1. CORRECTED Z-SCORE (Equal Weights)")
    print("-" * 50)
    
    from corrected_zscore_shape_query import CorrectedZScoreShapeQuery
    
    zscore_query = CorrectedZScoreShapeQuery(
        csv_file_path="final_006_cleaned.csv",
        cache_dir="distance_matrices_zscore_corrected_full",
        num_shapes=None,
        combination_method="weighted_sum",
        debug=False
    )
    
    zscore_metrics = evaluate_with_1000_queries(zscore_query, k=10)
    
    print(f"Precision@10:     {zscore_metrics['mean_precision@k']:.3f} ± {zscore_metrics['std_precision@k']:.3f}")
    print(f"Queries:          {zscore_metrics['n_queries']}")
    
    # Test 2: Min-Max Normalized (Optimized Weights)
    print("\\n2. MIN-MAX NORMALIZED (Optimized Weights)")
    print("-" * 50)
    
    from final_shape_query import FinalShapeQuery
    import pandas as pd
    
    # Load optimized weights
    weights_df = pd.read_csv("../../optimization_results_final/optimized_weights_final.csv")
    optimized_weights = {}
    for _, row in weights_df.iterrows():
        optimized_weights[row['descriptor']] = row['weight']
    
    minmax_query = FinalShapeQuery(
        csv_file_path="final_006_cleaned.csv",
        cache_dir="../../distance_matrices_raw_normalized_full",
        weights=optimized_weights,
        num_shapes=None,
        debug=False
    )
    
    minmax_metrics = evaluate_with_1000_queries(minmax_query, k=10)
    
    print(f"Precision@10:     {minmax_metrics['mean_precision@k']:.3f} ± {minmax_metrics['std_precision@k']:.3f}")
    print(f"Queries:          {minmax_metrics['n_queries']}")
    
    # Comparison
    print("\\n" + "="*60)
    print("FAIR COMPARISON RESULTS")
    print("="*60)
    
    zscore_precision = zscore_metrics['mean_precision@k']
    minmax_precision = minmax_metrics['mean_precision@k']
    
    print(f"Z-Score (Equal):      {zscore_precision:.3f} ± {zscore_metrics['std_precision@k']:.3f}")
    print(f"Min-Max (Optimized):  {minmax_precision:.3f} ± {minmax_metrics['std_precision@k']:.3f}")
    
    if zscore_precision > minmax_precision:
        improvement = ((zscore_precision - minmax_precision) / minmax_precision) * 100
        print(f"\\nZ-Score approach wins! (+{improvement:.1f}% better)")
        print("The corrected z-score approach with EQUAL weights already")
        print("outperforms the min-max approach with OPTIMIZED weights!")
    else:
        gap = ((minmax_precision - zscore_precision) / minmax_precision) * 100
        print(f"\\nMin-Max approach leads by {gap:.1f}%")
        print("Weight optimization for Z-Score approach should close this gap")
    
    # Calculate confidence intervals
    zscore_se = zscore_metrics['std_precision@k'] / (zscore_metrics['n_queries'] ** 0.5)
    minmax_se = minmax_metrics['std_precision@k'] / (minmax_metrics['n_queries'] ** 0.5)
    
    print(f"\\nStatistical Analysis:")
    print(f"Z-Score 95% CI:   [{zscore_precision - 1.96*zscore_se:.3f}, {zscore_precision + 1.96*zscore_se:.3f}]")
    print(f"Min-Max 95% CI:   [{minmax_precision - 1.96*minmax_se:.3f}, {minmax_precision + 1.96*minmax_se:.3f}]")
    
    return zscore_metrics, minmax_metrics

if __name__ == "__main__":
    compare_both_approaches()