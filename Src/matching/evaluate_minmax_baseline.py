"""
Evaluation of the min-max normalized approach with optimized weights.
This corresponds to the final_shape_query.py approach that achieved the previous best results.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from final_shape_query import FinalShapeQuery
import pandas as pd
import json

def evaluate_minmax_optimized():
    """
    Evaluate the min-max normalized approach with optimized weights.
    Uses the same robust evaluation parameters as the corrected z-score approach.
    """
    print("="*80)
    print("MIN-MAX NORMALIZED APPROACH - ROBUST EVALUATION")
    print("="*80)
    print("Approach: Raw distances + [0,1] min-max normalization + optimized weights")
    print("Parameters: k=10, 1000 random queries, full dataset")
    print("="*80 + "\\n")
    
    # Load optimized weights
    weights_df = pd.read_csv("../../optimization_results_final/optimized_weights_final.csv")
    optimized_weights = {}
    for _, row in weights_df.iterrows():
        optimized_weights[row['descriptor']] = row['weight']
    
    print("Optimized weights loaded:")
    for desc, weight in optimized_weights.items():
        print(f"  {desc}: {weight:.4f}")
    print()
    
    # Initialize with optimized weights
    shape_query = FinalShapeQuery(
        csv_file_path="final_006_cleaned.csv",
        cache_dir="../../distance_matrices_raw_normalized_full",
        weights=optimized_weights,
        num_shapes=None,  # Full dataset
        debug=True
    )
    
    print(f"Dataset: {len(shape_query.shape_names)} shapes, {len(set(shape_query.shape_classes))} classes")
    print(f"Average shapes per class: {len(shape_query.shape_names) / len(set(shape_query.shape_classes)):.1f}")
    
    print(f"\\nEvaluating with robust parameters...")
    print(f"  • k = 10 (precision@10)")
    print(f"  • 1000 random queries")
    print(f"  • Full dataset ({len(shape_query.shape_names)} shapes)")
    print(f"  • Optimized weights")
    
    # Evaluate with robust parameters
    metrics = shape_query.evaluate_retrieval(n_queries=1000, k=10)
    
    print(f"\\n" + "="*60)
    print("MIN-MAX NORMALIZED RESULTS (Optimized Weights)")
    print("="*60)
    print(f"Precision@10:     {metrics['mean_precision@k']:.3f} ± {metrics['std_precision@k']:.3f}")
    print(f"Queries:          {metrics['n_queries']}")
    print(f"Dataset size:     {len(shape_query.shape_names)} shapes")
    print(f"Classes:          {len(set(shape_query.shape_classes))}")
    
    # Calculate confidence interval (95%)
    std_error = metrics['std_precision@k'] / (metrics['n_queries'] ** 0.5)
    ci_95 = 1.96 * std_error
    
    print(f"\\nStatistical Analysis:")
    print(f"Standard Error:   {std_error:.4f}")
    print(f"95% Confidence:   [{metrics['mean_precision@k'] - ci_95:.3f}, {metrics['mean_precision@k'] + ci_95:.3f}]")
    
    print(f"\\n" + "="*80)
    print("BASELINE ESTABLISHED FOR MIN-MAX APPROACH")
    print("="*80)
    
    return metrics

if __name__ == "__main__":
    evaluate_minmax_optimized()