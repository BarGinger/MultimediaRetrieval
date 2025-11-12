
from corrected_zscore_shape_query import CorrectedZScoreShapeQuery

def evaluate_corrected_zscore_properly():
    """
    Evaluate the corrected z-score approach with robust evaluation parameters.
    """
    print("="*80)
    print("CORRECTED Z-SCORE APPROACH - PROPER EVALUATION")
    print("="*80)
    print("Parameters: k=10, 1000 random queries, full dataset")
    print("="*80 + "\\n")
    
    # Load the already computed approach (reuse cached matrices)
    shape_query = CorrectedZScoreShapeQuery(
        csv_file_path="final_006_cleaned.csv",
        cache_dir="distance_matrices_zscore_corrected_full",
        num_shapes=None,  # Full dataset
        combination_method="weighted_sum",  # Use the winner from comparison
        debug=True
    )
    
    print(f"Dataset: {len(shape_query.shape_names)} shapes, {len(set(shape_query.shape_classes))} classes")
    print(f"Average shapes per class: {len(shape_query.shape_names) / len(set(shape_query.shape_classes)):.1f}")
    
    print(f"\\nEvaluating with robust parameters...")
    print(f"  • k = 10 (precision@10)")
    print(f"  • 1000 random queries")
    print(f"  • Full dataset ({len(shape_query.shape_names)} shapes)")
    
    # Evaluate with robust parameters
    metrics = shape_query.evaluate_retrieval(n_queries=1000, k=10)
    
    print(f"\\n" + "="*60)
    print("CORRECTED Z-SCORE BASELINE RESULTS")
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
    print("BASELINE ESTABLISHED - READY FOR WEIGHT OPTIMIZATION")
    print("="*80)
    
    return metrics

if __name__ == "__main__":
    evaluate_corrected_zscore_properly()