"""
Compute final distance matrices for the complete dataset (2,438 shapes) 
and evaluate baseline performance with equal weights.
"""

from final_shape_query import FinalShapeQuery
import pandas as pd

print("="*80)
print("FINAL APPROACH - FULL DATASET COMPUTATION")
print("="*80)

print("Computing raw distance matrices for the complete dataset:")
print("✅ 2,438 shapes across 69 classes")
print("✅ Raw EMD + Euclidean distances")
print("✅ [0,1] min-max normalization")
print("✅ Individual descriptor weights (equal baseline)")
print("✅ Cache for future optimization")
print()

try:
    print("Creating FinalShapeQuery with full dataset...")
    print("This will take some time for EMD computation but only needs to be done once.")
    print()
    
    # Use full dataset (no num_shapes limit)
    query_system = FinalShapeQuery(
        csv_file_path="Src/matching/final_006_cleaned.csv",
        cache_dir="distance_matrices_raw_normalized_full",  # Different cache dir for full dataset
        debug=True  # Show progress details
    )
    
    print("\\n" + "="*60)
    print("EVALUATING BASELINE PERFORMANCE")
    print("="*60)
    print("Using equal weights (1/11 = 0.0909 per descriptor)")
    print("Testing with 100 queries for robust statistics...")
    print()
    
    # Evaluate with equal weights
    metrics = query_system.evaluate_retrieval(n_queries=100, k=30)
    
    print("\\n" + "="*60)
    print("FULL DATASET BASELINE RESULTS:")
    print("="*60)
    print(f"Dataset: {len(query_system.shape_names)} shapes, {len(set(query_system.shape_classes))} classes")
    print(f"Approach: Raw distances with [0,1] normalization")
    print(f"Weights: Equal (0.0909 per descriptor)")
    print(f"Precision@30: {metrics['mean_precision@k']:.3f} ± {metrics['std_precision@k']:.3f}")
    print(f"Queries tested: {metrics['n_queries']}")
    
    # Performance assessment
    if metrics['mean_precision@k'] >= 0.25:
        print("\\n🎉 EXCELLENT BASELINE: Great foundation for optimization!")
        print("   Expected after optimization: 30%+ precision@30")
    elif metrics['mean_precision@k'] >= 0.20:
        print("\\n📈 GOOD BASELINE: Solid foundation for optimization")
        print("   Expected after optimization: 25-35% precision@30")
    elif metrics['mean_precision@k'] >= 0.15:
        print("\\n📊 DECENT BASELINE: Reasonable starting point")
        print("   Expected after optimization: 20-30% precision@30")
    else:
        print("\\n⚠️  LOW BASELINE: May need further refinement")
    
    # Compare with previous approaches
    print(f"\\n" + "="*60)
    print("COMPARISON WITH PREVIOUS APPROACHES:")
    print("="*60)
    print(f"Original z-score + abs():           ~0.5% precision@30")
    print(f"Z-score without abs() (negatives):  16.1% precision@30")
    print(f"Final raw [0,1] approach:           {metrics['mean_precision@k']:.1f}% precision@30")
    
    improvement_vs_original = metrics['mean_precision@k'] / 0.005
    improvement_vs_zscore = metrics['mean_precision@k'] / 0.161
    
    print(f"\\nImprovement vs original: {improvement_vs_original:.0f}× better")
    print(f"vs z-score approach: {improvement_vs_zscore:.1f}× {'better' if improvement_vs_zscore > 1 else 'performance'}")
    
    # Show distance matrix info
    print(f"\\n" + "="*60)
    print("DISTANCE MATRICES CACHED:")
    print("="*60)
    print(f"Location: {query_system.cache_dir}")
    
    import os
    if os.path.exists(query_system.cache_dir):
        files = [f for f in os.listdir(query_system.cache_dir) if f.endswith('.csv')]
        print(f"Files created: {len(files)}")
        for file in sorted(files):
            print(f"  ✅ {file}")
    
    # Show sample distance ranges
    print(f"\\nSample distance ranges (all normalized to [0,1]):")
    for desc in ['surface_area', 'compactness', 'A3_hist']:
        if desc in query_system.distance_matrices:
            matrix = query_system.distance_matrices[desc]
            non_zero = matrix[matrix > 0]
            if len(non_zero) > 0:
                print(f"  {desc:15s}: {non_zero.min():.6f} to {matrix.max():.6f}")
    
    # Save baseline results
    baseline_info = f"""FULL DATASET BASELINE RESULTS
============================
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset: {len(query_system.shape_names)} shapes, {len(set(query_system.shape_classes))} classes
Approach: Raw distances with [0,1] min-max normalization
Weights: Equal (0.0909 per descriptor)

Performance:
Precision@30: {metrics['mean_precision@k']:.3f} ± {metrics['std_precision@k']:.3f}
Queries tested: {metrics['n_queries']}

Distance matrices cached in: {query_system.cache_dir}

Individual descriptor weights (all equal):
"""
    
    for desc, weight in query_system.weights.items():
        baseline_info += f"  {desc}: {weight:.4f}\\n"
    
    baseline_info += f"""
Next steps:
1. Use Bayesian optimization to find optimal individual weights
2. Expected improvement: 5-15 percentage points
3. Target: 30%+ precision@30 (professor's expectation)

This baseline is ready for optimization!
"""
    
    with open("full_dataset_final_baseline.txt", "w") as f:
        f.write(baseline_info)
    
    print(f"\\nBaseline results saved to: full_dataset_final_baseline.txt")
    print(f"\\n🚀 READY FOR OPTIMIZATION!")
    print(f"   Distance matrices: Cached and ready")
    print(f"   Optimization target: 30%+ precision@30")
    print(f"   Next step: Integrate with Bayesian optimization framework")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\\n" + "="*80)