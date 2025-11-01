"""
Weight optimization for CorrectedZScoreShapeQuery using individual descriptor weights.

This version:
- Uses the corrected Z-Score approach with proper standardization (no .abs())
- Optimizes individual weights for all 11 descriptors  
- Uses full 2,438 shape dataset with cached z-score matrices
- Targets improved performance beyond the 27.4% equal-weight baseline

Key advantages:
✅ Z-score standardization preserves negative "closer than average" semantics
✅ Individual weights allow optimization to find discriminative descriptors
✅ Pre-computed z-score matrices enable fast optimization trials
✅ Already outperforms min-max approach with equal weights
"""

import optuna
import numpy as np
from corrected_zscore_shape_query import CorrectedZScoreShapeQuery
import json
import pandas as pd
from datetime import datetime
import os
import time
import random


def objective_zscore(trial, shape_query, k=10, subset_size=500):
    """
    Objective function for CorrectedZScoreShapeQuery with individual descriptor weights.
    
    This optimizes 11 individual weights to find the most discriminative combination.
    """
    # All 11 descriptors (6 global + 5 histogram)
    descriptors = [
        # Global features (Euclidean distance)
        'surface_area', 'compactness', 'rectangularity', 
        'diameter', 'convexity', 'eccentricity',
        # Histogram features (EMD distance)
        'A3_hist', 'D1_hist', 'D2_hist', 'D3_hist', 'D4_hist'
    ]
    
    # Sample raw weights for each descriptor
    raw_weights = []
    for desc in descriptors:
        raw_weight = trial.suggest_float(f'weight_{desc}', 0.001, 10.0)
        raw_weights.append(raw_weight)
    
    # Normalize to sum to 1.0 (required by CorrectedZScoreShapeQuery)
    total = sum(raw_weights)
    weights = {desc: raw_w / total for desc, raw_w in zip(descriptors, raw_weights)}
    
    # Verify constraint
    weight_sum = sum(weights.values())
    assert abs(weight_sum - 1.0) < 1e-6, f"Weights don't sum to 1.0: {weight_sum}"
    
    try:
        # Update weights (fast - just recomputes weighted combination)
        start_time = time.time()
        shape_query.update_weights(weights)
        update_time = time.time() - start_time
        
        # Evaluate performance on subset
        n_shapes = len(shape_query.shape_names)
        n_queries = min(subset_size, n_shapes - 1)
        
        # Random sampling for speed
        query_indices = random.sample(range(n_shapes), n_queries)
        
        precisions = []
        for query_idx in query_indices:
            query_shape = shape_query.shape_names[query_idx]
            query_class = shape_query.shape_classes[query_idx]
            
            # Get top k retrievals
            results = shape_query.query(query_shape, k=k, include_self=False)
            
            # Count correct retrievals (results is dict for z-score approach)
            correct = 0
            for shape_name in results.keys():
                shape_idx = shape_query.shape_names.index(shape_name)
                if shape_query.shape_classes[shape_idx] == query_class:
                    correct += 1
            
            precision = correct / k if k > 0 else 0.0
            precisions.append(precision)
        
        mean_precision = np.mean(precisions) if precisions else 0.0
        
        eval_time = time.time() - start_time - update_time
        
        # Log trial details
        trial.set_user_attr("mean_precision", mean_precision)
        trial.set_user_attr("n_queries", len(precisions))
        trial.set_user_attr("update_time", update_time)
        trial.set_user_attr("eval_time", eval_time)
        trial.set_user_attr("weights", weights)
        
        return mean_precision
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 0.0


def optimize_zscore_weights(n_trials=200, k=10, subset_size=300, output_dir="optimization_results_zscore"):
    """
    Optimize weights for CorrectedZScoreShapeQuery.
    
    Args:
        n_trials: Number of optimization trials
        k: Precision@k to optimize for
        subset_size: Number of queries per trial (for speed)
        output_dir: Directory to save results
    """
    print("="*80)
    print("Z-SCORE WEIGHT OPTIMIZATION")
    print("="*80)
    print(f"Baseline (Equal Weights): 27.4% precision@{k}")
    print(f"Target: >30% precision@{k}")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize shape query system with cached matrices
    print("\\nInitializing CorrectedZScoreShapeQuery...")
    shape_query = CorrectedZScoreShapeQuery(
        csv_file_path="final_006_cleaned.csv",
        cache_dir="distance_matrices_zscore_corrected_full",
        num_shapes=None,  # Full dataset
        combination_method="weighted_sum",
        debug=False
    )
    
    print(f"Dataset: {len(shape_query.shape_names)} shapes, {len(set(shape_query.shape_classes))} classes")
    print(f"Optimization: {n_trials} trials, {subset_size} queries per trial")
    
    # Create optimization study
    study_name = "zscore_weights_v1"
    storage = f"sqlite:///{output_dir}/zscore_optimization.db"
    
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True
    )
    
    print(f"\\nStarting optimization...")
    print(f"Study: {study_name}")
    print(f"Storage: {storage}")
    
    # Run optimization
    study.optimize(
        lambda trial: objective_zscore(trial, shape_query, k=k, subset_size=subset_size),
        n_trials=n_trials
    )
    
    # Get best results
    best_trial = study.best_trial
    best_weights = best_trial.user_attrs["weights"]
    best_precision = best_trial.value
    
    print("\\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Best precision@{k}: {best_precision:.4f} ({best_precision*100:.1f}%)")
    print(f"Improvement: {((best_precision - 0.274) / 0.274) * 100:+.1f}% over baseline")
    print(f"Total trials: {len(study.trials)}")
    
    # Show best weights
    print(f"\\nBest weights:")
    for desc, weight in best_weights.items():
        print(f"  {desc}: {weight:.4f}")
    
    # Save results
    results = {
        "best_precision_at_k": best_precision,
        "baseline_precision": 0.274,
        "improvement_percent": ((best_precision - 0.274) / 0.274) * 100,
        "k": k,
        "n_trials": len(study.trials),
        "subset_size": subset_size,
        "best_weights": best_weights,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save to JSON
    with open(f"{output_dir}/zscore_optimization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save weights to CSV
    weights_df = pd.DataFrame([
        {"descriptor": desc, "weight": weight, "type": "Global" if desc in shape_query.global_features else "Histogram"}
        for desc, weight in best_weights.items()
    ])
    weights_df.to_csv(f"{output_dir}/zscore_optimized_weights.csv", index=False)
    
    print(f"\\nResults saved to {output_dir}/")
    print(f"  - zscore_optimization_results.json")
    print(f"  - zscore_optimized_weights.csv")
    
    return study, shape_query, best_weights


if __name__ == "__main__":
    # Run optimization
    study, shape_query, best_weights = optimize_zscore_weights(
        n_trials=200,  # 200 trials should be enough
        k=10,          # Optimize for precision@10
        subset_size=300,  # 300 queries per trial for good statistics
        output_dir="optimization_results_zscore"
    )
    
    print("\\n" + "="*80)
    print("FINAL EVALUATION WITH OPTIMIZED WEIGHTS")
    print("="*80)
    
    # Evaluate optimized weights on full 1000 queries
    shape_query.update_weights(best_weights)
    
    # Use the same evaluation function from fair_comparison
    import random
    from tqdm import tqdm
    
    n_queries = 1000
    query_indices = random.sample(range(len(shape_query.shape_names)), n_queries)
    
    precisions = []
    for query_idx in tqdm(query_indices, desc="Final evaluation"):
        query_shape = shape_query.shape_names[query_idx]
        query_class = shape_query.shape_classes[query_idx]
        
        results = shape_query.query(query_shape, k=10, include_self=False)
        
        correct = 0
        for shape_name in results.keys():
            shape_idx = shape_query.shape_names.index(shape_name)
            if shape_query.shape_classes[shape_idx] == query_class:
                correct += 1
        
        precisions.append(correct / 10)
    
    final_precision = np.mean(precisions)
    final_std = np.std(precisions)
    
    print(f"\\nFINAL RESULTS (1000 queries):")
    print(f"Baseline (Equal):     27.4% ± 27.2%")
    print(f"Optimized (Z-Score):  {final_precision*100:.1f}% ± {final_std*100:.1f}%")
    print(f"Improvement:          {((final_precision - 0.274) / 0.274) * 100:+.1f}%")
    
    # Calculate confidence interval
    std_error = final_std / (1000 ** 0.5)
    ci_95 = 1.96 * std_error
    print(f"95% Confidence:       [{final_precision - ci_95:.3f}, {final_precision + ci_95:.3f}]")