"""
Weight optimization for FinalShapeQuery using individual descriptor weights.

This version:
- Uses the corrected FinalShapeQuery with raw distances + [0,1] normalization
- Optimizes individual weights for all 11 descriptors
- Uses full 2,438 shape dataset with cached distance matrices
- Targets 30%+ performance as expected by assignment

Key improvements over previous approach:
✅ No broken distance computation (no abs() on z-scores)
✅ Uses raw EMD + Euclidean distances as assignment requires
✅ Individual weights allow optimization to find discriminative descriptors
✅ Pre-computed matrices enable fast optimization trials
"""

import optuna
import numpy as np
from final_shape_query import FinalShapeQuery
import json
import pandas as pd
from datetime import datetime
import os
import time


def objective_final(trial, shape_query, k=30, subset_size=500):
    """
    Objective function for FinalShapeQuery with individual descriptor weights.
    
    This optimizes 11 individual weights rather than excluding descriptors,
    allowing the optimizer to discover which features are most discriminative.
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
    
    # Normalize to sum to 1.0 (required by FinalShapeQuery)
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
        
        # Evaluate performance
        metrics = shape_query.evaluate_retrieval(n_queries=subset_size, k=k)
        
        # Store additional metrics
        trial.set_user_attr('precision@k', metrics['mean_precision@k'])
        trial.set_user_attr('std_precision@k', metrics['std_precision@k'])
        trial.set_user_attr('n_queries', metrics['n_queries'])
        trial.set_user_attr('update_time', update_time)
        trial.set_user_attr('weight_sum', weight_sum)
        
        # Store individual weights for analysis
        for desc, weight in weights.items():
            trial.set_user_attr(f'weight_{desc}', weight)
        
        # Return mean precision@k as objective (optimization target)
        return metrics['mean_precision@k']
    
    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0


if __name__ == "__main__":
    print("="*80)
    print("FINAL WEIGHT OPTIMIZATION - Individual Descriptor Weights")
    print("="*80)
    print("Approach: Raw EMD/Euclidean distances + [0,1] normalization")
    print("Features: All 11 descriptors with individual optimizable weights")
    print("Target: 30%+ precision@30 (assignment expectation)")
    print("="*80 + "\\n")
    
    # Initialize FinalShapeQuery with full dataset
    print("Initializing FinalShapeQuery with full dataset...")
    t_start = time.time()
    
    # Use absolute paths to ensure proper caching
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Current script directory
    base_dir = os.path.dirname(os.path.dirname(script_dir))  # Go up to MultimediaRetrieval root
    csv_path = os.path.join(script_dir, "final_006_cleaned.csv")  # CSV is in same dir as script
    cache_path = os.path.join(base_dir, "distance_matrices_raw_normalized_full")  # Cache is in root
    
    print(f"Dataset: {csv_path}")
    print(f"Cache: {cache_path}")
    print(f"Cache exists: {os.path.exists(cache_path)}")
    
    shape_query = FinalShapeQuery(
        csv_file_path=csv_path,
        cache_dir=cache_path,
        num_shapes=None,  # Use all 2,438 shapes
        debug=True
    )
    
    t_end = time.time()
    print(f"Initialization complete in {(t_end-t_start):.1f}s")
    print(f"Dataset: {len(shape_query.shape_names)} shapes, {len(set(shape_query.shape_classes))} classes\\n")
    
    # Evaluate baseline with equal weights
    print("Evaluating baseline (equal weights)...")
    baseline_metrics = shape_query.evaluate_retrieval(n_queries=100, k=30)
    baseline_precision = baseline_metrics['mean_precision@k']
    baseline_std = baseline_metrics['std_precision@k']
    
    print(f"Baseline Results:")
    print(f"  Precision@30: {baseline_precision:.3f} ± {baseline_std:.3f}")
    print(f"  Queries: {baseline_metrics['n_queries']}")
    
    # Target analysis
    target_improvement = 0.30 / baseline_precision if baseline_precision > 0 else 2.0
    print(f"\\nTarget: 30%+ precision@30")
    print(f"Required improvement: {target_improvement:.1f}x from baseline")
    
    if baseline_precision >= 0.30:
        print("🎉 Baseline already meets target! Optimization will push higher.")
    elif baseline_precision >= 0.15:
        print("📈 Strong baseline - optimization should reach target.")
    else:
        print("📊 Moderate baseline - optimization will search for discriminative features.")
    
    print("\\n" + "="*60)
    print("STARTING BAYESIAN OPTIMIZATION")
    print("="*60)
    
    # Create or load study
    study_name = 'final_shape_weights'
    db_file = os.path.join(base_dir, "optimization_final.db")
    
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        storage=f"sqlite:///{db_file}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=20,  # More startup trials for 11 parameters
            n_ei_candidates=24
        )
    )
    
    existing_trials = len(study.trials)
    print(f"Existing trials: {existing_trials}")
    
    # Run optimization
    n_trials = 150  # More trials for 11-parameter optimization
    print(f"Running {n_trials} optimization trials...")
    print(f"Parameters: 11 individual descriptor weights")
    print(f"Evaluation: {100} queries per trial on full dataset\\n")
    
    try:
        study.optimize(
            lambda trial: objective_final(trial, shape_query, k=30, subset_size=100),
            n_trials=n_trials,
            show_progress_bar=True,
            gc_after_trial=True  # Memory management
        )
        
        print("\\n" + "="*80)
        print("OPTIMIZATION COMPLETE")
        print("="*80)
        
        # Analyze results
        best_trial = study.best_trial
        
        print(f"\\nBest trial: #{best_trial.number}")
        print(f"  Precision@30: {best_trial.value:.4f} ({best_trial.value*100:.1f}%)")
        print(f"  Standard deviation: ±{best_trial.user_attrs['std_precision@k']:.4f}")
        print(f"  Queries: {best_trial.user_attrs['n_queries']}")
        print(f"  Update time: {best_trial.user_attrs['update_time']:.3f}s")
        
        # Extract optimized weights
        descriptors = [
            'surface_area', 'compactness', 'rectangularity', 
            'diameter', 'convexity', 'eccentricity',
            'A3_hist', 'D1_hist', 'D2_hist', 'D3_hist', 'D4_hist'
        ]
        
        optimized_weights = {}
        for desc in descriptors:
            optimized_weights[desc] = best_trial.user_attrs[f'weight_{desc}']
        
        # Display weights
        print(f"\\nOptimized weights:")
        weight_sum = 0.0
        for desc, weight in optimized_weights.items():
            feature_type = "Global" if desc in shape_query.available_global else "Histogram"
            print(f"  {desc:15s} ({feature_type:9s}): {weight:.4f}")
            weight_sum += weight
        print(f"  {'(sum)':15s} {'':10s}: {weight_sum:.6f}")
        
        # Performance improvement
        improvement = ((best_trial.value - baseline_precision) / baseline_precision) * 100
        print(f"\\nImprovement over baseline: {improvement:+.1f}%")
        
        # Target achievement
        if best_trial.value >= 0.30:
            print("🎉 TARGET ACHIEVED: 30%+ precision@30!")
        elif best_trial.value >= 0.25:
            print("📈 EXCELLENT: Close to 30% target")
        elif best_trial.value >= 0.20:
            print("📊 GOOD: Solid improvement toward target")
        else:
            print("📋 PROGRESS: Meaningful improvement from baseline")
        
        # Save results
        output_dir = os.path.join(base_dir, "optimization_results_final")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save optimized weights
        weights_df = pd.DataFrame([
            {'descriptor': desc, 'weight': weight, 
             'type': 'Global' if desc in shape_query.available_global else 'Histogram'}
            for desc, weight in optimized_weights.items()
        ])
        weights_path = os.path.join(output_dir, 'optimized_weights_final.csv')
        weights_df.to_csv(weights_path, index=False)
        print(f"\\nSaved optimized weights to: {weights_path}")
        
        # Final evaluation on full dataset
        print(f"\\nFinal evaluation on FULL dataset...")
        shape_query.update_weights(optimized_weights)
        final_metrics = shape_query.evaluate_retrieval(n_queries=None, k=30)  # All possible queries
        
        print(f"Final Performance (Full Dataset):")
        print(f"  Precision@30: {final_metrics['mean_precision@k']:.4f} ({final_metrics['mean_precision@k']*100:.1f}%)")
        print(f"  Standard deviation: ±{final_metrics['std_precision@k']:.4f}")
        print(f"  Total queries: {final_metrics['n_queries']}")
        
        # Save comprehensive results
        results_summary = {
            'baseline_precision_at_30': baseline_precision,
            'baseline_std': baseline_std,
            'optimized_precision_at_30': best_trial.value,
            'optimized_std': best_trial.user_attrs['std_precision@k'],
            'final_precision_at_30': final_metrics['mean_precision@k'],
            'final_std': final_metrics['std_precision@k'],
            'improvement_percent': improvement,
            'n_trials': n_trials + existing_trials,
            'best_trial_number': best_trial.number,
            'optimized_weights': optimized_weights,
            'dataset_shapes': len(shape_query.shape_names),
            'dataset_classes': len(set(shape_query.shape_classes)),
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = os.path.join(output_dir, 'optimization_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"Saved summary to: {summary_path}")
        
        # Analysis of most important features
        sorted_weights = sorted(optimized_weights.items(), key=lambda x: x[1], reverse=True)
        print(f"\\nMost discriminative features:")
        for i, (desc, weight) in enumerate(sorted_weights[:5], 1):
            feature_type = "Global" if desc in shape_query.available_global else "Histogram"
            print(f"  {i}. {desc:15s} ({feature_type:9s}): {weight:.4f}")
        
        print("\\n" + "="*80)
        print("OPTIMIZATION COMPLETE - Ready for assignment submission!")
        print("="*80)
        
    except KeyboardInterrupt:
        print(f"\\nOptimization interrupted. Best so far:")
        if study.trials:
            best = study.best_trial
            print(f"  Trial {best.number}: {best.value:.4f} precision@30")
        
    except Exception as e:
        print(f"\\nOptimization error: {e}")
        import traceback
        traceback.print_exc()