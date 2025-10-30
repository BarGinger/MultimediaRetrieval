"""
Weight optimization for shape retrieval using Bayesian optimization.

This script optimizes descriptor weights to maximize retrieval quality metrics
(Precision@K, Recall@K, MAP) using class labels as ground truth.

Uses Optuna for efficient Bayesian optimization of the weight space.
"""

import optuna
import numpy as np
from shapeQuery import ShapeQuery
import json
import pandas as pd
from datetime import datetime


def objective(trial, shape_query, k=10, subset_size=200):
    """
    Objective function for Optuna optimization.
    
    Enforces constraint: weights sum to 1.0
    
    Args:
        trial: Optuna trial object
        shape_query: ShapeQuery instance with preloaded matrices
        k: Number of neighbors for evaluation
        subset_size: Number of shapes to evaluate on (for speed)
    
    Returns:
        Negative MAP (we minimize, Optuna maximizes)
    """
    # Suggest weights for each descriptor using Dirichlet-like sampling
    # This ensures weights sum to 1.0
    descriptors = ['A3', 'D1', 'D2', 'D3', 'D4', 
                   'surface_area', 'compactness', 'rectangularity',
                   'diameter', 'convexity', 'eccentricity']
    
    # Sample N-1 weights in [0, 1], then compute the last one
    # Use a softmax-like approach: sample raw values, then normalize
    raw_weights = []
    for i, desc in enumerate(descriptors):
        # Suggest a positive value (will be normalized)
        raw_weight = trial.suggest_float(f'raw_weight_{desc}', 0.01, 10.0)
        raw_weights.append(raw_weight)
    
    # Normalize to sum to 1.0
    total = sum(raw_weights)
    weights = {desc: raw_w / total for desc, raw_w in zip(descriptors, raw_weights)}
    
    # Verify constraint (for debugging)
    weight_sum = sum(weights.values())
    assert abs(weight_sum - 1.0) < 1e-6, f"Weights don't sum to 1.0: {weight_sum}"
    
    # Update distance matrix with new weights
    try:
        elapsed = shape_query.update_weights(weights, save_matrix=False)
        
        # Evaluate retrieval quality
        metrics = shape_query.evaluate_retrieval(k=k, subset_size=subset_size)
        
        # Primary objective: MAP (Mean Average Precision)
        # Also log other metrics for analysis
        trial.set_user_attr('precision@k', metrics['precision@k'])
        trial.set_user_attr('recall@k', metrics['recall@k'])
        trial.set_user_attr('mrr', metrics['mrr'])
        trial.set_user_attr('compute_time', elapsed)
        trial.set_user_attr('weight_sum', weight_sum)
        
        # Return MAP (Optuna will maximize this)
        return metrics['map']
    
    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0  # Return worst possible value


def optimize_weights(
    csv_file_path: str = "final_006_cleaned.csv",
    precomputed_dir: str = "distance_matrices_normalized_98",
    n_trials: int = 100,
    k: int = 10,
    subset_size: int = 200,
    output_dir: str = "optimization_results",
    study_name: str = None,
    storage: str = None
):
    """
    Run weight optimization.
    
    Args:
        csv_file_path: Path to shape features CSV
        precomputed_dir: Directory with precomputed distance matrices
        n_trials: Number of optimization trials
        k: Number of neighbors for evaluation
        subset_size: Number of shapes to evaluate on per trial
        output_dir: Directory to save results
        study_name: Name of the study (for resuming). If None, creates new study.
        storage: Optuna storage URL (e.g., "sqlite:///optimization.db"). 
                 If None, uses in-memory storage (cannot resume).
    """
    import os
    import time
    
    print("="*80)
    print("SHAPE RETRIEVAL WEIGHT OPTIMIZATION")
    print("="*80)
    print(f"Optimization settings:")
    print(f"  Trials: {n_trials}")
    print(f"  Evaluation metric: MAP (Mean Average Precision)")
    print(f"  K neighbors: {k}")
    print(f"  Evaluation subset size: {subset_size} shapes")
    print("="*80 + "\n")
    
    # Initialize ShapeQuery (loads all matrices once)
    print("Initializing ShapeQuery...")
    t_init_start = time.time()
    shape_query = ShapeQuery(
        csv_file_path=csv_file_path,
        precomputed_dir=precomputed_dir,
        compute_matrix=True,  # Compute initial matrix with default weights
        debug=False
    )
    t_init_end = time.time()
    print(f"Initialization complete in {(t_init_end-t_init_start):.1f}s")
    print(f"Loaded {len(shape_query.shapes)} shapes\n")
    
    # Evaluate baseline (current weights)
    print("Evaluating baseline weights...")
    baseline_metrics = shape_query.evaluate_retrieval(k=k, subset_size=subset_size)
    print(f"Baseline MAP: {baseline_metrics['map']:.4f}")
    print(f"Baseline Precision@{k}: {baseline_metrics['precision@k']:.4f}")
    print(f"Baseline Recall@{k}: {baseline_metrics['recall@k']:.4f}")
    print(f"Baseline MRR: {baseline_metrics['mrr']:.4f}\n")
    
    # Create optimization study
    print("Starting optimization...\n")
    
    # Use persistent storage if provided (allows resuming)
    if storage:
        study_name = study_name or 'shape_retrieval_optimization'
        print(f"Using persistent storage: {storage}")
        print(f"Study name: {study_name}")
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            storage=storage,
            load_if_exists=True,  # Resume if exists
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        print(f"Existing trials: {len(study.trials)}")
        print(f"Running {n_trials} additional trials\n")
    else:
        # In-memory study (cannot resume)
        study_name = study_name or f'shape_retrieval_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, shape_query, k, subset_size),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    
    # Best results
    best_trial = study.best_trial
    print(f"\nBest trial: #{best_trial.number}")
    print(f"  MAP: {best_trial.value:.4f}")
    print(f"  Precision@{k}: {best_trial.user_attrs['precision@k']:.4f}")
    print(f"  Recall@{k}: {best_trial.user_attrs['recall@k']:.4f}")
    print(f"  MRR: {best_trial.user_attrs['mrr']:.4f}")
    print(f"  Compute time: {best_trial.user_attrs['compute_time']:.3f}s")
    
    print(f"\nBest weights:")
    
    # Reconstruct normalized weights from raw weights
    descriptors = ['A3', 'D1', 'D2', 'D3', 'D4', 
                   'surface_area', 'compactness', 'rectangularity',
                   'diameter', 'convexity', 'eccentricity']
    
    raw_weights = []
    for desc in descriptors:
        raw_weight = best_trial.params[f'raw_weight_{desc}']
        raw_weights.append(raw_weight)
    
    # Normalize to get actual weights
    total = sum(raw_weights)
    best_weights = {desc: raw_w / total for desc, raw_w in zip(descriptors, raw_weights)}
    
    # Print and verify they sum to 1.0
    weight_sum = 0.0
    for desc, weight in best_weights.items():
        print(f"  {desc}: {weight:.4f}")
        weight_sum += weight
    print(f"  (sum: {weight_sum:.6f})")
    
    # Improvement
    improvement = ((best_trial.value - baseline_metrics['map']) / baseline_metrics['map']) * 100
    print(f"\nImprovement over baseline: {improvement:+.1f}%")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save best weights to CSV
    weights_df = pd.DataFrame([
        {'descriptor': desc, 'weight': weight}
        for desc, weight in best_weights.items()
    ])
    weights_path = os.path.join(output_dir, 'optimized_weights.csv')
    weights_df.to_csv(weights_path, index=False)
    print(f"\nSaved optimized weights to: {weights_path}")
    
    # Save full results
    results = {
        'optimization': {
            'n_trials': n_trials,
            'k': k,
            'subset_size': subset_size,
            'best_trial_number': best_trial.number,
            'timestamp': datetime.now().isoformat()
        },
        'baseline': {
            'weights': shape_query.current_weights,
            'metrics': baseline_metrics
        },
        'optimized': {
            'weights': best_weights,
            'metrics': {
                'map': best_trial.value,
                'precision@k': best_trial.user_attrs['precision@k'],
                'recall@k': best_trial.user_attrs['recall@k'],
                'mrr': best_trial.user_attrs['mrr']
            }
        },
        'improvement_percent': improvement
    }
    
    results_path = os.path.join(output_dir, 'optimization_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved full results to: {results_path}")
    
    # Save trial history
    trials_df = study.trials_dataframe()
    trials_path = os.path.join(output_dir, 'optimization_trials.csv')
    trials_df.to_csv(trials_path, index=False)
    print(f"Saved trial history to: {trials_path}")
    
    # Create visualization if optuna.visualization is available
    try:
        import optuna.visualization as vis
        
        # Optimization history
        fig_history = vis.plot_optimization_history(study)
        fig_history.write_html(os.path.join(output_dir, 'optimization_history.html'))
        
        # Parameter importances
        fig_importance = vis.plot_param_importances(study)
        fig_importance.write_html(os.path.join(output_dir, 'parameter_importances.html'))
        
        print(f"Saved visualizations to: {output_dir}/")
    except ImportError:
        print("Note: Install plotly for visualizations: pip install plotly")
    
    print("\n" + "="*80)
    
    return study, shape_query, best_weights


if __name__ == "__main__":
    # Run optimization
    # 
    # To use persistent storage (allows resuming):
    # study, shape_query, best_weights = optimize_weights(
    #     n_trials=100,
    #     k=10,
    #     subset_size=200,
    #     output_dir="optimization_results",
    #     study_name="shape_retrieval_v1",
    #     storage="sqlite:///optimization.db"
    # )
    #
    # To resume and run 100 more trials, just run the script again!
    # The study_name and storage must match.
    
    study, shape_query, best_weights = optimize_weights(
        n_trials=100,  # Adjust based on time budget
        k=30,  # Increased to match average class size (~35 shapes/class)
        subset_size=500,  # Larger subset for better generalization to full dataset
        output_dir="optimization_results",
        # Uncomment to enable persistent storage:
        study_name="shape_retrieval_v1",
        storage="sqlite:///optimization.db"
    )
    
    # Optionally: Evaluate best weights on full dataset
    print("\nEvaluating best weights on FULL dataset...")
    shape_query.update_weights(best_weights, save_matrix=True)
    full_metrics = shape_query.evaluate_retrieval(k=30, subset_size=None)  # Use same k
    
    print(f"Full dataset evaluation:")
    print(f"  MAP: {full_metrics['map']:.4f}")
    print(f"  Precision@30: {full_metrics['precision@k']:.4f}")
    print(f"  Recall@30: {full_metrics['recall@k']:.4f}")
    print(f"  MRR: {full_metrics['mrr']:.4f}")
    print(f"  Queries: {full_metrics['num_queries']}")
