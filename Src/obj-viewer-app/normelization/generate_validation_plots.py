"""
Comprehensive Validation Plot Generator for Normalization Pipeline
Generates publication-quality figures for academic report

Usage:
    python generate_validation_plots.py --input Datasets/UnifiedPreprocessed/validation_detailed.json --output figures/
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from matplotlib.gridspec import GridSpec
import argparse
from scipy import stats as scipy_stats
import os
import glob

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13


class ValidationPlotGenerator:
    def plot_pca_axis_angle_boxplot(self):
        print("📈 Generating PCA axis-to-world angle histograms...")
        import numpy as np
        import matplotlib.pyplot as plt
        import glob
        import os
        import json
        # World axes
        world_axes = np.eye(3)
        colors = ['red', 'green', 'blue']
        axis_labels = ['X-axis', 'Y-axis', 'Z-axis']
        axis_angles = [[], [], []]  # For each world axis
        # Find metadata files
        base_dir = os.path.join(str(self.output_dir.parent), '*', '*_metadata.json')
        metadata_files = glob.glob(base_dir)
        print(f"[INFO] Scanning {len(metadata_files)} metadata files for eigenvectors...")
        for meta_path in metadata_files:
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                pca_block = meta.get('normalization_info', {}).get('pca')
                eigvecs = None
                eigvals = None
                if pca_block and 'eigenvectors' in pca_block and 'eigenvalues' in pca_block:
                    eigvecs = np.array(pca_block['eigenvectors'])
                    eigvals = np.array(pca_block['eigenvalues'])
                if eigvecs is None or eigvals is None or eigvecs.shape != (3, 3) or eigvals.shape[0] != 3:
                    continue
                # Sort PCA axes by eigenvalue magnitude (descending)
                sort_idx = np.argsort(-eigvals)
                eigvecs_sorted = eigvecs[sort_idx]
                # For each world axis, find the angle to the corresponding sorted PCA axis
                for j in range(3):
                    angle = np.arccos(np.clip(abs(np.dot(eigvecs_sorted[j], world_axes[j])), -1.0, 1.0)) * 180.0 / np.pi
                    axis_angles[j].append(angle)
            except Exception as e:
                print(f"[ERROR] Failed to process {meta_path}: {e}")
                continue
        print(f"[DEBUG] axis_angles population: {[len(a) for a in axis_angles]}")
        for idx, arr in enumerate(axis_angles):
            print(f"[DEBUG] axis_angles[{idx}] first 10: {arr[:10]}")
            if arr:
                print(f"[DEBUG] axis_angles[{idx}] min: {min(arr)}, max: {max(arr)}")
        # Prepare data for histogram
        data = []
        for i in range(3):
            vals = [float(v) for v in axis_angles[i] if v is not None and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))]
            vals = [v for v in vals if v >= 0.01]
            data.append(vals)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        bins = np.linspace(0, 180, 40)
        for i in range(3):
            axes[i].hist(data[i], bins=bins, color=colors[i], alpha=0.7, edgecolor='black')
            axes[i].set_title(f'Histogram: {axis_labels[i]}')
            axes[i].set_xlabel('Angle (degrees)')
            axes[i].set_xlim(0, 180)
            axes[i].grid(True, alpha=0.3, axis='y')
        axes[0].set_ylabel('Frequency')
        fig.suptitle('Histogram: Angle between sorted PCA axes and world axes (corresponding axes)')
        output_path = self.output_dir / 'fig_pca_axis_angle_histogram.png'
        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"   ✅ Saved: {output_path}")
    def __init__(self, validation_data_path, output_dir):
        """
        Initialize plot generator
        
        Args:
            validation_data_path: Path to validation_detailed.json
            output_dir: Directory to save plots
        """
        self.validation_data_path = Path(validation_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load validation data
        print(f"📊 Loading validation data from: {self.validation_data_path}")
        with open(self.validation_data_path, 'r') as f:
            self.data = json.load(f)
        
        self.validations = self.data.get('detailed_validations', [])
        self.stats = self.data.get('validation_statistics', {})
        print(f"   Loaded {len(self.validations)} shape validations")
    
    def extract_metric(self, metric_path, default=None):
        """Extract metric values from nested validation data"""
        values = []
        for validation in self.validations:
            try:
                value = validation
                for key in metric_path.split('.'):
                    value = value[key]
                if value is not None and not (isinstance(value, float) and np.isinf(value)):
                    values.append(value)
            except (KeyError, TypeError):
                if default is not None:
                    values.append(default)
        return values
    
    def plot_overview_panel(self):
        """Figure 1: Multi-panel overview of key validation metrics"""
        print("📈 Generating Figure 1: Validation Overview Panel...")
        
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Centering Error Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        centering_errors = self.extract_metric('cross_step_validation.final_centering_error')
        if centering_errors:
            ax1.hist(np.log10(centering_errors), bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            ax1.axvline(np.log10(1e-10), color='red', linestyle='--', linewidth=2, label='Target threshold')
            ax1.set_xlabel('Log₁₀(Centering Error)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('(a) Centering Error Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Scaling Error Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        scaling_errors = self.extract_metric('cross_step_validation.final_scaling_error')
        scaling_errors_pos = [e for e in scaling_errors if e > 0]
        if scaling_errors_pos:
            ax2.hist(np.log10(scaling_errors_pos), bins=30, color='forestgreen', alpha=0.7, edgecolor='black')
            ax2.axvline(np.log10(1e-6), color='red', linestyle='--', linewidth=2, label='Target threshold')
            ax2.set_xlabel('Log₁₀(Scaling Error)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('(b) Scaling Error Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Alignment Quality Distribution
        ax3 = fig.add_subplot(gs[0, 2])
        alignment_quality = self.extract_metric('alignment_validation.alignment_quality')
        if alignment_quality:
            ax3.hist(alignment_quality, bins=30, color='coral', alpha=0.7, edgecolor='black')
            ax3.axvline(np.mean(alignment_quality), color='darkred', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(alignment_quality):.3f}')
            ax3.set_xlabel('Alignment Quality Score')
            ax3.set_ylabel('Frequency')
            ax3.set_title('(c) PCA Alignment Quality')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Success Rate by Category
        ax4 = fig.add_subplot(gs[1, :2])
        category_performance = self.data.get('category_performance', {})
        if category_performance:
            categories = []
            success_rates = []
            counts = []
            
            for cat, stats in category_performance.items():
                if stats['total'] > 0:
                    categories.append(cat)
                    success_rates.append(100 * stats['successful'] / stats['total'])
                    counts.append(stats['total'])
            
            # Sort by success rate
            sorted_indices = np.argsort(success_rates)[::-1][:20]  # Top 20 categories
            categories = [categories[i] for i in sorted_indices]
            success_rates = [success_rates[i] for i in sorted_indices]
            
            y_pos = np.arange(len(categories))
            bars = ax4.barh(y_pos, success_rates, color='teal', alpha=0.7, edgecolor='black')
            
            # Color code by success rate
            for bar, rate in zip(bars, success_rates):
                if rate >= 99:
                    bar.set_color('darkgreen')
                elif rate >= 95:
                    bar.set_color('forestgreen')
                elif rate >= 90:
                    bar.set_color('orange')
                else:
                    bar.set_color('crimson')
            
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(categories, fontsize=8)
            ax4.set_xlabel('Success Rate (%)')
            ax4.set_title('(d) Normalization Success Rate by Category (Top 20)')
            ax4.axvline(100, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax4.grid(True, alpha=0.3, axis='x')
        
        # 5. Overall Statistics Summary
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        
        summary_text = f"""
Overall Statistics:

Total Shapes: {len(self.validations)}

Centering:
  Mean: {np.mean(centering_errors):.2e}
  Max: {np.max(centering_errors):.2e}
  < 10⁻¹⁰: {100*np.mean(np.array(centering_errors) < 1e-10):.1f}%

Scaling:
  Mean: {np.mean(scaling_errors):.2e}
  Max: {np.max(scaling_errors):.2e}
  < 10⁻⁶: {100*np.mean(np.array(scaling_errors) < 1e-6):.1f}%

Alignment:
  Mean Quality: {np.mean(alignment_quality):.3f}
  Std: {np.std(alignment_quality):.3f}

Overall Success: {self.data['processing_summary']['success_rate']:.1f}%
        """
        
        ax5.text(0.1, 0.5, summary_text.strip(), fontsize=9, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle('Normalization Pipeline Validation Overview', fontsize=14, fontweight='bold')
        
        output_path = self.output_dir / 'fig1_validation_overview.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {output_path}")
        plt.close()
    
    def plot_step_progression(self):
        """Figure 2: Cross-step consistency analysis"""
        print("📈 Generating Figure 2: Step-by-Step Progression...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 1. Barycenter drift through pipeline
        ax1 = axes[0, 0]
        steps = ['original', 'resampled', 'translated', 'aligned', 'flipped', 'scaled']
        step_labels = ['Original', 'Remeshed', 'Translated', 'Aligned', 'Flipped', 'Scaled']
        
        mean_centers = []
        max_centers = []
        
        for step in steps:
            centers = self.extract_metric(f'centering_validation.{step}.distance_from_origin', default=np.nan)
            if centers:
                mean_centers.append(np.nanmean(centers))
                max_centers.append(np.nanmax(centers))
            else:
                mean_centers.append(np.nan)
                max_centers.append(np.nan)
        
        x_pos = np.arange(len(steps))
        ax1.plot(x_pos, mean_centers, 'o-', linewidth=2, markersize=8, label='Mean', color='steelblue')
        ax1.plot(x_pos, max_centers, 's--', linewidth=2, markersize=6, label='Maximum', color='coral')
        ax1.axhline(1e-10, color='red', linestyle='--', linewidth=1.5, label='Target threshold')
        ax1.set_yscale('log')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(step_labels, rotation=45, ha='right')
        ax1.set_ylabel('Distance from Origin')
        ax1.set_title('(a) Barycenter Drift Through Pipeline')
        ax1.legend()
        ax1.grid(True, alpha=0.3, which='both')
        
        # 2. Vertex displacement magnitudes
        ax2 = axes[0, 1]
        
        transitions = [
            ('original_to_resampled', 'Orig→Remesh'),
            ('resampled_to_translated', 'Remesh→Trans'),
            ('translated_to_aligned', 'Trans→Align'),
            ('aligned_to_flipped', 'Align→Flip'),
            ('flipped_to_scaled', 'Flip→Scale')
        ]
        
        displacement_data = []
        labels = []
        
        for trans_key, label in transitions:
            means = []
            for validation in self.validations:
                try:
                    disp = validation.get('vertex_displacement_analysis', {}).get(trans_key, {}).get('mean')
                    if disp is not None:
                        means.append(disp)
                except:
                    pass
            
            if means:
                displacement_data.append(means)
                labels.append(label)
        
        if displacement_data:
            bp = ax2.boxplot(displacement_data, tick_labels=labels, patch_artist=True, showfliers=False)
            for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'plum']):
                patch.set_facecolor(color)
            ax2.set_ylabel('Mean Vertex Displacement')
            ax2.set_title('(b) Vertex Displacement per Transformation')
            ax2.grid(True, alpha=0.3, axis='y')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Bounding box dimension evolution
        ax3 = axes[1, 0]
        
        bbox_dims_by_step = {step: {'x': [], 'y': [], 'z': []} for step in steps}
        
        for validation in self.validations:
            for step in steps:
                try:
                    bbox = validation['centering_validation'][step].get('barycenter')
                    # This is actually showing we need bounding box data
                    # We'll use what we have from scaling_validation
                    if step == 'scaled':
                        dims = validation.get('scaling_validation', {}).get('bounding_box', {}).get('dimensions')
                        if dims and len(dims) == 3:
                            bbox_dims_by_step[step]['x'].append(dims[0])
                            bbox_dims_by_step[step]['y'].append(dims[1])
                            bbox_dims_by_step[step]['z'].append(dims[2])
                except:
                    pass
        
        # Plot max dimension across steps
        max_dims_mean = []
        for step in steps:
            dims_x = bbox_dims_by_step[step]['x']
            dims_y = bbox_dims_by_step[step]['y']
            dims_z = bbox_dims_by_step[step]['z']
            
            if dims_x and dims_y and dims_z:
                max_dims = [max(x, y, z) for x, y, z in zip(dims_x, dims_y, dims_z)]
                max_dims_mean.append(np.mean(max_dims))
            else:
                max_dims_mean.append(np.nan)
        
        valid_indices = [i for i, v in enumerate(max_dims_mean) if not np.isnan(v)]
        if valid_indices:
            ax3.plot([steps[i] for i in valid_indices], [max_dims_mean[i] for i in valid_indices], 
                    'o-', linewidth=2, markersize=8, color='darkgreen')
            ax3.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label='Target (unit cube)')
            ax3.set_xticks(range(len(steps)))
            ax3.set_xticklabels(step_labels, rotation=45, ha='right')
            ax3.set_ylabel('Maximum Bounding Box Dimension')
            ax3.set_title('(c) Bounding Box Scaling Convergence')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Cumulative success rate
        ax4 = axes[1, 1]
        
        # Extract metrics for success rate calculation
        centering_errors = self.extract_metric('cross_step_validation.final_centering_error')
        scaling_errors = self.extract_metric('cross_step_validation.final_scaling_error')
        alignment_quality = self.extract_metric('alignment_validation.alignment_quality')
        
        success_metrics = {
            'Centered\n(< 10⁻¹⁰)': 100 * np.mean(np.array(centering_errors) < 1e-10) if centering_errors else 0,
            'Scaled\n(< 10⁻⁶)': 100 * np.mean(np.array(scaling_errors) < 1e-6) if scaling_errors else 0,
            'Aligned\n(quality > 0.9)': 100 * np.mean(np.array(alignment_quality) > 0.9) if alignment_quality else 0,
            'Flipped\n(success)': 100 * np.mean([v.get('flipping_validation', {}).get('flipping_successful', False) 
                                                 for v in self.validations]),
            'Overall\nSuccess': self.data['processing_summary']['success_rate']
        }
        
        metrics = list(success_metrics.keys())
        values = list(success_metrics.values())
        colors = ['green' if v >= 95 else 'orange' if v >= 90 else 'red' for v in values]
        
        bars = ax4.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax4.axhline(100, color='darkgreen', linestyle='--', linewidth=1, alpha=0.5, label='100%')
        ax4.axhline(95, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='95%')
        ax4.set_ylabel('Success Rate (%)')
        ax4.set_ylim([80, 105])
        ax4.set_title('(d) Per-Step Validation Success Rates')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        plt.suptitle('Cross-Step Consistency and Progression Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / 'fig2_step_progression.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {output_path}")
        plt.close()
    
    def plot_pca_analysis(self):
        """Figure 3: PCA and eigenvalue analysis"""
        print("📈 Generating Figure 3: PCA Quality Analysis...")
        
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        
        # Extract eigenvalue data
        lambda1_over_lambda2 = self.extract_metric('eigenvalue_analysis.lambda1_over_lambda2')
        lambda2_over_lambda3 = self.extract_metric('eigenvalue_analysis.lambda2_over_lambda3')
        condition_numbers = self.extract_metric('eigenvalue_analysis.condition_number')
        anisotropy_scores = self.extract_metric('eigenvalue_analysis.anisotropy_score')
        
        # 1. Eigenvalue ratio scatter
        ax1 = axes[0, 0]
        if lambda1_over_lambda2 and lambda2_over_lambda3:
            ax1.scatter(lambda1_over_lambda2, lambda2_over_lambda3, alpha=0.5, s=20, c='steelblue', edgecolors='black', linewidth=0.5)
            ax1.set_xlabel('λ₁ / λ₂ (Major/Medium)')
            ax1.set_ylabel('λ₂ / λ₃ (Medium/Minor)')
            ax1.set_title('(a) Eigenvalue Ratio Distribution')
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.grid(True, alpha=0.3, which='both')
            # Manual reference line for y=x in log-log axes
            min_val = max(min(lambda1_over_lambda2 + lambda2_over_lambda3), 1e-3)
            max_val = min(max(lambda1_over_lambda2 + lambda2_over_lambda3), 1e3)
            ref_vals = np.logspace(np.log10(min_val), np.log10(max_val), 100)
            ax1.plot(ref_vals, ref_vals, color='red', linestyle='--', linewidth=1, alpha=0.5, label='λ₁/λ₂ = λ₂/λ₃')
            ax1.legend()
        
        # 2. Condition number distribution
        ax2 = axes[0, 1]
        if condition_numbers:
            ax2.hist(np.log10(condition_numbers), bins=40, color='forestgreen', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Log₁₀(Condition Number κ)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('(b) Condition Number Distribution')
            ax2.axvline(np.log10(np.median(condition_numbers)), color='red', linestyle='--', 
                       linewidth=2, label=f'Median: {np.median(condition_numbers):.1f}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Anisotropy score
        ax3 = axes[0, 2]
        if anisotropy_scores:
            ax3.hist(anisotropy_scores, bins=40, color='coral', alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Anisotropy Score')
            ax3.set_ylabel('Frequency')
            ax3.set_title('(c) Shape Anisotropy Distribution')
            ax3.axvline(np.mean(anisotropy_scores), color='darkred', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(anisotropy_scores):.3f}')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Eigenvector alignment quality
        ax4 = axes[1, 0]
        major_to_x = self.extract_metric('alignment_validation.eigenvector_alignment.major_to_x')
        medium_to_y = self.extract_metric('alignment_validation.eigenvector_alignment.medium_to_y')
        minor_to_z = self.extract_metric('alignment_validation.eigenvector_alignment.minor_to_z')
        
        if major_to_x and medium_to_y and minor_to_z:
            alignment_data = [major_to_x, medium_to_y, minor_to_z]
            bp = ax4.boxplot(alignment_data, tick_labels=['Major→X', 'Medium→Y', 'Minor→Z'], 
                    patch_artist=True, showfliers=False)
            for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightyellow']):
                patch.set_facecolor(color)
            ax4.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label='Perfect alignment')
            ax4.set_ylabel('|Alignment| (dot product magnitude)')
            ax4.set_title('(d) Eigenvector-to-Axis Alignment')
            ax4.set_ylim([0.8, 1.05])
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Condition number vs category complexity
        ax5 = axes[1, 1]
        
        # Extract category and condition number pairs
        category_condition = {}
        for validation in self.validations:
            category = validation.get('category', 'Unknown')
            cond_num = validation.get('eigenvalue_analysis', {}).get('condition_number')
            if cond_num is not None and not np.isinf(cond_num):
                if category not in category_condition:
                    category_condition[category] = []
                category_condition[category].append(cond_num)
        
        # Plot top 15 categories by median condition number
        categories = list(category_condition.keys())
        medians = [np.median(category_condition[cat]) for cat in categories]
        
        sorted_indices = np.argsort(medians)[::-1][:15]
        top_categories = [categories[i] for i in sorted_indices]
        top_data = [category_condition[cat] for cat in top_categories]
        
        if top_data:
            bp = ax5.boxplot(top_data, tick_labels=top_categories, patch_artist=True, showfliers=False)
            for patch in bp['boxes']:
                patch.set_facecolor('lightcoral')
            ax5.set_ylabel('Condition Number κ')
            ax5.set_title('(e) Condition Number by Category (Top 15)')
            ax5.set_yscale('log')
            plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=7)
            ax5.grid(True, alpha=0.3, axis='y', which='both')
        
        # 6. Eigenvalue ordering verification
        ax6 = axes[1, 2]
        
        ordering_correct = []
        for validation in self.validations:
            is_correct = validation.get('alignment_validation', {}).get('eigenvalue_ordering_correct')
            if is_correct is not None:
                ordering_correct.append(is_correct)
        
        if ordering_correct:
            correct_count = sum(ordering_correct)
            total_count = len(ordering_correct)
            incorrect_count = total_count - correct_count
            
            ax6.pie([correct_count, incorrect_count], 
                   labels=[f'Correct\n({correct_count})', f'Incorrect\n({incorrect_count})'],
                   colors=['lightgreen', 'lightcoral'],
                   autopct='%1.1f%%',
                   startangle=90,
                   explode=(0.05, 0))
            ax6.set_title(f'(f) Eigenvalue Ordering Verification\n(Total: {total_count})')
        
        plt.suptitle('Principal Component Analysis Quality Assessment', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / 'fig3_pca_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {output_path}")
        plt.close()

        # --- Additional figure: Average explained variance (pie chart) ---
        # Robustly extract explained variance ratios from various possible locations in each validation record.
        explained_lists = []
        for v in self.validations:
            vals = None
            # Try common locations in order of preference
            # 1) pca.explained_variance_ratio
            pca_block = v.get('pca') or v.get('normalization_info', {}).get('pca') or v.get('normalization', {}).get('pca')
            if pca_block:
                vals = pca_block.get('explained_variance_ratio')

            # 2) alignment_validation.eigenvalues -> convert to explained ratio if present
            if vals is None:
                eigs = v.get('alignment_validation', {}).get('eigenvalues') or v.get('eigenvalue_analysis', {}).get('eigenvalues')
                if eigs and len(eigs) >= 3:
                    try:
                        eigs = np.array(eigs, dtype=float)
                        if eigs.sum() > 0:
                            vals = (eigs / eigs.sum()).tolist()
                    except Exception:
                        vals = None

            # 3) eigenvalue_analysis -> sometimes stores explained ratios directly
            if vals is None:
                vals = v.get('eigenvalue_analysis', {}).get('explained_variance_ratio')

            # 4) final fallback: look for any top-level 'explained_variance_ratio'
            if vals is None:
                vals = v.get('explained_variance_ratio')

            if vals and isinstance(vals, (list, tuple)) and len(vals) >= 1:
                # Ensure numeric and finite
                try:
                    arr_vals = [float(x) for x in vals]
                    # normalize if not summing to ~1.0
                    s = sum(arr_vals)
                    if s > 0:
                        arr_vals = [x / s for x in arr_vals]
                        explained_lists.append(arr_vals)
                except Exception:
                    continue

        # Filter out very short entries (require at least 3 components for stable PC1..PC3 aggregation when possible)
        if not explained_lists:
            return

        # Pad all arrays to the same length (up to max 10 components) by appending zeros
        max_len = max(len(x) for x in explained_lists)
        max_len = min(max_len, 10)
        padded = []
        for arr in explained_lists:
            a = list(arr[:max_len])
            if len(a) < max_len:
                a += [0.0] * (max_len - len(a))
            padded.append(a)

        arr = np.array(padded, dtype=float)
        mean_explained = np.mean(arr, axis=0)

        # Aggregate into PC1, PC2, PC3, Other (if more components)
        if mean_explained.size > 3:
            first3 = mean_explained[:3].tolist()
            other = mean_explained[3:].sum()
            labels = ['PC1', 'PC2', 'PC3', 'Other']
            sizes = first3 + [other]
        else:
            labels = [f'PC{i+1}' for i in range(mean_explained.size)]
            sizes = mean_explained.tolist()

        # Convert to percentages for display (sums to ~100)
        sizes_pct = [s * 100 for s in sizes]

        fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
        N = len(explained_lists)
        # autopct that shows percent on first line and estimated count on second line
        wedges, texts, autotexts = ax_pie.pie(
            sizes_pct,
            labels=labels,
            autopct=lambda pct: f"{pct:.1f}%",
            startangle=90,
            colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        )
        ax_pie.set_title(f'Average Variance Explained by Each Principal Component (N = {N})')
        # Add a small annotation below the pie with the exact mean percentages for reproducibility
        pct_text = ', '.join([f'{lab}: {val:.1f}%' for lab, val in zip(labels, [s*100 for s in sizes])])
        ax_pie.text(0, -1.15, pct_text, ha='center', va='top', fontsize=8, wrap=True)
        pie_out = self.output_dir / 'fig3_pca_explained_variance_pie.png'
        plt.savefig(pie_out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved PCA explained-variance pie: {pie_out}")
    
    def plot_moment_and_symmetry(self):
        """Figure 4: Moment test and symmetry analysis"""
        print("📈 Generating Figure 4: Moment Test and Symmetry Analysis...")

        # Create a single row with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)

        # Extract moment values
        moment_x = []
        moment_y = []
        moment_z = []

        for validation in self.validations:
            moments = validation.get('flipping_validation', {}).get('moment_test_values')
            if moments and len(moments) == 3:
                moment_x.append(moments[0])
                moment_y.append(moments[1])
                moment_z.append(moments[2])

        # 2. Moment sign consistency (left)
        ax2 = axes[0]

        all_positive = sum(1 for x, y, z in zip(moment_x, moment_y, moment_z)
                           if x >= -1e-10 and y >= -1e-10 and z >= -1e-10)
        has_negative = len(moment_x) - all_positive

        ax2.bar(['All Positive', 'Has Negative'], [all_positive, has_negative],
                color=['darkgreen', 'crimson'], alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Number of Shapes')
        ax2.set_title('(a) Moment Sign Consistency')
        ax2.grid(True, alpha=0.3, axis='y')

        # Add percentage labels (guard against division by zero)
        total = len(moment_x)
        if total > 0:
            for i, (label, value) in enumerate(zip(['All Positive', 'Has Negative'], [all_positive, has_negative])):
                ax2.text(i, value + max(1, total * 0.02), f'{100*value/total:.1f}%', ha='center', fontweight='bold')

        # 4. Symmetry classification (middle)
        ax4 = axes[1]

        symmetry_counts = {'Spherical': 0, 'Cylindrical': 0, 'Asymmetric': 0}

        for validation in self.validations:
            sym_class = validation.get('symmetry_analysis', {}).get('symmetry_classification', {})
            if sym_class.get('spherical', False):
                symmetry_counts['Spherical'] += 1
            elif sym_class.get('cylindrical', False):
                symmetry_counts['Cylindrical'] += 1
            elif sym_class.get('asymmetric', False):
                symmetry_counts['Asymmetric'] += 1

        if sum(symmetry_counts.values()) > 0:
            labels = list(symmetry_counts.keys())
            values = list(symmetry_counts.values())
            colors_sym = ['gold', 'skyblue', 'lightcoral']

            wedges, texts, autotexts = ax4.pie(values, labels=labels, colors=colors_sym, autopct='%1.1f%%',
                                              startangle=90, explode=(0.05, 0.02, 0))
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontweight('bold')
            ax4.set_title('(b) Shape Symmetry Classification')

        # 6. Moment magnitude vs symmetry (right)
        ax6 = axes[2]

        avg_moment_magnitudes = []
        symmetry_types = []

        for validation in self.validations:
            moments = validation.get('flipping_validation', {}).get('moment_test_values')
            sym_class = validation.get('symmetry_analysis', {}).get('symmetry_classification', {})

            if moments and len(moments) == 3:
                avg_mag = np.mean(np.abs(moments))
                avg_moment_magnitudes.append(avg_mag)

                if sym_class.get('spherical', False):
                    symmetry_types.append('Spherical')
                elif sym_class.get('cylindrical', False):
                    symmetry_types.append('Cylindrical')
                else:
                    symmetry_types.append('Asymmetric')

        if avg_moment_magnitudes and symmetry_types:
            df_moments = pd.DataFrame({
                'Moment Magnitude': avg_moment_magnitudes,
                'Symmetry': symmetry_types
            })

            sym_order = ['Spherical', 'Cylindrical', 'Asymmetric']
            sym_data = [df_moments[df_moments['Symmetry'] == sym]['Moment Magnitude'].values
                        for sym in sym_order if sym in df_moments['Symmetry'].values]
            sym_labels = [sym for sym in sym_order if sym in df_moments['Symmetry'].values]

            bp = ax6.boxplot(sym_data, labels=sym_labels, patch_artist=True, showfliers=False)
            for patch, color in zip(bp['boxes'], ['gold', 'skyblue', 'lightcoral'][:len(sym_data)]):
                patch.set_facecolor(color)
            ax6.set_ylabel('Average |Moment| Magnitude')
            ax6.set_title('(c) Moment Magnitude by Symmetry Type')
            ax6.set_yscale('log')
            ax6.grid(True, alpha=0.3, axis='y', which='both')

        plt.suptitle('Moment Test and Symmetry Analysis', fontsize=16, fontweight='bold', y=1)
        plt.tight_layout()
        output_path = self.output_dir / 'fig4_moment_symmetry.png'
        plt.savefig(output_path, dpi=300)
        print(f"   ✅ Saved: {output_path}")
        plt.close()
    
    def plot_numerical_precision(self):
        """Figure 5: Numerical precision and robustness"""
        print("📈 Generating Figure 5: Numerical Precision Analysis...")
        
        fig, ax = plt.subplots(figsize=(14, 8))

        centering_errors = self.extract_metric('cross_step_validation.final_centering_error')
        scaling_errors = self.extract_metric('cross_step_validation.final_scaling_error')

        # Centering error CDF (single panel)
        if centering_errors:
            sorted_errors = np.sort(centering_errors)
            cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

            ax.plot(sorted_errors, cdf * 100, linewidth=2, color='steelblue')
            ax.axvline(1e-10, color='red', linestyle='--', linewidth=2, label='Target (10⁻¹⁰)')
            ax.axvline(1e-13, color='orange', linestyle='--', linewidth=1.5, label='Machine ε region')
            ax.set_xscale('log')
            ax.set_xlabel('Centering Error')
            ax.set_ylabel('Cumulative Percentage (%)')
            ax.set_title('Centering Error CDF')
            ax.grid(True, alpha=0.3, which='both')
            ax.legend()

            # Add annotation for percentage below threshold
            pct_below = 100 * np.mean(np.array(centering_errors) < 1e-10)
            ax.text(0.6, 0.3, f'{pct_below:.1f}% below\ntarget threshold',
                    transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # # 2. Scaling error CDF
        # ax2 = axes[0, 1]
        # if scaling_errors:
        #     sorted_errors = np.sort(scaling_errors)
        #     cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            
        #     ax2.plot(sorted_errors, cdf * 100, linewidth=2, color='forestgreen')
        #     ax2.axvline(1e-6, color='red', linestyle='--', linewidth=2, label='Target (10⁻⁶)')
        #     ax2.axvline(1e-10, color='orange', linestyle='--', linewidth=1.5, label='Machine ε region')
        #     ax2.set_xscale('log')
        #     ax2.set_xlabel('Scaling Error')
        #     ax2.set_ylabel('Cumulative Percentage (%)')
        #     ax2.set_title('(b) Scaling Error CDF')
        #     ax2.grid(True, alpha=0.3, which='both')
        #     ax2.legend()
            
        #     pct_below = 100 * np.mean(np.array(scaling_errors) < 1e-6)
        #     ax2.text(0.6, 0.3, f'{pct_below:.1f}% below\ntarget threshold',
        #             transform=ax2.transAxes, fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3. Two-pass recentering analysis
        # ax3 = axes[0, 2]
        
        # recentered_count = 0
        # not_recentered_count = 0
        
        # for validation in self.validations:
        #     triggered = validation.get('recentering_analysis', {}).get('second_pass_triggered', False)
        #     if triggered:
        #         recentered_count += 1
        #     else:
        #         not_recentered_count += 1
        
        # if recentered_count + not_recentered_count > 0:
        #     labels = ['Two-Pass\nRecentering', 'Single Pass']
        #     sizes = [recentered_count, not_recentered_count]
        #     colors = ['lightcoral', 'lightgreen']
        #     explode = (0.1, 0)
            
        #     wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        #                                         startangle=90, explode=explode)
        #     for autotext in autotexts:
        #         autotext.set_color('black')
        #         autotext.set_fontweight('bold')
        #     ax3.set_title(f'(c) Recentering Pass Requirements\n(Total: {recentered_count + not_recentered_count})')
        
        # # 4. Error threshold compliance
        # ax4 = axes[1, 0]
        
        # thresholds = [1e-15, 1e-13, 1e-10, 1e-8]
        # threshold_labels = ['10⁻¹⁵\n(Machine ε)', '10⁻¹³', '10⁻¹⁰\n(Target)', '10⁻⁸']
        
        # centering_compliance = [100 * np.mean(np.array(centering_errors) < thresh) for thresh in thresholds]
        # scaling_compliance = [100 * np.mean(np.array(scaling_errors) < thresh * 1e4) for thresh in thresholds]  # Scaled thresholds
        
        # x = np.arange(len(threshold_labels))
        # width = 0.35
        
        # bars1 = ax4.bar(x - width/2, centering_compliance, width, label='Centering', color='steelblue', alpha=0.7, edgecolor='black')
        # bars2 = ax4.bar(x + width/2, scaling_compliance, width, label='Scaling', color='forestgreen', alpha=0.7, edgecolor='black')
        
        # ax4.set_ylabel('Compliance Rate (%)')
        # ax4.set_xlabel('Error Threshold')
        # ax4.set_title('(d) Error Threshold Compliance')
        # ax4.set_xticks(x)
        # ax4.set_xticklabels(threshold_labels)
        # ax4.legend()
        # ax4.grid(True, alpha=0.3, axis='y')
        # ax4.set_ylim([0, 105])
        
        # 5. Aspect ratio preservation
        # ax5 = axes[0, 1]
        
        # aspect_errors = self.extract_metric('aspect_ratio_analysis.preservation_error')
        
        # if aspect_errors:
        #     ax5.hist(np.log10(aspect_errors), bins=40, color='purple', alpha=0.7, edgecolor='black')
        #     ax5.axvline(np.log10(1e-6), color='red', linestyle='--', linewidth=2, label='Target (10⁻⁶)')
        #     ax5.set_xlabel('Log₁₀(Aspect Ratio Preservation Error)')
        #     ax5.set_ylabel('Frequency')
        #     ax5.set_title('(e) Aspect Ratio Preservation')
        #     ax5.legend()
        #     ax5.grid(True, alpha=0.3)
            
        #     pct_preserved = 100 * np.mean(np.array(aspect_errors) < 1e-6)
        #     ax5.text(0.6, 0.8, f'{pct_preserved:.1f}% perfectly\npreserved',
        #             transform=ax5.transAxes, fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 6. Precision summary table
        # ax6 = axes[1, 0]
        # ax6.axis('off')
        
        # summary_data = [
        #     ['Metric', 'Mean', 'Max', '< Target'],
        #     ['', '', '', ''],
        #     ['Centering Error', f'{np.mean(centering_errors):.2e}', f'{np.max(centering_errors):.2e}', 
        #      f'{100*np.mean(np.array(centering_errors) < 1e-10):.1f}%'],
        #     ['Scaling Error', f'{np.mean(scaling_errors):.2e}', f'{np.max(scaling_errors):.2e}', 
        #      f'{100*np.mean(np.array(scaling_errors) < 1e-6):.1f}%'],
        # ]
        
        # if aspect_errors:
        #     summary_data.append(['Aspect Error', f'{np.mean(aspect_errors):.2e}', f'{np.max(aspect_errors):.2e}', 
        #                         f'{100*np.mean(np.array(aspect_errors) < 1e-6):.1f}%'])
        
        # table = ax6.table(cellText=summary_data, cellLoc='center', loc='center',
        #                  colWidths=[0.3, 0.25, 0.25, 0.2])
        # table.auto_set_font_size(False)
        # table.set_fontsize(9)
        # table.scale(1, 2)
        
        # # Style header row
        # for i in range(4):
        #     table[(0, i)].set_facecolor('#40466e')
        #     table[(0, i)].set_text_props(weight='bold', color='white')
        
        # # Style data rows
        # for i in range(2, len(summary_data)):
        #     for j in range(4):
        #         if i % 2 == 0:
        #             table[(i, j)].set_facecolor('#f0f0f0')
        
        # ax6.set_title('(f) Precision Summary', fontweight='bold', pad=20)
        
        plt.suptitle('Numerical Precision — Centering Error CDF', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / 'fig5_numerical_precision.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {output_path}")
        plt.close()
    
    def plot_category_performance_heatmap(self):
        """Figure 6: Category performance heatmap"""
        print("📈 Generating Figure 6: Category Performance Heatmap...")
        
        # Collect per-category metrics
        category_metrics = {}
        
        for validation in self.validations:
            category = validation.get('category', 'Unknown')
            if category not in category_metrics:
                category_metrics[category] = {
                    'centering': [],
                    'scaling': [],
                    'alignment': [],
                    'flipping': [],
                    'overall': []
                }
            
            # Collect metrics
            centering_ok = validation.get('cross_step_validation', {}).get('final_centering_error', 1) < 1e-10
            scaling_ok = validation.get('cross_step_validation', {}).get('final_scaling_error', 1) < 1e-6
            alignment_ok = validation.get('alignment_validation', {}).get('alignment_quality', 0) > 0.9
            flipping_ok = validation.get('flipping_validation', {}).get('flipping_successful', False)
            overall_ok = validation.get('cross_step_validation', {}).get('overall_normalization_success', False)
            
            category_metrics[category]['centering'].append(centering_ok)
            category_metrics[category]['scaling'].append(scaling_ok)
            category_metrics[category]['alignment'].append(alignment_ok)
            category_metrics[category]['flipping'].append(flipping_ok)
            category_metrics[category]['overall'].append(overall_ok)
        
        # Compute success rates
        categories = []
        heatmap_data = []
        
        for category, metrics in category_metrics.items():
            if len(metrics['overall']) >= 3:  # Only include categories with at least 3 shapes
                categories.append(category)
                row = [
                    100 * np.mean(metrics['centering']),
                    100 * np.mean(metrics['scaling']),
                    100 * np.mean(metrics['alignment']),
                    100 * np.mean(metrics['flipping']),
                    100 * np.mean(metrics['overall'])
                ]
                heatmap_data.append(row)
        
        # Sort by overall success rate
        sorted_indices = np.argsort([row[4] for row in heatmap_data])[::-1]
        categories = [categories[i] for i in sorted_indices]
        heatmap_data = [heatmap_data[i] for i in sorted_indices]
        
        # Create a combined figure: heatmap (left) + three scatter panels for final bbox dimensions (right)
        fig = plt.figure(figsize=(14, max(8, len(categories) * 0.3)))
        gs = GridSpec(1, 4, figure=fig, width_ratios=[2, 1, 1, 1], wspace=0.4)

        # Heatmap in the first (wide) column
        ax = fig.add_subplot(gs[0, 0])
        heatmap_array = np.array(heatmap_data)
        im = ax.imshow(heatmap_array, cmap='RdYlGn', aspect='auto', vmin=80, vmax=100)

        # Set ticks and labels for heatmap
        ax.set_xticks(np.arange(5))
        ax.set_yticks(np.arange(len(categories)))
        ax.set_xticklabels(['Centering', 'Scaling', 'Alignment', 'Flipping', 'Overall'])
        ax.set_yticklabels(categories)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Add text annotations inside heatmap cells
        for i in range(len(categories)):
            for j in range(5):
                ax.text(j, i, f'{heatmap_array[i, j]:.1f}%', ha='center', va='center',
                        color='black', fontsize=7, fontweight='bold')

        # Colorbar for heatmap
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Success Rate (%)', rotation=270, labelpad=20)
        ax.set_title('Category-Specific Normalization Performance Heatmap', fontsize=14, fontweight='bold', pad=20)

        # Save the heatmap only
        plt.tight_layout()
        output_path = self.output_dir / 'fig6_category_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {output_path}")
        plt.close()

        # --- Create a separate figure with scatter plots of final bounding-box dimensions for all shapes ---
        metadata_glob = os.path.join(str(self.output_dir.parent), '*', '*_metadata.json')
        dims_all = []
        cats_all = []
        names_all = []
        for meta_path in glob.glob(metadata_glob):
            try:
                with open(meta_path, 'r') as mf:
                    meta = json.load(mf)
                # try multiple common locations for final bbox dims
                dims = meta.get('normalization_info', {}).get('final', {}).get('bounding_box', {}).get('dimensions')
                if dims is None:
                    dims = meta.get('final', {}).get('bounding_box', {}).get('dimensions')
                if dims and isinstance(dims, (list, tuple)) and len(dims) == 3:
                    dims_all.append([float(d) for d in dims])
                    cats_all.append(meta.get('category', 'Unknown'))
                    names_all.append(Path(meta_path).stem)
            except Exception:
                continue

        if dims_all:
            # Group dims by category for per-category subplots
            category_grouped = {}
            for dims, cat, name in zip(dims_all, cats_all, names_all):
                category_grouped.setdefault(cat, []).append((dims, name))

            # Prepare subplot grid: up to 5 columns per row
            cats_with_data = [c for c, lst in category_grouped.items() if lst]
            if not cats_with_data:
                return
            n_cats = len(cats_with_data)
            cols = min(5, n_cats)
            rows = int(np.ceil(n_cats / cols))
            fig2, axs2 = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 3.0), squeeze=False)

            for idx, cat in enumerate(cats_with_data):
                r = idx // cols
                c = idx % cols
                ax = axs2[r][c]
                entries = category_grouped[cat]
                arr = np.array([e[0] for e in entries])
                n = arr.shape[0]
                # jitter x positions for readability
                jitter = (np.random.RandomState(123).randn(n) * 0.06)
                xcoords = np.arange(n) + jitter
                ax.scatter(xcoords, arr[:, 0], s=18, alpha=0.8, color='red', label='X')
                ax.scatter(xcoords, arr[:, 1], s=18, alpha=0.8, color='green', label='Y')
                ax.scatter(xcoords, arr[:, 2], s=18, alpha=0.8, color='blue', label='Z')
                ax.set_title(f'{cat} (n={n})')
                ax.set_xlabel('Shape index (within category)')
                ax.set_ylabel('Dimension (units)')
                ax.grid(True, alpha=0.2)
                # Do not draw per-subplot legends (we'll add a single figure legend)
                pass

            # Hide any unused subplots
            total_plots = rows * cols
            for extra in range(n_cats, total_plots):
                r = extra // cols
                c = extra % cols
                axs2[r][c].axis('off')

            # Place the title slightly lower (y) so it sits closer to the subplots
            fig2.suptitle('Final Bounding Box Dimensions: per-category scatter (X=red, Y=green, Z=blue)', y=0.995)
            out2 = self.output_dir / 'fig6_bbox_dimensions_by_category.png'
            # Add a single figure-level legend in the top-right (below the suptitle)
            from matplotlib.lines import Line2D
            handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=6),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=6),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=6)]
            labels = ['X', 'Y', 'Z']
            # Slightly reduce top margin so title isn't too far; place legend below title at top-right
            fig2.tight_layout(rect=[0, 0, 1, 0.97])
            # Anchor legend slightly lower to avoid touching the title
            fig2.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.9), frameon=False)
            plt.savefig(out2, dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved: {out2}")
            plt.close(fig2)
    
    def plot_pca_axis_angle_violinplot(self):
        """Violin plot: Distribution of angles between sorted PCA axes and world axes"""
        import numpy as np
        import matplotlib.pyplot as plt
        import glob
        import os
        import json
        print("📈 Generating PCA axis-to-world angle violin plots...")
        world_axes = np.eye(3)
        colors = ['red', 'green', 'blue']
        axis_labels = ['X-axis', 'Y-axis', 'Z-axis']
        axis_angles = [[], [], []]  # For each world axis
        # Find metadata files
        base_dir = os.path.join(str(self.output_dir.parent), '*', '*_metadata.json')
        metadata_files = glob.glob(base_dir)
        print(f"[INFO] Scanning {len(metadata_files)} metadata files for eigenvectors...")
        for meta_path in metadata_files:
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                pca_block = meta.get('normalization_info', {}).get('pca')
                eigvecs = None
                eigvals = None
                if pca_block and 'eigenvectors' in pca_block and 'eigenvalues' in pca_block:
                    eigvecs = np.array(pca_block['eigenvectors'])
                    eigvals = np.array(pca_block['eigenvalues'])
                if eigvecs is None or eigvals is None or eigvecs.shape != (3, 3) or eigvals.shape[0] != 3:
                    continue
                # Sort PCA axes by eigenvalue magnitude (descending)
                sort_idx = np.argsort(-eigvals)
                eigvecs_sorted = eigvecs[sort_idx]
                # For each world axis, find the angle to the corresponding sorted PCA axis
                for j in range(3):
                    angle = np.arccos(np.clip(abs(np.dot(eigvecs_sorted[j], world_axes[j])), -1.0, 1.0)) * 180.0 / np.pi
                    axis_angles[j].append(angle)
            except Exception as e:
                print(f"[ERROR] Failed to process {meta_path}: {e}")
                continue
        print(f"[DEBUG] axis_angles population: {[len(a) for a in axis_angles]}")
        for idx, arr in enumerate(axis_angles):
            print(f"[DEBUG] axis_angles[{idx}] first 10: {arr[:10]}")
            if arr:
                print(f"[DEBUG] axis_angles[{idx}] min: {min(arr)}, max: {max(arr)}")
        # Prepare data for violin plot
        data = []
        for i in range(3):
            vals = [float(v) for v in axis_angles[i] if v is not None and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))]
            vals = [v for v in vals if v >= 0.01]
            data.append(vals)
        fig, ax = plt.subplots(figsize=(8, 5))
        parts = ax.violinplot(data, showmeans=True, showmedians=True, showextrema=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(axis_labels)
        ax.set_ylabel('Angle (degrees)')
        ax.set_title('Violin Plot: Angle between sorted PCA axes and world axes (corresponding axes)')
        ax.set_ylim(0, 180)
        ax.grid(True, alpha=0.3, axis='y')
        output_path = self.output_dir / 'fig_pca_axis_angle_violinplot.png'
        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"   ✅ Saved: {output_path}")
    def generate_all_plots(self):
        """Generate all validation plots"""
        print("\n" + "="*60)
        print("  GENERATING COMPREHENSIVE VALIDATION PLOTS")
        print("="*60 + "\n")
        
        try:
            self.plot_overview_panel()
        except Exception as e:
            print(f"   ⚠️ Error generating Figure 1: {e}")
        
        try:
            self.plot_step_progression()
        except Exception as e:
            print(f"   ⚠️ Error generating Figure 2: {e}")
        
        try:
            self.plot_pca_analysis()
        except Exception as e:
            print(f"   ⚠️ Error generating Figure 3: {e}")
        try:
            self.plot_pca_axis_angle_boxplot()
        except Exception as e:
            print(f"   ⚠️ Error generating PCA axis angle boxplot: {e}")

        try:
            self.plot_pca_axis_angle_violinplot()
        except Exception as e:
            print(f"   ⚠️ Error generating PCA axis angle violinplot: {e}")
        
        try:
            self.plot_moment_and_symmetry()
        except Exception as e:
            print(f"   ⚠️ Error generating Figure 4: {e}")
        
        try:
            self.plot_numerical_precision()
        except Exception as e:
            print(f"   ⚠️ Error generating Figure 5: {e}")
        
        try:
            self.plot_category_performance_heatmap()
        except Exception as e:
            print(f"   ⚠️ Error generating Figure 6: {e}")
        
        print("\n" + "="*60)
        print("  PLOT GENERATION COMPLETE")
        print("="*60)
        print(f"\n📁 All figures saved to: {self.output_dir.absolute()}\n")


def main():    
    """
        Main function to run the validation plot generator.
    """

    BASE = Path(__file__).parent.parent.parent.parent.resolve()
    SOURCE_ROOT = BASE / 'Datasets' / 'UnifiedPreprocessed' / 'Data'

    input_path = SOURCE_ROOT  / 'validation_detailed.json'
    output_path = SOURCE_ROOT / 'Validation_Figures'
    
    
    # Check if input exists
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        print(f"\nPlease run normalize_database.py first to generate validation data.")
        return 1
    
    # Generate plots
    generator = ValidationPlotGenerator(input_path, output_path)
    generator.generate_all_plots()
    
    return 0


if __name__ == '__main__':
    exit(main())
