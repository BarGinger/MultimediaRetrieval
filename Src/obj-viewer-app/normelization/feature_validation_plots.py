"""
Feature Extraction Validation Plots
====================================

This script creates comprehensive validation plots to demonstrate the effectiveness
of feature extraction and normalization processes. It compares features before and
after normalization to show that geometric properties are preserved while making
features scale-invariant.

Usage:
    Update the paths in the main() function configuration section, then run:
    python feature_validation_plots.py

Configuration:
    Edit the variables in main() to point to your CSV files:
    - original_csv: Path to CSV with features from original meshes
    - normalized_csv: Path to CSV with features from normalized meshes
    - output_dir_name: Directory name for saving plots
    - show_plots: Whether to display plots interactively
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib for high-quality plots
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Professional color palette
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent': '#F18F01',
    'success': '#C73E1D',
    'warning': '#F4D35E',
    'neutral': '#6C757D',
    'light': '#F8F9FA',
    'dark': '#343A40'
}

def load_and_prepare_data(original_path, normalized_path):
    """Load and prepare the feature data for analysis."""
    print("Loading data...")
    
    # Load datasets
    original_df = pd.read_csv(original_path)
    normalized_df = pd.read_csv(normalized_path)
    
    # Standardize column names
    for df in [original_df, normalized_df]:
        if 'class' in df.columns:
            df.rename(columns={'class': 'category'}, inplace=True)
        if 'shape_file' in df.columns:
            df.rename(columns={'shape_file': 'filename'}, inplace=True)
    
    print(f"Original dataset: {len(original_df)} shapes")
    print(f"Normalized dataset: {len(normalized_df)} shapes")
    
    # Identify feature columns (numeric columns excluding metadata)
    metadata_cols = ['category', 'filename', 'filepath']
    feature_cols = [col for col in original_df.columns 
                   if col not in metadata_cols and 
                   original_df[col].dtype in ['float64', 'int64']]
    
    print(f"Feature columns: {feature_cols}")
    
    return original_df, normalized_df, feature_cols

def create_output_directory(base_path="validation_plots"):
    """Create output directory for plots."""
    output_dir = Path(base_path)
    output_dir.mkdir(exist_ok=True)
    return output_dir

def plot_feature_correlations_before_after(original_df, normalized_df, feature_cols, output_dir, show_plots=False):
    """Plot 3: Feature correlation preservation before and after normalization."""
    print("Creating correlation preservation plots...")
    
    # Calculate correlations
    orig_corr = original_df[feature_cols].corr()
    norm_corr = normalized_df[feature_cols].corr()
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Original correlations
    sns.heatmap(orig_corr, annot=True, cmap='RdBu_r', center=0, 
                square=True, ax=ax1, cbar_kws={'shrink': 0.8})
    ax1.set_title('Feature Correlations\n(Original Data)', fontweight='bold', pad=20)
    
    # Normalized correlations  
    sns.heatmap(norm_corr, annot=True, cmap='RdBu_r', center=0,
                square=True, ax=ax2, cbar_kws={'shrink': 0.8})
    ax2.set_title('Feature Correlations\n(Normalized Data)', fontweight='bold', pad=20)
    
    # Correlation difference
    corr_diff = norm_corr - orig_corr
    sns.heatmap(corr_diff, annot=True, cmap='RdBu_r', center=0,
                square=True, ax=ax3, cbar_kws={'shrink': 0.8})
    ax3.set_title('Correlation Changes\n(Normalized - Original)', fontweight='bold', pad=20)
    
    plt.suptitle('Feature Correlation Analysis: Before vs After Normalization', 
                fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_dir / 'correlation_preservation.png')
    if show_plots:
        plt.show()
    plt.close()

def plot_feature_scaling_analysis(original_df, normalized_df, feature_cols, output_dir, show_plots=False):
    """Plot feature scaling and distribution changes."""
    print("Creating feature scaling analysis...")
    
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, feature in enumerate(feature_cols[:len(axes)]):
        if feature in original_df.columns and feature in normalized_df.columns:
            ax = axes[i]
            
            # Create histograms
            ax.hist(original_df[feature].dropna(), bins=30, alpha=0.7, 
                   color=COLORS['primary'], label='Original', density=True)
            ax.hist(normalized_df[feature].dropna(), bins=30, alpha=0.7,
                   color=COLORS['secondary'], label='Normalized', density=True)
            
            ax.set_title(f'{feature.replace("_", " ").title()}', fontweight='bold')
            ax.set_xlabel('Feature Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(feature_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Feature Distribution Changes: Original vs Normalized', 
                fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(output_dir / 'feature_scaling_analysis.png')
    if show_plots:
        plt.show()
    plt.close()

def plot_feature_scatter_comparison(original_df, normalized_df, feature_cols, output_dir, show_plots=False):
    """Plot before/after scatter plots showing correlation preservation."""
    print("Creating feature scatter comparison...")
    
    # Select key feature pairs for comparison
    feature_pairs = [
        ('num_vertices', 'num_faces'),
        ('compactness', 'rectangularity'),
        ('convexity', 'eccentricity')
    ]
    
    # Filter pairs that exist in data
    valid_pairs = [(f1, f2) for f1, f2 in feature_pairs 
                   if f1 in feature_cols and f2 in feature_cols]
    
    if not valid_pairs:
        print("No valid feature pairs found for scatter comparison")
        return
    
    fig, axes = plt.subplots(2, len(valid_pairs), figsize=(6*len(valid_pairs), 12))
    if len(valid_pairs) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (feat1, feat2) in enumerate(valid_pairs):
        # Original data
        axes[0, i].scatter(original_df[feat1], original_df[feat2], 
                          alpha=0.6, color=COLORS['primary'], s=30)
        axes[0, i].set_title(f'Original Data\n{feat1} vs {feat2}', fontweight='bold')
        axes[0, i].set_xlabel(feat1.replace('_', ' ').title())
        axes[0, i].set_ylabel(feat2.replace('_', ' ').title())
        axes[0, i].grid(True, alpha=0.3)
        
        # Normalized data
        axes[1, i].scatter(normalized_df[feat1], normalized_df[feat2],
                          alpha=0.6, color=COLORS['secondary'], s=30)
        axes[1, i].set_title(f'Normalized Data\n{feat1} vs {feat2}', fontweight='bold')
        axes[1, i].set_xlabel(feat1.replace('_', ' ').title())
        axes[1, i].set_ylabel(feat2.replace('_', ' ').title())
        axes[1, i].grid(True, alpha=0.3)
    
    plt.suptitle('Feature Relationships: Before vs After Normalization', 
                fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(output_dir / 'feature_scatter_comparison.png')
    if show_plots:
        plt.show()
    plt.close()

def plot_category_feature_distributions(df, feature_cols, title_suffix, output_dir, show_plots=False):
    """Plot feature distributions across categories."""
    print(f"Creating category feature distributions ({title_suffix})...")
    
    if 'category' not in df.columns:
        print("No category column found, skipping category analysis")
        return
    
    # Select top categories by count
    top_categories = df['category'].value_counts().head(8).index.tolist()
    df_subset = df[df['category'].isin(top_categories)]
    
    # Select key features for visualization
    key_features = ['num_vertices', 'num_faces', 'compactness', 'rectangularity']
    available_features = [f for f in key_features if f in feature_cols]
    
    if not available_features:
        available_features = feature_cols[:4]  # Use first 4 features if key ones not available
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(available_features[:4]):
        ax = axes[i]
        
        # Create box plot
        df_subset.boxplot(column=feature, by='category', ax=ax)
        ax.set_title(f'{feature.replace("_", " ").title()} by Category', fontweight='bold')
        ax.set_xlabel('Category')
        ax.set_ylabel(feature.replace('_', ' ').title())
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
        
        # Remove the automatic title from boxplot
        plt.setp(ax, title=f'{feature.replace("_", " ").title()} by Category')
    
    plt.suptitle(f'Feature Distributions Across Categories ({title_suffix})', 
                fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    filename = f'category_distributions_{title_suffix.lower().replace(" ", "_")}.png'
    plt.savefig(output_dir / filename)
    if show_plots:
        plt.show()
    plt.close()

def plot_pca_analysis(original_df, normalized_df, feature_cols, output_dir, show_plots=False):
    """Plot PCA analysis showing clustering and separability."""
    print("Creating PCA analysis...")
    
    # Prepare data for PCA
    orig_features = original_df[feature_cols].fillna(0)
    norm_features = normalized_df[feature_cols].fillna(0)
    
    # Standardize features
    scaler_orig = StandardScaler()
    scaler_norm = StandardScaler()
    
    orig_scaled = scaler_orig.fit_transform(orig_features)
    norm_scaled = scaler_norm.fit_transform(norm_features)
    
    # Apply PCA
    pca = PCA(n_components=2)
    orig_pca = pca.fit_transform(orig_scaled)
    norm_pca = pca.fit_transform(norm_scaled)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Get unique categories and colors
    if 'category' in original_df.columns:
        categories = original_df['category'].unique()[:10]  # Limit to 10 for visibility
        colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
        
        for i, category in enumerate(categories):
            orig_mask = original_df['category'] == category
            norm_mask = normalized_df['category'] == category
            
            ax1.scatter(orig_pca[orig_mask, 0], orig_pca[orig_mask, 1], 
                       c=[colors[i]], label=category, alpha=0.7, s=30)
            ax2.scatter(norm_pca[norm_mask, 0], norm_pca[norm_mask, 1],
                       c=[colors[i]], label=category, alpha=0.7, s=30)
        
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax1.scatter(orig_pca[:, 0], orig_pca[:, 1], alpha=0.7, color=COLORS['primary'])
        ax2.scatter(norm_pca[:, 0], norm_pca[:, 1], alpha=0.7, color=COLORS['secondary'])
    
    ax1.set_title('PCA: Original Features', fontweight='bold')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('PCA: Normalized Features', fontweight='bold')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Principal Component Analysis: Feature Space Visualization', 
                fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(output_dir / 'pca_analysis.png')
    if show_plots:
        plt.show()
    plt.close()

def plot_invariant_properties_validation(original_df, normalized_df, feature_cols, output_dir, show_plots=False):
    """Plot validation of properties that should remain invariant."""
    print("Creating invariant properties validation...")
    
    # Properties that should be invariant to scaling and translation
    invariant_features = ['compactness', 'rectangularity', 'convexity', 'eccentricity']
    available_invariant = [f for f in invariant_features if f in feature_cols]
    
    if not available_invariant:
        print("No invariant features found for validation")
        return
    
    n_features = len(available_invariant)
    fig, axes = plt.subplots(1, n_features, figsize=(5*n_features, 5))
    if n_features == 1:
        axes = [axes]
    
    for i, feature in enumerate(available_invariant):
        ax = axes[i]
        
        # Scatter plot of original vs normalized values
        ax.scatter(original_df[feature], normalized_df[feature], 
                  alpha=0.6, color=COLORS['primary'], s=30)
        
        # Add perfect correlation line
        min_val = min(original_df[feature].min(), normalized_df[feature].min())
        max_val = max(original_df[feature].max(), normalized_df[feature].max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'r--', linewidth=2, label='Perfect Correlation')
        
        # Calculate correlation
        corr = np.corrcoef(original_df[feature].fillna(0), 
                          normalized_df[feature].fillna(0))[0, 1]
        
        ax.set_title(f'{feature.replace("_", " ").title()}\n(r = {corr:.3f})', fontweight='bold')
        ax.set_xlabel('Original Values')
        ax.set_ylabel('Normalized Values')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Invariant Properties Validation\n(Should remain unchanged after normalization)', 
                fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(output_dir / 'invariant_properties_validation.png')
    if show_plots:
        plt.show()
    plt.close()

def generate_summary_report(original_df, normalized_df, feature_cols, output_dir):
    """Generate a summary report of the analysis."""
    print("Generating summary report...")
    
    report = []
    report.append("FEATURE EXTRACTION VALIDATION REPORT")
    report.append("=" * 50)
    report.append(f"Original dataset: {len(original_df)} shapes")
    report.append(f"Normalized dataset: {len(normalized_df)} shapes")
    report.append(f"Features analyzed: {len(feature_cols)}")
    report.append("")
    
    # Feature statistics
    report.append("FEATURE STATISTICS:")
    report.append("-" * 20)
    
    for feature in feature_cols:
        if feature in original_df.columns and feature in normalized_df.columns:
            orig_mean = original_df[feature].mean()
            norm_mean = normalized_df[feature].mean()
            orig_std = original_df[feature].std()
            norm_std = normalized_df[feature].std()
            
            report.append(f"{feature}:")
            report.append(f"  Original: μ={orig_mean:.3f}, σ={orig_std:.3f}")
            report.append(f"  Normalized: μ={norm_mean:.3f}, σ={norm_std:.3f}")
            report.append("")
    
    # Save report
    with open(output_dir / 'validation_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    print("Summary report saved")

def main():
    # =============================================================================
    # CONFIGURATION - Update these paths to your CSV files
    # =============================================================================
    
    # Path to original dataset analysis CSV
    original_csv = "../../Preprocessing/analysis_results.csv"
    
    # Path to normalized dataset analysis CSV  
    normalized_csv = "../../Datasets/UnifiedPreprocessed/analysis_results_data.csv"
    
    # Output directory for plots
    output_dir_name = "validation_plots"
    
    # Show plots interactively (True/False)
    show_plots = True
    
    # =============================================================================
    # END CONFIGURATION
    # =============================================================================
    
    print("Starting feature validation analysis...")
    print(f"Original CSV: {original_csv}")
    print(f"Normalized CSV: {normalized_csv}")
    
    # Load data
    original_df, normalized_df, feature_cols = load_and_prepare_data(
        original_csv, normalized_csv)
    
    # Create output directory
    output_dir = create_output_directory(output_dir_name)
    print(f"Saving plots to: {output_dir}")
    
    # Generate all plots
    plot_feature_correlations_before_after(original_df, normalized_df, feature_cols, 
                                          output_dir, show_plots)
    
    plot_feature_scaling_analysis(original_df, normalized_df, feature_cols, 
                                 output_dir, show_plots)
    
    plot_feature_scatter_comparison(original_df, normalized_df, feature_cols,
                                   output_dir, show_plots)
    
    plot_category_feature_distributions(original_df, feature_cols, "Original Data",
                                       output_dir, show_plots)
    
    plot_category_feature_distributions(normalized_df, feature_cols, "Normalized Data", 
                                       output_dir, show_plots)
    
    plot_pca_analysis(original_df, normalized_df, feature_cols, output_dir, show_plots)
    
    plot_invariant_properties_validation(original_df, normalized_df, feature_cols,
                                        output_dir, show_plots)
    
    # Generate summary report
    generate_summary_report(original_df, normalized_df, feature_cols, output_dir)
    
    print(f"\nValidation complete! All plots saved to: {output_dir}")
    print("Generated plots:")
    print("- correlation_preservation.png")
    print("- feature_scaling_analysis.png") 
    print("- feature_scatter_comparison.png")
    print("- category_distributions_original_data.png")
    print("- category_distributions_normalized_data.png")
    print("- pca_analysis.png")
    print("- invariant_properties_validation.png")
    print("- validation_report.txt")

if __name__ == "__main__":
    main()