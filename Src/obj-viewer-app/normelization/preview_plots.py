"""
Quick preview of generated validation plots
Opens all figures in sequence for review
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import sys

def preview_plots(figures_dir='validation_figures'):
    """
    Display all generated validation plots for review
    """
    figures_dir = Path(figures_dir)
    
    if not figures_dir.exists():
        print(f"Error: Figures directory not found: {figures_dir}")
        print(f"\nPlease run generate_validation_plots.py first.")
        return False
    
    # Expected figures
    figure_files = [
        ('fig1_validation_overview.png', 'Figure 1: Validation Overview Panel'),
        ('fig2_step_progression.png', 'Figure 2: Step-by-Step Progression'),
        ('fig3_pca_analysis.png', 'Figure 3: PCA Quality Analysis'),
        ('fig4_moment_symmetry.png', 'Figure 4: Moment Test and Symmetry'),
        ('fig5_numerical_precision.png', 'Figure 5: Numerical Precision'),
        ('fig6_category_heatmap.png', 'Figure 6: Category Performance Heatmap')
    ]
    
    print("\n" + "="*70)
    print("  VALIDATION PLOTS PREVIEW")
    print("="*70 + "\n")
    
    found_figures = []
    missing_figures = []
    
    for filename, title in figure_files:
        fig_path = figures_dir / filename
        if fig_path.exists():
            found_figures.append((fig_path, title))
        else:
            missing_figures.append((filename, title))
    
    if missing_figures:
        print("️ Missing figures:")
        for filename, title in missing_figures:
            print(f"   • {filename}")
        print()
    
    if not found_figures:
        print("No figures found!")
        return False
    
    print(f"Found {len(found_figures)} figures")
    print(f"\n Press any key to view next figure, 'q'to quit, 's'to skip preview\n")
    
    choice = input("Start preview? (y/n/s): ").lower()
    
    if choice == 's' or choice == 'n':
        print("\n Figures are available in:", figures_dir.absolute())
        return True
    
    # Display each figure
    for i, (fig_path, title) in enumerate(found_figures, 1):
        print(f"\nShowing {i}/{len(found_figures)}: {title}")
        
        try:
            img = mpimg.imread(fig_path)
            
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error loading {fig_path.name}: {e}")
    
    print("\n" + "="*70)
    print("  PREVIEW COMPLETE")
    print("="*70)
    print(f"\n All figures saved in: {figures_dir.absolute()}")
    print("\n Tips for using in report:")
    print("   • All figures are 300 DPI (publication quality)")
    print("   • Use \\includegraphics[width=\\textwidth]{path/to/figure}")
    print("   • Reference figures using \\ref{fig:validation_overview}")
    print("   • Discuss key findings in figure captions")
    print()
    
    return True


def list_plots(figures_dir='validation_figures'):
    """
    List all available plots with file sizes
    """
    figures_dir = Path(figures_dir)
    
    if not figures_dir.exists():
        print(f"Error: Figures directory not found: {figures_dir}")
        return False
    
    print("\n" + "="*70)
    print("  AVAILABLE VALIDATION PLOTS")
    print("="*70 + "\n")
    
    plot_files = sorted(figures_dir.glob('*.png'))
    
    if not plot_files:
        print("No plots found in directory.")
        return False
    
    total_size = 0
    
    print(f"{'Filename':<40} {'Size':>10}")
    print("-" * 70)
    
    for plot_file in plot_files:
        size_bytes = plot_file.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        total_size += size_bytes
        print(f"{plot_file.name:<40} {size_mb:>8.2f} MB")
    
    total_mb = total_size / (1024 * 1024)
    print("-" * 70)
    print(f"{'TOTAL':<40} {total_mb:>8.2f} MB")
    print(f"\n Location: {figures_dir.absolute()}")
    print(f"Total plots: {len(plot_files)}")
    print()
    
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Preview or list validation plots',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--dir', '-d', 
                       default='validation_figures',
                       help='Directory containing figures')
    parser.add_argument('--list', '-l',
                       action='store_true',
                       help='List plots instead of previewing')
    
    args = parser.parse_args()
    
    if args.list:
        success = list_plots(args.dir)
    else:
        success = preview_plots(args.dir)
    
    sys.exit(0 if success else 1)
