"""
Example workflow: Run normalization and generate validation plots
"""

import subprocess
import sys
from pathlib import Path

def run_normalization_and_plots():
    """
    Complete workflow:
    1. Run normalization pipeline
    2. Generate validation plots
    3. Display summary
    """
    
    print("="*70)
    print("  NORMALIZATION PIPELINE & VALIDATION PLOT GENERATION")
    print("="*70)
    print()
    
    # Step 1: Check if validation data already exists
    validation_file = Path("../../Datasets/UnifiedPreprocessed/validation_detailed.json")
    
    if not validation_file.exists():
        print("Step 1: Running normalization pipeline...")
        print("   (This may take a while for large datasets)")
        print()
        
        try:
            result = subprocess.run(
                [sys.executable, "normalize_database.py"],
                check=True,
                capture_output=False
            )
            print("\n Normalization complete!")
        except subprocess.CalledProcessError as e:
            print(f"\n Error running normalization: {e}")
            return False
    else:
        print("Validation data already exists, skipping normalization")
        print(f"   Found: {validation_file}")
    
    print()
    print("="*70)
    print()
    
    # Step 2: Generate validation plots
    print("Step 2: Generating validation plots...")
    print()
    
    try:
        result = subprocess.run(
            [sys.executable, "generate_validation_plots.py"],
            check=True,
            capture_output=False
        )
        print("\n Validation plots generated!")
    except subprocess.CalledProcessError as e:
        print(f"\n Error generating plots: {e}")
        return False
    
    print()
    print("="*70)
    print("  WORKFLOW COMPLETE")
    print("="*70)
    print()
    print("Output locations:")
    print(f"   Validation data: {validation_file}")
    print(f"   Figures: ./validation_figures/")
    print()
    print("Generated figures:")
    print("   • fig1_validation_overview.png     - Multi-panel validation overview")
    print("   • fig2_step_progression.png        - Step-by-step progression analysis")
    print("   • fig3_pca_analysis.png            - PCA quality metrics")
    print("   • fig4_moment_symmetry.png         - Moment test and symmetry")
    print("   • fig5_numerical_precision.png     - Numerical precision analysis")
    print("   • fig6_category_heatmap.png        - Category performance heatmap")
    print()
    print("Next steps:")
    print("   1. Review the generated figures")
    print("   2. Include them in your report")
    print("   3. Reference validation statistics in your analysis")
    print()
    
    return True


if __name__ == '__main__':
    success = run_normalization_and_plots()
    sys.exit(0 if success else 1)
