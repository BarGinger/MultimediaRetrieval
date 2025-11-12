import open3d as o3d
from pathlib import Path
import statistics
import matplotlib.pyplot as plt
import numpy as np

# Compare original vs resampled files
datasets_dir = Path('Datasets')
original_dir = datasets_dir / 'Data'
resampled_dir = datasets_dir / 'UnifiedPreprocessed' / 'Data'
TARGET_VERTEX_COUNT = 7500
# Custom range: 5000-10000 vertices is acceptable (matching resampling.py)
MIN_ACCEPTABLE_VERTICES = 5000
MAX_ACCEPTABLE_VERTICES = 10000

def compare_datasets():
    import pandas as pd
    csv_dir = Path('Preprocessing') / 'csv_exports'
    csv_dir.mkdir(parents=True, exist_ok=True)
    all_comparisons_csv = csv_dir / 'all_comparisons.csv'
    original_vertex_csv = csv_dir / 'original_vertex_counts.csv'
    resampled_vertex_csv = csv_dir / 'resampled_vertex_counts.csv'
    min_target = MIN_ACCEPTABLE_VERTICES  # 5000
    max_target = MAX_ACCEPTABLE_VERTICES  # 10000

    # If CSVs exist, load them, else recalc and export
    if all_comparisons_csv.exists() and original_vertex_csv.exists() and resampled_vertex_csv.exists():
        print('ℹ️ Loading comparison data from CSVs (no recalculation)...')
        comparisons_df = pd.read_csv(all_comparisons_csv)
        original_vertex_counts = pd.read_csv(original_vertex_csv)['original_vertices'].tolist()
        resampled_vertex_counts = pd.read_csv(resampled_vertex_csv)['resampled_vertices'].tolist()
        comparisons = comparisons_df.to_dict('records')
    else:
        print("COMPARISON: Original vs Resampled Dataset")
        print("=" * 60)
        original_vertex_counts = []
        resampled_vertex_counts = []
        comparisons = []
        print(f"Target: {TARGET_VERTEX_COUNT} vertices")
        print(f"Acceptable range: {min_target} - {max_target} vertices")
        print("-" * 60)
        # Process each category
        for category_dir in original_dir.iterdir():
            if not category_dir.is_dir():
                continue
            resampled_category_dir = resampled_dir / category_dir.name
            if not resampled_category_dir.exists():
                continue
            for obj_file in category_dir.glob('*.obj'):
                filename = obj_file.name
                if "UnifiedPreprocessed" in str(resampled_dir):
                    filename = filename.replace(".obj", "_unified.obj")
                resampled_file = resampled_category_dir / filename
                if not resampled_file.exists():
                    continue
                try:
                    # Read original mesh
                    original_mesh = o3d.io.read_triangle_mesh(str(obj_file))
                    if original_mesh.is_empty():
                        continue
                    original_vertices = len(original_mesh.vertices)
                    original_faces = len(original_mesh.triangles)
                    # Read resampled mesh
                    resampled_mesh = o3d.io.read_triangle_mesh(str(resampled_file))
                    if resampled_mesh.is_empty():
                        continue
                    resampled_vertices = len(resampled_mesh.vertices)
                    resampled_faces = len(resampled_mesh.triangles)
                    # Store data
                    original_vertex_counts.append(original_vertices)
                    resampled_vertex_counts.append(resampled_vertices)
                    # Determine what action should have been taken
                    if original_vertices < min_target:
                        expected_action = "subdivide"
                    elif original_vertices > max_target:
                        expected_action = "simplify"
                    else:
                        expected_action = "unchanged"
                    # Determine what actually happened
                    if abs(resampled_vertices - original_vertices) < 10:
                        actual_action = "unchanged"
                    elif resampled_vertices > original_vertices:
                        actual_action = "subdivided"
                    else:
                        actual_action = "simplified"
                    comparisons.append({
                        'file': obj_file.name,
                        'category': category_dir.name,
                        'original_vertices': original_vertices,
                        'resampled_vertices': resampled_vertices,
                        'original_faces': original_faces,
                        'resampled_faces': resampled_faces,
                        'expected_action': expected_action,
                        'actual_action': actual_action,
                        'change': resampled_vertices - original_vertices
                    })
                except Exception as e:
                    print(f"Error processing {obj_file}: {e}")
        # Export all comparisons and vertex counts to CSVs
        pd.DataFrame(comparisons).to_csv(all_comparisons_csv, index=False)
        pd.DataFrame({'original_vertices': original_vertex_counts}).to_csv(original_vertex_csv, index=False)
        pd.DataFrame({'resampled_vertices': resampled_vertex_counts}).to_csv(resampled_vertex_csv, index=False)
    
    print(f"COMPARISON STATISTICS:")
    print(f"Total files compared: {len(comparisons)}")
    print()
    
    # Original dataset stats
    print(f"ORIGINAL DATASET:")
    print(f"  Average vertices: {statistics.mean(original_vertex_counts):.0f}")
    print(f"  Median vertices: {statistics.median(original_vertex_counts):.0f}")
    print(f"  Min vertices: {min(original_vertex_counts)}")
    print(f"  Max vertices: {max(original_vertex_counts)}")
    print(f"  Standard deviation: {statistics.stdev(original_vertex_counts):.0f}")
    
    # Resampled dataset stats
    print(f"\nRESAMPLED DATASET:")
    print(f"  Average vertices: {statistics.mean(resampled_vertex_counts):.0f}")
    print(f"  Median vertices: {statistics.median(resampled_vertex_counts):.0f}")
    print(f"  Min vertices: {min(resampled_vertex_counts)}")
    print(f"  Max vertices: {max(resampled_vertex_counts)}")
    print(f"  Standard deviation: {statistics.stdev(resampled_vertex_counts):.0f}")
    
    # Improvement analysis
    original_within_tolerance = sum(1 for v in original_vertex_counts if min_target <= v <= max_target)
    resampled_within_tolerance = sum(1 for v in resampled_vertex_counts if min_target <= v <= max_target)
    
    print(f"\nTARGET COMPLIANCE:")
    print(f"  Original within acceptable range: {original_within_tolerance}/{len(original_vertex_counts)} ({original_within_tolerance/len(original_vertex_counts)*100:.1f}%)")
    print(f"  Resampled within acceptable range: {resampled_within_tolerance}/{len(resampled_vertex_counts)} ({resampled_within_tolerance/len(resampled_vertex_counts)*100:.1f}%)")
    print(f"  Improvement: {resampled_within_tolerance - original_within_tolerance} files (+{(resampled_within_tolerance - original_within_tolerance)/len(original_vertex_counts)*100:.1f}%)")
    
    # Action analysis
    action_stats = {}
    for comp in comparisons:
        key = f"{comp['expected_action']} -> {comp['actual_action']}"
        action_stats[key] = action_stats.get(key, 0) + 1
    
    print(f"\nACTION ANALYSIS:")
    for action, count in sorted(action_stats.items()):
        percentage = count / len(comparisons) * 100
        print(f"  {action:<25}: {count:>3} files ({percentage:>5.1f}%)")
    
    # Show biggest improvements and failures
    print(f"\nBIGGEST IMPROVEMENTS (moved closer to target):")
    improvements = [c for c in comparisons if abs(c['resampled_vertices'] - TARGET_VERTEX_COUNT) < abs(c['original_vertices'] - TARGET_VERTEX_COUNT)]
    improvements.sort(key=lambda x: abs(x['original_vertices'] - TARGET_VERTEX_COUNT) - abs(x['resampled_vertices'] - TARGET_VERTEX_COUNT), reverse=True)
    
    for i, comp in enumerate(improvements[:5]):
        original_diff = abs(comp['original_vertices'] - TARGET_VERTEX_COUNT)
        resampled_diff = abs(comp['resampled_vertices'] - TARGET_VERTEX_COUNT)
        improvement = original_diff - resampled_diff
        print(f"  {i+1}. {comp['category']:<15} {comp['file']:<25} {comp['original_vertices']:>5} → {comp['resampled_vertices']:>5} (improved by {improvement:+.0f})")
    
    print(f"\nWORST CASES (still far from target):")
    worst_cases = sorted(comparisons, key=lambda x: abs(x['resampled_vertices'] - TARGET_VERTEX_COUNT), reverse=True)
    
    for i, comp in enumerate(worst_cases[:5]):
        diff = comp['resampled_vertices'] - TARGET_VERTEX_COUNT
        print(f"  {i+1}. {comp['category']:<15} {comp['file']:<25} {comp['original_vertices']:>5} → {comp['resampled_vertices']:>5} ({diff:+d} from target)")
    
    # Distribution comparison
    print(f"\nDISTRIBUTION COMPARISON:")
    ranges = [
        (0, 2000, "Very Small"),
        (2000, 5000, "Small"),
        (5000, 10000, "Acceptable Range"),
        (10000, 15000, "Large"),
        (15000, float('inf'), "Very Large")
    ]
    
    print(f"{'Range':<15} {'Original':<12} {'Resampled':<12} {'Change':<10}")
    print("-" * 50)
    for min_v, max_v, label in ranges:
        original_count = sum(1 for v in original_vertex_counts if min_v <= v < max_v)
        resampled_count = sum(1 for v in resampled_vertex_counts if min_v <= v < max_v)
        change = resampled_count - original_count
        
        original_pct = original_count / len(original_vertex_counts) * 100
        resampled_pct = resampled_count / len(resampled_vertex_counts) * 100
        
        print(f"{label:<15} {original_count:>3} ({original_pct:>4.1f}%) {resampled_count:>3} ({resampled_pct:>4.1f}%) {change:>+3}")
    
    # Create visualization plots
    print("\n" + "="*60)
    print("GENERATING ANALYSIS PLOTS...")
    print("="*60)
    try:
        create_analysis_plots(original_vertex_counts, resampled_vertex_counts, comparisons)
    except Exception as e:
        print(f"Error creating plots: {e}")
        print("   Make sure matplotlib is installed: pip install matplotlib")

def create_analysis_plots(original_vertex_counts, resampled_vertex_counts, comparisons):
    """Generate a single combined figure with two side-by-side subplots.

    Left: Target compliance stacked bar (original vs resampled)
    Right: Distribution by size categories (grouped bars)
    """

    import pandas as pd
    figures_dir = Path('Preprocessing') / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = Path('Preprocessing') / 'csv_exports'
    csv_dir.mkdir(parents=True, exist_ok=True)

    # --- CSV Export Logic ---
    compliance_csv = csv_dir / 'compliance_stats.csv'
    dist_csv = csv_dir / 'size_distribution.csv'
    inside_csv = csv_dir / 'files_within_range.csv'
    outside_csv = csv_dir / 'files_outside_range.csv'

    # If CSVs exist, load them, else recalc and export
    if compliance_csv.exists() and dist_csv.exists() and inside_csv.exists() and outside_csv.exists():
        print('ℹ️ Loading plot data from CSVs (no recalculation)...')
        compliance_df = pd.read_csv(compliance_csv)
        dist_df = pd.read_csv(dist_csv)
        files_within = pd.read_csv(inside_csv)
        files_outside = pd.read_csv(outside_csv)
        # Use loaded data for plotting (not implemented here, but can be added if needed)
    else:
        print('ℹ️ Calculating plot data and exporting to CSVs...')
        # Compliance DataFrame
        compliance_df = pd.DataFrame({
            'Category': ['Original', 'Resampled'],
            'WithinRange': [sum(1 for v in original_vertex_counts if MIN_ACCEPTABLE_VERTICES <= v <= MAX_ACCEPTABLE_VERTICES),
                            sum(1 for v in resampled_vertex_counts if MIN_ACCEPTABLE_VERTICES <= v <= MAX_ACCEPTABLE_VERTICES)],
            'OutsideRange': [sum(1 for v in original_vertex_counts if not (MIN_ACCEPTABLE_VERTICES <= v <= MAX_ACCEPTABLE_VERTICES)),
                             sum(1 for v in resampled_vertex_counts if not (MIN_ACCEPTABLE_VERTICES <= v <= MAX_ACCEPTABLE_VERTICES))]
        })
        compliance_df.to_csv(compliance_csv, index=False)

        # Distribution DataFrame
        size_ranges = [
            (0, 2000, "Very Small"),
            (2000, 5000, "Small"),
            (5000, 10000, "Acceptable"),
            (10000, 15000, "Large"),
            (15000, float('inf'), "Very Large")
        ]
        dist_rows = []
        for min_v, max_v, label in size_ranges:
            dist_rows.append({
                'Range': label,
                'OriginalCount': sum(1 for v in original_vertex_counts if min_v <= v < max_v),
                'ResampledCount': sum(1 for v in resampled_vertex_counts if min_v <= v < max_v)
            })
        dist_df = pd.DataFrame(dist_rows)
        dist_df.to_csv(dist_csv, index=False)

        # Files inside/outside range
        files_within = pd.DataFrame([comp for comp in comparisons if MIN_ACCEPTABLE_VERTICES <= comp['resampled_vertices'] <= MAX_ACCEPTABLE_VERTICES])
        files_outside = pd.DataFrame([comp for comp in comparisons if not (MIN_ACCEPTABLE_VERTICES <= comp['resampled_vertices'] <= MAX_ACCEPTABLE_VERTICES)])
        files_within.to_csv(inside_csv, index=False)
        files_outside.to_csv(outside_csv, index=False)

    # Prepare shared data
    original_within = sum(1 for v in original_vertex_counts if MIN_ACCEPTABLE_VERTICES <= v <= MAX_ACCEPTABLE_VERTICES)
    resampled_within = sum(1 for v in resampled_vertex_counts if MIN_ACCEPTABLE_VERTICES <= v <= MAX_ACCEPTABLE_VERTICES)
    categories = ['Original', 'Resampled']
    within_range = [original_within, resampled_within]
    total_files = len(original_vertex_counts)
    outside_range = [total_files - original_within, total_files - resampled_within]

    size_ranges = [
        (0, 2000, "Very Small"),
        (2000, 5000, "Small"),
        (5000, 10000, "Acceptable"),
        (10000, 15000, "Large"),
        (15000, float('inf'), "Very Large")
    ]
    range_labels = [r[2] for r in size_ranges]
    original_dist = []
    resampled_dist = []
    for min_v, max_v, _ in size_ranges:
        original_count = sum(1 for v in original_vertex_counts if min_v <= v < max_v)
        resampled_count = sum(1 for v in resampled_vertex_counts if min_v <= v < max_v)
        original_dist.append(original_count)
        resampled_dist.append(resampled_count)

    # Create combined figure with four subplots in 2x2 grid, larger size for better visibility
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Target Compliance & Size Distribution', fontsize=16, fontweight='bold')
    ax1, ax2 = axs[0]
    ax3, ax4 = axs[1]

    # --- Subplot 1: Target compliance (percentage) ---
    # --- Subplot 1 & 2: Target compliance as pie charts ---
    pie_labels = ['Within Range', 'Outside Range']
    pie_colors = ['#2ca02c', '#d62728']
    for i, cat in enumerate(categories):
        ax = [ax1, ax2][i]
        values = [within_range[i], outside_range[i]]
        # Use the correct total for this pie (original vs resampled). This prevents counts from
        # being computed against an incorrect total when original/resampled arrays have different lengths.
        pie_total = sum(values)
        # autopct should show percentage and absolute count derived from the pie_total.
        def make_autopct(pie_total):
            def autopct(pct):
                # Compute the absolute count from the percentage; round to nearest int for display
                count = int(round(pct / 100.0 * pie_total))
                return f'{pct:.1f}%\n({count})'
            return autopct
        autopct = make_autopct(pie_total)
        wedges, texts, autotexts = ax.pie(values, labels=pie_labels, colors=pie_colors, autopct=autopct, startangle=90, counterclock=False, textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax.set_title(f'{cat} Compliance', fontsize=13)
        ax.axis('equal')


    # --- Subplot 3: Size distribution (percentage) ---
    x2 = np.arange(len(range_labels))
    width2 = 0.45
    max_height = max(original_dist + resampled_dist) if original_dist else 0
    original_pct = [o / total_files * 100 for o in original_dist]
    resampled_pct = [r / total_files * 100 for r in resampled_dist]
    bars3 = ax3.bar(x2 - width2/2, original_pct, width2, label='Original', alpha=0.75, color='#ff7f0e')
    bars4 = ax3.bar(x2 + width2/2, resampled_pct, width2, label='Resampled', alpha=0.75, color='#1f77b4')
    ax3.set_xlabel('Vertex Count Range', fontsize=12)
    ax3.set_ylabel('Percentage of Files (%)', fontsize=12)
    ax3.set_title('Size Distribution (%)', fontsize=13)
    ax3.set_xticks(x2)
    ax3.set_xticklabels(range_labels, rotation=25, ha='right', fontsize=11)
    ax3.legend(frameon=False, loc='upper left', fontsize=11, 
            #    bbox_to_anchor=(1.01, 1)
               )
    ax3.grid(axis='y', alpha=0.25, linestyle='--')
    y_max3 = ax3.get_ylim()[1]
    margin3 = y_max3 * 0.04
    for bar in bars3:
        height = bar.get_height()
        if height > y_max3 * 0.15:
            ax3.text(bar.get_x() + bar.get_width()/2, height/2, f'{height:.1f}%', ha='center', va='center', fontweight='bold', color='black', fontsize=11)
        else:
            y = min(height + 2, y_max3 - margin3)
            ax3.text(bar.get_x() + bar.get_width()/2, y, f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', color='black', fontsize=11)
    for bar in bars4:
        height = bar.get_height()
        if height > y_max3 * 0.15:
            ax3.text(bar.get_x() + bar.get_width()/2, height/2, f'{height:.1f}%', ha='center', va='center', fontweight='bold', color='black', fontsize=11)
        else:
            y = min(height + 2, y_max3 - margin3)
            ax3.text(bar.get_x() + bar.get_width()/2, y, f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', color='black', fontsize=11)

    # --- Subplot 4: Size distribution (count) ---
    bars5 = ax4.bar(x2 - width2/2, original_dist, width2, label='Original', alpha=0.75, color='#ff7f0e')
    bars6 = ax4.bar(x2 + width2/2, resampled_dist, width2, label='Resampled', alpha=0.75, color='#1f77b4')
    ax4.set_xlabel('Vertex Count Range', fontsize=12)
    ax4.set_ylabel('Number of Files', fontsize=12)
    ax4.set_title('Size Distribution (Count)', fontsize=13)
    ax4.set_xticks(x2)
    ax4.set_xticklabels(range_labels, rotation=25, ha='right', fontsize=11)
    ax4.legend(frameon=False, loc='upper left', fontsize=11,
                # bbox_to_anchor=(1.01, 1)
                )
    ax4.grid(axis='y', alpha=0.25, linestyle='--')
    y_max4 = ax4.get_ylim()[1]
    margin4 = y_max4 * 0.04
    for bar in bars5:
        height = bar.get_height()
        if height > y_max4 * 0.15:
            ax4.text(bar.get_x() + bar.get_width()/2, height/2, str(int(height)), ha='center', va='center', fontweight='bold', color='black', fontsize=11)
        else:
            y = min(height + 20, y_max4 - margin4)
            ax4.text(bar.get_x() + bar.get_width()/2, y, str(int(height)), ha='center', va='bottom', fontweight='bold', color='black', fontsize=11)
    for bar in bars6:
        height = bar.get_height()
        if height > y_max4 * 0.15:
            ax4.text(bar.get_x() + bar.get_width()/2, height/2, str(int(height)), ha='center', va='center', fontweight='bold', color='black', fontsize=11)
        else:
            y = min(height + 20, y_max4 - margin4)
            ax4.text(bar.get_x() + bar.get_width()/2, y, str(int(height)), ha='center', va='bottom', fontweight='bold', color='black', fontsize=11)

    fig.tight_layout(rect=(0, 0, 1, 0.98), pad=3.0)
    fig.subplots_adjust(left=0.07, right=0.95, top=0.90, bottom=0.10, wspace=0.15, hspace=0.25)
    combined_path = figures_dir / 'compliance_and_distribution.png'
    fig.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f'Saved combined figure to: {combined_path}')

    plt.show()
    return fig

if __name__ == '__main__':
    compare_datasets()
