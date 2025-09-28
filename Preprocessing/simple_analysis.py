import open3d as o3d
from pathlib import Path
import statistics
import matplotlib.pyplot as plt
import numpy as np

# Compare original vs resampled files
datasets_dir = Path('Datasets')
original_dir = datasets_dir / 'Data'
resampled_dir = datasets_dir / 'Data_resampled'
TARGET_VERTEX_COUNT = 7500
# Custom range: 5000-10000 vertices is acceptable (matching resampling.py)
MIN_ACCEPTABLE_VERTICES = 5000
MAX_ACCEPTABLE_VERTICES = 10000

def compare_datasets():
    print("COMPARISON: Original vs Resampled Dataset")
    print("=" * 60)    
    original_vertex_counts = []
    resampled_vertex_counts = []
    comparisons = []
    
    min_target = MIN_ACCEPTABLE_VERTICES  # 5000
    max_target = MAX_ACCEPTABLE_VERTICES  # 10000
    
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
        
        # Process each file in the category
        for obj_file in category_dir.glob('*.obj'):
            resampled_file = resampled_category_dir / obj_file.name
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
    
    if not comparisons:
        print("No files found for comparison!")
        return
    
    print(f"📊 COMPARISON STATISTICS:")
    print(f"Total files compared: {len(comparisons)}")
    print()
    
    # Original dataset stats
    print(f"📈 ORIGINAL DATASET:")
    print(f"  Average vertices: {statistics.mean(original_vertex_counts):.0f}")
    print(f"  Median vertices: {statistics.median(original_vertex_counts):.0f}")
    print(f"  Min vertices: {min(original_vertex_counts)}")
    print(f"  Max vertices: {max(original_vertex_counts)}")
    print(f"  Standard deviation: {statistics.stdev(original_vertex_counts):.0f}")
    
    # Resampled dataset stats
    print(f"\n📈 RESAMPLED DATASET:")
    print(f"  Average vertices: {statistics.mean(resampled_vertex_counts):.0f}")
    print(f"  Median vertices: {statistics.median(resampled_vertex_counts):.0f}")
    print(f"  Min vertices: {min(resampled_vertex_counts)}")
    print(f"  Max vertices: {max(resampled_vertex_counts)}")
    print(f"  Standard deviation: {statistics.stdev(resampled_vertex_counts):.0f}")
    
    # Improvement analysis
    original_within_tolerance = sum(1 for v in original_vertex_counts if min_target <= v <= max_target)
    resampled_within_tolerance = sum(1 for v in resampled_vertex_counts if min_target <= v <= max_target)
    
    print(f"\n🎯 TARGET COMPLIANCE:")
    print(f"  Original within acceptable range: {original_within_tolerance}/{len(original_vertex_counts)} ({original_within_tolerance/len(original_vertex_counts)*100:.1f}%)")
    print(f"  Resampled within acceptable range: {resampled_within_tolerance}/{len(resampled_vertex_counts)} ({resampled_within_tolerance/len(resampled_vertex_counts)*100:.1f}%)")
    print(f"  Improvement: {resampled_within_tolerance - original_within_tolerance} files (+{(resampled_within_tolerance - original_within_tolerance)/len(original_vertex_counts)*100:.1f}%)")
    
    # Action analysis
    action_stats = {}
    for comp in comparisons:
        key = f"{comp['expected_action']} -> {comp['actual_action']}"
        action_stats[key] = action_stats.get(key, 0) + 1
    
    print(f"\n🔄 ACTION ANALYSIS:")
    for action, count in sorted(action_stats.items()):
        percentage = count / len(comparisons) * 100
        print(f"  {action:<25}: {count:>3} files ({percentage:>5.1f}%)")
    
    # Show biggest improvements and failures
    print(f"\n✅ BIGGEST IMPROVEMENTS (moved closer to target):")
    improvements = [c for c in comparisons if abs(c['resampled_vertices'] - TARGET_VERTEX_COUNT) < abs(c['original_vertices'] - TARGET_VERTEX_COUNT)]
    improvements.sort(key=lambda x: abs(x['original_vertices'] - TARGET_VERTEX_COUNT) - abs(x['resampled_vertices'] - TARGET_VERTEX_COUNT), reverse=True)
    
    for i, comp in enumerate(improvements[:5]):
        original_diff = abs(comp['original_vertices'] - TARGET_VERTEX_COUNT)
        resampled_diff = abs(comp['resampled_vertices'] - TARGET_VERTEX_COUNT)
        improvement = original_diff - resampled_diff
        print(f"  {i+1}. {comp['category']:<15} {comp['file']:<25} {comp['original_vertices']:>5} → {comp['resampled_vertices']:>5} (improved by {improvement:+.0f})")
    
    print(f"\n❌ WORST CASES (still far from target):")
    worst_cases = sorted(comparisons, key=lambda x: abs(x['resampled_vertices'] - TARGET_VERTEX_COUNT), reverse=True)
    
    for i, comp in enumerate(worst_cases[:5]):
        diff = comp['resampled_vertices'] - TARGET_VERTEX_COUNT
        print(f"  {i+1}. {comp['category']:<15} {comp['file']:<25} {comp['original_vertices']:>5} → {comp['resampled_vertices']:>5} ({diff:+d} from target)")
    
    # Distribution comparison
    print(f"\n📊 DISTRIBUTION COMPARISON:")
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
        print(f"❌ Error creating plots: {e}")
        print("   Make sure matplotlib is installed: pip install matplotlib")

def create_analysis_plots(original_vertex_counts, resampled_vertex_counts, comparisons):
    """Generate a single combined figure with two side-by-side subplots.

    Left: Target compliance stacked bar (original vs resampled)
    Right: Distribution by size categories (grouped bars)
    """

    figures_dir = Path('Preprocessing') / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

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

    # Create combined figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))
    fig.suptitle('Target Compliance & Size Distribution', fontsize=14, fontweight='bold')

    # --- Subplot 1: Target compliance ---
    x = np.arange(len(categories))
    width = 0.55
    ax1.bar(x, within_range, width, label='Within Range', color='#2ca02c', alpha=0.85)
    ax1.bar(x, outside_range, width, bottom=within_range, label='Outside Range', color='#d62728', alpha=0.7)
    ax1.set_ylabel('Number of Files')
    ax1.set_title(f'Compliance ({MIN_ACCEPTABLE_VERTICES}-{MAX_ACCEPTABLE_VERTICES})')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    # Add a little headroom so the legend does not overlap bars
    max_total = max(within_range[i] + outside_range[i] for i in range(len(categories)))
    ax1.set_ylim(0, max_total * 1.12)
    # Place legend centered at top inside the extra headroom
    ax1.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))

    for i, (within, outside) in enumerate(zip(within_range, outside_range)):
        total = within + outside
        within_pct = within / total * 100
        outside_pct = outside / total * 100
        ax1.text(i, within/2, f'{within_pct:.1f}%', ha='center', va='center', fontweight='bold', color='white')
        ax1.text(i, within + outside/2, f'{outside_pct:.1f}%', ha='center', va='center', fontweight='bold')

    # --- Subplot 2: Distribution by size categories ---
    x2 = np.arange(len(range_labels))
    width2 = 0.35
    max_height = max(original_dist + resampled_dist) if original_dist else 0
    ax2.bar(x2 - width2/2, original_dist, width2, label='Original', alpha=0.75, color='#ff7f0e')
    ax2.bar(x2 + width2/2, resampled_dist, width2, label='Resampled', alpha=0.75, color='#1f77b4')
    ax2.set_xlabel('Vertex Count Range')
    ax2.set_ylabel('Number of Files')
    ax2.set_title('Size Distribution')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(range_labels, rotation=25, ha='right')
    ax2.legend(frameon=False)
    ax2.grid(axis='y', alpha=0.25, linestyle='--')

    for i, (o, r) in enumerate(zip(original_dist, resampled_dist)):
        ax2.text(i - width2/2, o + max_height*0.015, str(o), ha='center', va='bottom', fontsize=8)
        ax2.text(i + width2/2, r + max_height*0.015, str(r), ha='center', va='bottom', fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    combined_path = figures_dir / 'compliance_and_distribution.png'
    fig.savefig(combined_path, dpi=300)
    print(f'✅ Saved combined figure to: {combined_path}')

    # Display
    plt.show()
    return fig

if __name__ == '__main__':
    compare_datasets()
