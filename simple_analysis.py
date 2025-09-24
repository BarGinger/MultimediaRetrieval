import open3d as o3d
from pathlib import Path
import statistics

# Compare original vs resampled files
datasets_dir = Path('Datasets')
original_dir = datasets_dir / 'Data_sampled'
resampled_dir = datasets_dir / 'Data_sampled_resampled'
TARGET_VERTEX_COUNT = 5000
TOLERANCE = 0.25

def compare_datasets():
    print("COMPARISON: Original vs Resampled Dataset")
    print("=" * 60)    
    original_vertex_counts = []
    resampled_vertex_counts = []
    comparisons = []
    
    min_target = int(TARGET_VERTEX_COUNT * (1 - TOLERANCE))  # 3750
    max_target = int(TARGET_VERTEX_COUNT * (1 + TOLERANCE))  # 6250
    
    print(f"Target: {TARGET_VERTEX_COUNT} vertices (±{TOLERANCE*100}% tolerance)")
    print(f"Expected range: {min_target} - {max_target} vertices")
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
    print(f"  Original within tolerance: {original_within_tolerance}/{len(original_vertex_counts)} ({original_within_tolerance/len(original_vertex_counts)*100:.1f}%)")
    print(f"  Resampled within tolerance: {resampled_within_tolerance}/{len(resampled_vertex_counts)} ({resampled_within_tolerance/len(resampled_vertex_counts)*100:.1f}%)")
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
        (0, 1000, "Very Small"),
        (1000, 3750, "Small"),
        (3750, 6250, "Target Range"),
        (6250, 10000, "Large"),
        (10000, float('inf'), "Very Large")
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

if __name__ == '__main__':
    compare_datasets()
