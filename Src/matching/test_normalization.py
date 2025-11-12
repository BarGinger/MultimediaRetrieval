"""
Test script to verify that distance matrices are properly normalized to [0, 1] range.
"""

from shapeQuery import ShapeQuery
import os

# Clean cache to force recomputation
cache_dir = "total_distances"
if os.path.exists(cache_dir):
    import shutil
    print(f"Clearing cache directory: {cache_dir}")
    shutil.rmtree(cache_dir)
    print("Cache cleared\n")

# Create query system with debug mode to see normalization
print("=" * 60)
print("Testing Distance Matrix Normalization")
print("=" * 60)
print()

qs = ShapeQuery(
    num_shapes=10,  # Use 10 shapes for quick test
    debug=True  # Enable debug output to see normalization info
)

print("\n" + "=" * 60)
print("Distance Matrix Statistics")
print("=" * 60)

# Check the total distance matrix
dm = qs.distance_matrix
print(f"\nTotal Distance Matrix:")
print(f"  Shape: {dm.shape}")
print(f"  Min: {dm.min().min():.6f}")
print(f"  Max: {dm.max().max():.6f}")
print(f"  Mean: {dm.mean().mean():.6f}")
print(f"  Diagonal (should be 0): {[dm.iloc[i,i] for i in range(min(5, len(dm)))]}")

# Verify all values are non-negative
if (dm >= 0).all().all():
    print("\nSUCCESS: All distances are non-negative!")
else:
    negative_count = (dm < 0).sum().sum()
    print(f"\n✗ FAILED: Found {negative_count} negative distances!")

# Test a query
print("\n" + "=" * 60)
print("Testing Query with Normalized Distances")
print("=" * 60)
query_shape = qs.shape_names[0]
print(f"\nQuerying for: {query_shape}")
results = qs.query(query_shape, k=5)

print(f"\nTop 5 similar shapes:")
for i, (shape, dist) in enumerate(results, 1):
    print(f"  {i}. {shape}: {dist:.6f}")
