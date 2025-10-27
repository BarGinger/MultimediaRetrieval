"""Quick test to verify symmetric distance matrix fix."""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shapeQuery import ShapeQuery

# Delete any cached total distance matrices to force recomputation
import glob
cache_dir = os.path.join(os.path.dirname(__file__), "total_distances")
if os.path.exists(cache_dir):
    for f in glob.glob(os.path.join(cache_dir, "*.csv")):
        try:
            os.remove(f)
            print(f"Deleted cached file: {os.path.basename(f)}")
        except:
            pass

print("\n" + "="*60)
print("Testing symmetric distance matrix with 10 shapes")
print("="*60 + "\n")

qs = ShapeQuery(
    num_shapes=10,
    debug=False  # Less verbose for cleaner output
)

# Test query
test_shape = qs.shape_names[0]
print(f"\nQuerying for: {test_shape}")
print("="*60)
results = qs.query(test_shape, k=5)
print(results)
print()

# Verify distances are not all zero
non_zero = (results['distance'] > 0).sum()
if non_zero > 0:
    print(f"✓ SUCCESS: Found {non_zero} non-zero distances!")
else:
    print(f"✗ ISSUE: All distances are zero - matrix symmetry problem persists")

# Show some actual distance values from the matrix
print(f"\nSample distances from matrix (first 5x5 block):")
print(qs.distance_matrix.iloc[:5, :5])
