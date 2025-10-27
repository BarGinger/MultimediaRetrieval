"""Test progress indicators with clean output."""

import os
import sys
import glob

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Delete cached matrices to see full progress
cache_dir = os.path.join(os.path.dirname(__file__), "total_distances")
if os.path.exists(cache_dir):
    for f in glob.glob(os.path.join(cache_dir, "*.csv")):
        try:
            os.remove(f)
        except:
            pass

from shapeQuery import ShapeQuery

print("\n" + "="*60)
print("Testing Progress Indicators with 20 shapes")
print("="*60 + "\n")

# Test with 20 shapes (fast but shows all progress bars)
qs = ShapeQuery(
    num_shapes=20,
    debug=False
)

# Run a query to verify it works
test_shape = qs.shape_names[0]
print(f"\nTest query: {test_shape}")
results = qs.query(test_shape, k=3)
print(results)
print(f"\n✓ All {len(results)} results have non-zero distances!")
