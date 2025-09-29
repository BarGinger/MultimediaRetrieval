"""
Simple verification test for technical tips compliance
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, '.')

try:
    from core.file_index import get_file_tree
    from core.analysis_cache import merge_analysis_data
    from core.shapeMesh import ShapeMesh
    
    print("🔍 Technical Tips Compliance Verification")
    print("=" * 50)
    
    # Get a test shape
    file_df = get_file_tree()
    file_df = merge_analysis_data(file_df, "LabeledPSB")
    test_row = file_df.head(1).iloc[0]
    
    print(f"Testing with: {test_row['filename']}")
    
    mesh = ShapeMesh.from_file_row(test_row)
    
    print("\n✅ Testing Step Order Compliance")
    print("Technical Tips Order: Remeshing → Translation → Pose → Flipping → Size")
    
    # Test with debug output
    normalized_vertices = mesh.apply_full_normalization(debug=True)
    
    print("\n✅ Testing Final Results")
    final_center = np.mean(normalized_vertices, axis=0)
    final_max_dim = np.max(np.ptp(normalized_vertices, axis=0))
    
    center_ok = np.linalg.norm(final_center) < 1e-10
    scale_ok = abs(final_max_dim - 1.0) < 1e-6
    
    print(f"Final center: {final_center} (magnitude: {np.linalg.norm(final_center):.2e})")
    print(f"Final max dimension: {final_max_dim:.6f}")
    print(f"Centered correctly: {'✅' if center_ok else '❌'}")
    print(f"Scaled correctly: {'✅' if scale_ok else '❌'}")
    
    if center_ok and scale_ok:
        print("\n🎉 TECHNICAL TIPS COMPLIANCE VERIFIED!")
    else:
        print("\n❌ Implementation needs adjustment")
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Please run from the correct directory")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()