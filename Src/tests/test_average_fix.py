#!/usr/bin/env python3
"""
Test script to verify that average finding now works correctly.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from viewer.callbacks import update_file_list_internal

def main():
    print("🧪 Testing Average Finding Fix")
    print("=" * 50)
    
    # Test parameters for finding average vertices
    avg_filter = 'avg_vertices'
    selected_category = 'all'
    filename_filter = ''
    vertices_op = None
    vertices_val = None
    faces_op = None
    faces_val = None
    sort_field = 'category'
    sort_order = 'asc'
    selected_dataset = 'Data'
    
    print(f"📊 Testing average vertices with dataset: {selected_dataset}")
    
    try:
        # Call the internal file list function with avg_filter
        result = update_file_list_internal(
            avg_filter, selected_category, filename_filter, 
            vertices_op, vertices_val, faces_op, faces_val, 
            sort_field, sort_order, selected_dataset
        )
        
        print(f"✅ Average vertices callback executed successfully")
        print(f"📄 Returned {len(result)} buttons (should be 1 for average)")
        
        if len(result) == 1:
            print("✅ Correctly filtered to 1 average shape!")
            button_text = str(result[0])
            if 'Vertices:' in button_text and 'Faces:' in button_text:
                print("✅ Average shape has vertex/face counts!")
            else:
                print("⚠️ Average shape missing vertex/face data")
        else:
            print(f"❌ Expected 1 shape, got {len(result)}")
            
    except Exception as e:
        print(f"❌ Error testing average callback: {e}")
        import traceback
        traceback.print_exc()

    # Test average faces
    print(f"\n📊 Testing average faces...")
    try:
        result = update_file_list_internal(
            'avg_faces', selected_category, filename_filter, 
            vertices_op, vertices_val, faces_op, faces_val, 
            sort_field, sort_order, selected_dataset
        )
        
        print(f"✅ Average faces callback executed successfully")
        print(f"📄 Returned {len(result)} buttons (should be 1 for average)")
        
        if len(result) == 1:
            print("✅ Correctly filtered to 1 average shape!")
        else:
            print(f"❌ Expected 1 shape, got {len(result)}")
            
    except Exception as e:
        print(f"❌ Error testing average faces: {e}")

if __name__ == "__main__":
    main()