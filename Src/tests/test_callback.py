#!/usr/bin/env python3
"""
Test script to verify that the file list callback now shows vertex/face counts correctly.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from viewer.callbacks import update_file_list_internal

def main():
    print("🧪 Testing File List Callback")
    print("=" * 50)
    
    # Test parameters that would trigger the file list update
    avg_filter = 'none'
    selected_category = 'all'
    filename_filter = ''
    vertices_op = None
    vertices_val = None
    faces_op = None
    faces_val = None
    sort_field = 'category'
    sort_order = 'asc'
    selected_dataset = 'Data'
    
    print(f"📊 Testing with dataset: {selected_dataset}")
    
    try:
        # Call the internal file list function
        result = update_file_list_internal(
            avg_filter, selected_category, filename_filter, 
            vertices_op, vertices_val, faces_op, faces_val, 
            sort_field, sort_order, selected_dataset
        )
        
        print(f"✅ Callback executed successfully")
        print(f"📄 Returned {len(result)} buttons")
        
        # Check a few buttons to see if they have vertex/face data
        if len(result) > 0:
            first_button = result[0]
            button_text = str(first_button)
            
            if 'N/A' in button_text:
                print("❌ Still showing N/A for vertex/face counts")
            else:
                print("✅ Button contains vertex/face counts!")
                
            # Look for vertex/face indicators in the button text
            if 'Vertices:' in button_text and 'Faces:' in button_text:
                print("✅ Both vertices and faces are present in button")
            else:
                print("⚠️ Missing vertex or face information")
                
        else:
            print("❌ No buttons returned")
            
    except Exception as e:
        print(f"❌ Error testing callback: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()