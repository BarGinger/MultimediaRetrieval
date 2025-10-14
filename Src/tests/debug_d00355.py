#!/usr/bin/env python3
"""
Debug script to check step detection for D00355
"""

import sys
from pathlib import Path

# Add the src directory to Python path
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

def debug_step_detection():
    """Debug step detection for D00355"""
    from core.file_index import get_file_tree, get_available_steps
    
    # Get the UnifiedPreprocessed/Data dataset
    print("🔍 Debugging step detection for D00355...")
    
    file_df = get_file_tree(data_dir='UnifiedPreprocessed/Data')
    print(f"Total shapes found: {len(file_df)}")
    
    # Look for D00355 specifically
    d00355_rows = file_df[file_df['filename'].str.contains('D00355')]
    print(f"D00355 shapes found: {len(d00355_rows)}")
    
    if len(d00355_rows) > 0:
        row = d00355_rows.iloc[0]
        print(f"Shape: {row['filename']}")
        print(f"Has processing steps: {row.get('has_processing_steps', False)}")
        print(f"Step files keys: {list(row.get('step_files', {}).keys())}")
        
        # Test available steps
        availability = get_available_steps(row)
        print(f"Available step indices: {availability['available_step_indices']}")
        print(f"Missing step indices: {availability['missing_step_indices']}")
        print(f"Step availability: {availability['step_availability']}")
        
        # Check if remesh file actually exists
        step_files = row.get('step_files', {})
        print(f"Has 01_remeshed file: {'01_remeshed' in step_files}")
        
        return row
    else:
        print("❌ D00355 not found in dataset")
        return None

if __name__ == "__main__":
    debug_step_detection()