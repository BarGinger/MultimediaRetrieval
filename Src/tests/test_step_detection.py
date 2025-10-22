#!/usr/bin/env python3
"""
Test script for step detection functionality
"""

import sys
sys.path.append('.')

from core.file_index import get_file_tree_with_steps, detect_step_files, get_step_file_path, get_step_display_info
from pathlib import Path
import pandas as pd

def test_step_detection():
    """Test the step detection functionality"""
    print("🧪 Testing Step Detection Functionality")
    print("=" * 50)
    
    # Test 1: Detect step files for a known shape
    test_dir = Path("c:/Users/bar24/OneDrive - Universiteit Utrecht/Documents/School/UU Data Sceince MSc/2nd Year/Period 1/Multimedia Retrieval - INFOMR/Assignments/MultimediaRetrieval/Datasets/UnifiedPreprocessed/Data/Car")
    test_base = "m1485"
    
    print(f"1. Testing step detection for {test_base} in Car category")
    step_info = detect_step_files(test_dir, test_base)
    print(f"   Has steps: {step_info['has_steps']}")
    print(f"   Step count: {step_info['step_count']}")
    print(f"   Original file: {step_info['original_file']}")
    print(f"   Available steps: {list(step_info['steps'].keys())}")
    print()
    
    # Test 2: Get file tree with step detection
    print("2. Testing file tree with step detection for UnifiedPreprocessed dataset")
    try:
        df = get_file_tree_with_steps("UnifiedPreprocessed/Data")
        print(f"   Found {len(df)} shapes total")
        
        if not df.empty:
            # Show first few entries
            print(f"   First 5 entries:")
            for i, row in df.head().iterrows():
                print(f"     {row['category']}/{row['filename']}: has_steps={row.get('has_processing_steps', False)}, steps={row.get('available_steps', 0)}")
            
            # Statistics
            has_steps = df['has_processing_steps'].sum() if 'has_processing_steps' in df.columns else 0
            print(f"   Shapes with processing steps: {has_steps}/{len(df)}")
        print()
    except Exception as e:
        print(f"   Error: {e}")
        print()
    
    # Test 3: Test step file path resolution
    print("3. Testing step file path resolution")
    if not df.empty and 'has_processing_steps' in df.columns:
        # Find a shape with processing steps
        processed_shapes = df[df['has_processing_steps'] == True]
        if not processed_shapes.empty:
            test_row = processed_shapes.iloc[0]
            print(f"   Testing with shape: {test_row['category']}/{test_row['filename']}")
            
            for step in range(7):
                step_path = get_step_file_path(test_row, step)
                step_info = get_step_display_info(step)
                print(f"     Step {step} ({step_info['name']}): {Path(step_path).name}")
    print()
    
    # Test 4: Test original dataset compatibility
    print("4. Testing original dataset compatibility")
    try:
        original_df = get_file_tree_with_steps("Data")
        print(f"   Original Data dataset: {len(original_df)} shapes")
        
        if not original_df.empty:
            has_steps = original_df['has_processing_steps'].sum() if 'has_processing_steps' in original_df.columns else 0
            print(f"   Shapes with processing steps: {has_steps}/{len(original_df)} (should be 0 for original dataset)")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n✅ Step detection testing complete!")

if __name__ == "__main__":
    test_step_detection()