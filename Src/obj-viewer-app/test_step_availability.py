#!/usr/bin/env python3
"""
Test get_available_steps function
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from core.dataset_cache import get_cached_dataset_data
from core.file_index import get_available_steps

def test_available_steps():
    """Test get_available_steps function"""
    print("🧪 TESTING get_available_steps function")
    print("=" * 50)
    
    try:
        # Get cached data
        dataset = "UnifiedPreprocessed/Data"
        file_df = get_cached_dataset_data(dataset)
        
        if file_df is None or file_df.empty:
            print("❌ No cached data found")
            return
            
        print(f"📊 Found {len(file_df)} files in dataset")
        
        # Test with first few files
        for i in range(min(3, len(file_df))):
            row = file_df.iloc[i]
            filename = row.get('filename', 'UNKNOWN')
            
            print(f"\n🔍 Testing file {i}: {filename}")
            print(f"   Row keys: {list(row.keys())}")
            print(f"   has_processing_steps: {row.get('has_processing_steps', 'NOT_SET')}")
            print(f"   step_files: {row.get('step_files', 'NOT_SET')}")
            
            # Test the function
            result = get_available_steps(row)
            print(f"   Available steps: {result.get('available_step_indices', [])}")
            print(f"   Missing steps: {result.get('missing_step_indices', [])}")
            print(f"   Step availability: {result.get('step_availability', {})}")
            
        # Specifically test D00355 if it exists
        d00355_rows = file_df[file_df['filename'].str.contains('D00355', na=False)]
        if not d00355_rows.empty:
            print(f"\n🎯 TESTING D00355 specifically:")
            row = d00355_rows.iloc[0]
            filename = row.get('filename', 'UNKNOWN')
            print(f"   Filename: {filename}")
            result = get_available_steps(row)
            print(f"   Available steps: {result.get('available_step_indices', [])}")
            print(f"   Missing steps: {result.get('missing_step_indices', [])}")
        else:
            print("\n❌ D00355 not found in dataset")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_available_steps()