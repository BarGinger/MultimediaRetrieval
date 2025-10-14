#!/usr/bin/env python3
"""
Debug script to compare what get_cached_dataset_data vs get_analysis_data returns.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.dataset_cache import get_cached_dataset_data
from core.analysis_cache import get_analysis_data

def main():
    print("🔍 Data Source Comparison")
    print("=" * 50)
    
    test_dataset = 'Data'
    
    print(f"\n📊 Testing dataset: {test_dataset}")
    
    # Test cached dataset data
    print("\n1️⃣ get_cached_dataset_data():")
    try:
        cached_df = get_cached_dataset_data(test_dataset)
        print(f"   Rows: {len(cached_df)}")
        print(f"   Columns: {list(cached_df.columns)}")
        
        if 'num_vertices' in cached_df.columns:
            sample_vertices = cached_df['num_vertices'].iloc[0] if len(cached_df) > 0 else None
            print(f"   Sample vertices: {sample_vertices} ({type(sample_vertices).__name__})")
            null_count = cached_df['num_vertices'].isnull().sum()
            print(f"   Null vertices: {null_count}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test analysis data
    print("\n2️⃣ get_analysis_data():")
    try:
        analysis_df = get_analysis_data(test_dataset)
        if analysis_df is not None:
            print(f"   Rows: {len(analysis_df)}")
            print(f"   Columns: {list(analysis_df.columns)}")
            
            if 'num_vertices' in analysis_df.columns:
                sample_vertices = analysis_df['num_vertices'].iloc[0] if len(analysis_df) > 0 else None
                print(f"   Sample vertices: {sample_vertices} ({type(sample_vertices).__name__})")
        else:
            print(f"   None returned")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    main()