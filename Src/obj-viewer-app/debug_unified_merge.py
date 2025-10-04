#!/usr/bin/env python3
"""
Debug UnifiedPreprocessed dataset merge issues
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from core.dataset_cache import get_cached_dataset_data
from core.analysis_cache import get_analysis_data
import pandas as pd

def debug_unified_merge():
    """Debug why UnifiedPreprocessed dataset has zero vertex/face counts"""
    print("🔍 DEBUGGING UnifiedPreprocessed merge issue")
    print("=" * 60)
    
    try:
        # Get the file dataset
        dataset = "UnifiedPreprocessed/Data"
        file_df = get_cached_dataset_data(dataset)
        
        print(f"📁 File dataset: {len(file_df)} shapes")
        print(f"📁 Sample filenames:")
        for i in range(5):
            if i < len(file_df):
                row = file_df.iloc[i]
                print(f"   {i}: {row['category']}/{row['filename']} - V:{row.get('num_vertices', 'N/A')}, F:{row.get('num_faces', 'N/A')}")
        
        # Get analysis data
        analysis_df = get_analysis_data("UnifiedPreprocessed/Data")
        if analysis_df is not None:
            print(f"\n📊 Analysis dataset: {len(analysis_df)} shapes")
            print(f"📊 Sample analysis filenames:")
            for i in range(5):
                if i < len(analysis_df):
                    row = analysis_df.iloc[i]
                    print(f"   {i}: {row['category']}/{row['filename']} - V:{row.get('num_vertices', 'N/A')}, F:{row.get('num_faces', 'N/A')}")
        
            # Check for overlapping filenames
            file_names = set(file_df['filename'].tolist())
            analysis_names = set(analysis_df['filename'].tolist())
            
            print(f"\n🔄 Overlap analysis:")
            print(f"   File dataset unique filenames: {len(file_names)}")
            print(f"   Analysis dataset unique filenames: {len(analysis_names)}")
            print(f"   Common filenames: {len(file_names.intersection(analysis_names))}")
            
            # Show some examples of non-overlapping
            non_overlapping_file = file_names - analysis_names
            non_overlapping_analysis = analysis_names - file_names
            
            print(f"\n❌ File dataset filenames NOT in analysis (first 5):")
            for name in list(non_overlapping_file)[:5]:
                print(f"   {name}")
                
            print(f"\n❌ Analysis filenames NOT in file dataset (first 5):")
            for name in list(non_overlapping_analysis)[:5]:
                print(f"   {name}")
                
            # Try manual merge to see what happens
            print(f"\n🧪 Testing merge...")
            merged = pd.merge(
                file_df[['category', 'filename']].head(10), 
                analysis_df[['category', 'filename', 'num_vertices', 'num_faces']],
                on=['category', 'filename'], 
                how='left'
            )
            print(f"   Merged result shape: {merged.shape}")
            print(f"   Non-null vertices: {merged['num_vertices'].notna().sum()}")
            print(f"   Non-null faces: {merged['num_faces'].notna().sum()}")
        else:
            print("❌ Could not load analysis data")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_unified_merge()