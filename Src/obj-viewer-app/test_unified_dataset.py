#!/usr/bin/env python3
"""
Test the UnifiedPreprocessed dataset specifically
"""

import sys
sys.path.append('.')

from core.dataset_cache import get_cached_dataset_data, get_available_datasets
import pandas as pd

def test_unified_dataset():
    """Test the UnifiedPreprocessed dataset"""
    print("🧪 Testing UnifiedPreprocessed Dataset")
    print("=" * 50)
    
    datasets = get_available_datasets()
    print(f"Available datasets: {datasets}")
    
    if 'UnifiedPreprocessed/Data' in datasets:
        print(f"\n✅ UnifiedPreprocessed/Data found in available datasets")
        
        # Load the dataset
        print("Loading UnifiedPreprocessed/Data...")
        df = get_cached_dataset_data('UnifiedPreprocessed/Data')
        
        print(f"Dataset loaded: {len(df)} shapes")
        print(f"Columns: {list(df.columns)}")
        
        if not df.empty:
            # Check processing steps
            has_steps = df['has_processing_steps'].sum() if 'has_processing_steps' in df.columns else 0
            print(f"Shapes with processing steps: {has_steps}/{len(df)}")
            
            # Show some examples
            print("\nFirst 3 entries:")
            for i, row in df.head(3).iterrows():
                print(f"  {row['category']}/{row['filename']}: has_steps={row.get('has_processing_steps', False)}, available_steps={row.get('available_steps', 0)}")
            
            # Check step file structure
            if 'step_files' in df.columns:
                processed_shapes = df[df['has_processing_steps'] == True]
                if not processed_shapes.empty:
                    example = processed_shapes.iloc[0]
                    print(f"\nExample step files for {example['base_filename']}:")
                    for step, path in example['step_files'].items():
                        print(f"  {step}: {path.name}")
    else:
        print("❌ UnifiedPreprocessed/Data not found in available datasets")

if __name__ == "__main__":
    test_unified_dataset()