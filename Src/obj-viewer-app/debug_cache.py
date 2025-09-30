#!/usr/bin/env python3
"""
Quick debug script to test the new cache system and check what data is available.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.dataset_cache import get_available_datasets, get_cached_dataset_data, clear_dataset_cache

def main():
    print("🔍 Dataset Cache Debug")
    print("=" * 50)
    
    # Clear cache first to force fresh load
    print("🗑️ Clearing cache...")
    clear_dataset_cache(disk_cache=True)
    
    # Get available datasets
    print("\n📁 Available datasets:")
    datasets = get_available_datasets()
    for i, dataset in enumerate(datasets, 1):
        print(f"  {i}. {dataset}")
    
    if not datasets:
        print("❌ No datasets found!")
        return
    
    # Test loading the first dataset
    test_dataset = datasets[0]
    print(f"\n⚡ Testing dataset: {test_dataset}")
    
    try:
        df = get_cached_dataset_data(test_dataset)
        print(f"✅ Loaded {len(df)} shapes")
        print(f"📊 Columns: {list(df.columns)}")
        
        if len(df) > 0:
            print(f"\n🔍 Sample data:")
            sample = df.iloc[0]
            for col in df.columns:
                value = sample[col]
                print(f"  {col}: {value} ({type(value).__name__})")
                
            print(f"\n📈 Vertex/Face info:")
            if 'num_vertices' in df.columns:
                vertex_stats = df['num_vertices'].describe()
                print(f"  Vertices: min={vertex_stats['min']}, max={vertex_stats['max']}, mean={vertex_stats['mean']:.1f}")
            
            if 'num_faces' in df.columns:
                face_stats = df['num_faces'].describe()
                print(f"  Faces: min={face_stats['min']}, max={face_stats['max']}, mean={face_stats['mean']:.1f}")
                
        else:
            print("⚠️ No shapes found in dataset!")
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()