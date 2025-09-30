#!/usr/bin/env python3
"""
Test script to validate that the navigate_to_average function works correctly
by simulating the same filtering logic.
"""

import sys
from pathlib import Path
import pandas as pd

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.dataset_cache import get_cached_dataset_data
import fnmatch

def simulate_navigate_to_average(selected_category='all', filename_filter='', vertices_op=None, vertices_val=None, faces_op=None, faces_val=None, sort_field='category', sort_order='asc', selected_dataset='Data'):
    """
    Simulate the navigate_to_average logic to test if it works correctly.
    """
    print(f"🧪 Simulating navigation to average with:")
    print(f"   Dataset: {selected_dataset}")
    print(f"   Category: {selected_category}")
    print(f"   Filename filter: '{filename_filter}'")
    print(f"   Sort: {sort_field} ({sort_order})")
    
    # Get cached data (same as function)
    file_df = get_cached_dataset_data(selected_dataset)
    print(f"   📊 Total shapes: {len(file_df)}")
    
    # Apply category filter (same as function)
    df = file_df if selected_category == 'all' else file_df[file_df['category'] == selected_category]
    print(f"   📁 After category filter: {len(df)}")
    
    # Apply filename filtering (same as function)
    if filename_filter and filename_filter.strip() and not df.empty and 'filename' in df.columns:
        pattern = filename_filter.strip()
        mask = df['filename'].apply(lambda x: fnmatch.fnmatch(x.lower(), pattern.lower()))
        df = df[mask]
        print(f"   📄 After filename filter: {len(df)}")
    
    # Apply vertices filtering (same as function)
    if vertices_val is not None and vertices_val != '' and 'num_vertices' in df.columns:
        val = int(vertices_val)
        if vertices_op == 'eq':
            df = df[df['num_vertices'] == val]
        elif vertices_op == 'gt':
            df = df[df['num_vertices'] > val]
        elif vertices_op == 'lt':
            df = df[df['num_vertices'] < val]
        print(f"   🔺 After vertices filter: {len(df)}")
    
    # Apply faces filtering (same as function)
    if faces_val is not None and faces_val != '' and 'num_faces' in df.columns:
        val = int(faces_val)
        if faces_op == 'eq':
            df = df[df['num_faces'] == val]
        elif faces_op == 'gt':
            df = df[df['num_faces'] > val]
        elif faces_op == 'lt':
            df = df[df['num_faces'] < val]
        print(f"   🔷 After faces filter: {len(df)}")
    
    # Apply sorting (same as function)
    ascending = True if sort_order == 'asc' else False
    df = df.copy()
    if sort_field == 'category':
        df = df.sort_values(by=['category', 'filename'], ascending=ascending)
    elif sort_field in ['num_vertices', 'num_faces']:
        df[sort_field] = df[sort_field].fillna(0)
        df = df.sort_values(by=sort_field, ascending=ascending)
    
    df = df.reset_index(drop=True)
    print(f"   📋 After sorting and reset: {len(df)}")
    
    if df.empty:
        print("   ❌ No shapes after filtering!")
        return None
    
    # Find average vertices
    if 'num_vertices' in df.columns:
        valid = df['num_vertices'].dropna()
        if not valid.empty:
            avg_v = valid.mean()
            idx = (df['num_vertices'] - avg_v).abs().idxmin()
            avg_shape = df.iloc[idx]
            print(f"   ✅ Average vertices: {avg_v:.1f}")
            print(f"   🎯 Closest shape at index {idx}: {avg_shape['filename']}")
            print(f"      Category: {avg_shape['category']}")
            print(f"      Vertices: {avg_shape['num_vertices']}")
            print(f"      Faces: {avg_shape['num_faces']}")
            return idx
    
    return None

def main():
    print("🧪 Testing Navigate to Average Fix")
    print("=" * 60)
    
    # Test 1: Basic navigation (no filters)
    print("\n1️⃣ Test: Basic navigation (all shapes)")
    simulate_navigate_to_average()
    
    # Test 2: With category filter
    print("\n2️⃣ Test: With category filter")
    simulate_navigate_to_average(selected_category='Car')
    
    # Test 3: With filename filter
    print("\n3️⃣ Test: With filename filter")
    simulate_navigate_to_average(filename_filter='m*')
    
    # Test 4: Combined filters
    print("\n4️⃣ Test: Combined filters")
    simulate_navigate_to_average(selected_category='Car', filename_filter='m*')

if __name__ == "__main__":
    main()