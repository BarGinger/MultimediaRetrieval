#!/usr/bin/env python3
"""
Simple test to check if our analysis generation logic would work
"""

import os
import glob
import csv

def test_analysis_generation_logic():
    """Test the file discovery logic for analysis generation without dependencies"""
    print("🧪 TESTING ANALYSIS GENERATION LOGIC")
    print("=" * 50)
    
    # Get the base path
    base_path = r"c:\Users\bar24\OneDrive - Universiteit Utrecht\Documents\School\UU Data Sceince MSc\2nd Year\Period 1\Multimedia Retrieval - INFOMR\Assignments\MultimediaRetrieval"
    
    # Test datasets to analyze
    datasets_to_analyze = {
        "original_datasets": ["Data", "Data_sampled", "Data_resampled", "Data_sampled_resampled", "Data_sampled_resampled_normalized"],
        "processed_datasets": ["UnifiedPreprocessed/Data"]
    }
    
    print("\n📁 CHECKING DATASET AVAILABILITY:")
    print("-" * 30)
    
    results = {"original": [], "processed": []}
    
    # Check original datasets
    for dataset_name in datasets_to_analyze["original_datasets"]:
        dataset_path = os.path.join(base_path, "Datasets", dataset_name)
        if os.path.exists(dataset_path):
            # Count .obj files
            obj_files = glob.glob(os.path.join(dataset_path, "**", "*.obj"), recursive=True)
            print(f"✅ {dataset_name}: {len(obj_files)} .obj files")
            results["original"].append((dataset_name, len(obj_files)))
        else:
            print(f"❌ {dataset_name}: NOT FOUND")
    
    # Check processed datasets
    for dataset_name in datasets_to_analyze["processed_datasets"]:
        dataset_path = os.path.join(base_path, "Datasets", dataset_name)
        if os.path.exists(dataset_path):
            # Count *_05_scaled.obj files (processed files)
            scaled_files = glob.glob(os.path.join(dataset_path, "**", "*_05_scaled.obj"), recursive=True)
            print(f"✅ {dataset_name}: {len(scaled_files)} *_05_scaled.obj files")
            results["processed"].append((dataset_name, len(scaled_files)))
        else:
            print(f"❌ {dataset_name}: NOT FOUND")
    
    print("\n📊 ANALYSIS CSV GENERATION PLAN:")
    print("-" * 35)
    
    # Check existing analysis files
    preprocessing_path = os.path.join(base_path, "Preprocessing")
    existing_csvs = glob.glob(os.path.join(preprocessing_path, "analysis_results*.csv"))
    print(f"📂 Existing analysis CSVs in Preprocessing: {len(existing_csvs)}")
    for csv_file in existing_csvs:
        print(f"   - {os.path.basename(csv_file)}")
    
    # Plan for original datasets
    print(f"\n🎯 Would create analysis CSVs for {len(results['original'])} original datasets in: {preprocessing_path}")
    for dataset_name, file_count in results["original"]:
        print(f"   - analysis_results_{dataset_name.lower()}.csv ({file_count} files)")
    
    # Plan for processed datasets
    print(f"\n🎯 Would create analysis CSVs for {len(results['processed'])} processed datasets:")
    for dataset_name, file_count in results["processed"]:
        dataset_path = os.path.join(base_path, "Datasets", dataset_name)
        print(f"   - {dataset_path}\\analysis_results.csv ({file_count} files)")
    
    return results

if __name__ == "__main__":
    test_analysis_generation_logic()