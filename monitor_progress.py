#!/usr/bin/env python3
"""
Progress monitoring script for normalize_database.py
Run this while your main script is running to check progress
"""

import os
import time
from pathlib import Path
import json

# Configuration
output_dir = Path("c:/Users/bar24/OneDrive - Universiteit Utrecht/Documents/School/UU Data Sceince MSc/2nd Year/Period 1/Multimedia Retrieval - INFOMR/Assignments/MultimediaRetrieval/Datasets/UnifiedPreprocessed/Data")
total_expected_files = 2483

def count_processed_files():
    """Count how many shapes have been fully processed"""
    completed_shapes = 0
    total_obj_files = 0
    categories_processing = 0
    
    if not output_dir.exists():
        return 0, 0, 0
    
    for category_dir in output_dir.iterdir():
        if category_dir.is_dir():
            categories_processing += 1
            category_shapes = 0
            
            # Count OBJ files in this category
            obj_files = list(category_dir.glob("*.obj"))
            total_obj_files += len(obj_files)
            
            # Count completed shapes (those with 05_scaled.obj)
            scaled_files = list(category_dir.glob("*_05_scaled.obj"))
            completed_shapes += len(scaled_files)
            category_shapes = len(scaled_files)
            
            print(f"📁 {category_dir.name}: {category_shapes} shapes completed")
    
    return completed_shapes, total_obj_files, categories_processing

def check_validation_files():
    """Check if validation files are being created"""
    validation_files = list(output_dir.glob("**/*_validation.json"))
    metadata_files = list(output_dir.glob("**/*_metadata.json"))
    
    return len(validation_files), len(metadata_files)

def estimate_time_remaining(completed, total, start_time):
    """Estimate time remaining based on current progress"""
    if completed == 0:
        return "Calculating..."
    
    elapsed = time.time() - start_time
    rate = completed / elapsed  # shapes per second
    remaining = total - completed
    
    if rate > 0:
        time_remaining = remaining / rate
        hours = int(time_remaining // 3600)
        minutes = int((time_remaining % 3600) // 60)
        return f"{hours}h {minutes}m"
    else:
        return "Calculating..."

def main():
    print("🔍 Normalize Database Progress Monitor")
    print("=" * 50)
    
    start_time = time.time()
    
    while True:
        try:
            # Count progress
            completed, total_objs, categories = count_processed_files()
            validation_count, metadata_count = check_validation_files()
            
            # Calculate percentages
            progress_percent = (completed / total_expected_files) * 100 if total_expected_files > 0 else 0
            
            # Estimate time
            time_remaining = estimate_time_remaining(completed, total_expected_files, start_time)
            
            # Display status
            print(f"\n⏰ {time.strftime('%H:%M:%S')} - Progress Update:")
            print(f"✅ Completed shapes: {completed}/{total_expected_files} ({progress_percent:.1f}%)")
            print(f"📊 Total OBJ files: {total_objs}")
            print(f"📁 Categories processing: {categories}")
            print(f"🔍 Validation files: {validation_count}")
            print(f"📋 Metadata files: {metadata_count}")
            print(f"⏱️  Estimated time remaining: {time_remaining}")
            
            # Progress bar
            bar_length = 30
            filled_length = int(bar_length * completed / total_expected_files)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            print(f"📈 Progress: [{bar}] {progress_percent:.1f}%")
            
            # Health checks
            print(f"\n🏥 Health Checks:")
            
            # Check if we have all expected files per completed shape
            if completed > 0:
                expected_objs_per_shape = 7  # 00_original through 05_scaled + unified
                actual_ratio = total_objs / completed if completed > 0 else 0
                if actual_ratio >= 6:  # At least 6 files per shape (some might not need remeshing)
                    print(f"✅ File generation: {actual_ratio:.1f} files/shape (healthy)")
                else:
                    print(f"⚠️  File generation: {actual_ratio:.1f} files/shape (check for issues)")
            
            # Check validation ratio
            if completed > 0:
                validation_ratio = validation_count / completed
                if validation_ratio > 0.8:
                    print(f"✅ Validation files: {validation_ratio:.1%} (healthy)")
                else:
                    print(f"⚠️  Validation files: {validation_ratio:.1%} (some missing)")
            
            # Check for recent activity
            recent_files = []
            cutoff_time = time.time() - 300  # Last 5 minutes
            for file_path in output_dir.glob("**/*.obj"):
                if file_path.stat().st_mtime > cutoff_time:
                    recent_files.append(file_path)
            
            if recent_files:
                print(f"✅ Recent activity: {len(recent_files)} files created in last 5 minutes")
            else:
                print(f"⚠️  No recent activity (might be stuck or finished)")
            
            # Warning if progress is too slow
            if completed > 10:  # Only after some shapes are done
                elapsed_hours = (time.time() - start_time) / 3600
                rate_per_hour = completed / elapsed_hours
                if rate_per_hour < 100:  # Less than 100 shapes per hour
                    print(f"⚠️  Processing rate: {rate_per_hour:.1f} shapes/hour (might be slow)")
                else:
                    print(f"✅ Processing rate: {rate_per_hour:.1f} shapes/hour (good)")
            
            if completed >= total_expected_files:
                print(f"\n🎉 PROCESSING COMPLETE! All {total_expected_files} shapes processed!")
                break
            
            print(f"\n💡 Press Ctrl+C to stop monitoring")
            time.sleep(30)  # Update every 30 seconds
            
        except KeyboardInterrupt:
            print(f"\n👋 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Error in monitoring: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()