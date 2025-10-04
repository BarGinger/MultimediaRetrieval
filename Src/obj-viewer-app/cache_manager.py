#!/usr/bin/env python3
"""
Dataset Cache Management Utility

This script provides utilities for managing the persistent dataset cache.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path so we can import modules
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.dataset_cache import get_cache_info, clear_dataset_cache, preload_datasets


def print_cache_info():
    """Print detailed cache information."""
    print("📊 Dataset Cache Information")
    print("=" * 50)
    
    info = get_cache_info()
    
    print(f"Memory Cache:")
    print(f"  - Cached datasets: {len(info['cached_datasets'])}")
    print(f"  - Memory usage: {info['memory_cache_size_mb']:.1f} MB")
    
    print(f"\nDisk Cache:")
    print(f"  - Cache directory: {info['cache_directory']}")
    print(f"  - Disk usage: {info['disk_cache_size_mb']:.1f} MB")
    print(f"  - Cache files: {info['disk_cache_files']}")
    
    if info['cached_datasets']:
        print(f"\nCached Datasets:")
        for dataset in info['cached_datasets']:
            metadata = info['metadata'].get(dataset, {})
            print(f"  - {dataset}: {metadata.get('shape_count', 0)} shapes")
    else:
        print("\nNo datasets currently cached in memory.")


def clear_cache(disk_cache=False):
    """Clear the cache."""
    if disk_cache:
        print("🗑️  Clearing both memory and disk cache...")
        clear_dataset_cache(disk_cache=True)
        print("✅ Both memory and disk cache cleared!")
    else:
        print("🗑️  Clearing memory cache...")
        clear_dataset_cache(disk_cache=False)
        print("✅ Memory cache cleared!")


def preload_cache():
    """Preload all datasets into cache."""
    print("⚡ Preloading all datasets...")
    preload_datasets()
    print("✅ All datasets preloaded!")


def main(command=None):
    """Main command-line interface."""
    if len(sys.argv) < 2:
        print("Dataset Cache Management Utility")
        print("\nUsage:")
        print("  python cache_manager.py info          - Show cache information")
        print("  python cache_manager.py clear         - Clear memory cache")
        print("  python cache_manager.py clear-all     - Clear memory and disk cache")
        print("  python cache_manager.py preload       - Preload all datasets")
        return
    
    command = sys.argv[1].lower()
    
    if command == "info":
        print_cache_info()
    elif command == "clear":
        clear_cache(disk_cache=False)
    elif command == "clear-all":
        clear_cache(disk_cache=True)
    elif command == "preload":
        preload_cache()
    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: info, clear, clear-all, preload")


if __name__ == "__main__":
    main() 