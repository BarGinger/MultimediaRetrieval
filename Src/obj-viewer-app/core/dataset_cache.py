"""
High-performance dataset cache with persistent disk storage.
Eliminates repetitive file scanning and CSV merging operations.
Features intelligent cache invalidation based on file modification times.
"""

import pandas as pd
import os
import time
import pickle
import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple
from .file_index import get_file_tree
from .analysis_cache import get_analysis_data


class DatasetCache:
    """
    Singleton cache for pre-merged dataset information with persistent disk storage.
    Combines file tree data with analysis data once and caches both in memory and on disk.
    Automatically detects when source files change and re-merges only when necessary.
    """
    
    _instance: Optional['DatasetCache'] = None
    _cache: Dict[str, pd.DataFrame] = {}
    _metadata: Dict[str, dict] = {}
    _available_datasets: Optional[list] = None
    _cache_dir: Path = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatasetCache, cls).__new__(cls)
            cls._instance._initialize_cache_dir()
        return cls._instance
    
    def _initialize_cache_dir(self):
        """Initialize the cache directory."""
        # Create cache directory in the project root
        project_root = Path(__file__).resolve().parent.parent
        self._cache_dir = project_root / '.dataset_cache'
        self._cache_dir.mkdir(exist_ok=True)
        
        # Create metadata file if it doesn't exist
        metadata_file = self._cache_dir / 'metadata.json'
        if not metadata_file.exists():
            import json
            with open(metadata_file, 'w') as f:
                json.dump({}, f)
    
    def get_available_datasets(self) -> list:
        """Get list of available datasets (cached)."""
        if self._available_datasets is None:
            self._available_datasets = self._scan_available_datasets()
        return self._available_datasets
    
    def get_dataset_data(self, dataset: str, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get pre-merged dataset with file tree + analysis data.
        Uses persistent disk cache for lightning-fast subsequent loads.
        
        Args:
            dataset: Dataset name ('Data', 'Data_sampled', etc.)
            force_refresh: Force reload from disk and re-merge
            
        Returns:
            DataFrame with merged file and analysis data
        """
        # Check memory cache first
        if dataset in self._cache and not force_refresh:
            return self._cache[dataset].copy()
        
        # Check if we can load from disk cache
        cache_file = self._cache_dir / f"{self._safe_filename(dataset)}.pkl"
        metadata_file = self._cache_dir / f"{self._safe_filename(dataset)}_meta.json"
        
        if not force_refresh and cache_file.exists() and metadata_file.exists():
            if self._is_cache_valid(dataset, metadata_file):
                print(f"[DatasetCache] Loading {dataset} from disk cache...")
                start_time = time.time()
                
                try:
                    with open(cache_file, 'rb') as f:
                        merged_df = pickle.load(f)
                    
                    # Load metadata
                    import json
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Store in memory cache
                    self._cache[dataset] = merged_df
                    self._metadata[dataset] = metadata
                    
                    load_time = time.time() - start_time
                    print(f"[DatasetCache] Loaded {len(merged_df)} shapes from cache in {load_time:.3f}s")
                    
                    return merged_df.copy()
                    
                except Exception as e:
                    print(f"[DatasetCache] Failed to load cache for {dataset}: {e}")
                    # Fall through to re-merge
        
        # Need to merge from scratch
        print(f"[DatasetCache] Merging and caching data for {dataset}...")
        start_time = time.time()
        
        # Get file tree
        file_df = get_file_tree(dataset)
        
        # Fast merge with analysis data
        merged_df = self._fast_merge_analysis_data(file_df, dataset)
        
        # Cache in memory
        self._cache[dataset] = merged_df
        
        merge_time = time.time() - start_time
        metadata = {
            'load_time': merge_time,
            'shape_count': len(merged_df),
            'categories': merged_df['category'].nunique() if len(merged_df) > 0 else 0,
            'last_merged': time.time(),
            'source_timestamps': self._get_source_timestamps(dataset)
        }
        self._metadata[dataset] = metadata
        
        # Save to disk cache
        self._save_to_disk_cache(dataset, merged_df, metadata)
        
        print(f"[DatasetCache] Merged and cached {len(merged_df)} shapes in {merge_time:.2f}s")
        
        return merged_df.copy()
    
    def _is_cache_valid(self, dataset: str, metadata_file: Path) -> bool:
        """Check if disk cache is still valid by comparing file timestamps."""
        try:
            import json
            with open(metadata_file, 'r') as f:
                cached_metadata = json.load(f)
            
            if 'source_timestamps' not in cached_metadata:
                return False
            
            # Get current timestamps
            current_timestamps = self._get_source_timestamps(dataset)
            cached_timestamps = cached_metadata['source_timestamps']
            
            # Compare timestamps
            return current_timestamps == cached_timestamps
            
        except Exception as e:
            print(f"[DatasetCache] Cache validation failed for {dataset}: {e}")
            return False
    
    def _get_source_timestamps(self, dataset: str) -> dict:
        """Get modification timestamps of source files for cache validation."""
        timestamps = {}
        
        try:
            # Get dataset directory timestamp
            cwd = Path.cwd()
            dataset_path = f"Datasets/{dataset}"
            candidates = [cwd / dataset_path, cwd.parent / dataset_path, cwd.parent.parent / dataset_path]
            data_path = next((p for p in candidates if p.exists()), None)
            
            if data_path and data_path.exists():
                # Get overall directory modification time
                timestamps['dataset_dir'] = data_path.stat().st_mtime
                
                # Sample a few files for deeper validation (not all files for performance)
                sample_files = []
                for category_dir in data_path.iterdir():
                    if category_dir.is_dir():
                        obj_files = list(category_dir.glob("*.obj"))
                        if obj_files:
                            # Sample first and last file from each category
                            sample_files.extend([obj_files[0], obj_files[-1]])
                            if len(sample_files) >= 10:  # Limit sampling for performance
                                break
                
                # Get timestamps of sample files
                for i, file_path in enumerate(sample_files[:10]):
                    timestamps[f'sample_{i}'] = file_path.stat().st_mtime
            
            # Get analysis CSV timestamps if available
            analysis_paths = [
                f"Preprocessing/analysis_results_{dataset.lower()}.csv",
                f"Preprocessing/analysis_results.csv"
            ]
            
            for analysis_path in analysis_paths:
                for root_candidate in [cwd, cwd.parent, cwd.parent.parent]:
                    full_path = root_candidate / analysis_path
                    if full_path.exists():
                        timestamps['analysis_csv'] = full_path.stat().st_mtime
                        break
                        
        except Exception as e:
            print(f"[DatasetCache] Warning: Could not get timestamps for {dataset}: {e}")
        
        return timestamps
    
    def _save_to_disk_cache(self, dataset: str, data: pd.DataFrame, metadata: dict):
        """Save merged dataset to disk cache."""
        try:
            cache_file = self._cache_dir / f"{self._safe_filename(dataset)}.pkl"
            metadata_file = self._cache_dir / f"{self._safe_filename(dataset)}_meta.json"
            
            # Save data
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save metadata
            import json
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            print(f"[DatasetCache] Saved {dataset} to disk cache ({cache_file.stat().st_size / 1024 / 1024:.1f}MB)")
            
        except Exception as e:
            print(f"[DatasetCache] Warning: Could not save cache for {dataset}: {e}")
    
    def _safe_filename(self, dataset: str) -> str:
        """Convert dataset name to safe filename."""
        # Replace problematic characters
        safe_name = dataset.replace('/', '_').replace('\\', '_').replace(':', '_')
        return safe_name
    
    def get_dataset_metadata(self, dataset: str) -> dict:
        """Get metadata about a cached dataset."""
        return self._metadata.get(dataset, {})
    
    def preload_all_datasets(self):
        """Preload all available datasets into cache."""
        datasets = self.get_available_datasets()
        print(f"[DatasetCache] Preloading {len(datasets)} datasets...")
        
        total_start = time.time()
        cache_hits = 0
        cache_misses = 0
        
        for dataset in datasets:
            try:
                cache_file = self._cache_dir / f"{self._safe_filename(dataset)}.pkl"
                if cache_file.exists():
                    cache_hits += 1
                else:
                    cache_misses += 1
                    
                self.get_dataset_data(dataset)
            except Exception as e:
                print(f"[DatasetCache] Failed to preload {dataset}: {e}")
        
        total_time = time.time() - total_start
        print(f"[DatasetCache] Preloaded all datasets in {total_time:.2f}s")
        print(f"[DatasetCache] Cache performance: {cache_hits} hits, {cache_misses} misses")
    
    def clear_cache(self, disk_cache: bool = False):
        """Clear cached data."""
        self._cache.clear()
        self._metadata.clear()
        self._available_datasets = None
        
        if disk_cache:
            try:
                import shutil
                if self._cache_dir.exists():
                    shutil.rmtree(self._cache_dir)
                    self._initialize_cache_dir()
                print("[DatasetCache] Disk cache cleared")
            except Exception as e:
                print(f"[DatasetCache] Error clearing disk cache: {e}")
        
        print("[DatasetCache] Memory cache cleared")
    
    def get_cache_info(self) -> dict:
        """Get information about cached datasets."""
        disk_files = list(self._cache_dir.glob("*.pkl")) if self._cache_dir.exists() else []
        disk_size = sum(f.stat().st_size for f in disk_files) / 1024 / 1024  # MB
        
        return {
            'cached_datasets': list(self._cache.keys()),
            'memory_cache_size_mb': sum(df.memory_usage(deep=True).sum() for df in self._cache.values()) / 1024 / 1024,
            'disk_cache_size_mb': disk_size,
            'disk_cache_files': len(disk_files),
            'cache_directory': str(self._cache_dir),
            'metadata': self._metadata
        }
    
    def _scan_available_datasets(self) -> list:
        """Scan for available datasets in the file system."""
        datasets = []
        cwd = Path.cwd()
        
        # Look for Datasets folder
        candidates = [cwd / "Datasets", cwd.parent / "Datasets", cwd.parent.parent / "Datasets"]
        datasets_path = next((p for p in candidates if p.exists()), None)
        
        if datasets_path:
            print(f"[DatasetCache] Scanning datasets in: {datasets_path}")
            
            for item in datasets_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    # Check if it has the expected structure (category directories with .obj files)
                    has_shapes = self._check_for_valid_categories(item)
                    
                    if has_shapes:
                        datasets.append(item.name)
                        print(f"[DatasetCache] Found dataset: {item.name}")
                    
                    # Also check for nested datasets (e.g., UnifiedPreprocessed/Data)
                    else:
                        for nested_item in item.iterdir():
                            if nested_item.is_dir() and not nested_item.name.startswith('.'):
                                if self._check_for_valid_categories(nested_item):
                                    nested_name = f"{item.name}/{nested_item.name}"
                                    datasets.append(nested_name)
                                    print(f"[DatasetCache] Found nested dataset: {nested_name}")
        
        datasets.sort()  # Sort alphabetically
        print(f"[DatasetCache] Total datasets found: {len(datasets)}")
        return datasets
    
    def _check_for_valid_categories(self, dataset_path: Path) -> bool:
        """Check if a path contains valid category folders with OBJ files."""
        try:
            for category_dir in dataset_path.iterdir():
                if category_dir.is_dir() and not category_dir.name.startswith('.'):
                    # Check if category has OBJ files
                    obj_files = list(category_dir.glob("*.obj"))
                    if obj_files:
                        return True
        except Exception as e:
            print(f"[DatasetCache] Error checking {dataset_path}: {e}")
        return False
    
    def _fast_merge_analysis_data(self, file_df: pd.DataFrame, dataset: str) -> pd.DataFrame:
        """
        Fast merge of file data with analysis data.
        Similar to the existing fast_merge_analysis_data but optimized.
        """
        if len(file_df) == 0:
            return file_df
        
        analysis_df = get_analysis_data(dataset)
        if analysis_df is not None:
            # Optimize the merge operation
            file_df_copy = file_df.copy()
            analysis_df_copy = analysis_df.copy()
            
            # For UnifiedPreprocessed datasets, need to map processed filenames to original
            if 'UnifiedPreprocessed' in dataset:
                # Convert processed filenames to original base names:
                # m1337_05_scaled.obj -> m1337.obj
                # m1337_unified.obj -> m1337.obj
                file_df_copy['base_filename'] = file_df_copy['filename'].str.replace(r'_(\d{2}_.*|unified)\.obj$', '.obj', regex=True)
                
                # Ensure analysis CSV uses correct column names 
                if 'class' in analysis_df_copy.columns:
                    analysis_df_copy = analysis_df_copy.rename(columns={'class': 'category'})
                if 'shape_file' in analysis_df_copy.columns:
                    analysis_df_copy = analysis_df_copy.rename(columns={'shape_file': 'filename'})
                
                analysis_df_copy['base_filename'] = analysis_df_copy['filename']
                
                # Merge on category and base filename
                merged = pd.merge(
                    file_df_copy, 
                    analysis_df_copy[['category', 'base_filename', 'num_vertices', 'num_faces']],
                    on=['category', 'base_filename'], 
                    how='left'
                ).drop('base_filename', axis=1)
                
                print(f"[DEBUG] UnifiedPreprocessed merge: {len(file_df_copy)} files, {len(analysis_df_copy)} analysis rows, {merged['num_vertices'].notna().sum()} matches")
            else:
                # For other datasets, use the original conversion logic
                file_df_copy['base_filename'] = file_df_copy['filename'].str.replace('_unified.obj', '.obj')
                analysis_df_copy['base_filename'] = analysis_df_copy['filename']
                
                # Use efficient merge
                merged = pd.merge(
                    file_df_copy, 
                    analysis_df_copy[['category', 'base_filename', 'num_vertices', 'num_faces']],
                    on=['category', 'base_filename'], 
                    how='left'
                ).drop('base_filename', axis=1)
            
            # Fill missing values with reasonable defaults
            merged['num_vertices'] = merged['num_vertices'].fillna(0).astype('int32')
            merged['num_faces'] = merged['num_faces'].fillna(0).astype('int32')
            
            return merged
        else:
            # No cached data - add empty columns with proper types
            file_df = file_df.copy()
            file_df['num_vertices'] = 0
            file_df['num_faces'] = 0
            return file_df


# Global cache instance
_dataset_cache = DatasetCache()

def get_cached_dataset_data(dataset: str, force_refresh: bool = False) -> pd.DataFrame:
    """Get pre-merged dataset data from cache."""
    return _dataset_cache.get_dataset_data(dataset, force_refresh)

def get_available_datasets() -> list:
    """Get list of available datasets."""
    return _dataset_cache.get_available_datasets()

def preload_datasets():
    """Preload all datasets for faster switching."""
    _dataset_cache.preload_all_datasets()

def get_cache_info() -> dict:
    """Get cache information for debugging."""
    return _dataset_cache.get_cache_info()

def clear_dataset_cache(disk_cache: bool = False):
    """
    Clear the dataset cache.
    
    Args:
        disk_cache: If True, also clear persistent disk cache
    """
    _dataset_cache.clear_cache(disk_cache)