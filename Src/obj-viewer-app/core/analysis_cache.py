"""
Analysis CSV cache module to avoid redundant loading of analysis data.
This module provides a singleton pattern to load analysis CSV files only once
and reuse the data across the application.
"""

import pandas as pd
import os
from typing import Dict, Optional


class AnalysisCache:
    """Singleton cache for analysis CSV data."""
    
    _instance: Optional['AnalysisCache'] = None
    _cache: Dict[str, pd.DataFrame] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AnalysisCache, cls).__new__(cls)
        return cls._instance
    
    def get_analysis_data(self, dataset: str) -> Optional[pd.DataFrame]:
        """
        Get analysis data for a specific dataset.
        
        Args:
            dataset: Dataset name ('Data', 'Data_sampled', etc.)
            
        Returns:
            DataFrame with analysis data or None if not available
        """
        if dataset in self._cache:
            return self._cache[dataset].copy()  # Return copy to prevent modifications
        
        analysis_path = self._get_analysis_path(dataset)
        if not analysis_path:
            return None
            
        try:
            if os.path.exists(analysis_path):
                df = pd.read_csv(analysis_path)
                # Standardize column names
                df = df.rename(columns={
                    'class': 'category',
                    'shape_file': 'filename'
                })
                self._cache[dataset] = df
                return df.copy()
        except Exception as e:
            print(f"[DEBUG] Could not load analysis CSV for {dataset}: {e}")
            
        return None
    
    def _get_analysis_path(self, dataset: str) -> Optional[str]:
        """Get the analysis CSV path for a dataset."""
        path_mapping = {
            'Data': 'Preprocessing/analysis_results.csv',
            'Data_sampled': 'Preprocessing/analysis_results_sampled.csv',
            'Data_sampled_resampled': 'Preprocessing/analysis_results_sampled_resampled.csv',
            'Data_sampled_resampled_normalized': 'Preprocessing/analysis_results_sampled_resampled_normalized.csv'
        }
        return path_mapping.get(dataset)
    
    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
    
    def refresh_dataset(self, dataset: str):
        """Refresh data for a specific dataset."""
        if dataset in self._cache:
            del self._cache[dataset]
        return self.get_analysis_data(dataset)


# Convenience function for global access
def get_analysis_data(dataset: str) -> Optional[pd.DataFrame]:
    """Get analysis data for a dataset using the global cache."""
    cache = AnalysisCache()
    return cache.get_analysis_data(dataset)


def merge_analysis_data(file_df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """
    Merge file DataFrame with analysis data for a dataset.
    
    Args:
        file_df: DataFrame with file information
        dataset: Dataset name
        
    Returns:
        DataFrame with merged analysis data
    """
    analysis_df = get_analysis_data(dataset)
    if analysis_df is not None:
        # Merge on category and filename
        merged = pd.merge(
            file_df, 
            analysis_df[['category', 'filename', 'num_vertices', 'num_faces']],
            on=['category', 'filename'], 
            how='left'
        )
        return merged
    return file_df