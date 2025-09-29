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
        # Handle nested datasets (e.g., "UnifiedPreprocessed/Data")
        if "/" in dataset:
            # For nested datasets, try to find analysis for the base dataset
            base_dataset = dataset.split("/")[-1]  # Get "Data" from "UnifiedPreprocessed/Data"
            return self._get_analysis_path(base_dataset)
        
        # Direct dataset mappings
        path_mapping = {
            'Data': 'Preprocessing/analysis_results.csv',
            'Data_sampled': 'Preprocessing/analysis_results_sampled.csv',
            'Data_resampled': 'Preprocessing/analysis_results_resampled.csv',
            'Data_sampled_resampled': 'Preprocessing/analysis_results_sampled_resampled.csv',
            'Data_sampled_resampled_normalized': 'Preprocessing/analysis_results_sampled_resampled_normalized.csv',
            'UnifiedPreprocessed/Data': 'Datasets/UnifiedPreprocessed/analysis_results_data.csv',
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
    If analysis data is not available, compute basic stats on-the-fly.
    
    Args:
        file_df: DataFrame with file information
        dataset: Dataset name
        
    Returns:
        DataFrame with merged analysis data
    """
    analysis_df = get_analysis_data(dataset)
    
    if analysis_df is not None:
        # Handle filename matching for processed files
        # Processed files have "_unified.obj" suffix, original analysis has ".obj"
        file_df_copy = file_df.copy()
        
        # Create a mapping column for matching
        file_df_copy['base_filename'] = file_df_copy['filename'].str.replace('_unified.obj', '.obj')
        analysis_df_copy = analysis_df.copy()
        analysis_df_copy['base_filename'] = analysis_df_copy['filename']
        
        # Merge on category and base filename
        merged = pd.merge(
            file_df_copy, 
            analysis_df_copy[['category', 'base_filename', 'num_vertices', 'num_faces']],
            on=['category', 'base_filename'], 
            how='left'
        )
        
        # Drop the temporary matching column
        merged = merged.drop('base_filename', axis=1)
        
        # For rows without analysis data, compute on-the-fly
        missing_analysis = merged['num_vertices'].isna()
        if missing_analysis.any():
            print(f"Computing analysis for {missing_analysis.sum()} files without cached data...")
            for idx in merged[missing_analysis].index:
                try:
                    filepath = merged.loc[idx, 'filepath']
                    vertices, faces = compute_basic_analysis(filepath)
                    merged.loc[idx, 'num_vertices'] = len(vertices) if vertices is not None else 0
                    merged.loc[idx, 'num_faces'] = len(faces) if faces is not None else 0
                except Exception as e:
                    print(f"Warning: Could not analyze {filepath}: {e}")
                    merged.loc[idx, 'num_vertices'] = 0
                    merged.loc[idx, 'num_faces'] = 0
        
        return merged
    else:
        # No cached analysis available, compute everything on-the-fly
        print(f"No cached analysis for {dataset}, computing on-the-fly...")
        file_df_copy = file_df.copy()
        file_df_copy['num_vertices'] = 0
        file_df_copy['num_faces'] = 0
        
        for idx, row in file_df_copy.iterrows():
            try:
                vertices, faces = compute_basic_analysis(row['filepath'])
                file_df_copy.loc[idx, 'num_vertices'] = len(vertices) if vertices is not None else 0
                file_df_copy.loc[idx, 'num_faces'] = len(faces) if faces is not None else 0
            except Exception as e:
                print(f"Warning: Could not analyze {row['filepath']}: {e}")
        
        return file_df_copy


def compute_basic_analysis(filepath: str):
    """
    Compute basic analysis (vertices, faces) for an OBJ file.
    
    Args:
        filepath: Path to OBJ file
        
    Returns:
        tuple: (vertices, faces) or (None, None) if failed
    """
    try:
        from .obj_parser import OBJParser
        vertices, faces = OBJParser.parse_obj_file(filepath)
        return vertices, faces
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None, None