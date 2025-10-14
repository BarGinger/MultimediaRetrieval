"""
Analysis CSV cache module to avoid redundant loading of analysis data.
This module provides a singleton pattern to load analysis CSV files only once
and reuse the data across the application.
"""

import pandas as pd
import os
from pathlib import Path
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
        
        # Try multiple potential paths for each dataset
        potential_paths = []
        
        # Direct dataset mappings - Updated to match enhanced preprocessing script logic
        path_mapping = {
            # Original datasets - analysis files saved in Preprocessing folder
            'Data': [
                'Preprocessing/analysis_results_data.csv',
                'Preprocessing/analysis_results.csv',
                'Datasets/Data/analysis_results.csv'
            ],
            'Data_sampled': [
                'Preprocessing/analysis_results_data_sampled.csv',
                'Preprocessing/analysis_results_sampled.csv',
                'Datasets/Data_sampled/analysis_results.csv'
            ],
            'Data_resampled': [
                'Preprocessing/analysis_results_data_resampled.csv',
                'Preprocessing/analysis_results_resampled.csv',
                'Datasets/Data_resampled/analysis_results.csv'
            ],
            'Data_sampled_resampled': [
                'Preprocessing/analysis_results_data_sampled_resampled.csv',
                'Preprocessing/analysis_results_sampled_resampled.csv',
                'Datasets/Data_sampled_resampled/analysis_results.csv'
            ],
            'Data_sampled_resampled_normalized': [
                'Preprocessing/analysis_results_data_sampled_resampled_normalized.csv',
                'Preprocessing/analysis_results_sampled_resampled_normalized.csv',
                'Datasets/Data_sampled_resampled_normalized/analysis_results.csv'
            ],
            'Data_sampled_resampled_simple': [
                'Preprocessing/analysis_results_data_sampled_resampled_simple.csv',
                'Preprocessing/analysis_results_sampled_resampled_simple.csv',
                'Datasets/Data_sampled_resampled_simple/analysis_results.csv'
            ],
            # Processed datasets - analysis files saved in dataset folder
            'UnifiedPreprocessed/Data': [
                'Datasets/UnifiedPreprocessed/Data/analysis_results.csv',
                'Datasets/UnifiedPreprocessed/analysis_results.csv',
                'Preprocessing/analysis_results_resampled.csv'  # fallback
            ],
            'UnifiedPreprocessed_Data': [
                'Datasets/UnifiedPreprocessed/Data/analysis_results.csv',
                'Datasets/UnifiedPreprocessed/analysis_results.csv',
                'analysis_results_unified.csv'
            ],
        }
        
        potential_paths = path_mapping.get(dataset, [])
        
        # Only add generic patterns if no specific mapping exists
        if not potential_paths:
            generic_patterns = [
                f'Preprocessing/analysis_results_{dataset.lower()}.csv',
                f'analysis_results_{dataset.lower()}.csv',
                f'Datasets/{dataset}/analysis_results.csv'
            ]
            potential_paths.extend(generic_patterns)
        
        # Find project root and check all potential paths
        project_root = Path(__file__).resolve().parent.parent
        root_candidates = [project_root, project_root.parent, project_root.parent.parent]
        
        for candidate_root in root_candidates:
            for path in potential_paths:
                full_path = candidate_root / path
                if full_path.exists():
                    print(f"[DEBUG] Found analysis CSV for {dataset}: {full_path}")
                    return str(full_path)
        
        print(f"[DEBUG] No analysis CSV found for dataset: {dataset}")
        print(f"[DEBUG] Tried paths: {potential_paths}")
        return None
    
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
        # File list shows only *_05_scaled.obj files, but analysis CSV contains all files
        file_df_copy = file_df.copy()
        
        # Create mapping for different filename patterns
        file_df_copy['base_filename'] = file_df_copy['filename'].copy()
        
        # For processed datasets, map from displayed filename to analysis filename
        if 'UnifiedPreprocessed' in dataset:
            # For UnifiedPreprocessed datasets:
            # - App shows: m1337_05_scaled.obj 
            # - Analysis CSV might have: m1337_05_scaled.obj (if generated by enhanced script)
            #   or m1337.obj (if using fallback analysis from original dataset)
            
            # Try exact match first, then fallback to base name without step suffix
            file_df_copy['fallback_filename'] = file_df_copy['filename'].str.replace(r'_\d+_scaled\.obj$', '.obj', regex=True)
        else:
            # For original datasets, filenames should match directly
            file_df_copy['fallback_filename'] = file_df_copy['filename']
        
        analysis_df_copy = analysis_df.copy()
        
        # First try exact filename match
        merged = pd.merge(
            file_df_copy, 
            analysis_df_copy[['category', 'filename', 'num_vertices', 'num_faces']],
            on=['category', 'filename'], 
            how='left'
        )
        
        # For unmatched entries, try fallback filename matching
        unmatched_mask = merged['num_vertices'].isna()
        if unmatched_mask.any() and 'fallback_filename' in file_df_copy.columns:
            print(f"[DEBUG] Trying fallback filename matching for {unmatched_mask.sum()} entries...")
            
            # Create temporary merge with fallback filenames
            unmatched_df = file_df_copy[unmatched_mask].copy()
            unmatched_df['filename'] = unmatched_df['fallback_filename']  # Use fallback for matching
            
            fallback_merged = pd.merge(
                unmatched_df[['category', 'filename', 'filepath', 'fallback_filename']],
                analysis_df_copy[['category', 'filename', 'num_vertices', 'num_faces']],
                on=['category', 'filename'],
                how='left'
            )
            
            # Update the main merged dataframe with fallback results
            for idx in fallback_merged.index:
                if not pd.isna(fallback_merged.loc[idx, 'num_vertices']):
                    orig_idx = unmatched_df.index[idx]
                    merged.loc[orig_idx, 'num_vertices'] = fallback_merged.loc[idx, 'num_vertices']
                    merged.loc[orig_idx, 'num_faces'] = fallback_merged.loc[idx, 'num_faces']
        
        # Clean up temporary columns
        merged = merged.drop(['fallback_filename'], axis=1, errors='ignore')
        
        # For rows still without analysis data, compute on-the-fly
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