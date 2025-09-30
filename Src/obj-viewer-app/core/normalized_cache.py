"""
Efficient loader for pre-normalized shapes
"""

import os
import json
import numpy as np
from pathlib import Path
from core.obj_parser import OBJParser
from core.shapeMesh import ShapeMesh

class NormalizedShapeCache:
    """Efficient cache for pre-normalized shapes"""
    
    def __init__(self, normalized_shapes_dir=None):
        # Use the same path resolution pattern as file_index.py
        if normalized_shapes_dir is None:
            cwd = Path.cwd()
            dataset_path = "Datasets/NormalizedShapes"
            candidates = [cwd / dataset_path, cwd.parent / dataset_path, cwd.parent.parent / dataset_path]
            self.base_dir = next((p for p in candidates if p.exists()), candidates[-1])
        else:
            self.base_dir = Path(normalized_shapes_dir)
        self._cache = {}  # In-memory cache for frequently accessed shapes
        self._metadata_cache = {}
    
    def get_normalized_obj_path(self, filename, dataset):
        """Get path to normalized OBJ file"""
        base_name = Path(filename).stem
        return self.base_dir / dataset / f"{base_name}_normalized.obj"
    
    def get_metadata_path(self, filename, dataset):
        """Get path to metadata JSON file"""
        base_name = Path(filename).stem
        return self.base_dir / dataset / f"{base_name}_metadata.json"
    
    def is_normalized_available(self, filename, dataset):
        """Check if normalized version exists"""
        obj_path = self.get_normalized_obj_path(filename, dataset)
        metadata_path = self.get_metadata_path(filename, dataset)
        return obj_path.exists() and metadata_path.exists()
    
    def load_normalized_metadata(self, filename, dataset):
        """Load normalization metadata"""
        cache_key = f"{dataset}::{filename}::metadata"
        
        if cache_key in self._metadata_cache:
            return self._metadata_cache[cache_key]
        
        metadata_path = self.get_metadata_path(filename, dataset)
        
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Cache for future use
            self._metadata_cache[cache_key] = metadata
            return metadata
            
        except Exception as e:
            print(f"Error loading metadata for {filename}: {e}")
            return None
    
    def load_normalized_shape(self, filename, dataset, use_cache=True):
        """Load pre-normalized shape efficiently"""
        cache_key = f"{dataset}::{filename}::normalized"
        
        # Check in-memory cache first
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        obj_path = self.get_normalized_obj_path(filename, dataset)
        
        if not obj_path.exists():
            return None
        
        try:
            # Load normalized OBJ file
            vertices, faces = OBJParser.parse_obj_file(str(obj_path))
            
            # Load metadata
            metadata = self.load_normalized_metadata(filename, dataset)
            
            # Create ShapeMesh with normalized vertices
            mesh = ShapeMesh(
                vertices=vertices,
                faces=faces,
                category=metadata.get('category') if metadata else None,
                filename=filename,
                filepath=str(obj_path)
            )
            
            # Add normalization info from metadata
            if metadata:
                mesh._normalization_metadata = metadata['normalization_info']
            
            # Cache for future use (limit cache size)
            if use_cache:
                if len(self._cache) > 100:  # Limit cache size
                    # Remove oldest entry
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                
                self._cache[cache_key] = mesh
            
            return mesh
            
        except Exception as e:
            print(f"Error loading normalized shape {filename}: {e}")
            return None
    
    def get_normalization_stats(self, dataset):
        """Get normalization statistics for a dataset"""
        report_path = self.base_dir / "normalization_report.json"
        
        if not report_path.exists():
            return None
        
        try:
            with open(report_path, 'r') as f:
                report = json.load(f)
            return report
        except Exception as e:
            print(f"Error loading normalization report: {e}")
            return None
    
    def clear_cache(self):
        """Clear in-memory cache to free memory"""
        self._cache.clear()
        self._metadata_cache.clear()

# Global instance for easy access
normalized_cache = NormalizedShapeCache()