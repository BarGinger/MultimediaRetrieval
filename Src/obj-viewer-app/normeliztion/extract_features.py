"""
Feature Extraction Script for 3D Shapes
Runs all feature extraction methods from extractions.py on shapes in a dataset
Exports results to CSV with shape information and calculated features
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import time
import csv
import traceback

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.file_index import get_file_tree
from core.shapeMesh import ShapeMesh
from core.extractions import MeshExtractions


class FeatureExtractionProcessor:
    def __init__(self, output_base_dir=None, enabled_features=None):
        """
        Feature extraction processor for 3D shapes
        
        Parameters:
            output_base_dir (str or Path): Output directory for feature extraction results
            enabled_features (list): List of features to extract. If None, extracts all available features.
                                   Options: ['surface_area', 'compactness', 'rectangularity', 
                                           'diameter', 'convexity', 'eccentricity', 'basic_features', 'derived_features']
        """
        # Use same path resolution pattern
        if output_base_dir is None:
            cwd = Path.cwd()
            dataset_path = "Datasets/FeatureExtractions"
            candidates = [cwd / dataset_path, cwd.parent / dataset_path, cwd.parent.parent / dataset_path]
            self.output_base_dir = next((p for p in candidates if p.parent.exists()), candidates[-1])
        else:
            self.output_base_dir = Path(output_base_dir)
        
        # Define all available features
        self.all_extraction_features = [
            'surface_area',
            'compactness', 
            'rectangularity',
            'diameter',
            'convexity',
            'eccentricity'
        ]
        
        self.basic_features = ['volume', 'num_vertices', 'num_faces', 'bbox_dimensions']
        self.derived_features = ['aspect_ratios', 'sphericity', 'compactness_normalized']
        
        # Set enabled features
        if enabled_features is None:
            # Enable all features by default
            self.enabled_extraction_features = self.all_extraction_features.copy()
            self.enable_basic_features = True
            self.enable_derived_features = True
        else:
            self.enabled_extraction_features = []
            self.enable_basic_features = False
            self.enable_derived_features = False
            
            for feature in enabled_features:
                if feature in self.all_extraction_features:
                    self.enabled_extraction_features.append(feature)
                elif feature == 'basic_features':
                    self.enable_basic_features = True
                elif feature == 'derived_features':
                    self.enable_derived_features = True
        
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'start_time': None,
            'errors': [],
            'feature_stats': {
                'by_category': {},
                'feature_ranges': {}
            }
        }
        
        print(f"🔧 Feature extraction configuration:")
        print(f"   Extraction features: {self.enabled_extraction_features}")
        print(f"   Basic features: {'Enabled' if self.enable_basic_features else 'Disabled'}")
        print(f"   Derived features: {'Enabled' if self.enable_derived_features else 'Disabled'}")
    
    def setup_output_directories(self, datasets):
        """Create output directories for each dataset"""
        print(f"📁 Setting up output directories in: {self.output_base_dir.absolute()}")
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        for dataset in datasets:
            dataset_dir = self.output_base_dir / dataset
            dataset_dir.mkdir(parents=True, exist_ok=True)
            print(f"   Created: {dataset_dir}")
    
    def extract_shape_features(self, mesh, debug=False):
        """
        Extract all enabled features for a single shape
        
        Args:
            mesh: ShapeMesh object
            debug: If True, print feature values
            
        Returns:
            features: dict with all extracted features
        """
        features = {}
        
        try:
            # Basic shape information (if enabled)
            if self.enable_basic_features:
                features['num_vertices'] = mesh.num_vertices if mesh.num_vertices is not None else 0
                features['num_faces'] = mesh.num_faces if mesh.num_faces is not None else 0
                
                # Safe volume extraction
                try:
                    volume = mesh.volume
                    features['volume'] = float(volume) if volume is not None else 0.0
                except:
                    features['volume'] = 0.0
                
                # Bounding box information
                try:
                    dimensions = mesh.dimensions
                    if dimensions is not None and len(dimensions) >= 3:
                        features['bbox_width'] = float(dimensions[0])
                        features['bbox_height'] = float(dimensions[1])
                        features['bbox_depth'] = float(dimensions[2])
                        features['bbox_volume'] = float(np.prod(dimensions))
                    else:
                        features['bbox_width'] = 0.0
                        features['bbox_height'] = 0.0
                        features['bbox_depth'] = 0.0
                        features['bbox_volume'] = 0.0
                except:
                    features['bbox_width'] = 0.0
                    features['bbox_height'] = 0.0
                    features['bbox_depth'] = 0.0
                    features['bbox_volume'] = 0.0
            
            # Extract features using ShapeMesh properties (safer than MeshExtractions)
            for feature_name in self.enabled_extraction_features:
                try:
                    if feature_name == 'surface_area':
                        # Use ShapeMesh area property
                        area = mesh.area
                        features[feature_name] = float(area) if area is not None else 0.0
                        
                    elif feature_name == 'compactness':
                        # Safe compactness calculation
                        area = mesh.area
                        volume = mesh.volume
                        if area is not None and volume is not None and volume > 0:
                            features[feature_name] = float((area ** 1.5) / volume)
                        else:
                            features[feature_name] = 0.0
                            
                    elif feature_name == 'rectangularity':
                        # Use bounding box approximation since OBB not available
                        volume = mesh.volume
                        dimensions = mesh.dimensions
                        if volume is not None and dimensions is not None and len(dimensions) >= 3:
                            bbox_volume = np.prod(dimensions)
                            if bbox_volume > 0:
                                features[feature_name] = float(volume / bbox_volume)
                            else:
                                features[feature_name] = 0.0
                        else:
                            features[feature_name] = 0.0
                            
                    elif feature_name == 'diameter':
                        # Use ShapeMesh diameter property
                        diameter = mesh.diameter
                        features[feature_name] = float(diameter) if diameter is not None else 0.0
                        
                    elif feature_name == 'convexity':
                        # Use ShapeMesh convexity property
                        convexity = mesh.convexity
                        features[feature_name] = float(convexity) if convexity is not None else 0.0
                        
                    elif feature_name == 'eccentricity':
                        # Use ShapeMesh eccentricity property
                        eccentricity = mesh.eccentricity
                        features[feature_name] = float(eccentricity) if eccentricity is not None else 0.0
                    
                    else:
                        print(f"⚠️  Unknown feature: {feature_name}")
                        features[feature_name] = 0.0
                        
                    if debug and feature_name in features:
                        print(f"  {feature_name}: {features[feature_name]:.6f}")
                        
                except Exception as e:
                    print(f"❌ Error extracting {feature_name}: {e}")
                    features[feature_name] = 0.0
            
            # Additional derived features (if enabled)
            if self.enable_derived_features:
                try:
                    # Get dimensions safely
                    dimensions = mesh.dimensions
                    if dimensions is not None and len(dimensions) >= 3:
                        dims = np.array(dimensions)
                        
                        # Aspect ratios
                        if dims[1] > 0:
                            features['aspect_ratio_xy'] = float(dims[0] / dims[1])
                        else:
                            features['aspect_ratio_xy'] = 0.0
                            
                        if dims[2] > 0:
                            features['aspect_ratio_xz'] = float(dims[0] / dims[2])
                            features['aspect_ratio_yz'] = float(dims[1] / dims[2])
                        else:
                            features['aspect_ratio_xz'] = 0.0
                            features['aspect_ratio_yz'] = 0.0
                    else:
                        features['aspect_ratio_xy'] = 0.0
                        features['aspect_ratio_xz'] = 0.0
                        features['aspect_ratio_yz'] = 0.0
                    
                    # Sphericity approximation
                    if 'surface_area' in features and 'volume' in features:
                        surface_area = features.get('surface_area', 0)
                        volume = features.get('volume', 0)
                        if surface_area > 0 and volume > 0:
                            features['sphericity'] = float((np.pi ** (1/3)) * ((6 * volume) ** (2/3)) / surface_area)
                        else:
                            features['sphericity'] = 0.0
                    else:
                        features['sphericity'] = 0.0
                        
                    # Compactness alternative (normalized)
                    if 'surface_area' in features and 'volume' in features:
                        surface_area = features.get('surface_area', 0)
                        volume = features.get('volume', 0)
                        if volume > 0 and surface_area > 0:
                            features['compactness_normalized'] = float(surface_area / (volume ** (2/3)))
                        else:
                            features['compactness_normalized'] = 0.0
                    else:
                        features['compactness_normalized'] = 0.0
                        
                except Exception as e:
                    print(f"❌ Error calculating derived features: {e}")
                    # Set default values for derived features
                    features.update({
                        'aspect_ratio_xy': 0.0,
                        'aspect_ratio_xz': 0.0, 
                        'aspect_ratio_yz': 0.0,
                        'sphericity': 0.0,
                        'compactness_normalized': 0.0
                    })
            
        except Exception as e:
            print(f"❌ Error in feature extraction: {e}")
            # Initialize enabled features to 0 if extraction fails completely
            for feature_name in self.enabled_extraction_features:
                features[feature_name] = 0.0
            
            if self.enable_basic_features:
                features.update({
                    'num_vertices': 0, 'num_faces': 0, 'volume': 0.0,
                    'bbox_width': 0.0, 'bbox_height': 0.0, 'bbox_depth': 0.0, 'bbox_volume': 0.0
                })
            
            if self.enable_derived_features:
                features.update({
                    'aspect_ratio_xy': 0.0, 'aspect_ratio_xz': 0.0, 'aspect_ratio_yz': 0.0,
                    'sphericity': 0.0, 'compactness_normalized': 0.0
                })
        
        return features
    
    def process_shape(self, row, dataset_name, use_normalized=False):
        """
        Process a single shape for feature extraction
        
        Args:
            row: DataFrame row with shape information
            dataset_name: Name of the dataset being processed
            use_normalized: If True, try to load normalized version first
            
        Returns:
            dict: Shape information + extracted features, or None if failed
        """
        try:
            original_filepath = Path(row['filepath'])
            normalized_version_used = False
            
            # Check if we're working with a unified dataset (already processed files)
            is_unified_dataset = '_unified.obj' in original_filepath.name
            
            # Load the shape
            if use_normalized and not is_unified_dataset:
                # Try to load normalized version first (check both old and new formats)
                mesh = self._try_load_normalized_shape(row, dataset_name)
                if mesh is not None:
                    normalized_version_used = True
                else:
                    # Fallback to original
                    mesh = ShapeMesh.from_file_row(row, use_normalized=False)
            else:
                # Either not using normalized, or already working with unified files
                mesh = ShapeMesh.from_file_row(row, use_normalized=False)
                if is_unified_dataset:
                    normalized_version_used = True  # Unified files are inherently normalized
            
            if mesh is None:
                print(f"❌ Failed to load shape: {original_filepath.name}")
                return None
            
            # Extract features
            print(f"  🔍 Extracting features from: {original_filepath.name}")
            features = self.extract_shape_features(mesh, debug=False)
            
            # Calculate relative path from Datasets folder
            try:
                # Find the "Datasets" part in the path and get everything after it
                path_parts = original_filepath.parts
                datasets_index = next(i for i, part in enumerate(path_parts) if part == "Datasets")
                relative_path = "/".join(path_parts[datasets_index + 1:])
            except (StopIteration, ValueError):
                # Fallback to just dataset/category/filename if Datasets not found
                relative_path = f"{dataset_name}/{original_filepath.parent.name}/{original_filepath.name}"
            
            # Combine shape info with features
            result = {
                'filename': mesh.filename,
                'category': mesh.category,
                'filepath': relative_path,
                'file_size_bytes': row.get('size', 0),
                'dataset': dataset_name,
                'normalized_version_used': normalized_version_used
            }
            
            # Add all extracted features
            result.update(features)
            
            # Update statistics
            category = mesh.category or 'Unknown'
            if category not in self.stats['feature_stats']['by_category']:
                self.stats['feature_stats']['by_category'][category] = {
                    'count': 0,
                    'avg_features': {}
                }
            
            self.stats['feature_stats']['by_category'][category]['count'] += 1
            
            # Track feature ranges for validation (safe None handling)
            all_feature_names = list(features.keys())
            for feature_name in all_feature_names:
                if feature_name in features and features[feature_name] is not None:
                    feature_value = features[feature_name]
                    
                    # Ensure feature_value is a number
                    try:
                        feature_value = float(feature_value)
                    except (ValueError, TypeError):
                        continue  # Skip non-numeric values
                    
                    if feature_name not in self.stats['feature_stats']['feature_ranges']:
                        self.stats['feature_stats']['feature_ranges'][feature_name] = {
                            'min': feature_value,
                            'max': feature_value,
                            'values': []
                        }
                    
                    range_info = self.stats['feature_stats']['feature_ranges'][feature_name]
                    range_info['min'] = min(range_info['min'], feature_value)
                    range_info['max'] = max(range_info['max'], feature_value)
                    range_info['values'].append(feature_value)
            
            self.stats['successful'] += 1
            self.stats['total_processed'] += 1
            return result
            
        except Exception as e:
            import traceback
            error_msg = f"Error processing {row.get('filename', 'unknown')}: {str(e)}"
            print(f"\n❌ {error_msg}")
            traceback.print_exc()
            self.stats['errors'].append(error_msg)
            self.stats['failed'] += 1
            self.stats['total_processed'] += 1
            return None
    
    def _try_load_normalized_shape(self, row, dataset_name):
        """
        Try to load normalized version of shape, checking both old and new formats
        
        Returns:
            ShapeMesh if normalized version found and loaded, None otherwise
        """
        try:
            original_filepath = Path(row['filepath'])
            
            # Try new unified format first (e.g., m1337_unified.obj)
            unified_filepath = original_filepath.parent / f"{original_filepath.stem}_unified.obj"
            unified_meta_filepath = original_filepath.parent / f"{original_filepath.stem}_metadata.json"
            
            if unified_filepath.exists() and unified_meta_filepath.exists():
                print(f"  📦 Loading unified normalized version: {unified_filepath.name}")
                mesh = ShapeMesh(str(unified_filepath))
                if mesh.vertices is not None and len(mesh.vertices) > 0:
                    # Load metadata if available
                    try:
                        import json
                        with open(unified_meta_filepath, 'r') as f:
                            metadata = json.load(f)
                            mesh._normalization_metadata = metadata
                    except Exception as e:
                        print(f"  ⚠️ Could not load metadata: {e}")
                    return mesh
            
            # Try old normalized format (e.g., m1337_normalized.obj)
            normalized_filepath = original_filepath.parent / f"{original_filepath.stem}_normalized.obj"
            if normalized_filepath.exists():
                print(f"  📦 Loading legacy normalized version: {normalized_filepath.name}")
                mesh = ShapeMesh(str(normalized_filepath))
                if mesh.vertices is not None and len(mesh.vertices) > 0:
                    return mesh
            
            return None
            
        except Exception as e:
            print(f"  ⚠️ Error loading normalized version: {e}")
            return None
    
    def process_dataset(self, dataset_name, use_normalized=False):
        """
        Process all shapes in a dataset for feature extraction
        
        Args:
            dataset_name: Name of the dataset to process
            use_normalized: If True, prefer normalized shapes when available
            
        Returns:
            list: List of dictionaries with shape info and features
        """
        print(f"\nProcessing dataset: {dataset_name}")
        print("=" * 60)
        print(f"Feature Extraction Pipeline: Load → Extract Features → Export")
        if use_normalized:
            print("Using normalized shapes when available")
        print("=" * 60)
        
        # Get file list for dataset
        print(f"🔍 Loading dataset files from: {dataset_name}")
        print(f"   Current working directory: {Path.cwd()}")
        file_df = get_file_tree(data_dir=dataset_name)
        print(f"   Found {len(file_df)} files in dataset")
        
        if len(file_df) == 0:
            print(f"No files found for dataset {dataset_name}")
            return []
        
        print(f"Found {len(file_df)} shapes across {file_df['category'].nunique()} categories")
        
        results = []
        
        # Process each shape
        for idx, row in tqdm(file_df.iterrows(), total=len(file_df), desc=f"Extracting features from {dataset_name}"):
            result = self.process_shape(row, dataset_name, use_normalized=use_normalized)
            
            if result is not None:
                results.append(result)
            
            # Progress logging every 50 shapes
            if (idx + 1) % 50 == 0:
                success_rate = self.stats['successful'] / max(self.stats['total_processed'], 1) * 100
                print(f"\nProgress: {idx+1}/{len(file_df)} ({success_rate:.1f}% success rate)")
        
        return results
    
    def export_to_csv(self, results, dataset_name):
        """
        Export feature extraction results to CSV
        
        Args:
            results: List of dictionaries with shape info and features
            dataset_name: Name of the dataset
            
        Returns:
            Path: Path to the exported CSV file
        """
        if not results:
            print(f"⚠️  No results to export for {dataset_name}")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Sort by category and filename for better organization
        df = df.sort_values(['category', 'filename']).reset_index(drop=True)
        
        # Define column order for better readability
        basic_cols = ['filename', 'category', 'dataset', 'normalized_version_used']
        
        # Build feature columns based on what's enabled
        shape_info_cols = []
        if self.enable_basic_features:
            shape_info_cols = ['num_vertices', 'num_faces', 'volume', 'file_size_bytes']
            
        bbox_cols = []
        if self.enable_basic_features:
            bbox_cols = ['bbox_width', 'bbox_height', 'bbox_depth', 'bbox_volume']
            
        feature_cols = self.enabled_extraction_features  # Only enabled features
        
        derived_cols = []
        if self.enable_derived_features:
            derived_cols = ['aspect_ratio_xy', 'aspect_ratio_xz', 'aspect_ratio_yz', 'sphericity', 'compactness_normalized']
            
        meta_cols = ['filepath']
        
        # Reorder columns
        column_order = basic_cols + shape_info_cols + bbox_cols + feature_cols + derived_cols + meta_cols
        
        # Only include columns that exist in the DataFrame
        column_order = [col for col in column_order if col in df.columns]
        
        # Add any remaining columns
        remaining_cols = [col for col in df.columns if col not in column_order]
        column_order.extend(remaining_cols)
        
        df = df[column_order]
        
        # Create safe filename from dataset name (handle subdirectories)
        safe_dataset_name = dataset_name.replace('/', '_').replace('\\', '_').lower()
        csv_path = self.output_base_dir / f"features_{safe_dataset_name}.csv"
        
        # Ensure output directory exists before saving CSV
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export to CSV
        df.to_csv(csv_path, index=False, float_format='%.6f')
        
        print(f"✅ Features exported to: {csv_path}")
        print(f"   Dataset: {dataset_name}")
        print(f"   Shapes: {len(df)}")
        print(f"   Categories: {df['category'].nunique()}")
        print(f"   Features: {len(feature_cols + derived_cols)}")
        
        return csv_path
    
    def save_feature_extraction_report(self, datasets_processed):
        """Save comprehensive feature extraction report"""
        # Generate timestamp for unique filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = self.output_base_dir / f"feature_extraction_report_{timestamp}.json"
        
        processing_time = time.time() - self.stats['start_time']
        
        # Calculate feature statistics
        feature_stats = {}
        for feature_name, range_info in self.stats['feature_stats']['feature_ranges'].items():
            if range_info['values']:
                feature_stats[feature_name] = {
                    'min': float(range_info['min']),
                    'max': float(range_info['max']),
                    'mean': float(np.mean(range_info['values'])),
                    'std': float(np.std(range_info['values'])),
                    'median': float(np.median(range_info['values']))
                }
        
        report = {
            'processing_summary': {
                'datasets_processed': datasets_processed,
                'total_shapes': self.stats['total_processed'],
                'successful': self.stats['successful'],
                'failed': self.stats['failed'],
                'success_rate': self.stats['successful'] / max(self.stats['total_processed'], 1) * 100,
                'processing_time_seconds': processing_time,
                'shapes_per_second': self.stats['total_processed'] / max(processing_time, 1)
            },
            'features_extracted': self.enabled_extraction_features + (['basic_features'] if self.enable_basic_features else []) + (['derived_features'] if self.enable_derived_features else []),
            'feature_statistics': feature_stats,
            'by_category': self.stats['feature_stats']['by_category'],
            'errors': self.stats['errors'],
            'timestamp': timestamp,
            'report_filename': f"feature_extraction_report_{timestamp}.json"
        }
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"✅ JSON report saved successfully: {report_path}")
        except Exception as e:
            print(f"❌ Failed to save JSON report: {e}")
            
        return report
    
    def print_summary(self, report, csv_files):
        """Print comprehensive feature extraction summary"""
        print("\n" + "=" * 80)
        print("FEATURE EXTRACTION COMPLETE")
        print("=" * 80)
        
        summary = report['processing_summary']
        
        print(f"📊 Processing Summary:")
        print(f"   Datasets processed: {len(summary['datasets_processed'])}")
        print(f"   Total shapes processed: {summary['total_shapes']}")
        print(f"   Successful: {summary['successful']}")
        print(f"   Failed: {summary['failed']}")
        print(f"   Success rate: {summary['success_rate']:.1f}%")
        print(f"   Processing time: {summary['processing_time_seconds']:.1f}s")
        print(f"   Speed: {summary['shapes_per_second']:.1f} shapes/second")
        
        print(f"\n🔍 Features Extracted:")
        features = report['features_extracted']
        for i, feature in enumerate(features, 1):
            print(f"   {i:2d}. {feature}")
        
        print(f"\n📄 Generated CSV Files:")
        for csv_file in csv_files:
            print(f"   📁 {csv_file}")
        
        print(f"\n📁 Output Directory: {self.output_base_dir}")
        timestamp = report.get('timestamp', 'unknown')
        print(f"🕒 Report generated at: {timestamp}")
        print(f"📄 Detailed Report: {self.output_base_dir / f'feature_extraction_report_{timestamp}.json'}")
        
        if len(self.stats['errors']) > 0:
            print(f"\n⚠️  {len(self.stats['errors'])} errors occurred (see report for details)")


def main():
    """Main feature extraction function"""
    print("🔍 Starting Feature Extraction for 3D Shapes")
    
    # ===== EASY CONFIGURATION SECTION =====
    # Modify these settings to control which features are extracted
    
    # Dataset configuration
    datasets = ["Data", "UnifiedPreprocessed/Data"]  # Adjust as needed
    use_normalized = True  # Set to True to prefer normalized shapes when available
    
    # Feature selection - choose what to extract:
    # Option 1: Extract all features (default)
    # enabled_features = None
    
    # Option 2: Extract only specific features (uncomment and modify as needed)
    # enabled_features = ['surface_area', 'diameter', 'basic_features']
    
    # Option 3: Extract only basic shape info (fast)
    enabled_features = ['basic_features']
    
    # Option 4: Extract only geometric features (no basic info)
    # enabled_features = ['surface_area', 'compactness', 'diameter']
    
    # Option 5: Extract everything except problematic features
    # enabled_features = ['surface_area', 'diameter', 'basic_features', 'derived_features']
    
    # Available features:
    # Extraction features: 'surface_area', 'compactness', 'rectangularity', 'diameter', 'convexity', 'eccentricity'
    # Other features: 'basic_features' (vertices, faces, volume, bbox), 'derived_features' (aspect ratios, sphericity)
    
    # ===== END CONFIGURATION SECTION =====
    
    print("Features to extract:")
    if enabled_features is None:
        print("  • All available features")
    else:
        print(f"  • Selected features: {enabled_features}")
    
    # Initialize processor
    processor = FeatureExtractionProcessor(enabled_features=enabled_features)
    processor.stats['start_time'] = time.time()
    
    # Setup directories
    processor.setup_output_directories(datasets)
    
    csv_files = []
    
    # Process each dataset
    for dataset in datasets:
        try:
            print(f"\n🔄 Processing dataset: {dataset}")
            
            # Extract features from all shapes
            results = processor.process_dataset(dataset, use_normalized=use_normalized)
            
            # Export to CSV
            if results:
                csv_file = processor.export_to_csv(results, dataset)
                if csv_file:
                    csv_files.append(csv_file)
            else:
                print(f"⚠️  No valid results for dataset {dataset}")
                
        except Exception as e:
            print(f"❌ Failed to process dataset {dataset}: {str(e)}")
            traceback.print_exc()
    
    # Generate and save processing report
    report = processor.save_feature_extraction_report(datasets)
    processor.print_summary(report, csv_files)


if __name__ == "__main__":
    main()