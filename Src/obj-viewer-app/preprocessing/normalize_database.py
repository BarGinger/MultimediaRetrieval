"""
Step 3.1: Preprocessing - Normalize all shapes in the database and save them.
This creates normalized .obj files for efficient reuse during feature extraction and visualization.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import time

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.file_index import get_file_tree
from core.analysis_cache import merge_analysis_data
from core.shapeMesh import ShapeMesh

class NormalizationProcessor: 
    def __init__(self, output_base_dir=None):
        """
        Processor to normalize shapes and save them with metadata.

        Parameters:
            output_base_dir (str or Path): Base directory to save normalized shapes and metadata.
                        Defaults to "Datasets/NormalizedShapes" in the current or parent directories.
                    
        """

        # Use the same path resolution pattern as file_index.py
        if output_base_dir is None:
            cwd = Path.cwd()
            dataset_path = "Datasets/NormalizedShapes"
            candidates = [cwd / dataset_path, cwd.parent / dataset_path, cwd.parent.parent / dataset_path]
            self.output_base_dir = next((p for p in candidates if p.parent.exists()), candidates[-1])
        else:
            self.output_base_dir = Path(output_base_dir)
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'start_time': None,
            'errors': [],
            'normalization_stats': {
                'centering_errors': [],
                'scaling_errors': [],
                'by_category': {}
            }
        }
    
    def setup_output_directories(self, datasets):
        """
        Create normalized shape directories for each dataset
        
        Parameters:
            datasets (list of str): List of dataset names to create directories for.

        Returns:
            None
        """
        print(f"📁 Setting up output directories in: {self.output_base_dir.absolute()}")
        for dataset in datasets:
            dataset_dir = self.output_base_dir / dataset
            dataset_dir.mkdir(parents=True, exist_ok=True)
            print(f"   Created: {dataset_dir}")
    
    def save_normalized_obj(self, mesh, output_path):
        """
        Save normalized mesh as OBJ file

        Parameters:
            mesh (ShapeMesh): The ShapeMesh object to normalize and save.
            output_path (str or Path): Path to save the normalized OBJ file.

        Returns:
            None
        """
        normalized_vertices = mesh.apply_full_normalization()
        
        with open(output_path, 'w') as f:
            # Write header
            f.write(f"# Normalized OBJ file generated from {mesh.filename}\n")
            f.write(f"# Applied 4-step normalization: centering, PCA alignment, flipping, scaling\n")
            f.write(f"# Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write vertices
            for vertex in normalized_vertices:
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            
            # Write faces (convert to 1-indexed for OBJ format)
            for face in mesh.faces:
                if len(face) >= 3:
                    # OBJ uses 1-indexed vertices
                    face_indices = [str(idx + 1) for idx in face]
                    f.write(f"f {' '.join(face_indices)}\n")
    
    def save_normalization_metadata(self, mesh, output_path):
        """
        Save normalization metadata as JSON
        
        Parameters:
            mesh (ShapeMesh): The ShapeMesh object to extract metadata from.
            output_path (str or Path): Path to save the normalization metadata JSON file.
        Returns:
            None
        """
        norm_info = mesh.get_normalization_info()
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        metadata = {
            'original_filename': mesh.filename,
            'category': mesh.category,
            'normalization_info': convert_numpy(norm_info),
            'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'vertices_count': len(mesh.vertices),
            'faces_count': len(mesh.faces)
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def process_dataset(self, dataset_name):
        """
        Process all shapes in a dataset

        Parameters:
            dataset_name (str): The name of the dataset to process.
        
        Returns:
            None
        """
        print(f"\nProcessing dataset: {dataset_name}")
        print("=" * 50)
        
        # Get file list for dataset - use the specific data directory
        file_df = get_file_tree(data_dir=dataset_name)
        # Don't merge analysis data since we're working with raw file structure
        
        if len(file_df) == 0:
            print(f"No files found for dataset {dataset_name}")
            return
        
        print(f"Found {len(file_df)} shapes across {file_df['category'].nunique()} categories")
        
        # Process each shape
        for idx, row in tqdm(file_df.iterrows(), total=len(file_df), desc=f"Normalizing {dataset_name}"):
            try:
                # Create mesh
                mesh = ShapeMesh.from_file_row(row)
                
                # Preserve category folder structure
                category = row.get('category', 'Unknown')
                category_dir = self.output_base_dir / dataset_name / category
                category_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate output paths within category folder
                base_name = Path(mesh.filename).stem
                normalized_obj_path = category_dir / f"{base_name}_normalized.obj"
                metadata_path = category_dir / f"{base_name}_metadata.json"
                
                # Save normalized OBJ
                self.save_normalized_obj(mesh, normalized_obj_path)
                
                # Save metadata
                self.save_normalization_metadata(mesh, metadata_path)
                
                # Update statistics
                norm_info = mesh.get_normalization_info()
                center_error = np.linalg.norm(norm_info['final']['center'])
                scale_error = abs(norm_info['final']['max_dimension'] - 1.0)
                
                self.stats['normalization_stats']['centering_errors'].append(center_error)
                self.stats['normalization_stats']['scaling_errors'].append(scale_error)
                
                # Track by category
                category = row.get('category', 'Unknown')
                if category not in self.stats['normalization_stats']['by_category']:
                    self.stats['normalization_stats']['by_category'][category] = {
                        'count': 0, 'successful': 0
                    }
                
                self.stats['normalization_stats']['by_category'][category]['count'] += 1
                
                if center_error < 1e-10 and scale_error < 1e-6:
                    self.stats['normalization_stats']['by_category'][category]['successful'] += 1
                    self.stats['successful'] += 1
                else:
                    self.stats['failed'] += 1
                
                self.stats['total_processed'] += 1
                
                # Progress info will be shown via tqdm automatically
                
            except Exception as e:
                import traceback
                error_msg = f"Error processing {row.get('filename', 'unknown')}: {str(e)}"
                print(f"\nDETAILED ERROR: {error_msg}")
                print(f"Traceback:")
                traceback.print_exc()
                self.stats['errors'].append(error_msg)
                self.stats['failed'] += 1
                self.stats['total_processed'] += 1
                
                # Continue processing other files even if one fails
                continue
    
    def save_processing_report(self):
        """
        Save comprehensive processing report
        """ 
        report_path = self.output_base_dir / "normalization_report.json"
        
        # Calculate statistics
        processing_time = time.time() - self.stats['start_time']
        
        report = {
            'processing_summary': {
                'total_shapes': self.stats['total_processed'],
                'successful': self.stats['successful'],
                'failed': self.stats['failed'],
                'success_rate': self.stats['successful'] / max(self.stats['total_processed'], 1) * 100,
                'processing_time_seconds': processing_time,
                'shapes_per_second': self.stats['total_processed'] / max(processing_time, 1)
            },
            'normalization_quality': {
                'mean_centering_error': np.mean(self.stats['normalization_stats']['centering_errors']) if self.stats['normalization_stats']['centering_errors'] else 0,
                'max_centering_error': np.max(self.stats['normalization_stats']['centering_errors']) if self.stats['normalization_stats']['centering_errors'] else 0,
                'mean_scaling_error': np.mean(self.stats['normalization_stats']['scaling_errors']) if self.stats['normalization_stats']['scaling_errors'] else 0,
                'max_scaling_error': np.max(self.stats['normalization_stats']['scaling_errors']) if self.stats['normalization_stats']['scaling_errors'] else 0
            },
            'by_category': self.stats['normalization_stats']['by_category'],
            'errors': self.stats['errors']
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def print_summary(self, report):
        """
        Print processing summary
        
        Parameters:
            report (dict): The report dictionary generated by save_processing_report().

        Returns:
            None
        """
        print("\n" + "=" * 60)
        print("NORMALIZATION PREPROCESSING COMPLETE")
        print("=" * 60)
        
        summary = report['processing_summary']
        quality = report['normalization_quality']
        
        print(f"📊 Processing Summary:")
        print(f"   Total shapes processed: {summary['total_shapes']}")
        print(f"   Successful: {summary['successful']}")
        print(f"   Failed: {summary['failed']}")
        print(f"   Success rate: {summary['success_rate']:.1f}%")
        print(f"   Processing time: {summary['processing_time_seconds']:.1f}s")
        print(f"   Speed: {summary['shapes_per_second']:.1f} shapes/second")
        
        print(f"\n🎯 Normalization Quality:")
        print(f"   Mean centering error: {quality['mean_centering_error']:.2e}")
        print(f"   Max centering error: {quality['max_centering_error']:.2e}")
        print(f"   Mean scaling error: {quality['mean_scaling_error']:.2e}")
        print(f"   Max scaling error: {quality['max_scaling_error']:.2e}")
        
        print(f"\n📁 Output Directory: {self.output_base_dir}")
        print(f"📄 Detailed Report: {self.output_base_dir / 'normalization_report.json'}")
        
        if len(self.stats['errors']) > 0:
            print(f"\n⚠️  {len(self.stats['errors'])} errors occurred (see report for details)")

def main():
    """Main preprocessing function"""
    print("🚀 Starting Shape Normalization Preprocessing")
    print("This will normalize all shapes from Data_sampled_resampled_normalized and save them for efficient reuse.")
    
    # Use the preprocessed dataset with all categories
    datasets = ["Data_sampled_resampled_normalized"]  # The final preprocessed dataset
    
    # Initialize processor
    processor = NormalizationProcessor()
    processor.stats['start_time'] = time.time()
    
    # Setup directories
    processor.setup_output_directories(datasets)
    
    # Process each dataset
    for dataset in datasets:
        try:
            processor.process_dataset(dataset)
        except Exception as e:
            print(f"❌ Failed to process dataset {dataset}: {str(e)}")

    # Generate and save report
    report = processor.save_processing_report()
    processor.print_summary(report)

if __name__ == "__main__":
    main()