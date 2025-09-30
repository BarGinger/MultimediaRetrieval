"""
Unified Preprocessing & Normalization Script
Combines remeshing (Step 2) with your existing complete 4-step normalization (Step 3.1)
Order: Remeshing → Translation → PCA → Flipping → Scaling
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import time
import open3d as o3d
import csv

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.file_index import get_file_tree
from core.analysis_cache import merge_analysis_data
from core.shapeMesh import ShapeMesh

class UnifiedPreprocessingProcessor:
    def __init__(self, target_faces=5000, output_base_dir=None):
        """
        Unified processor combining remeshing + your existing complete normalization
        
        Parameters:
            target_faces (int): Target number of faces for remeshing
            output_base_dir (str or Path): Output directory for processed shapes
        """
        self.target_faces = target_faces
        
        # Use same path resolution as your normalize_database.py
        if output_base_dir is None:
            cwd = Path.cwd()
            dataset_path = "Datasets/UnifiedPreprocessed"
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
            'remeshing_stats': [],
            'normalization_stats': {
                'centering_errors': [],
                'scaling_errors': [],
                'by_category': {}
            }
        }
    
    def setup_output_directories(self, datasets):
        """Create output directories for each dataset"""
        print(f"📁 Setting up output directories in: {self.output_base_dir.absolute()}")
        for dataset in datasets:
            dataset_dir = self.output_base_dir / dataset
            dataset_dir.mkdir(parents=True, exist_ok=True)
            print(f"   Created: {dataset_dir}")
    
    def apply_remeshing_if_needed(self, mesh_path, target_faces, tolerance=0.2):
        """
        Apply remeshing if face count significantly differs from target
        
        Parameters:
            mesh_path (Path): Path to the OBJ file
            target_faces (int): Target number of faces
            tolerance (float): Tolerance for face count difference (0.2 = 20%)
            
        Returns:
            tuple: (vertices, faces, remeshed_flag)
        """
        try:
            # Load with Open3D for remeshing
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            
            if len(mesh.vertices) == 0:
                print(f"❌ Empty mesh: {mesh_path}")
                return None, None, False
            
            current_faces = len(mesh.triangles)
            
            # Check if remeshing is needed
            face_diff = abs(current_faces - target_faces)
            needs_remeshing = face_diff > target_faces * tolerance
            
            if needs_remeshing:
                print(f"  🔄 Remeshing from {current_faces} to ~{target_faces} faces")
                
                if current_faces > target_faces:
                    # Simplify mesh
                    mesh = mesh.simplify_quadric_decimation(target_faces)
                elif current_faces < target_faces * 0.5:
                    # Subdivide if too few faces
                    mesh = mesh.subdivide_midpoint(number_of_iterations=1)
                
                # Clean up mesh
                mesh.remove_degenerate_triangles()
                mesh.remove_duplicated_triangles()
                mesh.remove_duplicated_vertices()
                mesh.remove_non_manifold_edges()
                
                final_faces = len(mesh.triangles)
                print(f"  ✅ Remeshing result: {len(mesh.vertices)} vertices, {final_faces} faces")
                
                # Collect remeshing stats
                self.stats['remeshing_stats'].append({
                    'original_faces': current_faces,
                    'target_faces': target_faces,
                    'final_faces': final_faces,
                    'reduction_ratio': final_faces / current_faces if current_faces > 0 else 1.0
                })
                
                return np.asarray(mesh.vertices), np.asarray(mesh.triangles), True
            else:
                print(f"  ✅ No remeshing needed ({current_faces} faces within tolerance)")
                return np.asarray(mesh.vertices), np.asarray(mesh.triangles), False
                
        except Exception as e:
            print(f"❌ Remeshing failed for {mesh_path}: {e}")
            return None, None, False
    
    def process_shape(self, row, dataset_name):
        """
        Process a single shape through the complete unified pipeline:
        1. Load original mesh
        2. Apply remeshing if needed
        3. Apply your existing complete 4-step normalization
        4. Save results with metadata
        """
        try:
            original_filepath = Path(row['filepath'])
            
            # Smart skip logic: check if shape is already processed
            category = row.get('category', 'Unknown')
            category_dir = self.output_base_dir / dataset_name / category
            base_name = Path(row.get('filename', original_filepath.name)).stem
            normalized_obj_path = category_dir / f"{base_name}_unified.obj"
            metadata_path = category_dir / f"{base_name}_metadata.json"
            
            if normalized_obj_path.exists() and metadata_path.exists():
                # Check if output is newer than input
                output_mtime = min(normalized_obj_path.stat().st_mtime, metadata_path.stat().st_mtime)
                input_mtime = original_filepath.stat().st_mtime
                
                if output_mtime > input_mtime:
                    print(f"  ✅ Already processed: {original_filepath.name}")
                    self.stats['successful'] += 1
                    self.stats['total_processed'] += 1
                    return True
                else:
                    print(f"  🔄 Input file newer than output, reprocessing: {original_filepath.name}")
            
            # Step 1 & 2: Load and remesh if needed
            vertices, faces, was_remeshed = self.apply_remeshing_if_needed(
                original_filepath, self.target_faces
            )
            
            if vertices is None:
                print(f"❌ Failed to load/remesh: {original_filepath.name}")
                return False
            
            # Step 3: Create ShapeMesh with potentially remeshed data
            mesh = ShapeMesh(
                vertices=vertices,
                faces=faces,
                category=row.get('category'),
                filename=row.get('filename'),
                filepath=str(original_filepath),
                size=row.get('size')
            )
            
            # Step 4: Apply your existing complete 4-step normalization
            print(f"  🔧 Applying 4-step normalization...")
            normalized_vertices = mesh.apply_full_normalization(debug=False)
            
            # Step 5: Save results
            category_dir.mkdir(parents=True, exist_ok=True)
            
            # Save normalized OBJ (reuse your existing save logic)
            self.save_normalized_obj(mesh, normalized_vertices, normalized_obj_path, was_remeshed)
            
            # Save metadata (enhanced with remeshing info)
            self.save_enhanced_metadata(mesh, metadata_path, was_remeshed, len(vertices), len(faces))
            
            # Update statistics (reuse your existing logic)
            norm_info = mesh.get_normalization_info()
            center_error = np.linalg.norm(norm_info['final']['center'])
            scale_error = abs(norm_info['final']['max_dimension'] - 1.0)
            
            self.stats['normalization_stats']['centering_errors'].append(center_error)
            self.stats['normalization_stats']['scaling_errors'].append(scale_error)
            
            # Track by category (your existing logic)
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
            return True
            
        except Exception as e:
            import traceback
            error_msg = f"Error processing {row.get('filename', 'unknown')}: {str(e)}"
            print(f"\n❌ {error_msg}")
            traceback.print_exc()
            self.stats['errors'].append(error_msg)
            self.stats['failed'] += 1
            self.stats['total_processed'] += 1
            return False
    
    def save_normalized_obj(self, mesh, normalized_vertices, output_path, was_remeshed):
        """Save normalized mesh as OBJ file (enhanced version of your method)"""
        with open(output_path, 'w') as f:
            # Enhanced header
            f.write(f"# Unified Preprocessed OBJ file from {mesh.filename}\n")
            f.write(f"# Applied remeshing: {'Yes' if was_remeshed else 'No'}\n")
            f.write(f"# Applied 4-step normalization: centering, PCA alignment, flipping, scaling\n")
            f.write(f"# Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write vertices
            for vertex in normalized_vertices:
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            
            # Write faces (convert to 1-indexed for OBJ format)
            for face in mesh.faces:
                if len(face) >= 3:
                    face_indices = [str(idx + 1) for idx in face]
                    f.write(f"f {' '.join(face_indices)}\n")
    
    def save_enhanced_metadata(self, mesh, output_path, was_remeshed, final_vertices, final_faces):
        """Save enhanced metadata including remeshing info"""
        norm_info = mesh.get_normalization_info()
        
        # Convert numpy arrays to lists (your existing logic)
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
            'processing_info': {
                'remeshing_applied': was_remeshed,
                'target_faces': self.target_faces,
                'final_vertices_count': final_vertices,
                'final_faces_count': final_faces
            },
            'normalization_info': convert_numpy(norm_info),
            'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'pipeline_version': 'unified_v1.0'
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def process_dataset(self, dataset_name):
        """Process all shapes in a dataset (enhanced version of your method)"""
        print(f"\nProcessing dataset: {dataset_name}")
        print("=" * 60)
        print("Pipeline: Remeshing → Translation → PCA → Flipping → Scaling")
        print("=" * 60)
        
        # Get file list for dataset (your existing logic)
        file_df = get_file_tree(data_dir=dataset_name)
        
        if len(file_df) == 0:
            print(f"No files found for dataset {dataset_name}")
            return
        
        print(f"Found {len(file_df)} shapes across {file_df['category'].nunique()} categories")
        
        # Process each shape (your existing loop structure)
        for idx, row in tqdm(file_df.iterrows(), total=len(file_df), desc=f"Processing {dataset_name}"):
            success = self.process_shape(row, dataset_name)
            
            # Progress logging every 50 shapes
            if (idx + 1) % 50 == 0:
                success_rate = self.stats['successful'] / max(self.stats['total_processed'], 1) * 100
                print(f"\nProgress: {idx+1}/{len(file_df)} ({success_rate:.1f}% success rate)")
    
    def save_processing_report(self):
        """Save comprehensive processing report (enhanced version)"""
        report_path = self.output_base_dir / "unified_processing_report.json"
        
        processing_time = time.time() - self.stats['start_time']
        
        # Enhanced report with remeshing stats
        report = {
            'processing_summary': {
                'total_shapes': self.stats['total_processed'],
                'successful': self.stats['successful'],
                'failed': self.stats['failed'],
                'success_rate': self.stats['successful'] / max(self.stats['total_processed'], 1) * 100,
                'processing_time_seconds': processing_time,
                'shapes_per_second': self.stats['total_processed'] / max(processing_time, 1)
            },
            'remeshing_summary': {
                'shapes_remeshed': len(self.stats['remeshing_stats']),
                'target_faces': self.target_faces,
                'avg_reduction_ratio': np.mean([s['reduction_ratio'] for s in self.stats['remeshing_stats']]) if self.stats['remeshing_stats'] else 1.0
            },
            'normalization_quality': {
                'mean_centering_error': np.mean(self.stats['normalization_stats']['centering_errors']) if self.stats['normalization_stats']['centering_errors'] else 0,
                'max_centering_error': np.max(self.stats['normalization_stats']['centering_errors']) if self.stats['normalization_stats']['centering_errors'] else 0,
                'mean_scaling_error': np.mean(self.stats['normalization_stats']['scaling_errors']) if self.stats['normalization_stats']['scaling_errors'] else 0,
                'max_scaling_error': np.max(self.stats['normalization_stats']['scaling_errors']) if self.stats['normalization_stats']['scaling_errors'] else 0
            },
            'by_category': self.stats['normalization_stats']['by_category'],
            'errors': self.stats['errors'],
            'remeshing_details': self.stats['remeshing_stats']
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def print_summary(self, report):
        """Print comprehensive processing summary"""
        print("\n" + "=" * 80)
        print("UNIFIED PREPROCESSING & NORMALIZATION COMPLETE")
        print("=" * 80)
        
        summary = report['processing_summary']
        remesh_summary = report['remeshing_summary']
        quality = report['normalization_quality']
        
        print(f"📊 Processing Summary:")
        print(f"   Total shapes processed: {summary['total_shapes']}")
        print(f"   Successful: {summary['successful']}")
        print(f"   Failed: {summary['failed']}")
        print(f"   Success rate: {summary['success_rate']:.1f}%")
        print(f"   Processing time: {summary['processing_time_seconds']:.1f}s")
        print(f"   Speed: {summary['shapes_per_second']:.1f} shapes/second")
        
        print(f"\n🔄 Remeshing Summary:")
        print(f"   Shapes remeshed: {remesh_summary['shapes_remeshed']}")
        print(f"   Target faces: {remesh_summary['target_faces']}")
        print(f"   Avg reduction ratio: {remesh_summary['avg_reduction_ratio']:.3f}")
        
        print(f"\n🎯 Normalization Quality (using your existing verification):")
        print(f"   Mean centering error: {quality['mean_centering_error']:.2e}")
        print(f"   Max centering error: {quality['max_centering_error']:.2e}")
        print(f"   Mean scaling error: {quality['mean_scaling_error']:.2e}")
        print(f"   Max scaling error: {quality['max_scaling_error']:.2e}")
        
        # Check compliance with technical tips
        centered_ok = quality['max_centering_error'] < 1e-10
        scaled_ok = quality['max_scaling_error'] < 1e-6
        
        if centered_ok and scaled_ok:
            print(f"\n🎉 FULL TECHNICAL TIPS COMPLIANCE ACHIEVED!")
        else:
            print(f"\n⚠️  Some shapes may not meet technical tips precision requirements")
        
        print(f"\n📁 Output Directory: {self.output_base_dir}")
        print(f"📄 Detailed Report: {self.output_base_dir / 'unified_processing_report.json'}")
        
        if len(self.stats['errors']) > 0:
            print(f"\n⚠️  {len(self.stats['errors'])} errors occurred (see report for details)")

    def analyze_shape(self, shape_path):
        """
        Analyze a single OBJ shape file for vertex/face counts and properties
        Based on analyzer_tool.py pattern
        """
        try:
            with open(shape_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading {shape_path}: {e}")
            return None

        vertices = []
        faces = []
        face_types = set()

        for line in lines:
            line = line.strip()
            if line.startswith('v '):
                vertices.append(line)
            elif line.startswith('f '):
                face_parts = line.split()[1:]  # Skip 'f'
                faces.append(face_parts)
                face_types.add(len(face_parts))

        num_vertices = len(vertices)
        num_faces = len(faces)

        # Calculate bounding box if vertices exist
        if vertices:
            coords = []
            for v_line in vertices:
                parts = v_line.split()[1:]  # Skip 'v'
                coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
            coords = np.array(coords)
            
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)
            bbox = {
                'min': min_coords.tolist(),
                'max': max_coords.tolist(),
                'size': (max_coords - min_coords).tolist()
            }
        else:
            bbox = None

        return {
            'num_vertices': num_vertices,
            'num_faces': num_faces,
            'face_types': list(face_types),
            'bounding_box': bbox
        }

    def analyze_processed_dataset(self, dataset_name):
        """
        Generate analysis CSV for a processed dataset
        Based on analyzer_tool.py pattern with smart skip logic
        """
        dataset_path = self.output_base_dir / dataset_name
        csv_file = self.output_base_dir / f"analysis_results_{dataset_name.lower()}.csv"
        
        # Smart skip logic: check if CSV exists and is recent
        if csv_file.exists():
            csv_mtime = csv_file.stat().st_mtime
            dataset_mtime = max(
                (p.stat().st_mtime for p in dataset_path.rglob("*.obj")),
                default=0
            )
            
            if csv_mtime > dataset_mtime:
                print(f"📊 Analysis CSV for {dataset_name} is up to date, skipping generation")
                return csv_file
            else:
                print(f"📊 Dataset {dataset_name} has newer files, regenerating analysis CSV")
        else:
            print(f"📊 Generating analysis CSV for {dataset_name}")
        
        results = []
        total_shapes = 0
        
        # Count total shapes first for progress bar
        for class_folder in dataset_path.iterdir():
            if class_folder.is_dir():
                total_shapes += len(list(class_folder.glob("*.obj")))
        
        if total_shapes == 0:
            print(f"No OBJ files found in {dataset_path}")
            return None
        
        # Analyze each shape
        with tqdm(total=total_shapes, desc=f"Analyzing {dataset_name}") as pbar:
            for class_folder in dataset_path.iterdir():
                if not class_folder.is_dir():
                    continue
                
                class_name = class_folder.name
                for shape_file in class_folder.glob("*.obj"):
                    analysis = self.analyze_shape(shape_file)
                    if analysis:
                        results.append({
                            'class': class_name,
                            'shape_file': shape_file.name,
                            **analysis
                        })
                    pbar.update(1)
        
        # Save results to CSV
        fieldnames = ['class', 'shape_file', 'num_vertices', 'num_faces', 'face_types', 'bounding_box']
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                # Convert face_types and bounding_box to strings for CSV
                row = result.copy()
                row['face_types'] = ','.join(map(str, row['face_types']))
                row['bounding_box'] = json.dumps(row['bounding_box'])
                writer.writerow(row)
        
        print(f"✅ Analysis CSV saved: {csv_file}")
        print(f"   Analyzed {len(results)} shapes across {len(set(r['class'] for r in results))} classes")
        
        return csv_file

    def generate_analysis_for_all_datasets(self):
        """
        Generate analysis CSV files for all processed datasets
        """
        print("\n" + "=" * 60)
        print("GENERATING ANALYSIS CSV FILES")
        print("=" * 60)
        
        generated_csvs = []
        
        # Find all processed datasets
        if self.output_base_dir.exists():
            for dataset_dir in self.output_base_dir.iterdir():
                if dataset_dir.is_dir() and dataset_dir.name != '__pycache__':
                    csv_file = self.analyze_processed_dataset(dataset_dir.name)
                    if csv_file:
                        generated_csvs.append(csv_file)
        
        if generated_csvs:
            print(f"\n✅ Generated {len(generated_csvs)} analysis CSV files:")
            for csv_file in generated_csvs:
                print(f"   📄 {csv_file}")
        else:
            print("\n⚠️  No datasets found to analyze")
        
        return generated_csvs

def main():
    """Main unified preprocessing function"""
    print("🚀 Starting Unified Preprocessing & Normalization")
    print("Pipeline: Remeshing → Your Complete 4-Step Normalization")
    print("Leveraging your existing robust ShapeMesh implementation")
    
    # Use your preferred dataset
    datasets = ["Data"]  # Adjust as needed
    
    # Initialize processor with remeshing target
    processor = UnifiedPreprocessingProcessor(target_faces=5000)
    processor.stats['start_time'] = time.time()
    
    # Setup directories
    processor.setup_output_directories(datasets)
    
    # Check if processing is needed (smart skip logic)
    skip_processing = True
    for dataset in datasets:
        dataset_output_dir = processor.output_base_dir / dataset
        if not dataset_output_dir.exists() or len(list(dataset_output_dir.rglob("*.obj"))) == 0:
            skip_processing = False
            break
    
    if skip_processing:
        print("\n📋 All datasets appear to be already processed, skipping preprocessing and normalization")
        print("   To force reprocessing, delete the UnifiedPreprocessed folder")
    else:
        # Process each dataset
        for dataset in datasets:
            try:
                processor.process_dataset(dataset)
            except Exception as e:
                print(f"❌ Failed to process dataset {dataset}: {str(e)}")
        
        # Generate and save processing report
        report = processor.save_processing_report()
        processor.print_summary(report)
    
    # Always generate analysis CSV files (with their own smart skip logic)
    processor.generate_analysis_for_all_datasets()

if __name__ == "__main__":
    main()