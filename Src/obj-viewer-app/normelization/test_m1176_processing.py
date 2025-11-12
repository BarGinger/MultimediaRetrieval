"""
Direct test of m1176 processing with the fixed normalization code.
Tests the actual UnifiedPreprocessingProcessor with minimal cleaning and dynamic component validation.
"""

import sys
import open3d as o3d
from pathlib import Path
import numpy as np

# Import the processor
from normalize_database import UnifiedPreprocessingProcessor

def test_m1176_with_processor():
    """Test m1176 using the actual UnifiedPreprocessingProcessor"""
    
    mesh_path = Path(r"C:\Users\bar24\OneDrive - Universiteit Utrecht\Documents\School\UU Data Sceince MSc\2nd Year\Period 1\Multimedia Retrieval - INFOMR\Assignments\MultimediaRetrieval\Datasets\Jet\Jet\m1176.obj")
    output_dir = Path(r"C:\Users\bar24\OneDrive - Universiteit Utrecht\Documents\School\UU Data Sceince MSc\2nd Year\Period 1\Multimedia Retrieval - INFOMR\Assignments\MultimediaRetrieval\Datasets\UnifiedPreprocessed\JetTest")
    
    if not mesh_path.exists():
        print(f"File not found: {mesh_path}")
        return
    
    print("="*70)
    print("TESTING m1176 WITH UNIFIED PREPROCESSING PROCESSOR")
    print("="*70)
    print(f"Input: {mesh_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Create processor
    processor = UnifiedPreprocessingProcessor(target_vertices=7500)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    processor.output_base_dir = output_dir
    
    # Load original mesh
    print("1. Loading original mesh...")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    print(f"   Vertices: {len(mesh.vertices)}")
    print(f"   Triangles: {len(mesh.triangles)}")
    
    # Test remeshing with the processor's method
    print("\n2. Testing remeshing with processor's apply_remeshing_if_needed...")
    try:
        vertices, faces, was_remeshed = processor.apply_remeshing_if_needed(
            mesh_path, 
            target_vertices=7500,
            tolerance=0.2
        )
        
        if vertices is None:
            print("Remeshing returned None")
            return
        
        print(f"\n Remeshing completed!")
        print(f"   Original vertices: {len(mesh.vertices)}")
        print(f"   Result vertices: {len(vertices)}")
        print(f"   Result faces: {len(faces)}")
        print(f"   Was remeshed: {was_remeshed}")
        
        # Reconstruct mesh for saving
        remeshed_mesh = o3d.geometry.TriangleMesh()
        remeshed_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        remeshed_mesh.triangles = o3d.utility.Vector3iVector(faces)
        
        # Save output
        output_path = output_dir / "m1176_remeshed.obj"
        o3d.io.write_triangle_mesh(str(output_path), remeshed_mesh)
        print(f"\n Saved remeshed mesh to: {output_path}")
        
        # Now test full normalization pipeline
        print("\n3. Testing full normalization pipeline...")
        try:
            # Create a mock row for process_shape
            from pandas import Series
            row = Series({
                'filename': 'm1176.obj',
                'filepath': str(mesh_path),
                'category': 'Jet'
            })
            
            # Process the shape
            success = processor.process_shape(row, 'JetTest')
            
            if success:
                print("Full processing completed successfully!")
                
                # Check output files
                category_dir = output_dir / 'Jet'
                if category_dir.exists():
                    step_files = list(category_dir.glob("m1176_*.obj"))
                    print(f"\n Generated {len(step_files)} step files:")
                    for step_file in sorted(step_files):
                        mesh_check = o3d.io.read_triangle_mesh(str(step_file))
                        print(f"   {step_file.name}: {len(mesh_check.vertices)} vertices, {len(mesh_check.triangles)} triangles")
            else:
                print("Processing failed!")
                
        except Exception as e:
            print(f"Normalization pipeline error: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"Remeshing error: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

if __name__ == "__main__":
    test_m1176_with_processor()
