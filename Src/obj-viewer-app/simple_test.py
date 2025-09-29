"""
Simple test script to isolate the exact error
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from core.file_index import get_file_tree
from core.shapeMesh import ShapeMesh

def simple_test():
    """Test with minimal complexity"""
    print("🔍 Simple Test - Creating One ShapeMesh")
    print("=" * 40)
    
    try:
        # Get file list
        file_df = get_file_tree(data_dir="Data_sampled_resampled_normalized")
        print(f"Found {len(file_df)} files")
        
        # Get first row
        first_row = file_df.iloc[0]
        print(f"First file: {first_row['filename']}")
        print(f"Filepath: {first_row['filepath']}")
        print(f"Category: {first_row['category']}")
        
        # Try to create mesh
        print("Creating ShapeMesh...")
        mesh = ShapeMesh.from_file_row(first_row)
        print(f"✅ Success! Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
        
        # Try normalization
        print("Testing normalization...")
        normalized_vertices = mesh.apply_full_normalization()
        print(f"✅ Normalization success! Final shape: {normalized_vertices.shape}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_test()