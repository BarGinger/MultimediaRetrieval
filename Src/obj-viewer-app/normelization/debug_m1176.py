"""
Debug script for m1176 upsampling issue
Tests subdivision methods directly to understand what's happening
"""
import open3d as o3d
import numpy as np
from pathlib import Path

def analyze_connectivity(mesh, name="mesh"):
    """Analyze mesh connectivity"""
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    print(f"\n{name}:")
    print(f"  Vertices: {len(vertices)}")
    print(f"  Triangles: {len(triangles)}")
    
    if len(triangles) == 0:
        print("  ⚠️  No triangles!")
        return
    
    # Check for degenerate triangles
    degenerate = 0
    for tri in triangles:
        if tri[0] == tri[1] or tri[1] == tri[2] or tri[0] == tri[2]:
            degenerate += 1
    print(f"  Degenerate triangles: {degenerate}")
    
    # Check isolated vertices
    referenced = np.unique(triangles.flatten())
    isolated = len(vertices) - len(referenced)
    print(f"  Isolated vertices: {isolated}")
    
    # Count connected components
    adjacency = {i: set() for i in range(len(vertices))}
    for tri in triangles:
        adjacency[tri[0]].update([tri[1], tri[2]])
        adjacency[tri[1]].update([tri[0], tri[2]])
        adjacency[tri[2]].update([tri[0], tri[1]])
    
    visited = set()
    components = 0
    
    for start in range(len(vertices)):
        if start in visited:
            continue
        components += 1
        queue = [start]
        visited.add(start)
        while queue:
            current = queue.pop(0)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    
    print(f"  Connected components: {components}")
    
    if components > 1:
        print(f"  ⚠️  Mesh has {components} separate parts!")
    
    return components

def test_m1176():
    """Test m1176 subdivision"""
    mesh_path = Path(r"C:\Users\bar24\OneDrive - Universiteit Utrecht\Documents\School\UU Data Sceince MSc\2nd Year\Period 1\Multimedia Retrieval - INFOMR\Assignments\MultimediaRetrieval\Datasets\Jet\Jet\m1176.obj")
    
    if not mesh_path.exists():
        print(f"❌ File not found: {mesh_path}")
        return
    
    print("="*60)
    print("DEBUGGING m1176 SUBDIVISION")
    print("="*60)
    
    # Load mesh
    print("\n1. Loading original mesh...")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    components_original = analyze_connectivity(mesh, "Original mesh")
    
    # Clean mesh
    print("\n2. Cleaning mesh...")
    print("   Removing:")
    print("   - Degenerate triangles")
    mesh.remove_degenerate_triangles()
    print("   - Duplicated vertices")
    mesh.remove_duplicated_vertices()
    print("   - Duplicated triangles")
    mesh.remove_duplicated_triangles()
    print("   - Non-manifold edges")
    mesh.remove_non_manifold_edges()
    print("   - Unreferenced vertices")
    mesh.remove_unreferenced_vertices()
    
    components_cleaned = analyze_connectivity(mesh, "Cleaned mesh")
    
    if components_cleaned > components_original:
        print(f"\n⚠️  WARNING: Cleaning INCREASED components from {components_original} to {components_cleaned}!")
        print("   The cleaning process is breaking the mesh apart!")
    
    # Test Loop subdivision
    print("\n3. Testing Loop subdivision...")
    try:
        if hasattr(mesh, 'subdivide_loop'):
            mesh_loop = mesh.subdivide_loop(number_of_iterations=1)
            components_loop = analyze_connectivity(mesh_loop, "After Loop subdivision")
            
            if components_loop > 3:
                print(f"   ❌ Loop created TOO MANY components: {components_loop}")
        else:
            print("   ❌ Loop subdivision not available")
    except Exception as e:
        print(f"   ❌ Loop subdivision failed: {e}")
    
    # Test Midpoint subdivision
    print("\n4. Testing Midpoint subdivision...")
    try:
        mesh_midpoint = mesh.subdivide_midpoint(number_of_iterations=1)
        components_midpoint = analyze_connectivity(mesh_midpoint, "After Midpoint subdivision")
        
        if components_midpoint > 3:
            print(f"   ❌ Midpoint created TOO MANY components: {components_midpoint}")
        else:
            print(f"   ✅ Midpoint looks good with {components_midpoint} components")
    except Exception as e:
        print(f"   ❌ Midpoint subdivision failed: {e}")
    
    # Test without aggressive cleaning
    print("\n5. Testing with MINIMAL cleaning (only duplicates)...")
    mesh_minimal = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh_minimal.remove_duplicated_vertices()
    mesh_minimal.remove_duplicated_triangles()
    # Skip remove_non_manifold_edges and remove_degenerate_triangles
    components_minimal = analyze_connectivity(mesh_minimal, "Minimally cleaned mesh")
    
    try:
        mesh_minimal_subdivided = mesh_minimal.subdivide_midpoint(number_of_iterations=1)
        components_minimal_sub = analyze_connectivity(mesh_minimal_subdivided, "After Midpoint (minimal cleaning)")
        
        if components_minimal_sub <= 3:
            print(f"   ✅ SUCCESS! Minimal cleaning + midpoint = {components_minimal_sub} components")
    except Exception as e:
        print(f"   ❌ Minimal cleaning approach failed: {e}")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    
    if components_cleaned > components_original:
        print("🔧 The aggressive cleaning is BREAKING the mesh!")
        print("   Solution: Use minimal cleaning (only remove duplicates)")
    
    if components_loop > 10:
        print("🔧 Loop subdivision creates too many components")
        print("   Solution: Use midpoint subdivision instead")
    
    if components_midpoint <= 3:
        print("✅ Midpoint subdivision works well!")
    
    print("\n")

if __name__ == "__main__":
    test_m1176()
