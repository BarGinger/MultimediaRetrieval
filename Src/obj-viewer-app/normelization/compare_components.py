"""
Compare component statistics between original and processed meshes
"""
import open3d as o3d
import numpy as np

def get_component_stats(mesh_path):
    """Get statistics about connected components"""
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    # Get connected components
    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    
    # Get unique components
    unique_components = np.unique(triangle_clusters)
    n_components = len(unique_components)
    
    print(f"Mesh: {mesh_path.split('\\\\')[-1]}")
    print(f"Total triangles: {len(mesh.triangles)}")
    print(f"Total vertices: {len(mesh.vertices)}")
    print(f"Connected components: {n_components}\n")
    
    # Print component stats sorted by size
    component_info = []
    for comp_id in unique_components:
        mask = triangle_clusters == comp_id
        n_tris = np.sum(mask)
        area = cluster_area[comp_id]
        component_info.append((n_tris, area))
    
    component_info.sort(reverse=True)  # Sort by triangle count
    
    for i, (n_tris, area) in enumerate(component_info):
        percentage = (n_tris / len(mesh.triangles)) * 100
        print(f"  Component {i+1}: {n_tris:5d} triangles ({percentage:5.1f}%), area: {area:.4f}")
    
    return n_components, component_info

if __name__ == "__main__":
    print("="*70)
    print("COMPONENT COMPARISON")
    print("="*70)
    
    # Original mesh
    original_path = r"C:\Users\bar24\OneDrive - Universiteit Utrecht\Documents\School\UU Data Sceince MSc\2nd Year\Period 1\Multimedia Retrieval - INFOMR\Assignments\MultimediaRetrieval\Datasets\Jet\Jet\m1176.obj"
    print("\n ORIGINAL MESH (before processing):")
    print("-" * 70)
    orig_n, orig_info = get_component_stats(original_path)
    
    # Processed mesh
    processed_path = r"C:\Users\bar24\OneDrive - Universiteit Utrecht\Documents\School\UU Data Sceince MSc\2nd Year\Period 1\Multimedia Retrieval - INFOMR\Assignments\MultimediaRetrieval\Datasets\UnifiedPreprocessed\JetTest\m1176_remeshed.obj"
    print("\n PROCESSED MESH (after Midpoint subdivision):")
    print("-" * 70)
    proc_n, proc_info = get_component_stats(processed_path)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Original components: {orig_n}")
    print(f"Processed components: {proc_n}")
    
    if orig_n == proc_n:
        print("\n SUCCESS: Component count preserved during subdivision!")
        print("   The mesh topology was NOT fragmented by processing.")
    elif proc_n > orig_n:
        print(f"\n️ WARNING: {proc_n - orig_n} extra component(s) created during processing")
        print("   Some fragmentation occurred, but may be acceptable if minor.")
    else:
        print(f"\n️ INFO: {orig_n - proc_n} fewer component(s) after processing")
        print("   Some components may have been merged during processing.")
    
    # Check if the major component is still major
    orig_major_pct = (orig_info[0][0] / sum(info[0] for info in orig_info)) * 100
    proc_major_pct = (proc_info[0][0] / sum(info[0] for info in proc_info)) * 100
    
    print(f"\nMain component size:")
    print(f"  Original: {orig_major_pct:.1f}% of triangles")
    print(f"  Processed: {proc_major_pct:.1f}% of triangles")
    
    if abs(orig_major_pct - proc_major_pct) < 5:
        print("Main component size preserved (±5%)")
    else:
        print(f"️ Main component size changed by {abs(orig_major_pct - proc_major_pct):.1f}%")
