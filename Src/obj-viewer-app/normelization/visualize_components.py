"""
Visualize the connected components of a mesh
"""
import open3d as o3d
import numpy as np

def visualize_components(mesh_path):
    """Visualize each connected component in a different color"""
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    # Get connected components
    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    
    # Get unique components
    unique_components = np.unique(triangle_clusters)
    n_components = len(unique_components)
    
    print(f"Mesh: {mesh_path}")
    print(f"Total triangles: {len(mesh.triangles)}")
    print(f"Total vertices: {len(mesh.vertices)}")
    print(f"Connected components: {n_components}\n")
    
    # Print component stats
    for i, comp_id in enumerate(unique_components):
        mask = triangle_clusters == comp_id
        n_tris = np.sum(mask)
        area = cluster_area[comp_id]
        print(f"Component {i+1}: {n_tris} triangles, area: {area:.4f}")
    
    # Color each component differently by vertex
    # Assign colors per vertex based on which component their triangles belong to
    vertex_colors = np.zeros((len(mesh.vertices), 3))
    np.random.seed(42)  # For consistent colors
    
    triangles = np.asarray(mesh.triangles)
    for comp_id in unique_components:
        mask = triangle_clusters == comp_id
        color = np.random.rand(3)
        # Get all vertices in this component
        component_triangles = triangles[mask]
        component_vertices = np.unique(component_triangles.flatten())
        vertex_colors[component_vertices] = color
    
    # Set vertex colors
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    
    # Compute normals for better visualization
    mesh.compute_vertex_normals()
    
    print("\nVisualizing mesh with colored components...")
    print("Each component is shown in a different color")
    o3d.visualization.draw_geometries([mesh], 
                                       window_name="Component Visualization",
                                       width=1200,
                                       height=800)

if __name__ == "__main__":
    # Visualize original mesh
    original_path = r"C:\Users\bar24\OneDrive - Universiteit Utrecht\Documents\School\UU Data Sceince MSc\2nd Year\Period 1\Multimedia Retrieval - INFOMR\Assignments\MultimediaRetrieval\Datasets\Jet\Jet\m1176.obj"
    print("="*70)
    print("ORIGINAL MESH")
    print("="*70)
    visualize_components(original_path)
    
    # Visualize processed mesh
    processed_path = r"C:\Users\bar24\OneDrive - Universiteit Utrecht\Documents\School\UU Data Sceince MSc\2nd Year\Period 1\Multimedia Retrieval - INFOMR\Assignments\MultimediaRetrieval\Datasets\UnifiedPreprocessed\JetTest\m1176_remeshed.obj"
    print("\n" + "="*70)
    print("PROCESSED MESH (Midpoint Subdivision)")
    print("="*70)
    visualize_components(processed_path)
