def compute_vertex_normals(vertices, faces):
    normals = np.zeros(vertices.shape, dtype=float)
    for tri in faces:
        v1, v2, v3 = vertices[tri]
        face_normal = np.cross(v2 - v1, v3 - v1)
        norm = np.linalg.norm(face_normal) + 1e-8
        if norm > 0:
            face_normal /= norm
        for idx in tri:
            normals[idx] += face_normal
    norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    normals /= norms
    return normals
import numpy as np
import plotly.graph_objects as go

def _wireframe_edges(vertices: np.ndarray, faces: np.ndarray):
    if faces.size == 0:
        return [], [], []
    edges = set()
    for tri in faces:
        tri = list(tri)
        for i in range(3):
            a, b = tri[i], tri[(i+1)%3]
            edges.add(tuple(sorted((a, b))))
    x, y, z = [], [], []
    for a, b in edges:
        x.extend([vertices[a,0], vertices[b,0], None])
        y.extend([vertices[a,1], vertices[b,1], None])
        z.extend([vertices[a,2], vertices[b,2], None])
    return x, y, z

def create_3d_plot(vertices: np.ndarray,
                   faces: np.ndarray,
                   title: str = "3D Shape",
                   show_wireframe: bool = False,
                   mesh_color: str = "lightblue",
                   smooth_shading: bool = False,
                   camera_config: dict = None,
                   use_rotated_vertices: bool = True):
    fig = go.Figure()
    if vertices.size == 0:
        fig.add_annotation(text="No data to display", showarrow=False)
        fig.update_layout(height=600, margin=dict(l=0, r=0, t=30, b=0), title=title)
        return fig

    if faces.size > 0:
        x, y, z = vertices.T
        i, j, k = faces.T
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color=mesh_color, opacity=0.7, name="Mesh",
            flatshading=not smooth_shading,
            lighting=dict(ambient=0.18, diffuse=1, fresnel=0.1, specular=1, roughness=0.05),
            lightposition=dict(x=100, y=200, z=0)
        ))
        if show_wireframe:
            wx, wy, wz = _wireframe_edges(vertices, faces)
            fig.add_trace(go.Scatter3d(
                x=wx, y=wy, z=wz, mode="lines",
                line=dict(color="black", width=2),
                name="Wireframe", hoverinfo="skip"
            ))
    else:
        x, y, z = vertices.T
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="markers",
                                   marker=dict(size=2, color=mesh_color),
                                   name="Point Cloud"))

    # Always add a red dot at the origin (0,0,0) to highlight the barycenter
    # Make it always visible by rendering it last and with special properties
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers",
        marker=dict(
            size=8,
            color="red", 
            symbol="circle",
            line=dict(color="darkred", width=2),  # Dark red outline for better visibility
            opacity=0.9  # Slightly transparent so it's visible even when occluded
        ),
        name="Barycenter (Origin)",
        hovertemplate="<b>Barycenter</b><br>Position: (0, 0, 0)<extra></extra>",
        showlegend=True
    ))

    # Configure camera
    if camera_config:
        camera_dict = {
            "eye": camera_config.get("eye", {"x": 1.5, "y": 1.5, "z": 1.5}),
            "center": camera_config.get("center", {"x": 0, "y": 0, "z": 0}),
            "up": camera_config.get("up", {"x": 0, "y": 0, "z": 1})
        }
    else:
        camera_dict = {"eye": {"x": 1.5, "y": 1.5, "z": 1.5}}

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="data",
            camera=camera_dict
        ),
        height=350,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig
