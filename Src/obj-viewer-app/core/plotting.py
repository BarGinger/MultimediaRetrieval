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
                   mesh_color: str = "lightblue"):
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

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=600,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig
