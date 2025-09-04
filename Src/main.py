"""
Main module for the application.
Last updated: 4/9/2025
"""

def parse_obj_file(filepath):
    """Parse OBJ file and extract vertices."""
    vertices = []
    
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('v '):  # Vertex line
                parts = line.split()
                if len(parts) >= 4:  # v x y z
                    x, y, z = map(float, parts[1:4])
                    vertices.append([x, y, z])
    
    return vertices

if __name__ == "__main__":
    import plotly.graph_objects as go
    import numpy as np

    # Parse OBJ file
    path = r"Data/Cup/D00035.obj"
    vertices = parse_obj_file(path)
    pts = np.array(vertices)
    
    if len(pts) > 0:
        x, y, z = pts.T
        fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, color='lightpink', opacity=0.50)])
        fig.show(renderer="browser")
    else:
        print("No vertices found in the OBJ file")