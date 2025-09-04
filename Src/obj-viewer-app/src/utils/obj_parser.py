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