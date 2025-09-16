from pathlib import Path
import numpy as np

class OBJParser:
    """Parser for OBJ 3D mesh files"""

    @staticmethod
    def parse_obj_file(filepath: str | Path):
        """
        Parse OBJ file and return vertices and faces.
        Returns:
            vertices: (N,3) float64
            faces:    (M,3) int64 (0-based indices)
        """
        vertices = []
        faces = []

        with open(filepath, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue

                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 4:
                        x, y, z = map(float, parts[1:4])
                        vertices.append([x, y, z])

                elif line.startswith("f "):
                    parts = line.split()
                    if len(parts) >= 4:
                        idxs = []
                        for part in parts[1:]:
                            idxs.append(int(part.split("/")[0]) - 1)
                        if len(idxs) == 3:
                            faces.append(idxs)
                        elif len(idxs) == 4:  # quad -> 2 tris
                            faces.append([idxs[0], idxs[1], idxs[2]])
                            faces.append([idxs[0], idxs[2], idxs[3]])

        return np.array(vertices, dtype=float), np.array(faces, dtype=int)
