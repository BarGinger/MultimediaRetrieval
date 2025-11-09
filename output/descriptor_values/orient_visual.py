#%%

import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path

# ==== CONFIG ====
OBJ_NAME = "m533_05_scaled.obj"   # <- put your OBJ filename here
VIEW_DIR = np.array([0.0, 0.0, 1.0])  # direction you consider "camera toward"
OUT_BEFORE = "before_orientation.png"
OUT_AFTER  = "after_orientation.png"
# ===============

def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def face_normals(vertices, faces):
    v0 = vertices[faces[:,0]]
    v1 = vertices[faces[:,1]]
    v2 = vertices[faces[:,2]]
    n = np.cross(v1 - v0, v2 - v0)
    # avoid zero-area faces: normalise but keep zeros as zero
    lens = np.linalg.norm(n, axis=1, keepdims=True)
    nz = lens[:,0] > 0
    n[nz] = n[nz] / lens[nz]
    return n

def cw_mask(vertices, faces, view_dir):
    n = face_normals(vertices, faces)
    s = (n @ view_dir)  # dot with view direction
    # CW if dot < 0, CCW if dot >= 0
    return s < 0.0, s

def render_mesh(vertices, faces, colours, out_path, elev=20, azim=30):
    # vertices: (N,3), faces: (M,3), colours: (M,) strings or RGB tuples
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    tris = vertices[faces]
    coll = Poly3DCollection(tris, linewidths=0.2, edgecolor='k')
    coll.set_facecolor(colours)
    ax.add_collection3d(coll)

    # autoscale
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) / 2.0
    size = (maxs - mins).max()
    ax.set_xlim(center[0] - size/2, center[0] + size/2)
    ax.set_ylim(center[1] - size/2, center[1] + size/2)
    ax.set_zlim(center[2] - size/2, center[2] + size/2)

    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)

def main():
    obj_path = Path(OBJ_NAME)
    if not obj_path.exists():
        raise FileNotFoundError(f"Couldn't find {obj_path.resolve()}")

    # Load mesh without automatic processing so we see current windings
    mesh = trimesh.load(obj_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        # If the OBJ is a scene, merge into a single Trimesh
        mesh = trimesh.util.concatenate(tuple(g for g in mesh.geometry.values()))

    # Ensure we have triangles
    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError("The OBJ appears to have no faces.")
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy().astype(np.int32)

    view_dir = unit(VIEW_DIR)

    # ---------- BEFORE ----------
    cw, _ = cw_mask(vertices, faces, view_dir)
    colours_before = np.where(cw, '#d62728', '#2ca02c')  # red / green
    render_mesh(vertices, faces, colours_before, OUT_BEFORE)

    # ---------- FIX ORIENTATION ----------
    # This makes face windings consistent and normals coherent.
    # Trimesh tries to orient faces so normals point outward for closed meshes.
    mesh_fixed = mesh.copy()
    mesh_fixed.fix_normals()  # includes winding fixes where needed

    vertices2 = mesh_fixed.vertices
    faces2 = mesh_fixed.faces.astype(np.int32)

    # ---------- AFTER ----------
    cw2, _ = cw_mask(vertices2, faces2, view_dir)
    # Expect all (or nearly all) to be CCW relative to the view_dir -> all green
    colours_after = np.where(cw2, '#d62728', '#2ca02c')
    render_mesh(vertices2, faces2, colours_after, OUT_AFTER)

    # Quick stats in the console
    print(f"[BEFORE]  CW faces: {int(cw.sum())} / {len(faces)}")
    print(f"[AFTER ]  CW faces: {int(cw2.sum())} / {len(faces2)}")
    print(f"Saved: {OUT_BEFORE}, {OUT_AFTER}")

if __name__ == "__main__":
    main()

# %%
