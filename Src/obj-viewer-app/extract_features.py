from pathlib import Path
import pandas as pd
import numpy as np
import traceback
from tqdm import tqdm

from core.shapeMesh import ShapeMesh
from core.extractions import MeshExtractions
from core.transformations import MeshTransformations

MESH_EXTS = {".obj"} 

# Scalar features only (histogram descriptors like A3/D1/D2 are omitted)
FEATURES = [
    "surface_area",
    "volume",
    "compactness",
    "rectangularity",
    "diameter",
    "convexity",
    "eccentricity",
]


def compute_scalar_features(mesh: ShapeMesh) -> dict:
    """
    Compute all scalar MeshExtractions for a mesh safely.
    Returns a dict {feature_name: float}. Missing/failed become 0.0
    """
    out = {}
    try:
        # Pre-compute shared values to avoid recomputation when possible
        # (These helpers accept optional S, V where applicable)
        S = None
        V = None

        # surface_area
        try:
            S = MeshExtractions.surface_area(mesh)
            out["surface_area"] = float(S)
        except Exception:
            out["surface_area"] = 0.0
            S = None

        # volume
        try:
            V = MeshExtractions.volume(mesh)
            out["volume"] = float(V)
        except Exception:
            out["volume"] = 0.0
            V = None

        # compactness (uses S and V if available)
        try:
            out["compactness"] = float(MeshExtractions.compactness(mesh, S=S, V=V))
        except Exception:
            out["compactness"] = 0.0

        # rectangularity (uses surface areas; robust to internal fallback)
        try:
            out["rectangularity"] = float(MeshExtractions.rectangularity(mesh))
        except Exception:
            out["rectangularity"] = 0.0

        # diameter
        try:
            out["diameter"] = float(MeshExtractions.diameter(mesh))
        except Exception:
            out["diameter"] = 0.0

        # convexity (V_mesh / V_hull)
        try:
            out["convexity"] = float(MeshExtractions.convexity(mesh, V_mesh=V))
        except Exception:
            out["convexity"] = 0.0

        # eccentricity (eigenvalue ratio)
        try:
            out["eccentricity"] = float(MeshExtractions.eccentricity(mesh))
        except Exception:
            out["eccentricity"] = 0.0

    except Exception:
        # catastrophic failure: set all zeros
        for f in FEATURES:
            out.setdefault(f, 0.0)

    # Ensure numeric outputs
    for k, v in list(out.items()):
        try:
            out[k] = float(v)
        except Exception:
            out[k] = 0.0

    return out


def prepare_copy(mesh: ShapeMesh) -> ShapeMesh:
    """Make a lightweight copy to avoid mutating the original when prepping."""
    return ShapeMesh(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces.copy(),
        category=mesh.category,
        filename=mesh.filename,
        face_types=mesh.face_types,
        bounding_box=mesh.bounding_box,
        size=mesh.size,
        filepath=mesh.filepath,
        base_mesh=mesh.base_mesh,
    )


def scan_mesh_files(root_dir: Path):
    """Yield Path objects for all mesh files under root_dir recursively"""
    for p in root_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in MESH_EXTS:
            if p.stem.lower().endswith("_prepared"):
                continue  # skip already prepared meshes
            yield p



def run(main_folder: str, output_csv: str):
    root = Path(main_folder).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Main folder not found: {root}")

    # Ensure output folder exists; if you want a fresh run each time, uncomment the next two lines:
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    # if Path(output_csv).exists():
    #     Path(output_csv).unlink()

    # Fixed column order
    ordered_cols = ["name"] + [f"{f}_before" for f in FEATURES] + [f"{f}_after" for f in FEATURES]

    # If file doesn't exist yet, create it with header only
    if not Path(output_csv).exists():
        pd.DataFrame(columns=ordered_cols).to_csv(output_csv, index=False)

    mesh_files = list(scan_mesh_files(root))   # so tqdm knows total
    for mesh_path in tqdm(mesh_files, desc="Processing meshes", unit="mesh"):
        rel_name = str(mesh_path.relative_to(root))
        try:
            mesh_raw = ShapeMesh.from_file(str(mesh_path))

            feats_before = compute_scalar_features(mesh_raw)
            mesh_prepped = MeshTransformations.prepare_for_extraction(prepare_copy(mesh_raw))
            feats_after  = compute_scalar_features(mesh_prepped)

            prepared_path = mesh_path.with_name(mesh_path.stem + "_prepared" + mesh_path.suffix)
            mesh_prepped.save_as_obj(str(prepared_path))


            row = {"name": rel_name}
            for f in FEATURES:
                row[f"{f}_before"] = feats_before.get(f, 0.0)
                row[f"{f}_after"]  = feats_after.get(f, 0.0)

        except Exception:
            traceback.print_exc()
            row = {"name": rel_name}
            for f in FEATURES:
                row[f"{f}_before"] = 0.0
                row[f"{f}_after"]  = 0.0

        # Align to ordered columns (fills missing with NaN; we coerce to 0.0 for safety)
        df_row = pd.DataFrame([row], columns=ordered_cols).fillna(0.0)

        # Append without header (already written once above)
        df_row.to_csv(output_csv, mode="a", header=False, index=False, float_format="%.6f")

    print(f"✅ Wrote: {output_csv}")



if __name__ == "__main__":
    # Example usage:
    #   python tools/run_extractions_before_after.py "/path/to/main/folder" "output/features_before_after.csv"
    import sys
    if len(sys.argv) != 3:
        print("Usage: python run_extractions_before_after.py <main_folder> <output_csv>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])