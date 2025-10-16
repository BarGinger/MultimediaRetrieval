from pathlib import Path
import pandas as pd
import numpy as np
import traceback
from tqdm import tqdm

from core.shapeMesh import ShapeMesh
from core.extractions import MeshExtractions
from core.transformations import MeshTransformations

MESH_EXTS = {".obj"} 

FEATURES = [
    "min_A3",
    "max_A3",
    "min_D1",
    "max_D1",
    "min_D2",
    "max_D2",
    "min_D3",
    "max_D3",
    "min_D4",
    "max_D4"
]


def compute_values(mesh: ShapeMesh) -> dict:
    out = {}
    try:
        # A3
        try:
            out["min_A3"], out["max_A3"] = MeshExtractions.A3(mesh)
        except Exception:
            out["min_A3"], out["max_A3"] = (0.0, 0.0)

        # D1
        try:
            out["min_D1"], out["max_D1"] = MeshExtractions.D1(mesh)
        except Exception:
            out["min_D1"], out["max_D1"] = (0.0, 0.0)

        # D2
        try:
            out["min_D2"], out["max_D2"] = MeshExtractions.D2(mesh)
        except Exception:
            out["min_D2"], out["max_D2"] = (0.0, 0.0)
        
        # D3
        try:
            out["min_D3"], out["max_D3"] = MeshExtractions.D3(mesh)
        except Exception:
            out["min_D3"], out["max_D3"] = (0.0, 0.0)
        
        # D4
        try:
            out["min_D4"], out["max_D4"] = MeshExtractions.D4(mesh)
        except Exception:
            out["min_D4"], out["max_D4"] = (0.0, 0.0)




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