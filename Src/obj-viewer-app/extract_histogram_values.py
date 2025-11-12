from pathlib import Path
import pandas as pd
import traceback
from tqdm import tqdm

from core.shapeMesh import ShapeMesh
from core.extractions_minmax import MeshExtractionsMinMax

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
            out["min_A3"], out["max_A3"] = MeshExtractionsMinMax.A3(mesh)
        except Exception:
            out["min_A3"], out["max_A3"] = (0.0, 0.0)

        # D1
        try:
            out["min_D1"], out["max_D1"] = MeshExtractionsMinMax.D1(mesh)
        except Exception:
            out["min_D1"], out["max_D1"] = (0.0, 0.0)

        # D2
        try:
            out["min_D2"], out["max_D2"] = MeshExtractionsMinMax.D2(mesh)
        except Exception:
            out["min_D2"], out["max_D2"] = (0.0, 0.0)
        
        # D3
        try:
            out["min_D3"], out["max_D3"] = MeshExtractionsMinMax.D3(mesh)
        except Exception:
            out["min_D3"], out["max_D3"] = (0.0, 0.0)
        
        # D4
        try:
            out["min_D4"], out["max_D4"] = MeshExtractionsMinMax.D4(mesh)
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


def scan_mesh_files(root_dir: Path):
    """Yield Path objects for all mesh files under root_dir recursively"""
    for p in root_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in MESH_EXTS:
            name = p.stem.lower()
            if "_unified_prepared" in name:
                yield p


def run(main_folder: str, output_csv: str):
    root = Path(main_folder).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Main folder not found: {root}")

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    ordered_cols = ["name"] + FEATURES


    if not Path(output_csv).exists():
        pd.DataFrame(columns=ordered_cols).to_csv(output_csv, index=False)

    mesh_files = list(scan_mesh_files(root))
    for mesh_path in tqdm(mesh_files, desc="Processing unified_prepared meshes", unit="mesh"):
        rel_name = str(mesh_path.relative_to(root))
        try:
            mesh = ShapeMesh.from_file(str(mesh_path))
            feats = compute_values(mesh)
            row = {"name": rel_name, **feats}
        except Exception:
            print(f"Error processing mesh: {rel_name}")
            traceback.print_exc()
            row = {"name": rel_name}
            for f in FEATURES:
                row[f] = 0.0

        # Append één rij in de vaste kolomvolgorde
        df_row = pd.DataFrame([row], columns=ordered_cols).fillna(0.0)
        df_row.to_csv(output_csv, mode="a", header=False, index=False, float_format="%.6f")

    print(f"Wrote: {output_csv}")


if __name__ == "__main__":
    # Example usage:
    #   python tools/run_extractions_before_after.py "/path/to/main/folder" "output/features_before_after.csv"
    import sys
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print("Usage: python run_unified_prepared_to_csv.py <main_folder> [output_csv]")
        sys.exit(1)

    main_folder = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) == 3 else "features_unified_prepared.csv"
    run(main_folder, output_csv)
