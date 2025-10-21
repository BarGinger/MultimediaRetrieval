from pathlib import Path
import traceback
from tqdm import tqdm

from core.shapeMesh import ShapeMesh
from core.transformations import MeshTransformations

MESH_EXTS = {".obj"} 

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
    """Yield Path objects for all mesh files under root_dir recursively.
    Only take files with '_unified' in the name and skip those with '_prepared'."""
    for p in root_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in MESH_EXTS:
            name = p.stem.lower()
            if "_prepared" in name:
                continue  # skip already prepared meshes
            if "_unified" in name:
                yield p


def run(main_folder: str):
    root = Path(main_folder).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Main folder not found: {root}")

    mesh_files = list(scan_mesh_files(root))   # so tqdm knows total
    for mesh_path in tqdm(mesh_files, desc="Processing meshes", unit="mesh"):
        rel_name = str(mesh_path.relative_to(root))
        try:
            mesh_raw = ShapeMesh.from_file(str(mesh_path))
            mesh_prepped = MeshTransformations.prepare_for_extraction(prepare_copy(mesh_raw))

            prepared_path = mesh_path.with_name(mesh_path.stem + "_prepared" + mesh_path.suffix)
            mesh_prepped.save_as_obj(str(prepared_path))

        except Exception:
            print(f"❌ Error processing mesh: {rel_name}")
            traceback.print_exc()

    print("✅ All meshes processed.")


if __name__ == "__main__":
    # Example usage:
    #   python run_extractions_before_after.py "/path/to/main/folder"
    import sys
    if len(sys.argv) != 2:
        print("Usage: python run_extractions_before_after.py <main_folder>")
        sys.exit(1)
    run(sys.argv[1])
