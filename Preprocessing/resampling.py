import os
import open3d as o3d
from pathlib import Path

# --- Toggle for sampled dataset ---
USE_SAMPLED_DATASET = True  # Set to True to use Data_sampled, False for full Data
BASE = Path(__file__).parent.parent.resolve()
SOURCE_ROOT = BASE / ('Data_sampled' if USE_SAMPLED_DATASET else 'Data')
TARGET_ROOT = BASE / ('Data_sampled_resampled' if USE_SAMPLED_DATASET else 'Data_resampled')
TARGET_VERTEX_COUNT = 5000


def resample_mesh(input_path, output_path, target_vertices=TARGET_VERTEX_COUNT):
    mesh = o3d.io.read_triangle_mesh(str(input_path))
    if mesh.is_empty():
        print(f"❌ Empty mesh: {input_path}")
        return False
    if len(mesh.vertices) <= target_vertices:
        o3d.io.write_triangle_mesh(str(output_path), mesh)
        return True
    # Simplify mesh
    try:
        mesh_simplified = mesh.simplify_quadric_decimation(target_vertices)
        o3d.io.write_triangle_mesh(str(output_path), mesh_simplified)
        return True
    except Exception as e:
        print(f"❌ Error simplifying {input_path}: {e}")
        return False

def main():
	print(f"Resampling meshes from {SOURCE_ROOT} to {TARGET_ROOT} (target: {TARGET_VERTEX_COUNT} vertices)")
	for category_dir in SOURCE_ROOT.iterdir():
		if not category_dir.is_dir():
			continue
		out_category_dir = TARGET_ROOT / category_dir.name
		out_category_dir.mkdir(parents=True, exist_ok=True)
		for obj_file in category_dir.glob('*.obj'):
			out_file = out_category_dir / obj_file.name
			success = resample_mesh(obj_file, out_file)
			if success:
				print(f"✅ {obj_file} -> {out_file}")
			else:
				print(f"❌ Failed: {obj_file}")

if __name__ == '__main__':
	main()
