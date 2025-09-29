from pathlib import Path
import pandas as pd

def get_file_tree(data_dir: str = "Data") -> pd.DataFrame:
    """
    Return DataFrame with columns: category, filename, filepath, size.
    Searches CWD/../.. for 'Datasets/{data_dir}' to be dev-friendly.
    """
    files_data = []
    cwd = Path.cwd()

    # Look for data_dir within Datasets folder
    dataset_path = f"Datasets/{data_dir}"
    candidates = [cwd / dataset_path, cwd.parent / dataset_path, cwd.parent.parent / dataset_path]
    data_path = next((p for p in candidates if p.exists()), candidates[0])

    if data_path.exists():
        # Special handling for NormalizedShapes dataset which has nested structure
        if data_dir == "NormalizedShapes":
            # Look for subdatasets within NormalizedShapes
            for subdataset_dir in data_path.iterdir():
                if subdataset_dir.is_dir():
                    # Each subdataset contains category directories
                    for category_dir in subdataset_dir.iterdir():
                        if category_dir.is_dir():
                            category = category_dir.name
                            for obj_file in category_dir.glob("*.obj"):
                                files_data.append({
                                    "category": category,
                                    "filename": obj_file.name,
                                    "filepath": str(obj_file),
                                    "size": obj_file.stat().st_size
                                })
        else:
            # Normal structure: Datasets/DataName/CategoryName/*.obj
            for category_dir in data_path.iterdir():
                if category_dir.is_dir():
                    category = category_dir.name
                    for obj_file in category_dir.glob("*.obj"):
                        files_data.append({
                            "category": category,
                            "filename": obj_file.name,
                            "filepath": str(obj_file),
                            "size": obj_file.stat().st_size
                        })

    df = pd.DataFrame(files_data)
    return df
