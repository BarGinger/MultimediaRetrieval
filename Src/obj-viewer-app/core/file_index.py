from pathlib import Path
import pandas as pd

def get_file_tree(data_dir: str = "Data") -> pd.DataFrame:
    """
    Return DataFrame with columns: category, filename, filepath, size.
    Searches CWD/../.. for 'Data' to be dev-friendly.
    """
    files_data = []
    cwd = Path.cwd()

    candidates = [cwd / data_dir, cwd.parent / data_dir, cwd.parent.parent / data_dir]
    data_path = next((p for p in candidates if p.exists()), candidates[0])

    if data_path.exists():
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
