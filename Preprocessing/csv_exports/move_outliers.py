
import os
import pandas as pd
import shutil
import json

def main():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "files_outside_range.csv")
    data_dir = os.path.abspath(os.path.join(base_dir, "..", "..", "Datasets", "UnifiedPreprocessed", "Data"))
    outliers_dir = os.path.abspath(os.path.join(base_dir, "..", "..", "Datasets", "UnifiedPreprocessed", "outliers"))

    # Ensure outliers directory exists
    os.makedirs(outliers_dir, exist_ok=True)

    # Read CSV
    df = pd.read_csv(csv_path)
    outlier_names = df.loc[df['to_exclude'], 'file'].str.replace('.obj', '', regex=False)

    # The 9 expected suffixes for each shape
    SUFFIXES = [
        '_metadata.json',
        '_validation.json',
        '_00_original.obj',
        '_01_remeshed.obj',
        '_02_translated.obj',
        '_03_aligned.obj',
        '_04_flipped.obj',
        '_05_scaled.obj',
        '_unified.obj',
    ]

    for name in outlier_names:
        # Find the category from the metadata JSON in the Data folder
        found_category = None
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file == f"{name}_metadata.json":
                    metadata_path = os.path.join(root, file)
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        found_category = metadata.get('category')
                    except Exception as e:
                        print(f"Warning: Could not read category from {metadata_path}: {e}")
                    break
            if found_category:
                break
        if not found_category:
            print(f"Warning: Could not determine category for {name}, skipping.")
            continue
        # Move only the 9 expected files
        for suffix in SUFFIXES:
            src = os.path.join(data_dir, found_category, f"{name}{suffix}")
            if os.path.exists(src):
                dst = os.path.join(outliers_dir, f"{name}{suffix}")
                print(f"Moving {src} -> {dst}")
                shutil.move(src, dst)
            else:
                print(f"File not found, skipping: {src}")

if __name__ == "__main__":
    main()
