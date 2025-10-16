import os
import shutil
import json

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
outliers_dir = os.path.abspath(os.path.join(base_dir, "..", "..", "Datasets", "UnifiedPreprocessed", "outliers"))
data_dir = os.path.abspath(os.path.join(base_dir, "..", "..", "Datasets", "UnifiedPreprocessed", "Data"))

# List all files in outliers folder
for file in os.listdir(outliers_dir):
    if file.endswith("_metadata.json"):
        # Use metadata to determine category and base name
        metadata_path = os.path.join(outliers_dir, file)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        category = metadata.get("category")
        base_name = file.replace("_metadata.json", "")
        # Move all files with this base name back to the correct category folder
        for f2 in os.listdir(outliers_dir):
            if f2.startswith(base_name):
                src = os.path.join(outliers_dir, f2)
                dst_dir = os.path.join(data_dir, category)
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir, f2)
                print(f"Moving {src} -> {dst}")
                shutil.move(src, dst)
