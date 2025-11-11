# make_metadata.py
import pandas as pd
import re
from pathlib import Path

FEATURES_CSV = "final_006_cleaned.csv"
OUT = "metadata.csv"

def to_filename(name: str) -> str:
    s = str(name)
    if s.lower().endswith(".obj"):
        return s

    m = re.match(r"([A-Za-z0-9]+)", s)
    if m:
        return f"{m.group(1)}_unified.obj"
    return s

df = pd.read_csv(FEATURES_CSV, dtype=str)
if "shape" not in df.columns:
    raise RuntimeError("Expected 'shape' column in features CSV")

meta = pd.DataFrame({
    "filename": df["shape"].map(to_filename),
    # Add whatever else you want to return with each neighbor:
    # "shape": df["shape"],
    # "surface_area": df["surface_area"].astype(float),
    # ...
})

meta = meta.drop_duplicates(subset=["filename"])
meta.to_csv(OUT, index=False)
print(f"Wrote {OUT} with {len(meta)} rows")
