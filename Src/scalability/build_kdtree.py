# %%
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import joblib  

FEATURES_CSV = "final_006_cleaned.csv"
OUTPUT_FILE = "kdtree.joblib"

# Load feature data 
df = pd.read_csv(FEATURES_CSV)
df = df[['shape', 'surface_area', 'compactness', 'rectangularity', 'diameter', 'convexity', 'eccentricity', 'A3_hist', 'D1_hist', 'D2_hist', 'D3_hist', 'D4_hist']]
for hist in ['A3_hist', 'D1_hist', 'D2_hist', 'D3_hist', 'D4_hist']:
    hist_df = df[hist].str.split(";", expand=True)
    hist_df = hist_df.astype(np.float32)
    hist_cols = [f"{hist}_{i}" for i in range(hist_df.shape[1])]
    hist_df.columns = hist_cols
    df = pd.concat([df.drop(columns=[hist]), hist_df], axis=1)
print(df.head())
feat_cols = [c for c in df.columns if c != "shape"]

names = df["shape"].astype(str).to_numpy()
X = df[feat_cols].to_numpy(dtype=np.float32)

# Build the KDTree
print(f"Building KDTree for {X.shape[0]} vectors of dimension {X.shape[1]}...")
tree = KDTree(X, metric="euclidean", leaf_size=40)

# Save
joblib.dump(
    {
        "tree": tree,
        "X_shape": X.shape,
        "names": names,
        "feature_columns": feat_cols,
    },
    OUTPUT_FILE,
)
print(f"KDTree saved to {OUTPUT_FILE}")

# %%
