"""
Find rows in a CSV file where any feature column contains NaN or 0 values.

Reads: output/final_feature_values.csv
Writes: output/final_feature_values_nan_or_zero.csv

Usage:
    python scripts/find_nan_or_zero_rows.py
"""
import pandas as pd
from pathlib import Path

INPUT_PATH = Path("Src/matching/final_006.csv")
OUTPUT_PATH = Path("Src/matching/final_006_nan_or_zero.csv")
ID_COLUMN = "name"  # Change if your ID column is named differently

def main():
    df = pd.read_csv(INPUT_PATH)
    # Exclude ID column from feature columns
    feature_cols = [c for c in df.columns if c != ID_COLUMN]
    # Find rows with any NaN or 0 in feature columns
    mask_nan = df[feature_cols].isna().any(axis=1)
    mask_zero = (df[feature_cols] == 0).any(axis=1)
    mask = mask_nan | mask_zero

    out_df = df.loc[mask]
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Found {len(out_df)} rows with NaN or 0 in any feature. Saved to {OUTPUT_PATH}")

    # Drop those rows from the original and save cleaned version
    cleaned_df = df.loc[~mask]
    cleaned_path = INPUT_PATH.parent / (INPUT_PATH.stem + '_cleaned.csv')
    cleaned_df.to_csv(cleaned_path, index=False)
    print(f"Saved cleaned CSV (no NaN/0 rows) to {cleaned_path} ({len(cleaned_df)} rows)")

if __name__ == "__main__":
    main()
