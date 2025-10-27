"""Merge final_006.csv into analysis_results_unifiedPreprocessed_data.csv.

This script takes the rows in the final CSV (which contain only
*_06_fill_holes_and_orientation.obj files) and copies a set of
columns into all rows that belong to the same base object (e.g. m1337)
in the analysis CSV (all processing steps).

Default paths (relative to repo root) can be overridden via CLI.

Usage (powershell):
    python .\scripts\merge.py
    python .\scripts\merge.py --final output/final_006.csv --analysis Datasets/UnifiedPreprocessed/Data/analysis_results_unifiedPreprocessed_data.csv

The script creates a backup of the analysis CSV before overwriting it.
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
import sys
import pandas as pd


DEFAULT_FINAL = Path("output") / "final_006.csv"
DEFAULT_ANALYSIS = Path("Datasets") / "UnifiedPreprocessed" / "Data" / "analysis_results_unifiedPreprocessed_data.csv"

COPY_COLS = [
    "surface_area",
    "compactness",
    "rectangularity",
    "diameter",
    "convexity",
    "eccentricity",
    "A3_hist",
    "A3_bins",
    "D1_hist",
    "D1_bins",
    "D2_hist",
    "D2_bins",
    "D3_hist",
    "D3_bins",
    "D4_hist",
    "D4_bins",
    "shape_file",
    "name",
    "class_b",
]


def detect_filename_column(df: pd.DataFrame) -> str | None:
    """Return a column name from df that looks like it contains the filename.

    Preference order: shape_file, shape, file, filename
    """
    for cand in ("shape_file", "shape", "file", "filename"):
        if cand in df.columns:
            return cand
    return None


def base_id_from_filename(fname: str) -> str:
    """Extract base id from filename like m1337_06_fill_holes_and_orientation.obj -> m1337

    This uses the part before the first underscore which matches the repository naming.
    """
    if not isinstance(fname, str):
        return ""
    return fname.split("_")[0]


def main(final_csv: Path, analysis_csv: Path, backup: bool = True) -> int:
    if not final_csv.exists():
        print(f"ERROR: final CSV not found: {final_csv}")
        return 2
    if not analysis_csv.exists():
        print(f"ERROR: analysis CSV not found: {analysis_csv}")
        return 2

    print(f"Reading final CSV: {final_csv}")
    final_df = pd.read_csv(final_csv, dtype=str)
    print(f"Reading analysis CSV: {analysis_csv}")
    analysis_df = pd.read_csv(analysis_csv, dtype=str)

    fname_col_final = detect_filename_column(final_df)
    fname_col_analysis = detect_filename_column(analysis_df)

    if fname_col_final is None:
        print("ERROR: cannot detect filename column in final CSV. Expected 'shape_file' or 'shape'.")
        return 2
    if fname_col_analysis is None:
        print("ERROR: cannot detect filename column in analysis CSV. Expected 'shape_file' or 'shape'.")
        return 2

    print(f"Using filename column '{fname_col_final}' in final and '{fname_col_analysis}' in analysis.")

    # normalize columns: ensure COPY_COLS exist in final_df (missing ones will be added as NaN)
    for col in COPY_COLS:
        if col not in final_df.columns:
            final_df[col] = pd.NA

    # build mapping from base_id -> row (prefer the first or last if duplicates; final CSV should have only 06 rows)
    final_df = final_df.copy()
    final_df["_base_id"] = final_df[fname_col_final].apply(base_id_from_filename)

    # If there are multiple final rows per base (unlikely), keep the last one
    final_latest = final_df.groupby("_base_id", sort=False).last().reset_index()

    # prepare dicts for lookup
    lookup = {}
    for _, row in final_latest.iterrows():
        base = row["_base_id"]
        lookup[base] = {col: row.get(col, pd.NA) for col in COPY_COLS}

    # add base id to analysis
    analysis_df = analysis_df.copy()
    analysis_df["_base_id"] = analysis_df[fname_col_analysis].apply(base_id_from_filename)

    # ensure columns exist in analysis
    for col in COPY_COLS:
        if col not in analysis_df.columns:
            analysis_df[col] = pd.NA

    # map values
    def get_value(base: str, col: str):
        entry = lookup.get(base)
        if not entry:
            return pd.NA
        return entry.get(col, pd.NA)

    total_rows = len(analysis_df)
    filled_counts = {col: 0 for col in COPY_COLS}
    missing_counts = {col: 0 for col in COPY_COLS}

    for col in COPY_COLS:
        # vectorized map: map base_id -> value
        analysis_df[col] = analysis_df["_base_id"].map(lambda b: get_value(b, col))

        # consider a value present only if it's not NA and not an empty string
        non_empty_mask = analysis_df[col].notna() & (analysis_df[col].astype(str).str.strip() != "")
        non_empty = int(non_empty_mask.sum())
        missing = int(total_rows - non_empty)
        filled_counts[col] = non_empty
        missing_counts[col] = missing

    # summary
    print(f"Processed {total_rows} analysis rows.")
    for col in COPY_COLS:
        print(f"  -> {col}: {filled_counts[col]} non-empty values, {missing_counts[col]} empty/NA values")

    # backup
    if backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = analysis_csv.with_suffix(analysis_csv.suffix + f".{stamp}.bak")
        print(f"Creating backup: {bak}")
        shutil.copy2(analysis_csv, bak)

    # drop helper column and write back
    analysis_df = analysis_df.drop(columns=["_base_id"], errors="ignore")

    out_path = analysis_csv
    print(f"Writing merged analysis CSV to: {out_path}")
    analysis_df.to_csv(out_path, index=False)

    print("Done.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge final_006.csv into analysis CSV for UnifiedPreprocessed/Data")
    parser.add_argument("--final", type=Path, default=DEFAULT_FINAL, help="Path to final CSV (default: output/final_006.csv)")
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS, help="Path to analysis CSV to update (default: Datasets/UnifiedPreprocessed/Data/analysis_results_unifiedPreprocessed_data.csv)")
    parser.add_argument("--no-backup", dest="backup", action="store_false", help="Do not create a timestamped backup of the analysis CSV")
    args = parser.parse_args()
    sys.exit(main(args.final, args.analysis, backup=args.backup))
