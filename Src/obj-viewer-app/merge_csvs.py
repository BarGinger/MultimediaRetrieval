"""merge_csvs.py

Merge two CSV files on a primary key column (default: "name").

Behavior:
- Loads both CSVs using pandas.
- Merges on the key using a configurable join type (inner/left/right/outer).
- Columns from the first file appear first (preserving their order).
- Columns from the second file that are not present in the first are appended in their original order.
- If a non-key column exists in both files, the column from the second file will be renamed by appending a suffix (default: "_b") to avoid collisions; use --overwrite to prefer second file values into the original column name.

Usage:
    python Src\obj-viewer-app\merge_csvs.py file_a.csv file_b.csv -o merged.csv

Options:
    --key KEY        Primary key column name (default: name)
    --how {inner,left,right,outer}  Join type (default: outer)
    --suffix S       Suffix to append to overlapping columns from second file (default: _b)
    --overwrite      Overwrite same-named non-key columns with values from the second file (no suffix)

Requires: pandas
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
from typing import Optional

try:
    import pandas as pd
except Exception as e:
    raise ImportError("pandas is required to run this script. Install with `pip install pandas`.") from e


def merge_csvs(path_a: Path, path_b: Path, out: Path, key: str = "name", how: str = "outer", suffix: str = "_b", overwrite: bool = False) -> None:
    """Merge two CSV files and write the result to out.

    Args:
        path_a: first CSV file (its columns appear first in the output)
        path_b: second CSV file (its non-key columns are appended)
        out: output CSV path
        key: primary key column name
        how: join type (inner/left/right/outer)
        suffix: suffix to append to overlapping non-key columns from file B (ignored if overwrite=True)
        overwrite: if True, values from file B overwrite same-named columns from file A (no suffix)
    """
    if not path_a.exists():
        raise FileNotFoundError(f"File not found: {path_a}")
    if not path_b.exists():
        raise FileNotFoundError(f"File not found: {path_b}")

    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)

    if key not in df_a.columns:
        raise KeyError(f"Key column '{key}' not found in {path_a}")
    if key not in df_b.columns:
        raise KeyError(f"Key column '{key}' not found in {path_b}")

    # Determine suffixes for merge. pandas raises if overlapping columns exist and no suffix provided.
    if overwrite:
        # merge with a temporary suffix for B, then copy values from suffixed columns into A's columns
        tmp_suffix = '_tmp_b'
        merged = pd.merge(df_a, df_b, on=key, how=how, suffixes=("", tmp_suffix))
        # For every non-key column that exists in both A and B, copy values from the suffixed B column
        for c in df_b.columns:
            if c == key:
                continue
            suffixed = c + tmp_suffix
            if suffixed in merged.columns:
                # Overwrite/assign B's values into the original column name
                merged[c] = merged[suffixed]
                # drop the suffixed column afterwards
                merged.drop(columns=[suffixed], inplace=True)
    else:
        merged = pd.merge(df_a, df_b, on=key, how=how, suffixes=("", suffix))

    # Build final column order: columns from A first (preserve order), then non-key columns from B that are
    # not present in A (preserve B order). Handle renamed columns created by suffixes.
    a_cols = list(df_a.columns)
    b_cols = list(df_b.columns)

    final_cols = []
    # Start with A's columns
    for c in a_cols:
        final_cols.append(c)

    # For B, append columns that are not the key and not already included
    for c in b_cols:
        if c == key:
            continue
        if c in a_cols:
            if overwrite:
                # column exists in A and we want to overwrite: keep the original name (already present)
                # nothing to append; values from B already in merged under that name
                continue
            else:
                # column exists in A and we kept suffix for B -> merged column name is c + suffix
                col_b_name = c + suffix
                if col_b_name in merged.columns:
                    final_cols.append(col_b_name)
                else:
                    # fallback: maybe pandas merged without suffix (unlikely here)
                    if c in merged.columns:
                        final_cols.append(c)
        else:
            # column unique to B
            if c in merged.columns:
                final_cols.append(c)

    # Ensure key is first column (if desired) — keep original as in df_a if present
    if final_cols and final_cols[0] != key:
        # Move key to front if it's present
        if key in final_cols:
            final_cols.remove(key)
            final_cols.insert(0, key)
        elif key in merged.columns:
            final_cols.insert(0, key)

    # As a safety, include any other merged columns that were not listed (avoid dropping unexpected cols)
    for c in merged.columns:
        if c not in final_cols:
            final_cols.append(c)

    merged = merged.loc[:, final_cols]

    merged.to_csv(out, index=False)


def _parse_args(argv: Optional[list] = None):
    p = argparse.ArgumentParser(description="Merge two CSV files on a key and preserve column ordering.")
    p.add_argument('file_a', type=Path, help='First CSV file (its columns appear first)')
    p.add_argument('file_b', type=Path, help='Second CSV file (its non-key columns are appended)')
    p.add_argument('-o', '--out', type=Path, default=Path('merged.csv'), help='Output CSV path')
    p.add_argument('--key', type=str, default='name', help='Primary key column name (default: name)')
    p.add_argument('--how', type=str, choices=['inner', 'left', 'right', 'outer'], default='outer', help='Join type (default: outer)')
    p.add_argument('--suffix', type=str, default='_b', help='Suffix to append to overlapping non-key columns from second file (default: _b)')
    p.add_argument('--overwrite', action='store_true', help='If set, values from file B overwrite same-named columns from file A (no suffix)')
    return p.parse_args(argv)


def main(argv: Optional[list] = None):
    args = _parse_args(argv)
    try:
        merge_csvs(args.file_a, args.file_b, args.out, key=args.key, how=args.how, suffix=args.suffix, overwrite=args.overwrite)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
