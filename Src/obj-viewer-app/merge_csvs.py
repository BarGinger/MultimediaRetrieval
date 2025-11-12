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

    Parameters:
        path_a (Path): First CSV file (its columns appear first in the output).
        path_b (Path): Second CSV file (its non-key columns are appended).
        out (Path): Output CSV path.
        key (str): Primary key column name.
        how (str): Join type (inner/left/right/outer).
        suffix (str): Suffix to append to overlapping non-key columns from file B (ignored if overwrite=True).
        overwrite (bool): If True, values from file B overwrite same-named columns from file A (no suffix).

    Returns:
        None

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


def merge_analysis_with_features(path_analysis: Path, path_features: Path, out: Path, left_key: str = 'shape_file', right_key: str = 'name', how: str = 'left', overwrite: bool = True, version_suffix: str | None = '_06_fill_holes_and_orientation.obj') -> None:
    """Specialized merge for analysis_results (A) and final_features (B).

    Produces output with columns: shape, class, <analysis columns except left_key/class>, 
    <feature columns except right_key/class and duplicates>.

    Parameters:
        path_analysis (Path): Path to analysis results CSV file.
        path_features (Path): Path to features CSV file.
        out (Path): Output CSV path.
        left_key (str): Column in analysis CSV (default 'shape_file').
        right_key (str): Column in features CSV (default 'name').
        how (str): Join type (default 'left').
        overwrite (bool): If True, prefer values from features file.
        version_suffix (str | None): Only keep analysis rows whose filename ends with this suffix.

    Returns:
        None

    """
    if not path_analysis.exists():
        raise FileNotFoundError(f"File not found: {path_analysis}")
    if not path_features.exists():
        raise FileNotFoundError(f"File not found: {path_features}")

    df_a = pd.read_csv(path_analysis)
    df_b = pd.read_csv(path_features)

    # If a version_suffix is provided, keep only rows in analysis whose filename ends with it
    if version_suffix:
        def filename_from_path(x: str) -> str:
            try:
                return Path(str(x)).name
            except Exception:
                return str(x)

        mask = df_a[left_key].astype(str).apply(lambda s: filename_from_path(s).endswith(version_suffix))
        df_a = df_a[mask].copy()

    if left_key not in df_a.columns:
        raise KeyError(f"Left key '{left_key}' not found in {path_analysis}")
    if right_key not in df_b.columns:
        raise KeyError(f"Right key '{right_key}' not found in {path_features}")

    # Preserve original analysis order by adding a dedicated order column.
    # Reset the index to ensure a dense 0..N-1 ordering, then set a unique temporary column.
    df_a = df_a.reset_index(drop=True)
    order_col = '__merge_order__'
    df_a[order_col] = df_a.index

    # Merge, prefer values from features (B) when overwrite=True
    if overwrite:
        tmp_suffix = '_tmp_b'
        merged = pd.merge(df_a, df_b, left_on=left_key, right_on=right_key, how=how, suffixes=("", tmp_suffix))
        # For overlapping non-key columns, copy values from suffixed B into the original name
        for c in df_b.columns:
            if c == right_key:
                continue
            suff = c + tmp_suffix
            if suff in merged.columns:
                merged[c] = merged[suff]
                merged.drop(columns=[suff], inplace=True)
    else:
        merged = pd.merge(df_a, df_b, left_on=left_key, right_on=right_key, how=how, suffixes=("", "_b"))

    # Restore analysis original order when possible, then remove the temporary column safely
    if order_col in merged.columns:
        try:
            merged.sort_values(order_col, inplace=True)
        except Exception:
            # If sorting fails for any reason, proceed without reordering
            pass
        # Safely drop the temporary order column if present
        merged.drop(columns=[order_col], errors='ignore', inplace=True)

    # Create the 'shape' column: prefer the features filename (right_key) if available
    if right_key in merged.columns:
        merged['shape'] = merged[right_key]
    elif left_key in merged.columns:
        merged['shape'] = merged[left_key]
    else:
        merged['shape'] = ''

    # Determine class: prefer features' class (from df_b) if present, otherwise take from df_a
    class_from_b = 'class' if 'class' in df_b.columns else None
    class_from_a = 'class' if 'class' in df_a.columns else None
    if class_from_b and class_from_b in merged.columns:
        merged['class'] = merged[class_from_b]
    elif class_from_a and class_from_a in merged.columns:
        merged['class'] = merged[class_from_a]
    else:
        merged['class'] = ''

    # Build output column order
    cols_a = list(df_a.columns)
    cols_b = list(df_b.columns)

    # Exclude the left_key, class and the temporary order column from the analysis columns
    analysis_cols = [c for c in cols_a if c not in (left_key, 'class', order_col)]
    feature_cols = [c for c in cols_b if c not in (right_key, 'class') and c not in cols_a]

    out_cols = ['shape', 'class'] + analysis_cols + feature_cols

    # Safety: append any remaining columns
    for c in merged.columns:
        if c not in out_cols:
            out_cols.append(c)

    # Filter out any columns that are not present in the merged DataFrame (safe-write)
    out_cols = [c for c in out_cols if c in merged.columns]

    merged.to_csv(out, columns=out_cols, index=False)


def _parse_args(argv: Optional[list] = None):
    """Parse command line arguments.

    Parameters:
        argv (Optional[list]): Command line arguments list.

    Returns:
        argparse.Namespace: Parsed arguments.

    """
    p = argparse.ArgumentParser(description="Merge two CSV files on a key and preserve column ordering.")
    p.add_argument('file_a', type=Path, help='First CSV file (its columns appear first)')
    p.add_argument('file_b', type=Path, help='Second CSV file (its non-key columns are appended)')
    p.add_argument('-o', '--out', type=Path, default=Path('merged.csv'), help='Output CSV path')
    p.add_argument('--key', type=str, default='name', help='Primary key column name (default: name)')
    p.add_argument('--how', type=str, choices=['inner', 'left', 'right', 'outer'], default='outer', help='Join type (default: outer)')
    p.add_argument('--suffix', type=str, default='_b', help='Suffix to append to overlapping non-key columns from second file (default: _b)')
    p.add_argument('--overwrite', action='store_true', help='If set, values from file B overwrite same-named columns from file A (no suffix)')
    p.add_argument('--analysis-merge', action='store_true', help='Specialized merge: analysis_results (file_a) with final_features (file_b)')
    p.add_argument('--left-key', type=str, default='shape_file', help='Key column in analysis file (default: shape_file)')
    p.add_argument('--right-key', type=str, default='name', help='Key column in features file (default: name)')
    p.add_argument('--version-suffix', type=str, default='_06_fill_holes_and_orientation.obj', help='Only keep analysis rows whose filename ends with this suffix (default: _06_fill_holes_and_orientation.obj)')
    return p.parse_args(argv)


def main(argv: Optional[list] = None):
    """Main entry point for the merge script.

    Parameters:
        argv (Optional[list]): Command line arguments list.

    Returns:
        int: Exit code (0 for success, 2 for error).

    """
    args = _parse_args(argv)
    try:
        if args.analysis_merge:
            merge_analysis_with_features(
                args.file_a,
                args.file_b,
                args.out,
                left_key=args.left_key,
                right_key=args.right_key,
                how=args.how,
                overwrite=args.overwrite,
                version_suffix=args.version_suffix,
            )
        else:
            merge_csvs(args.file_a, args.file_b, args.out, key=args.key, how=args.how, suffix=args.suffix, overwrite=args.overwrite)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
