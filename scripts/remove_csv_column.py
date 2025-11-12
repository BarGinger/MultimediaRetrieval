"""
File: scripts/remove_csv_column.py
Last modified: 01-11-2025
Utility: remove a column from a CSV quickly.

Usage:
    python scripts\remove_csv_column.py path/to/file.csv column_name [--out out.csv] [--inplace]

If --inplace is set, the original file will be backed up to file.csv.bak and the new file will overwrite the original.
Requires pandas.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd


def remove_column(path: Path, column: str, out: Path | None = None, inplace: bool = False) -> int:
    """
    Remove a column from a CSV file.

    Parameters:
        path (Path): Path to the input CSV file.
        column (str): Name of the column to remove.
        out (Path | None): Path to output CSV file (if not inplace).
        inplace (bool): Whether to overwrite the original file (with backup).
    Returns:
        int: 0 on success, 2 if file not found.
    """
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 2
    df = pd.read_csv(path)
    if column not in df.columns:
        print(f"Column '{column}' not found in {path}. No changes made.")
        return 0
    df = df.drop(columns=[column])
    if inplace:
        bak = path.with_suffix(path.suffix + '.bak')
        path.replace(bak)
        df.to_csv(path, index=False)
        print(f"Dropped column '{column}' and overwrote {path} (backup at {bak}).")
    else:
        if out is None:
            out = path.with_name(path.stem + f"_no_{column}" + path.suffix)
        df.to_csv(out, index=False)
        print(f"Dropped column '{column}' and wrote output to {out}.")
    return 0


def _parse_args(argv=None):
    """
    Parse command line arguments.
    
    Parameters:
        argv (list[str] | None): List of command line arguments (or None to use sys.argv).

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    p = argparse.ArgumentParser(description="Remove a column from a CSV quickly.")
    p.add_argument('file', type=Path, help='CSV file path')
    p.add_argument('column', type=str, help='Column name to remove')
    p.add_argument('--out', type=Path, default=None, help='Output path (when not using --inplace)')
    p.add_argument('--inplace', action='store_true', help='Overwrite the original file (backup created)')
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    return remove_column(args.file, args.column, out=args.out, inplace=args.inplace)


if __name__ == '__main__':
    raise SystemExit(main())
