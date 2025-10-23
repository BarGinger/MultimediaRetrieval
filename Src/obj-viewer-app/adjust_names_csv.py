r"""adjust_names_csv.py

Normalize the `name` column in CSV files by either stripping a number of leading path components
or by replacing a filename suffix in the basename. This is useful when one CSV stores full paths
like:
    Datasets\UnifiedPreprocessed\Data\ClassName\file_unified_prepared.obj
and you want to normalize them to:
    ClassName\file_06_fill_holes_and_orientation.obj

Two modes are supported:
- default: strip the first N path components from the `name` value (see --strip)
- suffix-replace: provide two strings OLD NEW via --suffix-replace OLD NEW and the script will
  replace the trailing OLD on the file stem with NEW (extension preserved).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import csv
import sys
import os
from typing import Optional
import re


def strip_leading_components(s: str, n: int) -> str:
    if s is None:
        return s
    # Support both forward and backward slashes; normalize to OS sep, then split
    s_norm = s.replace('/', os.path.sep).replace('\\', os.path.sep)
    parts = s_norm.split(os.path.sep)
    if len(parts) <= n:
        return os.path.sep.join(parts[-1:]) if parts else s
    return os.path.sep.join(parts[n:])


def adjust_csv(input_csv: Path, output_csv: Path, key: str = 'name', strip: int = 3, inplace: bool = False, suffix_replace: tuple | None = None, split_class: bool = False) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    if inplace:
        output_csv = input_csv

    rows = []
    with input_csv.open('r', newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        if key not in reader.fieldnames:
            raise KeyError(f"Key column '{key}' not found in {input_csv}")
        fieldnames = list(reader.fieldnames)
        # If requested, ensure a 'class' column exists
        if split_class and 'class' not in fieldnames:
            fieldnames.append('class')
        for r in reader:
            val = r.get(key, '')
            if suffix_replace:
                old, new = suffix_replace
                # Parse path, replace trailing old on the stem if present
                # Support both slashes
                norm = val.replace('/', os.path.sep).replace('\\', os.path.sep)
                parts = norm.split(os.path.sep)
                if parts:
                    stem = Path(parts[-1]).stem
                    suffix = Path(parts[-1]).suffix
                    if stem.endswith(old):
                        stem = stem[: -len(old)] + new
                    parts[-1] = stem + suffix
                    new_path = os.path.sep.join(parts)
                    r[key] = new_path
                else:
                    r[key] = val
            else:
                r[key] = strip_leading_components(val, strip)
            # Optionally split class and filename into separate columns
            if split_class:
                # Normalize separators and split
                norm = r[key].replace('/', os.path.sep).replace('\\', os.path.sep)
                parts = norm.split(os.path.sep)
                if len(parts) >= 2:
                    class_raw = parts[-2]
                    filename = parts[-1]
                elif parts:
                    class_raw = ''
                    filename = parts[-1]
                else:
                    class_raw = ''
                    filename = r[key]

                r[key] = filename
                r['class'] = class_raw
            rows.append(r)

    # If split_class was requested, ensure the column order is: key, class, <other original columns>
    if split_class:
        # preserve original field order but force key then class
        other_cols = [c for c in fieldnames if c not in (key, 'class')]
        ordered_fieldnames = [key, 'class'] + other_cols
    else:
        ordered_fieldnames = fieldnames

    with output_csv.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=ordered_fieldnames)
        writer.writeheader()
        for r in rows:
            # Ensure all keys exist in the row dict to avoid KeyError
            out_row = {c: r.get(c, '') for c in ordered_fieldnames}
            writer.writerow(out_row)


def _parse_args(argv: Optional[list] = None):
    p = argparse.ArgumentParser(description='Strip leading path components from `name` column in a CSV')
    p.add_argument('input_csv', type=Path, help='Input CSV path')
    p.add_argument('output_csv', type=Path, nargs='?', help='Output CSV path (defaults to input with .fixed.csv)')
    p.add_argument('--strip', type=int, default=3, help='Number of leading components to remove (default: 3)')
    p.add_argument('--key', type=str, default='name', help='Column name containing the path (default: name)')
    p.add_argument('--inplace', action='store_true', help='Modify the input file in-place')
    p.add_argument('--suffix-replace', nargs=2, metavar=('OLD', 'NEW'), help='Replace trailing OLD suffix on file stem with NEW (preserve extension). If supplied, this mode runs instead of --strip')
    p.add_argument('--split-class', action='store_true', help='Split the path in `name` into filename and class (parent folder)')
    return p.parse_args(argv)


def main(argv: Optional[list] = None):
    args = _parse_args(argv)
    inp = args.input_csv
    out = args.output_csv if args.output_csv is not None else inp.with_suffix('.fixed.csv')
    try:
        adjust_csv(
            inp,
            out,
            key=args.key,
            strip=args.strip,
            inplace=args.inplace,
            suffix_replace=tuple(args.suffix_replace) if args.suffix_replace else None,
            split_class=args.split_class,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
