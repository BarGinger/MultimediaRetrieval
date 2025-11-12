"""
File: scripts/rename_suffix_recursive.py
Last modified: 01-11-2025

Simple renamer for dataset folders.

Given a root folder that contains category subfolders, scan each immediate
subfolder for files whose names end with the source suffix and rename them by
replacing that suffix with the destination suffix.

Usage:
  python rename_suffix_recursive.py <root_folder> [--src SRC] [--dst DST] [--dry-run] [--force]

Defaults:
  src: _unified_prepared.obj
  dst: _06_fill_holes_and_orientation.obj
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys


def process_folder(root: Path, src_suffix: str, dst_suffix: str, dry_run: bool, force: bool) -> int:
    """
    Scan immediate subdirectories of root and rename matching files.
    
    Parameters:
        root (Path): Root folder containing category subfolders.
        src_suffix (str): Source suffix to match.
        dst_suffix (str): Destination suffix to use.
        dry_run (bool): If True, only print what would be done.
        force (bool): If True, overwrite target files if they exist.

    Returns:
        int: Number of files renamed (or that would be renamed in dry-run).
    """
    if not root.exists():
        raise FileNotFoundError(f"Root path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root}")

    renamed = 0

    # First scan files directly under root (in case files are not grouped)
    for path in root.glob(f"*{src_suffix}"):
        if not path.is_file():
            continue
        new_name = path.name[: -len(src_suffix)] + dst_suffix
        new_path = path.with_name(new_name)
        if new_path.exists() and not force:
            logging.warning("Skipping because target exists: %s -> %s", path, new_path)
            continue
        if dry_run:
            print(f"DRY RUN: {path} -> {new_path}")
            renamed += 1
            continue
        if new_path.exists() and force:
            new_path.unlink()
        path.rename(new_path)
        logging.info("Renamed: %s -> %s", path, new_path)
        renamed += 1

    # Then scan each immediate subdirectory
    for child in root.iterdir():
        if not child.is_dir():
            continue
        for path in child.glob(f"*{src_suffix}"):
            if not path.is_file():
                continue
            new_name = path.name[: -len(src_suffix)] + dst_suffix
            new_path = path.with_name(new_name)
            if new_path.exists() and not force:
                logging.warning("Skipping because target exists: %s -> %s", path, new_path)
                continue
            if dry_run:
                print(f"DRY RUN: {path} -> {new_path}")
                renamed += 1
                continue
            if new_path.exists() and force:
                new_path.unlink()
            path.rename(new_path)
            logging.info("Renamed: %s -> %s", path, new_path)
            renamed += 1

    return renamed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rename files in immediate subfolders of a dataset root.")
    parser.add_argument("root", nargs="?", default="Datasets/UnifiedPreprocessed/Data", help="Root folder containing category subfolders")
    parser.add_argument("--src", default="_unified_prepared.obj", help="Source suffix to match")
    parser.add_argument("--dst", default="_06_fill_holes_and_orientation.obj", help="Destination suffix to use")
    parser.add_argument("--dry-run", default=False, action="store_true", help="Don't perform changes, only print what would be done")
    parser.add_argument("--force", action="store_true", help="Overwrite target files if they exist")
    parser.add_argument("--verbose", "-v", action="count", default=0, help="Increase verbosity")

    args = parser.parse_args(argv)

    log_level = logging.WARNING
    if args.verbose >= 1:
        log_level = logging.INFO
    if args.verbose >= 2:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    root = Path(args.root)

    try:
        count = process_folder(root, args.src, args.dst, dry_run=args.dry_run, force=args.force)
    except (FileNotFoundError, NotADirectoryError) as e:
        logging.error(e)
        return 2

    if args.dry_run:
        print(f"Dry-run found {count} files that would be renamed.")
    else:
        print(f"Renamed {count} files.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
