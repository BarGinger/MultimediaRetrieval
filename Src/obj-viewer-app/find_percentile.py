"""
find_percentile.py

Compute a global percentile (e.g., 95th or 99th) for each descriptor (A3, D1..D4)
based on the per-shape histograms stored in output/descriptors_all_histograms.csv.

This script reconstructs the empirical distribution by summing normalized histograms
(per-shape histograms are treated as probability mass over their bins). It then
computes the percentile cutoff for each descriptor and writes a single-row CSV:

    output/descriptors_global_percentiles_{p}.csv

Columns: name, percentile_A3, percentile_D1, ...

Usage:
    python Src/obj-viewer-app/find_percentile.py --csv output/descriptors_all_histograms.csv --p 0.99

"""
from __future__ import annotations
import csv
from pathlib import Path
import argparse
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_DEFAULT = REPO_ROOT / "output" / "descriptors_all_histograms.csv"
OUT_DIR = REPO_ROOT / "output"
DESCRIPTORS = ["A3", "D1", "D2", "D3", "D4"]


def parse_semicolon_floats(s: str):
    if s is None or s == "":
        return np.array([])
    parts = [p.strip() for p in s.split(';') if p.strip() != '']
    return np.array([float(p) for p in parts], dtype=float)


def load_rows(csv_path: Path):
    rows = []
    with csv_path.open('r', newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            entry = {'name': r.get('name')}
            for desc in DESCRIPTORS:
                entry[f'{desc}_hist'] = parse_semicolon_floats(r.get(f'{desc}_hist', ''))
                entry[f'{desc}_bins'] = parse_semicolon_floats(r.get(f'{desc}_bins', ''))
            rows.append(entry)
    return rows


def bin_centers_from_bins(bins: np.ndarray, hist: np.ndarray):
    # If bins represent edges (len = len(hist)+1) compute centers, else use as-is
    if bins is None or len(bins) == 0:
        return np.array([])
    if len(bins) == len(hist) + 1:
        return (bins[:-1] + bins[1:]) / 2.0
    return bins


def compute_percentiles(rows, percentile: float):
    """Return dict descriptor->value for the percentile.

    Approach: for each descriptor, build a combined empirical distribution by
    summing per-shape histograms (normalize each shape's hist to sum=1 to treat as pmf),
    mapping to a common set of bin centers if needed. To avoid expensive resampling,
    we compute percentile by concatenating (hist mass, centers) across shapes then
    sorting by center — equivalent to merging distributions on the value axis.

    This assumes histograms are comparable (same bin centers or overlapping).
    """
    result = {}
    for desc in DESCRIPTORS:
        masses = []
        centers = []
        for r in rows:
            hist = r.get(f"{desc}_hist")
            bins = r.get(f"{desc}_bins")
            if hist is None or len(hist) == 0:
                continue
            c = bin_centers_from_bins(bins, hist)
            if c is None or len(c) == 0:
                continue
            h = np.asarray(hist, dtype=float)
            # normalize shape histogram to sum to 1 (if already normalized, this is a no-op)
            s = h.sum()
            if s <= 0:
                continue
            h = h / s
            masses.append(h)
            centers.append(c)
        if len(masses) == 0:
            result[desc] = float('nan')
            continue
        # Concatenate into a single value-mass list and sort by value
        all_centers = np.concatenate(centers)
        all_masses = np.concatenate(masses)
        # sort by center
        order = np.argsort(all_centers)
        all_centers = all_centers[order]
        all_masses = all_masses[order]
        # normalize total mass so cumulative runs from 0..1 (percentile expects fraction)
        total_mass = float(all_masses.sum())
        if total_mass <= 0.0:
            result[desc] = float('nan')
            continue
        all_masses = all_masses / total_mass
        # cumulative mass
        cum = np.cumsum(all_masses)
        # find first index where cum >= percentile
        idx = np.searchsorted(cum, percentile, side='left')
        if idx >= len(all_centers):
            # return the largest center if percentile at 1.0
            val = float(all_centers[-1])
        else:
            # interpolate between centers for a smoother cutoff
            if idx == 0:
                val = float(all_centers[0])
            else:
                c0, c1 = float(all_centers[idx - 1]), float(all_centers[idx])
                m0, m1 = float(cum[idx - 1]), float(cum[idx])
                if m1 == m0:
                    val = c1
                else:
                    frac = (percentile - m0) / (m1 - m0)
                    val = c0 + frac * (c1 - c0)
        result[desc] = val
    return result


def write_percentile_csv(p: float, percentiles: dict, out_dir: Path):
    p_int = int(round(p * 100))
    out_path = out_dir / f"descriptors_global_percentiles_{p_int}.csv"
    fieldnames = ['name'] + [f'percentile_{d}' for d in DESCRIPTORS]
    row = {'name': f'percentile_{p_int}'}
    for d in DESCRIPTORS:
        v = percentiles.get(d)
        row[f'percentile_{d}'] = f"{v:.6f}" if (v is not None and not np.isnan(v)) else ''
    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    print(f"Wrote percentiles to: {out_path}")
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path, default=CSV_DEFAULT)
    parser.add_argument('--p', type=float, default=0.99, help='percentile as fraction (0-1)')
    args = parser.parse_args()

    rows = load_rows(args.csv)
    percentiles = compute_percentiles(rows, args.p)
    out = write_percentile_csv(args.p, percentiles, OUT_DIR)
    print(percentiles)
