"""
descriptor_plots.py

Read output/descriptors_all_histograms.csv and produce comparison grids per descriptor.

Outputs (one PNG per descriptor) are written to:
    output/descriptor_histogram_comparisons/

Behavior:
- Load CSV where each row contains name and semicolon-separated hist and bins for descriptors A3, D1..D4.
- Group shapes by class (derived from the path in `name` column, e.g. "Datasets/UnifiedPreprocessed/Data/Apartment/xxx.obj" -> class "Apartment").
- Select ~20 classes that have similar number of shapes. Strategy: compute class sizes and pick 20 classes closest to the median class size.
- For each descriptor (A3, D1, D2, D3, D4) create an MxN grid of small subplots where each subplot shows all shapes in that class as line plots (histogram values vs bin centers). Use semi-transparent colored lines so many shapes can be compared.
- Save PNGs as: output/descriptor_histogram_comparisons/{descriptor}_class_comparison.png

Usage:
    python -m Src.obj-viewer-app.descriptor_plots --csv output/descriptors_all_histograms.csv --out output/descriptor_histogram_comparisons --classes 20

"""
from __future__ import annotations
import csv
import os
import math
from pathlib import Path
from collections import defaultdict
import argparse

import numpy as np
import matplotlib
# Use Agg backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:
    # fallback noop
    def tqdm(x, **kw):
        return x

CSV_DEFAULT = Path("output/descriptors_all_histograms.csv")
OUT_DIR_DEFAULT = Path("output/descriptor_histogram_comparisons")
# DESCRIPTORS = ["A3", "D1", "D2", "D3", "D4"]
DESCRIPTORS = ["D1"]

# Human-readable note for descriptors that were transformed to length-like units
# D3: sqrt(area) -> length, D4: cbrt(volume) -> length
TRANSFORMS = {
    "D3": "sqrt(area)",
    "D4": "cbrt(volume)",
}


def load_percentile_99(csv_path: Path):
    """Load percentiles from output/descriptors_global_percentiles_99.csv.

    Returns dict descriptor->float or empty dict if file not found or invalid.
    """
    if not csv_path.exists():
        return {}
    out = {}
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            row = next(reader, None)
            if row is None:
                return {}
            for d in DESCRIPTORS:
                key = f"percentile_{d}"
                v = row.get(key, "")
                try:
                    out[d] = float(v)
                except Exception:
                    pass
    except Exception:
        return {}
    return out


def parse_semicolon_floats(s: str) -> np.ndarray:
    if s is None or s == "":
        return np.array([])
    parts = s.split(";")
    try:
        return np.array([float(p) for p in parts])
    except Exception:
        # tolerate stray spaces
        return np.array([float(p.strip()) for p in parts if p.strip() != ""])


def load_histograms(csv_path: Path):
    """Return list of rows: dict with keys: name, and for each descriptor hist and bins arrays"""
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            entry = {"name": r.get("name")}
            for desc in DESCRIPTORS:
                hist_key = f"{desc}_hist"
                bins_key = f"{desc}_bins"
                entry[f"{desc}_hist"] = parse_semicolon_floats(r.get(hist_key, ""))
                entry[f"{desc}_bins"] = parse_semicolon_floats(r.get(bins_key, ""))
            rows.append(entry)
    return rows


def class_from_name(name: str) -> str:
    # Expect path like Datasets/UnifiedPreprocessed/Data/<Class>/<file>
    parts = Path(name).parts
    # find 'Data' then take the next path component as class if present
    for i, p in enumerate(parts):
        if p.lower() == "data" and i + 1 < len(parts):
            return parts[i + 1]
    # fallback: take parent folder name
    return Path(name).parent.name


def select_classes_by_size(rows, n_classes=20):
    classes = defaultdict(list)
    for r in rows:
        cls = class_from_name(r["name"]) or "<unknown>"
        classes[cls].append(r)
    # compute sizes
    sizes = {c: len(v) for c, v in classes.items()}
    if len(sizes) <= n_classes:
        return classes
    # find median size
    all_sizes = np.array(list(sizes.values()))
    median = int(np.median(all_sizes))
    # sort classes by abs(size - median)
    sorted_classes = sorted(sizes.items(), key=lambda it: (abs(it[1] - median), -it[1]))
    chosen = sorted_classes[:n_classes]
    result = {c: classes[c] for c, _ in chosen}
    return result


def plot_descriptor_grid(descriptor: str, class_map: dict, out_dir: Path, per_row=5):
    """Create a grid of plots, one subplot per class. Each subplot shows all shapes in that class for the given descriptor."""
    n_classes = len(class_map)
    if n_classes == 0:
        return
    ncols = per_row
    nrows = math.ceil(n_classes / ncols)
    fig_w = ncols * 3
    fig_h = nrows * 2.5
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    classes = list(class_map.items())
    cmap = plt.get_cmap("tab20")

    # Attempt to load global percentile (99) and compute 10% buffered upper bound
    repo_root = Path(__file__).resolve().parents[2]
    perc_csv = repo_root / "output" / "descriptors_global_percentiles_99.csv"
    # perc_csv = repo_root / "output" / "descriptors_custom_range.csv"
    percentiles = load_percentile_99(perc_csv)
    buffered_upper = {}
    for d in DESCRIPTORS:
        pv = percentiles.get(d)
        if pv is not None:
            buffered_upper[d] = pv * 1.10
        else:
            buffered_upper[d] = None

    for idx, (cls, items) in enumerate(classes):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]
        # plot each shape's histogram as a line using bin centers
        max_len = 0
        # find representative upper_bound for this descriptor/class
        rep_upper = buffered_upper.get(descriptor)
        if rep_upper is None:
            # fallback: look for first item's bins[-1]
            for item in items:
                bins = item.get(f"{descriptor}_bins")
                if bins is not None and len(bins) > 0:
                    try:
                        rep_upper = float(bins[-1])
                        break
                    except Exception:
                        rep_upper = None
        for j, item in enumerate(items):
            hist = item.get(f"{descriptor}_hist")
            bins = item.get(f"{descriptor}_bins")
            if hist is None or bins is None or len(hist) == 0 or len(bins) == 0:
                continue
            # if number of bins == number of hist + 1 (edges), convert to centers
            if len(bins) == len(hist) + 1:
                centers = (bins[:-1] + bins[1:]) / 2.0
            else:
                centers = bins
            max_len = max(max_len, len(centers))
            color = cmap(j % 20)
            ax.plot(centers, hist, color=color, alpha=0.6, linewidth=0.8)
        # draw vertical dashed line for upper bound and annotate overflow
        if rep_upper is not None:
            # vertical line at upper bound in data coordinates
            ax.axvline(x=rep_upper, color='k', linestyle='--', linewidth=0.7, alpha=0.7)
            # annotate in axes coords (upper-right)
            try:
                transform_note = TRANSFORMS.get(descriptor, None)
                if transform_note:
                    label = f"last bin = overflow ≥ {rep_upper:.6g} ({transform_note})"
                else:
                    label = f"last bin = overflow ≥ {rep_upper:.6g}"
            except Exception:
                label = "last bin = overflow"
            ax.text(0.98, 0.95, label, transform=ax.transAxes, ha='right', va='top', fontsize=6, color='k', alpha=0.8)
        ax.set_title(f"{cls} ({len(items)})", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_xlim(left=0)
    # hide empty subplots
    for k in range(n_classes, nrows * ncols):
        r = k // ncols
        c = k % ncols
        axes[r][c].axis('off')

    transform_note = TRANSFORMS.get(descriptor)
    title = f"Descriptor {descriptor} — per-class distributions (each line = shape)"
    if transform_note:
        title = f"{title} — values: {transform_note}"
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{descriptor}_class_comparison.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main(csv_path: Path, out_dir: Path, n_classes: int):
    rows = load_histograms(csv_path)
    class_map = select_classes_by_size(rows, n_classes=n_classes)
    print(f"Selected {len(class_map)} classes for plotting")
    for desc in DESCRIPTORS:
        print(f"Plotting {desc}...")
        plot_descriptor_grid(desc, class_map, out_dir)
    print("Done. Files written to", out_dir)


if __name__ == "__main__":
    # python -m Src.obj-viewer-app.descriptor_plots --input output/descriptors_all_histograms.csv --out output/descriptor_histogram_comparisons
    parser = argparse.ArgumentParser(description="Plot descriptor histogram comparisons per class")
    parser.add_argument("--csv", type=Path, default=CSV_DEFAULT, help="path to descriptors_all_histograms.csv")
    parser.add_argument("--out", type=Path, default=OUT_DIR_DEFAULT, help="output directory for PNGs")
    parser.add_argument("--classes", type=int, default=20, help="number of classes to select and plot")
    args = parser.parse_args()
    main(args.csv, args.out, args.classes)
