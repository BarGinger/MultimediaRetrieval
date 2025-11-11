#%%
# -*- coding: utf-8 -*-
# ============================================================
# Min→Max previews + boxplots + closest-items table (5 targets)
# ============================================================

# -------- CONFIG --------
CSV_PATH    = "analysis_results_unifiedPreprocessed_data.csv"
IMAGE_DIR   = "."                 # images live in working dir; or set to "images"
SAVE_DIR    = "preview_scales_out"
ID_COL      = "shape_file"        # column that identifies the image/item
CLASS_COL   = "class"             # class/label column
SHOW_FLIERS = True                # set True to show outliers on boxplots
WHIS        = (0, 100)            # whiskers min–max; use 1.5 to use Tukey rule
THUMB_ZOOM  = 0.08                # thumbnail size on the scales
IMG_EXTS    = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
JITTER_Y    = [0.0, 0.14, -0.14, 0.26, -0.26]   # offsets to reduce overlap

# Verbose logging while resolving images (set False once it all works)
VERBOSE_IO  = True
# ------------------------

import os
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

os.makedirs(SAVE_DIR, exist_ok=True)

def _read_df(csv_path: str, id_col: str, class_col: str) -> pd.DataFrame:
    """Load DataFrame with a fallback if the first column is an index."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    # 1) Try normal read
    df = pd.read_csv(csv_path)
    # remove the row where filename = m1097_unified_prepared.png
    df.drop(df[df[id_col] == "m1097_unified_prepared.png"].index, inplace=True)
    if id_col in df.columns and (class_col in df.columns or class_col is None):
        return df
    # 2) Fallback: treat first column as index
    df2 = pd.read_csv(csv_path, header=0, index_col=0)
    if id_col in df2.columns or class_col in df2.columns:
        return df2
    # 3) Give up with helpful message
    raise ValueError(
        f"Could not find '{id_col}' (and/or '{class_col}') in CSV even after fallback. "
        f"Columns present: {df.columns.tolist()}"
    )

def read_img_any(path: Path) -> np.ndarray:
    """Try matplotlib first, then PIL (converts to RGBA)."""
    try:
        return mpimg.imread(str(path))
    except Exception:
        return np.array(Image.open(str(path)).convert("RGBA"))

def load_image_verbose(name_or_path, image_dir="."):
    """
    Robust resolver:
    - strip whitespace
    - use basename only (ignore any folder parts found in the CSV)
    - if suffix not in IMG_EXTS, replace with common image suffixes (stem + .jpg/.png/...)
    - try case-insensitive glob stem.* in IMAGE_DIR as a last resort
    Returns (ndarray, resolved_path) or (None, [attempts...])
    """
    raw = str(name_or_path).strip()
    p = Path(raw)
    stem = p.stem
    base = p.name
    attempts = []

    def _log(*a):
        if VERBOSE_IO:
            print(*a)

    # 0) If CSV already has an image extension, try directly with basename in IMAGE_DIR
    if Path(base).suffix.lower() in IMG_EXTS:
        cand = Path(image_dir) / base
        attempts.append(str(cand))
        _log("[try]", cand)
        if cand.exists():
            try:
                img = read_img_any(cand)
                _log("[ok]  ", cand, getattr(img, "shape", None))
                return img, str(cand)
            except Exception as e:
                _log("[read-error]", cand, "->", e)

    # 1) Replace non-image suffix with common image suffixes in IMAGE_DIR (basename only)
    for ext in IMG_EXTS:
        cand = Path(image_dir) / f"{stem}{ext}"
        attempts.append(str(cand))
        _log("[try]", cand)
        if cand.exists():
            try:
                img = read_img_any(cand)
                _log("[ok]  ", cand, getattr(img, "shape", None))
                return img, str(cand)
            except Exception as e:
                _log("[read-error]", cand, "->", e)

    # 2) Case-insensitive glob search for stem.*
    glob_pattern = str((Path(image_dir) / f"{stem}.*")).replace("\\", "/")
    matches = glob.glob(glob_pattern)
    if matches:
        for m in matches:
            attempts.append(m)
        # prefer known image extensions first
        preferred = [m for m in matches if Path(m).suffix.lower() in IMG_EXTS]
        pick = preferred[0] if preferred else matches[0]
        try:
            img = read_img_any(Path(pick))
            _log("[ok(glob)]", pick)
            return img, pick
        except Exception as e:
            _log("[read-error(glob)]", pick, "->", e)

    _log(f"[miss] for {name_or_path} ; tried:", attempts)
    return None, attempts

def numeric_columns(df: pd.DataFrame, id_col: str, class_col: str):
    bad_prefixes = ("unnamed",)
    bad_exact = {"index"}
    cols = []
    for c in df.select_dtypes(include="number").columns:
        cl = c.lower()
        if c in (id_col, class_col):
            continue
        if cl.startswith(bad_prefixes) or cl in bad_exact:
            continue
        cols.append(c)
    if not cols:
        raise ValueError("No numeric columns found (besides ID/CLASS).")
    return cols

def make_boxplots(df: pd.DataFrame, class_col: str, value_cols, show_fliers=False, whis=(0,100)):
    """Boxplots per column, grouped by class, classes sorted by median."""
    if class_col not in df.columns:
        print(f"[warn] '{class_col}' not found — skipping boxplots.")
        return
    for col in value_cols:
        medians = df.groupby(class_col)[col].median().sort_values()
        sorted_classes = medians.index.tolist()
        data = [df.loc[df[class_col] == cl, col].dropna().values for cl in sorted_classes]

        plt.figure(figsize=(8, max(1.8, len(sorted_classes) * 0.25)))
        plt.boxplot(
            data,
            labels=[str(cl) for cl in sorted_classes],
            showfliers=show_fliers,
            vert=False,
            whis=whis,
            flierprops={"markersize": 3, "marker": "o", "alpha": 0.6},
        )
        plt.title(f"{col} by {class_col} (sorted by median)")
        plt.ylabel(class_col)
        plt.xlabel(col)
        plt.tight_layout()
        plt.show()

def compute_closest_table(df: pd.DataFrame, id_col: str, class_col: str, value_cols):
    """Compute closest rows to 5 evenly spaced targets per numeric column."""
    rows = []
    for col in value_cols:
        s = df[col]
        valid_mask = s.notna().to_numpy()
        if not valid_mask.any():
            continue
        values = s.to_numpy()
        valid_values = values[valid_mask]
        valid_indices = np.nonzero(valid_mask)[0]

        col_min = float(np.nanmin(values))
        col_max = float(np.nanmax(values))
        targets = np.linspace(col_min, col_max, 5)
        labels = ["min", "25%", "50%", "75%", "max"]

        for label, t in zip(labels, targets):
            diffs = np.abs(valid_values - t)
            j = int(np.argmin(diffs))
            idx = int(valid_indices[j])
            rows.append({
                "column": col,
                "target_label": label,
                "target_value": float(t),
                "closest_filename": str(df.at[idx, id_col]) if id_col in df.columns else None,
                "closest_class": df.at[idx, class_col] if class_col in df.columns else None,
                "closest_value": float(df.at[idx, col]),
                "abs_diff": float(abs(df.at[idx, col] - t)),
            })

    closest_df = pd.DataFrame(rows, columns=[
        "column", "target_label", "target_value",
        "closest_filename", "closest_class", "closest_value", "abs_diff"
    ]).sort_values(["column", "target_value"]).reset_index(drop=True)

    closest_df.to_csv("closest_points_summary.csv", index=False)
    print("[saved] closest_points_summary.csv")
    return closest_df

def print_closest_tables(closest_df: pd.DataFrame):
    """
    Pretty-print, per numeric column, the filenames and values that are
    closest to 0/25/50/75/100% (min..max).
    """
    label_map = {"min": "0%", "25%": "25%", "50%": "50%", "75%": "75%", "max": "100%"}
    order = ["0%", "25%", "50%", "75%", "100%"]

    with pd.option_context("display.float_format", lambda v: f"{v:.4g}"):
        for col, sub in closest_df.groupby("column", sort=False):
            view = (
                sub.assign(pct=sub["target_label"].map(label_map))
                   .sort_values("target_value")
                   .set_index("pct")[["closest_filename", "closest_value"]]
                   .reindex(order)
            )
            print(f"\n=== {col} ===")
            print(view.to_string(na_rep=""))

def draw_preview_scales(df: pd.DataFrame, id_col: str, class_col: str, value_cols):
    """Draw one min→max preview scale per numeric column with 5 thumbnails."""
    for col in value_cols:
        s = df[col]
        valid_mask = s.notna().to_numpy()
        if not valid_mask.any():
            print(f"[skip] {col}: all NaN")
            continue

        values = s.to_numpy()
        valid_vals = values[valid_mask]
        valid_idx  = np.nonzero(valid_mask)[0]

        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        rng  = vmax - vmin
        pad  = 0.02 * (rng if rng != 0 else 1.0)
        targets = np.linspace(vmin, vmax, 5)
        labels  = ["min", "25%", "50%", "75%", "max"]

        # choose closest rows
        picks = []
        for label, t in zip(labels, targets):
            diffs = np.abs(valid_vals - t)
            j = int(np.argmin(diffs))
            idx = int(valid_idx[j])
            picks.append({
                "label": label,
                "target": float(t),
                "idx": idx,
                "x": float(df.at[idx, col]),
                "filename": str(df.at[idx, id_col]),
                "klass": df.at[idx, class_col] if class_col in df.columns else None
            })

        # optional sanity print of what exists on disk for each stem
        if VERBOSE_IO:
            for q in picks:
                stem = Path(q["filename"].strip()).stem
                found = glob.glob(str((Path(IMAGE_DIR) / f"{stem}.*")).replace("\\", "/"))
                print(f"[stem-check] {stem} -> {found}")

        # figure
        fig, ax = plt.subplots(figsize=(10, 3.0), constrained_layout=True)
        ax.set_xlim(vmin - pad, vmax + pad)
        ax.set_ylim(-0.7, 0.7)
        ax.axhline(0, lw=2, alpha=0.3, zorder=1)

        xticks = np.linspace(vmin, vmax, 5)
        ax.set_xticks(xticks)
        ax.set_yticks([])
        ax.set_title(f"{col}: representative items across the range")

        drawn = 0
        for i, p in enumerate(picks):
            y = JITTER_Y[i % len(JITTER_Y)]
            img, resolved = load_image_verbose(p["filename"], IMAGE_DIR)

            if isinstance(img, np.ndarray):
                im = OffsetImage(img, zoom=THUMB_ZOOM, resample=True)
                ab = AnnotationBbox(
                    im, (p["x"], y),
                    frameon=True,
                    bboxprops=dict(boxstyle="round,pad=0.2", lw=0.5, alpha=0.9),
                    zorder=10,
                    clip_on=False
                )
                ax.add_artist(ab)
                drawn += 1
                if VERBOSE_IO:
                    print(f"[draw] {resolved} at x={p['x']:.4g}")
            else:
                # Fallback marker (so the position is still visible)
                ax.plot(p["x"], y, "o", ms=8, zorder=5)
                if VERBOSE_IO:
                    print(f"[fallback-marker] {p['filename']} at x={p['x']:.4g} ; tried: {resolved}")

            label_text = f'{p["label"]}\n{p.get("klass","")}\n{p["x"]:.3g}'
            ax.text(p["x"], y + 0.1, label_text, ha="center", va="bottom",
                    fontsize=8, zorder=11, clip_on=False)

        if drawn == 0:
            tried = [load_image_verbose(p["filename"], IMAGE_DIR)[1] for p in picks]
            print(
                f"[warn] No images drawn for column '{col}'. "
                f"Skipping this plot. Tried paths (per pick): {tried}"
            )
            plt.close(fig)
            continue

        ax.text(vmin, -0.56, f"min\n{vmin:.3g}", ha="left",  va="top", fontsize=8)
        ax.text(vmax, -0.56, f"max\n{vmax:.3g}", ha="right", va="top", fontsize=8)

        out_path = Path(SAVE_DIR) / f"{col}_preview_scale.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print("[saved]", out_path)

def main():
    print("CWD:", os.getcwd())
    print("CSV exists?", os.path.exists(CSV_PATH))
    df = _read_df(CSV_PATH, ID_COL, CLASS_COL)

    # Drop index-artifact columns like 'Unnamed: 0', 'index', etc.
    artifact_like = [c for c in df.columns if c.lower().startswith("unnamed") or c.lower() in {"index"}]
    if artifact_like:
        print("[info] dropping artifact columns:", artifact_like)
        df = df.drop(columns=artifact_like)

    print("DF shape:", df.shape)

    # numeric columns (exclude id/class if numeric)
    value_cols = numeric_columns(df, ID_COL, CLASS_COL)
    print("Numeric columns:", value_cols)

    # ---- 1) Boxplots ----
    make_boxplots(df, CLASS_COL, value_cols, show_fliers=SHOW_FLIERS, whis=WHIS)

    # ---- 2) Closest items table (5 targets) ----
    closest_df = compute_closest_table(df, ID_COL, CLASS_COL, value_cols)

    # ---- 2b) Print neat per-column 0/25/50/75/100% table ----
    print_closest_tables(closest_df)

    # ---- 3) Preview scales with thumbnails ----
    draw_preview_scales(df, ID_COL, CLASS_COL, value_cols)

    print("\nDone. Images in:", os.path.abspath(SAVE_DIR))

if __name__ == "__main__":
    main()

# %%
