"""
Standardize numeric features in a CSV using z-score (mean/std).

Reads:  output/features_unified_prepared.csv
Writes: output/features_unified_prepared_standardized.csv

Behavior:
- Preserves the 'name' column (first column).
- Standardizes all other numeric columns: (x - mean) / std.
- If a column has std == 0 (constant column), std is set to 1 to avoid division-by-zero
  so the standardized values become 0.

Usage:
    python scripts standardize_features.py

This script is intentionally small and dependency-light (uses pandas, numpy).
"""
from pathlib import Path
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def standardize_csv(input_path: Path, output_path: Path, id_column: str = "name") -> None:
    df = pd.read_csv(input_path)

    if id_column not in df.columns:
        raise ValueError(f"Identifier column '{id_column}' not found in {input_path}")

    # Separate id column and feature columns
    ids = df[id_column]
    features = df.drop(columns=[id_column])

    # Ensure features are numeric where possible
    numeric_features = features.apply(pd.to_numeric, errors="coerce")

    # Compute mean and std (population std, ddof=0) as in the slide notation
    means = numeric_features.mean(axis=0, skipna=True)
    stds = numeric_features.std(axis=0, ddof=0, skipna=True)

    # Avoid division by zero: replace zeros with 1.0
    zero_std = stds == 0
    if zero_std.any():
        stds[zero_std] = 1.0

    standardized = (numeric_features - means) / stds

    # Where original conversion to numeric produced NaN (non-numeric), keep original values
    non_numeric_mask = features.isna() & numeric_features.isna()

    # Build final dataframe: id column + standardized numeric columns + any original non-numeric columns
    out_df = pd.concat([ids.reset_index(drop=True), standardized.reset_index(drop=True)], axis=1)

    # For any columns that were non-numeric (all NaN after to_numeric), fall back to original
    for col in features.columns:
        if numeric_features[col].isna().all():
            out_df[col] = features[col].values

    # Re-order columns to put id_column first
    cols = [id_column] + [c for c in out_df.columns if c != id_column]
    out_df = out_df[cols]

    # Write output
    out_df.to_csv(output_path, index=False)
    print(f"Wrote standardized CSV to: {output_path}")


def safe_filename(s: str) -> str:
    # Replace unsafe characters for filenames
    return "".join([c if c.isalnum() or c in (' ', '.', '_', '-') else '_' for c in s]).replace(' ', '_')


def plot_features(df: pd.DataFrame, id_column: str, out_dir: Path) -> None:
    """Create histogram + boxplot for each numeric feature in df (excluding id_column).

    Saves one PNG per feature into out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    features = df.drop(columns=[id_column])
    # Convert to numeric where possible
    numeric = features.apply(pd.to_numeric, errors='coerce')

    for col in numeric.columns:
        col_data = numeric[col].dropna()
        if col_data.empty:
            continue

        # Use constrained_layout and larger figure so annotations and titles don't overlap
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]}, constrained_layout=True)

        hist = sns.histplot(col_data, ax=axes[0], kde=False, bins=40, color='C0')
        # Annotate counts above each bar (vertical text)
        # Annotate bars but add a small offset so labels don't sit on the bar top.
        ylim = axes[0].get_ylim()
        y_pad = max((ylim[1] - ylim[0]) * 0.02, 1.0)
        for patch in hist.patches:
            height = patch.get_height()
            if height <= 0:
                continue
            x = patch.get_x() + patch.get_width() / 2
            y = height + y_pad
            axes[0].text(x, y, f"{int(height)}", ha='center', va='bottom', rotation=90, fontsize=7,
                         clip_on=False, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.2))
        axes[0].set_title(f"After normalization (z-score) — histogram of {col}")
        axes[0].set_ylabel('Count')

        sns.boxplot(x=col_data, ax=axes[1], color='C1')
        axes[1].set_xlabel(f"{col} (after — standardized)")

        # Add statistical annotation for the 'after' (standardized) data
        a = col_data
        mean_a = a.mean()
        std_a = a.std(ddof=0)
        pct1 = (a.abs() <= 1).mean() * 100
        pct2 = (a.abs() <= 2).mean() * 100
        pct3 = (a.abs() <= 3).mean() * 100
        maxabs = a.abs().max()

        # Use raw f-string mathtext for bold labels and left-align the box
        stats_text = (
            fr"$\bf{{after}}:$\n"
            fr"$\bf{{mean}}:$ {mean_a:.3g}    $\bf{{std}}:$ {std_a:.3g}\n"
            fr"$\bf{{|z|<=1}}:$ {pct1:.1f}%    $\bf{{|z|<=2}}:$ {pct2:.1f}%\n"
            fr"$\bf{{|z|<=3}}:$ {pct3:.1f}%    $\bf{{max|z|}}:$ {maxabs:.3g}"
        )

        # Place stats box slightly lower to avoid clashing with axis title
        axes[0].text(0.02, 0.90, stats_text, transform=axes[0].transAxes,
                     ha='left', va='top', fontsize=9,
                     bbox=dict(facecolor='white', alpha=0.85, edgecolor='black', boxstyle='round'))

        # Ensure some extra room at the top for titles and boxes
        try:
            fig.subplots_adjust(top=0.88)
        except Exception:
            pass

        filename = out_dir / f"feature_{safe_filename(col)}.png"
        fig.savefig(filename, dpi=150)
        plt.close(fig)
        print(f"Saved plot: {filename}")


def plot_features_combined(df_before: pd.DataFrame, df_after: pd.DataFrame, id_column: str, out_dir: Path) -> None:
    """Create combined before/after histogram+boxplot per feature.

    Layout: 2 rows x 2 columns per figure
      [hist before] [hist after]
      [box before]  [box after]
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    fb = df_before.drop(columns=[id_column]).apply(pd.to_numeric, errors='coerce')
    fa = df_after.drop(columns=[id_column]).apply(pd.to_numeric, errors='coerce')

    cols = [c for c in fb.columns if c in fa.columns]
    for col in cols:
        bdata = fb[col].dropna()
        adata = fa[col].dropna()
        if bdata.empty and adata.empty:
            continue

        # Larger combined figure. Reserve an extra bottom row for a stats table that
        # spans both columns so the information doesn't overflow the plot area.
        fig = plt.figure(figsize=(12, 8), constrained_layout=True)
        gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[3, 1, 0.6])

        # Create axes from the gridspec
        ax_hist_b = fig.add_subplot(gs[0, 0])
        ax_hist_a = fig.add_subplot(gs[0, 1])
        ax_box_b = fig.add_subplot(gs[1, 0])
        ax_box_a = fig.add_subplot(gs[1, 1])
        ax_table = fig.add_subplot(gs[2, :])  # span both columns

        # Histograms
        hist_b = sns.histplot(bdata, ax=ax_hist_b, bins=40, color='C0')
        ylim_b = ax_hist_b.get_ylim()
        ypad_b = max((ylim_b[1] - ylim_b[0]) * 0.02, 1.0)
        for patch in hist_b.patches:
            h = patch.get_height()
            if h <= 0:
                continue
            x = patch.get_x() + patch.get_width() / 2
            y = h + ypad_b
            ax_hist_b.text(x, y, f"{int(h)}", ha='center', va='bottom', rotation=90, fontsize=7,
                           clip_on=False, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.2))
        ax_hist_b.set_title(f"Before normalization (original units) — {col}")
        ax_hist_b.set_ylabel('Count')

        hist_a = sns.histplot(adata, ax=ax_hist_a, bins=40, color='C2')
        ylim_a = ax_hist_a.get_ylim()
        ypad_a = max((ylim_a[1] - ylim_a[0]) * 0.02, 1.0)
        for patch in hist_a.patches:
            h = patch.get_height()
            if h <= 0:
                continue
            x = patch.get_x() + patch.get_width() / 2
            y = h + ypad_a
            ax_hist_a.text(x, y, f"{int(h)}", ha='center', va='bottom', rotation=90, fontsize=7,
                           clip_on=False, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.2))
        ax_hist_a.set_title(f"After normalization (z-score) — {col}")

        # Boxplots
        sns.boxplot(x=bdata, ax=ax_box_b, color='C1')
        ax_box_b.set_xlabel(f"{col} (before — original units)")

        sns.boxplot(x=adata, ax=ax_box_a, color='C3')
        ax_box_a.set_xlabel(f"{col} (after — z-score)")

        # Compute and add stats: before (raw) and after (z)
        # Before stats
        if not bdata.empty:
            b_mean = bdata.mean()
            b_std = bdata.std(ddof=0)
            b_min = bdata.min()
            b_max = bdata.max()
            b_median = bdata.median()
        else:
            b_mean = b_std = b_min = b_max = b_median = float('nan')

        # After stats
        if not adata.empty:
            a_mean = adata.mean()
            a_std = adata.std(ddof=0)
            a_pct1 = (adata.abs() <= 1).mean() * 100
            a_pct2 = (adata.abs() <= 2).mean() * 100
            a_pct3 = (adata.abs() <= 3).mean() * 100
            a_maxabs = adata.abs().max()
        else:
            a_mean = a_std = a_pct1 = a_pct2 = a_pct3 = a_maxabs = float('nan')

        # Build a table for stats so the info has its own subplot. This prevents
        # overflow and keeps the layout consistent across features.
        # Compute median for after as well
        a_median = adata.median() if not adata.empty else float('nan')

        col_labels = ['mean', 'std', 'median', 'min', 'max', '|z|<=1', '|z|<=2', '|z|<=3', 'max|z|']

        before_row = [
            f"{b_mean:.3g}",
            f"{b_std:.3g}",
            f"{b_median:.3g}",
            f"{b_min:.3g}",
            f"{b_max:.3g}",
            '-', '-', '-', '-'
        ]

        after_row = [
            f"{a_mean:.3g}",
            f"{a_std:.3g}",
            f"{a_median:.3g}",
            '-', '-',
            f"{a_pct1:.1f}%", f"{a_pct2:.1f}%", f"{a_pct3:.1f}%", f"{a_maxabs:.3g}"
        ]

        cell_text = [before_row, after_row]
        row_labels = ['before', 'after']

        # Draw the table centered in the bottom subplot
        ax_table.axis('off')
        table = ax_table.table(cellText=cell_text, colLabels=col_labels, rowLabels=row_labels,
                               cellLoc='center', loc='center')
        # Styling: font size, scale
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.6)

        # Header and row-label styling: header is at row -1, row labels are at col -1
        ncols = len(col_labels)
        nrows = len(cell_text)

        # Header and row-label styling: some Matplotlib versions don't expose
        # header cells via table[(-1, c)] so iterate over all cells and match
        # header/row labels by text content to apply styling safely.
        celld = table.get_celld()
        for (r, c), cell in celld.items():
            txt = cell.get_text().get_text()
            # header cells: text matches one of the column labels
            if txt in col_labels:
                cell.set_facecolor('#d3d3d3')
                cell.get_text().set_fontweight('bold')
                cell.get_text().set_ha('center')
                cell.get_text().set_va('center')
            # row label cells: text matches one of the row labels
            elif txt in row_labels:
                cell.get_text().set_fontweight('bold')
                cell.get_text().set_ha('center')
                cell.get_text().set_va('center')
            else:
                # data cells: zebra-striping by row index (header handled above)
                try:
                    if (r % 2) == 0:
                        cell.set_facecolor('#ffffff')
                    else:
                        cell.set_facecolor('#f7f7f7')
                except Exception:
                    cell.set_facecolor('#ffffff')
                cell.get_text().set_ha('center')
                cell.get_text().set_va('center')

        # Improve cell borders by increasing edge width
        for key, cell in table.get_celld().items():
            cell.set_linewidth(0.8)

        # Ensure some extra room at the top for titles and boxes; constrained_layout helps, but adjust if needed
        try:
            fig.subplots_adjust(top=0.88)
        except Exception:
            pass

        filename = out_dir / f"feature_combined_{safe_filename(col)}.png"
        fig.savefig(filename, dpi=150)
        plt.close(fig)
        print(f"Saved combined plot: {filename}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", help="Input CSV path",
                        default="output/features_unified_prepared.csv")
    parser.add_argument("--output", "-o", help="Output CSV path",
                        default="output/features_unified_prepared_standardized.csv")
    parser.add_argument("--id-column", "-c", help="Identifier column name",
                        default="name")
    parser.add_argument("--plot", action="store_true", help="Generate per-feature plots before and after standardization", default=True)
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(2)

    try:
        # If plotting requested, generate combined before/after plots
        if getattr(args, 'plot', False):
            try:
                df_before = pd.read_csv(input_path)
            except Exception as e:
                print(f"Warning: failed to read 'before' CSV for plotting: {e}")
                df_before = None

        standardize_csv(input_path, output_path, id_column=args.id_column)

        if getattr(args, 'plot', False):
            try:
                df_after = pd.read_csv(output_path)
                plot_dir_combined = output_path.parent / 'feature_plots' / 'combined'
                if df_before is None:
                    # Fall back: only plot after using single plot function
                    plot_features(df_after, args.id_column, plot_dir_combined)
                else:
                    plot_features_combined(df_before, df_after, args.id_column, plot_dir_combined)
            except Exception as e:
                print(f"Warning: failed to produce combined plots: {e}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
