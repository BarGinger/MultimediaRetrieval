"""
Normalize (standardize) pairwise distance matrices for each descriptor.

- Reads CSV distance matrices from an input folder (default: distance_matrices)
  with filenames like distances_A3.csv, distances_D1.csv, ...
- Computes mean and std over the OFF-DIAGONAL lower-triangular entries only
  (i > j), ignoring NaNs.
- Standardizes those entries: z = (d - mean) / std.
- Keeps the diagonal as 0.0 (sanity check) and the upper triangle as NaN
  (since distances are symmetric and aren't recomputed).
- Writes normalized matrices to an output folder (default: distance_matrices_normalized)
  preserving the same filenames.

Note: If std is 0 or extremely small, it falls back to std = 1 to avoid
division-by-zero, effectively producing zeros for all standardized entries.
"""

import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm


def normalize_distance_matrices(input_dir: str = "distance_matrices",
                                output_dir: str = "distance_matrices_normalized") -> None:
    """
    Normalize all distance matrices found in the input directory and save
    the standardized versions to the output directory.

    Args:
        input_dir: Directory containing distances_*.csv files
        output_dir: Directory to save normalized matrices
    """
    # Resolve directories relative to this script if not absolute
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if not os.path.isabs(input_dir):
        input_dir = os.path.join(script_dir, input_dir)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(script_dir, output_dir)

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Find all distance matrix CSVs
    pattern = os.path.join(input_dir, "distances_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files matching {pattern}")
        return

    print(f"Normalizing {len(files)} distance matrices from: {input_dir}")
    print(f"Saving results to: {output_dir}\n")

    for file_path in tqdm(files, desc="Normalizing matrices"):
        try:
            df = pd.read_csv(file_path, index_col=0)
        except Exception as e:
            print(f"\nWarning: Could not read {file_path}: {e}")
            continue

        # Convert to numpy array for efficient masking
        mat = df.values.astype(float)
        n = mat.shape[0]
        if mat.shape[0] != mat.shape[1]:
            print(f"\nWarning: Skipping non-square matrix: {file_path}")
            continue

        # Mask for lower triangle excluding diagonal
        lower_mask = np.tril(np.ones((n, n), dtype=bool), k=-1)

        # Extract valid off-diagonal distances (ignore NaN)
        lower_vals = mat[lower_mask]
        valid_vals = lower_vals[~np.isnan(lower_vals)]

        if valid_vals.size == 0:
            print(f"\nWarning: No valid distances found in lower triangle for {file_path}")
            # Still write out a copy with the same structure
            out_df = df.copy()
            out_path = os.path.join(output_dir, os.path.basename(file_path))
            out_df.to_csv(out_path)
            continue

        mean = float(np.mean(valid_vals))
        std = float(np.std(valid_vals, ddof=0))  # population std for consistency
        if std < 1e-12:
            std = 1.0  # avoid division by zero; results will be ~0

        # Prepare output matrix: NaN upper triangle, 0 on diagonal
        norm_mat = np.full_like(mat, np.nan)
        np.fill_diagonal(norm_mat, 0.0)

        # Standardize only the lower triangle values
        z_vals = (lower_vals - mean) / std
        norm_mat[lower_mask] = z_vals

        # Build DataFrame and save
        out_df = pd.DataFrame(norm_mat, index=df.index, columns=df.columns)
        out_path = os.path.join(output_dir, os.path.basename(file_path))
        out_df.to_csv(out_path)

        # Optionally, save stats next to the normalized file
        stats_path = out_path.replace('.csv', '_stats.csv')
        pd.DataFrame({
            'stat': ['mean', 'std', 'count'],
            'value': [mean, std, valid_vals.size]
        }).to_csv(stats_path, index=False)

    print("\nNormalization complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Normalize (z-score) distance matrices")
    parser.add_argument("--input-dir", type=str, default="distance_matrices",
                        help="Directory containing distances_*.csv files")
    parser.add_argument("--output-dir", type=str, default="distance_matrices_normalized",
                        help="Directory to save normalized matrices")

    args = parser.parse_args()

    normalize_distance_matrices(args.input_dir, args.output_dir)
