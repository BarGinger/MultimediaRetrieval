"""find_minmax.py

Read `output/descriptors_minmax.csv` and compute global minima for each min_* column
and global maxima for each max_* column. Output a single-row CSV file
`output/descriptors_global_minmax.csv` with the aggregated results.

Usage:
    python Src/find_minmax.py

The script assumes it's run from the repository root.
"""
import csv
from pathlib import Path

# Use repository root (two parents above this file) so paths point to MMR/output
REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = REPO_ROOT / "output" / "descriptors_minmax.csv"
OUTPUT_CSV = REPO_ROOT / "output" / "descriptors_global_minmax.csv"


def aggregate_min_max(input_path: Path, output_path: Path):
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("Input CSV has no header")

        # Prepare dictionaries to track global minima and maxima
        mins = {}
        maxs = {}

        # Identify min_/max_ columns
        min_cols = [c for c in fieldnames if c.startswith('min_')]
        max_cols = [c for c in fieldnames if c.startswith('max_')]

        # Initialize
        for c in min_cols:
            mins[c] = float('inf')
        for c in max_cols:
            maxs[c] = float('-inf')

        # Read rows and update
        for row in reader:
            for c in min_cols:
                v = row.get(c, '')
                if v != '':
                    try:
                        fv = float(v)
                        if fv < mins[c]:
                            mins[c] = fv
                    except ValueError:
                        pass
            for c in max_cols:
                v = row.get(c, '')
                if v != '':
                    try:
                        fv = float(v)
                        if fv > maxs[c]:
                            maxs[c] = fv
                    except ValueError:
                        pass

    # Prepare output header: keep 'name' then all min_ then all max_
    out_fieldnames = ['name'] + sorted(min_cols) + sorted(max_cols)

    # Compose a single row where 'name' can be set to 'global_minmax'
    out_row = {'name': 'global_minmax'}
    for c in min_cols:
        out_row[c] = f"{mins[c]:.6f}" if mins[c] != float('inf') else ''
    for c in max_cols:
        out_row[c] = f"{maxs[c]:.6f}" if maxs[c] != float('-inf') else ''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerow(out_row)

    print(f"Wrote aggregated min/max to: {output_path}")


if __name__ == '__main__':
    aggregate_min_max(INPUT_CSV, OUTPUT_CSV)
