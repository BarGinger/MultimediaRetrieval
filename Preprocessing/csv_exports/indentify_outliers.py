import pandas as pd
import os

# Parameters
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "files_outside_range.csv")
MIN_ACCEPTABLE_VERTICES = 5000
MAX_ACCEPTABLE_VERTICES = 10000
TOLERANCE_PERCENT = 0.15  # 15% tolerance

# Read CSV
df = pd.read_csv(CSV_PATH)

# Add 'to_exclude' column
def is_outlier(v, min_v, max_v, tol_percent):
    # Use percentage-based tolerance
    tol_min = min_v * tol_percent
    tol_max = max_v * tol_percent
    return not ((min_v - tol_min) <= v <= (max_v + tol_max))


def percent_diff_from_range(v, min_v, max_v):
    """
    Compute signed percent difference relative to the acceptable range.
    - Negative value: how far below the minimum (in % of min)
    - Positive value: how far above the maximum (in % of max)
    - Zero: within the acceptable range
    """
    try:
        if v < min_v:
            return (v - min_v) / float(min_v) * 100.0
        if v > max_v:
            return (v - max_v) / float(max_v) * 100.0
        return 0.0
    except Exception:
        return float('nan')

df['to_exclude'] = df['resampled_vertices'].apply(
    lambda v: is_outlier(v, MIN_ACCEPTABLE_VERTICES, MAX_ACCEPTABLE_VERTICES, TOLERANCE_PERCENT)
)

# Add percent difference column for easier explanation (negative = below min, positive = above max)
df['percent_diff'] = df['resampled_vertices'].apply(
    lambda v: round(percent_diff_from_range(v, MIN_ACCEPTABLE_VERTICES, MAX_ACCEPTABLE_VERTICES), 2)
)

# Print counts of True and False in 'to_exclude'
true_count = df['to_exclude'].sum()
false_count = (~df['to_exclude']).sum()
print(f"Count to_exclude=True: {true_count}")
print(f"Count to_exclude=False: {false_count}")

# Show top 5 largest negative and positive percent diffs for quick inspection
if 'percent_diff' in df.columns:
    below = df.nsmallest(5, 'percent_diff')[['file', 'percent_diff']]
    above = df.nlargest(5, 'percent_diff')[['file', 'percent_diff']]
    print('\nTop 5 below min (most negative %):')
    print(below.to_string(index=False))
    print('\nTop 5 above max (most positive %):')
    print(above.to_string(index=False))

# Save updated CSV
df.to_csv(CSV_PATH, index=False)
print(f"Updated CSV with 'to_exclude' column saved to {CSV_PATH}")