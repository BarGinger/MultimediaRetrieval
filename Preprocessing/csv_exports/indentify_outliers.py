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

df['to_exclude'] = df['resampled_vertices'].apply(
    lambda v: is_outlier(v, MIN_ACCEPTABLE_VERTICES, MAX_ACCEPTABLE_VERTICES, TOLERANCE_PERCENT)
)

# Print counts of True and False in 'to_exclude'
true_count = df['to_exclude'].sum()
false_count = (~df['to_exclude']).sum()
print(f"Count to_exclude=True: {true_count}")
print(f"Count to_exclude=False: {false_count}")

# Save updated CSV
df.to_csv(CSV_PATH, index=False)
print(f"Updated CSV with 'to_exclude' column saved to {CSV_PATH}")