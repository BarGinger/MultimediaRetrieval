import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
CSV_PATH = "csv.csv"   # <--- change this
OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8")  # nicer styling for report figures

# ---------------------------------------------------------------------
# 1. Read CSV and basic preprocessing
# ---------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

# Make sure num_vertices is numeric
df["num_vertices"] = pd.to_numeric(df["num_vertices"], errors="coerce")

# Extract shape_id and stage from the filename, e.g. m1337_05_scaled.obj
df["shape_id"] = df["shape_file"].str.split("_").str[0]
df["stage"] = df["shape_file"].str.split("_").str[1]  # "00", "01", ..., "06"

# ---------------------------------------------------------------------
# 2. Filter stages 05 and 06 and pair them up
# ---------------------------------------------------------------------
# Keep class from stage 05 (they should all match stage 06 anyway)
df_05 = df[df["stage"] == "05"][["shape_id", "class", "num_vertices"]].rename(
    columns={"num_vertices": "num_vertices_05"}
)
df_06 = df[df["stage"] == "06"][["shape_id", "num_vertices"]].rename(
    columns={"num_vertices": "num_vertices_06"}
)

pairs = pd.merge(df_05, df_06, on="shape_id", how="inner")

# ---------------------------------------------------------------------
# 3. Compute differences
# ---------------------------------------------------------------------
pairs["diff_vertices"] = pairs["num_vertices_06"] - pairs["num_vertices_05"]
pairs["abs_diff_vertices"] = pairs["diff_vertices"].abs()
pairs["rel_change"] = pairs["diff_vertices"] / pairs["num_vertices_05"]

print("Number of shapes with both 05 and 06 stages:", len(pairs))

unchanged = (pairs["diff_vertices"] == 0).sum()
print(f"Unchanged vertex count: {unchanged} shapes "
      f"({unchanged / len(pairs) * 100:.1f}%)")

top5 = pairs.sort_values("abs_diff_vertices", ascending=False).head(5)
print("\nTop 5 largest changes in number of vertices (06 vs 05):")
print(top5[["shape_id", "class", "num_vertices_05", "num_vertices_06",
            "diff_vertices", "rel_change"]])

# ---------------------------------------------------------------------
# 4. Histogram: full distribution + zoomed non-zero part
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

# Left: full histogram with log y-scale
axes[0].hist(pairs["diff_vertices"], bins=30, edgecolor="black")
axes[0].set_title("Change in number of vertices (06 - 05)\nFull distribution")
axes[0].set_xlabel("Difference in num_vertices")
axes[0].set_ylabel("Count of shapes")
axes[0].axvline(0, linestyle="--", linewidth=1)
axes[0].set_yscale("log")  # log scale so the huge zero peak doesn't dominate

# Right: zoomed histogram excluding zeros
nonzero = pairs[pairs["diff_vertices"] != 0]["diff_vertices"]
if len(nonzero) > 0:
    axes[1].hist(nonzero, bins=30, edgecolor="black")
    axes[1].set_title("Change in number of vertices (06 - 05)\nNon-zero only")
    axes[1].set_xlabel("Difference in num_vertices")
    axes[1].axvline(0, linestyle="--", linewidth=1)
else:
    axes[1].text(0.5, 0.5, "No non-zero differences",
                 ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set_axis_off()

fig.suptitle("Vertex count changes between stage 05 and 06", fontsize=14)
fig.tight_layout(rect=[0, 0.0, 1, 0.95])
fig.savefig(os.path.join(OUT_DIR, "vertex_diff_histograms.png"), dpi=300)

# ---------------------------------------------------------------------
# 5. Bar chart: Top 5 largest changes (absolute)
# ---------------------------------------------------------------------
top5_sorted = top5.sort_values("abs_diff_vertices", ascending=False)

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(top5_sorted["shape_id"], top5_sorted["diff_vertices"])

ax.set_title("Top 5 changes in number of vertices (06 - 05)")
ax.set_xlabel("Shape ID")
ax.set_ylabel("Difference in num_vertices")
ax.axhline(0, linewidth=1, color="black")

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f"{int(height)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "vertex_diff_top5.png"), dpi=300)

# ---------------------------------------------------------------------
# 6. Per-class analysis
# ---------------------------------------------------------------------
# Aggregate per class
class_stats = (
    pairs
    .groupby("class")
    .agg(
        mean_abs_diff=("abs_diff_vertices", "mean"),
        max_abs_diff=("abs_diff_vertices", "max"),
        changed=("abs_diff_vertices", lambda x: (x != 0).sum()),
        total=("abs_diff_vertices", "size")
    )
)

class_stats["unchanged"] = class_stats["total"] - class_stats["changed"]
class_stats["frac_changed"] = class_stats["changed"] / class_stats["total"]

print("\nPer-class vertex change stats:")
print(class_stats)

# Sort classes by mean_abs_diff for nicer plotting
class_stats_sorted = class_stats.sort_values("mean_abs_diff", ascending=False)

# ---------------------------------------------------------------------
# 6a. Bar chart: mean absolute change per class
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4))

ax.bar(class_stats_sorted.index, class_stats_sorted["mean_abs_diff"])
ax.set_title("Mean absolute change in number of vertices per class")
ax.set_xlabel("Class")
ax.set_ylabel("Mean |Δ num_vertices|")
ax.tick_params(axis='x', rotation=45)
for label in ax.get_xticklabels():
    label.set_horizontalalignment('right')

# Optional: add value labels if there are not too many classes
for i, (cls, row) in enumerate(class_stats_sorted.iterrows()):
    ax.annotate(f"{row['mean_abs_diff']:.1f}",
                xy=(i, row['mean_abs_diff']),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "vertex_diff_mean_per_class.png"), dpi=300)

# ---------------------------------------------------------------------
# 6b. Stacked bar: changed vs unchanged per class
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4))

width = 0.6
x = range(len(class_stats_sorted))

ax.bar(x, class_stats_sorted["unchanged"], width, label="Unchanged")
ax.bar(x, class_stats_sorted["changed"], width,
       bottom=class_stats_sorted["unchanged"], label="Changed")

ax.set_title("Number of meshes per class\nChanged vs unchanged vertex count")
ax.set_xlabel("Class")
ax.set_ylabel("Number of meshes")
ax.set_xticks(list(x))
ax.set_xticklabels(class_stats_sorted.index, rotation=45, ha="right")
ax.legend()

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "vertex_diff_changed_vs_unchanged_per_class.png"),
            dpi=300)

plt.show()  # optional
