#%%
# import the csv
import csv
import pandas as pd
with open('features_unified_prepared.csv', mode='r') as file:
    csvFile = csv.reader(file)
    df = pd.read_csv("features_unified_prepared.csv", header=0)
    rectangularity = pd.read_csv("rectangularity_unified_prepared.csv", header=0)

# %%
df[["class", "filename"]] = df["name"].str.split(r"\\", n=1, expand=True)
df = df.reindex(columns=['class', 'filename', 'surface_area', 'volume', 'compactness', 'rectangularity', 'diameter', 'convexity', 'eccentricity'])

# %%
import matplotlib.pyplot as plt

# numeric columns except "class" and "filename"
value_cols = [c for c in df.select_dtypes(include='number').columns
              if c not in ('class', 'filename')]

for col in value_cols:
    # calculate median per class for this column
    medians = df.groupby("class")[col].median().sort_values()

    # get sorted classes by median
    sorted_classes = medians.index.tolist()

    # collect data in that order
    data = [df.loc[df["class"] == cl, col].dropna().values for cl in sorted_classes]

    plt.figure(figsize=(8, len(sorted_classes) * 0.2))
    plt.boxplot(data, labels=[str(cl) for cl in sorted_classes],
                showfliers=False, vert=False)  # <- horizontal!
    plt.title(f"{col} by class (sorted by median)")
    plt.ylabel("Class")
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

# %%
