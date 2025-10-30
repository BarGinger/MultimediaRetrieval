#%%
import pandas as pd

df = pd.read_csv("total_distances_597a37344657.csv", header=0, index_col=0)
labels = pd.read_csv("class_labels.csv", header=0, index_col=0)


#%%
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np

# df: square, labelled distance matrix
df = df.loc[df.index, df.index]

D = df.to_numpy(dtype=float)
np.fill_diagonal(D, 0.0)
D = 0.5 * (D + D.T)
D[D < 0] = 0.0
if not np.isfinite(D).all():
    raise ValueError("Distance matrix has NaN/Inf.")

tsne = TSNE(
    n_components=2,
    metric='precomputed',
    perplexity=30,
    learning_rate='auto',
    init='random',     # <-- change here
    random_state=42,
    verbose=1
)
Y = tsne.fit_transform(D)
embedding = pd.DataFrame(Y, index=df.index, columns=['x', 'y'])
print(embedding.head())

# %%
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

merged = embedding.merge(labels, left_index=True, right_on='shape')

# sort classes alphabetically or by some known order
merged['class'] = merged['class'].astype(str)
ordered_classes = sorted(merged['class'].unique())

# map each class to an integer
encoder = LabelEncoder()
encoder.classes_ = np.array(ordered_classes)
merged['class_code'] = encoder.transform(merged['class'])

# plot using continuous colormap
plt.figure(figsize=(8,8))
sc = plt.scatter(
    merged['x'], merged['y'],
    c=merged['class_code'],
    cmap='hsv',   # hsv cycles smoothly through hues
    s=15,
    alpha=0.8
)
plt.colorbar(sc, label='Ordered class index')
plt.xlabel('t-SNE 1'); plt.ylabel('t-SNE 2')
plt.title('t-SNE embedding coloured by ordered class similarity')
plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Merge embedding and labels
merged = embedding.merge(labels, left_index=True, right_on='shape')

# Get unique classes
classes = merged['class'].unique()
n_classes = len(classes)

# Compute grid size (e.g. 9×8 = 72 subplots for 69 classes)
cols = int(np.ceil(np.sqrt(n_classes)))
rows = int(np.ceil(n_classes / cols))

# Create figure
fig, axes = plt.subplots(rows, cols, figsize=(cols*2.2, rows*2.2))
axes = axes.flatten()

for i, c in enumerate(classes):
    ax = axes[i]
    subset = merged[merged['class'] == c]
    
    # Plot all points in grey for context
    ax.scatter(merged['x'], merged['y'], s=5, color='lightgrey', alpha=0.3)
    # Highlight this class in colour
    ax.scatter(subset['x'], subset['y'], s=10, color='tab:blue', alpha=0.9)
    
    ax.set_title(str(c), fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

# Hide any unused subplots
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Inputs:
# - df: square distance matrix (index = shape names, columns = shape names)
# - labels: DataFrame with columns ['shape', 'class']

K = 10  # number of nearest neighbours to evaluate

# 1) Align labels to df index order
df = df.loc[df.index, df.index]
lab = labels.set_index('shape').reindex(df.index)['class']

if lab.isna().any():
    missing = lab[lab.isna()].index.tolist()[:5]
    raise ValueError(f"Missing class labels for some shapes, e.g. {missing}")

# 2) Prepare distance matrix (exclude self-distances)
D = df.to_numpy(dtype=float)
np.fill_diagonal(D, np.inf)  # so self never selected among neighbours

n = D.shape[0]
classes = lab.to_numpy()

# 3) For each row, get indices of K nearest neighbours
#    argpartition is O(n) per row and much faster than full argsort
nn_idx = np.argpartition(D, K, axis=1)[:, :K]  # (n, K) unordered among K
# Optionally sort those K by actual distance for neatness (not strictly needed)
row_indices = np.arange(n)[:, None]
nn_dists = D[row_indices, nn_idx]
order_within_k = np.argsort(nn_dists, axis=1)
nn_idx = nn_idx[row_indices, order_within_k]    # now sorted (n, K)

# 4) Compute correctness@K per sample
nn_classes = classes[nn_idx]                    # (n, K) class of each neighbour
same_class = (nn_classes == classes[:, None])   # (n, K) boolean
correct_k = same_class.sum(axis=1)              # (n,) count of correct neighbours

# 5) Aggregate per class
results = pd.DataFrame({
    'shape': df.index,
    'class': classes,
    f'correct@{K}': correct_k,
    f'perc@{K}': correct_k / K * 100.0
})

per_class = (results
             .groupby('class', sort=False)
             .agg(**{
                 f'avg_correct@{K}': (f'correct@{K}', 'mean'),
                 f'avg_perc@{K}%': (f'perc@{K}', 'mean'),
                 'n_items': ('shape', 'count')
             })
             .sort_values(by=f'avg_perc@{K}%', ascending=False))

print(per_class.head(10))

# 6) Plot: one bar per class (average % correct out of 10)
plt.figure(figsize=(max(10, len(per_class)*0.18), 5))
plt.bar(per_class.index.astype(str), per_class[f'avg_perc@{K}%'])
plt.ylabel(f'Average precision@{K} (%)')
plt.title(f'Per-class average of correct neighbours among top {K}')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt
import pandas as pd

# labels: DataFrame with ['shape', 'class']
class_counts = labels['class'].value_counts()
total = len(labels)

# Compute random probability (chance that a random shape is from same class)
random_prob = class_counts / total * 100  # percentage

# Build DataFrame and sort by highest probability
random_baseline = (
    pd.DataFrame({
        'class': class_counts.index,
        'n_items': class_counts.values,
        'random_prob_%': random_prob.values
    })
    .sort_values('random_prob_%', ascending=False)
    .set_index('class')
)

print(random_baseline.head())

# Plot
plt.figure(figsize=(max(10, len(random_baseline)*0.18), 5))
plt.bar(random_baseline.index.astype(str), random_baseline['random_prob_%'], color='steelblue')
plt.ylabel('Random match probability (%)')
plt.title('Baseline probability of correct match by random sampling (sorted)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ---- 1️⃣  Per-class precision results (from your earlier computation)
# Make sure you already have `per_class` with 'avg_perc@10%' column
# and `labels` with ['shape', 'class']

# ---- 2️⃣  Random baseline probability
class_counts = labels['class'].value_counts()
total = len(labels)
random_baseline = pd.DataFrame({
    'class': class_counts.index,
    'random_prob_%': class_counts.values / total * 100
}).set_index('class')

# ---- 3️⃣  Combine both
comparison = per_class.join(random_baseline, how='left')

# ---- 4️⃣  Sort by highest actual accuracy
comparison = comparison.sort_values(by='avg_perc@10%', ascending=False)

# ---- 5️⃣  Plot
x = np.arange(len(comparison))
width = 0.4

plt.figure(figsize=(max(10, len(comparison)*0.18), 5))
plt.bar(x - width/2, comparison['avg_perc@10%'], width, label='Actual @10', color='tab:blue')
plt.bar(x + width/2, comparison['random_prob_%'], width, label='Random chance', color='tab:red')

plt.xticks(x, comparison.index.astype(str), rotation=90)
plt.ylabel('Average % of correct neighbours')
plt.title('Per-class retrieval accuracy vs random baseline')
plt.legend()
plt.tight_layout()
plt.show()
