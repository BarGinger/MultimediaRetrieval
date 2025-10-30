import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
from matplotlib import colors

class TopologyGraph:

    def generate_graph():
        os.chdir(os.path.dirname(__file__))
        print(f"Current working directory: {os.getcwd()}")
        df = pd.read_csv("total_distances_optimized.csv", header=0, index_col=0)

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
            perplexity=50,
            learning_rate='auto',
            init='random',     # <-- change here
            random_state=69,
            verbose=1,
            early_exaggeration=12.0,
            max_iter=5000
        )
        Y = tsne.fit_transform(D)
        embedding = pd.DataFrame(Y, index=df.index, columns=['x', 'y'])
        print(embedding.head())
        embedding.to_csv("topology_graph.csv")


    def print_graph(name: str = "", n: int = 0):
        # Work relative to this script file
        try:
            os.chdir(os.path.dirname(__file__))
        except NameError:
            pass

        print(f"Current working directory: {os.getcwd()}")

        # --- Load embedding and labels
        embedding = pd.read_csv("topology_graph.csv", header=0, index_col=0)
        labels = pd.read_csv("class_labels.csv", header=0, index_col=0)

        if "shape" not in labels.columns:
            labels = labels.reset_index().rename(columns={"index": "shape"})

        merged = embedding.merge(labels, left_index=True, right_on="shape", how="left")

        # Encode classes
        merged["class"] = merged["class"].astype(str)
        ordered_classes = sorted(merged["class"].dropna().unique())
        encoder = LabelEncoder()
        encoder.classes_ = np.array(ordered_classes)
        merged["class_code"] = encoder.transform(merged["class"])

        n_classes = len(ordered_classes)
        cmap = plt.get_cmap("hsv", n_classes)
        norm = colors.BoundaryNorm(boundaries=np.arange(-0.5, n_classes + 0.5, 1), ncolors=n_classes)

        plt.figure(figsize=(8, 8))
        base_alpha = 0.8 if (name == "" or name is None) else 0.1
        base_size = 15  # <-- uniform point size for all shapes

        sc = plt.scatter(
            merged["x"], merged["y"],
            c=merged["class_code"],
            cmap=cmap, norm=norm,
            s=base_size,
            alpha=base_alpha
        )
        cb = plt.colorbar(sc, label="Class")
        cb.set_ticks(range(n_classes))
        cb.set_ticklabels(ordered_classes)
        plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")

        title = "t-SNE embedding"
        out_png = "topology_graph.png"

        # --- Focus mode
        if isinstance(name, str) and len(name) > 0:
            title += f" • focus: {name} (n={n})"
            if name not in merged["shape"].values:
                raise ValueError(f"Object '{name}' not found in embedding/labels.")

            target_row = merged.loc[merged["shape"] == name].iloc[0]
            target_class = target_row["class"]
            same_class_mask = (merged["class"] == target_class)

            # Neighbours
            dist_df = pd.read_csv("total_distances_optimized.csv", header=0, index_col=0)
            dist_df = dist_df.loc[dist_df.index, dist_df.index]
            if name not in dist_df.index:
                raise ValueError(f"Object '{name}' not found in distance matrix.")

            row = dist_df.loc[name].copy().drop(labels=[name], errors="ignore")
            row = pd.to_numeric(row, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            neigh_list = row.nsmallest(max(0, int(n))).index.tolist() if n and n > 0 else []

            target_mask = (merged["shape"] == name)
            neigh_mask  = merged["shape"].isin(neigh_list)
            same_class_only_mask = same_class_mask & ~target_mask & ~neigh_mask

            # Same-class non-neighbours
            if same_class_only_mask.any():
                plt.scatter(
                    merged.loc[same_class_only_mask, "x"],
                    merged.loc[same_class_only_mask, "y"],
                    c=merged.loc[same_class_only_mask, "class_code"],
                    cmap=cmap, norm=norm,
                    s=base_size, alpha=1.0, marker="o", edgecolors="none"
                )

            # Neighbours (same + diff class, their own colours)
            neigh_same_class_mask = neigh_mask & (merged["class"] == target_class)
            neigh_diff_class_mask = neigh_mask & (merged["class"] != target_class)

            if neigh_same_class_mask.any():
                plt.scatter(
                    merged.loc[neigh_same_class_mask, "x"],
                    merged.loc[neigh_same_class_mask, "y"],
                    c=merged.loc[neigh_same_class_mask, "class_code"],
                    cmap=cmap, norm=norm,
                    s=base_size, alpha=1.0, marker="^",
                    linewidths=0.4, edgecolors="black",
                    label="Same-class neighbours"
                )

            if neigh_diff_class_mask.any():
                plt.scatter(
                    merged.loc[neigh_diff_class_mask, "x"],
                    merged.loc[neigh_diff_class_mask, "y"],
                    c=merged.loc[neigh_diff_class_mask, "class_code"],
                    cmap=cmap, norm=norm,
                    s=base_size, alpha=1.0, marker="^",
                    linewidths=0.4, edgecolors="black",
                    label="Different-class neighbours"
                )

            # Target
            plt.scatter(
                merged.loc[target_mask, "x"],
                merged.loc[target_mask, "y"],
                c=merged.loc[target_mask, "class_code"],
                cmap=cmap, norm=norm,
                s=base_size, alpha=1.0, marker="s",
                linewidths=0.6, edgecolors="black", zorder=3
            )

            legend_elems = [
                Line2D([0], [0], marker='o', linestyle='None', markersize=5, label='All points (10%)', alpha=0.1),
                Line2D([0], [0], marker='o', linestyle='None', markersize=5, label=f"Same class: {target_class}", alpha=1.0),
                Line2D([0], [0], marker='^', linestyle='None', markersize=5, label="Same-class neighbours", markeredgecolor="black"),
                Line2D([0], [0], marker='^', linestyle='None', markersize=5, label="Different-class neighbours", markeredgecolor="black"),
                Line2D([0], [0], marker='s', linestyle='None', markersize=5, label='Target object', markeredgecolor="black"),
            ]
            plt.legend(handles=legend_elems, loc="best", frameon=True)
            out_png = f"topology_graph_{name}_n{len(neigh_list)}.png"

        plt.title(title)
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_png}")



    if __name__ == "__main__":
        print_graph("m1338_06_fill_holes_and_orientation.obj", n=5)
