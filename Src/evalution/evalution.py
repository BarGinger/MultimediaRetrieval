"""Evaluation utilities for Step 6: compute retrieval quality metrics.

This script compares three distance/ranking outputs against ground-truth class labels
and computes a broad set of binary-relevance metrics per query, per class, and overall.

Usage (from project root):
    python -m Src.obj-viewer-app.evalution.evalution

By default it looks for the matching files under `Src/matching/` and the ground truth
analysis file under `Datasets/UnifiedPreprocessed/Data/analysis_results_unifiedPreprocessed_data.csv`.

The script writes per-approach CSV summaries into `Reports/evaluation/` and prints a short summary.
"""
from __future__ import annotations
import os
import math
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def progress_iterable(iterable, desc: Optional[str] = None):
    """Return an iterable with progress display: tqdm if available, else a simple generator

    desc is used for printed messages when tqdm is not installed.
    """
    if tqdm is not None:
        return tqdm(iterable, desc=desc)

    # Simple fallback generator with coarse progress prints
    def _gen():
        # try to get length for percent reporting
        total = len(iterable) if hasattr(iterable, '__len__') else None
        i = 0
        for item in iterable:
            i += 1
            if total:
                # print for 0%,10%,..,100%
                if i == 1 or i == total or i % max(1, total // 10) == 0:
                    print(f"{desc or 'Progress'}: {i}/{total}")
            else:
                if i % 100 == 0:
                    print(f"{desc or 'Progress'}: processed {i} items")
            yield item

    return _gen()


DEFAULT_MATCHING = [
    "Src/matching/matrix_minmax_optimized.csv",
    "Src/matching/matrix_rank_based_optimized.csv",
    "Src/matching/matrix_weighted_sum.csv",
]

DEFAULT_ANALYSIS = "Datasets/UnifiedPreprocessed/Data/analysis_results_unifiedPreprocessed_data.csv"


def normalize_id(name: str) -> str:
    """Normalize a filename to a short id used to match across tables.

    Strategy: remove path, lowercase, remove extension, then take leading alphanumeric token
    (characters before first underscore) if present. This works with names like
    'm1337_unified.obj' and 'm1337_06_fill_holes_and_orientation.obj' mapping to 'm1337'.
    """
    if pd.isna(name):
        return ""
    s = os.path.basename(str(name)).lower()
    if s.endswith('.obj'):
        s = s[:-4]
    # If there's a leading alphanumeric token, return it
    import re
    m = re.match(r"^([a-z0-9]+)", s)
    if m:
        return m.group(1)
    return s


def load_analysis_labels(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Try to find filename column (common names: 'shape_file','filename','name')
    filename_col = None
    for c in ['shape_file', 'filename', 'name', 'shape', 'file']:
        if c in df.columns:
            filename_col = c
            break
    if filename_col is None:
        # fallback to first column
        filename_col = df.columns[0]
    df = df.rename(columns={filename_col: 'filename'})
    if 'class' not in df.columns:
        # try class-like columns
        for c in ['class', 'label', 'category', 'class_b']:
            if c in df.columns:
                df = df.rename(columns={c: 'class'})
                break
    if 'class' not in df.columns:
        raise RuntimeError('Could not find class column in analysis CSV')
    df['id'] = df['filename'].apply(normalize_id)
    return df[['filename', 'id', 'class']]


def load_distance_matrix(path: str) -> pd.DataFrame:
    """Load a distance matrix CSV into a DataFrame.

    Accepts either a full matrix (rows and columns are filenames) or a table with
    a leading filename column and distance columns.
    """
    df = pd.read_csv(path, index_col=None)
    # If DataFrame seems square and header parsed as columns, return as-is
    if df.shape[0] == df.shape[1] and df.columns.dtype == object:
        return df
    # If there's a filename/index column (first column contains strings or paths), set it as index
    first_col = df.columns[0]
    try:
        if df[first_col].dtype == object or df[first_col].astype(str).str.contains('.obj|/|\\').any():
            df2 = df.set_index(first_col)
            return df2
    except Exception:
        pass
    return df


def map_columns_to_ids(df: pd.DataFrame) -> Dict[str, str]:
    """Return map colname -> normalized id."""
    mapping = {}
    for c in df.columns:
        mapping[c] = normalize_id(c)
    # if index is present and named, include it too
    try:
        idx = df.index
        if idx is not None:
            for v in idx[:10]:
                mapping[str(v)] = normalize_id(str(v))
    except Exception:
        pass
    return mapping


def compute_metrics_for_query(tp: int, fp: int, fn: int, tn: int) -> Dict[str, Optional[float]]:
    # Handle zero divisors gracefully
    def safe_div(a, b):
        return float(a) / float(b) if b and not math.isclose(b, 0.0) else None

    metrics = {}
    metrics['TP'] = tp
    metrics['FP'] = fp
    metrics['FN'] = fn
    metrics['TN'] = tn
    P = tp + fn
    N = tn + fp
    metrics['TPR'] = safe_div(tp, P)  # recall / sensitivity
    metrics['TNR'] = safe_div(tn, N)  # specificity
    metrics['PPV'] = safe_div(tp, tp + fp)  # precision
    metrics['NPV'] = safe_div(tn, tn + fn)
    metrics['FNR'] = safe_div(fn, fn + tp)
    metrics['FPR'] = safe_div(fp, fp + tn)
    metrics['FDR'] = safe_div(fp, fp + tp)
    metrics['FOR'] = safe_div(fn, fn + tn)
    metrics['TS'] = safe_div(tp, tp + fp + fn)  # Jaccard / Critical Success Index
    metrics['ACC'] = safe_div(tp + tn, P + N)
    # F1
    if metrics['PPV'] is not None and metrics['TPR'] is not None and (metrics['PPV'] + metrics['TPR'])>0:
        metrics['F1'] = 2.0 * metrics['PPV'] * metrics['TPR'] / (metrics['PPV'] + metrics['TPR'])
    else:
        metrics['F1'] = None
    # F_beta variants (beta=0.5 and beta=2)
    def f_beta(ppv, tpr, beta):
        if ppv is None or tpr is None:
            return None
        b2 = beta * beta
        denom = b2 * ppv + tpr
        if denom == 0:
            return None
        return (1.0 + b2) * ppv * tpr / denom
    metrics['F0_5'] = f_beta(metrics['PPV'], metrics['TPR'], 0.5)
    metrics['F2'] = f_beta(metrics['PPV'], metrics['TPR'], 2.0)
    # MCC
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom and denom > 0:
        metrics['MCC'] = (tp * tn - fp * fn) / math.sqrt(denom)
    else:
        metrics['MCC'] = None
    # BM and MK
    metrics['BM'] = None if (metrics['TPR'] is None or metrics['TNR'] is None) else (metrics['TPR'] + metrics['TNR'] - 1.0)
    metrics['MK'] = None if (metrics['PPV'] is None or metrics['NPV'] is None) else (metrics['PPV'] + metrics['NPV'] - 1.0)
    # Likelihood ratios
    metrics['LR+'] = None
    metrics['LR-'] = None
    if metrics['TPR'] is not None and metrics['FPR'] is not None and metrics['FPR'] and not math.isclose(metrics['FPR'], 0.0):
        metrics['LR+'] = metrics['TPR'] / metrics['FPR']
    if metrics['FNR'] is not None and metrics['TNR'] is not None and metrics['TNR'] and not math.isclose(metrics['TNR'], 0.0):
        metrics['LR-'] = metrics['FNR'] / metrics['TNR']
    return metrics


def compute_dcg(labels: List[int], k: Optional[int] = None) -> float:
    """Compute Discounted Cumulative Gain at position k.
    
    labels: relevance scores (binary 1/0 or graded) in ranked order
    k: position cutoff (if None, use all positions)
    Returns: DCG value
    """
    labels_arr = np.asarray(labels, dtype=float)
    if k is not None:
        labels_arr = labels_arr[:k]
    if labels_arr.size == 0:
        return 0.0
    # DCG = sum of (rel_i / log2(i+1)) for i=1 to k
    # Using log2(i+2) because positions are 0-indexed but formula uses 1-indexed
    positions = np.arange(1, labels_arr.size + 1)
    dcg = np.sum(labels_arr / np.log2(positions + 1))
    return float(dcg)


def compute_ndcg(labels: List[int], k: Optional[int] = None) -> Optional[float]:
    """Compute Normalized Discounted Cumulative Gain at position k.
    
    labels: relevance scores (binary 1/0 or graded) in ranked order
    k: position cutoff (if None, use all positions)
    Returns: NDCG value (0 to 1), or None if no relevant items
    """
    dcg = compute_dcg(labels, k)
    # Ideal DCG: sort labels in descending order
    ideal_labels = sorted(labels, reverse=True)
    idcg = compute_dcg(ideal_labels, k)
    if idcg == 0.0:
        return None  # no relevant items
    return dcg / idcg


def compute_roc_auc(labels: List[int], scores: List[float]) -> Optional[float]:
    """Compute ROC AUC using trapezoidal integration.

    labels: binary relevance (1 positive, 0 negative)
    scores: higher means more likely positive. For our distances, we pass -distance so
    higher score -> more similar.
    Returns None if AUC is undefined (no positives or no negatives).
    """
    # convert to numpy
    y = np.asarray(labels, dtype=int)
    x = np.asarray(scores, dtype=float)
    # need at least one pos and one neg
    if y.sum() == 0 or (y.size - y.sum()) == 0:
        return None
    # sort by descending score (highest similarity first)
    order = np.argsort(-x)
    y_sorted = y[order]
    # compute cumulative true positives and false positives
    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)
    P = float(y.sum())
    N = float(y.size - y.sum())
    # TPR and FPR arrays; add starting (0,0) and ending (1,1)
    tpr = np.concatenate(([0.0], tp_cum / P, [1.0]))
    fpr = np.concatenate(([0.0], fp_cum / N, [1.0]))
    # Compute AUC using trapezoidal rule over FPR(x) vs TPR(y)
    # Ensure monotonicity of FPR; it's non-decreasing by construction.
    auc = np.trapz(tpr, fpr)
    return float(auc)


def compute_roc_curve(labels: List[int], scores: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (fpr, tpr) arrays for a given labels/scores pair.

    Uses the same ranking and construction as compute_roc_auc.
    """
    y = np.asarray(labels, dtype=int)
    x = np.asarray(scores, dtype=float)
    # need at least one pos and one neg
    if y.sum() == 0 or (y.size - y.sum()) == 0:
        return np.array([]), np.array([])
    order = np.argsort(-x)
    y_sorted = y[order]
    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)
    P = float(y.sum())
    N = float(y.size - y.sum())
    tpr = np.concatenate(([0.0], tp_cum / P, [1.0]))
    fpr = np.concatenate(([0.0], fp_cum / N, [1.0]))
    return fpr, tpr


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, out_path: Optional[str] = None, title: Optional[str] = None):
    """Plot a single ROC curve and optionally save it to disk.

    Requires matplotlib; if not present, raises ImportError.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise ImportError('matplotlib required to plot ROC curves') from e
    if fpr.size == 0 or tpr.size == 0:
        raise ValueError('Empty ROC arrays')
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f'AUC={np.trapz(tpr, fpr):.4f}')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title or 'ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True)
    if out_path:
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        return out_path
    return fig


def plot_mean_roc_for_class(dist_df: pd.DataFrame, analysis_df: pd.DataFrame, class_name: str, out_path: Optional[str] = None, n_points: int = 200):
    """Compute and plot the averaged ROC curve for a class.

    This pools per-query ROC curves for queries belonging to `class_name`, interpolates
    TPR values onto a common FPR grid, averages them, and plots the mean ROC with
    +/- std shading.
    """
    try:
        import matplotlib.pyplot as plt
        from scipy import interpolate
    except Exception:
        raise ImportError('matplotlib and scipy required to plot mean ROC')

    analysis_df = analysis_df.copy()
    # build id->class mapping
    id_to_class = analysis_df.set_index('id')['class'].to_dict()
    # find queries of this class
    queries = [q for q in dist_df.index if id_to_class.get(normalize_id(str(q))) == class_name]
    if not queries:
        raise ValueError(f'No queries found for class {class_name}')
    fpr_grid = np.linspace(0.0, 1.0, n_points)
    tpr_interp_list = []
    for q in queries:
        row = dist_df.loc[q]
        try:
            s = pd.Series(row.values, index=dist_df.columns)
        except Exception:
            s = pd.Series(row)
        ranked_all = s.sort_values(ascending=True).index.tolist()
        labels = [1 if id_to_class.get(normalize_id(c)) == class_name else 0 for c in ranked_all]
        scores = []
        for c in ranked_all:
            try:
                scores.append(-float(s[c]))
            except Exception:
                scores.append(0.0)
        fpr, tpr = compute_roc_curve(labels, scores)
        if fpr.size == 0:
            continue
        interp_fn = interpolate.interp1d(fpr, tpr, bounds_error=False, fill_value=(0.0, 1.0))
        tpr_interp = interp_fn(fpr_grid)
        tpr_interp_list.append(tpr_interp)
    if not tpr_interp_list:
        raise ValueError('No valid ROC curves for class')
    tprs = np.vstack(tpr_interp_list)
    mean_tpr = np.nanmean(tprs, axis=0)
    std_tpr = np.nanstd(tprs, axis=0)
    mean_auc = np.trapz(mean_tpr, fpr_grid)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr_grid, mean_tpr, label=f'Mean AUC={mean_auc:.4f}')
    ax.fill_between(fpr_grid, np.maximum(mean_tpr - std_tpr, 0.0), np.minimum(mean_tpr + std_tpr, 1.0), alpha=0.2)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(out_path or f'Mean ROC for class {class_name}')
    ax.legend(loc='lower right')
    ax.grid(True)
    if out_path:
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        return out_path
    return fig


def evaluate_distance_matrix(dist_df: pd.DataFrame, analysis_df: pd.DataFrame, top_n: int = 10, out_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a distance matrix and return (per_query_df, per_class_summary_df).

    per_query_df contains metrics for each query (one row per query). per_class_summary_df
    contains averaged metrics per class plus overall weighted averages.
    """
    # Build mapping id -> class and id -> filenames list
    analysis_df = analysis_df.copy()
    id_to_class = analysis_df.set_index('id')['class'].to_dict()
    # class sizes (count of rows per class)
    class_sizes = analysis_df.groupby('class').size().to_dict()
    total_shapes = len(analysis_df)

    # Prepare column ids for dist_df
    df = dist_df.copy()
    col_ids = {c: normalize_id(c) for c in df.columns}

    per_query_rows = []

    # Prepare confusion accumulation structures
    classes = list(class_sizes.keys())
    confusion_counts: Dict[str, Dict[str, int]] = {cls: {c2: 0 for c2 in classes} for cls in classes}
    class_query_counts: Dict[str, int] = {cls: 0 for cls in classes}

    # For micro-AUROC calculations: collect pooled labels/scores globally and per-class
    pooled_labels_all: List[int] = []
    pooled_scores_all: List[float] = []
    pooled_by_class: Dict[str, Dict[str, List]] = {}

    # Build mapping from id to available filenames in analysis (to compute P)
    id_to_filenames = analysis_df.groupby('id')['filename'].apply(list).to_dict()

    # iterate queries with progress reporting
    rows = list(df.iterrows())
    for qname, row in progress_iterable(rows, desc=f'Queries ({len(rows)})'):
        qname_str = str(qname)
        qid = normalize_id(qname_str)
        # Determine query class using id mapping (fall back to exact filename match)
        qclass = id_to_class.get(qid)
        if qclass is None:
            matches = analysis_df[analysis_df['filename'] == qname_str]
            if not matches.empty:
                qclass = matches.iloc[0]['class']
                qid = matches.iloc[0]['id']
        if qclass is None:
            # Unknown query: skip
            continue

        # Build series of distances: row may be a numpy array or pandas Series
        distances = row
        try:
            s = pd.Series(distances.values, index=df.columns)
        except Exception:
            s = pd.Series(distances)
        # Remove self-matches: compare normalized ids
        def is_self(col):
            return normalize_id(col) == qid
        s = s[[not is_self(c) for c in s.index]]

        # Full ranked list (ascending distances => more similar)
        ranked_all = s.sort_values(ascending=True).index.tolist()

        # Compute P (positives available excluding query itself)
        P = max(0, len(id_to_filenames.get(qid, [])) - 1)

        # Average Precision (AP) across full ranking using class-level relevance
        if P > 0:
            sum_prec = 0.0
            rel_count = 0
            for idx, cand in enumerate(ranked_all, start=1):
                cid = normalize_id(cand)
                cand_class = id_to_class.get(cid)
                if cand_class == qclass:
                    rel_count += 1
                    sum_prec += rel_count / idx
            AP = (sum_prec / P) if P > 0 else None
        else:
            AP = None

        # Prepare labels and scores for ROC/AUROC computation
        # labels: 1 if same class, 0 otherwise; scores: -distance (higher -> more similar)
        labels: List[int] = []
        scores: List[float] = []
        for col in ranked_all:
            cid = normalize_id(col)
            cand_class = id_to_class.get(cid)
            labels.append(1 if cand_class == qclass else 0)
            # distance value from series s
            try:
                scores.append(-float(s[col]))
            except Exception:
                # fallback: use 0
                scores.append(0.0)

        ROC_AUC = compute_roc_auc(labels, scores)

        # Compute NDCG for top-N and full ranking
        # For full ranking NDCG (all retrieved items)
        NDCG_full = compute_ndcg(labels)
        # For top-N NDCG
        NDCG_topN = compute_ndcg(labels, k=top_n)

        # Append to pooled lists for micro-AUROC
        pooled_labels_all.extend(labels)
        pooled_scores_all.extend(scores)
        if qclass not in pooled_by_class:
            pooled_by_class[qclass] = {'labels': [], 'scores': []}
        pooled_by_class[qclass]['labels'].extend(labels)
        pooled_by_class[qclass]['scores'].extend(scores)

        # Top-N evaluation
        ranked = ranked_all
        topk = ranked[:top_n]
        tp = 0
        for cand in topk:
            cid = normalize_id(cand)
            cand_class = id_to_class.get(cid)
            if cand_class == qclass:
                tp += 1
        fp = top_n - tp

        # accumulate confusion counts for top-N
        if qclass in confusion_counts:
            class_query_counts[qclass] = class_query_counts.get(qclass, 0) + 1
            for cand in topk:
                cid = normalize_id(cand)
                cand_class = id_to_class.get(cid)
                if cand_class is None:
                    continue
                if cand_class not in confusion_counts[qclass]:
                    confusion_counts[qclass][cand_class] = 0
                confusion_counts[qclass][cand_class] += 1

        # For top-N retrieval evaluation, only consider the top-N items
        # FN: relevant items NOT in top-N (but available in database)
        fn = max(0, P - tp)
        # TN: For top-N, this should be 0 since we only evaluate what we retrieved
        # In retrieval, we don't have "true negatives" in the traditional sense
        # because we only look at retrieved items, not all non-retrieved items
        tn = 0

        metrics = compute_metrics_for_query(tp, fp, fn, tn)

        row_out = {
            'query': qname_str,
            'id': qid,
            'class': qclass,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'P': P,
            'N_total': total_shapes - 1,
            'AP': AP,
            'ROC_AUC': ROC_AUC,
            'NDCG': NDCG_full,
            'NDCG_top{}'.format(top_n): NDCG_topN,
        }
        row_out.update(metrics)
        per_query_rows.append(row_out)

    per_query_df = pd.DataFrame(per_query_rows)

    # Per-class averages (mean over queries in class)
    class_summary = per_query_df.groupby('class').mean(numeric_only=True)
    # Also compute overall weighted averages (weighted by class size)
    metrics_cols = [c for c in class_summary.columns if c not in ('tp','fp','fn','tn','P','N_total')]
    total = per_query_df.shape[0]
    # compute per-class counts
    class_counts = per_query_df.groupby('class').size().to_dict()
    # overall (by-shape) mean
    overall_by_shape = per_query_df[metrics_cols].mean(skipna=True).to_dict()
    # weighted by class size
    weighted_by_class = {}
    for m in metrics_cols:
        num = 0.0
        den = 0.0
        for cls, val in class_summary[m].items():
            cnt = class_counts.get(cls, 0)
            if pd.isna(val):
                continue
            num += float(val) * float(cnt)
            den += float(cnt)
        weighted_by_class[m] = (num / den) if den>0 else None

    # Compute micro-AUROC per-class and overall (pooled across queries)
    per_class_micro_auc: Dict[str, Optional[float]] = {}
    for cls in class_summary.index:
        data = pooled_by_class.get(cls)
        if data is None:
            per_class_micro_auc[cls] = None
        else:
            per_class_micro_auc[cls] = compute_roc_auc(data['labels'], data['scores'])

    overall_micro_auc = compute_roc_auc(pooled_labels_all, pooled_scores_all) if pooled_labels_all else None
    # overall macro AUROC: mean of per-class mean ROC_AUC (class_summary already contains mean ROC_AUC)
    overall_macro_auc = None
    if 'ROC_AUC' in class_summary.columns:
        overall_macro_auc = float(class_summary['ROC_AUC'].mean(skipna=True)) if not class_summary['ROC_AUC'].dropna().empty else None

    # assemble summary DataFrame
    summary_rows = []
    for cls, vals in class_summary.iterrows():
        row = {'class': cls, 'class_size': int(class_sizes.get(cls, 0))}
        for col in class_summary.columns:
            row[col] = vals[col]
        # attach micro AUROC for this class if available
        row['ROC_AUC_micro'] = per_class_micro_auc.get(cls)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    # Append overall rows
    overall_row = {'class': '__overall_by_query__', 'class_size': total_shapes}
    overall_row.update(overall_by_shape)
    # micro (pooled across queries) and macro (mean of per-class means)
    overall_row['ROC_AUC_micro'] = overall_micro_auc
    overall_row['ROC_AUC_macro'] = overall_macro_auc
    # macro by class: mean of class_summary (distinct from per-query overall)
    macro_by_class = class_summary.mean(skipna=True).to_dict()
    macro_row = {'class': '__macro_by_class__', 'class_size': len(class_summary)}
    macro_row.update(macro_by_class)
    macro_row['ROC_AUC_micro'] = None
    macro_row['ROC_AUC_macro'] = overall_macro_auc
    summary_df = pd.concat([summary_df, pd.DataFrame([overall_row, macro_row])], ignore_index=True, sort=False)

    # If out_dir is provided, generate plots: per-class mean ROC, per-class AP histogram,
    # overall ROC, overall AP histogram, and top-N confusion heatmap.
    if out_dir:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except Exception:
            print('matplotlib required to generate plots; skipping plot generation')
        else:
            plots_dir = Path(out_dir) / 'plots'
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Per-class plots
            for cls in progress_iterable(classes, desc='Classes'):
                try:
                    outp = plots_dir / f'{cls}_mean_roc.png'
                    try:
                        plot_mean_roc_for_class(dist_df, analysis_df, cls, out_path=str(outp))
                    except Exception as e:
                        # skip if not enough data
                        print(f'Could not plot mean ROC for class {cls}: {e}')

                    # histogram of AP for queries in this class
                    ap_vals = per_query_df[per_query_df['class'] == cls]['AP'].dropna().astype(float)
                    if not ap_vals.empty:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.hist(ap_vals, bins=20, color='C0', edgecolor='black')
                        ax.set_title(f'AP distribution for class {cls}')
                        ax.set_xlabel('Average Precision (AP)')
                        ax.set_ylabel('Count')
                        fig.savefig(plots_dir / f'{cls}_AP_histogram.png', bbox_inches='tight')
                        plt.close(fig)
                except Exception as e:
                    print(f'Error while generating plots for class {cls}: {e}')

            # Overall pooled ROC
            try:
                if pooled_labels_all and pooled_scores_all:
                    fpr, tpr = compute_roc_curve(pooled_labels_all, pooled_scores_all)
                    if fpr.size and tpr.size:
                        plot_roc_curve(fpr, tpr, out_path=str(plots_dir / 'overall_roc.png'), title='Overall ROC (pooled)')
            except Exception as e:
                print(f'Could not plot overall ROC: {e}')

            # Overall AP histogram
            try:
                ap_vals_all = per_query_df['AP'].dropna().astype(float)
                if not ap_vals_all.empty:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.hist(ap_vals_all, bins=30, color='C1', edgecolor='black')
                    ax.set_title('AP distribution (all queries)')
                    ax.set_xlabel('Average Precision (AP)')
                    ax.set_ylabel('Count')
                    fig.savefig(plots_dir / 'AP_histogram_overall.png', bbox_inches='tight')
                    plt.close(fig)
            except Exception as e:
                print(f'Could not plot AP histogram overall: {e}')

            # Confusion heatmap (top-N retrieved classes per query averaged by query class)
            try:
                labels_sorted = classes
                mat = np.zeros((len(classes), len(classes)), dtype=float)
                for i, cls in enumerate(classes):
                    qcnt = class_query_counts.get(cls, 0)
                    if qcnt == 0:
                        continue
                    for j, cls2 in enumerate(classes):
                        # average proportion of top-N retrieved assigned to cls2 for queries in cls
                        mat[i, j] = confusion_counts.get(cls, {}).get(cls2, 0) / float(qcnt * top_n)
                fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.5), max(6, len(classes) * 0.5)))
                im = ax.imshow(mat, interpolation='nearest', cmap='viridis')
                ax.set_xticks(range(len(classes)))
                ax.set_yticks(range(len(classes)))
                ax.set_xticklabels(classes, rotation=90, fontsize=8)
                ax.set_yticklabels(classes, fontsize=8)
                ax.set_xlabel('Retrieved class')
                ax.set_ylabel('Query class')
                ax.set_title(f'Top-{top_n} retrieval confusion heatmap')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                fig.savefig(plots_dir / f'confusion_top{top_n}_heatmap.png', bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f'Could not plot confusion heatmap: {e}')

    return per_query_df, summary_df


def run_evaluation(matching_files: List[str], analysis_file: str, top_n: int, out_dir: str):
    analysis_df = load_analysis_labels(analysis_file)
    os.makedirs(out_dir, exist_ok=True)
    # Collect results per approach to enable combined comparisons
    all_per_query: Dict[str, pd.DataFrame] = {}
    all_summary: Dict[str, pd.DataFrame] = {}
    for path in progress_iterable(list(matching_files), desc='Matching files'):
        if not os.path.exists(path):
            print(f"Warning: matching file not found: {path}")
            continue
        base = Path(path).stem
        # Per-approach output folder to avoid overwriting plots
        approach_out = Path(out_dir) / base
        approach_out.mkdir(parents=True, exist_ok=True)
        per_query_out = Path(approach_out) / f"{base}_per_query_top{top_n}.csv"
        summary_out = Path(approach_out) / f"{base}_per_class_summary_top{top_n}.csv"
        
        # Check if CSVs already exist (cache); if so, load them instead of recalculating
        if per_query_out.exists() and summary_out.exists():
            print(f"Loading cached results for {base} from {per_query_out} and {summary_out}")
            per_query_df = pd.read_csv(per_query_out)
            summary_df = pd.read_csv(summary_out)
        else:
            print(f"Evaluating {base} ...")
            dist_df = load_distance_matrix(path)
            per_query_df, summary_df = evaluate_distance_matrix(dist_df, analysis_df, top_n=top_n, out_dir=str(approach_out))
            per_query_df.to_csv(per_query_out, index=False)
            summary_df.to_csv(summary_out, index=False)
            print(f"Wrote {per_query_out} and {summary_out}")
        
        all_per_query[base] = per_query_df
        all_summary[base] = summary_df

    # After processing all approaches, generate combined CSVs and comparison plots

    def generate_comparison_plots(all_summary: Dict[str, pd.DataFrame], out_dir: str):
        """Create combined CSVs and plots to compare approaches across classes and overall."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Build combined per-class DataFrame: index=class, columns=(approach, metric)
        # Include core assignment metrics (accuracy, precision/recall, specificity/sensitivity)
        # plus additional retrieval metrics (AP, NDCG, etc.)
        metrics_to_plot = ['ACC', 'PPV', 'TPR', 'TNR', 'F1', 'AP', 'ROC_AUC', 'MCC', 'NDCG', 'NDCG_top{}'.format(top_n)]
        classes = set()
        for df in all_summary.values():
            classes.update(df['class'].tolist())
        classes = sorted(classes)
        combined_rows = []
        for approach, df in all_summary.items():
            for _, row in df.iterrows():
                cls = row['class']
                r = {'approach': approach, 'class': cls}
                for m in metrics_to_plot:
                    r[m] = row.get(m, None)
                combined_rows.append(r)
        combined_df = pd.DataFrame(combined_rows)
        combined_csv = out_dir / 'combined_per_class_summary.csv'
        combined_df.to_csv(combined_csv, index=False)

        # Create per-class comparison heatmaps (rows=classes, cols=approaches) for each metric
        if plt is not None:
            approaches = sorted(all_summary.keys())
            n_classes = len(classes)
            n_approaches = len(approaches)
            
            # Use consistent colormap across all heatmaps for easier comparison
            # Collect all values across all metrics to determine global vmin/vmax
            all_metric_values = []
            for m in metrics_to_plot:
                for approach in approaches:
                    sub = combined_df[combined_df['approach'] == approach]
                    for cls in classes:
                        vals = sub[sub['class'] == cls][m]
                        if not vals.empty and pd.notna(vals.values[0]):
                            all_metric_values.append(float(vals.values[0]))
            
            # Use global min/max for consistent colormap (but still allow per-metric override for special cases)
            global_vmin = min(all_metric_values) if all_metric_values else 0.0
            global_vmax = max(all_metric_values) if all_metric_values else 1.0
            
            for m in metrics_to_plot:
                # Per-metric normalization and colormap choices
                # Use global scale for most metrics, but keep MCC with diverging colormap
                if m == 'MCC':
                    vmin, vmax = -1.0, 1.0
                    cmap_name = 'RdBu_r'  # reversed so blue=negative, red=positive
                else:
                    # Use consistent viridis colormap with global scale
                    vmin, vmax = global_vmin, global_vmax
                    cmap_name = 'viridis'

                # build matrix (classes x approaches)
                mat = np.full((n_classes, n_approaches), np.nan, dtype=float)
                for j, approach in enumerate(approaches):
                    sub = combined_df[combined_df['approach'] == approach]
                    for i, cls in enumerate(classes):
                        vals = sub[sub['class'] == cls][m]
                        if not vals.empty and pd.notna(vals.values[0]):
                            mat[i, j] = float(vals.values[0])

                # display labels: capitalize and replace underscores
                display_classes = [str(c).replace('_', ' ').title() for c in classes]

                fig, ax = plt.subplots(figsize=(3 + n_approaches * 1.2, max(5, n_classes * 0.35)))
                im = ax.imshow(mat, aspect='auto', interpolation='nearest', cmap=cmap_name, vmin=vmin, vmax=vmax)
                ax.set_yticks(np.arange(n_classes))
                ax.set_yticklabels(display_classes, fontsize=9)
                ax.set_xticks(np.arange(n_approaches))
                ax.set_xticklabels([a.replace('_', ' ').title() for a in approaches], rotation=45, ha='right', fontsize=10)
                ax.set_title(f'Per-class comparison — {m}', fontsize=12, weight='bold')
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=10)
                # annotate values inside cells with contrast-aware text color
                # determine threshold from colormap midpoint
                import matplotlib
                cmap_obj = matplotlib.colormaps.get_cmap(cmap_name)
                if vmin is not None and vmax is not None:
                    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
                else:
                    norm = matplotlib.colors.Normalize(vmin=np.nanmin(mat), vmax=np.nanmax(mat))
                for i in range(n_classes):
                    for j in range(n_approaches):
                        val = mat[i, j]
                        if not np.isnan(val):
                            # get luminance by mapping to RGBA then to perceived brightness
                            rgba = cmap_obj(norm(val))
                            # perceived brightness formula
                            r, g, b = rgba[0], rgba[1], rgba[2]
                            lum = 0.299 * r + 0.587 * g + 0.114 * b
                            text_color = 'white' if lum < 0.6 else 'black'
                            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8, color=text_color)
                ax.tick_params(axis='y', which='both', labelsize=9)
                ax.tick_params(axis='x', which='both', labelsize=10)
                fig.savefig(out_dir / f'heatmap_per_class_vs_approach_{m}.png', bbox_inches='tight', dpi=200)
                plt.close(fig)

            # Overall comparison using the overall rows
            overall_rows = []
            for approach, df in all_summary.items():
                overall = df[df['class'].isin(['__overall_by_query__', '__macro_by_class__'])]
                # pick overall_by_query
                if not overall.empty:
                    row = overall[overall['class'] == '__overall_by_query__']
                    if row.empty:
                        row = overall.iloc[[0]]
                else:
                    row = None
                vals = {'approach': approach}
                if row is not None and not row.empty:
                    for m in metrics_to_plot:
                        vals[m] = float(row.iloc[0].get(m)) if pd.notna(row.iloc[0].get(m)) else np.nan
                else:
                    for m in metrics_to_plot:
                        vals[m] = np.nan
                overall_rows.append(vals)
            overall_df = pd.DataFrame(overall_rows)
            overall_csv = out_dir / 'combined_overall_summary.csv'
            overall_df.to_csv(overall_csv, index=False)

            # grouped bar chart overall with value labels and legend to the right
            # Map metric codes to readable names
            metric_names = {
                'ACC': 'Accuracy',
                'PPV': 'Precision',
                'TPR': 'Recall (Sensitivity)',
                'TNR': 'Specificity',
                'F1': 'F1-Score',
                'AP': 'Average Precision',
                'ROC_AUC': 'ROC AUC',
                'MCC': 'Matthews Correlation',
                'NDCG': 'NDCG (Full)',
                'NDCG_top10': 'NDCG@10'
            }
            
            fig, ax = plt.subplots(figsize=(14, 6))
            x = np.arange(len(overall_df)) * 2.5  # Increase spacing between approach groups
            width = 0.18  # Slightly wider bars
            bars_collection = []
            for i, m in enumerate(metrics_to_plot):
                vals = overall_df[m].astype(float).values
                # Use readable name in legend
                label = metric_names.get(m, m)
                bars = ax.bar(x + i * width, vals, width=width, label=label)
                bars_collection.append(bars)
                
                # Compute ranking for this metric (1=best, higher value is better for all our metrics)
                ranks = []
                for val in vals:
                    if np.isnan(val):
                        ranks.append('')
                    else:
                        # Count how many values are strictly greater (rank = number of better + 1)
                        rank = 1 + sum(1 for v in vals if not np.isnan(v) and v > val)
                        ranks.append(f'({rank})')
                
                # annotate values on bars with ranking
                for bar, rank in zip(bars, ranks):
                    h = bar.get_height()
                    if not np.isnan(h):
                        # Show value and rank
                        label_text = f'{h:.3f}\n{rank}' if rank else f'{h:.3f}'
                        ax.annotate(label_text, xy=(bar.get_x() + bar.get_width() / 2, h), 
                                    xytext=(0, 3), textcoords='offset points', 
                                    ha='center', va='bottom', fontsize=8)

            ax.set_xticks(x + width * (len(metrics_to_plot) - 1) / 2)
            ax.set_xticklabels(overall_df['approach'], rotation=45, ha='right', fontsize=10)
            ax.set_title('Overall comparison across approaches', fontsize=12, weight='bold')
            ax.tick_params(axis='y', labelsize=10)
            
            # Adjust y-axis limits to prevent label cutoff
            # Get current limits and add 15% padding at the top for labels
            ylim = ax.get_ylim()
            ax.set_ylim(ylim[0], ylim[1] * 1.15)
            
            leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            for text in leg.get_texts():
                text.set_fontsize(10)
            fig.savefig(out_dir / 'comparison_overall_metrics.png', bbox_inches='tight', dpi=200)
            plt.close(fig)
        else:
            print('matplotlib not available: skipping combined plots')

    # generate combined outputs
    generate_comparison_plots(all_summary, out_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--matching', nargs='*', default=DEFAULT_MATCHING, help='Matching/distance CSV files')
    p.add_argument('--analysis', default=DEFAULT_ANALYSIS, help='Analysis CSV with class labels')
    p.add_argument('--top', type=int, default=10, help='Top-N results to use for evaluation')
    p.add_argument('--out', default='Src/evalution/figures', help='Output folder for CSV summaries')
    args = p.parse_args()
    run_evaluation(args.matching, args.analysis, args.top, args.out)
    print(f'Evaluation completed. Results saved in {args.out}')


if __name__ == '__main__':
    main()
